"""SQLite-backed emitter for process-bigraph composites.

Stores each composite tick as one row in a SQLite ``.db`` file. Rows from
multiple simulations can share one file — they're partitioned by
``simulation_id``. The standalone helpers (:func:`load_history`,
:func:`list_simulations`, etc.) let callers inspect and analyze a
finished run without rebuilding a Composite or a Core.

Only Python's stdlib ``sqlite3`` is required.
"""

import copy
import dataclasses
import datetime
import json
import os
import sqlite3
import uuid
from typing import Dict, List, Optional

import numpy as np

from bigraph_schema import Edge, get_path, set_path
from process_bigraph.emitter import Emitter


__all__ = [
    'SQLiteEmitter',
    'save_simulation_metadata',
    'mark_simulation_finished',
    'list_simulations',
    'load_history',
    'load_simulation_metadata',
]


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    # pint.Quantity wired to a quantity[...] port: persist the bare magnitude
    # (mirrors ParquetEmitter._strip_units / _is_quantity) so history stays
    # numeric instead of a "<Quantity(...)>" repr string that the viewer would
    # render character-by-character. Duck-typed to avoid importing pint; the
    # returned magnitude may be an np scalar/array, which json re-serializes
    # through this same default (np.floating -> float, ndarray -> tolist()).
    if (
        hasattr(value, "magnitude")
        and hasattr(value, "units")
        and hasattr(value, "dimensionality")
    ):
        return value.magnitude
    # bigraph-schema Node dataclasses (String, Float, Integer, ...) can end
    # up wired into emitted state; fall back to their repr so history stays
    # serializable without dragging in the schema machinery.
    if dataclasses.is_dataclass(value):
        return dataclasses.asdict(value)
    return repr(value)


def _tree_copy(state):
    """Deep copy utility for nested simulation state (excluding Edge instances)."""
    if isinstance(state, dict):
        return {k: v for k, v in ((k, _tree_copy(v)) for k, v in state.items()) if v is not None}
    if isinstance(state, np.ndarray):
        return state.copy()
    if isinstance(state, Edge):
        return None
    return copy.deepcopy(state)


def _resolve_query_paths(paths, query):
    """Accept either the new ``paths`` kwarg or the legacy ``query`` kwarg."""
    if paths is None and query is not None:
        return query
    return paths


def _init_history_db(conn):
    """Create both history and simulations tables. Safe to call repeatedly."""
    conn.execute('PRAGMA journal_mode=WAL')
    conn.execute('PRAGMA synchronous=NORMAL')
    conn.execute('''
        CREATE TABLE IF NOT EXISTS history (
            simulation_id TEXT NOT NULL,
            step INTEGER NOT NULL,
            global_time REAL,
            state TEXT NOT NULL,
            PRIMARY KEY (simulation_id, step)
        )
    ''')
    conn.execute(
        'CREATE INDEX IF NOT EXISTS idx_history_sim_time '
        'ON history(simulation_id, global_time)'
    )
    conn.execute('''
        CREATE TABLE IF NOT EXISTS simulations (
            simulation_id TEXT PRIMARY KEY,
            name TEXT,
            started_at TEXT NOT NULL,
            completed_at TEXT,
            elapsed_seconds REAL,
            composite_config TEXT,
            metadata TEXT
        )
    ''')
    # Migrate older dbs that predate completed_at / elapsed_seconds.
    existing = {row[1] for row in conn.execute("PRAGMA table_info(simulations)")}
    if 'completed_at' not in existing:
        conn.execute('ALTER TABLE simulations ADD COLUMN completed_at TEXT')
    if 'elapsed_seconds' not in existing:
        conn.execute('ALTER TABLE simulations ADD COLUMN elapsed_seconds REAL')


def save_simulation_metadata(db_path, simulation_id, composite_config=None,
                             metadata=None, name=None):
    """Write or update the ``simulations`` row for a run.

    Call once per simulation, typically right after building the Composite,
    to record the config that produced the history rows. Idempotent: fields
    passed as ``None`` leave any existing value untouched, so you can fill
    in ``name`` first and ``composite_config`` later without clobbering.
    """
    conn = sqlite3.connect(db_path, isolation_level=None)
    try:
        _init_history_db(conn)
        conn.execute(
            'INSERT INTO simulations '
            '(simulation_id, name, started_at, composite_config, metadata) '
            'VALUES (?, ?, ?, ?, ?) '
            'ON CONFLICT(simulation_id) DO UPDATE SET '
            '  name = COALESCE(excluded.name, simulations.name), '
            '  composite_config = COALESCE(excluded.composite_config, simulations.composite_config), '
            '  metadata = COALESCE(excluded.metadata, simulations.metadata)',
            (
                simulation_id,
                name,
                datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
                json.dumps(composite_config, default=_json_default) if composite_config is not None else None,
                json.dumps(metadata, default=_json_default) if metadata is not None else None,
            ),
        )
    finally:
        conn.close()


def list_simulations(db_path) -> List[Dict]:
    """Return all recorded simulations in a history db, newest first.

    Each entry has ``simulation_id``, ``name``, ``started_at``,
    ``completed_at``, ``elapsed_seconds``, ``step_count``, and ``has_config``
    (True if a composite_config was saved). No core or Composite is required
    — use this to browse a db long after the runs that produced it.
    """
    if not os.path.exists(db_path):
        return []
    conn = sqlite3.connect(db_path)
    try:
        _init_history_db(conn)
        rows = conn.execute('''
            SELECT s.simulation_id, s.name, s.started_at, s.completed_at,
                   s.elapsed_seconds, s.composite_config,
                   (SELECT COUNT(*) FROM history h WHERE h.simulation_id = s.simulation_id)
            FROM simulations s
            ORDER BY s.started_at DESC
        ''').fetchall()
        # Also include sims that have history rows but no metadata row.
        orphan_rows = conn.execute('''
            SELECT h.simulation_id, NULL, NULL, NULL, NULL, NULL, COUNT(*)
            FROM history h
            WHERE h.simulation_id NOT IN (SELECT simulation_id FROM simulations)
            GROUP BY h.simulation_id
        ''').fetchall()
    finally:
        conn.close()

    return [
        {
            'simulation_id': sid,
            'name': name,
            'started_at': started_at,
            'completed_at': completed_at,
            'elapsed_seconds': elapsed,
            'step_count': step_count,
            'has_config': cfg is not None,
        }
        for (sid, name, started_at, completed_at, elapsed, cfg, step_count)
        in list(rows) + list(orphan_rows)
    ]


def load_history(db_path, simulation_id, paths: Optional[List] = None) -> List[Dict]:
    """Load a simulation's history from a db file. No core/Composite needed.

    Returns the same shape that ``SQLiteEmitter.query()`` returns, so plot
    and analysis code that consumed RAM/JSON emitter output works unchanged.
    """
    if not os.path.exists(db_path):
        return []
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.execute(
            'SELECT state FROM history WHERE simulation_id = ? ORDER BY step',
            (simulation_id,),
        )
        history = [json.loads(row[0]) for row in cursor.fetchall()]
    finally:
        conn.close()

    if isinstance(paths, list):
        results = []
        for t in history:
            result = {}
            for path in paths:
                element = get_path(t, path)
                result = set_path(result, path, element)
            results.append(result)
        return results
    return history


def load_simulation_metadata(db_path, simulation_id) -> Optional[Dict]:
    """Return the ``simulations`` row for a sim, or ``None`` if missing.

    Result dict has ``simulation_id``, ``name``, ``started_at``,
    ``completed_at``, ``elapsed_seconds``, ``composite_config`` (parsed
    from JSON), and ``metadata``.
    """
    if not os.path.exists(db_path):
        return None
    conn = sqlite3.connect(db_path)
    try:
        _init_history_db(conn)
        row = conn.execute(
            'SELECT simulation_id, name, started_at, completed_at, '
            'elapsed_seconds, composite_config, metadata '
            'FROM simulations WHERE simulation_id = ?',
            (simulation_id,),
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        return None
    sid, name, started_at, completed_at, elapsed, cfg, meta = row
    return {
        'simulation_id': sid,
        'name': name,
        'started_at': started_at,
        'completed_at': completed_at,
        'elapsed_seconds': elapsed,
        'composite_config': json.loads(cfg) if cfg else None,
        'metadata': json.loads(meta) if meta else None,
    }


def mark_simulation_finished(db_path, simulation_id, elapsed_seconds=None):
    """Stamp ``completed_at`` (UTC now) and ``elapsed_seconds`` on a run.

    Call this right after ``sim.run()`` returns. Safe to call multiple
    times — each call just overwrites the two fields.
    """
    conn = sqlite3.connect(db_path, isolation_level=None)
    try:
        _init_history_db(conn)
        completed_at = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
        conn.execute(
            'UPDATE simulations SET completed_at = ?, elapsed_seconds = ? '
            'WHERE simulation_id = ?',
            (completed_at, elapsed_seconds, simulation_id),
        )
    finally:
        conn.close()


class SQLiteEmitter(Emitter):
    """Append simulation state to a SQLite database each timestep.

    One row per step, with the full state tree stored as JSON. The database
    file is a single ``.db`` file that can be opened with any SQLite client,
    queried with SQL, and kept for long-term storage. Multiple simulations
    can share one file — rows are partitioned by ``simulation_id``.

    To record the composite config or other metadata alongside the history
    rows, call :func:`save_simulation_metadata` after constructing the
    Composite — the emitter itself only writes the per-step history rows.

    ``subsample`` records only every Nth composite tick (default 1 = every
    tick). The ``step`` column still stores the true composite tick number,
    so time-series produced from the history reflect the real cadence even
    though intermediate ticks were not persisted. Use this when a Composite
    fires the emitter very often (small intervals, long runs) and you don't
    need every tick in the archive — it's the cheapest way to shrink the
    write volume without losing the simulation's time axis.

    ``batch_size`` buffers up to N recorded rows in memory and flushes them
    in a single SQL transaction (default 1 = write each row immediately).
    With ``batch_size=100`` the per-row fsync overhead is amortized across
    the batch, giving a large speedup on high-frequency runs. Unflushed
    rows are guaranteed to be written on ``close()`` and on the next
    ``query()``; a hard crash before flush loses the buffered rows.
    """
    config_schema = {
        **Emitter.config_schema,
        'file_path': {'_type': 'string', '_default': './out'},
        'db_file': {'_type': 'string', '_default': 'history.db'},
        'simulation_id': {'_type': 'string', '_default': None},
        'name': {'_type': 'string', '_default': None},
        'subsample': {'_type': 'integer', '_default': 1},
        'batch_size': {'_type': 'integer', '_default': 1},
    }

    @classmethod
    def emitter_contract(cls):
        from viva_emitters.contract import EmitterContract
        return EmitterContract(output_kind="sqlite", output_uri_config_key="db_file")

    def __init__(self, config, core):
        super().__init__(config, core)
        self.simulation_id = config.get('simulation_id') or str(uuid.uuid4())
        self.file_path = config.get('file_path') or './out'
        os.makedirs(self.file_path, exist_ok=True)
        self.db_path = os.path.join(self.file_path, config.get('db_file') or 'history.db')

        subsample = config.get('subsample')
        self.subsample = 1 if subsample is None else int(subsample)
        if self.subsample < 1:
            raise ValueError(
                f'SQLiteEmitter subsample must be >= 1, got {self.subsample}'
            )

        batch_size = config.get('batch_size')
        self.batch_size = 1 if batch_size is None else int(batch_size)
        if self.batch_size < 1:
            raise ValueError(
                f'SQLiteEmitter batch_size must be >= 1, got {self.batch_size}'
            )

        self._conn = sqlite3.connect(self.db_path, isolation_level=None)
        _init_history_db(self._conn)

        name = config.get('name')
        if name is not None:
            save_simulation_metadata(self.db_path, self.simulation_id, name=name)

        self._step = 0
        self._batch = []

    def update(self, state) -> Dict:
        if self._conn is None:
            raise RuntimeError('SQLiteEmitter has been closed')
        # Advance the true composite tick counter on every call; only persist
        # the row when this tick falls on the subsample cadence. Ticks 0,
        # subsample, 2*subsample, ... are written (first tick always kept).
        step = self._step
        self._step += 1
        if step % self.subsample != 0:
            return {}

        global_time = state.get('global_time') if isinstance(state, dict) else None
        # Strip live Edge/process instances the same way RAMEmitter does;
        # otherwise wires that pull in process objects break JSON serialization.
        clean = _tree_copy(state)
        payload = json.dumps(clean, default=_json_default)
        self._batch.append((self.simulation_id, step, global_time, payload))
        if len(self._batch) >= self.batch_size:
            self._flush_batch()
        return {}

    def _flush_batch(self):
        """Write any buffered rows in a single SQL transaction."""
        if not self._batch or self._conn is None:
            return
        # Plain INSERT (not OR REPLACE): `step` is a per-run monotonic
        # counter, so a PK conflict here would indicate a real bug — fail
        # loudly rather than silently overwriting a row.
        self._conn.execute('BEGIN')
        try:
            self._conn.executemany(
                'INSERT INTO history '
                '(simulation_id, step, global_time, state) VALUES (?, ?, ?, ?)',
                self._batch,
            )
            self._conn.execute('COMMIT')
        except Exception:
            self._conn.execute('ROLLBACK')
            raise
        self._batch.clear()

    def query(self, paths=None, query=None):
        paths = _resolve_query_paths(paths, query)
        # Flush so buffered-but-unwritten rows are visible to the read.
        self._flush_batch()
        # Route through the standalone helper so the in-process and post-hoc
        # retrieval paths go through the same code.
        return load_history(self.db_path, self.simulation_id, paths=paths)

    def close(self):
        """Close the underlying SQLite connection explicitly.

        Flushes any buffered rows (from ``batch_size > 1``) so nothing is
        lost, then closes the connection. After calling ``close`` the
        emitter can no longer record new rows.
        """
        if self._conn is not None:
            try:
                self._flush_batch()
            finally:
                self._conn.close()
                self._conn = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
