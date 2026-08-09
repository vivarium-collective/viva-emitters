"""Tests for viva_emitters.sqlite_emitter.

Adapted from process-bigraph's tests.py — covers retrieval helpers,
the paths/query kwarg, subsample, batch_size, and close() semantics.
The composite-driven test from upstream is replaced by direct calls to
``emitter.update(state)`` so this suite does not depend on Composite.
"""

import os
import sqlite3
import tempfile

import pytest

from viva_emitters.sqlite_emitter import (
    SQLiteEmitter,
    save_simulation_metadata,
    mark_simulation_finished,
    list_simulations,
    load_history,
    load_simulation_metadata,
    _json_default,
)


class _FakeQuantity:
    """Duck-typed pint.Quantity: exposes magnitude/units/dimensionality.

    _json_default keys off these three attributes (mirroring
    ParquetEmitter._is_quantity), so this stands in for a real Quantity
    without a pint dependency in the test suite.
    """

    def __init__(self, magnitude, units='femtogram'):
        self.magnitude = magnitude
        self.units = units
        self.dimensionality = '[mass]'

    def __repr__(self):  # what the bug used to persist
        return f'<Quantity({self.magnitude}, {self.units!r})>'


def test_sqlite_emitter_basic_run(core):
    """A bare run records one row per update() call and is durable on disk."""
    tmp = tempfile.mkdtemp(prefix='sqlite_basic_')
    e = SQLiteEmitter({
        'emit': {'global_time': 'node', 'value': 'node'},
        'file_path': tmp, 'simulation_id': 'sim', 'db_file': 'test_history.db',
    }, core=core)
    for i in range(10):
        e.update({'global_time': float(i), 'value': i * 2})

    results = e.query()
    assert len(results) == 10
    assert results[-1]['global_time'] == 9.0
    assert results[-1]['value'] == 18

    # Db file exists and survives a fresh connection.
    db_path = os.path.join(tmp, 'test_history.db')
    assert os.path.exists(db_path)
    conn = sqlite3.connect(db_path)
    try:
        (count,) = conn.execute('SELECT COUNT(*) FROM history').fetchone()
        assert count == 10
    finally:
        conn.close()
    e.close()


def test_sqlite_emitter_retrieval_helpers(core):
    """The standalone helpers let callers inspect a db without a Composite."""
    tmp = tempfile.mkdtemp(prefix='sqlite_retrieval_')
    db_path = os.path.join(tmp, 'history.db')

    e1 = SQLiteEmitter({
        'emit': {'global_time': 'node'},
        'file_path': tmp, 'simulation_id': 'run-A', 'name': 'alpha',
    }, core=core)
    for i in range(4):
        e1.update({'global_time': float(i)})
    save_simulation_metadata(
        db_path, 'run-A',
        composite_config={'cells': {}},
        metadata={'experiment': 'alpha', 'notes': 'first run'},
    )
    mark_simulation_finished(db_path, 'run-A', elapsed_seconds=12.5)
    e1.close()

    e2 = SQLiteEmitter({
        'emit': {'global_time': 'node'},
        'file_path': tmp, 'simulation_id': 'run-B',
    }, core=core)
    for i in range(2):
        e2.update({'global_time': float(i)})
    e2.close()

    sims = {s['simulation_id']: s for s in list_simulations(db_path)}
    assert set(sims) == {'run-A', 'run-B'}
    assert sims['run-A']['step_count'] == 4
    assert sims['run-A']['elapsed_seconds'] == 12.5
    assert sims['run-A']['completed_at'] is not None
    assert sims['run-A']['has_config'] is True
    assert sims['run-B']['step_count'] == 2
    assert sims['run-B']['has_config'] is False
    assert sims['run-B']['elapsed_seconds'] is None

    history = load_history(db_path, 'run-A')
    assert len(history) == 4
    assert history[-1]['global_time'] == 3.0
    filtered = load_history(db_path, 'run-A', paths=[['global_time']])
    assert filtered == [{'global_time': t} for t in (0.0, 1.0, 2.0, 3.0)]

    meta = load_simulation_metadata(db_path, 'run-A')
    assert meta['name'] == 'alpha'
    assert meta['composite_config'] == {'cells': {}}
    assert meta['metadata']['experiment'] == 'alpha'
    assert meta['elapsed_seconds'] == 12.5
    assert load_simulation_metadata(db_path, 'nope') is None


def test_sqlite_emitter_query_paths_kwarg(core):
    """``query()`` accepts the new ``paths`` kwarg and the legacy ``query`` kwarg."""
    tmp = tempfile.mkdtemp(prefix='sqlite_paths_kwarg_')
    e = SQLiteEmitter({
        'emit': {'global_time': 'node', 'a': 'node', 'b': 'node'},
        'file_path': tmp, 'simulation_id': 'sim',
    }, core=core)
    for i in range(3):
        e.update({'global_time': float(i), 'a': i, 'b': i * 2})

    assert len(e.query()) == 3
    assert e.query(paths=[['a']]) == [{'a': 0}, {'a': 1}, {'a': 2}]
    assert e.query(query=[['a']]) == [{'a': 0}, {'a': 1}, {'a': 2}]
    # paths wins when both are provided.
    assert e.query(paths=[['a']], query=[['b']]) == [{'a': 0}, {'a': 1}, {'a': 2}]
    e.close()


def test_sqlite_emitter_subsample(core):
    """subsample=N writes every Nth composite tick (first tick always kept)."""
    tmp = tempfile.mkdtemp(prefix='sqlite_subsample_')
    e = SQLiteEmitter({
        'emit': {'global_time': 'node', 'v': 'node'},
        'file_path': tmp, 'simulation_id': 'sim',
        'subsample': 5,
    }, core=core)

    for i in range(20):
        e.update({'global_time': float(i), 'v': i})
    e.close()

    history = load_history(os.path.join(tmp, 'history.db'), 'sim')
    assert len(history) == 4
    assert [row['v'] for row in history] == [0, 5, 10, 15]
    assert [row['global_time'] for row in history] == [0.0, 5.0, 10.0, 15.0]

    # `step` column preserves the true composite tick number.
    conn = sqlite3.connect(os.path.join(tmp, 'history.db'))
    try:
        steps = [r[0] for r in conn.execute(
            'SELECT step FROM history WHERE simulation_id = ? ORDER BY step',
            ('sim',),
        ).fetchall()]
    finally:
        conn.close()
    assert steps == [0, 5, 10, 15]


def test_sqlite_emitter_subsample_rejects_bad_value(core):
    """subsample < 1 is rejected at construction time."""
    tmp = tempfile.mkdtemp(prefix='sqlite_subsample_bad_')
    with pytest.raises(ValueError):
        SQLiteEmitter({
            'emit': {},
            'file_path': tmp, 'simulation_id': 'sim',
            'subsample': 0,
        }, core=core)


def test_sqlite_emitter_batch_size(core):
    """batch_size buffers up to N rows and flushes them in one transaction."""
    tmp = tempfile.mkdtemp(prefix='sqlite_batch_')
    e = SQLiteEmitter({
        'emit': {'global_time': 'node', 'v': 'node'},
        'file_path': tmp, 'simulation_id': 'sim',
        'batch_size': 5,
    }, core=core)

    db_path = os.path.join(tmp, 'history.db')
    conn = sqlite3.connect(db_path)
    try:
        # Below the batch threshold: nothing on disk yet.
        for i in range(3):
            e.update({'global_time': float(i), 'v': i})
        (pending,) = conn.execute(
            'SELECT COUNT(*) FROM history WHERE simulation_id=?', ('sim',)
        ).fetchone()
        assert pending == 0

        # query() forces a flush.
        assert len(e.query()) == 3
        (after_query,) = conn.execute(
            'SELECT COUNT(*) FROM history WHERE simulation_id=?', ('sim',)
        ).fetchone()
        assert after_query == 3

        # Cross a batch boundary.
        for i in range(3, 8):
            e.update({'global_time': float(i), 'v': i})
        (after_boundary,) = conn.execute(
            'SELECT COUNT(*) FROM history WHERE simulation_id=?', ('sim',)
        ).fetchone()
        assert after_boundary == 8

        # Remaining buffered rows flush on close().
        for i in range(8, 10):
            e.update({'global_time': float(i), 'v': i})
        e.close()
        (final,) = conn.execute(
            'SELECT COUNT(*) FROM history WHERE simulation_id=?', ('sim',)
        ).fetchone()
        assert final == 10
    finally:
        conn.close()

    history = load_history(db_path, 'sim')
    assert [row['v'] for row in history] == list(range(10))


def test_sqlite_emitter_batch_size_rejects_bad_value(core):
    """batch_size < 1 is rejected at construction."""
    tmp = tempfile.mkdtemp(prefix='sqlite_batch_bad_')
    with pytest.raises(ValueError):
        SQLiteEmitter({
            'emit': {},
            'file_path': tmp, 'simulation_id': 'sim',
            'batch_size': 0,
        }, core=core)


def test_sqlite_emitter_close(core):
    """close() releases the connection, makes update() fail loudly, idempotent."""
    tmp = tempfile.mkdtemp(prefix='sqlite_close_')
    e = SQLiteEmitter({
        'emit': {'global_time': 'node'},
        'file_path': tmp, 'simulation_id': 'sim',
    }, core=core)
    e.update({'global_time': 0.0})
    e.close()

    # Idempotent.
    e.close()

    # No silent writes after close.
    with pytest.raises(RuntimeError):
        e.update({'global_time': 1.0})

    db_path = os.path.join(tmp, 'history.db')
    conn = sqlite3.connect(db_path)
    try:
        (count,) = conn.execute('SELECT COUNT(*) FROM history').fetchone()
        assert count == 1
    finally:
        conn.close()


def test_json_default_strips_quantity_units_to_magnitude():
    """A unit-bearing (pint.Quantity-like) leaf serializes to its bare
    magnitude, not a "<Quantity(...)>" repr string. Regression: units-on-ports
    mass listeners were persisted as repr strings and rendered char-by-char in
    the Results viewer."""
    import json

    q = _FakeQuantity(0.0843286761)
    # Direct: default() returns the magnitude, and json serializes it as a number.
    assert _json_default(q) == 0.0843286761
    payload = json.dumps({'mass': q}, default=_json_default)
    assert json.loads(payload) == {'mass': 0.0843286761}
    assert 'Quantity' not in payload


def test_sqlite_emitter_persists_quantity_as_magnitude(core):
    """End-to-end: a Quantity-valued observable lands in history as a float."""
    tmp = tempfile.mkdtemp(prefix='sqlite_quantity_')
    e = SQLiteEmitter({
        'emit': {'global_time': 'node', 'mass': 'node'},
        'file_path': tmp, 'simulation_id': 'sim',
    }, core=core)
    e.update({'global_time': 0.0, 'mass': _FakeQuantity(12.57241145)})
    e.close()

    rows = load_history(os.path.join(tmp, 'history.db'), 'sim')
    assert rows, 'expected one emitted row'
    assert rows[0]['mass'] == 12.57241145
