"""RunReader: emitter-agnostic reader for stored simulation runs.

Returns a uniform per-tick series (with generation + time) for any
observable, over parquet / sqlite / xarray-zarr stores.

Imports of heavy dependencies (duckdb, xarray, zarr) are lazy: they happen
inside the backend branches so importing this module does not force extras.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import polars as pl


# ---------------------------------------------------------------------------
# Public error types
# ---------------------------------------------------------------------------


class IdNotInCatalog(KeyError):
    """An element id was not found in the observable's name catalog."""


class CatalogUnavailable(LookupError):
    """A name catalog is required but not present for this observable/backend."""


# ---------------------------------------------------------------------------
# Public data types
# ---------------------------------------------------------------------------


@dataclass
class RunRef:
    """Reference to a stored run.

    Attributes:
        store: Path to the run's store (hive dir | .db file | .zarr root).
        kind:  Backend kind — ``"parquet"``, ``"sqlite"``, or ``"xarray"``.
               Auto-detected from ``store`` if ``None``.
        simulation_id: Optional SQLite-specific filter.  When set, only rows
               with this ``simulation_id`` are read from the history table.
               Useful when multiple runs share one ``.db`` file and you want
               to address a single simulation.
    """

    store: str
    kind: str | None = None
    simulation_id: str | None = None


# ---------------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------------


def _detect_kind(store: str) -> str:
    """Infer backend from the store path."""
    p = Path(store)
    # SQLite: an existing file with .db / .sqlite extension, or just the ext
    if p.is_file() and p.suffix in {".db", ".sqlite"}:
        return "sqlite"
    if p.suffix in {".db", ".sqlite"}:
        return "sqlite"
    # XArray: .zarr directory (suffix or path ending)
    if p.suffix == ".zarr" or str(store).endswith(".zarr"):
        return "xarray"
    if p.is_dir() and (p / ".zgroup").exists():
        return "xarray"
    # Parquet: directory containing a 'history' hive sub-tree
    if p.is_dir() and (p / "history").exists():
        return "parquet"
    if p.is_dir() and any(p.glob("**/history")):
        return "parquet"
    # Default fallbacks
    if p.is_dir():
        return "parquet"
    return "sqlite"


# ---------------------------------------------------------------------------
# Cumulative-time helper
# ---------------------------------------------------------------------------


def _cumulative_time(df: pl.DataFrame) -> pl.DataFrame:
    """Add ``abs_time`` column: gen-local ``time`` stitched into one axis.

    Each generation's time resets to 0.  ``abs_time`` appends each generation
    at ``prev_max_time + 1`` so the axis is strictly increasing across gens.

    Args:
        df: Must have ``generation`` (int) and ``time`` (float) columns.

    Returns:
        ``df`` with an additional ``abs_time`` (Float64) column, sorted by
        ``(generation, time)``.
    """
    if df.is_empty():
        return df.with_columns(pl.lit(0.0).cast(pl.Float64).alias("abs_time"))

    gens = sorted(df["generation"].unique().to_list())
    offset = 0.0
    offsets: dict = {}
    for g in gens:
        offsets[g] = offset
        gmax = df.filter(pl.col("generation") == g)["time"].max()
        offset += (gmax or 0.0) + 1.0

    gen_dtype = df.schema["generation"]
    off_df = pl.DataFrame({
        "generation": pl.Series(list(offsets.keys()), dtype=gen_dtype),
        "_off": pl.Series(list(offsets.values()), dtype=pl.Float64),
    })
    return (
        df.join(off_df, on="generation", how="left")
        .with_columns((pl.col("time") + pl.col("_off")).alias("abs_time"))
        .drop("_off")
        .sort(["generation", "time"])
    )


# ---------------------------------------------------------------------------
# by_generation helper
# ---------------------------------------------------------------------------


def by_generation(df: pl.DataFrame) -> dict[int, pl.DataFrame]:
    """Split a series DataFrame into per-generation slices.

    Args:
        df: DataFrame as returned by :py:meth:`RunReader.series` — must have
            a ``generation`` column.

    Returns:
        Mapping from generation integer to the corresponding subset of ``df``.
    """
    return {
        int(g): df.filter(pl.col("generation") == g)
        for g in sorted(df["generation"].unique().to_list())
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _dig(state: dict, dotted_path: str):
    """Navigate a nested dict by dotted path.

    Example::

        _dig({"a": {"b": 1}}, "a.b")  # → 1

    Raises:
        KeyError: if any segment of ``dotted_path`` is missing.
    """
    parts = dotted_path.split(".")
    val = state
    for p in parts:
        if not isinstance(val, dict):
            raise KeyError(
                f"Cannot dig into {type(val).__name__!r} at key {p!r}"
            )
        if p not in val:
            raise KeyError(p)
        val = val[p]
    return val


def _flatten_paths(d: dict, prefix: str = "") -> list[str]:
    """Flatten a nested dict to sorted dotted leaf-key paths.

    Example::

        _flatten_paths({"a": {"b": 1}, "c": 2})  # → ["a.b", "c"]
    """
    paths: list[str] = []
    for k, v in d.items():
        full = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict) and v:
            paths.extend(_flatten_paths(v, full))
        else:
            paths.append(full)
    return paths


def _parse_gen_from_sim_id(sim_id: str, fallback: int) -> int:
    """Try to parse a generation integer from a simulation_id string.

    Matches patterns like ``gen=1``, ``gen_1``, ``gen-1``, ``gen1``
    (case-insensitive).  Returns ``fallback`` when no pattern is found.
    """
    m = re.search(r"gen[=_-]?(\d+)", sim_id, re.IGNORECASE)
    if m:
        return int(m.group(1))
    return fallback


def _sim_ids_to_gen_map(sim_ids: list[str]) -> dict[str, int]:
    """Map each simulation_id to a generation integer.

    Tries to parse generation from the id (``gen=N`` patterns); falls back to
    0-based insertion order so the result is never empty for a non-empty store.
    """
    result: dict[str, int] = {}
    for i, sid in enumerate(sim_ids):
        result[sid] = _parse_gen_from_sim_id(sid, i)
    return result


# ---------------------------------------------------------------------------
# Series reduction helper
# ---------------------------------------------------------------------------


def _reduce_series(dfs: list[pl.DataFrame], op: str) -> pl.DataFrame:
    """Reduce a list of ``[generation, time, abs_time, value]`` DataFrames.

    All DataFrames must share the same ``(generation, time, abs_time)`` index.
    Supported ops: ``"sum"``, ``"mean"``, ``"max"``, ``"min"``.
    """
    _empty = pl.DataFrame({
        "generation": pl.Series([], dtype=pl.Int64),
        "time":       pl.Series([], dtype=pl.Float64),
        "abs_time":   pl.Series([], dtype=pl.Float64),
        "value":      pl.Series([], dtype=pl.Float64),
    })
    if not dfs:
        return _empty
    if len(dfs) == 1:
        return dfs[0]

    base = dfs[0].rename({"value": "v0"})
    for i, df in enumerate(dfs[1:], 1):
        base = base.join(
            df.select(["generation", "time", "abs_time",
                       pl.col("value").alias(f"v{i}")]),
            on=["generation", "time", "abs_time"],
            how="inner",
        )

    vcols = [pl.col(f"v{i}") for i in range(len(dfs))]
    if op == "sum":
        val_expr = pl.sum_horizontal(*vcols)
    elif op == "mean":
        val_expr = pl.mean_horizontal(*vcols)
    elif op == "max":
        val_expr = pl.max_horizontal(*vcols)
    elif op == "min":
        val_expr = pl.min_horizontal(*vcols)
    else:
        raise ValueError(f"Unknown reduce op: {op!r}")

    return (
        base
        .with_columns(val_expr.alias("value"))
        .select(["generation", "time", "abs_time", "value"])
    )


# ---------------------------------------------------------------------------
# RunReader
# ---------------------------------------------------------------------------

_PARQUET_ID_COLS = frozenset({
    "experiment_id", "variant", "lineage_seed",
    "generation", "agent_id",
    "time",        # legacy: real stores use global_time, not time
    "global_time", # real ParquetEmitter output
})
_SQLITE_ID_FIELDS = frozenset({"generation", "global_time"})


class RunReader:
    """Emitter-agnostic reader that returns a uniform per-tick series.

    Open with :py:meth:`open`, then call:

    * :py:meth:`observables` — list available observable ids,
    * :py:meth:`generations` — sorted generation indices,
    * :py:meth:`series` — polars DataFrame with ``generation``, ``time``,
      ``abs_time``, ``value``.

    Example::

        r = RunReader.open("/path/to/run")
        df = r.series("listeners.mass.cell_mass")
        by_gen = by_generation(df)
    """

    def __init__(self, ref: RunRef) -> None:
        self._ref = ref
        self._kind: str = ref.kind or _detect_kind(ref.store)

    @classmethod
    def open(cls, store: str | RunRef, kind: str | None = None) -> "RunReader":
        """Open a run store and return a :py:class:`RunReader`.

        Args:
            store: Path to the run store, or a :py:class:`RunRef`.
            kind:  Force backend — ``"sqlite"``, ``"parquet"``, or
                   ``"xarray"``.  Auto-detected if ``None``.
        """
        if isinstance(store, RunRef):
            return cls(store)
        return cls(RunRef(store=str(store), kind=kind))

    @property
    def kind(self) -> str:
        """Detected or forced backend kind."""
        return self._kind

    # ------------------------------------------------------------------
    # Public interface — dispatch on kind
    # ------------------------------------------------------------------

    def observables(self) -> list[str]:
        """Return sorted list of available observable ids for this run."""
        if self._kind == "sqlite":
            return self._sqlite_observables()
        if self._kind == "parquet":
            return self._parquet_observables()
        if self._kind == "xarray":
            return self._xarray_observables()
        raise ValueError(f"Unknown kind: {self._kind!r}")

    def generations(self) -> list[int]:
        """Return sorted list of generation indices present in the run."""
        if self._kind == "sqlite":
            return self._sqlite_generations()
        if self._kind == "parquet":
            return self._parquet_generations()
        if self._kind == "xarray":
            return self._xarray_generations()
        raise ValueError(f"Unknown kind: {self._kind!r}")

    def series(self, observable: str) -> pl.DataFrame:
        """Return a per-tick series for ``observable``.

        Args:
            observable: Backend-native observable id as listed by
                :py:meth:`observables` (e.g. ``"listeners.mass.cell_mass"``).

        Returns:
            Polars DataFrame with columns::

                generation : Int64   — generation index (first-class from backend)
                time       : Float64 — gen-local time (resets each generation)
                abs_time   : Float64 — cumulative time (stitched across gens)
                value      : Float64 — observable value

            Ordered by ``(generation, time)``.

        Raises:
            KeyError: If ``observable`` is not available in this run.
        """
        if self._kind == "sqlite":
            return self._sqlite_series(observable)
        if self._kind == "parquet":
            return self._parquet_series(observable)
        if self._kind == "xarray":
            return self._xarray_series(observable)
        raise ValueError(f"Unknown kind: {self._kind!r}")

    def summary(self) -> dict:
        """Canonical quantitative summary of the run — the shape every recorded
        simulation carries so tooling can render it uniformly.

        Returns a dict with whatever could be read (best-effort; never raises)::

            generations    : int  — number of generations in the run
            n_observables   : int  — number of readouts collected
            sim_minutes     : int  — simulated time (cumulative ``abs_time`` in
                                     seconds / 60), NOT wall-clock

        ``sim_minutes`` is derived from a scalar observable's ``abs_time``;
        list/array-typed observables can't be cast to a numeric series, so we
        try known scalars first and then fall back to scanning, guarding each
        read so one bad observable never costs the generation/readout counts.
        """
        out: dict = {}
        try:
            gens = self.generations()
            if gens:
                out["generations"] = len(gens)
        except Exception:  # noqa: BLE001
            pass
        try:
            obs = self.observables() or []
        except Exception:  # noqa: BLE001
            obs = []
        if obs:
            out["n_observables"] = len(obs)
            for cand in ("listeners.mass.cell_mass", "listeners.mass.dry_mass",
                         "listeners.mass.cellMass", "global_time", "time", *obs):
                if cand not in obs:
                    continue
                try:
                    df = self.series(cand)
                except Exception:  # noqa: BLE001
                    continue
                if df is not None and len(df) and "abs_time" in df.columns:
                    out["sim_minutes"] = int(round(float(df["abs_time"].max()) / 60.0))
                    break
        return out

    # ------------------------------------------------------------------
    # SQLite backend
    # ------------------------------------------------------------------

    def _sqlite_rows(self) -> list[tuple]:
        """Read all history rows: [(simulation_id, step, global_time, state_dict), ...].

        Filters by ``RunRef.simulation_id`` when that attribute is set.
        Rows are ordered by ``(simulation_id, step)`` so multi-generation
        databases arrive in a stable, predictable order.
        """
        import json
        import sqlite3

        con = sqlite3.connect(self._ref.store)
        try:
            sim_id_filter = self._ref.simulation_id
            if sim_id_filter is not None:
                rows = con.execute(
                    "SELECT simulation_id, step, global_time, state "
                    "FROM history WHERE simulation_id = ? "
                    "ORDER BY simulation_id, step",
                    (sim_id_filter,),
                ).fetchall()
            else:
                rows = con.execute(
                    "SELECT simulation_id, step, global_time, state "
                    "FROM history ORDER BY simulation_id, step"
                ).fetchall()
        finally:
            con.close()
        return [
            (sim_id, step, gtime, json.loads(state))
            for sim_id, step, gtime, state in rows
        ]

    def _sqlite_observables(self) -> list[str]:
        rows = self._sqlite_rows()
        if not rows:
            return []
        _, _, _, state = rows[0]
        obs_state = {k: v for k, v in state.items() if k not in _SQLITE_ID_FIELDS}
        return sorted(_flatten_paths(obs_state))

    def _sqlite_generations(self) -> list[int]:
        rows = self._sqlite_rows()
        if not rows:
            return []

        # Prefer explicit generation key in state (legacy / custom runner).
        gen_from_state = {
            state.get("generation")
            for _, _, _, state in rows
            if "generation" in state
        }
        if gen_from_state and None not in gen_from_state:
            return sorted(int(g) for g in gen_from_state)

        # Fall back: derive generation from DISTINCT simulation_ids.
        # The real runner names per-gen sims with a gen=N suffix so we can
        # parse the generation index; if that fails, use 0-based index so we
        # NEVER return empty for a non-empty history.
        sim_ids = sorted(set(r[0] for r in rows))
        gen_map = _sim_ids_to_gen_map(sim_ids)
        return sorted(set(gen_map.values()))

    def _sqlite_series(self, observable: str) -> pl.DataFrame:
        _empty = pl.DataFrame({
            "generation": pl.Series([], dtype=pl.Int64),
            "time": pl.Series([], dtype=pl.Float64),
            "abs_time": pl.Series([], dtype=pl.Float64),
            "value": pl.Series([], dtype=pl.Float64),
        })
        rows = self._sqlite_rows()
        if not rows:
            return _empty

        # Validate — check observable exists in first row's state.
        _, _, _, first_state = rows[0]
        try:
            _dig(first_state, observable)
        except KeyError:
            raise KeyError(observable)

        # Build simulation_id → generation mapping once, used for every row.
        sim_ids = sorted(set(r[0] for r in rows))
        gen_map = _sim_ids_to_gen_map(sim_ids)

        data = []
        for sim_id, _step, gtime, state in rows:
            # Prefer explicit generation from state; fall back to derived gen.
            gen = state.get("generation")
            if gen is None:
                gen = gen_map.get(sim_id, 0)
            try:
                val = _dig(state, observable)
            except KeyError:
                continue
            data.append({
                "generation": int(gen),
                "time": float(gtime if gtime is not None else 0.0),
                "value": float(val),
            })

        if not data:
            return _empty

        df = pl.DataFrame(data).sort(["generation", "time"])
        df = _cumulative_time(df)
        return df.select(["generation", "time", "abs_time", "value"])

    # ------------------------------------------------------------------
    # Parquet backend
    # ------------------------------------------------------------------

    def _parquet_all_sql(self):
        """Return ``(conn, history_sql, config_sql)``; result is cached."""
        if not hasattr(self, "_pq_cache"):
            from viva_emitters.parquet_emitter import create_duckdb_conn, dataset_sql

            p = Path(self._ref.store)
            out_dir = str(p.parent)
            exp_id = p.name
            conn = create_duckdb_conn()
            history_sql, config_sql, _ = dataset_sql(out_dir, [exp_id])
            self._pq_cache = (conn, history_sql, config_sql)
        return self._pq_cache

    def _parquet_conn_sql(self):
        """Return ``(conn, history_sql)`` — backward-compatible shim."""
        conn, history_sql, _ = self._parquet_all_sql()
        return conn, history_sql

    def _parquet_observables(self) -> list[str]:
        from viva_emitters.parquet_emitter import list_columns

        conn, history_sql = self._parquet_conn_sql()
        all_cols = list_columns(conn, history_sql)
        # Exclude partition / id columns and bulk array columns (not scalars).
        obs_cols = [
            c for c in all_cols
            if c not in _PARQUET_ID_COLS and not c.startswith("bulk")
        ]
        return sorted(c.replace("__", ".") for c in obs_cols)

    def _parquet_generations(self) -> list[int]:
        conn, history_sql = self._parquet_conn_sql()
        result = conn.sql(
            f"SELECT DISTINCT generation FROM ({history_sql}) ORDER BY generation"
        ).pl()
        return result["generation"].cast(pl.Int64).to_list()

    def _parquet_series(self, observable: str) -> pl.DataFrame:
        from viva_emitters.parquet_emitter import list_columns

        col_name = observable.replace(".", "__")
        conn, history_sql = self._parquet_conn_sql()

        # Validate — check column exists in schema.
        available = list_columns(conn, history_sql, col_name)
        if not available:
            raise KeyError(observable)

        # Detect whether the store uses 'global_time' (real emitter) or 'time'
        # (older / hand-built stores).  Select it aliased to 'time' so the
        # returned DataFrame always has a 'time' column regardless of origin.
        all_cols = list_columns(conn, history_sql)
        time_col = "global_time" if "global_time" in all_cols else "time"

        quoted_col = f'"{col_name}"'
        query = f"""
            SELECT
                generation,
                {time_col} AS time,
                {quoted_col} AS value
            FROM ({history_sql})
            ORDER BY generation, {time_col}
        """
        df = conn.sql(query).pl()
        df = df.with_columns(
            pl.col("generation").cast(pl.Int64),
            pl.col("time").cast(pl.Float64),
            pl.col("value").cast(pl.Float64),
        )
        df = _cumulative_time(df)
        return df.select(["generation", "time", "abs_time", "value"])

    def _parquet_catalog(self, observable: str) -> list | None:
        """Return name catalog for *observable* from the parquet store.

        * ``"bulk"`` → reads the ``bulk__id`` column (first-row constant).
        * Other observables → reads ``output_metadata__<col>`` from the
          configuration parquet (written by :func:`parquet_emitter.field_metadata`).
        * Returns ``None`` when no catalog is available.
        """
        from viva_emitters.parquet_emitter import list_columns, METADATA_PREFIX

        conn, history_sql, config_sql = self._parquet_all_sql()

        if observable == "bulk":
            try:
                result = conn.sql(
                    f'SELECT first("bulk__id") FROM ({history_sql})'
                ).fetchone()
                if result and result[0] is not None:
                    return list(result[0])
            except Exception:
                pass
            return None

        # Listener / other vector observables → output_metadata in config.
        col_key = observable.replace(".", "__")
        meta_col = METADATA_PREFIX + col_key

        try:
            config_cols = list_columns(conn, config_sql)
        except Exception:
            return None

        if meta_col not in config_cols:
            return None

        try:
            result = conn.sql(
                f'SELECT first("{meta_col}") FROM ({config_sql})'
            ).fetchone()
            if result and result[0] is not None:
                return list(result[0])
        except Exception:
            pass
        return None

    def _parquet_select_element(self, col_name: str, idx: int) -> pl.DataFrame:
        """Extract the scalar at position *idx* from list column *col_name* per tick."""
        from viva_emitters.parquet_emitter import list_columns

        conn, history_sql, _ = self._parquet_all_sql()
        all_cols = list_columns(conn, history_sql)
        time_col = "global_time" if "global_time" in all_cols else "time"

        # DuckDB arrays are 1-indexed.
        query = f"""
            SELECT
                generation,
                {time_col} AS time,
                "{col_name}"[{idx + 1}] AS value
            FROM ({history_sql})
            ORDER BY generation, {time_col}
        """
        df = conn.sql(query).pl()
        df = df.with_columns(
            pl.col("generation").cast(pl.Int64),
            pl.col("time").cast(pl.Float64),
            pl.col("value").cast(pl.Float64),
        )
        df = _cumulative_time(df)
        return df.select(["generation", "time", "abs_time", "value"])

    def _parquet_aggregate_elements(
        self, col_name: str, indices: list[int], op: str
    ) -> pl.DataFrame:
        """Reduce multiple list-column elements in a single DuckDB query."""
        from viva_emitters.parquet_emitter import list_columns

        conn, history_sql, _ = self._parquet_all_sql()
        all_cols = list_columns(conn, history_sql)
        time_col = "global_time" if "global_time" in all_cols else "time"

        # DuckDB uses 1-indexed array access.
        elem_exprs = [f'"{col_name}"[{idx + 1}]' for idx in indices]

        if op == "sum":
            agg_expr = " + ".join(elem_exprs)
        elif op == "mean":
            n = float(len(indices))
            agg_expr = f"({' + '.join(elem_exprs)}) / {n}"
        elif op == "max":
            agg_expr = (
                elem_exprs[0]
                if len(elem_exprs) == 1
                else f"greatest({', '.join(elem_exprs)})"
            )
        elif op == "min":
            agg_expr = (
                elem_exprs[0]
                if len(elem_exprs) == 1
                else f"least({', '.join(elem_exprs)})"
            )
        else:
            raise ValueError(f"Unknown op: {op!r}")

        query = f"""
            SELECT
                generation,
                {time_col} AS time,
                ({agg_expr}) AS value
            FROM ({history_sql})
            ORDER BY generation, {time_col}
        """
        df = conn.sql(query).pl()
        df = df.with_columns(
            pl.col("generation").cast(pl.Int64),
            pl.col("time").cast(pl.Float64),
            pl.col("value").cast(pl.Float64),
        )
        df = _cumulative_time(df)
        return df.select(["generation", "time", "abs_time", "value"])

    # ------------------------------------------------------------------
    # XArray backend
    # ------------------------------------------------------------------

    def _xarray_open(self):
        """Open the DataTree from the zarr store (lazy import)."""
        import xarray as xr

        return xr.open_datatree(self._ref.store, engine="zarr")

    def _xarray_time_ds(self, dt):
        """Return the xarray Dataset that holds ``emitstep_gen=`` coordinates.

        Hand-crafted fixtures (e.g. tests) put time coords at the root ``"/"``
        node.  ``XArrayEmitter`` writes them at its partition path (e.g.
        ``/experiment_id=X/variant=0/lineage_seed=0``).  This helper
        searches root first, then all groups, so both layouts work.
        """
        from viva_emitters.xarray_emitter.storage import TIME_COO_PREFIX

        # Fast path: root node (hand-crafted test fixtures)
        root_ds = dt["/"].ds
        if any(c.startswith(TIME_COO_PREFIX) for c in root_ds.coords):
            return root_ds

        # Scan partition / child groups (real XArrayEmitter stores)
        for path in dt.groups:
            if path == "/":
                continue
            ds = dt[path].ds
            if any(c.startswith(TIME_COO_PREFIX) for c in ds.coords):
                return ds

        return root_ds  # fallback — empty, gen_info will be []

    def _xarray_gen_info(self, dt) -> list[tuple[int, str, str]]:
        """Return list of (gen_int, time_coo_name, time_var_name) sorted by gen.

        Reads from whichever node holds ``emitstep_gen=N`` coordinates
        (root for hand-crafted fixtures; partition path for XArrayEmitter).
        """
        # Import constants lazily — only executed on xarray backend
        from viva_emitters.xarray_emitter.storage import (
            TIME_COO_PREFIX,
            TIME_VAR_PREFIX,
        )

        time_ds = self._xarray_time_ds(dt)
        gen_info = []
        for coo_name in time_ds.coords:
            if coo_name.startswith(TIME_COO_PREFIX):
                # e.g. "emitstep_gen=1" → extract "1" after last "="
                gen = int(coo_name.rsplit("=", 1)[-1])
                time_var = f"{TIME_VAR_PREFIX}gen={gen}"
                gen_info.append((gen, coo_name, time_var))
        return sorted(gen_info, key=lambda x: x[0])

    def _xarray_observables(self) -> list[str]:
        dt = self._xarray_open()
        obs = []
        for path in dt.groups:
            if path == "/":
                continue
            node_ds = dt[path].ds
            has_gen_vars = any(
                v.startswith("generation=") for v in node_ds.data_vars
            )
            if has_gen_vars:
                obs.append(path.lstrip("/").replace("/", "."))
        return sorted(obs)

    def _xarray_generations(self) -> list[int]:
        dt = self._xarray_open()
        return [g for g, _, _ in self._xarray_gen_info(dt)]

    def _xarray_series(self, observable: str) -> pl.DataFrame:
        dt = self._xarray_open()
        node_path = "/" + observable.replace(".", "/")

        if node_path not in dt.groups:
            raise KeyError(observable)

        node_ds = dt[node_path].ds
        time_ds = self._xarray_time_ds(dt)
        gen_info = self._xarray_gen_info(dt)

        rows = []
        for gen, _coo_name, time_var in gen_info:
            var_name = f"generation={gen}"
            if var_name not in node_ds.data_vars:
                continue
            if time_var not in time_ds.data_vars:
                continue
            times = time_ds[time_var].values.ravel()
            values = node_ds[var_name].values.ravel()
            for t, v in zip(times, values):
                rows.append({
                    "generation": gen,
                    "time": float(t),
                    "value": float(v),
                })

        if not rows:
            # Observable node exists but no generation variables
            raise KeyError(observable)

        df = pl.DataFrame(rows).sort(["generation", "time"])
        df = _cumulative_time(df)
        return df.select(["generation", "time", "abs_time", "value"])

    def _xarray_catalog(self, observable: str) -> list | None:
        """Return the ``id_<var>`` coordinate values for *observable*, or ``None``."""
        try:
            from viva_emitters.xarray_emitter.storage import VAR_COO_PREFIX
        except ImportError:
            return None

        dt = self._xarray_open()
        node_path = "/" + observable.replace(".", "/")
        if node_path not in dt.groups:
            return None

        node_ds = dt[node_path].ds
        var_name = observable.split(".")[-1]
        id_coord = f"{VAR_COO_PREFIX}{var_name}"
        if id_coord in node_ds.coords:
            return list(node_ds.coords[id_coord].values)
        return None

    def _xarray_select_element(self, col_name: str, idx: int) -> pl.DataFrame:
        """Extract element at *idx* from an array observable per tick (XArray)."""
        observable = col_name.replace("__", ".")
        dt = self._xarray_open()
        node_path = "/" + observable.replace(".", "/")

        if node_path not in dt.groups:
            raise KeyError(observable)

        node_ds = dt[node_path].ds
        root_ds = dt["/"].ds
        gen_info = self._xarray_gen_info(dt)

        rows = []
        for gen, _coo_name, time_var in gen_info:
            var_name = f"generation={gen}"
            if var_name not in node_ds.data_vars or time_var not in root_ds.data_vars:
                continue
            times = root_ds[time_var].values.ravel()
            arr = node_ds[var_name].values
            # 2D (ticks × elements) → take column at idx; 1D fallback.
            if arr.ndim == 2:
                values = arr[:, idx]
            else:
                values = arr.ravel()
            for t, v in zip(times, values):
                rows.append({"generation": gen, "time": float(t), "value": float(v)})

        if not rows:
            raise KeyError(observable)

        df = pl.DataFrame(rows).sort(["generation", "time"])
        df = _cumulative_time(df)
        return df.select(["generation", "time", "abs_time", "value"])

    # ------------------------------------------------------------------
    # SQLite catalog + element extraction
    # ------------------------------------------------------------------

    def _sqlite_catalog(self, observable: str) -> list | None:
        """Return catalog from SQLite ``simulations`` metadata, or ``None``."""
        try:
            from viva_emitters.sqlite_emitter import load_simulation_metadata
        except ImportError:
            return None

        rows = self._sqlite_rows()
        if not rows:
            return None

        # Try each distinct sim_id (preserve first-seen order).
        seen: dict[str, None] = {}
        for r in rows:
            seen[r[0]] = None
        db_path = self._ref.store

        for sim_id in seen:
            try:
                meta = load_simulation_metadata(db_path, sim_id)
                if meta is None:
                    continue
                output_meta = (meta.get("metadata") or {}).get("output_metadata", {})
                val = output_meta
                for part in observable.split("."):
                    if not isinstance(val, dict):
                        val = None
                        break
                    val = val.get(part, {})
                if isinstance(val, list) and val:
                    return val
            except Exception:
                continue
        return None

    def _sqlite_select_element(self, col_name: str, idx: int) -> pl.DataFrame:
        """Extract element at *idx* from a list-valued observable per tick (SQLite)."""
        observable = col_name.replace("__", ".")
        _empty = pl.DataFrame({
            "generation": pl.Series([], dtype=pl.Int64),
            "time":       pl.Series([], dtype=pl.Float64),
            "abs_time":   pl.Series([], dtype=pl.Float64),
            "value":      pl.Series([], dtype=pl.Float64),
        })

        rows = self._sqlite_rows()
        if not rows:
            return _empty

        sim_ids = sorted(set(r[0] for r in rows))
        gen_map = _sim_ids_to_gen_map(sim_ids)

        data = []
        for sim_id, _step, gtime, state in rows:
            gen = state.get("generation")
            if gen is None:
                gen = gen_map.get(sim_id, 0)
            try:
                val_list = _dig(state, observable)
                if not isinstance(val_list, (list, tuple)) or idx >= len(val_list):
                    continue
                data.append({
                    "generation": int(gen),
                    "time": float(gtime if gtime is not None else 0.0),
                    "value": float(val_list[idx]),
                })
            except (KeyError, TypeError, IndexError):
                continue

        if not data:
            return _empty

        df = pl.DataFrame(data).sort(["generation", "time"])
        df = _cumulative_time(df)
        return df.select(["generation", "time", "abs_time", "value"])

    # ------------------------------------------------------------------
    # Backend dispatch helpers
    # ------------------------------------------------------------------

    def _backend_select_element(self, col_name: str, idx: int) -> pl.DataFrame:
        """Dispatch single-element extraction to the appropriate backend."""
        if self._kind == "parquet":
            return self._parquet_select_element(col_name, idx)
        if self._kind == "sqlite":
            return self._sqlite_select_element(col_name, idx)
        if self._kind == "xarray":
            return self._xarray_select_element(col_name, idx)
        raise ValueError(f"Unknown kind: {self._kind!r}")

    # ------------------------------------------------------------------
    # Public catalog API
    # ------------------------------------------------------------------

    # Maps index_by.type → (catalog_observable, data_col_name).
    # ``catalog_observable`` is passed to catalog()/resolve_id() to look up
    # the element index; ``data_col_name`` is the parquet/sqlite column that
    # holds the raw array (with ``__`` separators, NOT dots).
    _SELECT_TYPE_MAP: dict[str, tuple[str, str]] = {
        "bulk_id": ("bulk", "bulk__count"),
        "monomer_id": ("listeners.monomer_counts", "listeners__monomer_counts"),
    }

    # Maps catalog observable name → raw data column name (``__``-separated).
    # Used by aggregate_series to resolve the correct data column for observables
    # whose catalog name differs from their data column name (e.g. "bulk" whose
    # count data lives in "bulk__count", not "bulk").
    _OBSERVABLE_DATA_COL_MAP: dict[str, str] = {
        "bulk": "bulk__count",
    }

    def _resolve_observable_col(self, observable: str) -> tuple[str, str]:
        """Return ``(catalog_observable, data_col_name)`` for *observable*.

        For most observables the data column is simply
        ``observable.replace('.', '__')``.  Special cases (e.g. ``"bulk"``)
        are handled via :attr:`_OBSERVABLE_DATA_COL_MAP`.

        Args:
            observable: Dotted observable name (e.g. ``"bulk"``,
                ``"listeners.monomer_counts"``).

        Returns:
            ``(catalog_observable, data_col_name)`` where *catalog_observable*
            is passed to :py:meth:`resolve_id` and *data_col_name* is the raw
            column name used by the backend (``__``-separated, no dots).
        """
        data_col = self._OBSERVABLE_DATA_COL_MAP.get(
            observable, observable.replace(".", "__")
        )
        return observable, data_col

    def catalog(self, observable: str) -> list | None:
        """Return the element-name catalog for an array observable, or ``None``.

        Args:
            observable: Observable id as returned by :py:meth:`observables`
                (dotted path, e.g. ``"listeners.monomer_counts"``).  Use
                ``"bulk"`` to get the bulk-molecule id list.

        Returns:
            Ordered list of element names when a catalog is present,
            ``None`` otherwise.
        """
        if self._kind == "parquet":
            return self._parquet_catalog(observable)
        if self._kind == "xarray":
            return self._xarray_catalog(observable)
        if self._kind == "sqlite":
            return self._sqlite_catalog(observable)
        raise ValueError(f"Unknown kind: {self._kind!r}")

    def resolve_id(self, observable: str, element_id: str) -> int:
        """Return the 0-based index of *element_id* in the observable's catalog.

        Args:
            observable: Observable id (dotted path or ``"bulk"``).
            element_id: Name to look up.

        Returns:
            0-based integer index.

        Raises:
            :exc:`CatalogUnavailable`: No catalog exists for this observable.
            :exc:`IdNotInCatalog`: *element_id* is not in the catalog.
        """
        cat = self.catalog(observable)
        if cat is None:
            raise CatalogUnavailable(
                f"No catalog available for {observable!r} in {self._kind!r} store"
            )
        try:
            return cat.index(element_id)
        except ValueError:
            raise IdNotInCatalog(
                f"{element_id!r} not in catalog for {observable!r}"
            ) from None

    def select(self, index_by: dict) -> pl.DataFrame:
        """Return a ``[generation, time, abs_time, value]`` series for one element.

        Args:
            index_by: Selector dict with mandatory key ``"type"`` and ``"value"``.

            Supported types:

            * ``"bulk_id"`` — *value* is a bulk molecule id; observable is
              ``"bulk"`` / data column ``bulk__count``.
            * ``"monomer_id"`` — *value* is a monomer name; observable is
              ``"listeners.monomer_counts"``.
            * ``"listener_id"`` — *value* is a named element of a listener
              vector; requires ``"observable"`` key in *index_by*.
            * ``"literal_index"`` — *value* is a 0-based integer; requires
              ``"observable"`` key.  No catalog lookup is performed.

        Returns:
            Polars DataFrame with columns ``[generation, time, abs_time, value]``,
            same shape as :py:meth:`series`.

        Raises:
            :exc:`IdNotInCatalog`: Unknown id for catalog-backed types.
            :exc:`CatalogUnavailable`: Catalog required but absent.
        """
        type_ = index_by["type"]
        value = index_by["value"]

        if type_ == "literal_index":
            obs = index_by.get("observable")
            if obs is None:
                raise ValueError(
                    "literal_index selector requires 'observable' in index_by"
                )
            col_name = obs.replace(".", "__")
            return self._backend_select_element(col_name, int(value))

        if type_ in self._SELECT_TYPE_MAP:
            catalog_obs, data_col = self._SELECT_TYPE_MAP[type_]
            idx = self.resolve_id(catalog_obs, str(value))
            return self._backend_select_element(data_col, idx)

        if type_ == "listener_id":
            obs = index_by.get("observable")
            if not obs:
                raise ValueError(
                    "listener_id selector requires 'observable' in index_by"
                )
            idx = self.resolve_id(obs, str(value))
            col_name = obs.replace(".", "__")
            return self._backend_select_element(col_name, idx)

        raise ValueError(f"Unknown index_by type: {type_!r}")

    def aggregate_series(
        self, observable: str, op: str, over: list[str]
    ) -> pl.DataFrame:
        """Return a per-tick series reduced across named elements.

        Args:
            observable: Observable id whose catalog will be used to resolve
                *over* ids (dotted path, e.g. ``"listeners.monomer_counts"``).
            op: Reduction operator — one of ``"sum"``, ``"mean"``, ``"max"``,
                ``"min"``.
            over: Ordered list of element names to include.  **Every** name
                must exist in the catalog; an unknown name raises
                :exc:`IdNotInCatalog` (never silently dropped).

        Returns:
            Polars DataFrame ``[generation, time, abs_time, value]``, same
            shape as :py:meth:`series`.

        Raises:
            :exc:`IdNotInCatalog`: Any element in *over* is unknown.
            :exc:`CatalogUnavailable`: No catalog for *observable*.
            :exc:`ValueError`: Unknown *op*.
        """
        if op not in {"sum", "mean", "max", "min"}:
            raise ValueError(f"Unknown aggregate op: {op!r}. Use sum/mean/max/min.")

        # Resolve catalog observable and data column — handles cases where the
        # catalog observable name differs from the raw data column (e.g. "bulk"
        # whose count data lives in "bulk__count", not "bulk").
        catalog_obs, col_name = self._resolve_observable_col(observable)

        # Resolve all ids up-front — raises IdNotInCatalog for any unknown.
        indices = [self.resolve_id(catalog_obs, id_) for id_ in over]

        if self._kind == "parquet":
            return self._parquet_aggregate_elements(col_name, indices, op)

        # General fallback: pull per-element series and reduce in polars.
        dfs = [self._backend_select_element(col_name, idx) for idx in indices]
        return _reduce_series(dfs, op)
