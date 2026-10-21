"""Lightweight long-form Parquet emitter for process-bigraph composites.

Unlike the vEcoli-style :class:`~viva_emitters.parquet_emitter.ParquetEmitter`
(a per-tick, hive-partitioned dataset keyed by experiment/variant/seed/
generation, read back through DuckDB, needing the heavy ``[parquet]`` extra),
this emitter writes a **single tidy long-form Parquet file** and needs only
``pyarrow``.

It subclasses :class:`~process_bigraph.emitter.RAMEmitter`, so it keeps the full
in-memory history — ``query()`` / ``gather_emitter_results()`` keep working
exactly as with a RAMEmitter — and it additionally serializes that history to a
Parquet file. The write happens on ``close(success=True)`` and, by default,
whenever results are gathered (``query()``), so the file "just appears" even in
pipelines that only ever call ``gather_emitter_results`` and never ``close``.

Two output shapes are supported, chosen by config:

* **nested-timecourse** (``path_levels`` set) — when an emitted value is itself a
  nested dict of pre-computed time series
  ``{d1: {d2: ... {var: [...], <time_key>: [...]}}}`` (e.g. the BioModels corpus
  ``results`` store, nested ``biomodel_id -> job -> engine -> {variable: series}``),
  the nesting path is split into the ``path_levels`` columns, the ``time_key``
  array becomes the ``time`` column, and its sibling arrays become ``value`` rows
  tagged by ``variable``. With ``path_levels=["biomodel_id", "job", "engine"]``
  this reproduces the corpus schema
  ``(biomodel_id, job, engine, variable, time, value)``.

* **tick-stacked** (default, no ``path_levels``) — the natural long form for a
  time-stepped composite: each recorded tick contributes rows; every numeric
  scalar leaf of the emitted state tree becomes a ``(column, step, value)`` row
  where ``column`` is the dotted path to the leaf. (Array leaves are expanded
  with an extra ``index`` position per element.)

Only ``pyarrow`` is required (the ``[parquet-lite]`` extra).
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError as e:  # pragma: no cover - exercised only without the extra
    raise ImportError(
        "viva_emitters.long_form_parquet_emitter requires the [parquet-lite] "
        "extra. Install with: pip install 'viva-emitters[parquet-lite]'. "
        f"(missing: {e.name})"
    ) from e

from process_bigraph.emitter import RAMEmitter

__all__ = ["LongFormParquetEmitter", "read_long_form"]

# Top-level state keys that are bookkeeping, never data, and are dropped before
# flattening the tick-stacked form.
_BOOKKEEPING = frozenset({"global_time", "time"})


def _thin(seq: Sequence[float], max_points: int) -> List[float]:
    """Uniformly downsample a sequence to at most ``max_points`` points.

    ``max_points <= 0`` disables downsampling (returns the sequence as a list).
    """
    out = list(seq)
    n = len(out)
    if max_points <= 0 or n <= max_points:
        return out
    stride = n / max_points
    return [out[int(i * stride)] for i in range(max_points)]


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _as_float_list(seq: Any) -> Optional[List[float]]:
    """Coerce a 1-D sequence of numbers to a list of floats, else ``None``."""
    if isinstance(seq, (list, tuple)):
        try:
            return [float(x) for x in seq]
        except (TypeError, ValueError):
            return None
    # numpy array without importing numpy at module import time
    tolist = getattr(seq, "tolist", None)
    if callable(tolist):
        try:
            flat = tolist()
        except Exception:
            return None
        if isinstance(flat, (list, tuple)):
            try:
                return [float(x) for x in flat]
            except (TypeError, ValueError):
                return None
    return None


class LongFormParquetEmitter(RAMEmitter):
    """RAMEmitter that also writes a single tidy long-form Parquet file.

    Config (beyond RAMEmitter's ``subsample`` / ``max_len``):

    ``file_path``     output directory (created if missing). Default ``./out``.
    ``parquet_file``  file name within ``file_path``. Default ``history.parquet``.
    ``source_key``    top-level emitted-state key to serialize (e.g. ``results``);
                      ``None`` (default) serializes the whole state tree.
    ``path_levels``   list of column names for the nested-timecourse form; when
                      set, the serializer descends that many dict levels and the
                      remaining leaf dict is ``{variable: series}``. When unset,
                      the tick-stacked form is used.
    ``time_key``      leaf key holding the shared x-axis in nested form. Default
                      ``time``.
    ``variable_col`` / ``time_col`` / ``value_col`` / ``column_col`` / ``step_col``
                      / ``index_col`` — output column names (sensible defaults).
    ``value_dtype``   ``float32`` (default) or ``float64`` for time/value columns.
    ``compression``   parquet codec. Default ``zstd``.
    ``max_points``    downsample each series to at most N points (nested form) or
                      cap array-leaf expansion (tick form). ``0`` = no limit.
    ``write_on_query``  write the parquet whenever ``query()`` runs. Default True.
    """

    config_schema = {
        **RAMEmitter.config_schema,
        "file_path": {"_type": "string", "_default": "./out"},
        "parquet_file": {"_type": "string", "_default": "history.parquet"},
        "source_key": {"_type": "string", "_default": None},
        "path_levels": {"_type": "list", "_default": None},
        "time_key": {"_type": "string", "_default": "time"},
        "variable_col": {"_type": "string", "_default": "variable"},
        "time_col": {"_type": "string", "_default": "time"},
        "value_col": {"_type": "string", "_default": "value"},
        "column_col": {"_type": "string", "_default": "column"},
        "step_col": {"_type": "string", "_default": "step"},
        "index_col": {"_type": "string", "_default": "index"},
        "value_dtype": {"_type": "string", "_default": "float32"},
        "compression": {"_type": "string", "_default": "zstd"},
        "max_points": {"_type": "integer", "_default": 0},
        "write_on_query": {"_type": "boolean", "_default": True},
    }

    @classmethod
    def emitter_contract(cls):
        from viva_emitters.contract import EmitterContract

        return EmitterContract(output_kind="parquet", output_uri_config_key="parquet_file")

    def __init__(self, config, core):
        super().__init__(config, core)
        self.file_path = config.get("file_path") or "./out"
        self.parquet_file = config.get("parquet_file") or "history.parquet"
        os.makedirs(self.file_path, exist_ok=True)
        self.out_path = os.path.join(self.file_path, self.parquet_file)

        self.source_key = config.get("source_key")
        self.path_levels = list(config.get("path_levels") or []) or None
        self.time_key = config.get("time_key") or "time"
        self.variable_col = config.get("variable_col") or "variable"
        self.time_col = config.get("time_col") or "time"
        self.value_col = config.get("value_col") or "value"
        self.column_col = config.get("column_col") or "column"
        self.step_col = config.get("step_col") or "step"
        self.index_col = config.get("index_col") or "index"
        self.value_dtype = config.get("value_dtype") or "float32"
        self.compression = config.get("compression") or "zstd"
        mp = config.get("max_points")
        self.max_points = 0 if mp is None else int(mp)
        woq = config.get("write_on_query")
        self.write_on_query = True if woq is None else bool(woq)

        self._dirty = False  # True when history changed since the last write

    # -- capture -----------------------------------------------------------
    def update(self, state) -> Dict:
        result = super().update(state)
        # super().update appends to self.history on the subsample cadence.
        self._dirty = True
        return result

    def query(self, paths=None, schema=None, query=None):
        if self.write_on_query and self._dirty:
            self.write()
        return super().query(paths=paths, schema=schema, query=query)

    def close(self, success: bool = True) -> None:
        if success:
            self.write()
        parent_close = getattr(super(), "close", None)
        if callable(parent_close):
            try:
                parent_close(success=success)
            except TypeError:
                parent_close()

    # -- serialize ---------------------------------------------------------
    def write(self, path: Optional[str] = None) -> Optional[str]:
        """Serialize the recorded history to a Parquet file.

        Returns the written path, or ``None`` when there is nothing to write.
        """
        table = self._build_table()
        if table is None or table.num_rows == 0:
            return None
        out = path or self.out_path
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        pq.write_table(table, out, compression=self.compression)
        self._dirty = False
        return out

    def _value_type(self) -> "pa.DataType":
        return pa.float32() if self.value_dtype == "float32" else pa.float64()

    def _build_table(self) -> Optional["pa.Table"]:
        if not self.history:
            return None
        if self.path_levels:
            return self._build_nested_table()
        return self._build_tick_table()

    def _source(self, tick: dict) -> dict:
        if not isinstance(tick, dict):
            return {}
        if self.source_key:
            src = tick.get(self.source_key)
            return src if isinstance(src, dict) else {}
        return {k: v for k, v in tick.items() if k not in _BOOKKEEPING}

    # nested-timecourse: {d1: {... : {var: series, time_key: series}}}
    def _build_nested_table(self) -> Optional["pa.Table"]:
        levels = self.path_levels
        depth = len(levels)
        # The series are complete values per tick, so the final recorded tick
        # carries the full timecourse; serialize that one.
        src = self._source(self.history[-1])
        cols: Dict[str, list] = {lv: [] for lv in levels}
        cols[self.variable_col] = []
        cols[self.time_col] = []
        cols[self.value_col] = []

        def descend(node: Any, prefix: Tuple[str, ...]):
            if len(prefix) == depth:
                if not isinstance(node, dict):
                    return
                time_axis = _as_float_list(node.get(self.time_key))
                for var, series in node.items():
                    if var == self.time_key:
                        continue
                    ys = _as_float_list(series)
                    if ys is None:
                        continue
                    t = time_axis if time_axis is not None else list(range(len(ys)))
                    n = min(len(t), len(ys))
                    if n == 0:
                        continue
                    t = _thin(t[:n], self.max_points)
                    ys = _thin(ys[:n], self.max_points)
                    for lv, name in zip(levels, prefix):
                        cols[lv].extend([name] * len(ys))
                    cols[self.variable_col].extend([str(var)] * len(ys))
                    cols[self.time_col].extend(t)
                    cols[self.value_col].extend(ys)
                return
            if isinstance(node, dict):
                for key, child in node.items():
                    descend(child, prefix + (str(key),))

        descend(src, ())
        if not cols[self.value_col]:
            return None
        vtype = self._value_type()
        arrays = {lv: pa.array(cols[lv], type=pa.string()) for lv in levels}
        arrays[self.variable_col] = pa.array(cols[self.variable_col], type=pa.string())
        arrays[self.time_col] = pa.array(cols[self.time_col], type=vtype)
        arrays[self.value_col] = pa.array(cols[self.value_col], type=vtype)
        order = list(levels) + [self.variable_col, self.time_col, self.value_col]
        return pa.table({k: arrays[k] for k in order})

    # tick-stacked: numeric leaves of each recorded tick -> (column, step, value)
    def _build_tick_table(self) -> Optional["pa.Table"]:
        columns: List[str] = []
        steps: List[int] = []
        indices: List[int] = []
        values: List[float] = []
        any_index = False

        def flatten(node: Any, prefix: str, step: int):
            nonlocal any_index
            if isinstance(node, dict):
                for key, child in node.items():
                    sub = f"{prefix}.{key}" if prefix else str(key)
                    flatten(child, sub, step)
                return
            if _is_number(node):
                columns.append(prefix)
                steps.append(step)
                indices.append(0)
                values.append(float(node))
                return
            arr = _as_float_list(node)
            if arr is not None:
                arr = _thin(arr, self.max_points)
                for i, val in enumerate(arr):
                    any_index = True
                    columns.append(prefix)
                    steps.append(step)
                    indices.append(i)
                    values.append(val)

        for step, tick in enumerate(self.history):
            flatten(self._source(tick), "", step)

        if not values:
            return None
        vtype = self._value_type()
        data = {
            self.column_col: pa.array(columns, type=pa.string()),
            self.step_col: pa.array(steps, type=pa.int64()),
        }
        if any_index:
            data[self.index_col] = pa.array(indices, type=pa.int64())
        data[self.value_col] = pa.array(values, type=vtype)
        return pa.table(data)


def read_long_form(path: str) -> "pa.Table":
    """Read a long-form Parquet file written by :class:`LongFormParquetEmitter`."""
    return pq.read_table(path)
