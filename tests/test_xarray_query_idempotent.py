"""Regression tests for ``XArrayEmitter.query()`` as a pure, idempotent read.

Two defects motivated these tests (found building the results-wired slice):

1. ``query()`` used the normal append-based write path (``flush(final=False)``),
   which re-appended the in-memory buffer to the store on every call — so
   resolving results twice grew the row count (6 -> 9 -> 12) and corrupted the
   store. Resolving a ``results`` handle must be idempotent.
2. ``flush(final=False)`` asserts ``buf_tix == buf_size``, so ``query()`` raised
   ``AssertionError`` on any run whose tick count is not ``buffer_size``-aligned.

The fix routes the trailing buffer through the terminal (``final=True``) write
path exactly once (guarded by ``_flushed``), which truncates a partial buffer
instead of asserting, and never re-appends on a repeat read.
"""

import pytest

pytest.importorskip("xarray")
pytest.importorskip("zarr")

import xarray as xr  # noqa: E402

from bigraph_schema import allocate_core, set_path  # noqa: E402
from process_bigraph.composite import Composite, Process  # noqa: E402
from process_bigraph.emitter import collect_input_ports  # noqa: E402

BUFFER_SIZE = 3


class Counter(Process):
    """Minimal process: increments a scalar store by 1 each tick."""

    config_schema = {}

    def inputs(self):
        return {"value": "float"}

    def outputs(self):
        return {"value": "float"}

    def update(self, state, interval):
        return {"value": 1.0}


def _inject_emitter_as_step(composite, core, config, address):
    wires = collect_input_ports(composite.state)
    emit = {port: "node" for port in wires}
    emitter_state = {
        "_type": "step",
        "address": address,
        "config": {**config, "emit": emit},
        "inputs": wires,
    }
    path = ("emitter",)
    composite.merge({}, set_path({}, path, emitter_state))
    _, instance = core.traverse(composite.schema, composite.state, path)
    composite.step_paths[path] = instance
    composite.build_step_network()


def _run_emitter(tmp_path, n_ticks, *, buffer_size=BUFFER_SIZE):
    """Drive an XArrayEmitter for ``n_ticks`` and return the (still-open) instance."""
    from viva_emitters.xarray_emitter import XArrayEmitter
    from viva_emitters.xarray_emitter.view import view_from_emit_paths

    core = allocate_core()
    core.register_link("Counter", Counter)
    core.register_link("XArrayEmitter", XArrayEmitter)

    doc = {
        "counter": {
            "_type": "process",
            "address": "local:Counter",
            "config": {},
            "inputs": {"value": ["counter_store", "value"]},
            "outputs": {"value": ["counter_store", "value"]},
            "interval": 1.0,
        },
        "counter_store": {"value": 0.0},
    }
    composite = Composite({"state": doc}, core=core)

    wires = collect_input_ports(composite.state)
    emit_ports = [p for p in wires if p != "global_time"]
    store = str(tmp_path / "flat.zarr")
    config = {
        "out_uri": store,
        "strategy": "flat",
        "emit_root": [],
        "transducer": {
            "predicate": [[{"subsample": {"interval": 1}}]],
            "buffer": {"size": buffer_size},
        },
        "view": view_from_emit_paths(emit_ports, dtype="<f8"),
        "writer": {
            "backend": "zarr",
            "store": store,
            "buffers_per_chunk": 1,
            "backend_config": {"format": 3},
        },
        "metadata": {"experiment_id": "flat-run"},
        "metadata_keys": [],
        "metadata_validators": {},
        "output_metadata": {},
        "debug": False,
    }
    _inject_emitter_as_step(composite, core, config, "local:XArrayEmitter")
    composite.run(n_ticks)
    return composite.state["emitter"]["instance"], store


def _row_count(tree):
    """Length of the time axis of the first data variable found in the tree."""
    for node in tree.subtree:
        for _name, da in node.data_vars.items():
            return da.values.ravel().shape[0]
    raise AssertionError(f"no data variable in tree:\n{tree}")


def _first_series(tree):
    for node in tree.subtree:
        for _name, da in node.data_vars.items():
            return da.values.ravel().tolist()
    raise AssertionError(f"no data variable in tree:\n{tree}")


def test_query_is_idempotent_full_buffer(tmp_path):
    """Repeated query() on a buffer-aligned run returns the same data; no growth.

    Regression for defect #1: query() re-appended the buffer each call (6->9->12).
    """
    # 8 emit steps with buffer_size 3 leaves the buffer exactly full at query
    # time (two full buffers already on disk + a full trailing buffer in memory).
    emitter, _store = _run_emitter(tmp_path, 8)
    assert not emitter._closed

    first = _row_count(emitter.query())
    second = _row_count(emitter.query())
    third = _row_count(emitter.query())

    assert first == second == third, (
        f"query() must be idempotent, got {first} -> {second} -> {third}"
    )
    # Values must be identical too, not merely the same length.
    assert _first_series(emitter.query()) == _first_series(emitter.query())


def test_query_tolerates_partial_buffer(tmp_path):
    """query() must not raise on a run whose length is not buffer_size-aligned.

    Regression for defect #2: flush(final=False) asserted buf_tix == buf_size.
    """
    # 7 emit steps with buffer_size 3 leaves a partial (2-row) trailing buffer.
    emitter, _store = _run_emitter(tmp_path, 7)
    assert not emitter._closed
    assert 0 < emitter.transducer.buf_tix < emitter.transducer.buf_size

    tree = emitter.query()  # must NOT raise on the partial buffer
    n_partial = _row_count(tree)

    # Trailing partial rows are included (they are not silently dropped).
    assert n_partial >= emitter.transducer.buf_tix
    # And the read is still idempotent on a partial buffer.
    assert _row_count(emitter.query()) == n_partial


def test_query_does_not_grow_store_on_disk(tmp_path):
    """A second query() must not change the on-disk row count (no mutation)."""
    emitter, store = _run_emitter(tmp_path, 7)

    emitter.query()
    on_disk_after_first = _row_count(xr.open_datatree(store, engine="zarr"))
    emitter.query()
    on_disk_after_second = _row_count(xr.open_datatree(store, engine="zarr"))

    assert on_disk_after_first == on_disk_after_second


def test_close_after_query_succeeds_without_double_count(tmp_path):
    """close(success=True) after query() still finalizes and does not re-append.

    Confirms the normal close() semantics are preserved and the trailing buffer
    lands on disk exactly once across a query()+close() sequence.
    """
    emitter, store = _run_emitter(tmp_path, 7)

    rows_from_query = _row_count(emitter.query())
    emitter.close(success=True)
    assert emitter._closed

    rows_after_close = _row_count(xr.open_datatree(store, engine="zarr"))
    assert rows_after_close == rows_from_query
    # A query() after close() is still a safe, idempotent read.
    assert _row_count(emitter.query()) == rows_after_close
