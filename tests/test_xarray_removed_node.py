"""A leaf path whose NODE is removed by a topology rewrite must be DROPPED, not
crash the run on the "Missing emit paths" check.

This is the biofilm-emergence case: free bacteria are top-level nodes with
`biomass` leaves; on attachment they become cells nested under a `biofilm` node,
so the original `.../bacterium_N/biomass` paths disappear. The xarray emitter
already drops shape-varying / non-representable ports; it must likewise drop a
path whose node vanished, rather than raising KeyError('Missing emit paths').
"""
import warnings

import pytest

pytest.importorskip("xarray")
pytest.importorskip("zarr")

from bigraph_schema import allocate_core  # noqa: E402
from process_bigraph.composite import Composite, Process  # noqa: E402
from process_bigraph.emitter import collect_input_ports  # noqa: E402

from test_xarray_dynamic_vector import _base_config, _inject_emitter_as_step  # noqa: E402


class NodeDropProcess(Process):
    """A colony that REMOVES one of its cells at tick 2 — a genuine place-graph
    rewrite that deletes a node (and its leaf paths) from the tree."""

    config_schema = {}

    def __init__(self, config=None, core=None):
        super().__init__(config, core)
        self._t = 0

    def inputs(self):
        return {"colony": "tree[node]"}

    def outputs(self):
        return {"colony": "overwrite[tree[node]]"}

    def update(self, state, interval):
        self._t += 1
        if self._t == 2:
            return {"colony": {k: v for k, v in state["colony"].items() if k != "cell_gone"}}
        return {}


def test_removed_node_path_dropped_no_crash(tmp_path):
    core = allocate_core()
    core.register_link("NodeDropProcess", NodeDropProcess)
    from viva_emitters.xarray_emitter import XArrayEmitter
    core.register_link("XArrayEmitter", XArrayEmitter)

    doc = {
        "colony_proc": {
            "_type": "process", "address": "local:NodeDropProcess", "config": {},
            "inputs": {"colony": ["colony_store"]},
            "outputs": {"colony": ["colony_store"]},
            "interval": 1.0,
        },
        "colony_store": {
            "_type": "tree[node]",
            "cell_gone": {"biomass": 1.0},
            "cell_keep": {"biomass": 1.0},
        },
    }
    composite = Composite({"state": doc}, core=core)
    wires = collect_input_ports(composite.state)
    emit_ports = [p for p in wires if p != "global_time"]

    store = str(tmp_path / "colony.zarr")
    config = _base_config(store, emit_ports)
    _inject_emitter_as_step(composite, core, config, "local:XArrayEmitter")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        composite.run(4)  # must NOT raise when cell_gone disappears at tick 2
        emitter = composite.state["emitter"]["instance"]
        try:
            emitter.close(success=True)
        except Exception:
            pass

    msgs = [str(w.message) for w in caught]
    assert any("cell_gone" in m and "dropping" in m.lower() for m in msgs), msgs
