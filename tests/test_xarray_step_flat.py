"""Load-bearing test: XArrayEmitter runs as a normal process-bigraph Step.

This is the crux of the interchangeable-emitter framework (Task 2). It proves
that XArrayEmitter can be injected into a flat composite via the SAME standard
mechanism every other Emitter uses (``collect_input_ports`` wiring + the
composite step network), with NO v2ecoli colony/lineage envelope and NO
external driver loop. The composite itself drives the emitter via ``run(N)``.
"""

import pytest

pytest.importorskip("xarray")
pytest.importorskip("zarr")

import xarray as xr  # noqa: E402

from bigraph_schema import allocate_core, set_path  # noqa: E402
from process_bigraph.composite import Composite, Process  # noqa: E402
from process_bigraph.emitter import collect_input_ports  # noqa: E402


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
    """Inject an emitter into a composite the standard way, with full config.

    Mirrors ``process_bigraph.emitter.add_emitter_to_composite`` (collect input
    ports, merge an emitter step spec, rebuild the step network) but lets the
    caller supply the rich XArrayEmitter config alongside the auto-derived
    ``emit`` wiring. Returns the wires so the test can build a matching view.
    """
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
    return wires


def test_xarray_emitter_runs_as_flat_step(tmp_path):
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

    # Build the emitter config FIRST so the view matches the flat wires.
    wires = collect_input_ports(composite.state)
    emit_ports = [p for p in wires if p != "global_time"]
    assert "counter_store/value" in emit_ports

    store = str(tmp_path / "flat.zarr")
    config = {
        "out_uri": store,
        "strategy": "flat",       # opt-out of colony/lineage semantics
        "emit_root": [],          # consume the flat wired state directly
        "transducer": {
            "predicate": [[{"subsample": {"interval": 1}}]],
            "buffer": {"size": 3},
        },
        "view": view_from_emit_paths(emit_ports, dtype="<f8"),
        "writer": {
            "backend": "zarr",
            "store": store,
            "buffers_per_chunk": 1,
            "backend_config": {"format": 3},
        },
        # Minimal metadata: enough to construct + alloc + open the store.
        "metadata": {"experiment_id": "flat-run"},
        "metadata_keys": [],
        "metadata_validators": {},
        "output_metadata": {},
        "debug": False,
    }

    _inject_emitter_as_step(composite, core, config, "local:XArrayEmitter")

    # Drive via the composite itself — NOT an external emitter loop.
    composite.run(6)

    emitter = composite.state["emitter"]["instance"]
    try:
        emitter.close(success=True)
    except Exception:
        pass

    # Read the zarr store back and confirm the scalar time series is present.
    tree = xr.open_datatree(store, engine="zarr")
    # Locate the single data variable across the whole tree.
    series = None
    for node in tree.subtree:
        for name, da in node.data_vars.items():
            vals = da.values.ravel().tolist()
            if len(vals) >= 6:
                series = vals
                break
        if series is not None:
            break

    assert series is not None, f"no time series written; tree=\n{tree}"
    assert len(series) >= 6
    # Counter increments by 1 each tick -> monotonic non-decreasing.
    assert all(b >= a for a, b in zip(series, series[1:])), series
    assert series[-1] > series[0]
