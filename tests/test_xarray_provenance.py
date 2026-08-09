"""Round-trip tests: XArrayEmitter writes run provenance to the zarr root attrs.

Task D1. A caller attaches immutable run provenance (which composite + what
config produced this output) via the ``provenance`` config field. When
non-empty it must be written VERBATIM to the store's ROOT group attributes
under the single key ``"provenance"`` and survive the async buffer +
consolidation, so a FRESH reader opening the finalized store sees it. Empty /
absent provenance => no ``provenance`` root attr at all.

These tests drive a real emit + finalize (via a composite, the same way
``test_xarray_step_flat`` does) and then reopen the store with an independent
``zarr.open_group`` handle — NOT the writer's — to prove durability on disk.
"""

import os

import pytest

pytest.importorskip("xarray")
pytest.importorskip("zarr")

import zarr  # noqa: E402

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


def _build_and_run(tmp_path, core, provenance):
    """Build a flat composite with an XArrayEmitter, run it, finalize.

    Returns ``(store, root_path)`` where ``root_path`` is the on-disk group the
    writer treats as ROOT (``store/<independent_path>``), suitable for a fresh
    ``zarr.open_group`` reader. ``provenance`` may be an empty dict.
    """
    from viva_emitters.xarray_emitter import XArrayEmitter
    from viva_emitters.xarray_emitter.view import view_from_emit_paths

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

    store = str(tmp_path / "prov.zarr")
    config = {
        "out_uri": store,
        "strategy": "flat",
        "emit_root": [],
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
        "metadata": {"experiment_id": "prov-run"},
        "metadata_keys": [],
        "metadata_validators": {},
        "output_metadata": {},
        "provenance": provenance,
        "debug": False,
    }

    # Inject the emitter as a step the standard way (mirrors
    # add_emitter_to_composite + the rich XArrayEmitter config).
    emit = {port: "node" for port in wires}
    emitter_state = {
        "_type": "step",
        "address": "local:XArrayEmitter",
        "config": {**config, "emit": emit},
        "inputs": wires,
    }
    path = ("emitter",)
    composite.merge({}, set_path({}, path, emitter_state))
    _, instance = core.traverse(composite.schema, composite.state, path)
    composite.step_paths[path] = instance
    composite.build_step_network()

    composite.run(6)

    emitter = composite.state["emitter"]["instance"]
    # The writer treats this group as ROOT for attr writes.
    root_path = os.path.join(store, str(emitter.partition.independent_path))
    emitter.close(success=True)
    return store, root_path


def test_provenance_written_to_root_attrs_verbatim(tmp_path):
    """Non-empty provenance persists verbatim to the ROOT attrs of the store.

    Proven with a FRESH ``zarr.open_group`` handle after finalize +
    consolidation — not the writer's in-memory group.
    """
    provenance = {
        "composite": "pkg.composites.x",
        "config": {"seed": 0, "condition": "with_aa"},
        "run_id": "r1",
    }
    core = allocate_core()
    _store, root_path = _build_and_run(tmp_path, core, provenance)

    root = zarr.open_group(root_path, mode="r")
    assert "provenance" in root.attrs
    assert dict(root.attrs["provenance"]) == provenance


def test_no_provenance_writes_no_root_attr(tmp_path):
    """Empty provenance => the finalized store has NO ``provenance`` root attr."""
    core = allocate_core()
    _store, root_path = _build_and_run(tmp_path, core, {})

    root = zarr.open_group(root_path, mode="r")
    assert "provenance" not in dict(root.attrs)
