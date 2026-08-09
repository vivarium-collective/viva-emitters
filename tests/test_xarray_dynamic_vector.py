"""Generic emit-all path: lazy 1-D vector promotion + graceful drop.

These tests exercise the ``view_from_emit_paths(...)`` generic path (empty
``coords`` / ``output_metadata == {}``) the dashboard uses. In that path EVERY
port is allocated as a scalar buffer, so a port that emits a numpy vector used
to crash the transducer write with a broadcast/shape mismatch.

The fix: at write time, a scalar-allocated port (``VariableSpec.coord is None``)
that receives a stable 1-D vector is lazily *promoted* to a coord-bearing vector
variable (indistinguishable from a caller-supplied-coords variable), while a
port whose value is >1-D, or whose 1-D length changes across ticks, is
*dropped* (warn once, excluded from the store) instead of crashing.

The caller-supplied-coords path (v2ecoli ``output_metadata`` / ``LabeledArray``)
must remain untouched — see ``test_caller_coord_vector_still_works``.
"""

import warnings

import numpy as np
import pytest

pytest.importorskip("xarray")
pytest.importorskip("zarr")

import xarray as xr  # noqa: E402

from bigraph_schema import allocate_core, set_path  # noqa: E402
from process_bigraph.composite import Composite, Process  # noqa: E402
from process_bigraph.emitter import collect_input_ports  # noqa: E402


# --------------------------------------------------------------------------- #
# Minimal processes
# --------------------------------------------------------------------------- #


class Counter(Process):
    """Increments a scalar store by 1 each tick."""

    config_schema = {}

    def inputs(self):
        return {"value": "float"}

    def outputs(self):
        return {"value": "float"}

    def update(self, state, interval):
        return {"value": 1.0}


class VecProcess(Process):
    """Adds a fixed length-3 vector to an ``array[float]`` store each tick."""

    config_schema = {}

    def inputs(self):
        return {"v": "array[float]"}

    def outputs(self):
        return {"v": "array[float]"}

    def update(self, state, interval):
        return {"v": np.array([1.0, 2.0, 3.0])}


class MatrixProcess(Process):
    """Adds a fixed (2, 2) array to an ``array[float]`` store each tick.

    A 2-D per-tick value is not representable in the fixed-shape store, so the
    emitter must DROP this port (warn once, exclude) rather than crash.
    """

    config_schema = {}

    def inputs(self):
        return {"m": "array[float]"}

    def outputs(self):
        return {"m": "array[float]"}

    def update(self, state, interval):
        return {"m": np.zeros((2, 2))}


class GrowingVecProcess(Process):
    """Emits a 1-D vector whose LENGTH grows every tick (like a colony whose
    cell count changes). After the port is promoted to a fixed length, a later
    length change must DROP it (route back through the dynamic path) rather than
    crash the strict direct write."""

    config_schema = {}

    def __init__(self, config=None, core=None):
        super().__init__(config, core)
        self._n = 2

    def inputs(self):
        return {"v": "array[float]"}

    def outputs(self):
        return {"v": "array[float]"}

    def update(self, state, interval):
        out = np.ones(self._n, dtype=float)
        self._n += 1
        return {"v": out}


class StringProcess(Process):
    """Emits a STRING scalar (e.g. a colony agent id like 'a_0') each tick.

    Under the generic emit-all ("node") path the port is scalar-allocated as a
    numeric buffer, so a non-numeric scalar is not representable — the emitter
    must DROP it rather than crash on xarray's float coercion.
    """

    config_schema = {}

    def inputs(self):
        return {"aid": "string"}

    def outputs(self):
        return {"aid": "string"}

    def update(self, state, interval):
        return {"aid": "a_0"}




# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _inject_emitter_as_step(composite, core, config, address):
    """Inject an emitter into a composite the standard way, with full config."""
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


def _base_config(store, emit_ports, *, coords=None, output_metadata=None):
    from viva_emitters.xarray_emitter.view import view_from_emit_paths

    return {
        "out_uri": store,
        "strategy": "flat",
        "emit_root": [],
        "transducer": {
            "predicate": [[{"subsample": {"interval": 1}}]],
            "buffer": {"size": 3},
        },
        "view": view_from_emit_paths(emit_ports, dtype="<f8", coords=coords),
        "writer": {
            "backend": "zarr",
            "store": store,
            "buffers_per_chunk": 1,
            "backend_config": {"format": 3},
        },
        "metadata": {"experiment_id": "flat-run"},
        "metadata_keys": [],
        "metadata_validators": {},
        "output_metadata": output_metadata or {},
        "debug": False,
    }


def _all_data_vars(tree):
    """Map leaf var-name -> DataArray across the whole DataTree."""
    out = {}
    for node in tree.subtree:
        for name, da in node.data_vars.items():
            out[(str(node.path), name)] = da
    return out


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #


def test_generic_path_scalar_plus_vector(tmp_path):
    """Case 1: scalar + 1-D vector (len 3) through the generic emit-all path.

    BEFORE the fix this crashes at ``child_vars[...][t_ix] = val``. AFTER the
    fix the store holds the vector var with a coord dim of length 3 (correct
    per-tick values) and the scalar var (correct per-tick values).
    """
    core = allocate_core()
    core.register_link("Counter", Counter)
    core.register_link("VecProcess", VecProcess)
    from viva_emitters.xarray_emitter import XArrayEmitter

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
        "vec": {
            "_type": "process",
            "address": "local:VecProcess",
            "config": {},
            "inputs": {"v": ["vec_store", "v"]},
            "outputs": {"v": ["vec_store", "v"]},
            "interval": 1.0,
        },
        "counter_store": {"value": 0.0},
        "vec_store": {"v": np.zeros(3, dtype=float)},
    }
    composite = Composite({"state": doc}, core=core)

    wires = collect_input_ports(composite.state)
    emit_ports = [p for p in wires if p != "global_time"]
    assert "counter_store/value" in emit_ports
    assert "vec_store/v" in emit_ports

    store = str(tmp_path / "flat.zarr")
    config = _base_config(store, emit_ports)
    _inject_emitter_as_step(composite, core, config, "local:XArrayEmitter")

    composite.run(6)
    emitter = composite.state["emitter"]["instance"]
    try:
        emitter.close(success=True)
    except Exception:
        pass

    tree = xr.open_datatree(store, engine="zarr")
    dvars = _all_data_vars(tree)

    # locate the vector var (2-D: time x coord) and the scalar var (1-D)
    vec_da = None
    scalar_da = None
    for (_, _name), da in dvars.items():
        if da.ndim == 2 and 3 in da.shape:
            vec_da = da
        elif da.ndim == 1 and da.values.size >= 6:
            scalar_da = da

    assert vec_da is not None, f"vector var missing; tree=\n{tree}"
    assert scalar_da is not None, f"scalar var missing; tree=\n{tree}"

    # vector: coord dim length 3, per-tick row == k * [1, 2, 3]
    coord_len = [s for d, s in vec_da.sizes.items() if s == 3][0]
    assert coord_len == 3
    rows = vec_da.values
    # order the two dims as (time, coord)
    if rows.shape[0] == 3 and rows.shape[1] != 3:
        rows = rows.T
    for k in range(rows.shape[0]):
        np.testing.assert_allclose(rows[k], k * np.array([1.0, 2.0, 3.0]))

    # scalar: monotonic increasing counter
    svals = scalar_da.values.ravel().tolist()
    assert all(b >= a for a, b in zip(svals, svals[1:])), svals
    assert svals[-1] > svals[0]


def test_generic_path_2d_port_dropped_with_warning(tmp_path):
    """Case 2: a 2-D port completes the run, is ABSENT from the store, warns."""
    core = allocate_core()
    core.register_link("Counter", Counter)
    core.register_link("MatrixProcess", MatrixProcess)
    from viva_emitters.xarray_emitter import XArrayEmitter

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
        "mat": {
            "_type": "process",
            "address": "local:MatrixProcess",
            "config": {},
            "inputs": {"m": ["mat_store", "m"]},
            "outputs": {"m": ["mat_store", "m"]},
            "interval": 1.0,
        },
        "counter_store": {"value": 0.0},
        "mat_store": {"m": np.zeros((2, 2), dtype=float)},
    }
    composite = Composite({"state": doc}, core=core)

    wires = collect_input_ports(composite.state)
    emit_ports = [p for p in wires if p != "global_time"]
    assert "mat_store/m" in emit_ports

    store = str(tmp_path / "drop.zarr")
    config = _base_config(store, emit_ports)
    _inject_emitter_as_step(composite, core, config, "local:XArrayEmitter")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        composite.run(6)  # must NOT raise
        emitter = composite.state["emitter"]["instance"]
        try:
            emitter.close(success=True)
        except Exception:
            pass

    # a drop warning was raised mentioning the 2-D port
    msgs = [str(w.message) for w in caught]
    assert any("mat_store/m" in m and "dropping" in m.lower() for m in msgs), msgs

    tree = xr.open_datatree(store, engine="zarr")
    dvars = _all_data_vars(tree)
    # the dropped port must NOT appear anywhere as a 2-D (time x 4) var
    for (node, _name), da in dvars.items():
        assert "mat_store" not in node, f"dropped port present: {node}"

    # the scalar counter still made it
    scalar = None
    for da in dvars.values():
        if da.ndim == 1 and da.values.size >= 6:
            scalar = da.values.ravel().tolist()
    assert scalar is not None and scalar[-1] > scalar[0]


def test_generic_path_string_scalar_dropped_no_crash(tmp_path):
    """A string scalar (colony agent id 'a_0') into a numeric buffer must be
    DROPPED (warn once), not crash the run on float coercion."""
    core = allocate_core()
    core.register_link("Counter", Counter)
    core.register_link("StringProcess", StringProcess)
    from viva_emitters.xarray_emitter import XArrayEmitter

    core.register_link("XArrayEmitter", XArrayEmitter)

    doc = {
        "counter": {
            "_type": "process", "address": "local:Counter", "config": {},
            "inputs": {"value": ["counter_store", "value"]},
            "outputs": {"value": ["counter_store", "value"]},
            "interval": 1.0,
        },
        "agent": {
            "_type": "process", "address": "local:StringProcess", "config": {},
            "inputs": {"aid": ["agent_store", "aid"]},
            "outputs": {"aid": ["agent_store", "aid"]},
            "interval": 1.0,
        },
        "counter_store": {"value": 0.0},
        "agent_store": {"aid": "a_0"},
    }
    composite = Composite({"state": doc}, core=core)

    wires = collect_input_ports(composite.state)
    emit_ports = [p for p in wires if p != "global_time"]
    assert "agent_store/aid" in emit_ports

    store = str(tmp_path / "str.zarr")
    config = _base_config(store, emit_ports)
    _inject_emitter_as_step(composite, core, config, "local:XArrayEmitter")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        composite.run(6)  # must NOT raise (pre-fix: ValueError float coercion)
        emitter = composite.state["emitter"]["instance"]
        try:
            emitter.close(success=True)
        except Exception:
            pass

    msgs = [str(w.message) for w in caught]
    assert any("agent_store/aid" in m and "dropping" in m.lower() for m in msgs), msgs

    tree = xr.open_datatree(store, engine="zarr")
    dvars = _all_data_vars(tree)
    for (node, _name), da in dvars.items():
        assert "agent_store" not in node, f"dropped string port present: {node}"
    # the numeric counter still made it through
    scalar = None
    for da in dvars.values():
        if da.ndim == 1 and da.values.size >= 6:
            scalar = da.values.ravel().tolist()
    assert scalar is not None and scalar[-1] > scalar[0]


def test_promoted_vector_length_change_dropped_no_crash(tmp_path):
    """A generic port promoted to a length-N vector, then emitting a DIFFERENT
    length (a colony cell divides), must be DROPPED — not crash the strict
    direct write. Exercises routing promoted ports back through _write_dynamic."""
    core = allocate_core()
    core.register_link("GrowingVecProcess", GrowingVecProcess)
    from viva_emitters.xarray_emitter import XArrayEmitter

    core.register_link("XArrayEmitter", XArrayEmitter)

    doc = {
        "grow": {
            "_type": "process", "address": "local:GrowingVecProcess", "config": {},
            "inputs": {"v": ["grow_store", "v"]},
            "outputs": {"v": ["grow_store", "v"]},
            "interval": 1.0,
        },
        "grow_store": {"v": np.ones(2, dtype=float)},
    }
    composite = Composite({"state": doc}, core=core)
    wires = collect_input_ports(composite.state)
    emit_ports = [p for p in wires if p != "global_time"]
    assert "grow_store/v" in emit_ports

    store = str(tmp_path / "grow.zarr")
    config = _base_config(store, emit_ports)
    _inject_emitter_as_step(composite, core, config, "local:XArrayEmitter")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        composite.run(5)  # must NOT raise once the length changes past promotion
        emitter = composite.state["emitter"]["instance"]
        try:
            emitter.close(success=True)
        except Exception:
            pass

    msgs = [str(w.message) for w in caught]
    assert any("grow_store/v" in m and "dropping" in m.lower() for m in msgs), msgs


def test_caller_coord_vector_still_works(tmp_path):
    """Case 3: caller-supplied coords vector path is unchanged (no promote/drop).

    Passing ``coords={"vec_store/v": [...]}`` to ``view_from_emit_paths`` sets
    ``metadata=True`` so the port is allocated as a coord-bearing vector BEFORE
    any write. The fix's lazy/drop branch must never touch this path.
    """
    core = allocate_core()
    core.register_link("VecProcess", VecProcess)
    from viva_emitters.xarray_emitter import XArrayEmitter

    core.register_link("XArrayEmitter", XArrayEmitter)

    doc = {
        "vec": {
            "_type": "process",
            "address": "local:VecProcess",
            "config": {},
            "inputs": {"v": ["vec_store", "v"]},
            "outputs": {"v": ["vec_store", "v"]},
            "interval": 1.0,
        },
        "vec_store": {"v": np.zeros(3, dtype=float)},
    }
    composite = Composite({"state": doc}, core=core)

    wires = collect_input_ports(composite.state)
    emit_ports = [p for p in wires if p != "global_time"]
    assert "vec_store/v" in emit_ports

    store = str(tmp_path / "coord.zarr")
    # caller supplies coords keyed by the emit-port name
    coords = {"vec_store/v": ["a", "b", "c"]}
    config = _base_config(store, emit_ports, coords=coords, output_metadata=coords)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _inject_emitter_as_step(composite, core, config, "local:XArrayEmitter")
        composite.run(6)
        emitter = composite.state["emitter"]["instance"]
        try:
            emitter.close(success=True)
        except Exception:
            pass

    # no drop/promote warnings on the caller-coord path
    assert not [w for w in caught if "dropping" in str(w.message).lower()]

    tree = xr.open_datatree(store, engine="zarr")
    dvars = _all_data_vars(tree)
    vec_da = None
    for da in dvars.values():
        if da.ndim == 2 and 3 in da.shape:
            vec_da = da
    assert vec_da is not None, f"caller-coord vector missing; tree=\n{tree}"
    rows = vec_da.values
    if rows.shape[0] == 3 and rows.shape[1] != 3:
        rows = rows.T
    for k in range(rows.shape[0]):
        np.testing.assert_allclose(rows[k], k * np.array([1.0, 2.0, 3.0]))


def test_late_scalar_to_vector_promotion_warns():
    """Case 4: XarrayBuffer warns when scalar→vector promotion occurs at buf_tix > 0.

    Tests the ``_promote_port`` warning path directly, without going through
    the full composite / bigraph schema machinery.  The scenario:

    - ``write(buf_tix=0, …, scalar)``   — port allocated as scalar; no promotion
    - ``write(buf_tix=1, …, 1-D array)`` — first sight of a vector, buf_tix > 0
      → lossy promotion (earlier scalar sample silently zeroed) → must warn once.

    Assertions:
    (a) the second write does not raise,
    (b) a warning is raised whose text names the port (``lv_store/v``) and
        mentions discarded / earlier samples.
    """
    from viva_emitters.xarray_emitter.transducer import XarrayBuffer
    from viva_emitters.xarray_emitter.view import ForestView, view_from_emit_paths
    from viva_emitters.xarray_emitter.storage import XarrayStoragePartition
    from viva_emitters.xarray_emitter._base import StoragePartition

    emit_ports = ["lv_store/v"]
    view_config = view_from_emit_paths(emit_ports, dtype="<f8")
    forest_view = ForestView.from_dict(view_config)

    base_part = StoragePartition(experiment_id="test", variant=0, lineage_seed=0)
    xpart = XarrayStoragePartition.cast(base_part)

    buf = XarrayBuffer(view=forest_view, emit_root=())
    buf.assemble(xpart, {})
    buf.alloc(4, {})

    # tick 0: scalar → allocated as scalar; no promotion (buf_tix == 0)
    buf.write(0, 0, 0.0, {"lv_store/v": np.float64(7.0)})

    # tick 1: 1-D vector → late promotion (buf_tix == 1 > 0) → must warn
    with pytest.warns(UserWarning, match=r"lv_store/v") as warning_list:
        buf.write(1, 1, 1.0, {"lv_store/v": np.array([1.0, 2.0, 3.0])})

    # (a) no crash; (b) warning text mentions discarded / earlier samples
    msgs = [str(w.message) for w in warning_list]
    assert any(
        "discarded" in m or "earlier" in m for m in msgs
    ), f"expected 'discarded'/'earlier' in warning text, got: {msgs}"
