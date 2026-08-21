"""Regression: metadata-emission fixes for generation > 1 and appending buffers.

Two bugs, both invisible on a single-buffer, generation-1 run (which is what the
other tests mostly exercise):

1. **generation > 1 arrays got default chunking.** The emitter coupled
   coordinate-data emission and *encoding* computation under one flag that is
   only true for the first buffer of generation 1. So every ``generation > 1``
   partition's freshly created arrays were written with NO encoding and fell
   back to Zarr's default chunks instead of the intended ``b * buf_size``.

2. **child-variable unit attributes were mis-keyed and then erased.** The unit
   annotation was stored under the per-generation ``generation=N`` key instead
   of the variable's own name (so every variable's unit clobbered a shared
   key), and appending buffers re-assembled the child node from data variables
   only — with no attrs — so ``dump_to_store`` erased the units written by the
   first buffer.

Driving two colony generations (``"0"`` then ``"01"``) with a unit-bearing view
and 6 emits per generation (buffer size 3 -> two full buffers, i.e. an appending
buffer) exercises both.
"""

import pytest

pytest.importorskip("xarray")
pytest.importorskip("zarr")

import xarray as xr  # noqa: E402
import zarr  # noqa: E402

from bigraph_schema import allocate_core  # noqa: E402

BUF_SIZE = 3
# > 1 so the intended time chunk (b * buf_size == 6) differs from the size of a
# single buffer (3). Otherwise the very first buffer write creates the array at
# its own length and the default chunking coincidentally matches the intended
# one, hiding the generation>1 encoding bug.
BUFFERS_PER_CHUNK = 2
N_EMITS = 6  # two full buffers => the second flush is an appending buffer
UNIT = "[fg]"


def _config(store, agent_id):
    return {
        "emit": {},
        "out_uri": store,
        "strategy": "colony",
        "emit_root": ["agents", agent_id],
        "transducer": {
            "predicate": [[{"subsample": {"interval": 1}}]],
            "buffer": {"size": BUF_SIZE},
        },
        "view": [
            {
                "root": ("listeners",),
                "metadata": False,
                "variables": {
                    "mass": [{"path": "listeners/mass", "dtype": "<f8", "unit": UNIT}],
                },
            }
        ],
        "writer": {
            "backend": "zarr",
            "store": store,
            "buffers_per_chunk": BUFFERS_PER_CHUNK,
            "backend_config": {"format": 3},
        },
        "metadata": {
            "experiment_id": "multigen-run",
            "variant": 0,
            "lineage_seed": 0,
            "agent_id": agent_id,
        },
        "metadata_keys": [],
        "metadata_validators": {},
        "output_metadata": {},
        "debug": False,
    }


def _drive(core, store, agent_id):
    from viva_emitters.xarray_emitter import XArrayEmitter

    emitter = XArrayEmitter(_config(store, agent_id), core=core)
    for i in range(N_EMITS):
        emitter.update({
            "global_time": float(i),
            "agents": {agent_id: {"listeners": {"mass": 10.0 + i}}},
        })
    try:
        emitter.close(success=True)
    except Exception:
        pass


def _arrays_ending(store, suffix):
    """All zarr arrays in `store` whose dotted path ends with `suffix`."""
    root = zarr.open_group(store, mode="r")
    out = {}

    def walk(group, path):
        for name, arr in group.arrays():
            out[f"{path}/{name}".lstrip("/")] = arr
        for name, sub in group.groups():
            walk(sub, f"{path}/{name}")

    walk(root, "")
    return {k: v for k, v in out.items() if k.endswith(suffix)}


@pytest.fixture
def two_gen_store(tmp_path):
    core = allocate_core()
    store = str(tmp_path / "multigen.zarr")
    _drive(core, store, "0")    # generation 1
    _drive(core, store, "01")   # generation 2
    return store


def test_generation_2_array_uses_intended_chunking(two_gen_store):
    """The generation-2 data array must be chunked at ``b * buf_size`` along
    time, not Zarr's default (which for these params would be the whole
    6-length array in a single chunk)."""
    gen2 = _arrays_ending(two_gen_store, "generation=2")
    assert gen2, "no generation=2 data array written"
    expected = BUFFERS_PER_CHUNK * BUF_SIZE  # == 3
    for path, arr in gen2.items():
        assert arr.shape[0] == N_EMITS, (path, arr.shape)
        assert arr.chunks[0] == expected, (
            f"{path}: time chunk {arr.chunks[0]} != intended {expected} "
            f"(shape {arr.shape}) -> generation>1 array fell back to default "
            f"chunking (the bug)")


def test_unit_attrs_keyed_by_variable_and_survive_appending_buffer(two_gen_store):
    """The ``mass`` unit must be stored under the variable name and still be
    present after the appending (second) buffer flush."""
    tree = xr.open_datatree(two_gen_store, engine="zarr", consolidated=False)

    unit_keys = set()
    for node in tree.subtree:
        for key, val in node.attrs.items():
            if val == UNIT:
                unit_keys.add(key)

    assert unit_keys, (
        "the 'mass' unit attribute is absent -> it was erased on the appending "
        "buffer (no attrs re-attached), or never written under a readable key")
    # Fix 1: keyed by the variable's own name, never the generation suffix.
    assert "mass" in unit_keys, f"unit not keyed by var name: {unit_keys}"
    assert not any("generation=" in k for k in unit_keys), (
        f"unit mis-keyed under a per-generation key: {unit_keys}")
