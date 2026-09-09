"""Lineage-scoped XArrayEmitter: one emitter drives a whole cell lineage,
advancing generation internally (``advance_generation``) instead of the driver
building and closing a fresh emitter per generation.

Regression for the CD2 zarr "zero-arrays + gen-crash" failure. Root cause: the
lineage driver wrapped each per-generation ``close()`` in
``except AssertionError: pass`` (assuming "already on disk"); when a generation
failed to persist, the next generation's store re-open ``_check_group`` crashed
and the store held only group skeletons. ``advance_generation`` moves the
per-generation finalize (flush -> mark_success -> consolidate) into the emitter
and GUARANTEES it runs before the next generation opens, so:

  * every generation's arrays (data + time coordinate) are durably on disk,
  * each finished generation is readable *before* the lineage completes
    (incremental durability, like the parquet emitter), and
  * a SHORT generation (fewer emits than ``buffer.size``) still persists.

These are exactly the properties the zero-arrays store lacked.
"""

import pytest

pytest.importorskip("xarray")
pytest.importorskip("zarr")

import xarray as xr  # noqa: E402
import zarr  # noqa: E402

from bigraph_schema import allocate_core  # noqa: E402

from viva_emitters.xarray_emitter import XArrayEmitter  # noqa: E402

# buffer.size deliberately LARGER than the per-generation emit count, so each
# generation's data lives only in the trailing (never-filled) buffer -- the
# exact CD2 regime where the bug bit. If advance_generation did not force a
# terminal flush + consolidate, nothing would reach disk.
BUF_SIZE = 600
N_EMITS_PER_GEN = 5   # << BUF_SIZE
N_GENS = 3
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
        "view": [{
            "root": ("listeners",),
            "metadata": False,
            "variables": {
                "mass": [{"path": "listeners/mass", "dtype": "<f8", "unit": UNIT}],
            },
        }],
        "writer": {
            "backend": "zarr",
            "store": store,
            "buffers_per_chunk": 1,
            "backend_config": {"format": 3},
        },
        "metadata": {
            "experiment_id": "lineage-run", "variant": 0,
            "lineage_seed": 0, "agent_id": agent_id,
        },
        "metadata_keys": [], "metadata_validators": {},
        "output_metadata": {}, "debug": False,
    }


def _daughter(agent_id: str) -> str:
    """Phylogeny key one generation deeper (generation == len(agent_id))."""
    return agent_id + "0"


def _arrays(store):
    root = zarr.open_group(store, mode="r")
    out = {}

    def walk(group, path):
        for name, arr in group.arrays():
            out[f"{path}/{name}".lstrip("/")] = arr
        for name, sub in group.groups():
            walk(sub, f"{path}/{name}")

    walk(root, "")
    return out


def test_lineage_emitter_persists_every_generation(tmp_path):
    """One emitter, N short generations via advance_generation -> every
    generation's data + time coordinate is on disk with nonzero chunks."""
    core = allocate_core()
    store = str(tmp_path / "lineage.zarr")

    agent_id = "0"
    emitter = XArrayEmitter(_config(store, agent_id), core=core)
    t = 0
    for gen in range(N_GENS):
        for _ in range(N_EMITS_PER_GEN):
            emitter.update({
                "global_time": float(t),
                "agents": {agent_id: {"listeners": {"mass": 10.0 + t}}},
            })
            t += 1
        if gen < N_GENS - 1:
            agent_id = _daughter(agent_id)
            emitter.advance_generation(agent_id=agent_id, success=True)
    emitter.close(success=True)

    arrays = _arrays(store)
    # every generation must have a time-coordinate array AND a data array,
    # each with the full per-generation length and real (nonzero) bytes.
    for gen in range(1, N_GENS + 1):
        time_arrs = {k: a for k, a in arrays.items() if k.endswith(f"time_gen={gen}")}
        data_arrs = {k: a for k, a in arrays.items() if k.endswith(f"generation={gen}")}
        assert time_arrs, f"generation {gen}: no time array on disk (zero-arrays bug)"
        assert data_arrs, f"generation {gen}: no data array on disk (zero-arrays bug)"
        for k, a in {**time_arrs, **data_arrs}.items():
            assert a.shape[0] == N_EMITS_PER_GEN, (k, a.shape)
            assert a.nbytes_stored() > 0, f"{k}: zero stored bytes"


def test_finished_generation_is_readable_before_lineage_completes(tmp_path):
    """Incremental durability: after advancing off generation 1, generation 1's
    store is already consistent and readable (as parquet would be) -- it does
    NOT wait for the terminal close()."""
    core = allocate_core()
    store = str(tmp_path / "incremental.zarr")

    agent_id = "0"
    emitter = XArrayEmitter(_config(store, agent_id), core=core)
    for i in range(N_EMITS_PER_GEN):
        emitter.update({
            "global_time": float(i),
            "agents": {agent_id: {"listeners": {"mass": 10.0 + i}}},
        })
    # advance to generation 2 WITHOUT closing the lineage
    emitter.advance_generation(agent_id=_daughter(agent_id), success=True)

    # generation 1 is already durable: a fresh reader (no terminal close) sees it
    arrays = _arrays(store)
    gen1_time = {k: a for k, a in arrays.items() if k.endswith("time_gen=1")}
    assert gen1_time, "generation 1 not durable after advancing (lost until close)"
    for a in gen1_time.values():
        assert a.shape[0] == N_EMITS_PER_GEN
        assert a.nbytes_stored() > 0

    # and the store opens cleanly as a datatree mid-lineage
    tree = xr.open_datatree(store, engine="zarr", consolidated=False)
    assert tree is not None


def test_next_generation_check_group_finds_parent(tmp_path):
    """The gen-3 _check_group crash cannot occur: because advance_generation
    guarantees the parent generation persisted its time coordinate, opening the
    next generation never raises FileNotFoundError."""
    core = allocate_core()
    store = str(tmp_path / "checkgroup.zarr")

    agent_id = "0"
    emitter = XArrayEmitter(_config(store, agent_id), core=core)
    # three short generations back-to-back; each advance re-opens + _check_group
    for gen in range(3):
        for i in range(N_EMITS_PER_GEN):
            emitter.update({
                "global_time": float(gen * 100 + i),
                "agents": {agent_id: {"listeners": {"mass": 1.0 + i}}},
            })
        if gen < 2:
            agent_id = _daughter(agent_id)
            emitter.advance_generation(agent_id=agent_id, success=True)  # must not raise
    emitter.close(success=True)

    arrays = _arrays(store)
    assert any(k.endswith("generation=3") for k in arrays), "generation 3 missing"
