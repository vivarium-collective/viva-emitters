"""Open-path heal for the multi-seed-gang ``_check_group`` residual.

One emitter drives a lineage; at each division ``advance_generation`` consolidates
the finished generation, then opens the next via ``_open_store`` ->
``_open_group(use_consolidated=True)`` -> ``_check_group``, which finds the prior
generation's time-coordinate array ONLY through the ON-DISK consolidated manifest.

Observed on the multi-seed ray-mnp gang (not single-seed/local): a generation
opens and its ``_check_group`` raises ``FileNotFoundError`` "Missing path from
previous generation" even though the prior generation's data is physically on the
store -- the on-disk consolidated manifest was stale/incomplete when the next
generation read it. (Whether from an S3 manifest write/read gap or an incremental
reconsolidate miss, the symptom is identical: manifest missing, data present.)

The fix HEALS at open: when the manifest lacks the prior generation but the live
store has it, rebuild the manifest from the live store and proceed; a prior
generation genuinely absent from the store still fails loud.

These tests construct the exact bug state LOCALLY -- build a lineage, then strip a
generation's time-coordinate entry from the on-disk manifest while leaving its
data -- so the heal is exercised deterministically without a gang or S3.
"""

import pytest

pytest.importorskip("xarray")
pytest.importorskip("zarr")

import zarr
from bigraph_schema import allocate_core
from zarr.core.group import ConsolidatedMetadata
from zarr.core.sync import sync

from viva_emitters.xarray_emitter import XArrayEmitter
from viva_emitters.xarray_emitter.zarr_writer import (
    _replace_consolidated_metadata,
)

BUF_SIZE = 600
N_EMITS_PER_GEN = 5
UNIT = "[fg]"
INDEP = "experiment_id=lineage-run/variant=0/lineage_seed=0"  # partition.independent_path


def _config(store, agent_id):
    return {
        "emit": {}, "out_uri": store, "strategy": "colony",
        "emit_root": ["agents", agent_id],
        "transducer": {"predicate": [[{"subsample": {"interval": 1}}]],
                       "buffer": {"size": BUF_SIZE}},
        "view": [{"root": ("listeners",), "metadata": False,
                  "variables": {"mass": [{"path": "listeners/mass",
                                          "dtype": "<f8", "unit": UNIT}]}}],
        "writer": {"backend": "zarr", "store": store, "buffers_per_chunk": 1,
                   "backend_config": {"format": 3}},
        "metadata": {"experiment_id": "lineage-run", "variant": 0,
                     "lineage_seed": 0, "agent_id": agent_id},
        "metadata_keys": [], "metadata_validators": {},
        "output_metadata": {}, "debug": False,
    }


def _build_lineage(store, n_gens):
    """Drive ONE emitter through n_gens; leaves a store with a COMPLETE manifest."""
    core = allocate_core()
    aid = "0"
    em = XArrayEmitter(_config(store, aid), core=core)
    t = 0
    for gen in range(n_gens):
        for _ in range(N_EMITS_PER_GEN):
            em.update({"global_time": float(t),
                       "agents": {aid: {"listeners": {"mass": 10.0 + t}}}})
            t += 1
        if gen < n_gens - 1:
            aid = aid + "0"
            em.advance_generation(agent_id=aid, success=True)
    em.close(success=True)


def _manifest_keys(store):
    g = zarr.open_group(store, path=INDEP, zarr_format=3,
                        use_consolidated=True, mode="r")
    cm = g.metadata.consolidated_metadata
    return None if cm is None else set(cm.metadata.keys())


def _strip_from_manifest(store, key):
    """Remove one top-level entry from the ON-DISK consolidated manifest, leaving
    all array data in place -- reproduces 'manifest incomplete, data present'."""
    g = zarr.open_group(store, path=INDEP, zarr_format=3,
                        use_consolidated=True, mode="a")
    cm = g.metadata.consolidated_metadata
    assert cm is not None and key in cm.metadata, f"{key} not in manifest {set(cm.metadata)}"
    kept = {k: v for k, v in cm.metadata.items() if k != key}
    ag = _replace_consolidated_metadata(
        g._async_group, ConsolidatedMetadata(metadata=kept))
    sync(ag._save_metadata())


def _open_next_gen(store, agent_id):
    """Open a FRESH emitter at the given generation (== len(agent_id)) and emit
    one row -- exercises _open_store -> _check_group against the on-disk manifest."""
    core = allocate_core()
    em = XArrayEmitter(_config(store, agent_id), core=core)
    em.update({"global_time": 999.0,
               "agents": {agent_id: {"listeners": {"mass": 1.0}}}})
    em.close(success=True)


def test_open_heals_incomplete_manifest(tmp_path):
    """Prior gen present on store but stripped from the manifest: opening the next
    generation HEALS (rebuilds the manifest) instead of raising, and the repaired
    manifest contains the prior gen's time coordinate again."""
    store = str(tmp_path / "l.zarr")
    _build_lineage(store, n_gens=2)          # gens 1,2 on store, complete manifest
    _strip_from_manifest(store, "emitstep_gen=2")  # break gen2's time coord entry
    assert "emitstep_gen=2" not in (_manifest_keys(store) or set())

    # Fresh open of gen3 (agent "000") -> _check_group looks for gen2's time coord.
    _open_next_gen(store, "000")             # must NOT raise (heal)

    # The manifest was repaired from the live store.
    assert "emitstep_gen=2" in (_manifest_keys(store) or set())


def test_open_still_fails_when_prior_gen_truly_absent(tmp_path):
    """A prior generation genuinely absent from the store must STILL fail loud --
    the heal only covers a manifest gap, never a generation that never persisted."""
    store = str(tmp_path / "l.zarr")
    _build_lineage(store, n_gens=1)          # only gen1 on store
    # Open gen3 (agent "000"): its parent gen2 never existed -> real absence,
    # which must still raise the loud FileNotFoundError (the heal never masks it).
    with pytest.raises(FileNotFoundError, match="Missing path from previous generation"):
        _open_next_gen(store, "000")
