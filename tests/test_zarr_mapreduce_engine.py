"""End-to-end smoke test for the ported zarr map-reduce ENGINE foundation.

Proves the reconciled storage glue (WorkflowConfig / WorkflowPaths / Substore /
XarrayStoragePartition workflow methods) and the read-side zarr_utils helpers
work against a workflow store actually produced by viva's XArrayEmitter — i.e.
the multi-lineage, multi-generation ``experiment_id=/variant=/lineage_seed=``
layout the engine walks.

The full ``ZarrMapReduce`` driver (~12 abstract hooks) is exercised by the
concrete moving-average pipeline downstream; here we cover the foundation:
substore discovery + async array read.
"""

import asyncio
from pathlib import Path

import pytest

pytest.importorskip("xarray")
pytest.importorskip("zarr")

from bigraph_schema import allocate_core  # noqa: E402

from viva_emitters.xarray_emitter.storage import (  # noqa: E402
    WorkflowConfig,
    WorkflowPaths,
    Substore,
    XarrayStoragePartition,
)
from viva_emitters.xarray_emitter.zarr_utils import (  # noqa: E402
    get_async_group,
)

EXPERIMENT_ID = "e"
N_LINEAGES = 2


def _emitter_config(store_root, agent_id, variant, lineage_seed):
    return {
        "emit": {},
        "out_uri": store_root,
        "strategy": "colony",
        "emit_root": ["agents", agent_id],
        "transducer": {
            "predicate": [[{"subsample": {"interval": 1}}]],
            "buffer": {"size": 3},
        },
        "view": [
            {
                "root": ("listeners",),
                "metadata": False,
                "variables": {
                    "mass": [{"path": "listeners/mass", "dtype": "<f8", "unit": "[fg]"}],
                },
            }
        ],
        "writer": {
            "backend": "zarr",
            "store": store_root,
            "buffers_per_chunk": 1,
            "backend_config": {"format": 3},
        },
        "metadata": {
            "experiment_id": EXPERIMENT_ID,
            "variant": variant,
            "lineage_seed": lineage_seed,
            "agent_id": agent_id,
        },
        "metadata_keys": [],
        "metadata_validators": {},
        "output_metadata": {},
        "debug": False,
    }


@pytest.fixture
def workflow_store(tmp_path):
    """A real workflow store: 1 variant x 2 lineages x 2 generations."""
    from viva_emitters.xarray_emitter import XArrayEmitter

    out_dir = str(tmp_path)
    store_root = str(tmp_path / EXPERIMENT_ID / "store")
    core = allocate_core()
    for lineage in range(N_LINEAGES):
        for agent_id in ("0", "01"):  # generations 1 and 2
            em = XArrayEmitter(
                _emitter_config(store_root, agent_id, 0, lineage), core=core)
            for i in range(6):  # two buffers of 3
                em.update({
                    "global_time": float(i),
                    "agents": {agent_id: {"listeners": {"mass": 10.0 + i}}},
                })
            try:
                em.close(success=True)
            except Exception:
                pass
    return {"out_dir": out_dir, "store_root": store_root}


def _config(out_dir):
    return {
        "experiment_id": EXPERIMENT_ID,
        "emitter": "xarray",
        "emitter_arg": {"out_dir": out_dir},
        "variants": {},
        "n_init_sims": N_LINEAGES,
        "skip_baseline": False,  # -> num_variants == 1 (the baseline, variant=0)
    }


def test_workflow_paths_locate_discovers_real_viva_substores(workflow_store):
    wc = WorkflowConfig.build(_config(workflow_store["out_dir"]))
    assert wc.is_uri is False

    wp = WorkflowPaths.locate(wc)
    assert len(wp) == N_LINEAGES
    subs = sorted((s.variant, s.lineage) for s in wp)
    assert subs == [("variant=0", "lineage_seed=0"), ("variant=0", "lineage_seed=1")]


def test_from_substore_round_trips(workflow_store):
    wc = WorkflowConfig.build(_config(workflow_store["out_dir"]))
    sub = Substore("variant=0", "lineage_seed=1")
    part = XarrayStoragePartition.from_substore(wc, sub, generation=2)
    assert part.variant == 0
    assert part.lineage_seed == 1
    assert part.generation == 2
    assert part.agent_id == "00"
    assert str(part.independent_path) == "experiment_id=e/variant=0/lineage_seed=1"


def test_zarr_utils_async_read_over_substore(workflow_store):
    from zarr.api.asynchronous import open_group

    wc = WorkflowConfig.build(_config(workflow_store["out_dir"]))
    wp = WorkflowPaths.locate(wc)
    sub = next(iter(wp))
    sub_path = f"{Path(wp.root).name}/{sub}"

    async def read():
        root = await open_group(store=workflow_store["store_root"], mode="r")
        g = await get_async_group(root, sub_path)
        async for name, node in g.members(max_depth=None):
            if hasattr(node, "shape") and name.endswith("generation=2"):
                data = await node.getitem(Ellipsis)
                return name, tuple(data.shape), data[:3].tolist()
        return None

    result = asyncio.run(read())
    assert result is not None, "no generation=2 array read from substore"
    _name, shape, head = result
    assert shape == (6,)
    assert head == [10.0, 11.0, 12.0]
