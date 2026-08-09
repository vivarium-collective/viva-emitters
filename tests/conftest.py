"""Shared fixtures for viva-emitters tests."""

import pytest

from bigraph_schema import allocate_core


@pytest.fixture
def core():
    """A fresh process-bigraph Core for each test.

    Uses ``bigraph_schema.allocate_core()`` — the same factory
    process-bigraph itself uses to build its default core in tests.
    """
    return allocate_core()


@pytest.fixture
def minimal_xarray_config(tmp_path):
    """A minimum-valid config dict for viva_emitters.xarray_emitter.XArrayEmitter.

    Sufficient to build XarrayTransducer / ForestView / AsyncBufferWriter
    without raising. Used by tests that don't care about specific transducer
    behavior — only that the emitter constructs without error.

    The store path uses ``tmp_path`` so each test gets an isolated directory.
    No metadata is provided, so ``open_store`` is never called and no
    filesystem writes occur during construction.
    """
    store = str(tmp_path / "x.zarr")
    return {
        "emit": {},
        "out_uri": store,
        "transducer": {
            "predicate": [[{"subsample": {"interval": 1}}]],
            "buffer": {"size": 3},
        },
        "view": [
            {
                "root": ("listeners",),
                "variables": {
                    "dummy_var": [{"path": "dummy/val", "dtype": "<f4"}],
                },
            }
        ],
        "writer": {
            "backend": "zarr",
            "store": store,
            "buffers_per_chunk": 1,
            "backend_config": {"format": 3},
        },
        "metadata": {},
        "metadata_keys": [],
        "metadata_validators": {},
        "output_metadata": {},
        "debug": False,
    }
