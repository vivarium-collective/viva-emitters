"""Unit tests for viva_emitters.xarray_emitter."""

import pytest

# Skip the entire file if the [xarray] extra isn't installed.
pytest.importorskip("xarray")
pytest.importorskip("zarr")


def test_base_module_imports():
    from viva_emitters.xarray_emitter._base import (
        BufferedEmitter, StoragePartition, BlockingExecutor,
    )
    assert BufferedEmitter is not None
    assert StoragePartition is not None
    assert BlockingExecutor is not None


def test_storage_partition_dataclass():
    from viva_emitters.xarray_emitter._base import StoragePartition

    p = StoragePartition(
        experiment_id="exp1", variant=2, lineage_seed=7, agent_id="01"
    )
    assert p.generation == 2  # len(agent_id)
    assert p.parent.agent_id == "0"


def test_buffered_emitter_inherits_viva_emitter():
    from viva_emitters.xarray_emitter._base import BufferedEmitter
    from process_bigraph.emitter import Emitter
    assert issubclass(BufferedEmitter, Emitter)


def test_all_submodules_import():
    """Smoke check: every vendored sub-module imports without error."""
    for name in (
        "transducer", "view", "storage", "writer", "zarr_writer",
        "emit_path", "emit_predicate", "utils",
    ):
        __import__(f"viva_emitters.xarray_emitter.{name}")


def test_xarray_emitter_imports():
    from viva_emitters.xarray_emitter import XArrayEmitter
    from process_bigraph.emitter import Emitter
    assert issubclass(XArrayEmitter, Emitter)


def test_xarray_emitter_config_schema_has_expected_keys():
    from viva_emitters.xarray_emitter import XArrayEmitter
    for key in (
        "emit", "out_uri", "transducer", "view", "writer",
        "metadata", "metadata_keys", "metadata_validators",
        "output_metadata", "debug",
    ):
        assert key in XArrayEmitter.config_schema, f"missing: {key}"


def test_metadata_validators_failure_raises(minimal_xarray_config, core):
    from viva_emitters.xarray_emitter import XArrayEmitter

    cfg = {
        **minimal_xarray_config,
        "metadata": {"single_daughters": False},
        "metadata_validators": {"single_daughters": True},
    }
    with pytest.raises(ValueError, match="single_daughters"):
        XArrayEmitter(config=cfg, core=core)


def test_empty_metadata_validators_no_op(minimal_xarray_config, core):
    """Empty validators dict => no validation error from the validator path."""
    from viva_emitters.xarray_emitter import XArrayEmitter

    cfg = {**minimal_xarray_config, "metadata_validators": {}}
    # If construction raised a ValueError from a validator, the test would fail.
    # With empty validators, construction must succeed.
    emitter = XArrayEmitter(config=cfg, core=core)
    assert emitter is not None
    emitter.close()  # tidy up the writer


# ---------------------------------------------------------------------------
# out_uri property — required by the dashboard's _flush_step_emitters gate
# ---------------------------------------------------------------------------

def test_async_zarr_buffer_writer_has_out_uri(tmp_path):
    """AsyncBufferWriter.out_uri must return str(config['store']).

    Without this property, the dashboard's _flush_step_emitters helper
    returns False from _buffers(inst) and skips close()/consolidate(),
    leaving the zarr store unconsolidated → sqlite fallback.
    """
    from viva_emitters.xarray_emitter.writer import AsyncBufferWriter

    store = str(tmp_path / "test_writer.zarr")
    config = {
        "backend": "zarr",
        "store": store,
        "buffers_per_chunk": 1,
        "backend_config": {"format": 3},
    }
    writer = AsyncBufferWriter.dispatch(config)
    assert hasattr(writer, "out_uri"), "writer.out_uri attribute must exist"
    assert writer.out_uri == store


def test_xarray_emitter_writer_out_uri_accessible(minimal_xarray_config, core):
    """XArrayEmitter.writer.out_uri is reachable without AttributeError.

    The dashboard checks hasattr(inst.writer, 'out_uri') — this test
    confirms the emitter's writer exposes out_uri after construction.
    Also covers emitter.py:200 (self.writer.out_uri in query()).
    """
    from viva_emitters.xarray_emitter import XArrayEmitter

    emitter = XArrayEmitter(config=minimal_xarray_config, core=core)
    # must not raise AttributeError
    uri = emitter.writer.out_uri
    assert uri == minimal_xarray_config["writer"]["store"]
    emitter.close()
