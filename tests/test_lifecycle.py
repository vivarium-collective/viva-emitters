"""Tests for viva_emitters.lifecycle — the per-agent emitter registry.

These exercise the framework-generic lifecycle utility with a duck-typed
fake emitter (no heavy parquet deps required) and, where the [parquet]
extra is available, against a real ParquetEmitter to confirm the
finalize-before-teardown flow flushes the trailing partial batch and writes
the success sentinel.
"""

from __future__ import annotations

import pytest

from viva_emitters import lifecycle
from viva_emitters.lifecycle import (
    clear_registry,
    finalize_emitter_for_agent,
    get_emitter,
    register_emitter,
    registered_agent_ids,
    unregister_emitter,
)


@pytest.fixture(autouse=True)
def _clean_registry():
    """Each test starts and ends with an empty process-global registry."""
    clear_registry()
    yield
    clear_registry()


class _FakeEmitter:
    """Duck-typed stand-in: records close() calls."""

    def __init__(self):
        self.closed_with = []  # list of success flags passed to close()

    def close(self, success: bool = False):
        self.closed_with.append(success)


def test_register_get_unregister_roundtrip():
    assert get_emitter("00") is None
    e = _FakeEmitter()
    register_emitter("00", e)
    assert get_emitter("00") is e
    # agent_id is coerced to str, so an int key finds the same slot.
    register_emitter(0, e)
    assert get_emitter("0") is e
    assert set(registered_agent_ids()) == {"00", "0"}
    assert unregister_emitter("00") is e
    assert get_emitter("00") is None
    # Unregistering a missing key returns None, doesn't raise.
    assert unregister_emitter("nope") is None


def test_finalize_closes_and_unregisters():
    e = _FakeEmitter()
    register_emitter("0", e)
    finalized = finalize_emitter_for_agent("0", success=True)
    assert finalized is True
    assert e.closed_with == [True]
    # Finalize is one-shot: the emitter is gone from the registry.
    assert get_emitter("0") is None
    # A second finalize is a no-op (no emitter registered).
    assert finalize_emitter_for_agent("0", success=True) is False
    assert e.closed_with == [True]


def test_finalize_missing_agent_returns_false():
    assert finalize_emitter_for_agent("missing") is False


def test_finalize_default_success_is_true():
    e = _FakeEmitter()
    register_emitter("0", e)
    finalize_emitter_for_agent("0")
    assert e.closed_with == [True]


def test_finalize_unregisters_even_when_close_raises():
    class _Boom:
        def close(self, success: bool = False):
            raise RuntimeError("boom")

    register_emitter("0", _Boom())
    with pytest.raises(RuntimeError, match="boom"):
        finalize_emitter_for_agent("0")
    # Still unregistered, so a broken emitter isn't retried forever.
    assert get_emitter("0") is None


def test_module_level_exports_match_package_exports():
    """``from viva_emitters import finalize_emitter_for_agent`` works."""
    import viva_emitters

    assert viva_emitters.finalize_emitter_for_agent is finalize_emitter_for_agent
    assert viva_emitters.register_emitter is register_emitter


# ---------------------------------------------------------------------------
# Real-emitter integration: the lifecycle finalize flushes a ParquetEmitter's
# trailing partial batch and writes the success sentinel before teardown.
# ---------------------------------------------------------------------------

try:
    import duckdb  # noqa: F401
    import polars as pl  # noqa: F401

    _HAVE_PARQUET = True
except ImportError:
    _HAVE_PARQUET = False


@pytest.mark.skipif(not _HAVE_PARQUET, reason="[parquet] extra not installed")
def test_finalize_flushes_real_parquet_emitter(tmp_path, core):
    """finalize_emitter_for_agent flushes buffered rows + writes the sentinel.

    Mirrors the division-boundary flow: a per-generation parquet emitter is
    constructed and registered under its agent_id, buffers fewer rows than a
    full batch, and is finalized (instead of being torn down silently) so its
    partial batch lands on disk and the success sentinel is written.
    """
    import os

    from viva_emitters.parquet_emitter import ParquetEmitter

    out_dir = str(tmp_path / "out")
    emitter = ParquetEmitter(
        config={
            "out_dir": out_dir,
            "batch_size": 100,  # larger than the rows we emit -> partial batch
            "threaded": False,
            "partitioning_keys": ["experiment_id", "agent_id"],
            "metadata": {"experiment_id": "exp", "agent_id": "0"},
        },
        core=core,
    )
    emitter.last_batch_future.result()
    register_emitter("0", emitter)

    for t in range(5):
        emitter.update({"global_time": float(t)})

    # Before finalize the partial batch is only in memory; finalize lands it.
    assert finalize_emitter_for_agent("0", success=True) is True
    assert get_emitter("0") is None

    # Trailing partial batch persisted: all 5 rows readable.
    assert len(emitter.query()) == 5

    # Success sentinel written under the partition path.
    sentinel = os.path.join(
        emitter.out_uri, "exp", "success",
        "experiment_id=exp", "agent_id=0", "s.pq",
    )
    assert os.path.exists(sentinel), f"missing success sentinel: {sentinel}"
