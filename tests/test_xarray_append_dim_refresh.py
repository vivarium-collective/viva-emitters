"""Deterministic tests for the first-flush append-dim visibility refresh.

Regression coverage for an intermittent ``AssertionError`` in
``AsyncZarrBufferWriter.update_transport`` (zarr==3.1.6):

    assert self.store._append_dim in self.store.get_dimensions()

At the first buffer flush of a generation, ``update_transport`` refreshes the
``ZarrStore`` member cache via ``_fetch_members()`` (a fresh directory listing)
and then requires the generation-specific append dimension to be visible to
``get_dimensions()``. The fresh listing runs on Zarr's background event-loop
thread right after the first-flush writes; there is a narrow window in which a
just-written child ``zarr.json`` is not yet readable, so zarr drops that member
from the listing and the append dimension is transiently absent — the single
``_fetch_members()`` sample misses it and the assertion fires intermittently.

``_refresh_store_members`` fixes this by re-listing (bounded, with a short
back-off) until the append dimension is actually present. These tests exercise
that loop against a **real** ``xarray.backends.ZarrStore`` (real
``get_dimensions`` / ``members``), simulating the transient window by having
``_fetch_members`` omit the append-dim array for the first few calls.
"""

import types

import pytest

pytest.importorskip("xarray")
pytest.importorskip("zarr")

import numpy as np  # noqa: E402
import xarray as xr  # noqa: E402
from xarray.backends import ZarrStore  # noqa: E402

from viva_emitters.xarray_emitter.zarr_writer import AsyncZarrBufferWriter  # noqa: E402


APPEND_DIM = "elapsed_time_gen=1"


def _make_real_store(tmp_path):
    """Write a tiny zarr store carrying a root-level ``APPEND_DIM`` coordinate
    and open it as a real ``ZarrStore`` with member caching enabled."""
    store_path = str(tmp_path / "s.zarr")
    ds = xr.Dataset(
        {"value": ((APPEND_DIM,), np.arange(3, dtype="<f8"))},
        coords={APPEND_DIM: np.arange(3, dtype="<i8")},
    )
    ds.to_zarr(store_path, zarr_format=3, consolidated=False, mode="w")

    store = ZarrStore.open_group(
        store_path, mode="a", cache_members=True, zarr_format=3
    )
    # emulate the pre-first-flush cache captured in `_open_store`: a snapshot that
    # predates the append-dim array (here, simply empty)
    store._members = {}
    store._append_dim = APPEND_DIM
    return store


def _bind_refresh(store):
    """A minimal object exposing the surface ``_refresh_store_members`` needs,
    bound to the real method under test. Returns a zero-arg callable that runs
    the refresh for the store's current ``_append_dim``."""
    obj = types.SimpleNamespace(
        store=store,
        _warnings_eval_effect=[],
        _APPEND_DIM_REFRESH_ATTEMPTS=AsyncZarrBufferWriter._APPEND_DIM_REFRESH_ATTEMPTS,
        # keep the test fast: no real back-off needed for the simulated window
        _APPEND_DIM_REFRESH_BACKOFF=0.0,
    )
    bound = types.MethodType(AsyncZarrBufferWriter._refresh_store_members, obj)
    return lambda: bound(store._append_dim)


def test_append_dim_missing_from_stale_snapshot(tmp_path):
    """Sanity: the pre-refresh (single-sample) snapshot can lack the append dim."""
    store = _make_real_store(tmp_path)
    # the stale snapshot set in `_make_real_store` is empty
    assert APPEND_DIM not in store.get_dimensions()
    # but a genuine fresh listing DOES contain it
    store._members = store._fetch_members()
    assert APPEND_DIM in store.get_dimensions()


def test_refresh_retries_until_append_dim_visible(tmp_path, monkeypatch):
    """A transient window that omits the append-dim array for the first few
    fetches must be re-listed until the dimension appears."""
    store = _make_real_store(tmp_path)
    real_fetch = ZarrStore._fetch_members  # unbound; the genuine fresh listing

    miss_first = 3
    calls = {"n": 0}

    def flaky_fetch(self):
        calls["n"] += 1
        if calls["n"] <= miss_first:
            # transient consistency window: append-dim array not yet visible
            return {}
        return real_fetch(self)

    # `_fetch_members` is a slot-backed method, so patch it on the class
    monkeypatch.setattr(ZarrStore, "_fetch_members", flaky_fetch)

    _bind_refresh(store)()

    # the invariant the caller asserts must now hold, established (not sampled)
    assert APPEND_DIM in store.get_dimensions()
    assert calls["n"] == miss_first + 1  # retried past the transient misses


def test_refresh_is_bounded_and_does_not_mask_genuine_absence(tmp_path, monkeypatch):
    """If the append dim never becomes visible, the loop is bounded and leaves
    the store in a state where the caller's assertion still fires (a genuine
    failure is not silently masked)."""
    store = _make_real_store(tmp_path)
    calls = {"n": 0}

    def always_empty(self):
        calls["n"] += 1
        return {}

    monkeypatch.setattr(ZarrStore, "_fetch_members", always_empty)

    _bind_refresh(store)()

    assert APPEND_DIM not in store.get_dimensions()
    assert calls["n"] == AsyncZarrBufferWriter._APPEND_DIM_REFRESH_ATTEMPTS
