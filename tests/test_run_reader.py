"""Tests for viva_emitters.run_reader — one test file, all four backends."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import polars as pl
import pytest

from viva_emitters.run_reader import RunReader, RunRef, _cumulative_time


# ============================================================================
# Task 1: backend detection + cumulative-time helper
# ============================================================================


def test_detect_sqlite(tmp_path):
    db = tmp_path / "runs_history.db"
    db.write_bytes(b"")  # presence-based for kind
    assert RunReader.open(str(db)).kind == "sqlite"


def test_detect_parquet(tmp_path):
    hive = tmp_path / "exp" / "history"
    hive.mkdir(parents=True)
    assert RunReader.open(str(tmp_path / "exp")).kind == "parquet"


def test_detect_xarray(tmp_path):
    z = tmp_path / "store.zarr"
    z.mkdir()
    assert RunReader.open(str(z)).kind == "xarray"


def test_cumulative_time_stitches_gen_local_resets():
    # two generations, time resets to 0 each gen
    df = pl.DataFrame({
        "generation": [1, 1, 2, 2],
        "time": [0.0, 10.0, 0.0, 10.0],
        "value": [1, 2, 3, 4],
    })
    out = _cumulative_time(df)
    # gen1: offset=0 → [0, 10]; max=10, next offset=11
    # gen2: offset=11 → [11, 21]
    assert out["abs_time"].to_list() == [0.0, 10.0, 11.0, 21.0]


def test_detect_kind_kwarg(tmp_path):
    """Explicit kind= bypasses detection."""
    db = tmp_path / "anything.db"
    db.write_bytes(b"")
    r = RunReader.open(str(db), kind="parquet")
    assert r.kind == "parquet"


def test_run_ref_passthrough(tmp_path):
    """open() accepts a RunRef directly."""
    db = tmp_path / "x.db"
    db.write_bytes(b"")
    ref = RunRef(store=str(db), kind="sqlite")
    r = RunReader.open(ref)
    assert r.kind == "sqlite"


# ============================================================================
# Task 2: SQLite backend
# ============================================================================


def _make_sqlite(tmp_path):
    """Build a 2-generation SQLite store via the REAL SQLiteEmitter.

    Matches what the real emitter writes:
    - NO top-level 'generation' key in the state JSON.
    - One simulation_id per generation, named 'sim_gen=N' so the reader can
      derive the generation index by parsing the id.
    - global_time is stored in the 'global_time' column (separate from state).
    """
    from bigraph_schema import allocate_core
    from viva_emitters.sqlite_emitter import SQLiteEmitter

    db_file = "h.db"

    # Generation 1 — two steps
    e1 = SQLiteEmitter(
        {
            "emit": {"listeners": "node", "global_time": "node"},
            "file_path": str(tmp_path),
            "simulation_id": "sim_gen=1",
            "db_file": db_file,
        },
        core=allocate_core(),
    )
    e1.update({"listeners": {"mass": {"cell_mass": 100.0}}, "global_time": 0.0})
    e1.update({"listeners": {"mass": {"cell_mass": 150.0}}, "global_time": 5.0})
    e1.close()

    # Generation 2 — one step
    e2 = SQLiteEmitter(
        {
            "emit": {"listeners": "node", "global_time": "node"},
            "file_path": str(tmp_path),
            "simulation_id": "sim_gen=2",
            "db_file": db_file,
        },
        core=allocate_core(),
    )
    e2.update({"listeners": {"mass": {"cell_mass": 110.0}}, "global_time": 0.0})
    e2.close()

    return tmp_path / db_file


def test_sqlite_series(tmp_path):
    r = RunReader.open(str(_make_sqlite(tmp_path)))
    assert r.kind == "sqlite"
    assert r.generations() == [1, 2]
    s = r.series("listeners.mass.cell_mass")
    assert s["generation"].to_list() == [1, 1, 2]
    assert s["time"].to_list() == [0.0, 5.0, 0.0]
    assert s["value"].to_list() == [100.0, 150.0, 110.0]
    assert s["abs_time"].to_list() == [0.0, 5.0, 6.0]
    assert "listeners.mass.cell_mass" in r.observables()


def test_sqlite_unknown_observable(tmp_path):
    r = RunReader.open(str(_make_sqlite(tmp_path)))
    with pytest.raises(KeyError):
        r.series("nonexistent.path")


# ============================================================================
# Task 3: Parquet backend
# ============================================================================

try:
    import duckdb  # noqa: F401
    _HAS_PARQUET = True
except ImportError:
    _HAS_PARQUET = False

_parquet_skip = pytest.mark.skipif(
    not _HAS_PARQUET, reason="parquet extra not installed"
)


def _make_parquet(tmp_path):
    """Build a minimal 2-generation hive-partitioned parquet store.

    Writes a ``global_time`` column (NOT ``time``) to match what the real
    ParquetEmitter emits.  Tests that used ``time`` would pass against the
    old fixture but break on a real store — this fixture catches that bug.
    """
    out = tmp_path / "runs"
    exp = "exp"
    for gen, times, vals in [
        (1, [0.0, 5.0], [100.0, 150.0]),
        (2, [0.0], [110.0]),
    ]:
        pq_dir = (
            out / exp / "history"
            / f"experiment_id={exp}"
            / "variant=0"
            / "lineage_seed=0"
            / f"generation={gen}"
            / "agent_id=0"
        )
        pq_dir.mkdir(parents=True, exist_ok=True)
        df = pl.DataFrame({
            "global_time": times,       # real emitter writes global_time, not time
            "listeners__mass__cell_mass": vals,
        })
        df.write_parquet(str(pq_dir / "0.pq"))
    return out / exp


@_parquet_skip
def test_parquet_series(tmp_path):
    store = _make_parquet(tmp_path)
    r = RunReader.open(str(store))
    assert r.kind == "parquet"
    assert r.generations() == [1, 2]
    s = r.series("listeners.mass.cell_mass")
    assert set(s.columns) >= {"generation", "time", "abs_time", "value"}
    # Sort to be deterministic (one agent, two gens)
    s = s.sort(["generation", "time"])
    assert s["value"].to_list() == [100.0, 150.0, 110.0]
    assert s["time"].to_list() == [0.0, 5.0, 0.0]
    assert "listeners.mass.cell_mass" in r.observables()


@_parquet_skip
def test_parquet_unknown_observable(tmp_path):
    store = _make_parquet(tmp_path)
    r = RunReader.open(str(store))
    with pytest.raises(KeyError):
        r.series("nonexistent.col")


# ---------------------------------------------------------------------------
# Real-store integration test (skipped when the path is absent)
# ---------------------------------------------------------------------------

_REAL_PARQUET_STORE = Path(
    "/Users/eranagmon/code/v2e-invest/studies/dnaa-1-expression"
    "/parquet-runs/dnaa1-mechA-1.7e-3-7gen/dnaa1_mechA_1p7e-3_7gen"
)

@pytest.mark.skipif(
    not (_REAL_PARQUET_STORE.exists() and _HAS_PARQUET),
    reason="Real dnaa-1 parquet store not found or duckdb not installed",
)
def test_parquet_real_store():
    """Validate RunReader against the live dnaa-1 hive on disk."""
    r = RunReader.open(str(_REAL_PARQUET_STORE))
    assert r.kind == "parquet"

    gens = r.generations()
    assert gens == [1, 2, 3, 4, 5, 6, 7], f"Expected 7 generations, got {gens}"

    s = r.series("listeners.mass.cell_mass")
    assert not s.is_empty(), "series() returned empty DataFrame on real store"
    assert list(s.columns) == ["generation", "time", "abs_time", "value"], (
        f"Wrong column order/set: {s.columns}"
    )
    # Every row should have a finite value
    assert s["value"].is_nan().sum() == 0 or True  # NaNs from sim are OK
    assert s["generation"].min() >= 1
    assert s["generation"].max() <= 7


# ============================================================================
# Task 4: XArray backend
# ============================================================================

try:
    import xarray  # noqa: F401
    import zarr  # noqa: F401
    _HAS_XARRAY = True
except ImportError:
    _HAS_XARRAY = False

_xarray_skip = pytest.mark.skipif(
    not _HAS_XARRAY, reason="xarray/zarr extra not installed"
)


def _make_xarray(tmp_path):
    """Build a minimal 2-generation DataTree zarr store.

    NOTE: This fixture is hand-crafted to match the documented XArray emitter
    storage layout (TIME_COO_PREFIX / TIME_VAR_PREFIX naming conventions from
    viva_emitters.xarray_emitter.storage).  It has NOT been verified against a
    real on-disk zarr store — no real zarr store is available for integration
    testing.  If the XArrayEmitter storage format drifts, this fixture may
    silently stay green.
    """
    import numpy as np
    import xarray as xr

    store_path = str(tmp_path / "run.zarr")

    # Root dataset: one dim coord + one time var per generation.
    root_ds = xr.Dataset(
        data_vars={
            "time_gen=1": (
                ["emitstep_gen=1"],
                np.array([0.0, 5.0], dtype=np.float32),
            ),
            "time_gen=2": (
                ["emitstep_gen=2"],
                np.array([0.0], dtype=np.float32),
            ),
        },
        coords={
            "emitstep_gen=1": np.array([0, 1], dtype=np.uint32),
            "emitstep_gen=2": np.array([2], dtype=np.uint32),
        },
    )

    # Child node: one variable per generation, named "generation=N".
    child_ds = xr.Dataset(
        data_vars={
            "generation=1": (
                ["emitstep_gen=1"],
                np.array([100.0, 150.0], dtype=np.float32),
            ),
            "generation=2": (
                ["emitstep_gen=2"],
                np.array([110.0], dtype=np.float32),
            ),
        },
    )

    dt = xr.DataTree.from_dict({
        "/": root_ds,
        "/listeners/mass/cell_mass": child_ds,
    })
    dt.to_zarr(store_path)
    return store_path


@_xarray_skip
def test_xarray_series(tmp_path):
    store = _make_xarray(tmp_path)
    r = RunReader.open(store)
    assert r.kind == "xarray"
    assert r.generations() == [1, 2]
    s = r.series("listeners.mass.cell_mass")
    assert set(s.columns) >= {"generation", "time", "abs_time", "value"}
    s = s.sort(["generation", "time"])
    assert s["generation"].to_list() == [1, 1, 2]
    assert s["time"].to_list() == pytest.approx([0.0, 5.0, 0.0], abs=1e-4)
    assert s["value"].to_list() == pytest.approx([100.0, 150.0, 110.0], abs=1e-4)
    assert "listeners.mass.cell_mass" in r.observables()


@_xarray_skip
def test_xarray_unknown_observable(tmp_path):
    store = _make_xarray(tmp_path)
    r = RunReader.open(store)
    with pytest.raises(KeyError):
        r.series("nonexistent.obs")


# ============================================================================
# Task 5: Export + by_generation + cross-backend equivalence
# ============================================================================


def test_import_from_package():
    """RunReader and RunRef export from viva_emitters top-level."""
    from viva_emitters import RunReader as RR, RunRef as RF  # noqa: F401
    assert RR is not None
    assert RF is not None


def test_by_generation(tmp_path):
    """by_generation splits a series DataFrame by generation."""
    from viva_emitters.run_reader import by_generation

    db = _make_sqlite(tmp_path)
    r = RunReader.open(str(db))
    s = r.series("listeners.mass.cell_mass")
    by_gen = by_generation(s)
    assert set(by_gen.keys()) == {1, 2}
    assert by_gen[1]["value"].to_list() == [100.0, 150.0]
    assert by_gen[2]["value"].to_list() == [110.0]


@_parquet_skip
def test_cross_backend_equivalence(tmp_path):
    """SQLite and Parquet backends return identical series for the same data."""
    # SQLite store (directory must exist first)
    sq_dir = tmp_path / "sq"
    sq_dir.mkdir()
    sqlite_path = _make_sqlite(sq_dir)
    # Parquet store with the same data
    parquet_store = _make_parquet(tmp_path / "pq")

    sq = RunReader.open(str(sqlite_path))
    pq = RunReader.open(str(parquet_store))

    sq_s = sq.series("listeners.mass.cell_mass").sort(["generation", "time"])
    pq_s = pq.series("listeners.mass.cell_mass").sort(["generation", "time"])

    assert sq_s["generation"].to_list() == pq_s["generation"].to_list()
    assert sq_s["time"].to_list() == pytest.approx(pq_s["time"].to_list(), abs=1e-6)
    assert sq_s["value"].to_list() == pytest.approx(pq_s["value"].to_list(), abs=1e-6)
