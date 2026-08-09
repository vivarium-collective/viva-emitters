"""Tests for RunReader catalog extensions (sub-project #2).

Task 1: catalog() per backend
Task 2: resolve_id() + select()
Task 3: aggregate_series()
Task 4: golden (real hive bulk + synthetic aggregate)
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from viva_emitters.run_reader import (
    RunReader,
    IdNotInCatalog,
    CatalogUnavailable,
)

# ============================================================================
# Parquet extra guard
# ============================================================================

try:
    import duckdb  # noqa: F401
    _HAS_PARQUET = True
except ImportError:
    _HAS_PARQUET = False

_parquet_skip = pytest.mark.skipif(
    not _HAS_PARQUET, reason="parquet extra not installed"
)

# ============================================================================
# Fixtures
# ============================================================================

# Catalog names and per-tick values for the synthetic self-describing store.
CATALOG_NAMES = ["DnaA_ATP", "DnaA_ADP", "DnaA_free"]
# gen=1: t=0 → [10,2,3]; t=5 → [11,2.5,3.5]
# gen=2: t=0 → [12,3,4]
GEN1_VALS = [[10.0, 2.0, 3.0], [11.0, 2.5, 3.5]]
GEN2_VALS = [[12.0, 3.0, 4.0]]


def _make_self_describing_parquet(tmp_path):
    """Synthetic hive with output_metadata__listeners__monomer_counts catalog."""
    out = tmp_path / "runs"
    exp = "exp"

    for gen, times, vals in [
        (1, [0.0, 5.0], GEN1_VALS),
        (2, [0.0], GEN2_VALS),
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
            "global_time": pl.Series(times, dtype=pl.Float64),
            "listeners__monomer_counts": pl.Series(vals, dtype=pl.List(pl.Float64)),
        })
        df.write_parquet(str(pq_dir / "0.pq"))

    # Config with the per-observable name catalog.
    config_dir = (
        out / exp / "configuration"
        / f"experiment_id={exp}"
        / "variant=0"
        / "lineage_seed=0"
        / "generation=0"
        / "agent_id=0"
    )
    config_dir.mkdir(parents=True, exist_ok=True)
    config_df = pl.DataFrame({
        "output_metadata__listeners__monomer_counts": [CATALOG_NAMES],
    })
    config_df.write_parquet(str(config_dir / "config.pq"))

    return out / exp


def _make_basic_parquet(tmp_path):
    """Basic scalar-observable hive with NO output_metadata catalog."""
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
            "global_time": pl.Series(times, dtype=pl.Float64),
            "listeners__mass__cell_mass": pl.Series(vals, dtype=pl.Float64),
        })
        df.write_parquet(str(pq_dir / "0.pq"))
    return out / exp


# Real dnaa hive (read-only, NEVER modified)
_REAL_PARQUET_STORE = Path(
    "/Users/eranagmon/code/v2e-invest/studies/dnaa-1-expression"
    "/parquet-runs/dnaa1-mechA-1.7e-3-7gen/dnaa1_mechA_1p7e-3_7gen"
)
_real_skip = pytest.mark.skipif(
    not (_REAL_PARQUET_STORE.exists() and _HAS_PARQUET),
    reason="Real dnaa-1 parquet store not found or duckdb not installed",
)

# ============================================================================
# Task 1: catalog()
# ============================================================================


@_real_skip
def test_catalog_bulk_real_hive():
    """catalog('bulk') on the real hive returns a non-empty bulk__id list."""
    r = RunReader.open(str(_REAL_PARQUET_STORE))
    cat = r.catalog("bulk")
    assert cat is not None, "catalog('bulk') returned None on real hive"
    assert len(cat) > 0
    # Real bulk ids contain compartment brackets like [c], [p], etc.
    assert any("[" in x for x in cat), f"No bracketed ids in first 5: {cat[:5]}"


@_parquet_skip
def test_catalog_listener_synthetic(tmp_path):
    """catalog('listeners.monomer_counts') returns the output_metadata names."""
    store = _make_self_describing_parquet(tmp_path)
    r = RunReader.open(str(store))
    cat = r.catalog("listeners.monomer_counts")
    assert cat == CATALOG_NAMES


@_parquet_skip
def test_catalog_no_catalog_returns_none(tmp_path):
    """catalog() returns None for a scalar observable with no output_metadata."""
    store = _make_basic_parquet(tmp_path)
    r = RunReader.open(str(store))
    assert r.catalog("listeners.mass.cell_mass") is None


# ============================================================================
# Task 2: resolve_id() + select()
# ============================================================================


@_parquet_skip
def test_resolve_id_correct_indices(tmp_path):
    """resolve_id returns 0-based index for each known catalog element."""
    store = _make_self_describing_parquet(tmp_path)
    r = RunReader.open(str(store))
    assert r.resolve_id("listeners.monomer_counts", "DnaA_ATP") == 0
    assert r.resolve_id("listeners.monomer_counts", "DnaA_ADP") == 1
    assert r.resolve_id("listeners.monomer_counts", "DnaA_free") == 2


@_parquet_skip
def test_resolve_id_unknown_raises_id_not_in_catalog(tmp_path):
    """resolve_id raises IdNotInCatalog for an unknown element."""
    store = _make_self_describing_parquet(tmp_path)
    r = RunReader.open(str(store))
    with pytest.raises(IdNotInCatalog):
        r.resolve_id("listeners.monomer_counts", "NONEXISTENT_MOLECULE")


@_parquet_skip
def test_resolve_id_no_catalog_raises_catalog_unavailable(tmp_path):
    """resolve_id raises CatalogUnavailable when no catalog exists for the observable."""
    store = _make_basic_parquet(tmp_path)
    r = RunReader.open(str(store))
    with pytest.raises(CatalogUnavailable):
        r.resolve_id("listeners.mass.cell_mass", "some_id")


@_parquet_skip
def test_select_monomer_id(tmp_path):
    """select(monomer_id) returns standard shape with correct element values."""
    store = _make_self_describing_parquet(tmp_path)
    r = RunReader.open(str(store))
    df = r.select({"type": "monomer_id", "value": "DnaA_ATP"})
    # DnaA_ATP is index 0 → values: gen1=[10, 11], gen2=[12]
    assert list(df.columns) == ["generation", "time", "abs_time", "value"]
    df = df.sort(["generation", "time"])
    assert df["value"].to_list() == pytest.approx([10.0, 11.0, 12.0])
    assert df["generation"].to_list() == [1, 1, 2]
    assert df["time"].to_list() == pytest.approx([0.0, 5.0, 0.0])
    # abs_time: gen1 offset=0→[0,5]; gen1 max=5 → next offset=6; gen2=[6]
    assert df["abs_time"].to_list() == pytest.approx([0.0, 5.0, 6.0])


@_parquet_skip
def test_select_monomer_id_second_element(tmp_path):
    """select(monomer_id) with DnaA_ADP (index 1) returns correct values."""
    store = _make_self_describing_parquet(tmp_path)
    r = RunReader.open(str(store))
    df = r.select({"type": "monomer_id", "value": "DnaA_ADP"})
    df = df.sort(["generation", "time"])
    # DnaA_ADP is index 1 → values: [2, 2.5, 3]
    assert df["value"].to_list() == pytest.approx([2.0, 2.5, 3.0])


@_parquet_skip
def test_select_literal_index(tmp_path):
    """select(literal_index) works without a catalog; uses numeric index directly."""
    store = _make_self_describing_parquet(tmp_path)
    r = RunReader.open(str(store))
    # Index 1 → DnaA_ADP values: [2, 2.5, 3]
    df = r.select({
        "type": "literal_index",
        "value": 1,
        "observable": "listeners.monomer_counts",
    })
    assert list(df.columns) == ["generation", "time", "abs_time", "value"]
    df = df.sort(["generation", "time"])
    assert df["value"].to_list() == pytest.approx([2.0, 2.5, 3.0])


@_real_skip
def test_select_bulk_id_real_hive():
    """select(bulk_id) on real hive returns a non-empty count series."""
    r = RunReader.open(str(_REAL_PARQUET_STORE))
    cat = r.catalog("bulk")
    assert cat is not None
    some_id = cat[0]
    df = r.select({"type": "bulk_id", "value": some_id})
    assert list(df.columns) == ["generation", "time", "abs_time", "value"]
    assert not df.is_empty()
    # After Float64 cast, dtype is Float64
    assert df["value"].dtype == pl.Float64


# ============================================================================
# Task 3: aggregate_series()
# ============================================================================


@_parquet_skip
def test_aggregate_series_sum(tmp_path):
    """aggregate_series sum across all 3 elements returns per-tick sum."""
    store = _make_self_describing_parquet(tmp_path)
    r = RunReader.open(str(store))
    df = r.aggregate_series("listeners.monomer_counts", "sum", over=CATALOG_NAMES)
    assert list(df.columns) == ["generation", "time", "abs_time", "value"]
    df = df.sort(["generation", "time"])
    # gen1 t=0: 10+2+3=15; t=5: 11+2.5+3.5=17; gen2 t=0: 12+3+4=19
    assert df["value"].to_list() == pytest.approx([15.0, 17.0, 19.0])


@_parquet_skip
def test_aggregate_series_mean(tmp_path):
    """aggregate_series mean."""
    store = _make_self_describing_parquet(tmp_path)
    r = RunReader.open(str(store))
    df = r.aggregate_series("listeners.monomer_counts", "mean", over=CATALOG_NAMES)
    df = df.sort(["generation", "time"])
    assert df["value"].to_list() == pytest.approx([15.0 / 3, 17.0 / 3, 19.0 / 3], rel=1e-5)


@_parquet_skip
def test_aggregate_series_max(tmp_path):
    """aggregate_series max selects the largest element per tick."""
    store = _make_self_describing_parquet(tmp_path)
    r = RunReader.open(str(store))
    df = r.aggregate_series("listeners.monomer_counts", "max", over=CATALOG_NAMES)
    df = df.sort(["generation", "time"])
    # max of [10,2,3]=10, [11,2.5,3.5]=11, [12,3,4]=12
    assert df["value"].to_list() == pytest.approx([10.0, 11.0, 12.0])


@_parquet_skip
def test_aggregate_series_min(tmp_path):
    """aggregate_series min over a subset of elements."""
    store = _make_self_describing_parquet(tmp_path)
    r = RunReader.open(str(store))
    df = r.aggregate_series(
        "listeners.monomer_counts", "min", over=["DnaA_ATP", "DnaA_ADP"]
    )
    df = df.sort(["generation", "time"])
    # min of [10,2], [11,2.5], [12,3] → [2, 2.5, 3]
    assert df["value"].to_list() == pytest.approx([2.0, 2.5, 3.0])


@_parquet_skip
def test_aggregate_series_unknown_id_raises(tmp_path):
    """aggregate_series raises IdNotInCatalog for any unknown id in over."""
    store = _make_self_describing_parquet(tmp_path)
    r = RunReader.open(str(store))
    with pytest.raises(IdNotInCatalog):
        r.aggregate_series(
            "listeners.monomer_counts", "sum", over=["DnaA_ATP", "BOGUS_MOLECULE"]
        )


# ============================================================================
# Task 3b: aggregate_series for bulk observable (review bug fix)
# ============================================================================

# Bulk synthetic constants
_BULK_IDS = ["ATP[c]", "GTP[c]", "CTP[c]"]
# gen=1: t=0 → [100,200,300]; t=5 → [110,210,310]
# gen=2: t=0 → [120,220,320]
_BULK_GEN1 = [[100.0, 200.0, 300.0], [110.0, 210.0, 310.0]]
_BULK_GEN2 = [[120.0, 220.0, 320.0]]


def _make_bulk_parquet(tmp_path):
    """Synthetic hive with bulk__id catalog and bulk__count data columns."""
    out = tmp_path / "bulk_runs"
    exp = "exp"

    for gen, times, counts in [
        (1, [0.0, 5.0], _BULK_GEN1),
        (2, [0.0], _BULK_GEN2),
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
            "global_time": pl.Series(times, dtype=pl.Float64),
            "bulk__id": pl.Series([_BULK_IDS] * len(times), dtype=pl.List(pl.String)),
            "bulk__count": pl.Series(counts, dtype=pl.List(pl.Float64)),
        })
        df.write_parquet(str(pq_dir / "0.pq"))

    return out / exp


@_parquet_skip
def test_aggregate_series_bulk_synthetic(tmp_path):
    """aggregate_series('bulk','sum') uses bulk__count, not 'bulk' (review bug)."""
    store = _make_bulk_parquet(tmp_path)
    r = RunReader.open(str(store))

    # Sanity: individual selects work
    atp_df = r.select({"type": "bulk_id", "value": "ATP[c]"}).sort(["generation", "time"])
    gtp_df = r.select({"type": "bulk_id", "value": "GTP[c]"}).sort(["generation", "time"])
    expected = (atp_df["value"] + gtp_df["value"]).to_list()

    # This must not raise BinderException
    agg = r.aggregate_series("bulk", "sum", over=["ATP[c]", "GTP[c]"])
    agg = agg.sort(["generation", "time"])
    assert list(agg.columns) == ["generation", "time", "abs_time", "value"]
    # gen1 t=0: 100+200=300; t=5: 110+210=320; gen2 t=0: 120+220=340
    assert agg["value"].to_list() == pytest.approx([300.0, 320.0, 340.0])
    assert agg["value"].to_list() == pytest.approx(expected)


@_parquet_skip
def test_aggregate_series_bulk_all_three_synthetic(tmp_path):
    """aggregate_series('bulk','sum') over all 3 bulk ids = element-wise sum of selects."""
    store = _make_bulk_parquet(tmp_path)
    r = RunReader.open(str(store))

    # sum of all three: [100+200+300, 110+210+310, 120+220+320] = [600, 630, 660]
    agg = r.aggregate_series("bulk", "sum", over=_BULK_IDS)
    agg = agg.sort(["generation", "time"])
    assert agg["value"].to_list() == pytest.approx([600.0, 630.0, 660.0])


@_real_skip
def test_aggregate_series_bulk_real_hive():
    """aggregate_series('bulk','sum') on real hive matches sum of individual selects."""
    r = RunReader.open(str(_REAL_PARQUET_STORE))
    bulk_ids = ["ATP[c]", "ATP[p]", "ATP[e]"]

    # Individual selects (the known-working path)
    dfs = [
        r.select({"type": "bulk_id", "value": bid}).sort(["generation", "time"])
        for bid in bulk_ids
    ]
    expected_sum = (dfs[0]["value"] + dfs[1]["value"] + dfs[2]["value"]).to_list()

    # aggregate_series must return the same values
    agg = r.aggregate_series("bulk", "sum", over=bulk_ids)
    agg = agg.sort(["generation", "time"])
    assert list(agg.columns) == ["generation", "time", "abs_time", "value"]
    assert len(agg) == len(dfs[0])
    assert agg["value"].to_list() == pytest.approx(expected_sum, rel=1e-5)


# ============================================================================
# Task 4: error types importable + column shape consistency
# ============================================================================


def test_error_types_importable():
    """IdNotInCatalog and CatalogUnavailable are importable from viva_emitters."""
    from viva_emitters import IdNotInCatalog as Inc, CatalogUnavailable as Cu  # noqa: F401
    assert issubclass(Inc, KeyError)
    assert issubclass(Cu, LookupError)


@_parquet_skip
def test_select_returns_standard_shape(tmp_path):
    """select() returns exactly [generation, time, abs_time, value] in that order."""
    store = _make_self_describing_parquet(tmp_path)
    r = RunReader.open(str(store))
    df = r.select({"type": "monomer_id", "value": "DnaA_free"})
    assert df.columns == ["generation", "time", "abs_time", "value"]
    assert df.dtypes == [pl.Int64, pl.Float64, pl.Float64, pl.Float64]


@_parquet_skip
def test_aggregate_returns_standard_shape(tmp_path):
    """aggregate_series() returns exactly [generation, time, abs_time, value]."""
    store = _make_self_describing_parquet(tmp_path)
    r = RunReader.open(str(store))
    df = r.aggregate_series("listeners.monomer_counts", "sum", over=CATALOG_NAMES)
    assert df.columns == ["generation", "time", "abs_time", "value"]
    assert df.dtypes == [pl.Int64, pl.Float64, pl.Float64, pl.Float64]
