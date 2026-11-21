"""Tests for the canonical per-step ``RunReader.rows()`` accessor and the shared
bare-string-measure resolver — the G2 convergence seam.

The second test is the SHARED FIXTURE the engine grades against: a tiny composite
module function over ``rows`` + its grading through
``grade_bare_string_measure`` must equal the direct computation, so the engine's
``evaluate_test`` (calling the same resolver + accessor) grades byte-identically
to the workbench.
"""
from types import ModuleType

import polars as pl
import pytest

from viva_emitters.run_reader import RunReader
from viva_emitters.measure_resolver import (
    grade_bare_string_measure,
    resolve_measure_fn,
)

try:
    import duckdb  # noqa: F401
    _HAS_PARQUET = True
except ImportError:
    _HAS_PARQUET = False

_parquet_skip = pytest.mark.skipif(not _HAS_PARQUET, reason="parquet extra not installed")

# Storage uses ``__`` separators in column names; RunReader exposes observable
# ids (and therefore rows() keys) in dotted form.
STORE_MASS = "listeners__mass__cell_mass"
STORE_ACET = "listeners__acetate"
CELL_MASS = "listeners.mass.cell_mass"
ACETATE = "listeners.acetate"


def _make_two_obs_parquet(tmp_path):
    """Two scalar observables across 3 ticks (gen1: t=0,5; gen2: t=0)."""
    out = tmp_path / "runs"
    exp = "exp"
    for gen, times, mass, acet in [
        (1, [0.0, 5.0], [100.0, 150.0], [0.2, 0.4]),
        (2, [0.0], [110.0], [0.5]),
    ]:
        pq_dir = (
            out / exp / "history"
            / f"experiment_id={exp}" / "variant=0" / "lineage_seed=0"
            / f"generation={gen}" / "agent_id=0"
        )
        pq_dir.mkdir(parents=True, exist_ok=True)
        pl.DataFrame({
            "global_time": pl.Series(times, dtype=pl.Float64),
            STORE_MASS: pl.Series(mass, dtype=pl.Float64),
            STORE_ACET: pl.Series(acet, dtype=pl.Float64),
        }).write_parquet(str(pq_dir / "0.pq"))
    return out / exp


@_parquet_skip
def test_rows_are_per_tick_across_observables(tmp_path):
    r = RunReader.open(str(_make_two_obs_parquet(tmp_path)))
    rows = r.rows()
    # 3 ticks total, ordered by (generation, time)
    assert len(rows) == 3
    for row in rows:
        assert {"generation", "time", "abs_time"} <= set(row)
        assert CELL_MASS in row and ACETATE in row
    # values align per tick (first tick of gen 1)
    first = rows[0]
    assert first["generation"] == 1
    assert first["time"] == 0.0
    assert first[CELL_MASS] == 100.0
    assert first[ACETATE] == 0.2
    # ordering: last row is gen 2
    assert rows[-1]["generation"] == 2
    assert rows[-1][CELL_MASS] == 110.0


@_parquet_skip
def test_rows_observable_subset(tmp_path):
    r = RunReader.open(str(_make_two_obs_parquet(tmp_path)))
    rows = r.rows(observables=[CELL_MASS])
    assert len(rows) == 3
    assert CELL_MASS in rows[0]
    assert ACETATE not in rows[0]


@_parquet_skip
def test_grade_bare_string_measure_is_byte_identical(tmp_path):
    """The shared grading path == direct computation over the same rows.

    This is the cross-package contract: the engine, calling the same resolver +
    accessor on the same store, gets the same float the workbench does.
    """
    r = RunReader.open(str(_make_two_obs_parquet(tmp_path)))

    # A tiny composite module exposing a bare-string measure `mean_cell_mass`.
    mod = ModuleType("fake_composite")

    def mean_cell_mass(rows):
        vals = [row[CELL_MASS] for row in rows if row.get(CELL_MASS) is not None]
        return sum(vals) / len(vals)

    mod.mean_cell_mass = mean_cell_mass

    graded = grade_bare_string_measure(r, mod, "mean_cell_mass")
    direct = float(mean_cell_mass(r.rows()))

    assert graded == direct  # byte-identical
    assert graded == pytest.approx((100.0 + 150.0 + 110.0) / 3.0)

    # the resolver also accepts a module object and returns the callable
    assert resolve_measure_fn(mod, "mean_cell_mass") is mean_cell_mass


def test_resolve_measure_fn_missing_raises():
    mod = ModuleType("fake_composite")
    with pytest.raises(AttributeError):
        resolve_measure_fn(mod, "no_such_measure")
