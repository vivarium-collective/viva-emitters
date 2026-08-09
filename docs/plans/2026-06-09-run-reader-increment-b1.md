# RunReader (Increment B1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development / executing-plans. Steps use `- [ ]` checkboxes.

**Goal:** A single emitter-agnostic `RunReader` in `viva-emitters` that, given a stored run (parquet / sqlite / xarray-zarr), returns a uniform per-tick **series** for any observable, tagged with **generation** and **time** — the foundation the study-outcome evaluator (Increment B2) reads from.

**Architecture:** One class `RunReader` wrapping the existing per-backend read primitives (`dataset_sql`/`read_stacked_columns`, `load_history`/raw sqlite, `xr.open_datatree`+`ForestView`). It normalizes all three into a polars DataFrame with columns `["generation", "time", "value"]`. Generation structure is first-class in every backend (parquet `generation` column; sqlite `generation` in state + per-gen `simulation_id`; zarr `emitstep_gen=N`/`time_gen=N` coords) — the reader surfaces it, never infers it. NB: `global_time` is **gen-local** (resets each generation); the reader returns gen-local `time` plus an absolute `abs_time` (cumulative).

**Tech Stack:** Python 3.11+, polars, duckdb, sqlite3 (stdlib), xarray+zarr, numpy; pytest. Backed by the grounding in the program memory + spec `docs/specs/2026-06-09-study-run-outcome-spine-design.md` (in viva-superpowers).

**Repo:** `viva-emitters` only.

---

## Contract (the shape every backend normalizes to)

```python
# viva_emitters/run_reader.py
@dataclass
class RunRef:
    store: str            # path to the run's store (hive dir | .db | .zarr/datatree root)
    kind: str | None = None   # "parquet" | "sqlite" | "xarray"; auto-detected if None

class RunReader:
    @classmethod
    def open(cls, store: str | RunRef, kind: str | None = None) -> "RunReader": ...
    @property
    def kind(self) -> str: ...
    def observables(self) -> list[str]: ...          # available observable ids
    def generations(self) -> list[int]: ...          # sorted gen indices present
    def series(self, observable: str) -> "pl.DataFrame":
        # columns: generation:int, time:float (gen-local), abs_time:float (cumulative), value:float
        # ordered by (generation, time)
```

`observable` ids are the backend-native names the reader also lists from `observables()` (parquet flattened column e.g. `listeners__mass__cell_mass`; sqlite/xarray dotted state path e.g. `listeners.mass.cell_mass`). The reader accepts the dotted form and maps it to the backend's native key. **Arithmetic expressions over observables are NOT B1's job** — the evaluator (B2) composes them from single-observable series. `series` raises `KeyError` for an unknown observable (the evaluator turns that into the agent bucket, never a guess).

## File Structure
- Create: `viva_emitters/run_reader.py` — `RunRef`, `RunReader`, backend detection, the three readers, `_cumulative_time`.
- Modify: `viva_emitters/__init__.py` — export `RunReader`, `RunRef`.
- Test: `tests/test_run_reader.py` — fixtures per backend (mirror `tests/test_sqlite_emitter.py` / `test_parquet_emitter.py` / `test_xarray_emitter.py` + `conftest.py`).

---

## Task 1: Skeleton + backend detection + cumulative-time helper

**Files:** Create `viva_emitters/run_reader.py`; Test `tests/test_run_reader.py`.

- [ ] **Step 1: Failing test for detection + the time helper**

```python
# tests/test_run_reader.py
import polars as pl
from pathlib import Path
from viva_emitters.run_reader import RunReader, RunRef, _cumulative_time

def test_detect_sqlite(tmp_path):
    db = tmp_path / "runs_history.db"; db.write_bytes(b"")  # presence-based for kind
    assert RunReader.open(str(db)).kind == "sqlite"

def test_detect_parquet(tmp_path):
    hive = tmp_path / "exp" / "history"; hive.mkdir(parents=True)
    assert RunReader.open(str(tmp_path / "exp")).kind == "parquet"

def test_detect_xarray(tmp_path):
    z = tmp_path / "store.zarr"; z.mkdir()
    assert RunReader.open(str(z)).kind == "xarray"

def test_cumulative_time_stitches_gen_local_resets():
    # two generations, time resets to 0 each gen
    df = pl.DataFrame({"generation": [1,1,2,2], "time": [0.0, 10.0, 0.0, 10.0],
                       "value": [1,2,3,4]})
    out = _cumulative_time(df)
    assert out["abs_time"].to_list() == [0.0, 10.0, 11.0, 21.0]  # +max_prev+1 per gen
```

- [ ] **Step 2: Run — expect ImportError / fail**

Run: `.venv/bin/python -m pytest tests/test_run_reader.py -v`
Expected: FAIL (module/class missing).

- [ ] **Step 3: Implement skeleton + detection + `_cumulative_time`**

```python
# viva_emitters/run_reader.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import polars as pl

@dataclass
class RunRef:
    store: str
    kind: str | None = None

def _detect_kind(store: str) -> str:
    p = Path(store)
    if p.suffix == ".db" or (p.is_file() and p.suffix in {".db", ".sqlite"}):
        return "sqlite"
    if p.suffix == ".zarr" or (p.is_dir() and (p / ".zgroup").exists()) or str(p).endswith(".zarr"):
        return "xarray"
    if (p / "history").exists() or p.suffix == ".db" is False and (p.is_dir() and any(p.glob("**/history"))):
        return "parquet"
    # final fallbacks
    if p.is_dir():
        return "parquet"
    return "sqlite"

def _cumulative_time(df: pl.DataFrame) -> pl.DataFrame:
    """Add abs_time: gen-local `time` stitched into one axis (+max_prev_time+1 per gen)."""
    if df.is_empty():
        return df.with_columns(pl.lit(0.0).alias("abs_time"))
    gens = sorted(df["generation"].unique().to_list())
    offset, offsets = 0.0, {}
    for g in gens:
        offsets[g] = offset
        gmax = df.filter(pl.col("generation") == g)["time"].max()
        offset += (gmax or 0.0) + 1.0
    return df.with_columns(
        (pl.col("time") + pl.col("generation").replace(offsets, default=0.0)).alias("abs_time")
    )

class RunReader:
    def __init__(self, ref: RunRef):
        self._ref = ref
        self._kind = ref.kind or _detect_kind(ref.store)

    @classmethod
    def open(cls, store, kind: str | None = None) -> "RunReader":
        if isinstance(store, RunRef):
            return cls(store)
        return cls(RunRef(store=str(store), kind=kind))

    @property
    def kind(self) -> str:
        return self._kind
```

- [ ] **Step 4: Run — expect PASS** (`.venv/bin/python -m pytest tests/test_run_reader.py -v`)
- [ ] **Step 5: Commit** — `git add viva_emitters/run_reader.py tests/test_run_reader.py && git commit -m "feat(run_reader): skeleton, backend detection, cumulative-time helper"`

---

## Task 2: SQLite series + observables + generations

Read the emitter `history` table directly (`step`, `global_time`, `state` JSON). Generation comes from a `generation` key in the state dict; observable resolved by dotted path into the state tree.

**Files:** Modify `viva_emitters/run_reader.py`; Test `tests/test_run_reader.py`.

- [ ] **Step 1: Failing test (build a tiny 2-gen sqlite emitter db — mirror `tests/test_sqlite_emitter.py` for the table DDL)**

```python
import sqlite3, json
def _make_sqlite(tmp_path):
    db = tmp_path / "h.db"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE history (simulation_id TEXT, step INTEGER, global_time REAL, state TEXT, PRIMARY KEY(simulation_id, step))")
    rows = [
        ("s", 0, 0.0, json.dumps({"generation": 1, "listeners": {"mass": {"cell_mass": 100.0}}})),
        ("s", 1, 5.0, json.dumps({"generation": 1, "listeners": {"mass": {"cell_mass": 150.0}}})),
        ("s", 2, 0.0, json.dumps({"generation": 2, "listeners": {"mass": {"cell_mass": 110.0}}})),
    ]
    con.executemany("INSERT INTO history VALUES (?,?,?,?)", rows); con.commit(); con.close()
    return db

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
```

- [ ] **Step 2: Run — expect fail** (`series`/`generations`/`observables` not implemented).
- [ ] **Step 3: Implement the sqlite branch**

Add to `RunReader`: dispatch `series`/`observables`/`generations` on `self._kind`; implement `_sqlite_*` helpers that open the db, read `history` ordered by `step`, JSON-parse `state`, pull `generation` and the dotted observable via a `_dig(state, "a.b.c")` helper, build the polars DF, then `_cumulative_time`. `observables()` flattens the first row's state tree to dotted leaf paths.

- [ ] **Step 4: Run — expect PASS.**
- [ ] **Step 5: Commit** — `feat(run_reader): sqlite backend (series/observables/generations)`

---

## Task 3: Parquet series + observables + generations

Use the existing primitives: `dataset_sql(out_dir, [experiment_id])` → `history_sql`; `read_stacked_columns(history_sql, [col])` → polars DF with `generation, time, <col>`. Parse `out_dir`/`experiment_id` from the hive path.

**Files:** Modify `viva_emitters/run_reader.py`; Test `tests/test_run_reader.py`.

- [ ] **Step 1: Failing test — build a tiny hive (mirror `tests/test_parquet_emitter.py`)**

Write 2 generations of parquet under `{out}/{exp}/history/experiment_id=exp/variant=0/lineage_seed=0/generation={1,2}/agent_id=0/*.pq` with columns `time`, `listeners__mass__cell_mass`. Assert:
```python
r = RunReader.open(str(out / "exp"))
assert r.kind == "parquet"
assert r.generations() == [1, 2]
s = r.series("listeners.mass.cell_mass")   # dotted in, mapped to __ column
assert set(s.columns) >= {"generation", "time", "abs_time", "value"}
assert s["value"].to_list() == [expected...]
assert "listeners.mass.cell_mass" in r.observables()  # de-flattened from columns
```

- [ ] **Step 2: Run — expect fail.**
- [ ] **Step 3: Implement the parquet branch** — map dotted→`__` column; `dataset_sql`+`read_stacked_columns([col])`; rename col→`value`, keep `generation`,`time`; `_cumulative_time`. `observables()` = `list_columns(...)` de-flattened (`__`→`.`), excluding id cols. `generations()` = distinct `generation`.
- [ ] **Step 4: Run — expect PASS.**
- [ ] **Step 5: Commit** — `feat(run_reader): parquet backend via dataset_sql/read_stacked_columns`

---

## Task 4: Xarray series + observables + generations

Open with `xr.open_datatree(store, engine="zarr")`. Per-gen dims `emitstep_gen=N` + timestamp var `time_gen=N`. Map observable path → variable via `ForestView.leaves()` / `LeafView.var_name()`.

**Files:** Modify `viva_emitters/run_reader.py`; Test `tests/test_run_reader.py`.

- [ ] **Step 1: Failing test — build a tiny 2-gen datatree (mirror `tests/test_xarray_emitter.py`)** asserting `generations()==[1,2]`, `series("listeners.mass.cell_mass")` yields per-gen rows with `time` from `time_gen=N` and the value array, and `observables()` lists the leaf path.
- [ ] **Step 2: Run — expect fail.**
- [ ] **Step 3: Implement the xarray branch** — open datatree; enumerate generations from `emitstep_gen=N` coord names; for the observable, resolve leaf var (via `ForestView` or direct var lookup), read per gen the `(time_gen=N, var)` pair into rows; build DF; `_cumulative_time`. `observables()` = leaf paths.
- [ ] **Step 4: Run — expect PASS.**
- [ ] **Step 5: Commit** — `feat(run_reader): xarray/zarr backend (per-gen coords)`

---

## Task 5: Export + cross-backend equivalence + by_generation helper

**Files:** Modify `viva_emitters/__init__.py`, `viva_emitters/run_reader.py`; Test `tests/test_run_reader.py`.

- [ ] **Step 1: Failing test** — `from viva_emitters import RunReader, RunRef` works; add `by_generation(df) -> dict[int, pl.DataFrame]`; and an equivalence test: the SAME tiny dataset written to sqlite AND parquet yields identical `series(...)` `generation`/`time`/`value` (proves the normalization is backend-agnostic).
- [ ] **Step 2: Run — expect fail.**
- [ ] **Step 3: Implement** — add `by_generation`; export `RunReader`/`RunRef` from `__init__.py` (guard imports like the existing optional-extra pattern in `__init__.py`).
- [ ] **Step 4: Run full suite** — `.venv/bin/python -m pytest -q` — all green (existing 40 + new).
- [ ] **Step 5: Commit** — `feat(run_reader): export + by_generation + cross-backend equivalence`

---

## Self-Review
- **Spec coverage:** §4.2 (emitter-aware reader in viva-emitters, uniform interface over the 3 backends, observable resolution) → Tasks 2-5; generation/time normalization (gen-local reset stitched) → Task 1 + each backend. ✓ (Evaluator, schema, migration are B2/B3 — out of scope.)
- **Placeholder scan:** fixtures reference the existing per-backend test files for exact store-construction patterns; everything else is complete code.
- **Type consistency:** `series()` returns polars DF `["generation","time","abs_time","value"]` in all four tasks; `RunReader.open` / `.kind` / `.observables` / `.generations` / `.series` signatures stable across tasks; `_cumulative_time(df)->df` consistent.

## Notes for the executor
- Run via `.venv/bin/python -m pytest` (uv venv; no `pip` — use `uv pip` if installing).
- Reuse `tests/conftest.py` + the three `test_*_emitter.py` files for the exact store-construction idioms of each backend; don't invent store layouts.
- Keep `RunReader` import-light: import duckdb/xarray lazily inside the backend branches so importing the module doesn't require all extras (mirror `__init__.py`'s guarded imports).
- If a real backend store is needed to sanity-check, dnaa-1 has a real parquet hive (`v2e-invest/studies/dnaa-1-expression/parquet-runs/...`) and dnaa-2 a real sqlite (`v2e-invest/studies/dnaa-2-nucleotide-balance/runs.db`) — read-only, do not modify.
