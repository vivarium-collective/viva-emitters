# RunReader catalog + index_by/aggregate resolution (Readout-coord #2) — Plan

> REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Steps use `- [ ]`.

**Goal:** Teach `RunReader` to read the per-observable NAME CATALOG that self-describing stores now carry (sub-project #1) and to resolve a structured selector — `index_by={type, value}` and `aggregate={op, over:[ids]}` — to a numeric series **from the stored run alone** (no sim_data). This is what lets the evaluator compute the authored dnaa **vector** verdicts (e.g. "sum monomer_counts across DnaA forms").

**Architecture:** Extend `RunReader` (`viva_emitters/run_reader.py`) with `catalog()`, `resolve_id()`, a selector-aware series, and `aggregate_series()`. Catalog source per backend: parquet → `field_metadata(conn, config_sql, "<flattened path>")` for listener vectors (+ the `bulk__id` history column for bulk); xarray → the `id_<var>` coordinate; sqlite → `simulations.metadata["output_metadata"]`. Resolution: id → index via the catalog → pull/sum that element's per-tick series, returning the same `[generation, time, abs_time, value]` shape as `series()`. **Never-guess:** unknown id / absent catalog → raise a typed error (the evaluator routes to the agent bucket), never a fabricated value.

**Tech Stack:** Python 3.11+, polars, duckdb; pytest. Spec: `viva-superpowers/docs/specs/2026-06-09-readout-coordination-design.md` (#2). Builds on RunReader (merged #6).

---

## Contract (additions to RunReader)
```python
def catalog(self, observable: str) -> list[str] | None:
    # element-name catalog for an array observable; None if not catalogued
def resolve_id(self, observable: str, element_id: str) -> int:
    # index of element_id in catalog(observable); raises IdNotInCatalog if absent
def select(self, index_by: dict) -> pl.DataFrame:
    # index_by = {"type": bulk_id|monomer_id|listener_id|literal_index, "value": <id|int>}
    # -> [generation, time, abs_time, value] for that single element
def aggregate_series(self, observable: str, op: str, over: list[str]) -> pl.DataFrame:
    # op in {sum, mean, max, min}; over = element ids; -> [generation,time,abs_time,value]
```
`index_by.type` → (observable, catalog) mapping: `bulk_id`→ observable `bulk`/`bulk__count`, catalog = `bulk__id`; `monomer_id`→ observable `listeners.monomer_counts`, catalog via `output_metadata`; `listener_id`→ a named element of a listener vector via its `output_metadata` catalog; `literal_index`→ index directly, no catalog. (Keep the type→observable map small + overridable; the readout schema in #3 will supply it explicitly.)

## File map
- Modify: `viva_emitters/run_reader.py` (the methods above + per-backend catalog readers + error types `IdNotInCatalog`, `CatalogUnavailable`).
- Modify: `viva_emitters/__init__.py` (export the error types if useful).
- Test: `tests/test_run_reader_catalog.py`.

---

## Task 1: `catalog()` per backend
- [ ] **Step 1: Failing tests** — (a) bulk on the REAL dnaa hive (`/Users/eranagmon/code/v2e-invest/studies/dnaa-1-expression/parquet-runs/dnaa1-mechA-1.7e-3-7gen/dnaa1_mechA_1p7e-3_7gen`, skipif absent): `catalog("bulk")` returns the `bulk__id` list (non-empty, contains bracketed ids). (b) a SYNTHETIC parquet store written with an `output_metadata__listeners__monomer_counts` config column: `catalog("listeners.monomer_counts")` returns the names. (c) plain array with no catalog → `None`.
- [ ] **Step 2: Run → fail.**
- [ ] **Step 3: Implement `catalog(observable)`** — parquet: for `bulk` read the `bulk__id` column (first row, like `_helpers.bulk_field_ids`); else `field_metadata(conn, config_sql, observable.replace(".", "__"))` → list or None. xarray: read the `id_<var>` coord for the observable's variable (None if absent). sqlite: `load_simulation_metadata(...)["metadata"]["output_metadata"]` → dig the path. Dotted→`__`/path mapping consistent with `series()`.
- [ ] **Step 4: Run → pass.** **Step 5: Commit** — `feat(run_reader): catalog() reads per-observable name catalog (parquet/xarray/sqlite)`

## Task 2: `resolve_id()` + selector `select()`
- [ ] **Step 1: Failing tests** — on the synthetic store: `resolve_id("listeners.monomer_counts", "<name>")` → correct index; unknown id → `IdNotInCatalog`. `select({"type":"monomer_id","value":"<name>"})` → a `[generation,time,abs_time,value]` DF whose `value` is that element's per-tick count. `select({"type":"literal_index","value":3})` works with no catalog. On the real hive: `select({"type":"bulk_id","value":"<a real bulk id>"})` returns its count series.
- [ ] **Step 2: Run → fail.**
- [ ] **Step 3: Implement** — `resolve_id` via `catalog`. `select`: map `index_by.type`→(observable, catalog); resolve index (or use literal); pull element `idx` from the array column per tick (parquet: `named_idx`/a DuckDB `col[idx+1]` expression; xarray: `.isel` on the id dim; sqlite: index into the JSON array), returning the standard shape via the existing per-gen/time machinery. Absent catalog for a non-literal type → `CatalogUnavailable` (never guess).
- [ ] **Step 4: Run → pass.** **Step 5: Commit** — `feat(run_reader): resolve_id + select(index_by) for single elements`

## Task 3: `aggregate_series()`
- [ ] **Step 1: Failing tests** — synthetic store: `aggregate_series("listeners.monomer_counts", "sum", over=[id1,id2,id3])` → per-tick sum of those three elements (assert exact values + the `[generation,time,abs_time,value]` shape, ordered). `mean`/`max`/`min` likewise. One unknown id in `over` → `IdNotInCatalog` (never silently drop).
- [ ] **Step 2: Run → fail.**
- [ ] **Step 3: Implement** — resolve each id → index; pull each element's series; reduce per tick with the op (align on generation+abs_time; vectorized in polars/duckdb). 
- [ ] **Step 4: Run → pass.** **Step 5: Commit** — `feat(run_reader): aggregate_series over named ids (sum/mean/max/min)`

## Task 4: Golden — "sum across DnaA forms" on a real self-describing run
- [ ] **Step 1:** If a post-#1 self-describing run is available (or cheap to make in a tmp dir using the cache), assert `aggregate_series("listeners.monomer_counts", "sum", over=<DnaA form ids>)` returns a sane per-gen DnaA-pool series. Otherwise, document that the existing real hives predate #1 (no listener catalog) and prove the path on the synthetic store + the real-hive BULK path (which is self-describing via `bulk__id`). Skipif paths absent; NEVER modify v2e-invest.
- [ ] **Step 2: Full suite** `.venv/bin/python -m pytest -q` green (existing RunReader tests + new). **Step 3: Commit** — `test(run_reader): catalog/select/aggregate goldens`

---

## Self-Review
- Spec #2 coverage: catalog read (all backends) → T1; index_by resolution → T2; aggregate → T3; run-only (no sim_data) → all; never-guess (typed errors) → T2/T3. #3 (readout schema feeds the type→observable map) and #6 (evaluator calls these) are separate.
- No placeholders: per-backend catalog mechanics named; tests carry concrete assertions.
- Types: `select`/`aggregate_series` return the same `[generation,time,abs_time,value]` polars shape as `series()`; `catalog()->list|None`; errors `IdNotInCatalog`/`CatalogUnavailable`.

## Notes for executor
- `.venv/bin/python -m pytest`. Build synthetic self-describing stores by writing a small parquet hive with an `output_metadata__listeners__monomer_counts` config partition (mirror how `parquet_emitter._write_configuration` flattens `config["metadata"]["output_metadata"]`) — read `tests/test_run_reader.py` for the fixture idiom + `parquet_emitter.field_metadata`/`named_idx` signatures.
- Real dnaa hive is read-only. Existing real hives predate #1, so they have `bulk__id` (bulk catalog) but NOT listener-vector catalogs — test listener-vector resolution on synthetic stores.
