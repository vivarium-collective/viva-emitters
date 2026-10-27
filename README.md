# viva-emitters

Focused emitter library for [process-bigraph](https://github.com/vivarium-collective/process-bigraph)
composites. Hosts the heavy-dependency emitters out-of-tree so the
process-bigraph core stays lean.

The zero-dependency built-ins — `RAMEmitter`, `ConsoleEmitter`, `JSONEmitter`
— continue to live in `process_bigraph.emitter`. This package adds disk-backed
emitters (each a process-bigraph `Step`), an emitter-agnostic reader, and a
small per-agent emitter lifecycle registry.

## What it provides

| Emitter                 | Extra            | What it does                                              |
| ----------------------- | ---------------- | -------------------------------------------------------- |
| `SQLiteEmitter`         | `[sqlite]`       | One row per tick into a single `.db` file (stdlib).      |
| `ParquetEmitter`        | `[parquet]`      | Hive-partitioned Parquet dataset, batched writes.        |
| `LongFormParquetEmitter`| `[parquet-lite]` | Lightweight single-file long-form Parquet (pyarrow only).|
| `XArrayEmitter`         | `[xarray]`       | Zarr-backed N-dimensional dataset via xarray, unit-aware.|

Imports are lazy: importing `viva_emitters` never forces the heavy
dependencies — each emitter only becomes available when its extra is
installed.

Alongside the emitters the top-level package exports:

- **`RunReader`** (plus `RunRef`, `by_generation`) — an emitter-agnostic reader
  that returns a uniform per-tick series for any observable across
  parquet / sqlite / xarray-zarr stores, auto-detecting the backend from the
  store path.
- **Emitter lifecycle registry** — `register_emitter`, `get_emitter`,
  `unregister_emitter`, `clear_registry`, `registered_agent_ids`,
  `finalize_emitter_for_agent`: a pure-stdlib per-agent registry for tracking
  and finalizing emitters across a run.
- **`EmitterContract`** (plus `contract_for`, `register_contract`) — a minimal
  self-description (`output_kind`, `output_uri_config_key`) that lets a consumer
  read back a finished run without knowing which emitter wrote it.
- **Parquet reader helpers** re-exported from the top level
  (`create_duckdb_conn`, `read_stacked_columns`, `named_idx`,
  `ndidx_to_duckdb_expr`, ...) for working with `ParquetEmitter` datasets
  directly.

## Install

Install only the extras you need:

```bash
pip install 'viva-emitters[sqlite]'          # SQLiteEmitter (stdlib only)
pip install 'viva-emitters[parquet]'         # ParquetEmitter (duckdb, polars, ...)
pip install 'viva-emitters[parquet-lite]'    # LongFormParquetEmitter (pyarrow only)
pip install 'viva-emitters[xarray]'          # XArrayEmitter (xarray, zarr, pint, ...)
pip install 'viva-emitters[sqlite,parquet]'  # multiple at once
```

The `[sqlite]` extra is empty (Python's stdlib `sqlite3` is enough); it exists
for symmetry. The `[parquet]` extra pulls in `duckdb`, `polars`, `pyarrow`,
`fsspec`, and `tqdm`. The `[parquet-lite]` extra pulls in only `pyarrow`. The
`[xarray]` extra pulls in `xarray`, `zarr`, `zarrs`, `bottleneck`, `fsspec`,
`pint`, and `unum`.

From a checkout, for development:

```bash
uv pip install -e '.[parquet,dev]'   # or: pip install -e '.[parquet,dev]'
```

## Usage

Wire an emitter as a step inside a process-bigraph `Composite`, e.g. via the
usual `emitter_from_wires` helper:

```python
from viva_emitters import SQLiteEmitter, ParquetEmitter  # registers the Steps
from process_bigraph.emitter import emitter_from_wires

spec = emitter_from_wires(wires, address='local:SQLiteEmitter')
```

To read a stored run back — independently of which emitter wrote it:

```python
from viva_emitters import RunReader

reader = RunReader.open('path/to/run.db')   # or a .zarr dir / parquet hive dir
reader.observables()                        # available observables
reader.generations()                        # generation indices present
df = reader.series('mass')                  # uniform per-tick series (polars)
```

`RunReader` also exposes `select(index_by=...)`, `aggregate_series(...)`,
`catalog(...)`, `resolve_id(...)`, and `summary()` for indexed / aggregated
reads.

## Back-compat

The import package was renamed from `pbg_emitters` to `viva_emitters` as part of
the pbg → viva rebrand. A `pbg_emitters` shim still re-exports the public surface
(emitting a one-time `DeprecationWarning`); update imports to `viva_emitters`.

## Tests

```bash
uv pip install -e '.[parquet,dev]'
pytest
```

## License

MIT. See [LICENSE](LICENSE).
