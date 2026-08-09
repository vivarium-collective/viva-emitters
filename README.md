# viva-emitters

Focused emitter library for [process-bigraph](https://github.com/vivarium-collective/process-bigraph)
composites. Hosts the heavy-dependency emitters out-of-tree so the
process-bigraph core stays lean.

The zero-dependency built-ins — `RAMEmitter`, `ConsoleEmitter`, `JSONEmitter`
— continue to live in `process_bigraph.emitter`. This package adds:

| Emitter          | Extra        | What it does                                          |
| ---------------- | ------------ | ----------------------------------------------------- |
| `SQLiteEmitter`  | `[sqlite]`   | One row per tick into a single `.db` file (stdlib).   |
| `ParquetEmitter` | `[parquet]`  | Hive-partitioned Parquet dataset, batched writes.     |

## Install

```bash
pip install 'viva-emitters[sqlite]'        # SQLiteEmitter only
pip install 'viva-emitters[parquet]'       # ParquetEmitter only
pip install 'viva-emitters[sqlite,parquet]'  # both
```

The `[sqlite]` extra is empty (Python's stdlib `sqlite3` is enough); it exists
for symmetry. The `[parquet]` extra pulls in `duckdb`, `polars`, `pyarrow`,
`fsspec`, and `tqdm`.

## Usage

```python
from viva_emitters import SQLiteEmitter, ParquetEmitter

# Wire either as a step inside a process-bigraph Composite, e.g. via the
# usual emitter_from_wires helper:
#
#   from process_bigraph.emitter import emitter_from_wires
#   spec = emitter_from_wires(wires, address='local:SQLiteEmitter')
```

Reader helpers for the Parquet datasets (`create_duckdb_conn`,
`read_stacked_columns`, `named_idx`, `ndidx_to_duckdb_expr`, ...) are also
re-exported from the top-level package.

## Tests

```bash
pip install -e '.[parquet,dev]'
pytest
```

## License

MIT. See [LICENSE](LICENSE).
