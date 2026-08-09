"""Tests for viva_emitters.parquet_emitter.

Helper-function tests are ported from v2ecoli's
``tests/test_parquet_emitter_ported.py`` ``TestHelperFunctions`` class
(which in turn comes from vEcoli). The vEcoli-specific dtype-override
test was dropped since viva-emitters has no built-in overrides.

Integration tests exercise the emitter directly through its
``update`` / ``query`` / ``close`` API, without a Composite.
"""

from __future__ import annotations

import os
import re
import shutil
import tempfile

import pytest

try:
    import duckdb
    import numpy as np
    import polars as pl
except ImportError as e:  # pragma: no cover - skip whole module without extra
    pytest.skip(
        f"viva-emitters [parquet] extra not installed (missing: {e.name})",
        allow_module_level=True,
    )

from viva_emitters.parquet_emitter import (
    ParquetEmitter,
    _is_bookkeeping_field,
    _split_structured_arrays,
    create_duckdb_conn,
    flatten_dict,
    list_columns,
    named_idx,
    ndidx_to_duckdb_expr,
    np_dtype,
    quote_columns,
    union_pl_dtypes,
)


# ============================================================================
# Helper-function tests — ported from v2ecoli ``TestHelperFunctions``.
# ============================================================================


class TestHelperFunctions:
    @pytest.fixture
    def query_conn(self):
        conn = duckdb.connect(":memory:")
        df = pl.DataFrame(  # noqa: F841
            {
                "a": [[0.1, 0.0, 0.3], [0.4, 0.5, 0.0], [None, 0.8, 0.9]],
                "b": [
                    [[0.1, 0.2], [0.3, None]],
                    [[0.5, 0.6], [0.0, 0.8]],
                    [[0.9, 0.0], [1.1, 1.2]],
                ],
                "c": [[[0.1, 0.2], [0.3]], [[0.5], [0.0, 0.8]], [[0.9], [1.1]]],
            }
        )
        conn.sql("CREATE OR REPLACE TABLE test_table AS SELECT * FROM df")
        yield conn

    def test_named_idx(self, query_conn):
        col_expr = named_idx("a", ["col1", "col2", "col3"], [[0, 1, 2]])
        result = query_conn.sql(f"SELECT {col_expr} FROM test_table").pl()
        expected = pl.DataFrame(
            {"col1": [0.1, 0.4, None], "col2": [0.0, 0.5, 0.8], "col3": [0.3, 0.0, 0.9]}
        )
        assert result.equals(expected)

        col_expr = named_idx(
            "a", ["col1", "col2", "col3"], [[0, 1, 2]], zero_to_null=True
        )
        result = query_conn.sql(f"SELECT {col_expr} FROM test_table").pl()
        expected = pl.DataFrame(
            {
                "col1": [0.1, 0.4, None],
                "col2": [None, 0.5, 0.8],
                "col3": [0.3, None, 0.9],
            }
        )
        assert result.equals(expected)

        col_expr = named_idx(
            "b", ["col1", "col2", "col3", "col4"], [[0, 1], [0, 1]], zero_to_null=True
        )
        result = query_conn.sql(f"SELECT {col_expr} FROM test_table").pl()
        expected = pl.DataFrame(
            {
                "col1": [0.1, 0.5, 0.9],
                "col2": [0.2, 0.6, None],
                "col3": [0.3, None, 1.1],
                "col4": [None, 0.8, 1.2],
            }
        )
        assert result.equals(expected)

    def test_ndidx_to_duckdb_expr(self, query_conn):
        expr = ndidx_to_duckdb_expr("b", [0, 1])
        result = query_conn.sql(f"SELECT {expr} FROM test_table").pl()
        expected = pl.DataFrame({"b": [[[0.2]], [[0.6]], [[0.0]]]})
        assert result.equals(expected)

        expr = ndidx_to_duckdb_expr("b", [":", [True, False]])
        result = query_conn.sql(f"SELECT {expr} FROM test_table").pl()
        expected = pl.DataFrame({"b": [[[0.1], [0.3]], [[0.5], [0.0]], [[0.9], [1.1]]]})
        assert result.equals(expected)

        expr = ndidx_to_duckdb_expr("c", [[0], ":"])
        result = query_conn.sql(f"SELECT {expr} FROM test_table").pl()
        expected = pl.DataFrame({"c": [[[0.1, 0.2]], [[0.5]], [[0.9]]]})
        assert result.equals(expected)

    def test_flatten_dict(self):
        assert flatten_dict({"a": 1, "b": 2}) == {"a": 1, "b": 2}
        assert flatten_dict({"a": {"b": 1, "c": 2}, "d": 3}) == {
            "a__b": 1,
            "a__c": 2,
            "d": 3,
        }
        assert flatten_dict({"a": {"b": {"c": {"d": 1}}}, "e": 2}) == {
            "a__b__c__d": 1,
            "e": 2,
        }
        assert flatten_dict({}) == {}
        nested = flatten_dict({"a": [1, 2, 3], "b": {"c": np.array([4, 5, 6])}})
        assert nested["a"] == [1, 2, 3]
        np.testing.assert_array_equal(nested["b__c"], np.array([4, 5, 6]))

    def test_np_dtype(self):
        # Basic types
        assert np_dtype(1.0, "float_field") == np.float64
        assert np_dtype(True, "bool_field") == np.bool_
        assert np_dtype("text", "string_field") == np.dtypes.StringDType
        assert np_dtype(42, "int_field") == np.int64

        # Arrays with various dimensions
        assert np_dtype(np.array([1, 2, 3]), "array1d_field") == np.int64
        assert np_dtype(np.array([[1, 2], [3, 4]]), "array2d_field") == np.int64
        # Empty arrays still have a dtype
        assert np_dtype(np.array([]), "empty_array_field") == np.float64

        # Generic override path (no built-in overrides in viva-emitters)
        overrides = {
            "uint16_field": "UInt16",
            "listeners__*__index": "UInt32",
        }
        assert np_dtype(42, "uint16_field", overrides) == np.uint16
        assert np_dtype(42, "listeners__foo__index", overrides) == np.uint32
        # Non-matching name falls through to value-based dispatch.
        assert np_dtype(42, "other_field", overrides) == np.int64

        # Raise to trigger Polars fallback in the emitter
        with pytest.raises(ValueError, match="empty_list_field has unsupported"):
            np_dtype([[], [], None], "empty_list_field")
        with pytest.raises(ValueError, match="none_field has unsupported"):
            np_dtype(None, "none_field")
        with pytest.raises(ValueError, match="complex_field has unsupported type"):
            np_dtype(complex(1, 2), "complex_field")

    def test_union_pl_dtypes(self):
        # Basic types
        with pytest.raises(
            TypeError, match=re.escape("Incompatible inner types for field")
        ):
            union_pl_dtypes(pl.Int32, pl.Int64, "fail")
        with pytest.raises(
            TypeError, match=re.escape("Incompatible inner types for field")
        ):
            union_pl_dtypes(pl.Float32, pl.String, "fail")

        # Nested types
        with pytest.raises(
            TypeError, match=re.escape("Incompatible inner types for field")
        ):
            union_pl_dtypes(pl.List(pl.Int16), pl.List(pl.Int64), "nest")
        with pytest.raises(
            TypeError, match=re.escape("Incompatible inner types for field")
        ):
            union_pl_dtypes(pl.List(pl.UInt16), pl.List(pl.String), "nest_fail")
        with pytest.raises(
            TypeError, match=re.escape("Incompatible inner types for field")
        ):
            union_pl_dtypes(
                pl.List(pl.List(pl.UInt16)), pl.List(pl.String), "nest_fail"
            )
        with pytest.raises(
            TypeError, match=re.escape("Incompatible inner types for field")
        ):
            union_pl_dtypes(
                pl.List(pl.UInt16), pl.List(pl.Array(pl.String, (1,))), "nest_fail"
            )
        assert union_pl_dtypes(
            pl.List(pl.UInt16), pl.List(pl.Int64), "force_u32", pl.UInt32
        ) == pl.List(pl.UInt32)

        # Forced types
        assert union_pl_dtypes(pl.Int16, pl.UInt8, "force_u16", pl.UInt16) == pl.UInt16
        assert union_pl_dtypes(pl.UInt16, pl.Int64, "force_u32", pl.UInt32) == pl.UInt32
        assert (
            union_pl_dtypes(pl.UInt16, pl.String, "force_u32", pl.UInt32) == pl.UInt32
        )
        assert union_pl_dtypes(
            pl.List(pl.UInt16), pl.List(pl.String), "force_u32", pl.UInt32
        ) == pl.List(pl.UInt32)
        assert union_pl_dtypes(
            pl.List(pl.UInt16), pl.List(pl.Int64), "force_u32", pl.UInt32
        ) == pl.List(pl.UInt32)
        assert union_pl_dtypes(
            pl.Array(pl.UInt16, (1, 1)),
            pl.List(pl.List(pl.Int64)),
            "force_u16",
            pl.UInt16,
        ) == pl.List(pl.List(pl.UInt16))

        # Null merge
        assert union_pl_dtypes(pl.Null, pl.Int64, "null_merge") == pl.Int64
        assert union_pl_dtypes(pl.Null, pl.Float64, "force_u16", pl.UInt16) == pl.UInt16
        assert union_pl_dtypes(
            pl.Null, pl.List(pl.Int64), "force_u16", pl.UInt16
        ) == pl.List(pl.UInt16)
        assert union_pl_dtypes(
            pl.List(pl.Null), pl.List(pl.List(pl.Float32)), "null_merge"
        ) == pl.List(pl.List(pl.Float32))
        assert union_pl_dtypes(
            pl.Array(pl.Null, (1, 1, 1)),
            pl.List(pl.Array(pl.Float32, (1, 1))),
            "null_merge",
        ) == pl.List(pl.List(pl.List(pl.Float32)))
        assert union_pl_dtypes(
            pl.List(pl.Null), pl.List(pl.String), "force_u16", pl.UInt16
        ) == pl.List(pl.UInt16)
        assert union_pl_dtypes(
            pl.List(pl.Null), pl.List(pl.List(pl.Int32)), "force_u32", pl.UInt32
        ) == pl.List(pl.List(pl.UInt32))
        assert union_pl_dtypes(
            pl.List(pl.Null), pl.List(pl.List(pl.List(pl.Int32))), "null_merge"
        ) == pl.List(pl.List(pl.List(pl.Int32)))
        assert union_pl_dtypes(
            pl.List(pl.Null),
            pl.List(pl.List(pl.List(pl.Int32))),
            "force_u32",
            pl.UInt32,
        ) == pl.List(pl.List(pl.List(pl.UInt32)))

    def test_quote_columns(self):
        # Singles
        assert quote_columns("simple") == '"simple"'
        assert quote_columns("with spaces") == '"with spaces"'
        assert quote_columns("with-hyphens") == '"with-hyphens"'
        assert quote_columns("with[brackets]") == '"with[brackets]"'
        assert quote_columns("with/slashes") == '"with/slashes"'
        # Pre-quoted (must be escaped)
        assert quote_columns('already"quoted') == '"already""quoted"'
        assert quote_columns('"fully"quoted"') == '"""fully""quoted"""'
        # Lists
        assert quote_columns(["col1", "col2", "col3"]) == ['"col1"', '"col2"', '"col3"']
        assert quote_columns(["with spaces", "with-hyphens"]) == [
            '"with spaces"',
            '"with-hyphens"',
        ]
        assert quote_columns(["normal", "space here", "hyphen-here", 'quote"here']) == [
            '"normal"',
            '"space here"',
            '"hyphen-here"',
            '"quote""here"',
        ]
        # Empty cases
        assert quote_columns("") == '""'
        assert quote_columns([]) == []

        # End-to-end with DuckDB
        with tempfile.TemporaryDirectory() as tmp_path:
            test_file = os.path.join(tmp_path, "weird_cols.parquet")
            test_data = pl.DataFrame(
                {
                    "simple": [1, 2, 3],
                    "with spaces": [4, 5, 6],
                    "with-hyphens": [7, 8, 9],
                    "with[brackets]": [10, 11, 12],
                    "with/slashes": [13, 14, 15],
                    'has"quote': [16, 17, 18],
                    "dot.name": [19, 20, 21],
                    "colon:name": [22, 23, 24],
                }
            )
            test_data.write_parquet(test_file, statistics=False)
            conn = create_duckdb_conn()
            for col in test_data.columns:
                quoted_col = quote_columns(col)
                result = conn.sql(f"SELECT {quoted_col} FROM '{test_file}'").pl()
                assert result.shape == (3, 1)
                assert result.columns[0] == col
                assert result[col].to_list() == test_data[col].to_list()
            weird_cols = ["with spaces", "with-hyphens", "with[brackets]", 'has"quote']
            quoted_cols = ", ".join(quote_columns(weird_cols))
            result = conn.sql(f"SELECT {quoted_cols} FROM '{test_file}'").pl()
            assert result.shape == (3, 4)
            for col in weird_cols:
                assert col in result.columns
                assert result[col].to_list() == test_data[col].to_list()

    def test_list_columns(self):
        with tempfile.TemporaryDirectory() as tmp_path:
            test_file = os.path.join(tmp_path, "test.parquet")
            test_data = pl.DataFrame(
                {
                    "col_a": [1, 2, 3],
                    "col_b": [4.0, 5.0, 6.0],
                    "listeners__mass__cell_mass": [7.0, 8.0, 9.0],
                    "listeners__mass__dry_mass": [10.0, 11.0, 12.0],
                    "listeners__growth__instantaneous_growth_rate": [0.1, 0.2, 0.3],
                    "bulk": [[1, 2], [3, 4], [5, 6]],
                }
            )
            test_data.write_parquet(test_file, statistics=False)
            conn = create_duckdb_conn()
            subquery = f"SELECT * FROM '{test_file}'"
            all_cols = list_columns(conn, subquery)
            assert len(all_cols) == 6
            assert "col_a" in all_cols
            assert "col_b" in all_cols
            assert "listeners__mass__cell_mass" in all_cols
            listener_cols = list_columns(conn, subquery, "listeners__*")
            assert len(listener_cols) == 3
            assert all(col.startswith("listeners__") for col in listener_cols)
            mass_cols = list_columns(conn, subquery, "listeners__mass__*")
            assert len(mass_cols) == 2
            assert "listeners__mass__cell_mass" in mass_cols
            assert "listeners__mass__dry_mass" in mass_cols
            no_match = list_columns(conn, subquery, "nonexistent__*")
            assert len(no_match) == 0
            col_pattern = list_columns(conn, subquery, "col_?")
            assert len(col_pattern) == 2
            assert "col_a" in col_pattern
            assert "col_b" in col_pattern
            exact = list_columns(conn, subquery, "bulk")
            assert exact == ["bulk"]


# ============================================================================
# Structured-array splitting helpers.
# ============================================================================


class TestSplitStructuredArrays:
    def test_split_structured_arrays_explodes_record_array(self):
        """Each non-bookkeeping field of a structured array becomes its own column."""
        rec = np.array(
            [(1, 10), (2, 20)], dtype=[("id", "i8"), ("count", "i8")]
        )
        out = _split_structured_arrays({"bulk": rec})

        assert set(out.keys()) == {"bulk__id", "bulk__count"}
        np.testing.assert_array_equal(out["bulk__id"], np.array([1, 2], dtype="i8"))
        np.testing.assert_array_equal(out["bulk__count"], np.array([10, 20], dtype="i8"))

    def test_split_structured_arrays_drops_bookkeeping_fields(self):
        """Fields matching ``_is_bookkeeping_field`` are silently dropped."""
        rec = np.array(
            [(1, 10, 0.5, 1.0, 1), (2, 20, 0.6, 2.0, 0)],
            dtype=[
                ("id", "i8"),
                ("count", "i8"),
                ("protein_submass", "f8"),
                ("massDiff_water", "f8"),
                ("_entryState", "i1"),
            ],
        )
        out = _split_structured_arrays({"unique": rec})

        assert set(out.keys()) == {"unique__id", "unique__count"}
        assert "unique__protein_submass" not in out
        assert "unique__massDiff_water" not in out
        assert "unique__massDiff_" not in out
        assert "unique___entryState" not in out

    def test_split_structured_arrays_passthrough_non_structured(self):
        """Scalars, lists, and plain (non-record) ndarrays pass through unchanged."""
        plain = np.array([1, 2, 3], dtype=np.int32)
        flat = {
            "scalar": 42,
            "string": "hello",
            "list_of_ints": [1, 2, 3],
            "plain_ndarray": plain,
            "ratio": 0.75,
            "bool_value": True,
        }
        out = _split_structured_arrays(flat)

        assert set(out.keys()) == set(flat.keys())
        assert out["scalar"] == 42
        assert out["string"] == "hello"
        assert out["list_of_ints"] == [1, 2, 3]
        # Plain ndarray must be the same object (no copy on passthrough).
        assert out["plain_ndarray"] is plain
        assert out["ratio"] == 0.75
        assert out["bool_value"] is True

    def test_is_bookkeeping_field_patterns(self):
        """Pattern coverage for the predicate."""
        assert _is_bookkeeping_field("protein_submass")
        assert _is_bookkeeping_field("_submass")
        assert _is_bookkeeping_field("massDiff_water")
        assert _is_bookkeeping_field("massDiff_")
        assert _is_bookkeeping_field("_entryState")
        # Non-matches.
        assert not _is_bookkeeping_field("id")
        assert not _is_bookkeeping_field("count")
        assert not _is_bookkeeping_field("submass_count")  # not the suffix
        assert not _is_bookkeeping_field("entryState")     # missing leading "_"
        assert not _is_bookkeeping_field("MassDiff_x")     # case-sensitive


# ============================================================================
# Integration tests — exercise ParquetEmitter through update/query/close.
# ============================================================================


class TestParquetEmitterIntegration:
    @pytest.fixture
    def temp_dir(self):
        tmp = tempfile.mkdtemp(prefix="parquet_emitter_")
        yield tmp
        shutil.rmtree(tmp, ignore_errors=True)

    def test_initialization_requires_destination(self, temp_dir, core):
        """Construction needs either out_dir or out_uri."""
        e = ParquetEmitter(
            config={"out_dir": temp_dir, "threaded": False}, core=core,
        )
        assert e.out_uri == os.path.abspath(temp_dir)
        assert e.batch_size == 400

        # out_uri wins over the absolute-path resolution.
        e2 = ParquetEmitter(
            config={
                "out_uri": "/some/explicit/path",
                "batch_size": 100,
                "threaded": False,
            },
            core=core,
        )
        assert e2.out_uri == "/some/explicit/path"
        assert e2.batch_size == 100

        with pytest.raises(ValueError, match="out_dir.*out_uri"):
            ParquetEmitter(config={"threaded": False}, core=core)

    def test_del_is_non_blocking_on_pending_future(self, temp_dir, core):
        """``__del__`` must never block on ``last_batch_future``.

        GC can invoke ``__del__`` re-entrantly at an arbitrary moment (e.g.
        mid type-dispatch while a *later* composite is being built); waiting on
        the write future there deadlocks. Regression for that hang: the
        finalizer must return promptly even when the future never resolves.
        """
        import threading
        from concurrent.futures import Future

        e = ParquetEmitter(
            config={"out_dir": temp_dir, "threaded": True}, core=core,
        )
        e.last_batch_future.result()  # let construction settle
        # Deadlock condition: a write future that never resolves on an open
        # emitter. Stub the flush so the pending future stays in place.
        e._flush_partial_batch = lambda: None
        e.last_batch_future = Future()  # pending forever
        e._closed = False

        done = threading.Event()
        threading.Thread(
            target=lambda: (type(e).__del__(e), done.set()), daemon=True
        ).start()
        assert done.wait(timeout=5.0), (
            "__del__ blocked on last_batch_future — a finalizer must not wait "
            "on a write future (it can be called re-entrantly during GC)"
        )
        assert e._closed is True

    def test_round_trip_basic(self, temp_dir, core):
        """Emit a few rows of mixed types and read them back via query()."""
        emitter = ParquetEmitter(
            config={
                "out_dir": temp_dir,
                "batch_size": 2,
                "threaded": False,
                "metadata": {"experiment_id": "round_trip"},
            },
            core=core,
        )
        emitter.last_batch_future.result()

        emitter.update({"time": 1.0, "value": 10, "ratio": 0.5})
        emitter.update({"time": 2.0, "value": 20, "ratio": 0.75})
        # Third update fills past the batch boundary.
        emitter.update({"time": 3.0, "value": 30, "ratio": 1.0})
        emitter.close(success=False)

        df = emitter.query()
        assert sorted(df.columns) == ["ratio", "time", "value"]
        # Order isn't guaranteed across files, so sort before comparing.
        df_sorted = df.sort("time")
        assert df_sorted["time"].to_list() == [1.0, 2.0, 3.0]
        assert df_sorted["value"].to_list() == [10, 20, 30]
        assert df_sorted["ratio"].to_list() == [0.5, 0.75, 1.0]

    def test_drop_shape_varying_column_does_not_crash(self, temp_dir, core):
        """A column that is a scalar at one tick and a length-N array at another
        cannot live in a single parquet column. The emitter DROPS it (with a
        warning) and finishes the run, instead of raising and sinking the whole
        simulation. The other columns persist.

        Regression: a full emit of baseline listeners hit
        ``listeners__rnap_data__rna_init_event_per_cistron`` (Int64 at t=0, then
        Array(Int64, 4538) later); ``union_pl_dtypes`` raised TypeError mid-run.
        """
        import numpy as np

        emitter = ParquetEmitter(
            config={
                "out_dir": temp_dir,
                "batch_size": 8,  # keep both ticks in one batch
                "threaded": False,
                "metadata": {"experiment_id": "ragged"},
            },
            core=core,
        )
        emitter.last_batch_future.result()

        with pytest.warns(UserWarning, match="dropping column"):
            emitter.update({"time": 1.0, "keep": 10, "ragged": 5})
            emitter.update({"time": 2.0, "keep": 20,
                            "ragged": np.array([1, 2, 3, 4])})
        emitter.close(success=False)

        assert "ragged" in emitter._dropped_cols
        df = emitter.query()
        assert "ragged" not in df.columns
        assert sorted(df.columns) == ["keep", "time"]
        df_sorted = df.sort("time")
        assert df_sorted["time"].to_list() == [1.0, 2.0]
        assert df_sorted["keep"].to_list() == [10, 20]

    def test_round_trip_quantity_unit_bearing_port(self, temp_dir, core):
        """A pint.Quantity (a ``quantity[...]`` port value) is stripped to its
        magnitude under the SAME column name, and the unit is recorded as
        Parquet file metadata — instead of crashing ``pl.Series`` on numpy
        object dtype, and without renaming the column.

        Regression: a Quantity made ``pl.Series([Quantity(...)])`` raise
        ``cannot parse numpy data type dtype('O')``. The column-name
        preservation matters so downstream name-based queries / reports keep
        working when a port is promoted to a unit-bearing type.
        """
        import glob
        import json

        import pyarrow.parquet as pq
        from bigraph_schema.units import units as ureg

        emitter = ParquetEmitter(
            config={
                "out_dir": temp_dir,
                "batch_size": 2,
                "threaded": False,
                "metadata": {"experiment_id": "quantity_port"},
            },
            core=core,
        )
        emitter.last_batch_future.result()

        emitter.update({"time": 1.0, "rate": ureg.Quantity(10.0, "gram / liter")})
        emitter.update({"time": 2.0, "rate": ureg.Quantity(20.0, "gram / liter")})
        emitter.close(success=False)

        df = emitter.query().sort("time")
        # Column name preserved (NOT renamed to rate__magnitude); holds the
        # plain-float magnitudes.
        assert "rate" in df.columns, df.columns
        assert "rate__magnitude" not in df.columns, df.columns
        assert df["rate"].to_list() == [10.0, 20.0]
        # Unit captured in-memory and persisted to Parquet file metadata.
        assert emitter.column_units.get("rate") == "gram / liter"
        files = glob.glob(
            os.path.join(temp_dir, "quantity_port", "history", "**", "*.pq"),
            recursive=True,
        )
        assert files, "no history parquet written"
        kv = pq.read_metadata(files[0]).metadata or {}
        column_units = json.loads(kv[b"column_units"])
        assert column_units["rate"] == "gram / liter"

    def test_close_writes_partial_batch_and_success_sentinel(self, temp_dir, core):
        """close(success=True) flushes the trailing batch + writes the sentinel."""
        emitter = ParquetEmitter(
            config={
                "out_dir": temp_dir,
                "batch_size": 4,
                "threaded": False,
                "partitioning_keys": ["experiment_id"],
                "metadata": {"experiment_id": "sentinel_exp"},
            },
            core=core,
        )
        emitter.last_batch_future.result()

        emitter.update({"time": 0.0, "field1": 10, "field2": 20.5})
        emitter.close(success=True)

        # Partial batch landed as <num_emits>.pq
        history_path = os.path.join(
            emitter.out_uri, "sentinel_exp", "history",
            "experiment_id=sentinel_exp", "1.pq",
        )
        assert os.path.exists(history_path), f"missing partial flush: {history_path}"
        t = pl.read_parquet(history_path)
        assert len(t) == 1
        assert t["field1"].to_list() == [10]
        assert t["field2"].to_list() == [20.5]

        # Success sentinel under the partition path.
        sentinel = os.path.join(
            emitter.out_uri, "sentinel_exp", "success",
            "experiment_id=sentinel_exp", "s.pq",
        )
        assert os.path.exists(sentinel), f"missing sentinel: {sentinel}"

    def test_parquet_emitter_handles_structured_bulk_state(self, temp_dir, core):
        """End-to-end: a structured-array field is split into per-field columns."""
        emitter = ParquetEmitter(
            config={
                "out_dir": temp_dir,
                "batch_size": 4,
                "threaded": False,
                "metadata": {"experiment_id": "structured"},
            },
            core=core,
        )
        emitter.last_batch_future.result()

        bulk_dtype = [
            ("id", "i8"),
            ("count", "i8"),
            ("protein_submass", "f8"),  # bookkeeping — must be dropped
        ]
        emitter.update({
            "time": 1.0,
            "bulk": np.array([(1, 10, 0.5), (2, 20, 0.6)], dtype=bulk_dtype),
        })
        emitter.update({
            "time": 2.0,
            "bulk": np.array([(1, 11, 0.7), (2, 21, 0.8)], dtype=bulk_dtype),
        })
        emitter.close(success=False)

        df = emitter.query().sort("time")

        # Structured array exploded into per-field columns; bookkeeping dropped.
        assert "bulk__id" in df.columns
        assert "bulk__count" in df.columns
        assert "bulk" not in df.columns
        assert "bulk__protein_submass" not in df.columns

        assert df["time"].to_list() == [1.0, 2.0]
        # The exploded fields land as list[T] per row.
        assert df["bulk__id"].to_list() == [[1, 2], [1, 2]]
        assert df["bulk__count"].to_list() == [[10, 20], [11, 21]]

    def test_ragged_array_polars_fallback(self, temp_dir, core):
        """A field whose shape changes mid-batch falls back to the Polars list path."""
        emitter = ParquetEmitter(
            config={
                "out_dir": temp_dir,
                "batch_size": 3,
                "threaded": False,
                "metadata": {"experiment_id": "ragged"},
            },
            core=core,
        )
        emitter.last_batch_future.result()

        emitter.update({"time": 1.0, "ragged": np.array([1, 2, 3])})
        # Shape change forces the field onto the Polars path.
        emitter.update({"time": 2.0, "ragged": np.array([4, 5, 6, 7])})
        emitter.update({"time": 3.0, "ragged": np.array([8])})
        emitter.close()

        df = emitter.query().sort("time")
        assert df["ragged"].to_list() == [[1, 2, 3], [4, 5, 6, 7], [8]]
        assert "ragged" in emitter.pl_serialized
