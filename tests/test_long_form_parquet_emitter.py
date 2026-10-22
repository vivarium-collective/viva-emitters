"""Tests for viva_emitters.long_form_parquet_emitter.LongFormParquetEmitter.

Driven by direct ``emitter.update(state)`` calls (no Composite dependency),
mirroring the sqlite/parquet emitter test style. Covers:

* nested-timecourse form reproduces the BioModels corpus schema
  (biomodel_id, job, engine, variable, time, value) with downsampling;
* tick-stacked form for a time-stepped composite;
* RAMEmitter compatibility (query()/history still work → gather keeps working);
* write-on-close and write-on-query.
"""

import os
import tempfile

import pyarrow.parquet as pq
import pytest

from viva_emitters.long_form_parquet_emitter import (
    LongFormParquetEmitter,
    read_long_form,
)


def _make(config, core):
    return LongFormParquetEmitter({**config}, core)


def test_nested_timecourse_reproduces_corpus_schema(core):
    with tempfile.TemporaryDirectory() as d:
        em = _make({
            "file_path": d,
            "parquet_file": "series.parquet",
            "source_key": "results",
            "path_levels": ["biomodel_id", "job", "engine"],
            "time_key": "time",
            "max_points": 0,
        }, core)
        state = {
            "results": {
                "BIOMD1": {
                    "auto_ten_seconds": {
                        "copasi": {
                            "time": [0.0, 1.0, 2.0],
                            "A": [10.0, 9.0, 8.0],
                            "B": [0.0, 1.0, 2.0],
                        },
                        "tellurium": {
                            "time": [0.0, 1.0, 2.0],
                            "A": [10.0, 9.1, 8.2],
                        },
                    },
                },
            },
        }
        em.update(state)
        out = em.write()
        assert out and os.path.exists(out)

        t = read_long_form(out)
        assert t.column_names == ["biomodel_id", "job", "engine", "variable", "time", "value"]
        # A (3) + B (3) for copasi, A (3) for tellurium = 9 rows; 'time' is not a row
        assert t.num_rows == 9
        rows = t.to_pylist()
        assert all(r["biomodel_id"] == "BIOMD1" for r in rows)
        assert set(r["variable"] for r in rows) == {"A", "B"}
        assert "time" not in set(r["variable"] for r in rows)
        # a specific value survived round-trip (float32)
        cop_A = [r["value"] for r in rows
                 if r["engine"] == "copasi" and r["variable"] == "A"]
        assert cop_A == pytest.approx([10.0, 9.0, 8.0])


def test_nested_downsampling(core):
    with tempfile.TemporaryDirectory() as d:
        em = _make({
            "file_path": d,
            "source_key": "results",
            "path_levels": ["biomodel_id", "job", "engine"],
            "max_points": 10,
        }, core)
        n = 200
        state = {"results": {"M": {"j": {"e": {
            "time": list(range(n)),
            "X": [float(i) for i in range(n)],
        }}}}}
        em.update(state)
        t = read_long_form(em.write())
        # one variable X, downsampled to 10 points
        assert t.num_rows == 10


def test_tick_stacked_form(core):
    with tempfile.TemporaryDirectory() as d:
        em = _make({"file_path": d, "parquet_file": "ticks.parquet"}, core)
        for i in range(3):
            em.update({"global_time": float(i), "cell": {"mass": float(10 + i)}})
        t = read_long_form(em.write())
        assert set(t.column_names) == {"column", "step", "value"}
        rows = t.to_pylist()
        # 3 ticks * 1 numeric leaf (cell.mass); global_time is bookkeeping, dropped
        assert len(rows) == 3
        assert all(r["column"] == "cell.mass" for r in rows)
        assert [r["value"] for r in rows] == pytest.approx([10.0, 11.0, 12.0])


def test_query_still_returns_history_and_writes(core):
    with tempfile.TemporaryDirectory() as d:
        em = _make({
            "file_path": d,
            "source_key": "results",
            "path_levels": ["m", "j", "e"],
            "write_on_query": True,
        }, core)
        state = {"results": {"M": {"j": {"e": {"time": [0.0, 1.0], "X": [1.0, 2.0]}}}}}
        em.update(state)
        # RAMEmitter history preserved -> gather_emitter_results keeps working
        hist = em.query()
        assert isinstance(hist, list) and len(hist) == 1
        assert hist[0]["results"]["M"]["j"]["e"]["X"] == [1.0, 2.0]
        # write_on_query produced the file
        assert os.path.exists(em.out_path)


def test_close_writes_on_success_only(core):
    with tempfile.TemporaryDirectory() as d:
        em = _make({
            "file_path": d, "parquet_file": "c.parquet",
            "source_key": "results", "path_levels": ["m", "j", "e"],
            "write_on_query": False,
        }, core)
        em.update({"results": {"M": {"j": {"e": {"time": [0.0], "X": [5.0]}}}}})
        assert not os.path.exists(em.out_path)  # nothing written yet
        em.close(success=True)
        assert os.path.exists(em.out_path)


def test_contract_is_parquet():
    c = LongFormParquetEmitter.emitter_contract()
    assert c.output_kind == "parquet"
    assert c.output_uri_config_key == "parquet_file"
