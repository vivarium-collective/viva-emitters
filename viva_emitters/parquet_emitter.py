"""Parquet-backed emitter for process-bigraph composites.

Writes each composite tick as a row in a hive-partitioned Parquet dataset,
buffering rows into batches before flushing them to disk through a
background thread. Reader helpers (:func:`create_duckdb_conn`,
:func:`dataset_sql`, :func:`read_stacked_columns`, etc.) build DuckDB
queries over the resulting dataset so downstream analysis code can pull
columns out of many cells / many runs without dragging in the original
Composite.

Requires the ``[parquet]`` extra (``duckdb``, ``polars``, ``pyarrow``,
``fsspec``, ``tqdm``).
"""

from __future__ import annotations

import fnmatch
import json
import os
import warnings
from concurrent.futures import Future, ThreadPoolExecutor, Executor
from typing import Any, Callable, Mapping, Optional, cast
from urllib import parse

try:
    import duckdb
    import fsspec
    import numpy as np
    import polars as pl
    import tqdm
    from fsspec import open as fsspec_open
    from fsspec.core import OpenFile, filesystem, url_to_fs
    from fsspec.spec import AbstractFileSystem
    from polars.datatypes import DataTypeClass
except ImportError as e:
    raise ImportError(
        f"viva_emitters.parquet_emitter requires the [parquet] extra. "
        f"Install with: pip install 'viva-emitters[parquet]'. (missing: {e.name})"
    ) from e

from process_bigraph.emitter import Emitter


# ==============================================================================


class BlockingExecutor(Executor):

    def __init__(self, *args) -> None:
        assert not len(args)
        super().__init__()

    def submit(self, fn: Callable, /, *args, **kwargs) -> Future:
        """
        Run a function in the current thread, and return a
        :py:class:`~concurrent.futures.Future` that is already done.
        """
        future: Future = Future()
        try:
            result = fn(*args, **kwargs)
            future.set_result(result)
        except Exception as e:
            future.set_exception(e)
        return future

    def shutdown(self, wait=True, *, cancel_futures=False) -> None:
        pass


# ==============================================================================


METADATA_PREFIX = "output_metadata__"
"""
In the configuration parquet, user-defined per-store metadata is written into
columns with this prefix so it can be queried alongside the rest of the config.
"""



# ==============================================================================


def json_to_parquet(
    emit_dict: dict[str, np.ndarray | list[pl.Series]],
    outfile: str,
    schema: dict[str, Any],
    filesystem: AbstractFileSystem,
    metadata: dict[str, str] | None = None,
):
    """Convert dictionary to Parquet.

    Args:
        emit_dict: Mapping from column names to NumPy arrays (fixed-shape)
            or lists of Polars Series (variable-shape).
        outfile: Path to output Parquet file. Can be local path or URI.
        schema: Full mapping of column names to Polars dtypes.
        filesystem: On local filesystem, fsspec filesystem needed to
            write Parquet file atomically.
        metadata: Optional file-level Parquet key/value metadata (e.g. the
            unit string for each unit-bearing column).
    """
    col_schema = {k: schema[k] for k in emit_dict}
    try:
        tbl = pl.DataFrame(emit_dict, schema=col_schema)
    except (TypeError, ValueError, pl.exceptions.PolarsError) as exc:
        # Some column's buffered rows can't be built into one Series — almost
        # always a field whose SHAPE varies across ticks (scalar on some ticks,
        # a vector on others), which is unstorable in a single rectangular
        # Parquet column. Crashing here loses the ENTIRE batch, so every
        # downstream viz/analysis then sees "no run data". Build column-by-column
        # instead and DROP only the genuinely unstorable column(s), preserving
        # every well-behaved observable so the batch is written and the viz has
        # data. (Unit metadata is file-level, so it is unaffected.)
        cols: list[pl.Series] = []
        dropped: list[str] = []
        for k, v in emit_dict.items():
            try:
                cols.append(pl.Series(k, v))
            except (TypeError, ValueError, pl.exceptions.PolarsError):
                dropped.append(k)
        tbl = pl.DataFrame(cols)
        print(
            f"parquet: dropped {len(dropped)} column(s) with an unstorable "
            f"time-varying shape so the batch could be written: {dropped[:8]}"
            + (" …" if len(dropped) > 8 else "")
            + f" [{type(exc).__name__}: {exc}]",
            flush=True,
        )
    # GCS should have atomic uploads, but on a local filesystem, DuckDB may fail
    # trying to read partially written Parquet files. Get around this by writing
    # to a temporary file and then renaming it to the final output file.
    temp_outfile = outfile
    if parse.urlparse(outfile).scheme in ("", "file", "local"):
        temp_outfile = outfile + ".tmp"
    tbl.write_parquet(
        temp_outfile,
        # File-level key/value metadata (column units, etc.).
        metadata=metadata or None,
        # Increase retry attempts to handle S3/GCS failures
        storage_options={"max_retries": 50, "retry_timeout_ms": 300000},
    )
    if temp_outfile != outfile:
        filesystem.mv(temp_outfile, outfile)


def union_by_name(query_sql: str) -> str:
    """
    Modifies SQL query string from :py:func:`~.dataset_sql` to
    include ``union_by_name = true`` in the DuckDB ``read_parquet``
    function. This allows data to be read from simulations that have
    different columns by filling in nulls and casting as necessary.
    This comes with a performance penalty and should be avoided if possible.

    Args:
        query_sql: SQL query string from :py:func:`~.dataset_sql`
    """
    return query_sql.replace(
        "hive_partitioning = true,", "hive_partitioning = true, union_by_name = true,"
    )


def create_duckdb_conn(
    temp_dir: str = "/tmp", object_store: str = "", cpus: Optional[int] = None
) -> duckdb.DuckDBPyConnection:
    """
    Create a DuckDB connection.

    Args:
        temp_dir: Temporary directory for spilling to disk.
        object_store: URI scheme of object store to register with DuckDB (e.g. "gcs" or "s3").
        cpus: Number of cores to use (by default, use all detected cores).
    """
    conn = duckdb.connect()
    if object_store == "s3":
        # Use DuckDB's built-in HTTPFS extension for S3
        # credential_chain uses AWS credential chain (env vars, ~/.aws/credentials,
        # EC2 instance role, etc.) - works automatically on EC2 with IAM roles
        conn.execute("INSTALL httpfs; LOAD httpfs;")
        conn.execute("""
            CREATE OR REPLACE SECRET secret (
                TYPE s3,
                PROVIDER credential_chain
            );
        """)
    elif object_store:
        # For GCS and other object stores, use fsspec
        conn.register_filesystem(filesystem(object_store))
    # Temp directory so DuckDB can spill to disk when data larger than RAM
    conn.execute(f"SET temp_directory = '{temp_dir}'")
    # Turning this off reduces RAM usage
    conn.execute("SET preserve_insertion_order = false")
    # Do not cache Parquet metadata to reduce RAM usage
    conn.execute("SET parquet_metadata_cache = false")
    # Turn off Parquet file caching to reduce RAM usage
    conn.execute("SET enable_external_file_cache = false")
    # Set number of threads for DuckDB
    if cpus is not None:
        conn.execute(f"SET threads = {cpus}")
    return conn


def dataset_sql(out_dir: str, experiment_ids: list[str]) -> tuple[str, str, str]:
    """
    Creates DuckDB SQL strings for sim outputs, configs, and metadata on which
    sims were successful.

    Args:
        out_dir: Path to output directory for workflows to retrieve data
            for (relative or absolute local path OR URI beginning with
            ``gcs://`` or ``gs://`` for Google Cloud Storage bucket)
        experiment_ids: List of experiment IDs to include in query. To read data
            from more than one experiment ID, the listeners in the output of the
            first experiment ID in the list must be a strict subset of the listeners
            in the output of the subsequent experiment ID(s).

    Returns:
        3-element tuple containing

        - **history_sql**: SQL query for sim output (see :py:func:`~.read_stacked_columns`),
        - **config_sql**: SQL query for sim configs (see :py:func:`~.field_metadata`
          and :py:func:`~.config_value`)
        - **success_sql**: SQL query for metadata marking successful sims
          (see :py:func:`~.read_stacked_columns`)

    """
    sql_queries = []
    for query_type in ("history", "configuration", "success"):
        query_files = []
        for experiment_id in experiment_ids:
            query_files.append(
                f"'{os.path.join(out_dir, experiment_id)}/{query_type}/*/*/*/*/*/*.pq'"
            )
        query_files = ", ".join(query_files)
        sql_queries.append(
            f"""
            FROM read_parquet(
                [{query_files}],
                hive_partitioning = true,
                hive_types = {{
                    'experiment_id': VARCHAR,
                    'variant': BIGINT,
                    'lineage_seed': BIGINT,
                    'generation': BIGINT,
                    'agent_id': VARCHAR,
                }}
            )
            """
        )
    return sql_queries[0], sql_queries[1], sql_queries[2]


def list_columns(
    conn: duckdb.DuckDBPyConnection, history_subquery: str, pattern: str | None = None
) -> list[str]:
    """
    Return list of columns in DuckDB subquery containing sim output data.

    Args:
        conn: DuckDB connection
        history_subquery: DuckDB query containing sim output data
        pattern: Optional glob pattern to filter column names
    """
    columns = (
        conn.sql(f"SELECT column_name FROM (DESCRIBE ({history_subquery}))")
        .pl()["column_name"]
        .to_list()
    )
    if pattern is not None:
        columns = fnmatch.filter(columns, pattern)
    return columns


def quote_columns(columns: str | list[str]) -> str | list[str]:
    """
    Given one or more raw column names (not DuckDB expressions),
    return the same column name(s) enclosed in
    double quotes to handle special characters (spaces, dashes, etc.).

    Args:
        columns: One or more column names
    """
    if isinstance(columns, str):
        # Escape existing double quotes by doubling them
        escaped = columns.replace('"', '""')
        return f'"{escaped}"'
    return [cast(str, quote_columns(col)) for col in columns]


def num_cells(conn: duckdb.DuckDBPyConnection, subquery: str) -> int:
    """
    Return cell count in DuckDB subquery containing ``experiment_id``,
    ``variant``, ``lineage_seed``, ``generation``, and ``agent_id`` columns.
    """
    return cast(
        tuple,
        conn.sql(f"""SELECT count(
        DISTINCT (experiment_id, variant, lineage_seed, generation, agent_id)
        ) FROM ({subquery})""").fetchone(),
    )[0]


def skip_n_gens(subquery: str, n: int) -> str:
    """
    Modifies a DuckDB SQL query to skip the first ``n`` generations of data.
    """
    return f"SELECT * FROM ({subquery}) WHERE generation > {n}"


def ndlist_to_ndarray(s) -> np.ndarray:
    """
    Convert a PyArrow series of nested lists with fixed dimensions into
    a Numpy ndarray. This should really only be necessary if you are trying
    to perform linear algebra (e.g. matrix multiplication, dot products) inside
    a user-defined function (see DuckDB documentation on Python Function API and
    ``func`` kwarg for :py:func:`~read_stacked_columns`).

    For elementwise arithmetic of two nested list columns, this can be used
    to define a custom DuckDB function as follows::

        import duckdb
        import polars as pl
        from viva_emitters import ndlist_to_ndarray
        def sum_arrays(col_0, col_1):
            return pl.Series(
                ndlist_to_ndarray(col_0) +
                ndlist_to_ndarray(col_1)
            ).to_arrow()
        conn = duckdb.connect()
        conn.create_function(
            "sum_2d_int_arrays", # Function name for use in SQL (must be unique)
            sum_arrays, # Python function that takes and returns PyArrow arrays
            [list[list[int]], list[list[int]]], # Input types (2D lists here)
            list[list[int]], # Return type (2D list here)
            type = "arrow" # Tell DuckDB function operates on Arrow arrays
        )
        conn.sql("SELECT sum_2d_int_arrays(int_col_0, int_col_1) from input_table")
        # Note that function must be registered under different name for each
        # set of unique input/output types
        conn.create_function(
            "sum_2d_int_and_float",
            sum_arrays,
            [list[list[int]], list[list[float]]], # Second input is 2D float array
            list[list[float]], # Adding int to float array gives float in Numpy
            type = "arrow"
        )
        conn.sql("SELECT sum_2d_int_and_float(int_col_0, float_col_0) from input_table")

    """
    inner_s = pl.Series(s)
    dimensions = []
    while inner_s.dtype.is_nested() and len(inner_s) > 0:
        inner_s = inner_s[0]
        dimensions.append(len(inner_s))
    inner_s = inner_s.dtype
    while inner_s.is_nested():
        inner_s = inner_s.inner  # type: ignore[attr-defined]
        dimensions.append(0)
    return pl.Series(s, dtype=pl.Array(inner_s, tuple(dimensions))).to_numpy()


def ndidx_to_duckdb_expr(
    name: str, idx: list[int | list[int] | list[bool] | str]
) -> str:
    """
    Returns a DuckDB expression for a column equivalent to converting each row
    of ``name`` into an ndarray ``name_arr`` (:py:func:`~.ndlist_to_ndarray`)
    and getting ``name_arr[idx]``. ``idx`` can contain 1D lists of integers,
    boolean masks, or ``":"`` (no 2D+ indices like ``x[[[1,2]]]``). See also
    :py:func:`~named_idx` if pulling out a relatively small set of indices.
    Automatically quotes column names to handle special characters. Do NOT
    use double quotes in ``name``.

    .. WARNING:: DuckDB arrays are 1-indexed so this function adds 1 to every
        supplied integer index!

    Args:
        name: Name of column to recursively index
        idx: To get all elements for a dimension, supply the string ``":"``.
            Otherwise, only single integers or 1D integer lists of indices are
            allowed for each dimension. Some examples::

                [0, 1] # First row, second column
                [[0, 1], 1] # First and second row, second column
                [0, 1, ":"] # First element of axis 1, second of 2, all of 3
                # Final example differs between this function and Numpy
                # This func: 1st and 2nd of axis 1, all of 2, 1st and 2nd of 3
                # Numpy: Complicated, see Numpy docs on advanced indexing
                [[0, 1], ":", [0, 1]]

    """
    quoted_name = f'"{name}"'
    idx = idx.copy()
    idx.reverse()
    # Construct expression from inside out (deepest to shallowest axis)
    first_idx = idx.pop(0)
    if isinstance(first_idx, list):
        # Python bools are instances of int so check bool first
        if isinstance(first_idx[0], bool):
            select_expr = f"list_where(x_0, {first_idx})"
        elif isinstance(first_idx[0], int):
            one_indexed_idx = ", ".join(str(i + 1) for i in first_idx)
            select_expr = f"list_select(x_0, [{one_indexed_idx}])"
        else:
            raise TypeError("Indices must be integers or boolean masks.")
    elif first_idx == ":":
        select_expr = "x_0"
    elif isinstance(first_idx, int):
        select_expr = f"list_select(x_0, [{int(first_idx) + 1}])"
    else:
        raise TypeError("All indices must be lists, ints, or ':'.")
    i = -1
    for i, indices in enumerate(idx):
        if isinstance(indices, list):
            if isinstance(indices[0], bool):
                select_expr = f"list_transform(list_where(x_{i + 1}, {indices}), lambda x_{i} : {select_expr})"
            elif isinstance(indices[0], int):
                one_indexed_idx = ", ".join(str(i + 1) for i in indices)
                select_expr = f"list_transform(list_select(x_{i + 1}, [{one_indexed_idx}]), lambda x_{i} : {select_expr})"
            else:
                raise TypeError("Indices must be integers or boolean masks.")
        elif indices == ":":
            select_expr = f"list_transform(x_{i + 1}, lambda x_{i} : {select_expr})"
        elif isinstance(indices, int):
            select_expr = f"list_transform(list_select(x_{i + 1}, [{int(indices) + 1}]), lambda x_{i} : {select_expr})"
        else:
            raise TypeError("All indices must be lists, ints, or ':'.")
    select_expr = select_expr.replace(f"x_{i + 1}", quoted_name)
    return select_expr + f" AS {quoted_name}"


def named_idx(
    col: str,
    names: list[str],
    idx: list[list[int]],
    zero_to_null: bool = False,
    _quote_col: bool = True,
) -> str:
    """
    Create DuckDB expressions for given indices from a list column. Can be
    used in ``columns`` kwarg of :py:func:`~.read_stacked_columns`. Since
    each index gets pulled out into its own column, this greatly simplifies
    aggregations like averages, etc. Only use this if the number of indices
    is relatively small (<100) and the list column is 1-dimensional. For 2+
    dimensions or >100 indices, see :py:func:`~.ndidx_to_duckdb_expr`.
    Automatically quotes column names to handle special characters.
    Do NOT use double quotes in ``names`` or ``col``.

    .. WARNING:: DuckDB arrays are 1-indexed so this function adds 1 to every
        supplied index!

    Args:
        col: Name of list column.
        names: List of names for the new columns. Length must match the
            number of index combinations in ``idx`` (4 for example below).
        idx: Integer indices to retrieve from each dimension of ``col``.
            For example, ``[[0, 1], [2, 3]]`` will retrieve the third and
            fourth elements of the second dimension for the first and second
            elements of the first dimension.
        zero_to_null: Whether to turn 0s into nulls. This is useful when
            dividing by the values in this column, as most DuckDB aggregation
            functions (e.g. ``avg``, ``max``) propagate NaNs but ignore nulls.
        _quote_col: Private argument to ensure ``col`` is quoted properly.

    Returns:
        DuckDB SQL expression for a set of named columns corresponding to
        the values at given indices of a list column
    """
    assert isinstance(idx[0], list), "idx must be a list of lists."
    # Quote column name on initial call
    if _quote_col:
        col = f'"{col}"'
    col_exprs = []
    if len(idx) == 1:
        for num, i in enumerate(idx[0]):
            quoted_name = f'"{names[num]}"'
            if zero_to_null:
                col_exprs.append(
                    f"CASE WHEN {col}[{i + 1}] = 0 THEN NULL ELSE {col}[{i + 1}] END AS {quoted_name}"
                )
            else:
                col_exprs.append(f"{col}[{i + 1}] AS {quoted_name}")
    else:
        col_counter = 0
        for i in idx[0]:
            sub_col_exprs = named_idx(
                f"{col}[{i + 1}]",
                names[col_counter:],
                idx[1:],
                zero_to_null,
                _quote_col=False,
            )
            col_counter += sub_col_exprs.count(", ") + 1
            col_exprs.append(sub_col_exprs)
    return ", ".join(col_exprs)


def field_metadata(
    conn: duckdb.DuckDBPyConnection, config_subquery: str, field: str
) -> list:
    """
    Gets the saved per-field metadata for a given field as a list.
    Automatically quotes the field name to handle special characters.
    Do NOT use double quotes in ``field``.

    Args:
        conn: DuckDB connection
        config_subquery: DuckDB query containing sim config data
        field: Name of field to get metadata for
    """
    metadata = cast(
        tuple,
        conn.sql(
            f'SELECT first("{METADATA_PREFIX + field}") FROM ({config_subquery})'
        ).fetchone(),
    )[0]
    if isinstance(metadata, list):
        return metadata
    return list(metadata)


def config_value(
    conn: duckdb.DuckDBPyConnection, config_subquery: str, field: str
) -> Any:
    """
    Gets the saved configuration option (anything in config JSON, with
    double underscore concatenation for nested fields due to
    :py:func:`~.flatten_dict`). Automatically quotes the field name to
    handle special characters. Do NOT use double quotes in ``field``.

    Args:
        conn: DuckDB connection
        config_subquery: DuckDB query containing sim config data
        field: Name of configuration option to get value of
    """
    return cast(
        tuple,
        conn.sql(f'SELECT first("{field}") FROM ({config_subquery})').fetchone(),
    )[0]


def plot_metadata(
    conn: duckdb.DuckDBPyConnection, config_subquery: str, variant_name: str
) -> dict[str, Any]:
    """
    Gets a dictionary suitable for use as ``metadata`` kwarg to a generic
    figure-export helper (e.g. ``export_figure(metadata=...)``).

    Args:
        conn: DuckDB connection
        config_subquery: DuckDB query containing sim config data
        variant_name: Name of variant
    """
    return {
        "git_hash": config_value(conn, config_subquery, "git_hash"),
        "time": config_value(conn, config_subquery, "time"),
        "description": config_value(conn, config_subquery, "description"),
        "variant_function": variant_name,
        "variant_index": conn.sql(f"SELECT DISTINCT variant FROM ({config_subquery})")
        .arrow()
        .to_pydict()["variant"],
        "seed": conn.sql(f"SELECT DISTINCT lineage_seed FROM ({config_subquery})")
        .arrow()
        .to_pydict()["lineage_seed"],
        "total_gens": cast(
            tuple,
            conn.sql(
                f"SELECT count(DISTINCT generation) FROM ({config_subquery})"
            ).fetchone(),
        )[0],
        "total_variants": cast(
            tuple,
            conn.sql(
                f"SELECT count(DISTINCT variant) FROM ({config_subquery})"
            ).fetchone(),
        )[0],
    }


def open_arbitrary_sim_data(sim_data_dict: dict[str, dict[int, Any]]) -> OpenFile:
    """
    Given a mapping from experiment ID(s) to mappings from variant ID(s)
    to sim_data path(s), pick an arbitrary sim_data to read.

    Args:
        sim_data_dict: Mapping passed in by a workflow runner / analysis driver.

    Returns:
        File object for arbitrarily chosen sim_data to be loaded
        with ``pickle.load``
    """
    sim_data_path = next(iter(next(iter(sim_data_dict.values())).values()))
    return fsspec_open(sim_data_path, "rb")


def read_stacked_columns(
    history_sql: str,
    columns: list[str],
    remove_first: bool = False,
    func: Optional[Callable[[pl.DataFrame], pl.DataFrame]] = None,
    conn: Optional[duckdb.DuckDBPyConnection] = None,
    order_results: bool = True,
    success_sql: Optional[str] = None,
) -> pl.DataFrame | str:
    """
    Loads columns for many cells. If you would like to perform more advanced
    computatations (aggregations, window functions, etc.) using the optimized
    DuckDB API, you can omit ``conn``, in which case this function will return
    an SQL string that can be used as a subquery. For computations that cannot
    be easily performed using the DuckDB API, you can define a custom function
    ``func`` that will be called on the data for each cell. By default, the
    return value (whether it be the actual data or an SQL subquery) will
    also include the ``experiment_id``, ``variant``, ``lineage_seed``,
    ``generation``, ``agent_id``, and ``time`` columns.

    .. hint:: To get a full list of columns in the output data that you can
        use in your ``columns`` SQL expressions, use :py:func:`~.list_columns`.

    .. warning:: If your raw column names contain special characters and you
        are not constructing your column expressions with
        :py:func:`~named_idx` or :py:func:`~ndidx_to_duckdb_expr`,
        the raw column names MUST be enclosed in double quotes
        to handle special characters (e.g. ``'"space and-hyphens"'``,
        ``"\"[brackets]\""``). Use :py:func:`~quote_columns` to quote
        these columns before constructing SQL expressions with them.

    For example, to get the average total concentration of three list elements
    with indices 100, 1000, and 10000 per cell::

        import duckdb
        from viva_emitters import dataset_sql, read_stacked_columns
        history_sql, config_sql, _ = dataset_sql('out/', 'exp_id')
        subquery = read_stacked_columns(
            history_sql,
            # Note DuckDB arrays are 1-indexed
            ["bulk[100 + 1] + bulk[1000 + 1] + bulk[10000 + 1] AS bulk_sum",
            "listeners__enzyme_kinetics__counts_to_molar AS counts_to_molar"],
            order_results=False,
        )
        query = '''
            SELECT avg(bulk_sum * counts_to_molar) AS avg_total_conc
            FROM ({subquery})
            GROUP BY experiment_id, variant, lineage_seed, generation, agent_id
            '''
        conn = duckdb.connect()
        data = conn.sql(query).pl()

    Args:
        history_sql: DuckDB SQL string from :py:func:`~.dataset_sql`,
            potentially with filters appended in ``WHERE`` clause
        columns: Names of columns to read data for. Alternatively, DuckDB
            expressions of columns (e.g. ``avg(listeners__mass__cell_mass) AS avg_mass``
            or the output of :py:func:`~.named_idx` or :py:func:`~.ndidx_to_duckdb_expr`).
        remove_first: Remove data for first timestep of each cell
        func: Function to call on data for each cell, should take and
            return a Polars DataFrame with columns equal to ``columns``
        conn: DuckDB connection instance with which to run query. Tweaked
            settings can be applied by callers via :py:func:`~.create_duckdb_conn`.
            Can be omitted to return SQL query string to be used as subquery
            instead of running query immediately and returning result.
        order_results: Whether to sort returned table by ``experiment_id``,
            ``variant``, ``lineage_seed``, ``generation``, ``agent_id``, and
            ``time``. If no ``conn`` is provided, this can usually be disabled
            and any sorting can be deferred until the last step in the query with
            a manual ``ORDER BY``. Doing this can greatly reduce RAM usage.
        success_sql: Final DuckDB SQL string from :py:func:`~.dataset_sql`.
            If provided, will be used to filter out unsuccessful sims.
    """
    id_cols = "experiment_id, variant, lineage_seed, generation, agent_id, time"
    columns_str = ", ".join(columns)
    sql_query = f"SELECT {columns_str}, {id_cols} FROM ({history_sql})"
    # Use a semi join to filter out unsuccessful sims
    if success_sql is not None:
        sql_query = f"""
            SELECT * FROM ({sql_query})
            SEMI JOIN ({success_sql})
            USING (experiment_id, variant, lineage_seed, generation, agent_id)
            """
    # Use an anti join to remove rows for first timestep of each sim
    if remove_first:
        sql_query = f"""
            SELECT * FROM ({sql_query})
            ANTI JOIN (
                SELECT experiment_id, variant, lineage_seed, generation,
                    agent_id, MIN(time) AS time
                FROM ({history_sql.replace("COLNAMEHERE", "time")})
                GROUP BY experiment_id, variant, lineage_seed, generation,
                    agent_id
            ) USING (experiment_id, variant, lineage_seed, generation,
                agent_id, time)
            """
    if func is not None:
        if conn is None:
            raise RuntimeError("`conn` must be provided with `func`.")
        # Get all cell identifiers
        cell_ids = conn.sql(f"""SELECT DISTINCT ON(experiment_id, variant,
            lineage_seed, generation, agent_id) experiment_id, variant,
            lineage_seed, generation, agent_id FROM ({history_sql}) ORDER BY {id_cols}
        """).fetchall()
        all_cell_tbls = []
        for experiment_id, variant, lineage_seed, generation, agent_id in tqdm.tqdm(
            cell_ids
        ):
            # Explicitly specify Hive partition because DuckDB
            # will otherwise spend a lot of time scanning all files
            cell_sql = sql_query.replace(
                "history/*/*/*/*/*",
                f"history/experiment_id={experiment_id}/variant={variant}/lineage_seed={lineage_seed}/generation={generation}/agent_id={agent_id}",
            )
            # Apply func to data for each cell
            all_cell_tbls.append(func(conn.sql(cell_sql).pl()))
        return pl.concat(all_cell_tbls)
    if order_results:
        query = f"SELECT * FROM ({sql_query}) ORDER BY {id_cols}"
    else:
        query = sql_query
    if conn is None:
        return query
    return conn.sql(query).pl()


_POLARS_DTYPE_BY_NAME: dict[str, Any] = {
    "Int8":   pl.Int8,
    "Int16":  pl.Int16,
    "Int32":  pl.Int32,
    "Int64":  pl.Int64,
    "UInt8":  pl.UInt8,
    "UInt16": pl.UInt16,
    "UInt32": pl.UInt32,
    "UInt64": pl.UInt64,
    "Float32": pl.Float32,
    "Float64": pl.Float64,
    "Boolean": pl.Boolean,
    "Utf8":    pl.Utf8,
}

_NUMPY_DTYPE_BY_POLARS_NAME: dict[str, Any] = {
    "Int8": np.int8, "Int16": np.int16, "Int32": np.int32, "Int64": np.int64,
    "UInt8": np.uint8, "UInt16": np.uint16, "UInt32": np.uint32, "UInt64": np.uint64,
    "Float32": np.float32, "Float64": np.float64,
    "Boolean": np.bool_,
}


def _lookup_dtype_override(field_name: str, overrides: dict[str, str]) -> Optional[str]:
    """Return the override dtype name for ``field_name``, or None.

    Exact-name matches win over fnmatch glob matches. Globs are checked in
    insertion order; first match wins.
    """
    if field_name in overrides:
        return overrides[field_name]
    for pattern, dtype_name in overrides.items():
        if any(c in pattern for c in "*?[") and fnmatch.fnmatchcase(field_name, pattern):
            return dtype_name
    return None


def _polars_dtype_from_override(field_name: str, overrides: dict[str, str]) -> Optional[Any]:
    """Resolve ``field_name``'s override to a polars dtype instance, or None."""
    if not overrides:
        return None
    hit = _lookup_dtype_override(field_name, overrides)
    if hit is None:
        return None
    pl_cls = _POLARS_DTYPE_BY_NAME.get(hit)
    return pl_cls() if pl_cls is not None else None


def np_dtype(val: Any, field_name: str, overrides: Optional[dict[str, str]] = None) -> Any:
    """Choose a NumPy dtype for ``val`` for the column named ``field_name``.

    Resolution order: (1) ``overrides`` (exact name, then fnmatch glob), (2)
    deep dispatch by value type. The deep dispatch raises ``ValueError`` on
    None / empty list / unsupported types so that ``ParquetEmitter.update``
    can fall back to Polars serialization. Bytes and datetime values also
    raise — Polars handles them more robustly than NumPy.
    """
    if overrides:
        hit = _lookup_dtype_override(field_name, overrides)
        if hit is not None:
            mapped = _NUMPY_DTYPE_BY_POLARS_NAME.get(hit)
            if mapped is not None:
                return mapped
    # Order matters: bool is a subclass of int in Python.
    if isinstance(val, float):
        return np.float64
    if isinstance(val, bool):
        return np.bool_
    if isinstance(val, int):
        return np.int64
    if isinstance(val, (str, np.str_)):
        return np.dtypes.StringDType
    if isinstance(val, np.generic):
        return val.dtype
    if isinstance(val, np.ndarray):
        return val.dtype
    if isinstance(val, (list, tuple)):
        if len(val) > 0:
            for inner_val in val:
                if inner_val is not None:
                    return np_dtype(inner_val, field_name, overrides)
    raise ValueError(f"{field_name} has unsupported type {type(val)}.")


def union_pl_dtypes(
    dt1: pl.DataType | DataTypeClass,
    dt2: pl.DataType | DataTypeClass,
    k: str,
    force_inner: Optional[pl.DataType | DataTypeClass] = None,
) -> pl.DataType | DataTypeClass:
    """
    Returns the more specific data type when combining two Polars datatypes.
    Mainly intended to fill out nested List types that contain Nulls.

    Args:
        dt1: First Polars datatype
        dt2: Second Polars datatype
        k: Name of column being combined (for error messages)
        force_inner: Force this inner type when possible

    Returns:
        The resulting datatype
    """
    if isinstance(dt1, (pl.List, pl.Array)) and isinstance(dt2, (pl.List, pl.Array)):
        # Recursively find the common type for inner elements
        inner_type = union_pl_dtypes(dt1.inner, dt2.inner, k, force_inner)
        return pl.List(inner_type)

    if dt1 == pl.Null:
        # To force a specific inner type, may need to recurse
        if force_inner is not None:
            if isinstance(dt2, (pl.List, pl.Array)):
                return pl.List(union_pl_dtypes(dt2.inner, dt2.inner, k, force_inner))
            return force_inner
        return dt2
    if dt2 == pl.Null:
        if force_inner is not None:
            if isinstance(dt1, (pl.List, pl.Array)):
                return pl.List(union_pl_dtypes(dt1.inner, dt1.inner, k, force_inner))
            return force_inner
        return dt1

    if force_inner is not None:
        return force_inner

    if dt1 == dt2:
        return dt1

    raise TypeError(
        f"Incompatible inner types for field {k}: {dt1} and {dt2}."
        " Please modify the store value to contain a consistent type."
    )


def flatten_dict(d: dict, separator: str = "__", prefix: str = "") -> dict:
    """Flatten a nested dict into a single-level dict keyed by joined paths.

    Default separator is ``"__"`` to match the parquet emitter's column-name
    convention.
    """
    out: dict = {}
    for k, v in d.items():
        key = f"{prefix}{separator}{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(flatten_dict(v, separator=separator, prefix=key))
        else:
            out[key] = v
    return out


def _is_bookkeeping_field(name: str) -> bool:
    """Predicate for structured-array sub-fields that are not analysis-interesting.

    When a numpy structured (record) array is split into per-field columns
    by :func:`_split_structured_arrays`, certain sub-fields are dropped to
    keep the per-row column count manageable. These tend to be allocator-
    or accounting-related fields that explode in width when projected as
    list columns but are only useful in-process.

    The dropped sub-field name patterns are common conventions in
    cell-simulation contexts (e.g. mass-balance bookkeeping on bulk
    molecule counts and per-instance metadata arrays):

    - ``*_submass`` — per-row mass-delta columns commonly attached to
      bulk-molecule structured arrays
    - ``massDiff_*`` — per-instance mass-delta columns commonly attached
      to per-molecule metadata arrays
    - ``_entryState`` — internal allocator tombstone / liveness flag

    Callers that want every field preserved should not call
    :func:`_split_structured_arrays`.
    """
    return (
        name.endswith("_submass")
        or name.startswith("massDiff_")
        or name == "_entryState"
    )


def _split_structured_arrays(flat: dict) -> dict:
    """Project numpy structured-record arrays into separate per-field columns.

    Polars cannot write a numpy structured (record) array as a single
    column, but it handles ``list[T]`` per-row just fine — so each
    non-bookkeeping field of the structured array becomes its own
    ``<key>__<field>`` entry. Bookkeeping fields (see
    :func:`_is_bookkeeping_field`) are dropped.

    Example: a structured array with dtype ``[('id', 'i8'), ('count', 'i8')]``
    stored under key ``'bulk'`` is projected to
    ``{'bulk__id': id_array, 'bulk__count': count_array}``.

    Non-structured values (scalars, plain ndarrays, lists, etc.) pass
    through unchanged.
    """
    out: dict = {}
    for k, v in flat.items():
        if isinstance(v, np.ndarray) and v.dtype.fields is not None:
            for f in (v.dtype.names or ()):
                if _is_bookkeeping_field(f):
                    continue
                out[f"{k}__{f}"] = v[f]
            continue
        out[k] = v
    return out


def pl_dtype_from_ndarray(arr: np.ndarray) -> pl.DataType:
    """
    Get Polars data type for a Numpy array, including nested lists.
    """
    # Must be size 1 in order for np.dtypes.StringDType to
    # convert to Polars String type
    pl_dtype = pl.Series(np.empty(1, dtype=arr.dtype)).dtype
    for _ in range(arr.ndim):
        pl_dtype = pl.List(pl_dtype)
    return pl_dtype


def _is_quantity(value: Any) -> bool:
    """Duck-type a ``pint.Quantity`` without importing pint.

    A unit-bearing value exposes ``magnitude``, ``units`` and
    ``dimensionality``; plain floats / numpy arrays / lists do not.
    """
    return (
        hasattr(value, "magnitude")
        and hasattr(value, "units")
        and hasattr(value, "dimensionality")
    )


def strip_quantities(flat: dict[str, Any], column_units: dict[str, str]) -> dict[str, Any]:
    """Replace unit-bearing leaves with their magnitude, recording the unit.

    The numpy/Polars buffering paths can only store native scalars/arrays.
    A ``pint.Quantity`` wired to a ``quantity[...]`` port is numpy ``object``
    dtype and makes ``pl.Series([v])`` raise ``cannot parse numpy data type
    dtype('O')``.

    Operating on the already-flattened dict, each ``Quantity`` leaf is
    replaced by its ``.magnitude`` **under the same column name** (so
    downstream queries / reports / visualizations that read the column by
    name keep working unchanged), and its unit string is recorded in
    ``column_units`` to be written as Parquet file metadata. This is the key
    difference from emitting ``{units, magnitude}`` and letting
    ``flatten_dict`` split it into ``…__magnitude`` / ``…__units__<sym>``
    columns — that renames the column and breaks name-based consumers.

    The emit schema is anyized to ``node`` (``process_bigraph.emitter
    .anyize_paths``), so the value's own type drives this — not the declared
    port schema. Native values are left untouched (hot path unaffected).
    """
    for k, v in list(flat.items()):
        if _is_quantity(v):
            column_units.setdefault(k, str(v.units))
            flat[k] = v.magnitude
    return flat


class ParquetEmitter(Emitter):
    """Generic Parquet emitter, hive-partitioned per ``partitioning_keys``.

    The process-bigraph lifecycle drives the emit cadence:

      - configuration is written at ``__init__`` time from ``config["metadata"]``
        (only when ``metadata`` is non-None);
      - history rows arrive via ``update(state)`` per composite tick;
      - ``close(success=...)`` flushes the buffer and (when ``success=True``
        and a partitioning layout is configured) writes the success sentinel.

    Durability-sensitive callers must call ``close()`` explicitly at end of
    composite run; ``__del__`` calls it defensively but interpreter-shutdown
    ordering is undefined.
    """

    config_schema = {
        **Emitter.config_schema,
        "out_dir":           {"_type": "string",       "_default": ""},
        "out_uri":           {"_type": "string",       "_default": ""},
        "batch_size":        {"_type": "integer",      "_default": 400},
        "threaded":          {"_type": "boolean",      "_default": True},
        "flatten_separator": {"_type": "string",       "_default": "__"},
        "partitioning_keys": {"_type": "list[string]", "_default": []},
        "dtype_overrides":   {"_type": "map[string]",  "_default": {}},
        "metadata":          {"_type": "map",          "_default": {}},
    }

    @classmethod
    def emitter_contract(cls):
        from viva_emitters.contract import EmitterContract
        return EmitterContract(output_kind="parquet", output_uri_config_key="out_uri")

    def __init__(self, config: dict[str, Any], core: Any) -> None:
        super().__init__(config, core)

        if "out_uri" in config and config["out_uri"]:
            self.out_uri: str = config["out_uri"]
        elif "out_dir" in config and config["out_dir"]:
            self.out_uri = os.path.abspath(config["out_dir"])
        else:
            raise ValueError(
                "ParquetEmitter requires either config['out_dir'] or config['out_uri']"
            )

        self.filesystem: AbstractFileSystem
        self.filesystem, _ = url_to_fs(self.out_uri)

        self.batch_size: int = int(config.get("batch_size", 400))
        self.threaded: bool = bool(config.get("threaded", True))
        self.flatten_separator: str = str(config.get("flatten_separator", "__"))
        self.partitioning_keys: list[str] = list(config.get("partitioning_keys") or [])
        self.dtype_overrides: dict[str, str] = dict(config.get("dtype_overrides") or {})

        self.executor: ThreadPoolExecutor | BlockingExecutor = (
            ThreadPoolExecutor(1) if self.threaded else BlockingExecutor()
        )

        self.buffered_emits: dict[str, Any] = {}
        self.pl_types: dict[str, pl.DataType | DataTypeClass] = {}
        self.np_types: dict[str, Any] = {}
        self.pl_serialized: set[str] = set()
        # Columns whose inner type varies irreconcilably across ticks (scalar at
        # one tick, length-N array at another). Dropped on first failure and
        # skipped thereafter so one ragged listener can't sink the whole run.
        self._dropped_cols: set[str] = set()
        # column name -> unit string, captured when a quantity[...] port emits
        # a pint.Quantity; written as Parquet file metadata so the unit travels
        # with the data without renaming the (magnitude-valued) column.
        self.column_units: dict[str, str] = {}
        self.num_emits: int = 0
        self.last_batch_future: Future = Future()
        self.last_batch_future.set_result(None)
        self.experiment_id: str = ""
        self.partitioning_path: str = ""
        self._closed: bool = False

        metadata = config.get("metadata")
        # Empty dict (the schema default) means "no metadata requested" —
        # don't run the one-shot configuration write.
        if metadata:
            self._write_configuration(dict(metadata))

    def _build_partitioning_path(self, metadata: dict[str, Any]) -> str:
        """Build the hive partition path from ``self.partitioning_keys``."""
        if not self.partitioning_keys:
            return ""
        parts: list[str] = []
        for key in self.partitioning_keys:
            if key not in metadata:
                raise KeyError(
                    f"ParquetEmitter partitioning_keys requires '{key}' "
                    f"in config['metadata']; got keys: {sorted(metadata)}"
                )
            parts.append(f"{key}={metadata[key]}")
        return os.path.join(*parts)

    def _write_configuration(self, metadata: dict[str, Any]) -> None:
        """Write the one-shot configuration parquet from ``metadata``."""
        self.experiment_id = str(metadata.get("experiment_id", "default"))
        self.partitioning_path = self._build_partitioning_path(metadata)

        flat = flatten_dict(metadata, separator=self.flatten_separator)
        config_emit: dict[str, Any] = {}
        config_schema: dict[str, pl.DataType] = {}
        for k, v in flat.items():
            try:
                v_np = np.asarray(v, dtype=np_dtype(v, k, self.dtype_overrides))
                config_emit[k] = v_np[np.newaxis]
                config_schema[k] = pl_dtype_from_ndarray(v_np)
            except (ValueError, TypeError):
                series = pl.Series([v])
                config_emit[k] = series
                config_schema[k] = series.dtype

        outfile = os.path.join(
            self.out_uri,
            self.experiment_id,
            "configuration",
            self.partitioning_path,
            "config.pq",
        )
        try:
            self.filesystem.delete(os.path.dirname(outfile), recursive=True)
        except (FileNotFoundError, OSError):
            pass
        self.filesystem.makedirs(os.path.dirname(outfile))
        self.last_batch_future = self.executor.submit(
            json_to_parquet, config_emit, outfile, config_schema, self.filesystem,
        )
        # Clear out any old history files for this partition.
        history_outdir = os.path.join(
            self.out_uri, self.experiment_id, "history", self.partitioning_path,
        )
        try:
            self.filesystem.delete(history_outdir, recursive=True)
        except (FileNotFoundError, OSError):
            pass

    def _history_metadata(self) -> dict[str, str] | None:
        """File-level Parquet metadata for history files.

        Carries the unit string for each unit-bearing (``quantity[...]``)
        column under the ``column_units`` key, JSON-encoded, so the unit
        travels with the data without renaming the magnitude-valued column.
        ``None`` when no unit-bearing columns have been emitted.
        """
        if not self.column_units:
            return None
        return {"column_units": json.dumps(self.column_units)}

    def update(self, state: dict[str, Any]) -> dict:
        """Buffer one history row; flush to parquet when ``batch_size`` reached.

        Each field takes one of two paths:

        * **NumPy path** (efficient, default for first encounter): the dtype
          is fixed from the first value, and rows are buffered into a
          ``(batch_size,) + shape`` ndarray. Subsequent emits with a
          different shape or that introduce a new field mid-batch raise
          ``ValueError``, which triggers a per-field fallback to the Polars
          path. The buffer is recreated at the start of each batch.

        * **Polars path** (fallback for ragged / nullable / object data): the
          field value becomes a ``pl.Series``; rows are buffered in a
          ``list[pl.Series | None]`` and the polars dtype is reconciled
          across rows via :py:func:`union_pl_dtypes`. ``dtype_overrides``
          provide a ``force_inner`` hint so e.g. all-null prefixes don't
          collapse to ``pl.Null``.
        """
        flat = flatten_dict(state, separator=self.flatten_separator)
        flat = _split_structured_arrays(flat)
        # Unit-bearing leaves (pint.Quantity on a quantity[...] port) are numpy
        # object dtype and can't go through the numpy/Polars paths. Strip each
        # to its magnitude under the SAME column name (downstream name-based
        # queries keep working) and record the unit, written as Parquet file
        # metadata at flush.
        flat = strip_quantities(flat, self.column_units)
        overrides = self.dtype_overrides
        emit_idx = self.num_emits % self.batch_size

        for k, v in flat.items():
            if k in self._dropped_cols:
                continue
            if k not in self.pl_serialized:
                try:
                    if k not in self.np_types:
                        self.np_types[k] = np_dtype(v, k, overrides)
                    v_np = np.asarray(v, dtype=self.np_types[k])
                    if k not in self.buffered_emits:
                        if emit_idx == 0:
                            self.buffered_emits[k] = np.zeros(
                                (self.batch_size,) + v_np.shape, dtype=v_np.dtype
                            )
                        else:
                            raise ValueError(f"Field {k} added mid-batch.")
                    if k not in self.pl_types:
                        self.pl_types[k] = pl_dtype_from_ndarray(v_np)
                    if v_np.shape != self.buffered_emits[k].shape[1:]:
                        raise ValueError(f"Shape mismatch for {k}.")
                    self.buffered_emits[k][emit_idx] = v_np
                    continue
                except (ValueError, TypeError):
                    self.pl_serialized.add(k)
                    if k in self.buffered_emits and isinstance(
                        self.buffered_emits[k], np.ndarray
                    ):
                        self.buffered_emits[k] = (
                            self.buffered_emits[k][:emit_idx].tolist()
                            + [None] * (self.batch_size - emit_idx)
                        )
            # Polars fallback. Two normalisations before wrapping in a
            # pl.Series so the Polars path handles real composite values:
            #
            # 1. Cast to the remembered numpy dtype (if any) so int32 vs
            #    int64 doesn't bounce inner types across rows. Without this
            #    union_pl_dtypes raises "Incompatible inner types".
            # 2. Convert multi-D ndarrays to nested Python lists. Polars's
            #    Series constructor can't parse a 2+D numpy array (errors
            #    out with "cannot parse numpy data type dtype('O')"), but
            #    happily reads a Python list-of-lists into a List(List(...))
            #    column. Common for fields whose first tick is shape (0, 0)
            #    (locks buffer to that 2D shape) and later ticks are (M, N).
            np_type = self.np_types.get(k)
            if np_type is not None and isinstance(v, np.ndarray):
                try:
                    v = v.astype(np_type, copy=False)
                except (TypeError, ValueError):
                    pass
            if isinstance(v, np.ndarray) and v.ndim > 1:
                v = v.tolist()
            ser = pl.Series([v])
            curr_type = self.pl_types.setdefault(k, pl.Null)
            if ser.dtype != curr_type:
                force_inner = _polars_dtype_from_override(k, overrides)
                try:
                    self.pl_types[k] = union_pl_dtypes(
                        curr_type, ser.dtype, k, force_inner
                    )
                except TypeError:
                    # Inner type varies irreconcilably across ticks (e.g. a
                    # listener that is a scalar at t=0 then a length-N array
                    # later): it can't live in one parquet column. Drop it
                    # rather than sinking the whole run — mirrors
                    # json_to_parquet's drop-on-fail at flush.
                    self._drop_column(k)
                    continue
            if k not in self.buffered_emits:
                self.buffered_emits[k] = [None] * self.batch_size
            self.buffered_emits[k][emit_idx] = ser[0]

        self.num_emits += 1
        if self.num_emits % self.batch_size == 0:
            # If last batch failed, that exception surfaces here.
            self.last_batch_future.result()
            outfile = os.path.join(
                self.out_uri,
                self.experiment_id or "default",
                "history",
                self.partitioning_path,
                f"{self.num_emits}.pq",
            )
            self.filesystem.makedirs(os.path.dirname(outfile), exist_ok=True)
            self.last_batch_future = self.executor.submit(
                json_to_parquet,
                self.buffered_emits,
                outfile,
                self.pl_types,
                self.filesystem,
                self._history_metadata(),
            )
            # Clear buffers so the background writer can't be racing with
            # the next batch's writes into the same arrays.
            if self.threaded:
                self.buffered_emits = {}
        return {}

    def _drop_column(self, k: str) -> None:
        """Permanently drop a column that can't be stored (irreconcilable inner
        types across ticks). Removes any buffered/typed state and records the
        name so later ticks skip it. Logged once for diagnosability."""
        if k in self._dropped_cols:
            return
        self._dropped_cols.add(k)
        self.buffered_emits.pop(k, None)
        self.pl_types.pop(k, None)
        self.np_types.pop(k, None)
        self.pl_serialized.discard(k)
        self.column_units.pop(k, None)
        warnings.warn(
            f"ParquetEmitter: dropping column {k!r} — its value type varies "
            f"across ticks and cannot be stored in a single parquet column.",
            stacklevel=2,
        )

    def _flush_partial_batch(self) -> None:
        """Write the trailing partial batch (rows after the last batch_size flush).

        No-op when there are no unflushed rows. Used by both ``close()``
        (for end-of-run durability) and ``query()`` (so a query on an open
        emitter sees in-memory rows).
        """
        if self.num_emits % self.batch_size == 0 or not self.buffered_emits:
            return
        # Wait for any in-flight batch first.
        self.last_batch_future.result()
        rows_in_batch = self.num_emits % self.batch_size
        trimmed = {k: v[:rows_in_batch] for k, v in self.buffered_emits.items()}
        outfile = os.path.join(
            self.out_uri,
            self.experiment_id or "default",
            "history",
            self.partitioning_path,
            f"{self.num_emits}.pq",
        )
        self.filesystem.makedirs(os.path.dirname(outfile), exist_ok=True)
        self.last_batch_future = self.executor.submit(
            json_to_parquet, trimmed, outfile, self.pl_types, self.filesystem,
            self._history_metadata(),
        )
        self.buffered_emits = {}

    def close(self, success: bool = False) -> None:
        """Flush remaining buffer; write success sentinel when partitioned + success."""
        if self._closed:
            return
        self._flush_partial_batch()
        # Wait for the executor's last in-flight write.
        self.last_batch_future.result()
        if isinstance(self.executor, ThreadPoolExecutor):
            self.executor.shutdown(wait=True)
        if success and self.partitioning_keys:
            success_file = os.path.join(
                self.out_uri,
                self.experiment_id or "default",
                "success",
                self.partitioning_path,
                "s.pq",
            )
            try:
                self.filesystem.delete(os.path.dirname(success_file), recursive=True)
            except (FileNotFoundError, OSError):
                pass
            self.filesystem.makedirs(os.path.dirname(success_file))
            pl.DataFrame({"success": [True]}).write_parquet(success_file)
        self._closed = True

    @staticmethod
    def flush_all_in_composite(composite: Any, success: bool = True) -> int:
        """Walk a composite's state, call close() on every ParquetEmitter step.

        Composites typically construct the ParquetEmitter inside their step
        factory; the driver script never sees the instance and so cannot
        call close() directly. Without an explicit close, a partial last
        batch (rows after the most recent batch_size flush) stays in
        memory and never lands on disk. Call this helper after the
        simulation loop ends to make the trailing batch durable.

        Returns the number of ParquetEmitter instances closed.
        """
        closed = 0
        def _walk(node: Any) -> None:
            nonlocal closed
            if isinstance(node, dict):
                inst = node.get("instance")
                if isinstance(inst, ParquetEmitter) and not inst._closed:
                    inst.close(success=success)
                    closed += 1
                for v in node.values():
                    _walk(v)
        _walk(getattr(composite, "state", None) or {})
        return closed

    def __del__(self) -> None:
        """Defensive flush on garbage collection.

        When the emitter is wired into a process-bigraph composite step, no
        explicit ``close()`` call happens at composite end-of-life — the step
        instance gets garbage-collected and any trailing partial batch is
        lost. ``__del__`` flushes the trailing rows so they land on disk.
        Callers that care about the success sentinel or about durable flushes
        during normal operation should still call ``close()`` explicitly.

        NON-BLOCKING by design: a finalizer must never block. GC can invoke
        ``__del__`` re-entrantly at an arbitrary moment — e.g. while a *later*
        composite is being built and the interpreter is mid type-dispatch —
        and ``close()``'s ``last_batch_future.result()`` wait deadlocks there
        (the awaited write can't make progress because the calling thread is
        the one stuck in __del__). So instead of ``close()``, we submit the
        trailing batch and shut the pool down *without* waiting; the
        ThreadPoolExecutor's own atexit join still lands the write before the
        interpreter exits.
        """
        try:
            if not getattr(self, "_closed", True):
                self._closed = True
                self._flush_partial_batch()
                if isinstance(self.executor, ThreadPoolExecutor):
                    self.executor.shutdown(wait=False)
        except Exception:
            # Never raise from __del__ — Python emits "Exception ignored in"
            # warnings for that and it pollutes test output / logs.
            pass

    def query(self, paths=None, query=None) -> Any:
        """Read back what was emitted. Returns a polars DataFrame."""
        # Flush so unwritten rows are visible.
        if not self._closed:
            self._flush_partial_batch()
            self.last_batch_future.result()
        history_dir = os.path.join(
            self.out_uri, self.experiment_id or "default", "history",
        )
        # Escape single quotes for the SQL string literal.
        escaped = history_dir.replace("'", "''")
        if self.partitioning_keys:
            sql = (
                f"SELECT * FROM read_parquet('{escaped}/**/*.pq', "
                f"hive_partitioning = true)"
            )
        else:
            sql = f"SELECT * FROM read_parquet('{escaped}/*.pq')"
        conn = duckdb.connect(":memory:")
        try:
            df = conn.execute(sql).pl()
        finally:
            conn.close()
        select = paths if paths is not None else query
        if isinstance(select, list):
            sep = self.flatten_separator
            cols = [sep.join(p) if isinstance(p, list) else p for p in select]
            df = df.select([c for c in cols if c in df.columns])
        return df
