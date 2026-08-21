
"""
Utilities for controlling low-level Zarr internals (read side).

Ported from vEcoli's ``ecoli.library.xarray_emitter.zarr_utils`` (PR #414). Only
the store-access and codec-parsing helpers used by the map-reduce engine
(:py:mod:`.zarr_mapreduce`) and by downstream analysis pipelines are kept here;
the consolidated-metadata management functions live in
:py:mod:`viva_emitters.xarray_emitter.zarr_writer` (viva-emitters keeps its own
diverged implementation of those).
"""

from lzma import FILTER_LZMA2, FORMAT_RAW
from typing import Any, Literal, cast

from numpy.typing import NDArray
from zarr.abc.codec import Codec
from zarr.abc.numcodec import Numcodec
from zarr.core.array import (
    Array,
    AsyncArray,
    _parse_chunk_encoding_v2,
    default_compressors_v3,
)
from zarr.core.dtype import parse_dtype
from zarr.core.group import AsyncGroup, Group
from zarr.core.indexing import BlockIndex
from zarr.core.metadata import v2, v3
from zarr.errors import UnstableSpecificationWarning, ZarrUserWarning

from .storage import VariableEncoding
from .utils import WarningFilter, filter_warnings

# ==============================================================================
# Zarr warnings
# ==============================================================================


zarr_warnings: dict[str, WarningFilter] = {
    "consolidated_metadata": WarningFilter(
        module="zarr.api.asynchronous",
        category=ZarrUserWarning,
        message="Consolidated metadata.*Zarr format 3",
        action="ignore"),
    "string": WarningFilter(
        module="zarr.core.dtype.npy.string",
        category=UnstableSpecificationWarning,
        message=".*data type.*Zarr V3",
        action="ignore"),
    "numcodecs": WarningFilter(
        module="zarr.codecs.numcodecs",
        category=ZarrUserWarning,
        message=".*Numcodecs codecs.*Zarr version 3 specification",
        action="ignore"),
    "zarrs": WarningFilter(
        module="zarrs.pipeline",
        category=UserWarning,
        message="Array is unsupported by ZarrsCodecPipeline",
        action="ignore")
}


# ==============================================================================
# Zarr codecs
# ==============================================================================


ZARR_FILTERS: dict[str, dict[int, list[dict[str, Any]]]] = {
    "delta": {
        2: [{"id": "delta", "dtype": None}],
        3: [{"name": "numcodecs.delta", "configuration": {"dtype": None}}]
    },
    "num": {
        2: [],
        3: []
    },
}
"""
Default filter codecs for :py:meth:`.AsyncZarrBufferWriter.var_codecs`, as a
function of the array category and the Zarr format.
"""
ZARR_COMPRESSORS: dict[str, dict[int, list[dict[str, Any]]]] = {
    "delta": {
        2: [{"id": "blosc", "cname": "zstd", "clevel": 6,
            "shuffle": -1, "blocksize": 0}],
        3: [{"name": "blosc", "configuration": {
            "cname": "zstd", "clevel": 6,
            "typesize": None, "shuffle": None, "blocksize": 0}}]
    },
    "num": {
        2: [{"id": "lzma", "check": -1, "preset": None,
             "format": FORMAT_RAW,
             "filters": [{"id": FILTER_LZMA2, "preset": 6}]}],
        3: [{"name": "numcodecs.lzma",
             "configuration": {"format": FORMAT_RAW,
                               "filters": [{"id": FILTER_LZMA2, "preset": 6}]}}]
    },
}
"""
Default compression codecs for :py:meth:`.AsyncZarrBufferWriter.var_codecs`, as
a function of the array category and the Zarr format.
"""


# ------------------------------------------------------------------------------


def parse_codecs(
    zarr_format: Literal[2, 3], /, *,
    codecs: dict[str, Any] | None = None,
    category: str | None = None,
    dtype: str | None = None
) -> VariableEncoding:
    """
    Translate a Vivarium JSON configuration into a codec specification that is
    interpretable by the Zarr API, leveraging Zarr internal functions for
    parsing.

    Used by: :py:meth:`.AsyncZarrBufferWriter.coo_codecs`,
    :py:meth:`.AsyncZarrBufferWriter.var_codecs`.
    """
    filters: tuple[Codec | Numcodec, ...] | None
    compressors: tuple[Codec | Numcodec | None, ...]

    # dispatch on spec type
    z = zarr_format
    if codecs or (category is not None):
        # fetch non-default config
        if codecs:
            # fetch custom JSON config
            assert category is None
            assert dtype is None
            _filters = codecs.get(f"filters_v{z}", [])
            _compressors = codecs.get(f"compressors_v{z}", [])
            if not (_filters or _compressors):
                raise KeyError(
                    f"Missing arguments:\n  "
                    f"{{\"filters_v{z}\": ..., \"compressors_v{z}\": ...}}")
        elif category is not None:
            # fetch library preset, supply data type information
            _filters = ZARR_FILTERS[category][z]
            _compressors = ZARR_COMPRESSORS[category][z]
            assert dtype is not None
            for f in _filters:
                if z == 2:
                    f["dtype"] = dtype
                else:
                    f["configuration"]["dtype"] = dtype
        # parse non-default config
        with filter_warnings(list(zarr_warnings.values())):
            if z == 2:
                filters = v2.parse_filters(_filters)
                compressors = tuple(map(v2.parse_compressor, _compressors))
            else:
                filters = v3.parse_codecs(_filters)
                compressors = v3.parse_codecs(_compressors)
    else:
        # fetch default config, supply data type information
        assert dtype is not None
        _dtype = parse_dtype(dtype, zarr_format=z)
        if z == 2:
            filters, compressor = _parse_chunk_encoding_v2(
                filters="auto", compressor="auto", dtype=_dtype
            )
            compressors = (compressor,)
        else:
            filters = ()
            compressors = default_compressors_v3(_dtype)

    return {"filters": filters, "compressors": compressors}


# ==============================================================================
# Zarr store access
# ==============================================================================


def get_group(group: Group, path: str) -> Group:
    """
    Access a Zarr store path known to be a :py:class:`~zarr.Group`.
    """
    return cast(Group, group[path])


def get_array(group: Group, path: str) -> Array:
    """
    Access a Zarr store path known to be an :py:class:`~zarr.Array`.
    """
    return cast(Array,  group[path])


def get_ndarray(group: Group, path: str) -> NDArray:
    """
    Access a Zarr store path known to be an :py:class:`~zarr.Array`, and
    retrieve the uncompressed array data.
    """
    return cast(NDArray,  get_array(group, path)[:])


def get_rectilinear_ndarray(group: Group, path: str) -> list:
    """
    Given a Zarr array that was stored using a `rectilinear chunk grid`_, return
    a list over its block views.

    .. _rectilinear chunk grid: https://zarr.readthedocs.io/en/stable/user-guide/examples/rectilinear_chunks/
    """
    arr = get_array(group, path)
    blk_view: BlockIndex = arr.blocks
    blk_ix = [range(len(dim)) for dim in arr.write_chunk_sizes]
    def get_blocks(outer: tuple[int,...], inner: list[range]):
        return ([get_blocks(outer + (i,), inner[1:]) for i in inner[0]]
                if inner else
                blk_view[outer])
    return get_blocks((), blk_ix)


async def get_async_group(group: AsyncGroup, path: str) -> AsyncGroup:
    """
    Access a Zarr store path known to be an :py:class:`~zarr.AsyncGroup`.
    """
    return cast(AsyncGroup, await group.getitem(path))


async def get_async_array(group: AsyncGroup, path: str) -> AsyncArray:
    """
    Access a Zarr store path known to be an :py:class:`~zarr.AsyncArray`.
    """
    return cast(AsyncArray, await group.getitem(path))
