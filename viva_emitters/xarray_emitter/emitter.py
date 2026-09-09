"""Generic XArrayEmitter — Zarr-via-Xarray writer for process-bigraph composites.

Vendored from vivarium-collective/vEcoli@b25ca24 (PR #414 head). Re-rooted
onto process_bigraph.emitter.Emitter via the BufferedEmitter base in _base.

The vivarium emit() two-channel handshake collapses into __init__ (one-shot
configuration: build partition, allocate transducer, open writer store)
plus update(state) (per-tick history). finalize() becomes close(success).
metadata_keys and metadata_validators are now config knobs.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from pprint import pp
from typing import Any

from process_bigraph.emitter import Emitter

from ._base import BufferedEmitter, StoragePartition
from .transducer import XarrayTransducer
from .storage import XarrayStoragePartition
from .writer import AsyncBufferWriter
from .utils import emitter_arg_error


class XArrayEmitter(BufferedEmitter):
    """Generic XArrayEmitter. See module docstring for the lifecycle mapping."""

    config_schema = {
        **Emitter.config_schema,
        "out_uri":             {"_type": "string", "_default": ""},
        "strategy":            {"_type": "string", "_default": "flat"},
        "emit_root":           {"_type": "list", "_default": []},
        "transducer":          {"_type": "map", "_default": {}},
        "view":                {"_type": "list", "_default": []},
        "writer":              {"_type": "map", "_default": {}},
        "metadata":            {"_type": "map", "_default": {}},
        "metadata_keys":       {"_type": "list[string]", "_default": []},
        "metadata_validators": {"_type": "map", "_default": {}},
        "output_metadata":     {"_type": "map", "_default": {}},
        "provenance":          {"_type": "map", "_default": {}},
        "debug":               {"_type": "boolean", "_default": False},
    }

    @classmethod
    def emitter_contract(cls):
        from viva_emitters.contract import EmitterContract
        return EmitterContract(output_kind="zarr", output_uri_config_key="out_uri")

    def __init__(self, config: dict[str, Any], core: Any) -> None:
        self.validate_config(config)
        #: The emitter's own config, retained so :py:meth:`.advance_generation`
        #: can rebuild the buffering pipeline for the next generation of a
        #: lineage from the same emitter instance.
        self._config: dict[str, Any] = config
        self.debug: bool = bool(config.get("debug", False))
        #: Partition strategy: "flat" (default, generic Step) or "colony"
        #: (v2ecoli lineage layout). Selects how `extract_partition` reads
        #: `agent_id`/`generation` from metadata.
        self._strategy: str = config.get("strategy") or "flat"
        self._metadata_keys: list[str] = list(config.get("metadata_keys") or [])
        self._metadata_validators: dict[str, Any] = dict(
            config.get("metadata_validators") or {}
        )
        #: Immutable run provenance (which composite + config produced this
        #: output) written verbatim to the zarr store's ROOT attributes under
        #: the "provenance" key, so a saved simulation is self-describing.
        #: Treated as an opaque JSON-serializable map — empty => no attr.
        self._provenance: dict[str, Any] = dict(config.get("provenance") or {})
        self._closed: bool = False
        #: True once the pending (possibly partial) in-memory buffer has been
        #: terminally flushed to the store. Shared by :py:meth:`close` and
        #: :py:meth:`query` so the trailing buffer lands on disk exactly once —
        #: repeated ``query()`` reads must not re-append it (idempotent read).
        self._flushed: bool = False

        # Unconditionally build the transducer and writer. Tests that only
        # exercise the validator path should supply a minimum-valid transducer
        # config (see the `minimal_xarray_config` fixture in tests/conftest.py).
        self._build_writer_and_transducer()

        # Call the BufferedEmitter base __init__ AFTER setting up attributes
        # (per the upstream warning that __init__ must be called at the end).
        BufferedEmitter.__init__(self, config, core)

        # vivarium's "configuration" emit happens here at construction time
        # when metadata is available. validate_metadata is called before
        # transducer.alloc so a validator mismatch raises ValueError early.
        self._alloc_and_open(dict(config.get("metadata") or {}))

    def _build_writer_and_transducer(self) -> None:
        """Allocate a fresh transducer + writer pair from :py:attr:`._config`.

        Called at construction and again by :py:meth:`.advance_generation` for
        each new generation of a lineage (each generation is a fresh single-use
        buffering pipeline over the SAME per-lineage store).
        """
        self.transducer = XarrayTransducer(self._config, debug=self.debug)
        self.writer = AsyncBufferWriter.dispatch(self._config["writer"])
        # Provenance is written to the store's ROOT attrs at finalize time,
        # by the writer, so it survives the async buffer + consolidation.
        self.writer.provenance = self._provenance

    def _alloc_and_open(self, metadata: dict[str, Any]) -> None:
        """Allocate this generation's buffer and open its partition in the store.

        When ``metadata`` is empty the store is left unopened (the store-less
        validator/test path); :py:meth:`.flush` and :py:meth:`.close` tolerate
        that. Otherwise the partition (``experiment_id``/``variant``/
        ``lineage_seed``, and — under the ``colony`` strategy — the
        ``agent_id`` that fixes ``generation``) is derived and the writer opens
        the corresponding ``generation=N`` partition in the per-lineage store.
        """
        self._flushed = False
        self._closed = False
        if metadata:
            self.validate_metadata(metadata)
            partition = self.extract_partition(metadata)
            extracted_meta = self.extract_metadata(metadata)
            coords = self._config.get("output_metadata") or {}
            self.transducer.alloc(
                partition=partition, metadata=extracted_meta, coords=coords,
            )
            self.writer.open_store(self.transducer.buffer)

    def advance_generation(
        self, *, agent_id: str | None = None,
        metadata: dict[str, Any] | None = None, success: bool = True,
    ) -> None:
        """Finalize the current generation and open the next one, in place.

        A lineage is driven by a SINGLE emitter: call :py:meth:`update` for a
        generation's history, then ``advance_generation`` at the division
        event, then :py:meth:`update` for the daughter, and so on, ending the
        lineage with :py:meth:`close`.

        The current generation is finalized *first* and unconditionally — its
        trailing (possibly sub-``buffer_size``) buffer is flushed, the cell
        division event (``success``) is marked, and consolidated metadata is
        written — so it is durably on disk before the next generation opens.
        This is the guarantee the old per-generation-emitter flow lacked: its
        driver wrapped ``close()`` in ``except AssertionError: pass``, so a
        generation that failed to persist left the store as bare group
        skeletons and crashed the next generation's ``_check_group``.

        Args:
          agent_id: Phylogeny key of the next generation (``generation ==
                    len(agent_id)`` under the colony strategy). Updates the
                    ``agent_id`` in both the stored metadata and the
                    agent-keyed ``emit_root`` so the daughter's payload is read
                    correctly. Ignored when ``metadata`` is given explicitly.
          metadata: Full metadata for the next generation. When omitted it is
                    derived from the emitter's config with ``agent_id`` applied.
          success:  Whether the finished generation reached its division event
                    (marks the ``division_reached`` attr the next generation's
                    ``_check_group`` requires). Defaults to ``True`` — a
                    generation you advance off of divided by definition.
        """
        if self.finalized or self._closed:
            raise RuntimeError(
                f"`{type(self).__name__}.advance_generation()` called after "
                f"the lineage was closed.")
        # 1. GUARANTEED finalize of the current generation (never swallowed).
        if not self._flushed:
            self.flush(final=True)
            self._flushed = True
        if self.writer is not None and self.writer._buffer is not None:
            if success:
                self.writer.mark_success()
            self.writer.close()

        # 2. Resolve the next generation's metadata + agent-keyed emit_root.
        if metadata is None:
            metadata = dict(self._config.get("metadata") or {})
            if agent_id is not None:
                metadata["agent_id"] = agent_id
                # Keep an optional caller-tracked ``generation`` field (the
                # colony ``generation == len(agent_id)``) consistent with the
                # new agent_id, so the stored metadata attr is not stale for
                # generation > 1. Only touched when the caller already carries
                # it (the storage partition itself derives generation from
                # agent_id regardless).
                if "generation" in metadata:
                    metadata["generation"] = len(agent_id)
        self._config["metadata"] = metadata
        if agent_id is not None and self._config.get("emit_root"):
            # the colony emit_root is ["agents", <agent_id>]; retarget its
            # agent element so the daughter's payload is stripped correctly.
            new_root = list(self._config["emit_root"])
            new_root[-1] = agent_id
            self._config["emit_root"] = new_root

        # 3. Rebuild the pipeline and open the next generation in the same store.
        self._build_writer_and_transducer()
        self.finalized = False
        self._alloc_and_open(metadata)

    @classmethod
    def validate_config(cls, config: dict[str, Any]) -> None:
        for key in ("transducer", "view", "writer"):
            if key not in config:
                raise KeyError(emitter_arg_error(
                    cls, "Missing argument", f'"{key}": ...'
                ))
        match config.get("debug", False):
            case bool():
                pass
            case debug:
                raise TypeError(emitter_arg_error(
                    cls, "Invalid argument", f'"debug": {debug}'
                ))

    def validate_metadata(self, metadata: dict[str, Any]) -> None:
        """Check config['metadata_validators'] against the supplied metadata."""
        for key, expected in self._metadata_validators.items():
            actual = metadata.get(key)
            if bool(actual) != bool(expected):
                raise ValueError(
                    f"\n  Metadata field unsupported by {type(self).__name__}:"
                    f'\n    {{"{key}": {actual}}}'
                )

    def extract_partition(self, metadata: dict[str, Any]) -> XarrayStoragePartition:
        return XarrayStoragePartition.cast(
            BufferedEmitter.extract_partition(
                self, metadata, strategy=self._strategy)
        )

    def extract_metadata(self, metadata: dict[str, Any]) -> dict[str, Any]:
        """Return the subset of `metadata` named by `self._metadata_keys`."""
        keys = self._metadata_keys
        if not keys:
            selected = dict(metadata)
        else:
            selected = {k: metadata[k] for k in keys if k in metadata}
        # Reduce to JSON-friendly types.
        for k, v in list(selected.items()):
            match v:
                case Path():
                    selected[k] = str(v)
                case datetime():
                    selected[k] = str(v.astimezone())
        if self.debug:
            hline = "-" * 79
            print(f"\nMetadata:\n{hline}")
            pp(selected)
            print(hline)
        return selected

    @property
    def partition(self) -> XarrayStoragePartition:
        assert self.transducer is not None
        return self.transducer.buffer.partition

    def flush(self, *, final: bool = False) -> None:
        if self.transducer is None or self.writer is None:
            return
        if self.writer._buffer is None:
            # store not yet opened (no metadata was provided at construction)
            return
        self.writer.write(self.transducer, final=final)

    def update(self, state: dict[str, Any]) -> dict:
        """Buffer one history row via the transducer; flush when buffer fills.

        The ``state`` dict must be shaped to match the underlying vEcoli/vivarium
        storage layout that the transducer expects: a top-level ``"agents"`` key
        keyed by ``agent_id`` (matching ``config["metadata"]["agent_id"]``) plus a
        top-level ``"time"`` key. Example::

            emitter.update({
                "time": 1.0,
                "agents": {"1": {"listeners": {"global_time": 1.0}}},
            })

        This nesting is inherited from PR #414's ``XarrayTransducer.write()``
        implementation (see ``transducer.py``); it's the cost of vendoring the
        upstream transducer/view machinery unchanged.
        """
        if self.transducer is None:
            return {}
        if not self.transducer.step(state):
            self.flush()
            self.transducer.shift()
            assert self.transducer.step(state)
        return {}

    def close(self, success: bool = False) -> None:
        """Flush final batch, finalize the buffered base, close the writer."""
        if self._closed:
            return
        # Terminal flush of the trailing (possibly partial) buffer — but only
        # if a prior query() has not already done it (see `_flushed`), so the
        # buffer is never appended to the store twice.
        if not self._flushed:
            self.flush(final=True)
            self._flushed = True
        if self.writer is not None and self.writer._buffer is not None:
            if success:
                self.writer.mark_success()
            self.writer.close()
        self.finalized = True
        self._closed = True

    def _finalize(self, *, success: bool) -> None:
        """Adapter for the BufferedEmitter abstract method."""
        self.close(success=success)

    def query(self, paths=None, query=None) -> Any:
        """Open the written Zarr store and return an xarray DataTree.

        A pure, **idempotent** read: calling ``query()`` repeatedly returns the
        same data and never mutates (grows) the store. Any rows still buffered
        in memory when ``query()`` is first called are flushed to the store
        **once** via the terminal (``final=True``) write path, which tolerates a
        partial — i.e. not ``buffer_size``-aligned — buffer (it truncates rather
        than asserting the buffer is exactly full). Subsequent calls skip the
        flush (guarded by ``_flushed``) and simply re-open the store, so
        resolving an emitter's results is repeatable.

        Note: like ``close()``, the terminal flush finalizes the in-memory
        buffer; the emitter is not expected to keep emitting after ``query()``.
        The normal per-tick write path and ``close()`` semantics are unchanged.
        """
        if (not self._closed and not self._flushed
                and self.transducer is not None
                and self.transducer.buf_tix > 0):
            # Persist the trailing (possibly partial) buffer exactly once.
            self.flush(final=True)
            # Await the async append so the just-written rows are on disk before
            # the fresh reader below opens the store.
            if self.writer is not None and self.writer._buffer is not None:
                self.writer.future.result()
            self._flushed = True
        import xarray as xr
        assert self.writer is not None
        tree = xr.open_datatree(self.writer.out_uri, engine="zarr")
        select = paths if paths is not None else query
        if isinstance(select, list):
            tree = tree[select] if hasattr(tree, "__getitem__") else tree
        return tree

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
