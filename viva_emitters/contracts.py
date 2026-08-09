"""Emitter-contract registry.

Resolution order in `contract_for`: a class that exposes an `emitter_contract()`
classmethod wins (self-description); otherwise the name is looked up in the
registry. RAMEmitter lives in process-bigraph (which we do not edit), so its
contract is registered here as a literal. Third-party emitters may register the
same way.
"""
from __future__ import annotations

from .contract import EmitterContract

_REGISTRY: dict[str, EmitterContract] = {}


def register_contract(key: str, contract: EmitterContract) -> None:
    _REGISTRY[str(key)] = contract


def contract_for(key) -> EmitterContract:
    m = getattr(key, "emitter_contract", None)
    if callable(m):
        return m()
    name = key if isinstance(key, str) else getattr(key, "__name__", str(key))
    if name in _REGISTRY:
        return _REGISTRY[name]
    raise KeyError(f"no emitter contract registered for {name!r}")


# RAMEmitter lives in process-bigraph (not editable here) — register a literal.
_RAM = EmitterContract(output_kind="ram", output_uri_config_key=None)
register_contract("ram", _RAM)
register_contract("RAMEmitter", _RAM)

# Self-register the viva emitters under short + class name. Guarded because the
# emitter modules may require extras that aren't installed (mirrors __init__).
try:
    from .sqlite_emitter import SQLiteEmitter
    register_contract("sqlite", SQLiteEmitter.emitter_contract())
    register_contract("SQLiteEmitter", SQLiteEmitter.emitter_contract())
except ImportError:
    pass

try:
    from .parquet_emitter import ParquetEmitter
    register_contract("parquet", ParquetEmitter.emitter_contract())
    register_contract("ParquetEmitter", ParquetEmitter.emitter_contract())
except ImportError:
    pass

try:
    from .xarray_emitter import XArrayEmitter
    register_contract("xarray", XArrayEmitter.emitter_contract())
    register_contract("XArrayEmitter", XArrayEmitter.emitter_contract())
except ImportError:
    pass
