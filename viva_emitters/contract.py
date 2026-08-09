"""Minimal self-described emitter contract.

Beyond the standard bigraph-schema interface (inputs()/outputs()/config_schema),
a dashboard/consumer needs only two facts to read back a finished run: the
output_kind (which external sink/reader) and which config key carries the store
path. Every emitter is a process-bigraph Step, so there is intentionally no
`driving_mode` field.
"""
from __future__ import annotations

import dataclasses

_OUTPUT_KINDS = frozenset({"sqlite", "zarr", "parquet", "ram"})


@dataclasses.dataclass(frozen=True)
class EmitterContract:
    output_kind: str
    output_uri_config_key: str | None = None

    def __post_init__(self) -> None:
        if self.output_kind not in _OUTPUT_KINDS:
            raise ValueError(
                f"invalid output_kind {self.output_kind!r}; "
                f"expected one of {sorted(_OUTPUT_KINDS)}")
