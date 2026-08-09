"""Generic XArrayEmitter for process-bigraph composites.

Vendored from vivarium-collective/vEcoli@b25ca24 (PR #414 head). Re-rooted
onto process_bigraph.emitter.Emitter. vEcoli-specific metadata keys and
validator checks are config-driven; a downstream builder can reproduce
vEcoli's exact behavior.
"""

import sys

# The xarray backend's implementation modules use Python 3.12+ syntax
# (PEP 695 type aliases / generic classes, PEP 701 f-strings). On < 3.12 those
# modules raise SyntaxError at import, which is NOT an ImportError and so would
# escape the optional-extra guards in ``viva_emitters/__init__`` and crash every
# importer of ``process_bigraph`` (which imports viva_emitters). Fail early here
# with a clean ImportError so the optional backend degrades gracefully instead.
if sys.version_info < (3, 12):
    raise ImportError(
        "viva_emitters.xarray_emitter requires Python >= 3.12 "
        "(uses PEP 695 / PEP 701 syntax); the [xarray] backend is unavailable "
        f"on this interpreter (Python {sys.version_info.major}.{sys.version_info.minor})."
    )

try:
    import xarray  # noqa: F401
    import zarr    # noqa: F401
    import zarrs   # noqa: F401
except ImportError as e:
    raise ImportError(
        f"viva_emitters.xarray_emitter requires the [xarray] extra "
        f"(pip install 'viva-emitters[xarray]'). (missing: {e.name})"
    ) from e

from viva_emitters.xarray_emitter.emitter import XArrayEmitter

__all__ = ["XArrayEmitter"]
