"""Back-compat shim: ``pbg_emitters`` was renamed to ``viva_emitters``.

The import package was renamed as part of the pbg -> viva rebrand. This shim keeps
every existing consumer working during the deprecation window:

  * ``import pbg_emitters`` / ``from pbg_emitters import X`` works
    (re-exports the new package's top-level ``__all__``);
  * ``import pbg_emitters.<sub>`` transparently resolves to
    ``viva_emitters.<sub>`` via a meta-path finder (so
    ``from pbg_emitters.run_reader import RunReader`` keeps working); and
  * ``python -m pbg_emitters.<sub>`` still executes (``get_code`` forwards the
    real module's code object to ``runpy``).

Importing anything under this package emits a one-time :class:`DeprecationWarning`.
Update imports to ``viva_emitters``; this shim is removed in a future major
release. (Mirrors the ``pbg_superpowers`` shim in viva-superpowers.)
"""
from __future__ import annotations

import importlib
import importlib.abc
import importlib.util
import sys
import warnings

warnings.warn(
    "pbg_emitters is renamed to viva_emitters; update your imports "
    "(the pbg_emitters alias is removed in a future major release).",
    DeprecationWarning,
    stacklevel=2,
)

_OLD = "pbg_emitters."
_NEW = "viva_emitters."


class _Redirect(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Forward ``pbg_emitters.<sub>`` imports to ``viva_emitters.<sub>``.

    ``create_module``/``exec_module`` handle ordinary ``import`` (the imported
    submodule object is aliased into ``sys.modules`` under both names), while
    ``get_code`` lets ``python -m pbg_emitters.<sub>`` execute the real module's
    code object as ``__main__``.
    """

    def _target(self, name: str) -> str:
        return _NEW + name[len(_OLD):]

    def find_spec(self, name, path=None, target=None):
        if not name.startswith(_OLD):
            return None
        real = importlib.util.find_spec(self._target(name))
        if real is None:
            return None
        spec = importlib.util.spec_from_loader(
            name,
            self,
            origin=real.origin,
            is_package=real.submodule_search_locations is not None,
        )
        if real.submodule_search_locations is not None:
            spec.submodule_search_locations = list(real.submodule_search_locations)
        return spec

    def create_module(self, spec):
        # Alias the fully-initialized new-package module under BOTH names so
        # `import a.b` and identity checks against either name agree.
        mod = importlib.import_module(self._target(spec.name))
        sys.modules[spec.name] = mod
        return mod

    def exec_module(self, module):  # already executed by import_module
        pass

    def get_code(self, name):
        # Support `python -m pbg_emitters.<sub>`: runpy needs a code object.
        target = self._target(name)
        return importlib.util.find_spec(target).loader.get_code(target)


sys.meta_path.insert(0, _Redirect())

_viva = importlib.import_module("viva_emitters")
__version__ = getattr(_viva, "__version__", "0.1.0")
# Re-export the new package's public surface so `from pbg_emitters import X`
# (e.g. RunReader, by_generation, SQLiteEmitter) keeps working. Prefer a declared
# ``__all__``; viva_emitters declares none, so fall back to every public
# (non-underscore) top-level attribute.
_names = getattr(_viva, "__all__", None) or [n for n in dir(_viva) if not n.startswith("_")]
globals().update({k: getattr(_viva, k) for k in _names})
