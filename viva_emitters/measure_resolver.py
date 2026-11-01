"""Shared resolver for bare-string measures.

A study can express an acceptance measure two ways:

* a DICT measure (``{kind, path}``) the engine resolves via its registered
  derived-scalar machinery, or
* a BARE-STRING measure (``measure: lysis_probability``) — the M-track /
  workbench convention — where the name is a function on the study's **composite
  module** that computes a float from the run's raw per-tick rows:
  ``lysis_probability(rows) -> float``.

This module is the ONE place both sides grade a bare-string measure, so the
workbench grading/observables path and the engine's ``evaluate_test`` /
``compute_outcomes`` produce byte-identical results instead of each
re-implementing "rows". It depends only on :class:`viva_emitters.run_reader.RunReader`
(for :py:meth:`~viva_emitters.run_reader.RunReader.rows`) and ``importlib`` — it
does NOT import the composite registry, so it stays a low-level leaf both
packages can import without a cycle. The caller supplies the composite module it
has already resolved (or a dotted module path); this module only calls
``getattr(module, name)`` and feeds it ``reader.rows()``.
"""

from __future__ import annotations

import importlib
from types import ModuleType
from typing import Any, Callable


def resolve_measure_fn(composite_module: ModuleType | str, name: str) -> Callable[[list[dict]], Any]:
    """Resolve a bare-string measure ``name`` to the composite module's function.

    Args:
        composite_module: The study's composite module (already imported) or its
            dotted import path (e.g. ``"viva_eukaryogenesis.composites.vesicle"``).
        name: The bare-string measure id (the function to call).

    Returns:
        The callable ``name(rows) -> value`` defined on the composite module.

    Raises:
        ModuleNotFoundError: If a dotted path cannot be imported.
        AttributeError: If the module has no callable attribute ``name``.
    """
    mod = (
        importlib.import_module(composite_module)
        if isinstance(composite_module, str)
        else composite_module
    )
    fn = getattr(mod, name, None)
    if fn is None or not callable(fn):
        raise AttributeError(
            f"composite module {getattr(mod, '__name__', mod)!r} has no callable "
            f"bare-string measure {name!r}"
        )
    return fn


def grade_bare_string_measure(
    reader: Any,
    composite_module: ModuleType | str,
    name: str,
    *,
    observables: list[str] | None = None,
) -> float:
    """Grade a bare-string measure: ``composite_module.name(reader.rows()) -> float``.

    THE shared grading path for bare-string measures. Both the workbench grading
    path and the engine's ``evaluate_test`` call this, so they grade from one
    ``rows`` representation and agree byte-for-byte.

    Args:
        reader: A :class:`viva_emitters.run_reader.RunReader` (anything exposing
            ``rows(observables=...) -> list[dict]``).
        composite_module: The study's composite module or its dotted import path.
        name: The bare-string measure id.
        observables: Optional subset of observables to include in ``rows`` (passed
            through to ``reader.rows``); default is all scalar observables.

    Returns:
        The measure value as a float.

    Raises:
        AttributeError: If the measure function is missing (see
            :func:`resolve_measure_fn`).
        TypeError/ValueError: If the function's return cannot be cast to float.
    """
    fn = resolve_measure_fn(composite_module, name)
    rows = reader.rows(observables=observables)
    return float(fn(rows))
