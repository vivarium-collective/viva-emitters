"""Per-agent emitter-lifecycle management for process-bigraph composites.

In a multi-generation / dividing-cell composite, an emitter (e.g. a
:class:`~viva_emitters.parquet_emitter.ParquetEmitter`) is constructed deep
inside a composite's step factory — the driver and the division step never
see the instance, so neither can call ``close()`` on it. When a cell divides
its agent subtree is torn down (e.g. via a ``{'agents': {'_remove': [...]}}``
update); without an explicit ``close(success=True)`` the parent emitter's
trailing partial batch (rows buffered since the last batch flush) is lost and
no success sentinel is written.

This module provides a small, generic registry keyed by ``agent_id`` so the
step that detects division can look the parent emitter up and finalize it
before the subtree is removed. It mirrors the pre-divide
``self.emitter.finalize()`` hook that workflow runners (e.g. vEcoli's
``ecoli_master_sim.py``) call when they own the emitter directly.

The registry is intentionally framework-generic: it stores any object and
calls ``close(success=...)`` on it (duck-typed). It carries no knowledge of
any particular composite, simulator, or partition layout — callers supply
the ``agent_id`` key and the domain-specific glue (deciding *when* to
finalize, re-pointing daughter partitions, etc.) stays in the caller.

The registry is module-global (process-scoped); a single composite run lives
in one process, so this matches the lifetime of the emitters it tracks.
Helpers are not thread-safe — register / finalize from the composite-driving
thread.
"""

from __future__ import annotations

from typing import Any, Optional

# agent_id -> live emitter instance. Populated when a composite's step factory
# constructs an emitter; consulted by the division/teardown step.
_EMITTERS_BY_AGENT: dict[str, Any] = {}


def register_emitter(agent_id: str, emitter: Any) -> None:
    """Track a live emitter under ``agent_id`` so it can be finalized later.

    A later ``register_emitter`` for the same ``agent_id`` overwrites the
    previous entry (the latest emitter on that slot wins).
    """
    _EMITTERS_BY_AGENT[str(agent_id)] = emitter


def get_emitter(agent_id: str) -> Optional[Any]:
    """Return the emitter registered for ``agent_id``, or ``None``."""
    return _EMITTERS_BY_AGENT.get(str(agent_id))


def unregister_emitter(agent_id: str) -> Optional[Any]:
    """Drop and return the emitter registered for ``agent_id`` (or ``None``)."""
    return _EMITTERS_BY_AGENT.pop(str(agent_id), None)


def clear_registry() -> None:
    """Forget all registered emitters (e.g. between independent runs/tests)."""
    _EMITTERS_BY_AGENT.clear()


def registered_agent_ids() -> list[str]:
    """Return the agent_ids currently registered (snapshot, for inspection)."""
    return list(_EMITTERS_BY_AGENT)


def finalize_emitter_for_agent(agent_id: str, success: bool = True) -> bool:
    """Close the emitter registered for ``agent_id`` and unregister it.

    This is the pre-teardown hook: it flushes the emitter's trailing partial
    batch and (for a partitioned emitter with ``success=True``) writes the
    success sentinel, then removes it from the registry so it is finalized at
    most once.

    The emitter is duck-typed: any object exposing ``close(success=...)``
    works. ``close`` is expected to be idempotent (the ParquetEmitter's is).

    Returns ``True`` if an emitter was found and finalized, ``False`` if no
    emitter was registered for ``agent_id``.
    """
    emitter = _EMITTERS_BY_AGENT.get(str(agent_id))
    if emitter is None:
        return False
    try:
        emitter.close(success=success)
    finally:
        # Unregister even if close() raised, so a broken emitter isn't
        # retried indefinitely on every teardown.
        _EMITTERS_BY_AGENT.pop(str(agent_id), None)
    return True
