
"""
Debugging, warnings & errors.

Also provides vivarium.core.types / vivarium.library.topology shims so that
vendored sub-modules do not need vivarium installed.
"""


from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, astuple
from itertools import starmap
from typing import Any, Literal
from warnings import filterwarnings, catch_warnings


# ---------------------------------------------------------------------------
# Vivarium shims — avoids pulling in vivarium as a dependency
# ---------------------------------------------------------------------------

# HierarchyPath mirrors vivarium.core.types.HierarchyPath
HierarchyPath = tuple[str, ...]


def get_in(d: Any, path: tuple, default: Any = None) -> Any:
    """Get the value from a nested dict by its path.

    Mirrors vivarium.library.topology.get_in.
    """
    if path:
        head = path[0]
        if isinstance(d, dict) and head in d:
            return get_in(d[head], path[1:], default)
        return default
    return d


def dict_to_paths(root: tuple, d: Any) -> list[tuple[tuple, Any]]:
    """Return all leaf (path, value) pairs from a nested dict.

    Mirrors vivarium.library.topology.dict_to_paths.
    """
    if isinstance(d, dict):
        deeper: list = []
        for key, down in d.items():
            deeper.extend(dict_to_paths(root + (key,), down))
        return deeper
    else:
        return [(root, d)]


# ==============================================================================


def indent(s: int, obj: object):
    return f"\n{s*" "}".join([""] + repr(obj).split("\n"))


# ------------------------------------------------------------------------------


def emitter_arg_error(obj: object | type, msg: str, args: str, /) -> str:
    return (f"\n  {msg} for "
            f"{(obj if isinstance(obj, type) else obj.__class__).__name__}:"
            f"\n    {{\"emitter_arg\": {{{args}}}}}")


# ------------------------------------------------------------------------------


@dataclass(kw_only=True, slots=True, frozen=True)
class WarningFilter:
    """
    Data class for specifying warning filters. Passed to
    :py:func:`warnings.filterwarnings` as a tuple, and to
    :py:func:`pytest.mark.filterwarnings` as a string.
    """

    action: Literal["ignore", "error"]
    message: str
    category: type[Warning]
    module: str

    def __post_init__(self) -> None:
        assert issubclass(self.category, Warning)

    def __str__(self) -> str:
        wtyp = self.category
        wmod = "" if ((wmod := wtyp.__module__) == "builtins") else f"{wmod}."
        w = f"{wmod}{wtyp.__name__}"
        return f"{self.action}:{self.message}:{w}:{self.module}"


@contextmanager
def filter_warnings(filters: list[WarningFilter]) -> Iterator[None]:
    """
    Context manager for activating a collection of warning filters using
    :py:func:`warnings.filterwarnings`.
    """
    with catch_warnings():
        list(starmap(filterwarnings, map(astuple, filters)))
        yield None
