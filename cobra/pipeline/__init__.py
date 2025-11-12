"""
Pipeline utilities for percentile collection/apply workflows.

The pct_* modules were moved from cobra.overwatch to cobra.pipeline to make
them reachable from both training and tooling stacks without depending on the
overwatch package.
"""

from importlib import import_module
from types import ModuleType
from typing import Any

__all__ = (
    "pct_apply",
    "pct_collect",
    "pct_schema",
)


def __getattr__(name: str) -> ModuleType:
    if name in __all__:
        module = import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def __dir__() -> Any:
    return sorted(list(globals().keys()) + list(__all__))

