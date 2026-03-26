"""
Lightweight package entrypoint for the Aletheia codebase.

Import concrete submodules explicitly to avoid loading the entire training
stack as a side effect of importing the package itself.
"""

from __future__ import annotations

from importlib import import_module
from types import ModuleType

__all__ = [
    "aletheia_actor_critic",
    "aletheia_api",
    "aletheia_config",
    "aletheia_foundation",
    "aletheia_train",
    "aletheia_twohot",
    "aletheia_world_model",
    "contracts",
    "training",
]


def __getattr__(name: str) -> ModuleType:
    if name in __all__:
        module = import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
