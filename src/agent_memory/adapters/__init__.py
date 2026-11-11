"""Harness adapters, keyed by CLI name and by ledger agent id."""

from __future__ import annotations

from .base import Adapter, UnsupportedTranscript
from .droid import DroidAdapter

_ADAPTERS: dict[str, type[DroidAdapter]] = {
    "droid": DroidAdapter,
    "factory-droid": DroidAdapter,
}

HARNESS_NAMES = ("droid",)


def get_adapter(name: str, *, record_pre_tool: bool = False) -> Adapter:
    try:
        cls = _ADAPTERS[name.lower()]
    except KeyError:
        known = ", ".join(HARNESS_NAMES)
        raise ValueError(f"no adapter for harness {name!r}; there is one for: {known}") from None
    return cls(record_pre_tool=record_pre_tool)


__all__ = ["HARNESS_NAMES", "Adapter", "UnsupportedTranscript", "get_adapter"]
