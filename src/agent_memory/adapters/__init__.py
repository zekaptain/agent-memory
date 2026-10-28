"""Harness adapters, keyed by CLI name and by ledger agent id."""

from __future__ import annotations

from .base import Adapter, UnsupportedTranscript
from .droid import DroidAdapter
from .vibe import VibeAdapter

_ADAPTERS: dict[str, type[DroidAdapter] | type[VibeAdapter]] = {
    "droid": DroidAdapter,
    "factory-droid": DroidAdapter,
    "vibe": VibeAdapter,
    "mistral-vibe": VibeAdapter,
}

HARNESS_NAMES = ("droid", "vibe")


def get_adapter(name: str, *, record_pre_tool: bool = False) -> Adapter:
    try:
        cls = _ADAPTERS[name.lower()]
    except KeyError:
        choices = ", ".join(HARNESS_NAMES)
        raise ValueError(f"unknown harness {name!r}; choose one of {choices}") from None
    return cls(record_pre_tool=record_pre_tool)


__all__ = ["HARNESS_NAMES", "Adapter", "UnsupportedTranscript", "get_adapter"]
