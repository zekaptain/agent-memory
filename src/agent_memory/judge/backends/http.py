"""Compatibility seam for judge HTTP mocks and judge-specific errors."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ... import endpoints
from ..store import JudgeError

DEFAULT_TIMEOUT = endpoints.DEFAULT_TIMEOUT


class HttpStatusError(JudgeError):
    """An HTTP rejection, distinguishable from transport and reply faults."""

    def __init__(self, message: str, code: int, *, latency_ms: float = 0.0) -> None:
        super().__init__(message)
        self.code = code
        self.latency_ms = latency_ms


def post_json(
    url: str,
    body: Mapping[str, Any],
    headers: Mapping[str, str] | None = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> tuple[dict[str, Any], float]:
    try:
        return endpoints.post_json(url, body, headers=headers, timeout=timeout)
    except endpoints.HttpStatusError as exc:
        raise HttpStatusError(str(exc), exc.code, latency_ms=exc.latency_ms) from None
    except endpoints.EgressError as exc:
        raise JudgeError(str(exc)) from None


def get_json(url: str, timeout: float = 30.0) -> dict[str, Any]:
    try:
        return endpoints.get_json(url, timeout=timeout)
    except endpoints.HttpStatusError as exc:
        raise HttpStatusError(str(exc), exc.code, latency_ms=exc.latency_ms) from None
    except endpoints.EgressError as exc:
        raise JudgeError(str(exc)) from None
