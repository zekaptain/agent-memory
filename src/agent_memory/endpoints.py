"""Shared model endpoints and the opt-in boundary for sending data off-machine."""

from __future__ import annotations

import ipaddress
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping
from typing import Any

from .db import AgentMemoryError

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
LLAMACPP_HOST = os.environ.get("LLAMACPP_HOST", "http://localhost:9931").rstrip("/")
REMOTE_ENV = "AGENT_MEMORY_JUDGE_ALLOW_REMOTE"
DEFAULT_TIMEOUT = 300.0
TIMEOUT_RETRIES = 1


class EgressError(AgentMemoryError):
    """A model request is forbidden or cannot complete."""


class HttpStatusError(EgressError):
    """An HTTP rejection with its status and elapsed request time."""

    def __init__(self, message: str, code: int, *, latency_ms: float = 0.0) -> None:
        super().__init__(message)
        self.code = code
        self.latency_ms = latency_ms


def remote_allowed() -> bool:
    return os.environ.get(REMOTE_ENV, "") == "1"


def is_loopback(url: str) -> bool:
    """Recognize literal loopback hosts without resolving DNS names."""
    try:
        host = urllib.parse.urlsplit(url).hostname
        if host == "localhost":
            return True
        address = ipaddress.ip_address(host or "")
    except ValueError:
        return False
    return (
        address in ipaddress.ip_network("127.0.0.0/8")
        if address.version == 4
        else address == ipaddress.ip_address("::1")
    )


def require_egress(url: str, what: str) -> None:
    if not is_loopback(url) and not remote_allowed():
        raise EgressError(
            f"{what} sends data to a non-loopback service. Set {REMOTE_ENV}=1 to allow it."
        )


def post_json(
    url: str,
    body: Mapping[str, Any],
    headers: Mapping[str, str] | None = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> tuple[dict[str, Any], float]:
    """POST JSON after checking egress. Retry a model-loading timeout once."""
    require_egress(url, "This model request")
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", **(headers or {})},
    )
    return _send(url, request, timeout)


def get_json(url: str, timeout: float = 30.0) -> dict[str, Any]:
    """GET a JSON object through the same egress boundary."""
    require_egress(url, "This model request")
    return _send(url, urllib.request.Request(url), timeout)[0]


def _send(
    url: str, request: urllib.request.Request, timeout: float
) -> tuple[dict[str, Any], float]:
    started = time.perf_counter()
    payload = b""
    for attempt in range(TIMEOUT_RETRIES + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = response.read()
            break
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:500]
            raise HttpStatusError(
                f"{url}: HTTP {exc.code}: {detail}",
                exc.code,
                latency_ms=(time.perf_counter() - started) * 1000,
            ) from None
        except TimeoutError:
            if attempt == TIMEOUT_RETRIES:
                raise EgressError(f"{url}: timed out after {timeout:.0f}s") from None
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, TimeoutError) and attempt < TIMEOUT_RETRIES:
                continue
            raise EgressError(f"{url}: {exc}") from None
        except OSError as exc:
            raise EgressError(f"{url}: {exc}") from None
    latency = (time.perf_counter() - started) * 1000
    try:
        parsed = json.loads(payload)
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise EgressError(f"{url}: response is not JSON") from None
    if not isinstance(parsed, dict):
        raise EgressError(f"{url}: expected a JSON object")
    return parsed, latency
