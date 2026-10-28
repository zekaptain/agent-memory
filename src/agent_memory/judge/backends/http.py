"""Minimal JSON-over-HTTP helper. Standard library only; tests replace `post_json`."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from collections.abc import Mapping
from typing import Any

from ..store import JudgeError

DEFAULT_TIMEOUT = 300.0
TIMEOUT_RETRIES = 1


def post_json(
    url: str,
    body: Mapping[str, Any],
    headers: Mapping[str, str] | None = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> tuple[dict[str, Any], float]:
    """POST a JSON body and return (parsed JSON, latency in ms). A timeout is retried once;
    a local model server that is swapping models in is the usual cause."""
    data = json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json", **(headers or {})}
    )
    started = time.perf_counter()
    payload = b""
    for attempt in range(TIMEOUT_RETRIES + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = response.read()
            break
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:500]
            raise JudgeError(f"{url}: HTTP {exc.code}: {detail}") from None
        except TimeoutError:
            if attempt == TIMEOUT_RETRIES:
                raise JudgeError(f"{url}: timed out after {timeout:.0f}s") from None
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, TimeoutError) and attempt < TIMEOUT_RETRIES:
                continue
            raise JudgeError(f"{url}: {exc}") from None
        except OSError as exc:
            raise JudgeError(f"{url}: {exc}") from None
    latency = (time.perf_counter() - started) * 1000
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise JudgeError(f"{url}: response is not JSON: {exc}") from None
    if not isinstance(parsed, dict):
        raise JudgeError(f"{url}: expected a JSON object")
    return parsed, latency
