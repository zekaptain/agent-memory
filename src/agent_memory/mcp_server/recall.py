"""Recall: run one strategy and record what it returned.

Every read of memories on an agent's behalf goes through `run_recall`, so each one leaves a
`memory_recall` event, a retrieval, and one item per memory returned. How the memories are
chosen is the strategy's business; that they are recorded is not.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from ..db import MemoryClient
from ..lifecycle import maintain_corrections


class Strategy(Protocol):
    """One way of choosing and ordering memories."""

    @property
    def name(self) -> str:
        """Kept on the retrieval. Versioned, so a change of method gets a new name."""
        ...

    @property
    def query_text(self) -> str | None:
        """What was asked for, already redacted; None when nothing was."""
        ...

    @property
    def scopes(self) -> Sequence[str]: ...

    @property
    def sql(self) -> str:
        """The statement that reads the candidates. Values reach it through sql_literal."""
        ...

    @property
    def params(self) -> dict[str, Any]:
        """Kept on the retrieval: what the statement and the ranking were given."""
        ...

    @property
    def request(self) -> dict[str, Any]:
        """Payload of the `memory_recall` event."""
        ...

    def items(self, rows: Sequence[tuple]) -> list[dict[str, Any]]:
        """The rows to show, in order and cut to the limit. Each item carries at least
        memory_id, content, score, score_components and status."""
        ...


@dataclass(frozen=True)
class Recalled:
    """What one recall returned. `reason` is set, and nothing was recorded, when the ledger
    could not be reached."""

    strategy: str
    retrieval_id: str | None = None
    request_event_id: str | None = None
    candidates: int = 0
    items: list[dict[str, Any]] = field(default_factory=list)
    reason: str | None = None


def run_recall(
    client: MemoryClient,
    strategy: Strategy,
    request_event: Callable[[dict[str, Any]], dict[str, Any]],
) -> Recalled:
    """Read with the strategy, then record the request, the retrieval and its items in one
    write. A recall that finds nothing is recorded the same way.

    `request_event` builds the `memory_recall` event row from the strategy's request; the
    caller owns the session and the redaction of what was asked."""
    maintain_corrections(client)
    started = time.perf_counter()
    sql = strategy.sql
    _, rows = client.query(sql)
    latency_ms = (time.perf_counter() - started) * 1000
    items = strategy.items(rows)
    event = request_event(strategy.request)
    retrieval_id = client.new_ids()[0]
    client.insert(
        {
            "events": [event],
            "retrievals": [
                {
                    "retrieval_id": retrieval_id,
                    "session_id": event["session_id"],
                    "request_event_id": event["event_id"],
                    "query_text": strategy.query_text,
                    "scopes": strategy.scopes,
                    "strategy": strategy.name,
                    "sql_used": sql,
                    "params": strategy.params,
                    "candidates": len(rows),
                    "returned": len(items),
                    "token_estimate": sum(len(i["content"]) // 4 for i in items),
                    "latency_ms": latency_ms,
                }
            ],
            "retrieval_items": [
                {
                    "retrieval_id": retrieval_id,
                    "rank": rank,
                    "memory_id": item["memory_id"],
                    "score": item["score"],
                    "score_components": item["score_components"],
                    "injected": True,
                    "status_at_recall": item["status"],
                }
                for rank, item in enumerate(items, start=1)
            ],
        }
    )
    return Recalled(
        strategy=strategy.name,
        retrieval_id=retrieval_id,
        request_event_id=event["event_id"],
        candidates=len(rows),
        # The status is kept on the retrieval item; every memory shown was active.
        items=[{k: v for k, v in item.items() if k != "status"} for item in items],
    )
