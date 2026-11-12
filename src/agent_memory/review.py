"""`mem review`: the memories awaiting approval, and a human's approval or retirement of
one. Registered on the main app from cli.py."""

from __future__ import annotations

import sys
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Annotated, Any

import typer

from .capture.session import Observed
from .config import Config, command_config
from .db import AgentMemoryError, MemoryClient, sql_literal
from .lifecycle import HUMAN_AGENT, LifecyclePlan
from .mcp_server.tools import MemoryTools

review_app = typer.Typer(
    help="The memories awaiting approval, and a human's approval or retirement of one.",
    no_args_is_help=True,
)

COMMAND = Observed("mem review", "record", "mem_cli_v1")
DONE = {"approve": "approved", "retire": "retired"}


class ReviewError(AgentMemoryError):
    """`mem review` did not do what was asked."""


@dataclass(frozen=True)
class Waiting:
    """A memory whose standing an approval would change: one in quarantine, or an active one
    that a tool's output asked to put in another memory's place."""

    memory_id: str
    scope: str
    type: str
    trust_level: str
    content: str
    created_at: datetime
    written_by: str
    status: str
    pending_reason: str | None
    quarantined_by: str | None
    quarantine_reason: str | None
    # The memories an approval would supersede, each with its content.
    replaces: tuple[tuple[str, str], ...]

    @property
    def why(self) -> str:
        if self.status != "quarantined":
            return "a tool's output asked for it to replace another memory"
        if self.quarantined_by:
            reason = self.quarantine_reason or "no reason given"
            return f"{self.quarantined_by} quarantined it: {reason}"
        return "it came from a tool's output and was never approved"


def waiting(
    client: MemoryClient, *, scopes: list[str] | None = None, memory_id: str | None = None
) -> list[Waiting]:
    """The memories awaiting approval, oldest first. It is not a recall: none of it is something an
    agent may be shown, and nothing is recorded about the reading."""
    filters = ""
    if scopes:
        filters += f" AND m.scope IN ({', '.join(sql_literal(s) for s in scopes)})"
    if memory_id:
        filters += f" AND m.memory_id = {sql_literal(memory_id)}"
    _, rows = client.query(
        "SELECT m.memory_id::VARCHAR, m.scope, m.type, m.trust_level, m.content, m.created_at, "
        "coalesce(s.agent_id, 'unknown'), st.status, st.pending_reason, a.actor, a.reason, "
        "p.predecessor_id::VARCHAR, old.content "
        "FROM memories m "
        "JOIN memory_status st ON st.memory_id = m.memory_id "
        "JOIN memory_eligibility e ON e.memory_id = m.memory_id "
        "LEFT JOIN memory_actions a "
        "ON a.action_id = e.review_id AND e.last_review = 'quarantine' "
        "LEFT JOIN sessions s ON s.session_id = m.created_by_session "
        "LEFT JOIN memory_proposal_status p ON p.successor_id = m.memory_id "
        "AND p.correction_status = 'pending' AND p.pending_reason = 'awaiting_approval' "
        "LEFT JOIN memories old ON old.memory_id = p.predecessor_id "
        f"WHERE (st.status = 'quarantined' OR p.proposal_id IS NOT NULL){filters} "
        "ORDER BY m.created_at, m.memory_id, p.predecessor_id"
    )
    found: dict[str, Waiting] = {}
    for *memory, predecessor, replaced in rows:
        current = found.setdefault(memory[0], Waiting(*memory, replaces=()))
        if predecessor and predecessor not in dict(current.replaces):
            found[memory[0]] = Waiting(
                *memory, replaces=(*current.replaces, (predecessor, replaced))
            )
    return list(found.values())


def reviewer(cfg: Config) -> MemoryTools:
    """What a human's review action is written with: the human agent, in a session of its own."""
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return MemoryTools(cfg, HUMAN_AGENT, harness_session_id=f"review-{stamp}", observed=COMMAND)


def _memory_id(value: str) -> str:
    try:
        return str(uuid.UUID(value))
    except ValueError:
        raise typer.BadParameter("give the memory's id, as `mem review list` prints it") from None


def _reason(value: str) -> str:
    if not value.strip():
        raise typer.BadParameter("say why")
    return value


SCOPES = Annotated[
    list[str] | None,
    typer.Option("--scope", help="Scope to read; repeatable. Default: every scope."),
]
MEMORY_ID = Annotated[str, typer.Argument(callback=_memory_id, help="The memory's id.")]
REASON = Annotated[
    str, typer.Option(callback=_reason, help="Why. It is kept with the review action, redacted.")
]


def _show(memory: Waiting) -> None:
    when = memory.created_at.astimezone().strftime("%Y-%m-%d %H:%M")
    print(memory.memory_id)
    print(
        f"  {memory.type} in scope {memory.scope}, {memory.trust_level} trust, "
        f"written by {memory.written_by} on {when}"
    )
    print(f"  waits because {memory.why}")
    _quote(memory.content)
    for predecessor, content in memory.replaces:
        print(f"  approving it replaces {predecessor}:")
        _quote(content)
    if memory.pending_reason == "stale_predecessor":
        print("  the memory it corrects was already replaced, so approval will not activate it")


def _quote(content: str) -> None:
    for line in content.splitlines():
        print(f"    {line}")


def _decide(cfg: Config, action: str, memory_id: str, reason: str) -> None:
    tools = reviewer(cfg)
    try:
        result: dict[str, Any] = tools.review(action, memory_id, reason)
    finally:
        tools.close()
    if result["queued"]:
        print(
            f"the server did not answer: {DONE[action]} {memory_id} once the spool is replayed "
            "(`mem ingest --spool`)"
        )
        return
    pending = f" ({result['pending_reason']})" if result["pending_reason"] else ""
    print(f"{DONE[action]} {result['resolved_memory_id']}: now {result['status']}{pending}")


@review_app.command("list")
def list_waiting(ctx: typer.Context, scope: SCOPES = None) -> None:
    """Every memory awaiting approval, oldest first: those in quarantine, and active ones a
    tool's output asked to put in another memory's place."""
    with MemoryClient.from_config(command_config(ctx)) as client:
        found = waiting(client, scopes=scope)
    if not found:
        print("nothing awaits approval")
        return
    for memory in found:
        _show(memory)
        print()
    count = len(found)
    print(f"{count} {'memory awaits' if count == 1 else 'memories await'} approval")


@review_app.command()
def approve(ctx: typer.Context, memory_id: MEMORY_ID, reason: REASON) -> None:
    """Let a memory awaiting approval be recalled. It shows the memory and asks for a yes
    typed at a terminal: an approval is a human's, so a yes that is piped in is refused."""
    if not sys.stdin.isatty():
        raise ReviewError(
            "approving a memory takes a human at a terminal; run `mem review approve` yourself"
        )
    cfg = command_config(ctx)
    with MemoryClient.from_config(cfg) as client:
        # A duplicate's id stands for its canonical, as it does in every review action.
        target = LifecyclePlan(client).resolve(memory_id)
        found = waiting(client, memory_id=target)
        if not found:
            _, rows = client.query(
                f"SELECT status FROM memory_status WHERE memory_id = {sql_literal(target)}"
            )
            raise ReviewError(
                f"memory {target} is {rows[0][0]}; there is nothing to approve"
                if rows
                else f"no memory {target}"
            )
    _show(found[0])
    print("Approve this memory? [y/N] ", end="", flush=True)
    if sys.stdin.readline().strip().lower() not in ("y", "yes"):
        raise ReviewError("not approved")
    _decide(cfg, "approve", target, reason)


@review_app.command()
def retire(ctx: typer.Context, memory_id: MEMORY_ID, reason: REASON) -> None:
    """Withdraw a memory for good, whether or not it awaits approval. Retirement is not undone."""
    _decide(command_config(ctx), "retire", memory_id, reason)
