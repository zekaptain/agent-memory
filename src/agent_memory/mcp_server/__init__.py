"""mem-mcp: the stdio MCP server harnesses attach to for remember/recall/cite/review."""

from __future__ import annotations

import os
import sys
from collections.abc import Sequence
from typing import Any

from mcp.server.mcpserver import MCPServer

from ..config import ConfigError, load_config
from .tools import MemoryTools, ToolError

INSTRUCTIONS = """Shared, auditable memory across coding agents.
Call recall before answering questions about the user, this project's history, or past
decisions, and pass its retrieval_id to cite when you answer. Call remember for durable
facts, decisions, preferences, and procedures (one statement per call; name supersedes
when correcting an older memory), not for transient task state. Use review to retire a
wrong memory or flag a contradiction.
Call relate to record how two named things connect (a person owns a service, a module
depends on a library, a decision replaced an approach): one subject, predicate, object per
call, with the entity names as they appear in the source. Call graph to walk those
connections from an entity when a question spans several facts, and cite its retrieval_id
too."""


def build_server(tools: MemoryTools) -> MCPServer:
    server = MCPServer(name="agent-memory", instructions=INSTRUCTIONS)

    @server.tool()
    def recall(
        query: str,
        scopes: list[str] | None = None,
        limit: int = 8,
        types: list[str] | None = None,
    ) -> dict[str, Any]:
        """Search memory. Returns ranked active memories with a retrieval_id for cite.
        query: free text; empty returns the most important recent memories.
        scopes: default ["global", <project>]. types: fact, preference, decision, procedure,
        observation, profile. Every call is logged with what was returned."""
        return tools.recall(query, scopes, limit, types)

    @server.tool()
    def remember(
        content: str,
        type: str,
        scope: str | None = None,
        key: str | None = None,
        confidence: float = 0.7,
        importance: float = 0.5,
        trust_level: str = "agent",
        supersedes: str | None = None,
        source_event_ids: list[str] | None = None,
        valid_from: str | None = None,
        expires_at: str | None = None,
        valid_to: str | None = None,
    ) -> dict[str, Any]:
        """Store one durable memory: a fact, preference, decision, procedure, observation,
        or profile attribute. Not for transient task state. trust_level: user (the user said
        it), agent (you inferred it), tool (came from tool output; starts quarantined).
        Pass supersedes=<memory_id> when correcting an existing memory. Identical content
        in the same scope is confirmed rather than duplicated."""
        return tools.remember(
            content,
            type,
            scope,
            key,
            confidence,
            importance,
            trust_level,
            supersedes,
            source_event_ids,
            valid_from,
            expires_at,
            valid_to,
        )

    @server.tool()
    def cite(
        retrieval_id: str | None = None,
        memory_ids: list[str] | None = None,
        context_event_ids: list[str] | None = None,
        answer_event_id: str | None = None,
    ) -> dict[str, Any]:
        """Record what an answer was based on: the recall it used and the memories it relied
        on. Call it when you answer using recalled memory. Without answer_event_id a
        placeholder answer event is created and returned."""
        return tools.cite(retrieval_id, memory_ids, context_event_ids, answer_event_id)

    @server.tool()
    def review(
        action: str,
        memory_id: str,
        reason: str,
        related_memory_id: str | None = None,
    ) -> dict[str, Any]:
        """Change a memory's standing. action: retire (wrong or obsolete), quarantine (hide
        until reviewed), approve (release from quarantine), confirm (still true),
        flag_contradiction (conflicts with related_memory_id). reason is required."""
        return tools.review(action, memory_id, reason, related_memory_id)

    @server.tool()
    def relate(
        subject: str,
        predicate: str,
        object: str,
        subject_type: str | None = None,
        object_type: str | None = None,
        content: str | None = None,
        scope: str | None = None,
        confidence: float = 0.7,
        importance: float = 0.5,
        trust_level: str = "agent",
        source_event_ids: list[str] | None = None,
        valid_from: str | None = None,
        expires_at: str | None = None,
        valid_to: str | None = None,
    ) -> dict[str, Any]:
        """Record one relation between two named entities as a fact memory:
        subject --[predicate]--> object, for example ("payments-api", "owned by", "Dana").
        Use short verb-phrase predicates and the entity names as written in the source.
        subject_type and object_type are optional labels such as PERSON, SERVICE, FILE,
        LIBRARY, DECISION. content is an optional fuller sentence; the default is the
        triple itself. The same triple in the same scope is confirmed, not duplicated."""
        return tools.relate(
            subject,
            predicate,
            object,
            subject_type,
            object_type,
            content,
            scope,
            confidence,
            importance,
            trust_level,
            source_event_ids,
            valid_from,
            expires_at,
            valid_to,
        )

    @server.tool()
    def graph(
        entity: str,
        hops: int = 1,
        scopes: list[str] | None = None,
        limit: int = 25,
        types: list[str] | None = None,
    ) -> dict[str, Any]:
        """Walk recorded relations outward from an entity. Returns the edges within hops
        (1 to 3) as items and as "(subject) --[predicate]--> (object)" triples you can
        reason over, plus a retrieval_id for cite. Entity names match case- and
        punctuation-insensitively. Use it for questions that chain facts: who owns the
        thing that depends on X, what replaced the approach a decision retired."""
        return tools.graph(entity, hops, scopes, limit, types)

    @server.tool()
    def status() -> dict[str, Any]:
        """Whether the memory server is reachable, this session's id, how much is waiting
        in the spool, and how many records the ledger rejected."""
        return tools.status()

    return server


def tools_from_env() -> MemoryTools:
    agent = os.environ.get("AGENT_MEMORY_AGENT")
    if not agent:
        raise ConfigError("AGENT_MEMORY_AGENT must name the harness (for example factory-droid).")
    cfg = load_config()
    return MemoryTools(
        cfg,
        agent,
        project=os.environ.get("AGENT_MEMORY_PROJECT") or None,
    )


def main(argv: Sequence[str] | None = None) -> int:
    try:
        tools = tools_from_env()
    except ConfigError as exc:
        print(f"mem-mcp: {exc}", file=sys.stderr)
        return 1
    server = build_server(tools)
    try:
        server.run(transport="stdio")
    finally:
        tools.close()
    return 0


__all__ = ["MemoryTools", "ToolError", "build_server", "main"]
