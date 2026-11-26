"""mem-mcp: the stdio MCP server harnesses attach to for remember/recall/cite/review."""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from mcp.server.mcpserver import MCPServer

from ..config import ConfigError, load_config
from ..lifecycle import HUMAN_AGENT
from .tools import MemoryTools, ToolError

MCP_WORDING_ENV = "AGENT_MEMORY_MCP_WORDING_FILE"

TOOL_NAMES: tuple[str, ...] = (
    "recall",
    "remember",
    "cite",
    "review",
    "relate",
    "graph",
    "status",
)

INSTRUCTIONS = """Shared, auditable memory across coding agents.
Call recall before answering questions about the user, this project's history, or past
decisions, and pass its retrieval_id to cite when you answer. Call remember for durable
facts, decisions, preferences, and procedures (one statement per call; name supersedes
when correcting an older memory), not for transient task state. Use review to retire a
wrong memory or flag a contradiction. A memory stored with trust_level tool, or one whose
content holds invisible or direction-changing characters or instruction-like text (whatever
its trust_level), starts quarantined and is not recalled until the user approves it with
`mem review`; you cannot approve one yourself.
Call relate to record how two named things connect (a person owns a service, a module
depends on a library, a decision replaced an approach): one subject, predicate, object per
call, with the entity names as they appear in the source. Call graph to walk those
connections from an entity when a question spans several facts, and cite its retrieval_id
too."""

DEFAULT_TOOL_DESCRIPTIONS: dict[str, str] = {
    "recall": (
        "Search memory. Returns ranked active memories with a retrieval_id for cite.\n"
        "query: free text; empty returns the most important recent memories.\n"
        'scopes: default ["global", <project>]. types: fact, preference, decision, procedure,\n'
        "observation, profile. Every call is logged with what was returned."
    ),
    "remember": (
        "Store one durable memory: a fact, preference, decision, procedure, observation,\n"
        "or profile attribute. Not for transient task state. trust_level: user (the user said\n"
        "it), agent (you inferred it), tool (came from tool output; quarantined until the user\n"
        "approves it). Content with invisible or direction-changing characters or\n"
        "instruction-like text is also written quarantined regardless of trust_level and waits\n"
        "for the user to approve it with `mem review`.\n"
        "Pass supersedes=<memory_id> when correcting an existing memory. Identical content\n"
        "in the same scope is confirmed rather than duplicated."
    ),
    "cite": (
        "Record what an answer was based on: the recall it used and the memories it relied\n"
        "on. Call it when you answer using recalled memory. Without answer_event_id a\n"
        "placeholder answer event is created and returned."
    ),
    "review": (
        "Change a memory's standing. action: retire (wrong or obsolete), quarantine (hide\n"
        "until the user reviews it), confirm (still true), flag_contradiction (conflicts with\n"
        "related_memory_id). reason is required. You cannot approve a quarantined memory:\n"
        "only the user can, with `mem review`."
    ),
    "relate": (
        "Record one relation between two named entities as a fact memory:\n"
        'subject --[predicate]--> object, for example ("payments-api", "owned by", "Dana").\n'
        "Use short verb-phrase predicates and the entity names as written in the source.\n"
        "subject_type and object_type are optional labels such as PERSON, SERVICE, FILE,\n"
        "LIBRARY, DECISION. content is an optional fuller sentence; the default is the\n"
        "triple itself. The same triple in the same scope is confirmed, not duplicated."
    ),
    "graph": (
        "Walk recorded relations outward from an entity. Returns the edges within hops\n"
        '(1 to 3) as items and as "(subject) --[predicate]--> (object)" triples you can\n'
        "reason over, plus a retrieval_id for cite. Entity names match case- and\n"
        "punctuation-insensitively. Use it for questions that chain facts: who owns the\n"
        "thing that depends on X, what replaced the approach a decision retired."
    ),
    "status": (
        "Whether the memory server is reachable, this session's id, how much is waiting\n"
        "in the spool, and how many records the ledger rejected."
    ),
}


def default_tool_wording() -> dict[str, Any]:
    """Return the installed default MCP instructions and per-tool descriptions."""
    return {
        "instructions": INSTRUCTIONS,
        "descriptions": dict(DEFAULT_TOOL_DESCRIPTIONS),
    }


def validate_tool_wording(data: Mapping[str, Any]) -> tuple[str, dict[str, str]]:
    """Validate an MCP wording mapping and return ``(instructions, descriptions)``."""
    if not isinstance(data, Mapping):
        raise ConfigError("memory tool wording must be a JSON object")
    raw_instructions = data.get("instructions")
    if not isinstance(raw_instructions, str) or not raw_instructions.strip():
        raise ConfigError("memory tool wording is missing non-empty 'instructions'")
    raw_descriptions = data.get("descriptions")
    if not isinstance(raw_descriptions, Mapping):
        raise ConfigError("memory tool wording is missing 'descriptions' mapping")

    expected = set(TOOL_NAMES)
    provided = set(raw_descriptions.keys())
    missing = sorted(expected - provided)
    extra = sorted(provided - expected)
    if missing:
        raise ConfigError(
            f"memory tool wording is missing descriptions for tool(s): {', '.join(missing)}"
        )
    if extra:
        raise ConfigError(f"memory tool wording names unknown tool(s): {', '.join(extra)}")

    descriptions: dict[str, str] = {}
    for name in TOOL_NAMES:
        val = raw_descriptions[name]
        if not isinstance(val, str) or not val.strip():
            raise ConfigError(f"memory tool wording description for {name!r} must be non-empty")
        descriptions[name] = val
    return (raw_instructions, descriptions)


def load_tool_wording_file(path: Path) -> tuple[str, dict[str, str]]:
    """Read and validate registered MCP instructions and tool descriptions from ``path``."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConfigError(f"cannot read memory tool wording file {path}: {exc}") from None
    except json.JSONDecodeError as exc:
        raise ConfigError(f"invalid JSON in memory tool wording file {path}: {exc}") from None
    return validate_tool_wording(raw)


def build_server(
    tools: MemoryTools,
    *,
    instructions: str | None = None,
    tool_descriptions: Mapping[str, str] | None = None,
) -> MCPServer:
    effective_instructions = INSTRUCTIONS if instructions is None else instructions
    if not isinstance(effective_instructions, str) or not effective_instructions.strip():
        raise ConfigError("MCP server instructions must be a non-empty string")

    if tool_descriptions is None:
        descriptions = dict(DEFAULT_TOOL_DESCRIPTIONS)
    else:
        _, descriptions = validate_tool_wording(
            {"instructions": effective_instructions, "descriptions": tool_descriptions}
        )

    server = MCPServer(name="agent-memory", instructions=effective_instructions)

    @server.tool(description=descriptions["recall"])
    def recall(
        query: str,
        scopes: list[str] | None = None,
        limit: int = 8,
        types: list[str] | None = None,
    ) -> dict[str, Any]:
        return tools.recall(query=query, scopes=scopes, limit=limit, types=types)

    @server.tool(description=descriptions["remember"])
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
        return tools.remember(
            content=content,
            memory_type=type,
            scope=scope,
            key=key,
            confidence=confidence,
            importance=importance,
            trust_level=trust_level,
            supersedes=supersedes,
            source_event_ids=source_event_ids,
            valid_from=valid_from,
            expires_at=expires_at,
            valid_to=valid_to,
        )

    @server.tool(description=descriptions["cite"])
    def cite(
        retrieval_id: str | None = None,
        memory_ids: list[str] | None = None,
        context_event_ids: list[str] | None = None,
        answer_event_id: str | None = None,
    ) -> dict[str, Any]:
        return tools.cite(
            retrieval_id=retrieval_id,
            memory_ids=memory_ids,
            context_event_ids=context_event_ids,
            answer_event_id=answer_event_id,
        )

    @server.tool(description=descriptions["review"])
    def review(
        action: str,
        memory_id: str,
        reason: str,
        related_memory_id: str | None = None,
    ) -> dict[str, Any]:
        return tools.review(
            action=action,
            memory_id=memory_id,
            reason=reason,
            related_memory_id=related_memory_id,
        )

    @server.tool(description=descriptions["relate"])
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
        return tools.relate(
            subject=subject,
            predicate=predicate,
            object=object,
            subject_type=subject_type,
            object_type=object_type,
            content=content,
            scope=scope,
            confidence=confidence,
            importance=importance,
            trust_level=trust_level,
            source_event_ids=source_event_ids,
            valid_from=valid_from,
            expires_at=expires_at,
            valid_to=valid_to,
        )

    @server.tool(description=descriptions["graph"])
    def graph(
        entity: str,
        hops: int = 1,
        scopes: list[str] | None = None,
        limit: int = 25,
        types: list[str] | None = None,
    ) -> dict[str, Any]:
        return tools.graph(entity=entity, hops=hops, scopes=scopes, limit=limit, types=types)

    @server.tool(description=descriptions["status"])
    def status() -> dict[str, Any]:
        return tools.status()

    return server


def tools_from_env() -> MemoryTools:
    agent = os.environ.get("AGENT_MEMORY_AGENT")
    if not agent:
        raise ConfigError("AGENT_MEMORY_AGENT must name the harness (for example factory-droid).")
    if agent == HUMAN_AGENT:
        # Whatever runs mem-mcp is an agent, and the human agent may approve memories.
        raise ConfigError(
            f"AGENT_MEMORY_AGENT={HUMAN_AGENT} is the human at the mem CLI; "
            "name the harness (for example factory-droid)."
        )
    cfg = load_config()
    return MemoryTools(
        cfg,
        agent,
        project=os.environ.get("AGENT_MEMORY_PROJECT") or None,
    )


def main() -> int:
    try:
        tools = tools_from_env()
        wording_path = os.environ.get(MCP_WORDING_ENV)
        if wording_path:
            instructions, descriptions = load_tool_wording_file(Path(wording_path))
            server = build_server(tools, instructions=instructions, tool_descriptions=descriptions)
        else:
            server = build_server(tools)
    except ConfigError as exc:
        print(f"mem-mcp: {exc}", file=sys.stderr)
        return 1
    try:
        server.run(transport="stdio")
    finally:
        tools.close()
    return 0


__all__ = [
    "DEFAULT_TOOL_DESCRIPTIONS",
    "INSTRUCTIONS",
    "MCP_WORDING_ENV",
    "TOOL_NAMES",
    "MemoryTools",
    "ToolError",
    "build_server",
    "default_tool_wording",
    "load_tool_wording_file",
    "main",
    "validate_tool_wording",
]
