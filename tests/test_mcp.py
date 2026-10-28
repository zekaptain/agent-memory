"""mem-mcp tools against a live server, through the MCP layer, and over stdio."""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from typing import Any

import pytest
from mcp.types import CallToolResult, TextContent

from agent_memory.capture import ids
from agent_memory.capture.pipeline import replay_spool
from agent_memory.db import MemoryClient, MemoryServer, sql_literal
from agent_memory.mcp_server import MemoryTools, ToolError, build_server
from agent_memory.mcp_server.ranking import normalize_query


@pytest.fixture
def tools(live_cfg):
    t = MemoryTools(live_cfg, "jetbrains-junie", project="demo")
    yield t
    t.close()


@pytest.fixture
def droid_tools(live_cfg):
    t = MemoryTools(live_cfg, "factory-droid", harness_session_id="hooked-1", project="demo")
    yield t
    t.close()


def q(client, sql):
    return client.query(sql)[1]


def structured(result: object) -> dict[str, Any]:
    assert isinstance(result, CallToolResult)
    assert result.structured_content is not None
    return result.structured_content


# sessions


def test_own_session_is_partial_and_created_once(tools, live_client):
    tools.status()
    tools.status()
    got = q(
        live_client,
        "SELECT agent_id, capture_completeness, project, harness_session_id FROM sessions "
        f"WHERE session_id = {sql_literal(tools.session_id)}",
    )
    assert len(got) == 1
    assert got[0][:3] == ("jetbrains-junie", "partial", "demo")
    assert got[0][3].startswith("mcp-")


def test_harness_session_is_shared_with_hooks(droid_tools, live_client):
    droid_tools.status()
    assert droid_tools.session_id == ids.session_id("factory-droid", "hooked-1")
    got = q(
        live_client,
        "SELECT capture_completeness FROM sessions "
        f"WHERE session_id = {sql_literal(droid_tools.session_id)}",
    )
    assert got == [("full",)]


# remember / recall / cite / review


def test_remember_recall_cite_review_chain(tools, live_client):
    kept = tools.remember(
        "The user prefers uv over pip for Python work", "preference", trust_level="user"
    )
    assert kept["status"] == "active"
    assert not kept["deduplicated"]
    tool_memory = tools.remember(
        "pytest is configured with testpaths = tests",
        "fact",
        trust_level="tool",
        source_event_ids=[],
    )
    assert tool_memory["status"] == "quarantined"
    retired = tools.remember(
        "The user prefers pip", "preference", trust_level="agent", scope="demo"
    )
    tools.review("retire", retired["memory_id"], "superseded by the uv preference")

    recall = tools.recall("which Python package manager does the user prefer?")
    assert not recall["degraded"]
    assert recall["strategy"] == "fts_bm25_v1"
    returned = [i["memory_id"] for i in recall["items"]]
    assert kept["memory_id"] in returned
    assert tool_memory["memory_id"] not in returned  # quarantined
    assert retired["memory_id"] not in returned  # retired
    top = recall["items"][0]
    assert set(top["score_components"]) == {"bm25", "importance", "confidence", "scope_boost"}
    assert "status" not in top

    rows = q(
        live_client,
        "SELECT strategy, candidates, returned, query_text, params::JSON::VARCHAR, sql_used "
        f"FROM retrievals WHERE retrieval_id = {sql_literal(recall['retrieval_id'])}",
    )
    strategy, candidates, returned_n, query_text, params, sql_used = rows[0]
    assert (strategy, returned_n) == ("fts_bm25_v1", len(recall["items"]))
    assert candidates == recall["candidates"]
    assert query_text == "which Python package manager does the user prefer?"
    assert "normalized" in params
    assert "match_bm25" in sql_used
    items = q(
        live_client,
        "SELECT rank, memory_id::VARCHAR, injected, status_at_recall FROM retrieval_items "
        f"WHERE retrieval_id = {sql_literal(recall['retrieval_id'])} ORDER BY rank",
    )
    assert items[0] == (1, top["memory_id"], True, "active")
    assert q(
        live_client,
        "SELECT kind, content FROM events "
        f"WHERE event_id = {sql_literal(recall['request_event_id'])}",
    ) == [("memory_recall", query_text)]

    cited = tools.cite(retrieval_id=recall["retrieval_id"], memory_ids=[kept["memory_id"]])
    assert cited["links"] == 2
    why = q(
        live_client,
        "SELECT via, memory_id::VARCHAR, status_at_recall, memory_status_now, "
        "source_relation, source_kind FROM why_answer "
        f"WHERE answer_event_id = {sql_literal(cited['answer_event_id'])} ORDER BY via",
    )
    assert (
        "retrieval",
        kept["memory_id"],
        "active",
        "active",
        "user_stated",
        "memory_write",
    ) in why
    assert ("cited_memory", kept["memory_id"], None, "active", "user_stated", "memory_write") in why
    placeholder = q(
        live_client,
        "SELECT kind, role, content, payload::JSON::VARCHAR FROM events "
        f"WHERE event_id = {sql_literal(cited['answer_event_id'])}",
    )[0]
    assert placeholder[:3] == ("assistant_msg", "assistant", None)
    assert '"placeholder":true' in placeholder[3].replace(" ", "")

    approved = tools.review("approve", tool_memory["memory_id"], "checked pyproject.toml")
    assert approved["status"] == "active"
    assert tool_memory["memory_id"] in [
        i["memory_id"] for i in tools.recall("pytest testpaths configuration")["items"]
    ]


def test_remember_dedupes_identical_content_in_scope(tools, live_client):
    first = tools.remember("Deploys go through ops/deploy.sh", "procedure", scope="dedupe")
    second = tools.remember("Deploys go through ops/deploy.sh", "procedure", scope="dedupe")
    assert second["memory_id"] == first["memory_id"]
    assert second["deduplicated"]
    other_scope = tools.remember("Deploys go through ops/deploy.sh", "procedure", scope="elsewhere")
    assert other_scope["memory_id"] != first["memory_id"]
    actions = q(
        live_client,
        "SELECT action, reason FROM memory_actions "
        f"WHERE memory_id = {sql_literal(first['memory_id'])}",
    )
    assert actions == [("confirm", "remember() with identical content")]


def test_remember_redacts_and_links_sources(tools, live_client):
    recall = tools.recall("anything at all")
    source = recall["request_event_id"]
    stored = tools.remember(
        "CI token is ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghij0123",
        "fact",
        source_event_ids=[source, "00000000-0000-0000-0000-000000000000"],
    )
    assert stored["ignored_source_event_ids"] == ["00000000-0000-0000-0000-000000000000"]
    content = q(
        live_client,
        f"SELECT content FROM memories WHERE memory_id = {sql_literal(stored['memory_id'])}",
    )
    assert content == [("CI token is [REDACTED:github_token]",)]
    relations = q(
        live_client,
        "SELECT event_id::VARCHAR, relation FROM memory_sources "
        f"WHERE memory_id = {sql_literal(stored['memory_id'])} ORDER BY relation, event_id",
    )
    assert (source, "derived_from") in relations
    assert len(relations) == 2


def test_supersedes_marks_the_old_memory(tools, live_client):
    old = tools.remember("The API lives on port 8000", "fact", scope="ports")
    new = tools.remember(
        "The API lives on port 8080", "fact", scope="ports", supersedes=old["memory_id"]
    )
    assert new["status"] == "active"
    assert q(
        live_client,
        f"SELECT status FROM memory_status WHERE memory_id = {sql_literal(old['memory_id'])}",
    ) == [("superseded",)]


def test_empty_query_returns_recent_by_importance(tools):
    tools.remember("low importance note", "observation", scope="recent", importance=0.1)
    tools.remember("high importance rule", "decision", scope="recent", importance=0.9)
    result = tools.recall("", scopes=["recent"], limit=5)
    assert result["strategy"] == "recent_v1"
    assert [i["content"] for i in result["items"]][:2] == [
        "high importance rule",
        "low importance note",
    ]


def test_types_filter_and_limit(tools):
    tools.remember("filter me: a procedure about ducks", "procedure", scope="filters")
    tools.remember("filter me: a fact about ducks", "fact", scope="filters")
    only_facts = tools.recall("ducks", scopes=["filters"], types=["fact"])
    assert [i["type"] for i in only_facts["items"]] == ["fact"]
    assert len(tools.recall("ducks", scopes=["filters"], limit=1)["items"]) == 1


@pytest.mark.parametrize(
    "query",
    [
        "ducks'; UPDATE memories SET content = 'x'; --",
        "ducks /* comment */",
        'ducks"; DELETE FROM memories; --',
        "it’s ‘ducks’",  # noqa: RUF001
        "ducks $$ CALL quack_stop('x') $$",
    ],
)
def test_hostile_queries_are_inert_and_stored_verbatim(tools, live_client, query):
    tools.remember("ducks like ponds", "fact", scope="hostile")
    before = q(live_client, "SELECT count(*) FROM memories")[0][0]
    result = tools.recall(query, scopes=["hostile"])
    assert not result["degraded"]
    assert [i["content"] for i in result["items"]] == ["ducks like ponds"]
    assert q(live_client, "SELECT count(*) FROM memories")[0][0] == before
    stored = q(
        live_client,
        "SELECT query_text FROM retrievals "
        f"WHERE retrieval_id = {sql_literal(result['retrieval_id'])}",
    )
    assert stored == [(query,)]


def test_normalize_query():
    assert normalize_query("It’s ‘Ducks’, isn't it?") == "it s ducks isn t it"  # noqa: RUF001
    assert normalize_query("   ") == ""


# argument errors


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (lambda t: t.remember("", "fact"), "content is required"),
        (lambda t: t.remember("x", "nonsense"), "type must be one of"),
        (lambda t: t.remember("x", "fact", trust_level="imported"), "trust_level"),
        (lambda t: t.remember("x", "fact", confidence=2), "between 0 and 1"),
        (lambda t: t.remember("x", "fact", supersedes="nope"), "must be a UUID"),
        (lambda t: t.cite(), "cite needs"),
        (lambda t: t.cite(memory_ids=["nope"]), "must be a UUID"),
        (
            lambda t: t.review("bogus", "00000000-0000-0000-0000-000000000000", "r"),
            "action must be one of",
        ),
        (
            lambda t: t.review("retire", "00000000-0000-0000-0000-000000000000", ""),
            "reason is required",
        ),
    ],
)
def test_bad_arguments_raise_tool_errors(tools, call, message):
    with pytest.raises(ToolError, match=message):
        call(tools)


# MCP layer


def test_mcp_server_exposes_tools_and_keeps_error_messages(tools, live_client):
    server = build_server(tools)

    async def go():
        names = [t.name for t in await server.list_tools()]
        assert names == ["recall", "remember", "cite", "review", "status"]
        remembered = await server.call_tool(
            "remember", {"content": "MCP layer works", "type": "fact", "scope": "mcp"}
        )
        assert structured(remembered)["status"] == "active"
        recalled = await server.call_tool("recall", {"query": "MCP layer", "scopes": ["mcp"]})
        assert structured(recalled)["items"][0]["content"] == "MCP layer works"
        status = await server.call_tool("status", {})
        assert structured(status)["reachable"] is True
        from mcp.server.mcpserver.exceptions import ToolError as McpToolError

        with pytest.raises(McpToolError, match="type must be one of"):
            await server.call_tool("remember", {"content": "x", "type": "nonsense"})

    asyncio.run(go())


# degraded mode


def test_degraded_mode_queues_and_replays(down_cfg):
    tools = MemoryTools(down_cfg, "amp", project="demo")
    try:
        recall = tools.recall("anything")
        assert recall["degraded"]
        assert recall["items"] == []
        assert "Cannot connect" in recall["reason"]
        queued = tools.remember("Queued while the server was down", "observation", scope="demo")
        assert queued["queued"]
        memory_id = queued["memory_id"]
        review = tools.review("quarantine", memory_id, "double-check later")
        assert review["queued"]
        cite = tools.cite(memory_ids=[memory_id])
        assert cite["degraded"]
        status = tools.status()
        assert status["reachable"] is False
        assert status["spool_pending_lines"] == 2

        server = MemoryServer(down_cfg, export_interval_hours=0)
        server.start()
        try:
            result = replay_spool(down_cfg)
            assert result.failed_lines == 0
            with MemoryClient.from_config(down_cfg) as client:
                assert q(
                    client,
                    f"SELECT content FROM memories WHERE memory_id = {sql_literal(memory_id)}",
                ) == [("Queued while the server was down",)]
                assert q(
                    client,
                    f"SELECT status FROM memory_status WHERE memory_id = {sql_literal(memory_id)}",
                ) == [("quarantined",)]
                assert q(client, "SELECT capture_completeness FROM sessions") == [("partial",)]
            # The same MemoryTools reconnects on its own.
            assert tools.status()["reachable"] is True
            assert tools.recall("queued server down", scopes=["demo"])["degraded"] is False
        finally:
            server.stop()
    finally:
        tools.close()


# stdio entry point


def test_mem_mcp_runs_over_stdio(live_cfg):
    from mcp.client.session import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client

    env = {
        **os.environ,
        "AGENT_MEMORY_HOME": str(live_cfg.home),
        "AGENT_MEMORY_PORT": str(live_cfg.port),
        "AGENT_MEMORY_AGENT": "jetbrains-junie",
        "AGENT_MEMORY_PROJECT": "stdio",
    }
    params = StdioServerParameters(
        command=sys.executable, args=["-m", "agent_memory.mcp_server"], env=env
    )

    async def go():
        async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
            await session.initialize()
            names = sorted(t.name for t in (await session.list_tools()).tools)
            assert names == ["cite", "recall", "remember", "review", "status"]
            result = await session.call_tool("status", {})
            assert result.is_error is False
            assert result.structured_content["reachable"] is True
            assert result.structured_content["project"] == "stdio"
            bad = await session.call_tool("remember", {"content": "x", "type": "nonsense"})
            assert bad.is_error is True
            first = bad.content[0]
            assert isinstance(first, TextContent)
            assert "type must be one of" in first.text

    asyncio.run(go())


def test_mem_mcp_needs_an_agent(live_cfg):
    env = {k: v for k, v in os.environ.items() if k != "AGENT_MEMORY_AGENT"}
    env["AGENT_MEMORY_HOME"] = str(live_cfg.home)
    proc = subprocess.run(
        [sys.executable, "-m", "agent_memory.mcp_server"],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 1
    assert "AGENT_MEMORY_AGENT" in proc.stderr
