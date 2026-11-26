"""Tests for `mem bridge`, `MEM-7` (`session_start_v1` / `mem context`), and Corvus attribution."""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from agent_memory.bridge import BridgeServer, run_bridge
from agent_memory.capture import attribution, ids
from agent_memory.capture.attribution import evidence_from_records, reconcile_answers
from agent_memory.capture.observations import covering
from agent_memory.capture.pipeline import deliver, replay_spool
from agent_memory.capture.record import Record
from agent_memory.cli import main
from agent_memory.db import MemoryClient, MemoryServer, sql_literal
from agent_memory.mcp_server import MemoryTools, ranking
from agent_memory.review import reviewer


def q(client: MemoryClient, sql: str) -> list[tuple[Any, ...]]:
    return client.query(sql)[1]


def _bridge_proc(
    home: Path, port: int, lines: list[str], *, extra_args: list[str] | None = None
) -> subprocess.CompletedProcess[str]:
    cmd = [
        sys.executable,
        "-m",
        "agent_memory.cli",
        "--home",
        str(home),
        "--port",
        str(port),
        "bridge",
        *(extra_args or []),
    ]
    return subprocess.run(
        cmd,
        input="\n".join(lines) + ("\n" if lines else ""),
        capture_output=True,
        text=True,
        timeout=30,
    )


def _corvus_records(
    *,
    harness_session_id: str,
    retrieval_id: str,
    placeholder_id: str,
    with_corvus_identity: bool,
) -> list[Record]:
    ts0 = datetime(2026, 11, 26, 12, 0, tzinfo=UTC)
    recall_payload: dict[str, Any] = {"tool_name": "recall", "tool_input": {"query": "uv"}}
    cite_payload: dict[str, Any] = {"tool_name": "cite", "tool_input": {}}
    if with_corvus_identity:
        recall_payload.update({"extension": "corvus", "memory_op": "recall"})
        cite_payload.update({"extension": "corvus", "memory_op": "cite"})

    def rec(
        kind: str,
        key: str,
        offset_s: int,
        *,
        role: str,
        content: str | None = None,
        payload: dict[str, Any] | None = None,
        parent_key: str | None = None,
        has_tool_calls: bool = False,
        turn_complete: bool = False,
    ) -> Record:
        return Record(
            agent_id="oh-my-pi",
            harness_session_id=harness_session_id,
            kind=kind,
            ts=ts0 + timedelta(seconds=offset_s),
            source_key=key,
            role=role,
            content=content,
            payload=payload,
            parent_key=parent_key,
            attribution={
                "turn_id": "turn-1",
                "turn_complete": turn_complete,
                "has_tool_calls": has_tool_calls,
            },
        )

    return [
        rec("user_msg", "u1", 0, role="user", content="How do I install deps?"),
        rec(
            "tool_call",
            "call-recall",
            1,
            role="assistant",
            payload=recall_payload,
            has_tool_calls=True,
        ),
        rec(
            "tool_result",
            "res-recall",
            2,
            role="tool",
            parent_key="call-recall",
            content=json.dumps({"degraded": False, "retrieval_id": retrieval_id, "items": []}),
        ),
        rec(
            "tool_call",
            "call-cite",
            3,
            role="assistant",
            payload=cite_payload,
            has_tool_calls=True,
        ),
        rec(
            "tool_result",
            "res-cite",
            4,
            role="tool",
            parent_key="call-cite",
            content=json.dumps({"degraded": False, "answer_event_id": placeholder_id, "links": 1}),
        ),
        rec(
            "assistant_msg",
            "reply-1",
            5,
            role="assistant",
            content="Use uv sync.",
            turn_complete=True,
        ),
    ]


# MemoryTools cwd + session_context + session_start_v1 + run_recall(injected)


def test_memory_tools_cwd_derives_project_and_records_session_cwd(live_cfg, live_client):
    tools = MemoryTools(
        live_cfg,
        "oh-my-pi",
        harness_session_id="cwd-derive-1",
        cwd="/tmp/workspaces/corvus-app/",
    )
    try:
        assert tools.project == "corvus-app"
        status = tools.status()
        assert status["reachable"] is True
        assert status["project"] == "corvus-app"
        rows = q(
            live_client,
            "SELECT cwd, project, metadata::JSON::VARCHAR FROM sessions "
            f"WHERE session_id = {sql_literal(tools.session_id)}",
        )
        assert rows[0][0] == "/tmp/workspaces/corvus-app/"
        assert rows[0][1] == "corvus-app"
    finally:
        tools.close()


def test_session_start_filters_active_only_and_orders_deterministically(live_cfg, live_client):
    tools = MemoryTools(live_cfg, "oh-my-pi", harness_session_id="ss-filter", project="proj-a")
    human = reviewer(live_cfg)
    try:
        # Active memories in proj-a and global.
        proj_high = tools.remember(
            "Project high\nwith  newlines", "decision", scope="proj-a", importance=0.8
        )
        proj_low = tools.remember("Project low", "fact", scope="proj-a", importance=0.4)
        global_top = tools.remember(
            "Global top importance", "preference", scope="global", importance=0.95
        )
        other_proj = tools.remember("Other project memory", "fact", scope="proj-b", importance=0.99)
        # Non-active memories that MUST NOT appear:
        quarantined = tools.remember(
            "Quarantined tool fact", "fact", scope="proj-a", trust_level="tool"
        )
        retired = tools.remember("Retired fact", "fact", scope="proj-a", importance=0.9)
        tools.review("retire", retired["memory_id"], "stale")
        expired = tools.remember(
            "Expired fact",
            "fact",
            scope="proj-a",
            expires_at=(datetime.now(UTC) - timedelta(hours=1)).isoformat(),
        )
        pending = tools.remember(
            "Future pending fact",
            "fact",
            scope="proj-a",
            valid_from=(datetime.now(UTC) + timedelta(days=1)).isoformat(),
        )
        old_fact = tools.remember("Old superseded fact", "fact", scope="proj-a", importance=0.3)
        new_fact = tools.remember(
            "New superseding fact",
            "fact",
            scope="proj-a",
            importance=0.7,
            supersedes=old_fact["memory_id"],
        )
        assert quarantined["status"] == "quarantined"
        assert expired["status"] == "expired"
        assert pending["status"] == "pending"
        assert new_fact["status"] == "active"

        ctx1 = tools.session_context()
        ctx2 = tools.session_context()
        assert not ctx1["degraded"]
        assert ctx1["strategy"] == ranking.SESSION_START_STRATEGY
        assert ctx1["text"] == ctx2["text"]

        ids_returned = [item["memory_id"] for item in ctx1["items"]]
        # Project scope comes before global, then importance DESC:
        assert ids_returned == [
            proj_high["memory_id"],
            new_fact["memory_id"],
            proj_low["memory_id"],
            global_top["memory_id"],
        ]
        # Excluded: other_proj, quarantined, retired, expired, pending, old_fact (superseded)
        for excluded in (
            other_proj["memory_id"],
            quarantined["memory_id"],
            retired["memory_id"],
            expired["memory_id"],
            pending["memory_id"],
            old_fact["memory_id"],
        ):
            assert excluded not in ids_returned

        expected_lines = [
            "# Active memories",
            f"- [{proj_high['memory_id']}] (proj-a/decision) Project high with newlines",
            f"- [{new_fact['memory_id']}] (proj-a/fact) New superseding fact",
            f"- [{proj_low['memory_id']}] (proj-a/fact) Project low",
            f"- [{global_top['memory_id']}] (global/preference) Global top importance",
        ]
        assert ctx1["text"] == "\n".join(expected_lines)
        assert ctx1["shown_count"] == 4
        assert ctx1["not_shown_count"] == 0
        assert ctx1["char_count"] == len(ctx1["text"])
        for item in ctx1["items"]:
            assert "status" not in item
            assert "injected" not in item

        # Also verify type filtering.
        only_pref = tools.session_context(types=["preference"])
        assert [i["memory_id"] for i in only_pref["items"]] == [global_top["memory_id"]]
    finally:
        human.close()
        tools.close()


def test_session_start_enforces_max_chars_max_items_and_records_unshown(live_cfg, live_client):
    tools = MemoryTools(live_cfg, "oh-my-pi", harness_session_id="ss-budget", project="budget-proj")
    try:
        m1 = tools.remember("First " + ("a" * 70), "fact", scope="budget-proj", importance=0.9)
        m2 = tools.remember("Second " + ("b" * 70), "fact", scope="budget-proj", importance=0.8)
        m3 = tools.remember("Third " + ("c" * 70), "fact", scope="budget-proj", importance=0.7)

        # Max chars tight enough to fit only 1 item (header ~17 + line ~120 = ~137 <= 200;
        # 2nd line would exceed 200).
        tight = tools.session_context(scopes=["budget-proj"], max_chars=200, max_items=10)
        assert tight["max_chars"] == 200
        assert tight["shown_count"] == 1
        assert tight["not_shown_count"] == 2
        assert len(tight["text"]) <= 200
        assert [i["memory_id"] for i in tight["items"]] == [m1["memory_id"]]

        ret_row = q(
            live_client,
            "SELECT strategy, candidates, returned, token_estimate "
            f"FROM retrievals WHERE retrieval_id = {sql_literal(tight['retrieval_id'])}",
        )[0]
        assert ret_row == (
            "session_start_v1",
            3,
            1,
            len(tight["items"][0]["content"]) // 4,
        )
        item_rows = q(
            live_client,
            "SELECT rank, memory_id::VARCHAR, injected, status_at_recall "
            "FROM retrieval_items "
            f"WHERE retrieval_id = {sql_literal(tight['retrieval_id'])} ORDER BY rank",
        )
        assert item_rows == [
            (1, m1["memory_id"], True, "active"),
            (2, m2["memory_id"], False, "active"),
            (3, m3["memory_id"], False, "active"),
        ]

        # Max items = 2 with generous max_chars.
        by_items = tools.session_context(scopes=["budget-proj"], max_items=2, max_chars=5000)
        assert by_items["shown_count"] == 2
        assert by_items["not_shown_count"] == 1
        assert [i["memory_id"] for i in by_items["items"]] == [
            m1["memory_id"],
            m2["memory_id"],
        ]

        # When 0 memories fit (single oversized memory > 200 chars), text is empty and recorded.
        tools.remember("Oversized " + ("z" * 300), "fact", scope="oversize-proj", importance=0.9)
        zero_fit = tools.session_context(scopes=["oversize-proj"], max_chars=50)
        assert zero_fit["max_chars"] == 200  # clamped min
        assert zero_fit["text"] == ""
        assert zero_fit["shown_count"] == 0
        assert zero_fit["not_shown_count"] == 1
        assert zero_fit["char_count"] == 0
        assert q(
            live_client,
            "SELECT strategy, candidates, returned FROM retrievals "
            f"WHERE retrieval_id = {sql_literal(zero_fit['retrieval_id'])}",
        ) == [("session_start_v1", 1, 0)]
    finally:
        tools.close()


def test_session_start_empty_ledger_is_recorded(live_cfg, live_client):
    tools = MemoryTools(live_cfg, "oh-my-pi", harness_session_id="ss-empty", project="empty-scope")
    try:
        res = tools.session_context(scopes=["empty-scope"])
        assert not res["degraded"]
        assert res["text"] == ""
        assert res["items"] == []
        assert res["shown_count"] == 0
        assert res["not_shown_count"] == 0
        assert res["char_count"] == 0
        assert res["retrieval_id"] is not None
        assert q(
            live_client,
            "SELECT strategy, candidates, returned FROM retrievals "
            f"WHERE retrieval_id = {sql_literal(res['retrieval_id'])}",
        ) == [("session_start_v1", 0, 0)]
    finally:
        tools.close()


# mem context CLI


def test_mem_context_cli_plain_json_empty_and_down(
    live_cfg, live_client, down_cfg, capsys, monkeypatch
):
    for var in ("AGENT_MEMORY_HOME", "AGENT_MEMORY_HOST", "AGENT_MEMORY_PORT"):
        monkeypatch.delenv(var, raising=False)

    # Empty scope: plain prints nothing, exits 0, still records retrieval.
    code = main(
        [
            "--home",
            str(live_cfg.home),
            "--port",
            str(live_cfg.port),
            "context",
            "--scope",
            "cli-empty",
            "--session",
            "ctx-cli-empty",
        ]
    )
    out = capsys.readouterr()
    assert (code, out.out, out.err) == (0, "", "")
    assert q(
        live_client,
        "SELECT count(*) FROM retrievals r JOIN sessions s USING (session_id) "
        "WHERE s.agent_id = 'human' AND s.harness_session_id = 'ctx-cli-empty' "
        "AND r.strategy = 'session_start_v1'",
    ) == [(1,)]

    # Seed memory via tools, then verify plain & --json output.
    tools = MemoryTools(live_cfg, "oh-my-pi", harness_session_id="ctx-seed", project="cli-proj")
    try:
        mem = tools.remember("Use uv run pytest", "procedure", scope="cli-proj", importance=0.8)
    finally:
        tools.close()

    code = main(
        [
            "--home",
            str(live_cfg.home),
            "--port",
            str(live_cfg.port),
            "context",
            "--cwd",
            "/tmp/work/cli-proj",
            "--scope",
            "cli-proj",
            "--session",
            "ctx-cli-plain",
        ]
    )
    out = capsys.readouterr()
    assert code == 0
    assert out.err == ""
    assert (
        out.out
        == f"# Active memories\n- [{mem['memory_id']}] (cli-proj/procedure) Use uv run pytest\n"
    )

    code = main(
        [
            "--home",
            str(live_cfg.home),
            "--port",
            str(live_cfg.port),
            "context",
            "--scope",
            "cli-proj",
            "--json",
        ]
    )
    out = capsys.readouterr()
    assert code == 0
    parsed = json.loads(out.out)
    assert parsed["degraded"] is False
    assert parsed["strategy"] == "session_start_v1"
    assert parsed["shown_count"] == 1
    assert parsed["items"][0]["memory_id"] == mem["memory_id"]

    # Unreachable server: silent stdout + exit 0 in plain mode; degraded JSON in --json mode.
    code = main(
        ["--home", str(down_cfg.home), "--port", str(down_cfg.port), "context", "--project", "p"]
    )
    out = capsys.readouterr()
    assert (code, out.out) == (0, "")

    code = main(
        [
            "--home",
            str(down_cfg.home),
            "--port",
            str(down_cfg.port),
            "context",
            "--project",
            "p",
            "--json",
        ]
    )
    out = capsys.readouterr()
    assert code == 0
    degraded = json.loads(out.out)
    assert degraded["degraded"] is True
    assert degraded["text"] == ""
    assert degraded["retrieval_id"] is None


# mem bridge


def test_mem_bridge_all_ops_session_switching_and_errors(live_cfg, live_client):
    s1 = "omp-session-one"
    s2 = "omp-session-two"
    requests = [
        "",  # blank line ignored
        "   ",  # whitespace ignored
        "{not valid json",
        json.dumps(["not", "an", "object"]),
        json.dumps({"id": "bad-no-hsid", "op": "status"}),
        json.dumps({"id": "bad-blank-hsid", "op": "status", "harness_session_id": "  "}),
        json.dumps({"id": "bad-op", "op": "nonexistent", "harness_session_id": s1}),
        json.dumps(
            {
                "id": "bad-args",
                "op": "recall",
                "harness_session_id": s1,
                "args": {"query": "x", "extra": 1},
            }
        ),
        json.dumps(
            {
                "id": "bad-limit-bool",
                "op": "recall",
                "harness_session_id": s1,
                "args": {"query": "x", "limit": True},
            }
        ),
        json.dumps(
            {
                "id": "rem-1",
                "op": "remember",
                "harness_session_id": s1,
                "cwd": "/tmp/repos/bridge-alpha",
                "args": {
                    "content": "Bridge alpha runs pytest",
                    "type": "fact",
                    "importance": 0.9,
                },
            }
        ),
        json.dumps(
            {
                "id": "rem-tool",
                "op": "remember",
                "harness_session_id": s1,
                "cwd": "/tmp/repos/bridge-alpha",
                "args": {
                    "content": "Tool derived memory",
                    "type": "observation",
                    "trust_level": "tool",
                },
            }
        ),
        json.dumps(
            {
                "id": "ctx-1",
                "op": "session_context",
                "harness_session_id": s1,
                "cwd": "/tmp/repos/bridge-alpha",
                "args": {"max_items": 5},
            }
        ),
        json.dumps(
            {
                "id": "rec-1",
                "op": "recall",
                "harness_session_id": s1,
                "cwd": "/tmp/repos/bridge-alpha",
                "args": {"query": "pytest"},
            }
        ),
        json.dumps(
            {
                "id": "status-s2",
                "op": "status",
                "harness_session_id": s2,
                "cwd": "/tmp/repos/bridge-beta",
                "project": "custom-beta",
                "args": {},
            }
        ),
        json.dumps(
            {
                "id": "rem-s1-again",
                "op": "remember",
                "harness_session_id": s1,
                "cwd": "/tmp/repos/bridge-alpha",
                "args": {"content": "", "type": "fact"},
            }
        ),
    ]
    proc = _bridge_proc(live_cfg.home, live_cfg.port, requests)
    assert proc.returncode == 0
    assert proc.stderr == ""
    responses = [json.loads(line) for line in proc.stdout.splitlines()]
    by_id = {r["id"]: r for r in responses if r["id"] is not None}

    # Invalid JSON & non-object responses have id=None.
    null_id_errs = [r for r in responses if r["id"] is None]
    assert [r["error"]["code"] for r in null_id_errs] == ["invalid_json", "invalid_args"]

    assert by_id["bad-no-hsid"]["ok"] is False
    assert by_id["bad-no-hsid"]["error"]["code"] == "invalid_args"
    assert by_id["bad-blank-hsid"]["error"]["code"] == "invalid_args"
    assert by_id["bad-op"]["error"]["code"] == "unknown_op"
    assert by_id["bad-args"]["error"]["code"] == "invalid_args"
    assert by_id["bad-limit-bool"]["error"]["code"] == "invalid_args"
    assert by_id["rem-s1-again"]["ok"] is False
    assert by_id["rem-s1-again"]["error"]["code"] == "validation_error"

    rem1 = by_id["rem-1"]
    assert rem1["ok"] is True
    mem_id = rem1["result"]["memory_id"]
    assert rem1["result"]["status"] == "active"

    tool_mem_id = by_id["rem-tool"]["result"]["memory_id"]
    assert by_id["rem-tool"]["result"]["status"] == "quarantined"

    ctx1 = by_id["ctx-1"]
    assert ctx1["ok"] is True
    assert ctx1["result"]["strategy"] == "session_start_v1"
    assert mem_id in ctx1["result"]["text"]
    assert tool_mem_id not in ctx1["result"]["text"]

    rec1 = by_id["rec-1"]
    assert rec1["ok"] is True
    retrieval_id = rec1["result"]["retrieval_id"]
    assert [i["memory_id"] for i in rec1["result"]["items"]] == [mem_id]

    st2 = by_id["status-s2"]
    assert st2["ok"] is True
    assert st2["result"]["agent_id"] == "oh-my-pi"
    assert st2["result"]["harness_session_id"] == s2
    assert st2["result"]["project"] == "custom-beta"
    assert st2["result"]["reachable"] is True

    # Follow-up process: test cite + refused approve + retire review.
    follow_up = [
        json.dumps(
            {
                "id": 10,
                "op": "cite",
                "harness_session_id": s1,
                "cwd": "/tmp/repos/bridge-alpha",
                "args": {"retrieval_id": retrieval_id, "memory_ids": [mem_id]},
            }
        ),
        json.dumps(
            {
                "id": 11,
                "op": "review",
                "harness_session_id": s1,
                "cwd": "/tmp/repos/bridge-alpha",
                "args": {
                    "action": "approve",
                    "memory_id": tool_mem_id,
                    "reason": "agent tries to self-approve",
                },
            }
        ),
        json.dumps(
            {
                "id": 12,
                "op": "review",
                "harness_session_id": s1,
                "cwd": "/tmp/repos/bridge-alpha",
                "args": {
                    "action": "retire",
                    "memory_id": tool_mem_id,
                    "reason": "retire via bridge",
                },
            }
        ),
    ]
    proc2 = _bridge_proc(live_cfg.home, live_cfg.port, follow_up)
    assert proc2.returncode == 0
    res2 = {r["id"]: r for r in (json.loads(line) for line in proc2.stdout.splitlines())}
    assert res2[10]["ok"] is True
    assert res2[10]["result"]["links"] == 2
    assert res2[11]["ok"] is False
    assert res2[11]["error"]["code"] == "validation_error"
    assert "mem review" in res2[11]["error"]["message"]
    assert res2[12]["ok"] is True
    assert res2[12]["result"]["status"] == "retired"

    # Verify DB recorded via mem-bridge and true cwd/project without any approve action.
    sid1 = ids.session_id("oh-my-pi", s1)
    sid2 = ids.session_id("oh-my-pi", s2)
    assert q(
        live_client,
        "SELECT cwd, project, metadata::JSON::VARCHAR FROM sessions "
        f"WHERE session_id = {sql_literal(sid1)}",
    ) == [("/tmp/repos/bridge-alpha", "bridge-alpha", '{"via":"mem-bridge"}')]
    assert q(
        live_client,
        f"SELECT cwd, project FROM sessions WHERE session_id = {sql_literal(sid2)}",
    ) == [("/tmp/repos/bridge-beta", "custom-beta")]
    assert q(
        live_client,
        "SELECT source_mode, adapter_version, exclusions::JSON::VARCHAR "
        f"FROM capture_observations WHERE session_id = {sql_literal(sid1)}",
    ) == [("record", "mem_bridge_v1", "[]")]
    assert q(
        live_client,
        f"SELECT action FROM memory_actions WHERE memory_id = {sql_literal(tool_mem_id)}",
    ) == [("retire",)]


def test_mem_bridge_rejects_human_agent(live_cfg):
    proc = _bridge_proc(live_cfg.home, live_cfg.port, [], extra_args=["--agent", "human"])
    assert proc.returncode == 1
    assert proc.stdout == ""
    assert "human" in proc.stderr

    # AGENT_MEMORY_AGENT=human is also rejected; non-human env var is honored.
    cmd = [
        sys.executable,
        "-m",
        "agent_memory.cli",
        "--home",
        str(live_cfg.home),
        "--port",
        str(live_cfg.port),
        "bridge",
    ]
    env_human = {**os.environ, "AGENT_MEMORY_AGENT": "human"}
    proc_env_human = subprocess.run(cmd, input="", env=env_human, capture_output=True, text=True)
    assert proc_env_human.returncode == 1
    assert proc_env_human.stdout == ""
    assert "human" in proc_env_human.stderr

    env_amp = {**os.environ, "AGENT_MEMORY_AGENT": "amp"}
    req = json.dumps({"id": 1, "op": "status", "harness_session_id": "env-agent-1"}) + "\n"
    proc_amp = subprocess.run(cmd, input=req, env=env_amp, capture_output=True, text=True)
    assert proc_amp.returncode == 0
    assert json.loads(proc_amp.stdout.strip())["result"]["agent_id"] == "amp"


def test_cli_lazy_imports_bridge():
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from agent_memory.cli import main; main(['--version']); "
            "assert 'agent_memory.bridge' not in sys.modules",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, proc.stderr


def test_mem_bridge_lru_eviction_closes_clients(live_cfg, monkeypatch):
    closed: list[str] = []
    orig_close = MemoryTools.close

    def tracking_close(self: MemoryTools) -> None:
        closed.append(self.harness_session_id)
        orig_close(self)

    monkeypatch.setattr(MemoryTools, "close", tracking_close)
    server = BridgeServer(live_cfg, "oh-my-pi", cache_size=2)
    try:
        for hsid in ("s1", "s2", "s1", "s3"):
            resp = server.handle_line(
                json.dumps({"id": hsid, "op": "status", "harness_session_id": hsid})
            )
            assert resp is not None
            assert resp["ok"] is True
        # s1 was touched before s3, so s2 was evicted.
        assert closed == ["s2"]
    finally:
        server.close()
    assert set(closed) == {"s2", "s1", "s3"}


def test_mem_bridge_degrades_and_spools_when_server_down(down_cfg):
    first_batch = [
        json.dumps(
            {
                "id": "r1",
                "op": "recall",
                "harness_session_id": "down-1",
                "args": {"query": "anything"},
            }
        ),
        json.dumps(
            {
                "id": "sc1",
                "op": "session_context",
                "harness_session_id": "down-1",
                "args": {"max_chars": 500},
            }
        ),
        json.dumps(
            {
                "id": "c1",
                "op": "cite",
                "harness_session_id": "down-1",
                "args": {"memory_ids": ["00000000-0000-0000-0000-000000000001"]},
            }
        ),
        json.dumps(
            {
                "id": "rem1",
                "op": "remember",
                "harness_session_id": "down-1",
                "cwd": "/tmp/down-proj",
                "args": {"content": "Queued via bridge", "type": "fact"},
            }
        ),
    ]
    stdout1 = io.StringIO()
    stderr1 = io.StringIO()
    assert (
        run_bridge(
            down_cfg,
            stdin=io.StringIO("\n".join(first_batch) + "\n"),
            stdout=stdout1,
            stderr=stderr1,
        )
        == 0
    )
    assert stderr1.getvalue() == ""
    by_id = {r["id"]: r for r in (json.loads(line) for line in stdout1.getvalue().splitlines())}

    assert by_id["r1"]["ok"] is True
    assert by_id["r1"]["result"]["degraded"] is True
    assert by_id["r1"]["result"]["items"] == []

    assert by_id["sc1"]["ok"] is True
    assert by_id["sc1"]["result"] == {
        "degraded": True,
        "reason": by_id["sc1"]["result"]["reason"],
        "retrieval_id": None,
        "request_event_id": None,
        "strategy": "session_start_v1",
        "text": "",
        "items": [],
        "shown_count": 0,
        "not_shown_count": 0,
        "char_count": 0,
        "max_chars": 500,
    }

    assert by_id["c1"]["ok"] is True
    assert by_id["c1"]["result"]["degraded"] is True
    assert by_id["c1"]["result"]["links"] == 0

    assert by_id["rem1"]["ok"] is True
    assert by_id["rem1"]["result"]["queued"] is True
    queued_id = by_id["rem1"]["result"]["memory_id"]

    second_batch = [
        json.dumps(
            {
                "id": "quar1",
                "op": "review",
                "harness_session_id": "down-1",
                "args": {
                    "action": "quarantine",
                    "memory_id": queued_id,
                    "reason": "quarantine queued while down",
                },
            }
        ),
        json.dumps(
            {
                "id": "app1",
                "op": "review",
                "harness_session_id": "down-1",
                "args": {
                    "action": "approve",
                    "memory_id": queued_id,
                    "reason": "must never spool",
                },
            }
        ),
        json.dumps(
            {
                "id": "st1",
                "op": "status",
                "harness_session_id": "down-1",
                "args": {},
            }
        ),
    ]
    stdout2 = io.StringIO()
    stderr2 = io.StringIO()
    assert (
        run_bridge(
            down_cfg,
            stdin=io.StringIO("\n".join(second_batch) + "\n"),
            stdout=stdout2,
            stderr=stderr2,
        )
        == 0
    )
    assert stderr2.getvalue() == ""
    by_id2 = {r["id"]: r for r in (json.loads(line) for line in stdout2.getvalue().splitlines())}

    assert by_id2["quar1"]["ok"] is True
    assert by_id2["quar1"]["result"]["queued"] is True
    assert by_id2["quar1"]["result"]["action"] == "quarantine"

    # approve is refused as validation_error and NEVER spooled.
    assert by_id2["app1"]["ok"] is False
    assert by_id2["app1"]["error"]["code"] == "validation_error"

    assert by_id2["st1"]["ok"] is True
    assert by_id2["st1"]["result"]["reachable"] is False
    assert by_id2["st1"]["result"]["spool_pending_lines"] == 2

    # Start server and replay spooled bridge writes.
    server = MemoryServer(down_cfg, export_interval_hours=0)
    server.start()
    try:
        replayed = replay_spool(down_cfg)
        assert replayed.failed_lines == 0
        with MemoryClient.from_config(down_cfg) as client:
            assert q(
                client,
                f"SELECT content, scope FROM memories WHERE memory_id = {sql_literal(queued_id)}",
            ) == [("Queued via bridge", "down-proj")]
            assert q(
                client,
                f"SELECT status FROM memory_status WHERE memory_id = {sql_literal(queued_id)}",
            ) == [("quarantined",)]
    finally:
        server.stop()


# Corvus native tool attribution


def test_corvus_native_tool_attribution_unit_and_reconcile(live_cfg, live_client):
    # Generic bare tool names without extension == "corvus" must still be rejected.
    assert attribution._memory_tool({"tool_name": "recall"}, "recall") is False
    assert attribution._memory_tool({"tool_name": "cite"}, "cite") is False
    assert (
        attribution._memory_tool(
            {"extension": "other", "memory_op": "recall", "tool_name": "recall"}, "recall"
        )
        is False
    )
    assert (
        attribution._memory_tool(
            {"extension": "corvus", "memory_op": "recall", "tool_name": "recall"}, "recall"
        )
        is True
    )
    assert (
        attribution._memory_tool(
            {"extension": "corvus", "memory_op": "cite", "tool_name": "cite"}, "cite"
        )
        is True
    )

    tools = MemoryTools(live_cfg, "oh-my-pi", harness_session_id="corvus-attr-1", project="corvus")
    try:
        recall = tools.recall("uv")
        cited = tools.cite(retrieval_id=recall["retrieval_id"])
    finally:
        tools.close()

    # Bare records without extension == "corvus" produce no attribution evidence.
    bare = _corvus_records(
        harness_session_id="corvus-attr-bare",
        retrieval_id=recall["retrieval_id"],
        placeholder_id=cited["answer_event_id"],
        with_corvus_identity=False,
    )
    assert evidence_from_records(bare) == []

    # Corvus-tagged records produce attribution evidence and reconcile answer_bindings.
    corvus_recs = _corvus_records(
        harness_session_id="corvus-attr-1",
        retrieval_id=recall["retrieval_id"],
        placeholder_id=cited["answer_event_id"],
        with_corvus_identity=True,
    )
    evidence = evidence_from_records(corvus_recs)
    assert len(evidence) == 1
    assert evidence[0]["placeholder_event_id"] == cited["answer_event_id"]
    assert [r["retrieval_id"] for r in evidence[0]["captured_retrievals"]] == [
        recall["retrieval_id"]
    ]

    observed_recs = [
        *corvus_recs[:-1],
        replace(
            corvus_recs[-1],
            observation={**covering(corvus_recs), "attribution_evidence": evidence},
        ),
    ]
    delivery = deliver(live_cfg, observed_recs, deadline=10)
    assert delivery.written is not None
    sid = ids.session_id("oh-my-pi", "corvus-attr-1")
    reconcile_answers(live_client, [sid])
    reply_id = ids.event_id("oh-my-pi", "corvus-attr-1", "assistant_msg", "reply-1")
    bindings = q(
        live_client,
        "SELECT answer_event_id::VARCHAR, reason_code, captured_retrieval_ids::VARCHAR[] "
        "FROM effective_answer_bindings "
        f"WHERE placeholder_event_id = {sql_literal(cited['answer_event_id'])}",
    )
    assert bindings == [(reply_id, "unique_captured_turn", [recall["retrieval_id"]])]
