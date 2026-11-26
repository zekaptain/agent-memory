"""Tests for usefulness recall and citation accounting (MEM-19)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any

from agent_memory.config import Config
from agent_memory.db import MemoryServer, initialize_home, open_direct
from agent_memory.mcp_server import MemoryTools
from agent_memory.smoke import free_port
from agent_memory.usefulness import (
    HarnessInvocation,
    HarnessResult,
    ScriptedHarnessAdapter,
    TrialConfig,
    TrialLedgerBaseline,
    account_trial_recall,
    capture_trial_baseline,
    run_trial,
)


def _git(repo: Path, *args: str) -> str:
    env = os.environ.copy()
    env.update(
        {
            "GIT_AUTHOR_NAME": "Test Author",
            "GIT_AUTHOR_EMAIL": "test@example.com",
            "GIT_COMMITTER_NAME": "Test Author",
            "GIT_COMMITTER_EMAIL": "test@example.com",
            "GIT_TERMINAL_PROMPT": "0",
        }
    )
    proc = subprocess.run(
        ["git", *args],
        cwd=str(repo),
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    return proc.stdout.strip()


def _make_repo_and_task(tmp_path: Path, *, project: str = "demo-proj") -> tuple[Path, Path]:
    repo = tmp_path / "source-repo"
    repo.mkdir(parents=True)
    _git(repo, "init", "-b", "main")
    (repo / "app.py").write_text("def answer():\n    return 1\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "base")
    commit = _git(repo, "rev-parse", "HEAD")

    exp_home = tmp_path / "exp"
    task_dir = exp_home / "tasks" / "t-recall"
    task_dir.mkdir(parents=True)
    (task_dir / "task.json").write_text(
        json.dumps(
            {
                "task_id": "t-recall",
                "repository": str(repo),
                "commit": commit,
                "project": project,
                "work_done_at": "2026-11-20T12:00:00Z",
                "task_family": "fam-1",
                "strata": ["ordinary"],
                "checks": {
                    "commands": [{"name": "ok", "command": [sys.executable, "-c", "print('ok')"]}]
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (task_dir / "request.md").write_text("Make answer() return 42.\n", encoding="utf-8")
    (task_dir / "required_behavior.md").write_text("- answer() returns 42\n", encoding="utf-8")
    return exp_home, task_dir


def _make_seeded_home_with_history(
    path: Path,
    project: str,
) -> tuple[Path, str, str, str]:
    """Create a stopped home containing two active memories plus pre-trial history.

    Returns ``(home_path, mem_id_1, mem_id_2, historical_retrieval_id)``.
    The historical retrieval and citation verify that pre-trial rows copied from
    the memory set are excluded from the trial's recall and citation counts.
    """
    cfg = Config(home=path, port=free_port())
    initialize_home(cfg)

    server = MemoryServer(cfg)
    server.start()
    try:
        tools = MemoryTools(cfg, "factory-droid", harness_session_id="hist-1", project=project)
        try:
            r1 = tools.remember(
                "answer() must return 42 for demo-proj",
                "fact",
                scope=project,
                trust_level="user",
            )
            r2 = tools.remember(
                "verify answer() returns 42 with pytest",
                "procedure",
                scope=project,
                trust_level="user",
            )
            mem1 = str(r1["memory_id"])
            mem2 = str(r2["memory_id"])

            # Add a historical retrieval + citation placeholder so pre-trial
            # retrievals/context_links/placeholders exist in the memory set.
            hist_rec = tools.recall("answer 42")
            assert not hist_rec["degraded"]
            assert len(hist_rec["items"]) == 2
            hist_rid = str(hist_rec["retrieval_id"])
            hist_cite = tools.cite(retrieval_id=hist_rid, memory_ids=[mem1])
            assert not hist_cite["degraded"]
        finally:
            tools.close()
    finally:
        server.stop()

    return path, mem1, mem2, hist_rid


def _tool_call_msg(
    msg_id: str, ts: str, call_id: str, name: str, tool_input: dict[str, Any]
) -> dict[str, Any]:
    return {
        "type": "message",
        "id": msg_id,
        "timestamp": ts,
        "message": {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": call_id, "name": name, "input": tool_input}],
        },
    }


def _tool_result_msg(
    msg_id: str,
    ts: str,
    call_id: str,
    content: Any,
    *,
    is_error: bool = False,
) -> dict[str, Any]:
    part: dict[str, Any] = {
        "type": "tool_result",
        "tool_use_id": call_id,
        "content": content,
    }
    if is_error:
        part["is_error"] = True
    return {
        "type": "message",
        "id": msg_id,
        "timestamp": ts,
        "message": {"role": "user", "content": [part]},
    }


def _assistant_reply_msg(msg_id: str, ts: str, text: str) -> dict[str, Any]:
    return {
        "type": "message",
        "id": msg_id,
        "timestamp": ts,
        "message": {"role": "assistant", "content": text},
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return path


def test_memory_arm_trial_imports_transcript_and_counts_recalls_and_citations(
    tmp_path: Path,
) -> None:
    """End-to-end check of every recall outcome and citation metric against the trial home."""
    project = "demo-proj"
    exp_home, task_dir = _make_repo_and_task(tmp_path, project=project)
    source_home, mem1, mem2, hist_rid = _make_seeded_home_with_history(
        tmp_path / "source-home", project
    )

    observed_ids: dict[str, str] = {}

    def scripted_memory_trial(inv: HarnessInvocation) -> HarnessResult:
        assert inv.memory_endpoint is not None
        ep = inv.memory_endpoint
        trial_cfg = Config(home=ep.home, host=ep.host, port=ep.port)
        tools = MemoryTools(trial_cfg, ep.agent_id, project=ep.project)
        try:
            # 1. Recall that returns memories (both mem1 and mem2).
            res_hit = tools.recall("answer 42")
            assert not res_hit["degraded"]
            assert len(res_hit["items"]) == 2
            rid_hit = str(res_hit["retrieval_id"])
            observed_ids["rid_hit"] = rid_hit

            # Turn 1 citation: single citation bound to a unique reply (cites mem1).
            cite_single = tools.cite(retrieval_id=rid_hit, memory_ids=[mem1])
            assert not cite_single["degraded"]
            p_single = str(cite_single["answer_event_id"])
            observed_ids["p_single"] = p_single

            # 2. Recall that is a real miss recorded in the ledger (returned == 0).
            res_miss = tools.recall("nonexistent_xylophone_token_999")
            assert not res_miss["degraded"]
            assert len(res_miss["items"]) == 0
            rid_miss = str(res_miss["retrieval_id"])
            observed_ids["rid_miss"] = rid_miss

            # Turn 5 citations: two citations in one turn -> multi-citation reply.
            cite_multi_1 = tools.cite(retrieval_id=rid_hit, memory_ids=[mem1])
            cite_multi_2 = tools.cite(retrieval_id=rid_hit, memory_ids=[mem2])
            p_multi_1 = str(cite_multi_1["answer_event_id"])
            p_multi_2 = str(cite_multi_2["answer_event_id"])
            observed_ids["p_multi_1"] = p_multi_1
            observed_ids["p_multi_2"] = p_multi_2

            # Turn 6 citation: ambiguous turn (two assistant replies) -> unbound placeholder.
            cite_ambig = tools.cite(retrieval_id=rid_hit, memory_ids=[mem1])
            p_ambig = str(cite_ambig["answer_event_id"])
            observed_ids["p_ambig"] = p_ambig

            # Uncaptured citation: never appears in the transcript -> unbound placeholder.
            cite_orphan = tools.cite(retrieval_id=rid_hit)
            p_orphan = str(cite_orphan["answer_event_id"])
            observed_ids["p_orphan"] = p_orphan
        finally:
            tools.close()

        fake_missing_rid = str(uuid.uuid4())
        observed_ids["fake_missing_rid"] = fake_missing_rid

        transcript_rows: list[dict[str, Any]] = [
            {
                "type": "session_start",
                "id": "trial-droid-sess-1",
                "timestamp": "2026-11-26T10:00:00Z",
                "cwd": str(inv.clone_dir),
            },
            # Turn 1: hit recall + single citation + unique reply
            _tool_call_msg(
                "m1",
                "2026-11-26T10:00:01Z",
                "call-hit",
                "agent-memory___recall",
                {"query": "answer 42"},
            ),
            _tool_result_msg("m2", "2026-11-26T10:00:02Z", "call-hit", json.dumps(res_hit)),
            _tool_call_msg(
                "m3",
                "2026-11-26T10:00:03Z",
                "call-cite-1",
                "agent-memory___cite",
                {"retrieval_id": rid_hit, "memory_ids": [mem1]},
            ),
            _tool_result_msg(
                "m4",
                "2026-11-26T10:00:04Z",
                "call-cite-1",
                json.dumps({"degraded": False, "answer_event_id": p_single, "links": 2}),
            ),
            _assistant_reply_msg("reply-1", "2026-11-26T10:00:05Z", "Updated answer() to 42."),
            {"type": "agent_turn_outcome", "turnId": "turn-1", "resultKind": "text"},
            # Turn 2: miss recall (0 items, recorded in trial ledger)
            _tool_call_msg(
                "m5",
                "2026-11-26T10:00:06Z",
                "call-miss",
                "agent-memory___recall",
                {"query": "nonexistent_xylophone_token_999"},
            ),
            _tool_result_msg("m6", "2026-11-26T10:00:07Z", "call-miss", json.dumps(res_miss)),
            _assistant_reply_msg("reply-2", "2026-11-26T10:00:08Z", "No extra facts found."),
            {"type": "agent_turn_outcome", "turnId": "turn-2", "resultKind": "text"},
            # Turn 3: degraded recall (answered without the ledger)
            _tool_call_msg(
                "m7",
                "2026-11-26T10:00:09Z",
                "call-degraded",
                "agent-memory___recall",
                {"query": "style"},
            ),
            _tool_result_msg(
                "m8",
                "2026-11-26T10:00:10Z",
                "call-degraded",
                json.dumps(
                    {
                        "degraded": True,
                        "reason": "ConnectionRefusedError: server unreachable",
                        "retrieval_id": None,
                        "items": [],
                    }
                ),
            ),
            # Turn 4a: missing record with empty items & unknown retrieval_id (MUST NOT be a miss!)
            _tool_call_msg(
                "m9",
                "2026-11-26T10:00:11Z",
                "call-missing-rid",
                "agent-memory___recall",
                {"query": "ghost"},
            ),
            _tool_result_msg(
                "m10",
                "2026-11-26T10:00:12Z",
                "call-missing-rid",
                json.dumps(
                    {
                        "degraded": False,
                        "retrieval_id": fake_missing_rid,
                        "items": [],
                    }
                ),
            ),
            # Turn 4b: missing record with no tool_result at all (e.g. aborted mid-call)
            _tool_call_msg(
                "m11",
                "2026-11-26T10:00:13Z",
                "call-missing-res",
                "agent-memory___recall",
                {"query": "cut off"},
            ),
            # Turn 4c: missing record pointing to a pre-trial historical retrieval.
            _tool_call_msg(
                "m12",
                "2026-11-26T10:00:14Z",
                "call-hist-rid",
                "agent-memory___recall",
                {"query": "stale"},
            ),
            _tool_result_msg(
                "m13",
                "2026-11-26T10:00:15Z",
                "call-hist-rid",
                json.dumps({"degraded": False, "retrieval_id": hist_rid, "items": []}),
            ),
            _assistant_reply_msg("reply-3", "2026-11-26T10:00:16Z", "Continuing."),
            {"type": "agent_turn_outcome", "turnId": "turn-3", "resultKind": "text"},
            # Turn 5: two citations bound to one reply -> multi-citation reply
            _tool_call_msg(
                "m14",
                "2026-11-26T10:00:17Z",
                "call-cite-m1",
                "agent-memory___cite",
                {"retrieval_id": rid_hit, "memory_ids": [mem1]},
            ),
            _tool_result_msg(
                "m15",
                "2026-11-26T10:00:18Z",
                "call-cite-m1",
                json.dumps({"degraded": False, "answer_event_id": p_multi_1, "links": 2}),
            ),
            _tool_call_msg(
                "m16",
                "2026-11-26T10:00:19Z",
                "call-cite-m2",
                "agent-memory___cite",
                {"retrieval_id": rid_hit, "memory_ids": [mem2]},
            ),
            _tool_result_msg(
                "m17",
                "2026-11-26T10:00:20Z",
                "call-cite-m2",
                json.dumps({"degraded": False, "answer_event_id": p_multi_2, "links": 2}),
            ),
            _assistant_reply_msg("reply-multi", "2026-11-26T10:00:21Z", "Verified with pytest."),
            {"type": "agent_turn_outcome", "turnId": "turn-5", "resultKind": "text"},
            # Turn 6: ambiguous turn (two text replies) -> unbound placeholder
            _tool_call_msg(
                "m18",
                "2026-11-26T10:00:22Z",
                "call-cite-ambig",
                "agent-memory___cite",
                {"retrieval_id": rid_hit, "memory_ids": [mem1]},
            ),
            _tool_result_msg(
                "m19",
                "2026-11-26T10:00:23Z",
                "call-cite-ambig",
                json.dumps({"degraded": False, "answer_event_id": p_ambig, "links": 2}),
            ),
            _assistant_reply_msg("reply-ambig-a", "2026-11-26T10:00:24Z", "First reply."),
            _assistant_reply_msg("reply-ambig-b", "2026-11-26T10:00:25Z", "Second reply."),
            {"type": "agent_turn_outcome", "turnId": "turn-6", "resultKind": "text"},
        ]

        transcript_path = _write_jsonl(
            inv.harness_home / ".factory" / "sessions" / "clone" / "trial-droid-sess-1.jsonl",
            transcript_rows,
        )
        return HarnessResult(
            status="completed",
            last_reply="Verified with pytest.",
            transcript_path=transcript_path,
            harness_session_id="trial-droid-sess-1",
        )

    record = run_trial(
        exp_home,
        "run-mem19",
        task_dir,
        TrialConfig(
            arm="memory",
            model="local-model",
            model_base_url="http://127.0.0.1:8080/v1",
            source_home=source_home,
            trial_id="t-recall-memory",
        ),
        adapter=ScriptedHarnessAdapter(scripted_memory_trial),
    )

    assert record.status == "completed", record.failure_detail
    assert record.recall_accounting is not None
    ra = record.recall_accounting

    # Transcript import succeeded.
    assert ra.transcript_import.status == "imported"
    assert ra.transcript_import.inserted > 0
    assert ra.transcript_import.spooled == 0
    assert ra.transcript_import.failed_lines == 0

    # Recall outcome counts: 1 returned + 1 miss + 1 degraded + 3 missing_record = 6 total.
    assert ra.made_no_recall is False
    assert ra.total_recalls == 6
    assert ra.returned_count == 1
    assert ra.miss_count == 1
    assert ra.degraded_count == 1
    assert ra.missing_record_count == 3
    assert ra.total_retrieval_latency_ms is not None
    assert ra.total_retrieval_latency_ms >= 0.0

    by_outcome = {r.outcome: [x for x in ra.recalls if x.outcome == r.outcome] for r in ra.recalls}
    assert by_outcome["returned"][0].retrieval_id == observed_ids["rid_hit"]
    assert set(by_outcome["returned"][0].returned_memory_ids) == {mem1, mem2}
    assert by_outcome["miss"][0].retrieval_id == observed_ids["rid_miss"]
    assert by_outcome["miss"][0].returned_count == 0
    assert by_outcome["degraded"][0].detail == "degraded_tool_result"
    missing_details = sorted(x.detail or "" for x in by_outcome["missing_record"])
    assert missing_details == [
        "missing_tool_result",
        "retrieval_not_in_ledger",
        "retrieval_not_in_ledger",
    ]

    # Citation counts: 5 placeholders created in this trial (pre-trial history excluded!).
    ci = ra.citations
    assert ci.citations_made == 5
    assert ci.cite_calls_in_transcript == 4
    assert ci.degraded_cite_calls == 0
    assert ci.placeholder_count == 5
    assert ci.bound_placeholder_count == 3
    assert ci.unbound_placeholder_count == 2
    assert set(ci.unbound_placeholder_ids) == {observed_ids["p_ambig"], observed_ids["p_orphan"]}

    # Replies bound: reply-1 has 1 citation (single), reply-multi has 2 citations (multi).
    assert ci.bound_reply_count == 2
    assert ci.single_citation_reply_count == 1
    assert ci.multi_citation_reply_count == 1
    assert len(ci.single_citation_reply_ids) == 1
    assert len(ci.multi_citation_reply_ids) == 1

    # Returned memories declared used: both mem1 and mem2 were returned and cited.
    assert set(ci.returned_memory_ids) == {mem1, mem2}
    assert set(ci.cited_memory_ids) == {mem1, mem2}
    assert ci.returned_memories_cited_count == 2
    assert set(ci.returned_memories_cited_ids) == {mem1, mem2}

    # Persisted record.json matches the in-memory RecallAccounting verbatim.
    persisted = json.loads(record.record_path.read_text(encoding="utf-8"))
    assert persisted["recall_accounting"] == ra.to_dict()

    # Independently verify directly against the stopped trial home's DuckDB tables.
    trial_home = Path(record.paths["trial_home"] or "")
    source_con = open_direct(source_home / "memory.duckdb", read_only=True)
    try:
        baseline = capture_trial_baseline(source_con)
    finally:
        source_con.close()

    trial_con = open_direct(trial_home / "memory.duckdb", read_only=True)
    try:
        # Confirm transcript session was imported into the trial home.
        imported_sessions = trial_con.execute(
            "SELECT harness_session_id, capture_completeness FROM session_summary "
            "WHERE harness_session_id = 'trial-droid-sess-1'"
        ).fetchall()
        assert imported_sessions == [("trial-droid-sess-1", "transcript_import")]

        # Confirm exact trial retrieval count in DuckDB (excluding baseline).
        all_rids = {
            str(row[0])
            for row in trial_con.execute("SELECT retrieval_id::VARCHAR FROM retrievals").fetchall()
        }
        new_rids = all_rids - baseline.retrieval_ids
        assert new_rids == {observed_ids["rid_hit"], observed_ids["rid_miss"]}

        # Confirm exact effective_answer_bindings counts in DuckDB for this trial's placeholders.
        trial_pids = {
            observed_ids["p_single"],
            observed_ids["p_multi_1"],
            observed_ids["p_multi_2"],
            observed_ids["p_ambig"],
            observed_ids["p_orphan"],
        }
        rows = trial_con.execute(
            "SELECT placeholder_event_id::VARCHAR, answer_event_id::VARCHAR, reason_code "
            "FROM effective_answer_bindings"
        ).fetchall()
        trial_bindings = {
            str(p): (str(a) if a else None, str(rc)) for p, a, rc in rows if str(p) in trial_pids
        }
        assert trial_bindings[observed_ids["p_single"]][1] == "unique_captured_turn"
        assert (
            trial_bindings[observed_ids["p_multi_1"]][0]
            == trial_bindings[observed_ids["p_multi_2"]][0]
        )
        assert trial_bindings[observed_ids["p_ambig"]] == (None, "ambiguous_captured_turn")
        assert observed_ids["p_orphan"] not in trial_bindings
    finally:
        trial_con.close()

    # Re-running standalone account_trial_recall against the stopped trial home (both
    # with transcript_path and falling back to imported ledger events) matches every count!
    standalone_with_file = account_trial_recall(
        trial_home,
        transcript_path=Path(record.paths["transcript_path"] or ""),
        baseline=baseline,
    )
    assert standalone_with_file.total_recalls == ra.total_recalls
    assert standalone_with_file.returned_count == ra.returned_count
    assert standalone_with_file.miss_count == ra.miss_count
    assert standalone_with_file.degraded_count == ra.degraded_count
    assert standalone_with_file.missing_record_count == ra.missing_record_count
    assert standalone_with_file.citations == ra.citations

    standalone_from_ledger = account_trial_recall(trial_home, baseline=baseline)
    assert standalone_from_ledger.total_recalls == ra.total_recalls
    assert standalone_from_ledger.returned_count == ra.returned_count
    assert standalone_from_ledger.miss_count == ra.miss_count
    assert standalone_from_ledger.degraded_count == ra.degraded_count
    assert standalone_from_ledger.missing_record_count == ra.missing_record_count
    assert standalone_from_ledger.citations == ra.citations


def test_memory_arm_trial_with_no_recall_and_no_memory_arm(tmp_path: Path) -> None:
    """Memory arm with zero recalls sets made_no_recall=True; no-memory arm has None."""
    exp_home, task_dir = _make_repo_and_task(tmp_path)

    def no_recall_agent(inv: HarnessInvocation) -> HarnessResult:
        transcript = _write_jsonl(
            inv.harness_home / ".factory" / "sessions" / "clone" / "sess-no-recall.jsonl",
            [
                {
                    "type": "session_start",
                    "id": "sess-no-recall",
                    "timestamp": "2026-11-26T11:00:00Z",
                    "cwd": str(inv.clone_dir),
                },
                _tool_call_msg(
                    "m1", "2026-11-26T11:00:01Z", "call-read", "Read", {"file_path": "app.py"}
                ),
                _tool_result_msg(
                    "m2", "2026-11-26T11:00:02Z", "call-read", "def answer(): return 1"
                ),
                _assistant_reply_msg("m3", "2026-11-26T11:00:03Z", "Done without recall."),
                {"type": "agent_turn_outcome", "turnId": "t1", "resultKind": "text"},
            ],
        )
        return HarnessResult(
            status="completed",
            last_reply="Done without recall.",
            transcript_path=transcript,
            harness_session_id="sess-no-recall",
        )

    mem_record = run_trial(
        exp_home,
        "run-norecall",
        task_dir,
        TrialConfig(
            arm="memory",
            model="local-model",
            model_base_url="http://127.0.0.1:8080/v1",
            trial_id="t-mem-none",
        ),
        adapter=ScriptedHarnessAdapter(no_recall_agent),
    )
    assert mem_record.recall_accounting is not None
    assert mem_record.recall_accounting.made_no_recall is True
    assert mem_record.recall_accounting.total_recalls == 0
    assert mem_record.recall_accounting.returned_count == 0
    assert mem_record.recall_accounting.miss_count == 0
    assert mem_record.recall_accounting.degraded_count == 0
    assert mem_record.recall_accounting.missing_record_count == 0
    assert mem_record.recall_accounting.total_retrieval_latency_ms is None
    assert mem_record.recall_accounting.citations.citations_made == 0
    assert mem_record.recall_accounting.transcript_import.status == "imported"

    nomem_record = run_trial(
        exp_home,
        "run-norecall",
        task_dir,
        TrialConfig(
            arm="no_memory",
            model="local-model",
            model_base_url="http://127.0.0.1:8080/v1",
            trial_id="t-nomem",
        ),
        adapter=ScriptedHarnessAdapter(
            lambda inv: HarnessResult(status="completed", last_reply="Done.")
        ),
    )
    assert nomem_record.recall_accounting is None
    persisted = json.loads(nomem_record.record_path.read_text(encoding="utf-8"))
    assert persisted["recall_accounting"] is None


def test_custom_server_name_graph_recall_and_missing_transcript_ledger_only(
    tmp_path: Path,
) -> None:
    """Custom MCP server name binds citations; missing transcript still counts ledger retrievals."""
    exp_home, task_dir = _make_repo_and_task(tmp_path)

    # 1. Custom memory_server_name + graph recall + citation binding.
    def custom_server_agent(inv: HarnessInvocation) -> HarnessResult:
        assert inv.memory_endpoint is not None
        ep = inv.memory_endpoint
        trial_cfg = Config(home=ep.home, host=ep.host, port=ep.port)
        tools = MemoryTools(trial_cfg, ep.agent_id, project=ep.project)
        try:
            g = tools.graph("demo-proj")
            assert not g["degraded"]
            cite = tools.cite(retrieval_id=str(g["retrieval_id"]))
            assert not cite["degraded"]
            placeholder = str(cite["answer_event_id"])
        finally:
            tools.close()

        transcript = _write_jsonl(
            inv.harness_home / ".factory" / "sessions" / "clone" / "sess-custom.jsonl",
            [
                {
                    "type": "session_start",
                    "id": "sess-custom",
                    "timestamp": "2026-11-26T12:00:00Z",
                    "cwd": str(inv.clone_dir),
                },
                _tool_call_msg(
                    "g1",
                    "2026-11-26T12:00:01Z",
                    "call-graph",
                    "custom-mem___graph",
                    {"entity": "demo-proj"},
                ),
                _tool_result_msg(
                    "g2",
                    "2026-11-26T12:00:02Z",
                    "call-graph",
                    [{"type": "text", "text": json.dumps(g)}],
                ),
                _tool_call_msg(
                    "c1",
                    "2026-11-26T12:00:03Z",
                    "call-cite",
                    "custom-mem___cite",
                    {"retrieval_id": str(g["retrieval_id"])},
                ),
                _tool_result_msg(
                    "c2",
                    "2026-11-26T12:00:04Z",
                    "call-cite",
                    json.dumps({"degraded": False, "answer_event_id": placeholder, "links": 1}),
                ),
                _assistant_reply_msg("r1", "2026-11-26T12:00:05Z", "Checked the graph."),
                {"type": "agent_turn_outcome", "turnId": "t-g", "resultKind": "text"},
            ],
        )
        return HarnessResult(
            status="completed",
            last_reply="Checked the graph.",
            transcript_path=transcript,
            harness_session_id="sess-custom",
        )

    rec_custom = run_trial(
        exp_home,
        "run-custom",
        task_dir,
        TrialConfig(
            arm="memory",
            model="local-model",
            model_base_url="http://127.0.0.1:8080/v1",
            memory_server_name="custom-mem",
            trial_id="t-custom-server",
        ),
        adapter=ScriptedHarnessAdapter(custom_server_agent),
    )
    assert rec_custom.recall_accounting is not None
    ra = rec_custom.recall_accounting
    assert ra.total_recalls == 1
    assert ra.miss_count == 1
    assert ra.recalls[0].operation == "graph"
    assert ra.recalls[0].source == "transcript_and_ledger"
    assert ra.citations.bound_placeholder_count == 1
    assert ra.citations.unbound_placeholder_count == 0
    assert ra.citations.single_citation_reply_count == 1

    # 2. Missing transcript file: ledger retrieval still counted as ledger_only.
    def missing_transcript_agent(inv: HarnessInvocation) -> HarnessResult:
        assert inv.memory_endpoint is not None
        ep = inv.memory_endpoint
        trial_cfg = Config(home=ep.home, host=ep.host, port=ep.port)
        tools = MemoryTools(trial_cfg, ep.agent_id, project=ep.project)
        try:
            tools.recall("something")
        finally:
            tools.close()
        return HarnessResult(
            status="failed",
            last_reply=None,
            failure_category="timeout",
            transcript_path=inv.harness_home / "does-not-exist.jsonl",
        )

    rec_missing = run_trial(
        exp_home,
        "run-custom",
        task_dir,
        TrialConfig(
            arm="memory",
            model="local-model",
            model_base_url="http://127.0.0.1:8080/v1",
            trial_id="t-missing-transcript",
        ),
        adapter=ScriptedHarnessAdapter(missing_transcript_agent),
    )
    assert rec_missing.recall_accounting is not None
    ra_m = rec_missing.recall_accounting
    assert ra_m.transcript_import.status == "missing"
    assert ra_m.made_no_recall is False
    assert ra_m.total_recalls == 1
    assert ra_m.miss_count == 1
    assert ra_m.recalls[0].source == "ledger_only"

    # Empty baseline constant sanity check.
    assert TrialLedgerBaseline.empty().session_ids == frozenset()


def test_tool_errors_malformed_results_degraded_cite_and_corrupt_transcript(
    tmp_path: Path,
) -> None:
    """Tool errors and non-JSON recall results count as missing_record, never miss."""
    exp_home, task_dir = _make_repo_and_task(tmp_path)

    def error_edge_cases_agent(inv: HarnessInvocation) -> HarnessResult:
        transcript = _write_jsonl(
            inv.harness_home / ".factory" / "sessions" / "clone" / "sess-edges.jsonl",
            [
                {
                    "type": "session_start",
                    "id": "sess-edges",
                    "timestamp": "2026-11-26T13:00:00Z",
                    "cwd": str(inv.clone_dir),
                },
                # Recall returning an MCP error string (is_error=True) -> missing_record
                _tool_call_msg(
                    "e1",
                    "2026-11-26T13:00:01Z",
                    "call-err",
                    "agent-memory___recall",
                    {"query": "x"},
                ),
                _tool_result_msg(
                    "e2", "2026-11-26T13:00:02Z", "call-err", "MCP error: timeout", is_error=True
                ),
                # Recall returning non-JSON text without is_error -> missing_record(no_retrieval_id)
                _tool_call_msg(
                    "e3",
                    "2026-11-26T13:00:03Z",
                    "call-raw",
                    "agent-memory___recall",
                    {"query": "y"},
                ),
                _tool_result_msg("e4", "2026-11-26T13:00:04Z", "call-raw", "plain text response"),
                # Degraded cite call -> counted in cite_calls_in_transcript & degraded_cite_calls
                _tool_call_msg(
                    "e5", "2026-11-26T13:00:05Z", "call-cite-deg", "agent-memory___cite", {}
                ),
                _tool_result_msg(
                    "e6",
                    "2026-11-26T13:00:06Z",
                    "call-cite-deg",
                    json.dumps({"degraded": True, "reason": "db down", "links": 0}),
                ),
                _assistant_reply_msg("e7", "2026-11-26T13:00:07Z", "Handled errors."),
                {"type": "agent_turn_outcome", "turnId": "t-err", "resultKind": "text"},
            ],
        )
        return HarnessResult(
            status="completed",
            last_reply="Handled errors.",
            transcript_path=transcript,
            harness_session_id="sess-edges",
        )

    rec = run_trial(
        exp_home,
        "run-edges",
        task_dir,
        TrialConfig(
            arm="memory",
            model="local-model",
            model_base_url="http://127.0.0.1:8080/v1",
            trial_id="t-edges",
        ),
        adapter=ScriptedHarnessAdapter(error_edge_cases_agent),
    )
    assert rec.recall_accounting is not None
    ra = rec.recall_accounting
    assert ra.total_recalls == 2
    assert ra.miss_count == 0
    assert ra.degraded_count == 0
    assert ra.missing_record_count == 2
    assert [r.detail for r in ra.recalls] == ["tool_error", "no_retrieval_id"]
    assert ra.citations.cite_calls_in_transcript == 1
    assert ra.citations.degraded_cite_calls == 1
    assert ra.citations.citations_made == 0
