"""Lifecycle contracts through real MCP, capture and Quack connections."""

import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from threading import Barrier
from uuid import uuid4

import duckdb
import pytest

from agent_memory.capture.pipeline import deliver, replay_spool
from agent_memory.capture.spool import Spool
from agent_memory.db import AgentMemoryError, MemoryClient, MemoryServer
from agent_memory.lifecycle import maintain_corrections
from agent_memory.mcp_server.tools import MemoryTools, ToolError
from agent_memory.review import reviewer


@pytest.fixture
def tools(live_cfg):
    tool = MemoryTools(live_cfg, "factory-droid", project=f"lifecycle-{uuid4()}")
    yield tool
    tool.close()


@pytest.fixture
def human(live_cfg):
    tool = reviewer(live_cfg)
    yield tool
    tool.close()


def states(tools, *mids):
    client = tools._client_or_raise()
    return [tools._status_of(client, mid) for mid in mids]


def test_unapproved_proposal_keeps_original_until_approval_and_never_revives(tools, human):
    old = tools.remember("original", "fact")["memory_id"]
    new = tools.remember("replacement", "fact", supersedes=old, trust_level="tool")
    mid = new["memory_id"]
    assert states(tools, old, mid) == ["active", "quarantined"]
    assert new["pending_reason"] == "awaiting_approval"
    tools.review("confirm", mid, "evidence only")
    assert states(tools, old, mid) == ["active", "quarantined"]
    human.review("approve", mid, "verified correction")
    assert states(tools, old, mid) == ["superseded", "active"]
    tools.review("quarantine", mid, "withdraw approval")
    tools.review("retire", mid, "withdraw correction")
    tools.review("confirm", old, "not restoration")
    assert states(tools, old, mid) == ["superseded", "retired"]
    assert tools.recall("")["items"] == []


def test_future_and_valid_to_are_not_recalled(tools):
    future = (datetime.now(UTC) + timedelta(days=1)).isoformat()
    past = (datetime.now(UTC) - timedelta(days=1)).isoformat()
    old = tools.remember("original", "fact")["memory_id"]
    new = tools.remember("future replacement", "fact", supersedes=old, valid_from=future)
    ended = tools.remember("ended", "fact", valid_to=past)
    assert states(tools, old, new["memory_id"], ended["memory_id"]) == [
        "active",
        "pending",
        "expired",
    ]
    assert new["pending_reason"] == "valid_from"
    assert [item["memory_id"] for item in tools.recall("")["items"]] == [old]


def test_dedupe_preserves_correction_intent_and_rejects_stale_or_self(tools):
    old = tools.remember("original", "fact")["memory_id"]
    canonical = tools.remember("canonical", "fact")["memory_id"]
    result = tools.remember("canonical", "fact", supersedes=old)
    assert result["memory_id"] == canonical
    assert result["deduplicated"]
    assert states(tools, old, canonical) == ["superseded", "active"]
    with pytest.raises(ToolError, match=r"stale|conflict"):
        tools.remember("competitor", "fact", supersedes=old)
    with pytest.raises(ToolError, match="self"):
        tools.remember("canonical", "fact", supersedes=canonical)
    assert tools.status()["spool_pending_lines"] == 0


def test_invalid_intervals_are_terminal_even_offline(down_cfg):
    tool = MemoryTools(down_cfg, "factory-droid")
    with pytest.raises(ToolError, match="interval"):
        tool.remember("bad range", "fact", valid_from="2026-01-02", valid_to="2026-01-01")
    assert tool.status()["spool_pending_lines"] == 0
    tool.close()


def test_queued_identity_resolves_dedup_only_after_restart(down_cfg):
    server = MemoryServer(down_cfg, export_interval_hours=0)
    tool = MemoryTools(down_cfg, "factory-droid", project="queue")
    queued = tool.remember("identical", "fact", trust_level="tool")["memory_id"]
    human = reviewer(down_cfg)
    assert human.review("approve", queued, "approve queued identity")["queued"]
    human.close()
    server.start()
    try:
        online = MemoryTools(down_cfg, "factory-droid", project="queue")
        canonical = online.remember("identical", "fact", trust_level="tool")["memory_id"]
        online.close()
        assert replay_spool(down_cfg).failed_lines == 0
        with MemoryClient.from_config(down_cfg) as client:
            _, rows = client.query(
                f"SELECT memory_id::VARCHAR FROM memories WHERE memory_id = '{queued}'"
            )
            assert rows == [(queued,)]
        assert states(tool, queued, canonical) == ["deduplicated", "active"]
        replacement = tool.remember("next", "fact", supersedes=canonical)["memory_id"]
        tool.review("quarantine", queued, "must review canonical, NOT successor")
        assert states(tool, canonical, replacement) == ["superseded", "active"]
        assert replay_spool(down_cfg).files == 0
    finally:
        tool.close()
        server.stop()


def wait_until(at):
    time.sleep(max(0, (at - datetime.now(UTC)).total_seconds()) + 0.02)


def waiting_correction(old, write, leads=(1.2, 5.0, 20.0, 80.0)):
    """Write a correction of `old` that waits for its time, on a machine of any speed.

    `write(old, due)` writes one dated `due` and returns its id and the status the ledger
    then gives it. `pending` says the writes were there first. Any other status says the
    time had already come: a correction that arrives then is applied at once and never
    waits as due work, so it becomes the memory to correct, with more time ahead.

    Returns the memory corrected, its correction, and the time the correction comes due.
    """
    for lead in leads:
        due = datetime.now(UTC) + timedelta(seconds=lead)
        new, status = write(old, due)
        if status == "pending":
            return old, new, due
        old = new
    raise AssertionError(f"a correction dated {leads[-1]} s ahead did not wait")


def test_due_approval_withdrawal_and_reapproval(tools, human):
    def propose_and_approve(old, due):
        new = tools.remember(
            f"due {due}", "fact", supersedes=old, trust_level="tool", valid_from=due.isoformat()
        )["memory_id"]
        return new, human.review("approve", new, "approved future correction")["status"]

    old, new, due = waiting_correction(
        tools.remember("old", "fact")["memory_id"], propose_and_approve
    )
    tools.review("quarantine", new, "withdraw before due")
    wait_until(due)
    assert [x["memory_id"] for x in tools.recall("")["items"]] == [old]
    assert states(tools, old, new) == ["active", "quarantined"]
    human.review("approve", new, "approve after boundary")
    assert states(tools, old, new) == ["superseded", "active"]


def test_startup_accepts_due_work_before_graph(down_cfg):
    server = MemoryServer(down_cfg, export_interval_hours=0)
    server.start()
    tool = MemoryTools(down_cfg, "factory-droid", project="startup")
    due = datetime.now(UTC) + timedelta(seconds=1.2)
    try:
        old = tool.relate("A", "owns", "B")["memory_id"]
        new = tool.remember("replaced edge", "fact", supersedes=old, valid_from=due.isoformat())[
            "memory_id"
        ]
    finally:
        tool.close()
        server.stop()
    wait_until(due)
    server = MemoryServer(down_cfg, export_interval_hours=0)
    server.start()
    try:
        assert states(tool, old, new) == ["superseded", "active"]
        assert tool.graph("A")["items"] == []
    finally:
        tool.close()
        server.stop()


def test_bounded_due_batch_never_exposes_two_versions(down_cfg):
    server = MemoryServer(down_cfg, export_interval_hours=0)
    server.start()
    tool = MemoryTools(down_cfg, "factory-droid", project="bounded")

    def propose(old, due):
        new = tool.remember(f"due {due}", "fact", supersedes=old, valid_from=due.isoformat())
        return new["memory_id"], new["status"]

    try:
        waiting = [
            waiting_correction(tool.remember(f"old {index}", "fact")["memory_id"], propose)
            for index in range(3)
        ]
        pairs = [(old, new) for old, new, _ in waiting]
        wait_until(max(due for _, _, due in waiting))
        client = tool._client_or_raise()
        assert maintain_corrections(client, limit=1) == 1
        assert sum(states(tool, old, new) == ["superseded", "active"] for old, new in pairs) == 1
        assert sum(states(tool, old, new) == ["active", "pending"] for old, new in pairs) == 2
        assert len(tool.recall("")["items"]) == 3
        assert all(states(tool, old, new) == ["superseded", "active"] for old, new in pairs)
    finally:
        tool.close()
        server.stop()


def test_no_due_work_path_does_not_query_memories(tools, monkeypatch):
    client = tools._client_or_raise()
    queries = []
    query = client.query

    def observed(sql):
        queries.append(sql)
        return query(sql)

    monkeypatch.setattr(client, "query", observed)
    maintain_corrections(client)
    assert len(queries) == 1
    assert "FROM pending_memory_corrections" in queries[0]
    assert "LIMIT 128" in queries[0]


def test_cross_scope_cycles_and_repeated_pending_proposals(tools, human):
    old = tools.remember("old", "fact")["memory_id"]
    with pytest.raises(ToolError, match="cross-scope"):
        tools.remember("new", "fact", supersedes=old, scope="another-scope")
    new = tools.remember("new", "fact", supersedes=old, trust_level="tool")["memory_id"]
    with pytest.raises(ToolError, match=r"cycle|pending predecessor"):
        tools.remember("old", "fact", supersedes=new)
    assert tools.remember("new", "fact", supersedes=old, trust_level="tool")["memory_id"] == new
    human.review("approve", new, "approve once")
    assert states(tools, old, new) == ["superseded", "active"]
    assert tools.status()["spool_pending_lines"] == 0


def test_dedupe_retains_sources_without_restoring_retired_memory(tools, human):
    source = tools.recall("")["request_event_id"]
    original = tools.remember("same", "fact")["memory_id"]
    tools.review("retire", original, "terminal")
    result = tools.remember("same", "fact", source_event_ids=[source])
    assert result["memory_id"] == original
    assert result["status"] == "retired"
    _, rows = tools._client_or_raise().query(
        f"SELECT event_id::VARCHAR FROM memory_sources WHERE memory_id = '{original}'"
    )
    assert (source,) in rows
    assert len(rows) == 3
    assert human.review("approve", original, "not restoration")["status"] == "retired"


def test_concurrent_acceptance_has_one_winner(tools, live_cfg, monkeypatch):
    old = tools.remember("original", "fact")["memory_id"]
    barrier = Barrier(2)
    insert = MemoryClient.insert

    def synchronized(client, rows):
        pairs = list(rows.items()) if isinstance(rows, dict) else list(rows)
        if any(
            table == "memories" and any(r.get("supersedes") == old for r in rs)
            for table, rs in pairs
        ):
            barrier.wait(timeout=10)
        return insert(client, pairs)

    monkeypatch.setattr(MemoryClient, "insert", synchronized)

    def compete(index):
        tool = MemoryTools(
            live_cfg, "factory-droid", project=tools.project, harness_session_id=str(uuid4())
        )
        try:
            return tool.remember(f"competitor {index}", "fact", supersedes=old)
        except ToolError as exc:
            return exc
        finally:
            tool.close()

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(compete, range(2)))
    winners = [r for r in results if isinstance(r, dict)]
    losers = [r for r in results if isinstance(r, ToolError)]
    assert len(winners) == len(losers) == 1
    assert not winners[0]["queued"]
    assert "conflict" in str(losers[0])
    assert states(tools, old, winners[0]["memory_id"]) == ["superseded", "active"]
    assert tools.status()["spool_pending_lines"] == 0
    assert len(tools.recall("")["items"]) == 1


def test_failure_after_acceptance_rolls_back_every_write(tools, monkeypatch):
    old = tools.remember("old", "fact")["memory_id"]
    client = tools._client_or_raise()
    insert = client.insert
    attempted = []

    def fail_after_acceptance(rows):
        pairs = list(rows)
        new = next(rs[0]["memory_id"] for table, rs in pairs if table == "memories")
        attempted.append(new)
        pairs.append(
            (
                "memory_sources",
                [
                    {
                        "memory_id": new,
                        "event_id": str(uuid4()),
                        "relation": "supports",
                    }
                ],
            )
        )
        insert(pairs)

    monkeypatch.setattr(client, "insert", fail_after_acceptance)
    with pytest.raises(ToolError, match="foreign key"):
        tools.remember("failed replacement", "fact", supersedes=old)
    assert states(tools, old, attempted[0]) == ["active", None]
    _, rows = client.query(f"SELECT 1 FROM memory_replacements WHERE predecessor_id = '{old}'")
    assert rows == []


def test_replay_lost_commit_ack_remains_pending_then_idempotent(down_cfg, monkeypatch):
    tool = MemoryTools(down_cfg, "factory-droid", project="lost-ack")
    mid = tool.remember("queued original", "fact")["memory_id"]
    successor = tool.remember("queued correction", "fact", supersedes=mid)["memory_id"]
    server = MemoryServer(down_cfg, export_interval_hours=0)
    server.start()
    insert = MemoryClient.insert
    lost = False

    def lost_ack(client, rows_by_table):
        nonlocal lost
        insert(client, rows_by_table)
        if not lost:
            lost = True
            raise AgentMemoryError("synthetic lost commit acknowledgement")

    try:
        with monkeypatch.context() as patch:
            patch.setattr(MemoryClient, "insert", lost_ack)
            with pytest.raises(AgentMemoryError, match="lost commit"):
                replay_spool(down_cfg)
        assert Spool(down_cfg.spool_dir).pending()
        result = replay_spool(down_cfg)
        assert result.failed_lines == 0
        assert states(tool, mid, successor) == ["superseded", "active"]
        _, rows = tool._client_or_raise().query(
            f"SELECT 1 FROM memory_replacements WHERE predecessor_id = '{mid}'"
        )
        assert len(rows) == 1
    finally:
        tool.close()
        server.stop()


def test_merge_canonical_preserves_alias_sources_approval_and_valid_to(tools, human):
    end = (datetime.now(UTC) + timedelta(days=1)).isoformat()
    source = tools.recall("")["request_event_id"]
    incoming = tools.relate(
        "Alias", "owns", "Service", trust_level="tool", valid_to=end, source_event_ids=[source]
    )["memory_id"]
    human.review("approve", incoming, "verified")
    merged = tools.merge_entity("Alias", "Canonical", "same entity")
    canonical = merged["superseded"][0]["by"]
    _, rows = tools._client_or_raise().query(
        f"SELECT valid_to FROM memories WHERE memory_id = '{canonical}'"
    )
    assert rows[0][0] == datetime.fromisoformat(end)
    second = tools.relate("Other alias", "owns", "Service", source_event_ids=[source])
    tools.merge_entity("Other alias", "Canonical", "same again")
    item = tools.graph("Canonical")["items"][0]
    assert item["memory_id"] == canonical
    assert set(item["aliases"]) == {"Alias", "Other alias"}
    _, sources = tools._client_or_raise().query(
        f"SELECT event_id FROM memory_sources WHERE memory_id = '{canonical}'"
    )
    assert len(sources) == 5  # two writes, two merges, and the shared source
    assert states(tools, incoming, canonical, second["memory_id"]) == [
        "superseded",
        "active",
        "retired",
    ]


def test_acceptance_guard_rejects_forged_legacy_method(tools):
    old = tools.remember("old", "fact")["memory_id"]
    new = tools.remember("new", "fact", supersedes=old, trust_level="tool")["memory_id"]
    client = tools._client_or_raise()
    _, rows = client.query(
        "SELECT action_id::VARCHAR, event_id::VARCHAR FROM memory_actions "
        f"WHERE memory_id = '{new}' AND action = 'propose_correction'"
    )
    with pytest.raises(duckdb.Error, match="invalid acceptance"):
        client.insert(
            {
                "memory_replacements": [
                    {
                        "predecessor_id": old,
                        "successor_id": new,
                        "proposal_id": rows[0][0],
                        "evidence_event_id": rows[0][1],
                        "method": "legacy_trusted_at_creation",
                    }
                ]
            }
        )
    assert states(tools, old, new) == ["active", "quarantined"]


def test_dedup_correction_keeps_future_intent(tools):
    old = tools.remember("original", "fact")["memory_id"]
    canonical = tools.remember("canonical", "fact")["memory_id"]
    future = (datetime.now(UTC) + timedelta(days=1)).isoformat()
    result = tools.remember("canonical", "fact", supersedes=old, valid_from=future)
    assert result["memory_id"] == canonical
    assert result["correction_status"] == "pending"
    assert result["correction_pending_reason"] == "valid_from"
    assert states(tools, old, canonical) == ["active", "active"]


def test_dedup_tool_proposal_cannot_borrow_agent_default_trust(tools, human):
    old = tools.remember("original", "fact")["memory_id"]
    canonical = tools.remember("canonical", "fact")["memory_id"]
    result = tools.remember("canonical", "fact", supersedes=old, trust_level="tool")
    assert result["correction_pending_reason"] == "awaiting_approval"
    assert states(tools, old, canonical) == ["active", "active"]
    tools.review("confirm", canonical, "not approval")
    assert states(tools, old, canonical) == ["active", "active"]
    human.review("approve", canonical, "approve correction")
    assert states(tools, old, canonical) == ["superseded", "active"]


def test_explicit_correction_can_restore_old_content_with_new_identity(tools):
    old = tools.remember("original", "fact")["memory_id"]
    changed = tools.remember("changed", "fact", supersedes=old)["memory_id"]
    restored = tools.remember("original", "fact", supersedes=changed)["memory_id"]
    assert restored not in (old, changed)
    assert states(tools, old, changed, restored) == ["superseded", "superseded", "active"]


def test_correcting_pending_predecessor_cannot_expose_two_versions(tools):
    old = tools.remember("old", "fact")["memory_id"]
    future = (datetime.now(UTC) + timedelta(days=1)).isoformat()
    pending = tools.remember("future", "fact", supersedes=old, valid_from=future)["memory_id"]
    with pytest.raises(ToolError, match="pending predecessor"):
        tools.remember("newer", "fact", supersedes=pending)
    assert [x["memory_id"] for x in tools.recall("")["items"]] == [old]


def test_replay_identical_already_accepted_correction_keeps_queued_identity(down_cfg):
    server = MemoryServer(down_cfg, export_interval_hours=0)
    server.start()
    tool = MemoryTools(down_cfg, "factory-droid", project="accepted-dedup")
    old = tool.remember("old", "fact")["memory_id"]
    tool.close()
    server.stop()
    queued = tool.remember("new", "fact", supersedes=old)["memory_id"]
    server = MemoryServer(down_cfg, export_interval_hours=0)
    server.start()
    try:
        canonical = tool.remember("new", "fact", supersedes=old)["memory_id"]
        assert replay_spool(down_cfg).failed_lines == 0
        assert states(tool, old, queued, canonical) == ["superseded", "deduplicated", "active"]
    finally:
        tool.close()
        server.stop()


def test_replay_rejects_an_approval_from_an_agents_session(down_cfg):
    tool = MemoryTools(down_cfg, "factory-droid", project="forged")
    held = tool.remember("read from a tool", "fact", trust_level="tool")["memory_id"]
    doubted = tool.remember("an inference", "fact")["memory_id"]
    # What a process holding the agent's spool could append without the review tool.
    spooled = [
        tool.session.record(
            "memory_write",
            f"spooled-{action}",
            role="memory",
            action={"memory_id": mid, "action": action, "actor": "human", "reason": "looks right"},
        )
        for mid, action in ((held, "approve"), (doubted, "retire"))
    ]
    deliver(down_cfg, spooled, spool_only=True)
    server = MemoryServer(down_cfg, export_interval_hours=0)
    server.start()
    try:
        result = replay_spool(down_cfg)
        assert result.failed_lines == 1
        assert "only a human approves a memory" in result.errors[0]
        assert Spool(down_cfg.spool_dir).rejected()[0].lines == 1
        assert states(tool, held, doubted) == ["quarantined", "retired"]
        # A queued review is its session's agent's, whoever the record says asked.
        _, rows = tool._client_or_raise().query(
            "SELECT action, actor FROM memory_actions "
            f"WHERE memory_id IN ('{held}', '{doubted}') AND action IN ('approve', 'retire')"
        )
        assert rows == [("retire", "factory-droid")]
    finally:
        tool.close()
        server.stop()


def test_replay_dependencies_across_interleaved_session_files(down_cfg):
    first = reviewer(down_cfg)
    second = MemoryTools(down_cfg, "factory-droid", harness_session_id="second", project="shared")
    old = first.remember("cross-session original", "fact", scope="shared")["memory_id"]
    new = second.remember("cross-session correction", "fact", supersedes=old, trust_level="tool")[
        "memory_id"
    ]
    # Appending to the first file makes its mtime newer, and makes file-level
    # ordering cyclic even though the individual records have no cycle.
    first.review("approve", new, "approve the other session's correction")
    server = MemoryServer(down_cfg, export_interval_hours=0)
    server.start()
    try:
        result = replay_spool(down_cfg)
        assert result.failed_lines == 0
        assert states(first, old, new) == ["superseded", "active"]
        assert replay_spool(down_cfg).files == 0
    finally:
        first.close()
        second.close()
        server.stop()
