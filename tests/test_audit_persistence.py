"""Regression checks for persistence issues found by the whole-codebase audit."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from threading import Event

import duckdb
import pytest

from agent_memory import db
from agent_memory.capture import ids
from agent_memory.capture.pipeline import deliver, replay_spool
from agent_memory.capture.record import SessionInfo, now
from agent_memory.capture.redact import Redactor
from agent_memory.capture.spool import Spool
from agent_memory.capture.writer import Writer
from agent_memory.db import MemoryClient, MemoryServer, sql_literal
from agent_memory.mcp_server.tools import MemoryTools
from tests.test_writer import rec

SECRET = "synthetic-audit-sensitive-value"


def test_structured_credentials_use_existing_assignment_rules(down_cfg):
    payload = {
        "nested": [{"password": SECRET, "api_key": SECRET}],
        "description": SECRET,
        "token_estimate": 200,
        "short": "safe",
    }
    redactor = Redactor()
    clean, count = redactor.value(payload)
    assert clean["nested"] == [
        {"password": "[REDACTED:generic_assignment]", "api_key": "[REDACTED:generic_assignment]"}
    ]
    assert clean["description"] == SECRET
    assert clean["token_estimate"] == 200
    assert count == 2
    assert redactor.value(clean) == (clean, 0)
    assert Redactor(rules=[]).value(payload) == (payload, 0)
    exact = Redactor(extra_secrets=[SECRET])
    assert exact.value({"password": SECRET}) == ({"password": "[REDACTED:mem_token]"}, 1)
    (path,) = deliver(
        down_cfg,
        [rec("audit-fields", "note", "n", payload={"password": SECRET})],
        spool_only=True,
    ).spooled
    assert SECRET not in path.read_text()


@pytest.mark.parametrize("valid_json", [False, True])
def test_rejected_spool_lines_cannot_hide_escaped_secrets(live_cfg, live_client, valid_json):
    secret = live_cfg.read_token()
    escaped = f"\\u{ord(secret[0]):04x}" + secret[1:]
    spool = Spool(live_cfg.spool_dir)
    record = rec(f"audit-escaped-{valid_json}", "note", "good")
    (path,) = spool.append([record])
    line = '{"v":999,"payload":{"credential":"' + escaped + '"}}'
    with path.open("a") as stream:
        stream.write((line if valid_json else line + " malformed") + "\n")
    assert replay_spool(live_cfg).failed_lines == 1
    for directory in (spool.done_dir, spool.failed_dir):
        for archived in directory.rglob("*.jsonl"):
            for raw in archived.read_text().splitlines():
                # Decode JSON escapes, including those embedded in an invalid record.
                assert secret not in raw.replace(f"\\u{ord(secret[0]):04x}", secret[0])
    sid = ids.session_id(record.agent_id, record.harness_session_id)
    rows = live_client.query(
        f"SELECT payload::JSON::VARCHAR FROM events WHERE session_id = {sql_literal(sid)}"
    )[1]
    assert secret not in repr([json.loads(row[0]) for row in rows if row[0]])


def test_poison_recovery_keeps_reversed_result_parent_link(live_cfg, live_client):
    call = rec("audit-poison-order", "tool_call", "tool")
    result = replace(call, kind="tool_result", parent_key="tool")
    poison = replace(call, kind="memory_write", source_key="poison", memory={"content": ""})
    Spool(live_cfg.spool_dir).append([result, call, poison])
    assert replay_spool(live_cfg).failed_lines == 1
    eid = ids.event_id(call.agent_id, call.harness_session_id, call.kind, call.source_key)
    sid = ids.session_id(call.agent_id, call.harness_session_id)
    assert live_client.query(
        "SELECT parent_event_id::VARCHAR FROM events "
        f"WHERE session_id = {sql_literal(sid)} AND kind = 'tool_result'"
    )[1] == [(eid,)]


def test_poison_recovery_does_not_write_children_of_a_failed_parent(live_cfg, live_client):
    parent = rec("audit-failed-parent", "tool_call", "tool", memory={"content": ""})
    result = replace(parent, kind="tool_result", memory=None, parent_key="tool")
    healthy = rec("audit-failed-parent", "note", "healthy")
    Spool(live_cfg.spool_dir).append([result, healthy, parent])
    assert replay_spool(live_cfg).failed_lines == 2
    sid = ids.session_id(parent.agent_id, parent.harness_session_id)
    assert live_client.query(
        f"SELECT kind FROM events WHERE session_id = {sql_literal(sid)} AND kind != 'capture_error'"
    )[1] == [("note",)]


def test_poison_recovery_retains_cycles_but_writes_unrelated_records(live_cfg, live_client):
    cyclic = rec("audit-replay-cycle", "note", "cycle", parent_key="cycle", parent_kind="note")
    healthy = rec("audit-replay-cycle", "note", "healthy")
    Spool(live_cfg.spool_dir).append([cyclic, healthy])
    assert replay_spool(live_cfg).failed_lines == 1
    sid = ids.session_id(cyclic.agent_id, cyclic.harness_session_id)
    assert live_client.query(
        f"SELECT content FROM events WHERE session_id = {sql_literal(sid)} AND kind = 'note'"
    )[1] == [("note healthy",)]


def test_spool_redacts_every_persisted_record_copy(live_cfg, live_client):
    secret = live_cfg.read_token()
    record = rec(
        "audit-private",
        "memory_write",
        "m1",
        content=secret,
        payload={"nested": [secret], secret: "sensitive key"},
        raw={"input": secret},
        session=SessionInfo(cwd=f"/tmp/{secret}", metadata={"nested": [secret]}),
        memory={"content": secret, "metadata": {"nested": [secret]}},
    )
    delivery = deliver(live_cfg, [record], spool_only=True)
    assert secret not in delivery.spooled[0].read_text()
    replay_spool(live_cfg)
    for table in ("events", "sessions", "memories"):
        assert secret not in repr(live_client.query(f"SELECT * FROM {table}")[1])
    sid = ids.session_id(record.agent_id, record.harness_session_id)
    ((redacted, payload),) = live_client.query(
        "SELECT redactions > 0, payload::JSON::VARCHAR FROM events "
        f"WHERE session_id = {sql_literal(sid)}"
    )[1]
    assert redacted
    # The raw input is stored without the secret, not left out.
    assert json.loads(payload)["raw"] == {"input": "[REDACTED:mem_token]"}


def test_writer_redacts_queued_memory_and_action(live_cfg, live_client):
    writer = Writer(live_client, live_cfg.blob_dir, Redactor(extra_secrets=[SECRET]))
    memory = rec(
        "audit-writer-private",
        "memory_write",
        "m",
        memory={"content": "safe", "key": SECRET, "metadata": {"nested": [SECRET]}},
    )
    mid = ids.memory_id(memory.agent_id, memory.harness_session_id, memory.source_key)
    action = replace(
        memory,
        source_key="a",
        memory=None,
        action={"action": "confirm", "memory_id": mid, "reason": SECRET},
    )
    writer.write([memory, action])
    for table in ("memories", "memory_actions"):
        assert SECRET not in repr(live_client.query(f"SELECT * FROM {table}")[1])


def test_replay_redacts_archived_and_failed_lines(live_cfg, live_client):
    secret = live_cfg.read_token()
    spool = Spool(live_cfg.spool_dir)
    (path,) = spool.append(
        [
            rec("audit-old-spool", "note", "n", content=secret),
            rec(
                "audit-old-spool",
                "memory_write",
                "bad",
                memory={"content": "", "metadata": {"s": secret}},
            ),
        ]
    )
    with path.open("a") as stream:
        stream.write(f"not JSON {secret}\n")
    result = replay_spool(live_cfg)
    assert result.failed_lines == 2
    for directory in (spool.done_dir, spool.failed_dir):
        for archived in directory.rglob("*.jsonl"):
            assert secret not in archived.read_text()


@pytest.mark.parametrize("operation", ["recall", "graph", "review", "remember", "merge"])
def test_mcp_redacts_before_deriving_persisted_values(live_cfg, live_client, operation):
    with_tools = MemoryTools(
        live_cfg, "factory-droid", harness_session_id=f"audit-mcp-{operation}", project="audit"
    )
    with_tools._redactor = Redactor(extra_secrets=[SECRET])
    try:
        if operation == "recall":
            result = with_tools.recall(SECRET, scopes=[SECRET], types=[SECRET])
        elif operation == "graph":
            result = with_tools.graph(SECRET, scopes=[SECRET], types=[SECRET])
        elif operation == "review":
            memory = with_tools.remember("safe", "fact")
            result = with_tools.review("confirm", memory["memory_id"], SECRET)
        elif operation == "remember":
            result = with_tools.remember("safe", "fact", scope=SECRET, key=SECRET)
        else:
            with_tools.relate("old-service", "uses", "database")
            result = with_tools.merge_entity("old-service", SECRET, SECRET)
            assert result["superseded"]
        assert not result.get("degraded")
        assert not result.get("queued")
        for table in ("events", "retrievals", "memories", "memory_actions"):
            column = "created_by_session" if table == "memories" else "session_id"
            assert SECRET not in repr(
                live_client.query(
                    f"SELECT * FROM {table} WHERE {column} = {sql_literal(with_tools.session_id)}"
                )[1]
            )
    finally:
        with_tools.close()


def test_replay_does_not_archive_a_concurrent_append(live_cfg, live_client, monkeypatch):
    spool = Spool(live_cfg.spool_dir)
    first = rec("audit-append", "note", "first")
    second = replace(first, source_key="second")
    spool.append([first])
    original = Writer.write
    appended = False

    def interleave(writer, records):
        nonlocal appended
        if not appended:
            appended = True
            spool.append([second])
        return original(writer, records)

    monkeypatch.setattr(Writer, "write", interleave)
    replay_spool(live_cfg)
    assert sum(s.lines for s in spool.status()) == 1
    replay_spool(live_cfg)
    sid = ids.session_id(first.agent_id, first.harness_session_id)
    assert live_client.query(f"SELECT count(*) FROM events WHERE session_id = {sql_literal(sid)}")[
        1
    ] == [(2,)]


def test_claim_waits_for_an_appender_with_an_open_descriptor(live_cfg, monkeypatch):
    import agent_memory.capture.spool as spool_module

    spool = Spool(live_cfg.spool_dir)
    first = rec("audit-open-append", "note", "first")
    (path,) = spool.append([first])
    opened, release, claiming = Event(), Event(), Event()
    real_open, real_claim = spool_module.os.open, Spool.claim

    def pause_open(file, flags, *args, **kwargs):
        fd = real_open(file, flags, *args, **kwargs)
        if file == path:
            opened.set()
            assert release.wait(5)
        return fd

    def signal_claim(owner, file):
        claiming.set()
        return real_claim(owner, file)

    monkeypatch.setattr(spool_module.os, "open", pause_open)
    monkeypatch.setattr(Spool, "claim", signal_claim)
    with ThreadPoolExecutor(max_workers=2) as pool:
        append = pool.submit(spool.append, [replace(first, source_key="second")])
        assert opened.wait(5)
        replay = pool.submit(replay_spool, live_cfg)
        try:
            assert claiming.wait(5)
        finally:
            release.set()
        append.result(timeout=5)
        assert replay.result(timeout=5).inserted == 3
    assert spool.pending() == []


def test_overlapping_replay_skips_claimed_files(live_cfg, monkeypatch):
    spool = Spool(live_cfg.spool_dir)
    spool.append([rec("audit-overlap", "note", "n")])
    real_write = Writer.write
    nested = []

    def interleave(writer, records):
        nested.append(replay_spool(live_cfg).files)
        return real_write(writer, records)

    monkeypatch.setattr(Writer, "write", interleave)
    assert replay_spool(live_cfg).files == 1
    assert nested == [0]


def test_interrupted_replay_remains_recoverable(live_cfg, live_client, monkeypatch):
    spool = Spool(live_cfg.spool_dir)
    spool.append([rec("audit-interrupted", "note", "first")])
    original = Writer.write

    def interrupted(*args):
        raise RuntimeError("synthetic interruption")

    monkeypatch.setattr(Writer, "write", interrupted)
    with pytest.raises(RuntimeError, match="synthetic interruption"):
        replay_spool(live_cfg)
    spool.append([rec("audit-interrupted", "note", "second")])
    monkeypatch.setattr(Writer, "write", original)
    result = replay_spool(live_cfg)
    assert result.inserted == 3
    assert spool.pending() == []


def test_queued_review_redacts_before_spooling_and_replay(down_cfg):
    secret = down_cfg.read_token()
    server = MemoryServer(down_cfg, export_interval_hours=0)
    tools = MemoryTools(down_cfg, "factory-droid", harness_session_id="audit-queued-review")
    server.start()
    try:
        memory = tools.remember("safe queued review target", "fact")
    finally:
        tools.close()
        server.stop()
    result = tools.review("confirm", memory["memory_id"], secret)
    assert result["queued"]
    assert secret not in Path(result["spool"][0]).read_text()
    tools.close()
    server.start()
    try:
        replay_spool(down_cfg)
        with MemoryClient.from_config(down_cfg) as client:
            assert secret not in repr(client.query("SELECT * FROM memory_actions")[1])
    finally:
        server.stop()


def test_writer_layers_reversed_deep_dependencies(live_cfg, live_client):
    writer = Writer(live_client, live_cfg.blob_dir, Redactor())
    root = rec("audit-depth", "note", "root")
    child = replace(root, source_key="child", parent_key="root", parent_kind="note")
    leaf = replace(child, source_key="leaf", parent_key="child")
    assert writer.write([leaf, child, root]).events == 3
    assert writer.write([leaf, child, root]).skipped == 3


def test_writer_queries_only_incoming_events_and_their_parents(live_cfg, live_client, monkeypatch):
    writer = Writer(live_client, live_cfg.blob_dir, Redactor())
    history = [rec("audit-bounded-lookup", "tool_call", str(i)) for i in range(1100)]
    writer.write(history)
    real_query = live_client.query
    event_results = []

    def measured(sql):
        result = real_query(sql)
        if sql.startswith("SELECT event_id::VARCHAR FROM events"):
            assert "event_id IN (" in sql
            event_results.append(len(result[1]))
        return result

    monkeypatch.setattr(live_client, "query", measured)
    child = rec("audit-bounded-lookup", "tool_result", "0", parent_key="0")
    assert writer.write([child]).events == 1
    assert event_results == [1]
    event_results.clear()
    assert writer.write(history).skipped == 1100
    assert event_results == [500, 500, 100]


@pytest.mark.parametrize("size", [1, 3])
def test_writer_rejects_parent_cycles_before_inserting(live_cfg, live_client, size):
    writer = Writer(live_client, live_cfg.blob_dir, Redactor())
    records = [
        rec(
            f"audit-cycle-{size}",
            "note",
            str(i),
            parent_key=str((i + 1) % size),
            parent_kind="note",
        )
        for i in range(size)
    ]
    result = writer.write(records)
    assert [r.index for r in result.rejected] == list(range(size))
    assert all("cycl" in r.reason for r in result.rejected)
    sid = ids.session_id(records[0].agent_id, records[0].harness_session_id)
    assert live_client.query(
        f"SELECT count(*) FROM sessions WHERE session_id = {sql_literal(sid)}"
    )[1] == [(0,)]


def test_queued_memories_and_actions_deduplicate_in_one_batch(live_cfg, live_client):
    writer = Writer(live_client, live_cfg.blob_dir, Redactor())
    memory = rec("audit-duplicate", "memory_write", "m", memory={"content": "one fact"})
    mid = ids.memory_id(memory.agent_id, memory.harness_session_id, memory.source_key)
    action = replace(
        memory, source_key="a", memory=None, action={"memory_id": mid, "action": "confirm"}
    )
    result = writer.write([memory, memory, action, action])
    assert (result.events, result.memories, result.actions, result.skipped) == (2, 1, 1, 2)
    assert writer.write([memory, action]).inserted == 0
    assert live_client.query(
        f"SELECT count(*) FROM memory_sources WHERE memory_id = {sql_literal(mid)}"
    )[1] == [(1,)]


@pytest.mark.parametrize("payload", [{"payload": {"bad": object()}}, {"seq": "not an integer"}])
def test_failed_stage_cleans_up_and_client_can_retry(live_client, payload):
    for _ in range(2):
        with pytest.raises((TypeError, duckdb.Error)):
            live_client.insert({"events": [payload]})
        assert (
            live_client.con.execute(
                "SELECT table_name FROM duckdb_tables() "
                "WHERE temporary AND table_name LIKE 'stage_%'"
            ).fetchall()
            == []
        )
    sid = live_client.new_ids()[0]
    live_client.insert(
        {
            "sessions": [
                {
                    "session_id": sid,
                    "agent_id": "factory-droid",
                    "capture_completeness": "full",
                    "started_at": now(),
                }
            ],
            "events": [
                {
                    "event_id": live_client.new_ids()[0],
                    "session_id": sid,
                    "kind": "note",
                    "content": "valid after failed stage",
                    "seq": 0,
                    "ts": now(),
                }
            ],
        }
    )
    assert live_client.query(f"SELECT content FROM events WHERE session_id = {sql_literal(sid)}")[
        1
    ] == [("valid after failed stage",)]


@pytest.mark.parametrize("failure", ["copy", "publish"])
def test_failed_export_cleans_only_its_staging(live_client, tmp_path, monkeypatch, failure):
    previous = live_client.export(tmp_path)
    unrelated = tmp_path / ".tmp-unrelated"
    unrelated.mkdir()
    real_replace = db.os.replace
    if failure == "copy":
        monkeypatch.setattr(db, "ALL_TABLES", ("agents", "missing_audit_table"))
    else:

        def fail_publish(source, target):
            if Path(target).name == "latest":
                raise OSError("synthetic publication failure")
            return real_replace(source, target)

        monkeypatch.setattr(db.os, "replace", fail_publish)
    with pytest.raises((OSError, duckdb.Error)):
        live_client.export(tmp_path)
    assert (tmp_path / "latest").resolve() == previous
    assert set(tmp_path.glob(".tmp-*")) == {unrelated}
    assert not list(tmp_path.glob(".latest-*"))
    monkeypatch.undo()
    assert live_client.export(tmp_path).is_dir()
