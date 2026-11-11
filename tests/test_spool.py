"""Spool and pipeline: outage handling, replay, bad lines, redaction on replay."""

from __future__ import annotations

import os
import threading
import time
from datetime import UTC, datetime
from typing import Any

import pytest

from agent_memory.capture import ids
from agent_memory.capture.pipeline import deliver, replay_spool
from agent_memory.capture.record import Record
from agent_memory.capture.spool import Spool
from agent_memory.capture.writer import Writer
from agent_memory.db import AgentMemoryError, MemoryClient, MemoryServer, sql_literal

NOW = datetime(2026, 10, 25, 12, 0, tzinfo=UTC)


def rec(hsid: str, key: str, **over: Any) -> Record:
    base: dict[str, Any] = {
        "agent_id": "factory-droid",
        "harness_session_id": hsid,
        "kind": "user_msg",
        "ts": NOW,
        "source_key": key,
        "content": f"msg {key}",
    }
    return Record(**{**base, **over})


def test_spool_layout_and_status(tmp_path):
    spool = Spool(tmp_path / "spool")
    paths = spool.append([rec("s/1", "a"), rec("s/1", "b"), rec("other", "c", agent_id="amp")])
    assert sorted(p.relative_to(spool.root).as_posix() for p in paths) == [
        "amp/other.jsonl",
        "factory-droid/s_1.jsonl",
    ]
    assert (paths[0].stat().st_mode & 0o777) == 0o600
    status = {s.agent_id: (s.files, s.lines) for s in spool.status()}
    assert status == {"amp": (1, 1), "factory-droid": (1, 2)}
    pending = spool.pending()
    assert [(f.agent_id, f.harness_session_id) for f in pending] == [
        ("amp", "other"),
        ("factory-droid", "s_1"),
    ]
    done = spool.mark_done(pending[0])
    assert done.parent == spool.done_dir / "amp"
    assert spool.status()[0].agent_id == "factory-droid"
    old = time.time() - 40 * 86400
    os.utime(done, (old, old))
    assert spool.purge_done(30) == [done]


def test_deliver_spools_when_the_server_is_down(down_cfg):
    delivery = deliver(
        down_cfg, [rec("d-1", "a", content="token ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghij0123")]
    )
    assert delivery.written is None
    assert delivery.error is not None
    assert "Cannot connect" in delivery.error
    (path,) = delivery.spooled
    text = path.read_text()
    assert "ghp_" not in text  # redacted before it touches disk
    assert "[REDACTED:github_token]" in text


def test_deliver_times_out_into_the_spool(down_cfg, monkeypatch):
    import agent_memory.capture.pipeline as pipeline

    class Hangs:
        def __init__(self, *a, **k):
            time.sleep(0.5)
            raise RuntimeError("unreachable")

    monkeypatch.setattr(pipeline.MemoryClient, "from_config", staticmethod(lambda cfg: Hangs()))
    delivery = deliver(down_cfg, [rec("d-2", "a")], deadline=0.05)
    assert delivery.spooled
    assert delivery.error == "timed out waiting for the server"


def test_spool_only_skips_the_server(live_cfg, live_client):
    delivery = deliver(live_cfg, [rec("so-1", "a")], spool_only=True)
    assert delivery.spooled
    assert delivery.written is None
    sid = ids.session_id("factory-droid", "so-1")
    assert live_client.query(
        f"SELECT count(*) FROM sessions WHERE session_id = {sql_literal(sid)}"
    )[1] == [(0,)]


def test_replay_lands_everything_and_moves_files(down_cfg):
    deliver(down_cfg, [rec("r-1", "a"), rec("r-1", "b")])
    deliver(down_cfg, [rec("r-2", "c", agent_id="amp")])
    server = MemoryServer(down_cfg, export_interval_hours=0)
    server.start()
    try:
        result = replay_spool(down_cfg)
        assert (result.files, result.inserted, result.skipped, result.failed_lines) == (2, 5, 0, 0)
        with MemoryClient.from_config(down_cfg) as client:
            assert client.query("SELECT count(*) FROM events")[1] == [(3,)]
        again = replay_spool(down_cfg)
        assert again.files == 0
    finally:
        server.stop()
    spool = Spool(down_cfg.spool_dir)
    assert spool.pending() == []
    assert len(list(spool.done_dir.rglob("*.jsonl"))) == 2


def test_bad_lines_become_capture_errors_and_are_kept(down_cfg):
    spool = Spool(down_cfg.spool_dir)
    (path,) = spool.append([rec("r-bad", "ok")])
    with path.open("a") as fh:
        fh.write("this is not a record\n")
        fh.write(
            '{"v": 1, "agent_id": "factory-droid", "harness_session_id": "r-bad", '
            '"kind": "nope", "ts": "2026-01-01T00:00:00Z", "source_key": "k"}\n'
        )
    server = MemoryServer(down_cfg, export_interval_hours=0)
    server.start()
    try:
        result = replay_spool(down_cfg)
        assert result.files == 1
        assert result.failed_lines == 2
        assert result.inserted == 4  # 1 session + 1 good event + 2 capture_error events
        failed = result.failed_paths[0]
        assert failed.parent == spool.failed_dir / "factory-droid"
        assert failed.read_text().count("\n") == 2
        with MemoryClient.from_config(down_cfg) as client:
            got = client.query(
                "SELECT content, payload::JSON::VARCHAR FROM events "
                "WHERE kind = 'capture_error' ORDER BY content"
            )[1]
        assert len(got) == 2
        assert any("not JSON" in c for c, _ in got)
        assert any("unknown event kind" in c for c, _ in got)
        assert all('"line":' in p for _, p in got)
    finally:
        server.stop()


def test_replay_applies_rules_added_after_spooling(down_cfg, monkeypatch):
    spool = Spool(down_cfg.spool_dir)
    spool.append([rec("r-late", "a", content="password: hunter2hunter2")])
    server = MemoryServer(down_cfg, export_interval_hours=0)
    server.start()
    try:
        replay_spool(down_cfg)
        with MemoryClient.from_config(down_cfg) as client:
            got = client.query("SELECT content, redactions FROM events")[1]
        assert got == [("password=[REDACTED:generic_assignment]", 1)]
    finally:
        server.stop()


def test_one_poison_record_does_not_block_the_file(down_cfg):
    spool = Spool(down_cfg.spool_dir)
    good = rec("r-poison", "good")
    poison = rec("r-poison", "bad", memory={"content": "", "type": "fact"})
    spool.append([good, poison])
    server = MemoryServer(down_cfg, export_interval_hours=0)
    server.start()
    try:
        result = replay_spool(down_cfg)
        assert result.failed_lines == 1
        assert result.errors
        with MemoryClient.from_config(down_cfg) as client:
            kinds = client.query("SELECT kind, count(*) FROM events GROUP BY kind ORDER BY kind")[1]
        assert kinds == [("capture_error", 1), ("user_msg", 1)]
    finally:
        server.stop()
    assert len(list(spool.failed_dir.rglob("*.jsonl"))) == 1


def hooks_inserting_together(monkeypatch) -> None:
    """Hold each connection's first insert until two writers have done their pre-reads."""
    barrier = threading.Barrier(2)
    real_insert = MemoryClient.insert
    held: set[int] = set()

    def insert_together(client, rows_by_table):
        if id(client) not in held:
            held.add(id(client))
            barrier.wait(10)
        return real_insert(client, rows_by_table)

    monkeypatch.setattr(MemoryClient, "insert", insert_together)


def test_two_hooks_racing_on_a_new_session_lose_nothing(live_cfg, live_client, monkeypatch):
    hsid = "d-race"
    hooks_inserting_together(monkeypatch)
    hooks = {
        "SessionStart": rec(hsid, "start", kind="session_start", payload={"hook": "SessionStart"}),
        "UserPromptSubmit": rec(hsid, "m1", payload={"hook": "UserPromptSubmit"}),
    }
    deliveries: dict[str, Any] = {}

    def run(name: str) -> None:
        try:
            deliveries[name] = deliver(live_cfg, [hooks[name]], deadline=30)
        except Exception as exc:
            deliveries[name] = exc

    threads = [threading.Thread(target=run, args=(name,)) for name in hooks]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(40)
    for name in hooks:
        delivery = deliveries[name]
        assert getattr(delivery, "written", None) is not None, (name, delivery)
        assert (delivery.written.rejected, delivery.spooled, delivery.failed_paths) == ([], [], [])
    sid = ids.session_id("factory-droid", hsid)
    assert live_client.query(
        f"SELECT kind FROM events WHERE session_id = {sql_literal(sid)} ORDER BY kind"
    )[1] == [("session_start",), ("user_msg",)]


def test_deliver_keeps_a_rejected_record_and_writes_the_rest(live_cfg, live_client):
    hsid = "d-rejected"
    poison = rec(hsid, "bad", kind="memory_write", memory={"content": "", "type": "fact"})
    delivery = deliver(live_cfg, [rec(hsid, "good"), poison], deadline=10)
    assert delivery.written is not None
    assert delivery.spooled == []
    assert [r.index for r in delivery.written.rejected] == [1]
    (kept,) = delivery.failed_paths
    assert kept == Spool(live_cfg.spool_dir).failed_dir / "factory-droid" / f"{hsid}.jsonl"
    assert [Record.from_json(line).source_key for line in kept.read_text().splitlines()] == ["bad"]
    sid = ids.session_id("factory-droid", hsid)
    events = live_client.query(
        f"SELECT kind, content FROM events WHERE session_id = {sql_literal(sid)} ORDER BY kind"
    )[1]
    assert [kind for kind, _ in events] == ["capture_error", "user_msg"]
    assert "no content" in events[0][1]


def test_a_delivery_the_ledger_refuses_entirely_is_kept_not_dropped(live_cfg, live_client):
    hsid = "d-all-rejected"
    poison = rec(hsid, "bad", kind="memory_write", memory={"content": ""})
    delivery = deliver(live_cfg, [poison], deadline=10)
    assert delivery.written is not None
    assert delivery.spooled == []
    (kept,) = delivery.failed_paths
    assert kept.read_text().count("\n") == 1
    sid = ids.session_id("factory-droid", hsid)
    assert live_client.query(f"SELECT kind FROM events WHERE session_id = {sql_literal(sid)}")[
        1
    ] == [("capture_error",)]


def test_a_capture_error_the_ledger_cannot_take_is_spooled(live_cfg, live_client, monkeypatch):
    hsid = "d-unreported"
    real_write = Writer.write
    calls: list[int] = []

    def fail_after_the_batch(writer, records):
        calls.append(len(records))
        if len(calls) == 2:
            raise AgentMemoryError("synthetic outage")
        return real_write(writer, records)

    poison = rec(hsid, "bad", kind="memory_write", memory={"content": ""})
    with monkeypatch.context() as patch:
        patch.setattr(Writer, "write", fail_after_the_batch)
        delivery = deliver(live_cfg, [rec(hsid, "good"), poison], deadline=10)
    assert delivery.written is not None
    assert len(delivery.failed_paths) == 1
    assert len(delivery.spooled) == 1
    assert "synthetic outage" in (delivery.error or "")
    sid = ids.session_id("factory-droid", hsid)
    kinds = f"SELECT kind FROM events WHERE session_id = {sql_literal(sid)} ORDER BY kind"
    assert live_client.query(kinds)[1] == [("user_msg",)]
    assert replay_spool(live_cfg).failed_lines == 0
    assert live_client.query(kinds)[1] == [("capture_error",), ("user_msg",)]


@pytest.mark.parametrize(
    "field",
    ['"payload": "not an object"', '"content": {"not": "text"}', '"memory": ["not an object"]'],
)
def test_a_line_with_the_wrong_shape_does_not_block_the_spool(down_cfg, field):
    spool = Spool(down_cfg.spool_dir)
    (path,) = spool.append([rec("r-shape", "ok")])
    with path.open("a") as fh:
        fh.write(
            '{"v": 1, "agent_id": "factory-droid", "harness_session_id": "r-shape", '
            f'"kind": "note", "ts": "2026-01-01T00:00:00Z", "source_key": "k", {field}}}\n'
        )
    server = MemoryServer(down_cfg, export_interval_hours=0)
    server.start()
    try:
        result = replay_spool(down_cfg)
        assert result.failed_lines == 1
        with MemoryClient.from_config(down_cfg) as client:
            kinds = client.query("SELECT kind, count(*) FROM events GROUP BY kind ORDER BY kind")[1]
        assert kinds == [("capture_error", 1), ("user_msg", 1)]
    finally:
        server.stop()
    assert spool.pending() == []


def test_a_record_that_breaks_the_writer_is_rejected_not_fatal(down_cfg):
    spool = Spool(down_cfg.spool_dir)
    odd = rec("r-odd", "odd", kind="memory_write", memory={"content": "x", "metadata": "oops"})
    spool.append([rec("r-odd", "good"), odd])
    server = MemoryServer(down_cfg, export_interval_hours=0)
    server.start()
    try:
        result = replay_spool(down_cfg)
        assert result.failed_lines == 1
        assert any("AttributeError" in error for error in result.errors)
        with MemoryClient.from_config(down_cfg) as client:
            kinds = client.query("SELECT kind, count(*) FROM events GROUP BY kind ORDER BY kind")[1]
        assert kinds == [("capture_error", 1), ("user_msg", 1)]
    finally:
        server.stop()
    assert spool.pending() == []


def test_rejected_records_return_to_the_spool_on_request(down_cfg):
    spool = Spool(down_cfg.spool_dir)
    memory = rec("r-retry", "m", kind="memory_write", memory={"content": "arrives late"})
    mid = ids.memory_id(memory.agent_id, memory.harness_session_id, memory.source_key)
    action = rec(
        "r-retry", "a", kind="memory_write", action={"memory_id": mid, "action": "confirm"}
    )
    spool.append([action])
    server = MemoryServer(down_cfg, export_interval_hours=0)
    server.start()
    try:
        # The action names a memory the ledger has not seen yet.
        assert replay_spool(down_cfg).failed_lines == 1
        assert [(s.agent_id, s.files, s.lines) for s in spool.rejected()] == [
            ("factory-droid", 1, 1)
        ]
        assert spool.retry_failed() == (1, 1)
        assert spool.rejected() == []
        assert [(s.agent_id, s.lines) for s in spool.status()] == [("factory-droid", 1)]
        spool.append([memory])
        assert replay_spool(down_cfg).failed_lines == 0
        with MemoryClient.from_config(down_cfg) as client:
            assert client.query(
                f"SELECT action FROM memory_actions WHERE memory_id = {sql_literal(mid)}"
            )[1] == [("confirm",)]
    finally:
        server.stop()
    assert spool.retry_failed() == (0, 0)


def test_retry_returns_a_file_kept_under_a_claim_name(tmp_path):
    spool = Spool(tmp_path / "spool")
    kept = spool.failed_dir / "factory-droid" / f"s-1.claim-{'0' * 32}.jsonl"
    kept.parent.mkdir(parents=True)
    kept.write_text(rec("s-1", "a").to_json() + "\n")
    assert spool.retry_failed() == (1, 1)
    assert [file.path for file in spool.pending()] == [spool.path_for("factory-droid", "s-1")]
