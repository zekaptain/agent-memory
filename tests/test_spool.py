"""Spool and pipeline: outage handling, replay, bad lines, redaction on replay."""

from __future__ import annotations

import os
import time
from datetime import UTC, datetime
from typing import Any

from agent_memory.capture import ids
from agent_memory.capture.pipeline import deliver, replay_spool
from agent_memory.capture.record import Record
from agent_memory.capture.spool import Spool
from agent_memory.db import MemoryClient, MemoryServer, sql_literal

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
    assert delivery.ok
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
