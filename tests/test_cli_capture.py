"""mem event, mem ingest, mem spool status, and the server's startup drain."""

from __future__ import annotations

import io
import json

import pytest

from agent_memory.capture import ids
from agent_memory.capture.spool import Spool
from agent_memory.db import MemoryClient, MemoryServer, sql_literal
from tests.conftest import FIXTURES
from tests.test_cli import Mem


@pytest.fixture
def live_mem(live_cfg, capsys, monkeypatch) -> Mem:
    for var in ("AGENT_MEMORY_HOME", "AGENT_MEMORY_HOST", "AGENT_MEMORY_PORT"):
        monkeypatch.delenv(var, raising=False)
    return Mem(home=live_cfg.home, port=live_cfg.port, capsys=capsys)


@pytest.fixture
def down_mem(down_cfg, capsys, monkeypatch) -> Mem:
    for var in ("AGENT_MEMORY_HOME", "AGENT_MEMORY_HOST", "AGENT_MEMORY_PORT"):
        monkeypatch.delenv(var, raising=False)
    return Mem(home=down_cfg.home, port=down_cfg.port, capsys=capsys)


def feed(monkeypatch, payload) -> None:
    monkeypatch.setattr(
        "sys.stdin", io.StringIO(json.dumps(payload) if not isinstance(payload, str) else payload)
    )


HOOK = {
    "session_id": "cli-1",
    "cwd": "/Users/tester/code/demo",
    "hook_event_name": "UserPromptSubmit",
    "prompt": "hello",
    "message_id": "m1",
}


def test_event_writes_through_a_live_server(live_mem, live_client, monkeypatch):
    feed(monkeypatch, HOOK)
    code, out, err = live_mem("event", "--harness", "droid")
    assert code == 0, err
    assert "1 session(s), 1 event(s)" in out
    feed(monkeypatch, HOOK)
    code, out, _ = live_mem("event", "--harness", "droid")
    assert code == 0
    assert "0 event(s)" in out
    assert "1 already present" in out
    sid = ids.session_id("factory-droid", "cli-1")
    got = live_client.query(f"SELECT content FROM events WHERE session_id = {sql_literal(sid)}")[1]
    assert got == [("hello",)]


def test_event_spools_when_down_and_still_exits_zero(down_mem, down_cfg, monkeypatch):
    feed(monkeypatch, HOOK)
    code, out, _ = down_mem("event", "--harness", "droid")
    assert code == 0
    assert "spooled to" in out
    assert Spool(down_cfg.spool_dir).status()[0].lines == 1
    code, out, _ = down_mem("spool", "status")
    assert code == 0
    assert "factory-droid" in out


def test_event_rejects_bad_input(live_mem, monkeypatch):
    feed(monkeypatch, "not json")
    code, _, err = live_mem("event", "--harness", "droid")
    assert code == 2
    assert "not JSON" in err
    feed(monkeypatch, {"hook_event_name": "Stop"})
    code, _, err = live_mem("event", "--harness", "droid")
    assert code == 2
    assert "session_id" in err
    feed(monkeypatch, HOOK)
    code, _, err = live_mem("event", "--harness", "letta")
    assert code == 2
    assert "unknown harness" in err


def test_event_with_nothing_to_record(live_mem, monkeypatch):
    feed(monkeypatch, {**HOOK, "hook_event_name": "Stop"})
    code, out, _ = live_mem("event", "--harness", "droid")
    assert code == 0
    assert "nothing to record" in out


def test_ingest_transcript_then_since_key(live_mem, live_client, tmp_path):
    transcript = tmp_path / "cli-transcript.jsonl"
    transcript.write_text((FIXTURES / "droid_session.jsonl").read_text())
    code, out, err = live_mem("ingest", "--harness", "droid", "--transcript", str(transcript))
    assert code == 0, err
    assert "1 session(s), 10 event(s)" in out
    code, out, _ = live_mem(
        "ingest",
        "--harness",
        "droid",
        "--transcript",
        str(transcript),
        "--since-key",
        "msg-a2",
        "--include-thinking",
    )
    assert code == 0
    assert "0 event(s)" in out
    sid = ids.session_id("factory-droid", "cli-transcript")
    kinds = live_client.query(
        f"SELECT kind, count(*) FROM events WHERE session_id = {sql_literal(sid)} "
        "GROUP BY kind ORDER BY kind"
    )[1]
    assert dict(kinds) == {
        "assistant_msg": 2,
        "capture_error": 1,
        "compaction": 1,
        "outcome": 1,
        "session_start": 1,
        "tool_call": 1,
        "tool_result": 1,
        "user_msg": 2,
    }
    redacted = live_client.query(
        f"SELECT content FROM events WHERE session_id = {sql_literal(sid)} "
        "AND kind = 'user_msg' ORDER BY seq"
    )[1][0][0]
    assert "[REDACTED:github_token]" in redacted


def test_ingest_argument_validation(live_mem, tmp_path):
    code, _, err = live_mem("ingest")
    assert code == 2
    assert "exactly one" in err
    transcript = tmp_path / "t.jsonl"
    transcript.write_text("")
    code, _, err = live_mem("ingest", "--transcript", str(transcript))
    assert code == 2
    assert "--harness" in err
    code, _, err = live_mem("ingest", "--harness", "vibe", "--transcript", str(transcript))
    assert code == 1
    assert "Phase 2" in err
    code, _, err = live_mem(
        "ingest", "--harness", "droid", "--transcript", str(tmp_path / "missing.jsonl")
    )
    assert code == 2


def test_ingest_spool_replays_and_purges(down_mem, down_cfg, monkeypatch):
    feed(monkeypatch, HOOK)
    assert down_mem("event", "--harness", "droid")[0] == 0
    server = MemoryServer(down_cfg, export_interval_hours=0)
    server.start()
    try:
        code, out, _ = down_mem("ingest", "--spool", "--purge-done", "0")
        assert code == 0
        assert "1 file(s), 2 rows inserted" in out
        assert "purged 1 replayed file(s)" in out
        with MemoryClient.from_config(down_cfg) as client:
            assert client.query("SELECT count(*) FROM events")[1] == [(1,)]
    finally:
        server.stop()
    assert not list(Spool(down_cfg.spool_dir).done_dir.rglob("*.jsonl"))


def test_serve_drains_the_spool_at_startup(down_mem, down_cfg, monkeypatch, capsys):
    feed(monkeypatch, HOOK)
    assert down_mem("event", "--harness", "droid")[0] == 0
    import agent_memory.cli as cli

    real_serve_forever = MemoryServer.serve_forever

    def serve_briefly(self, stop, **kwargs):
        stop.set()
        return real_serve_forever(self, stop, **kwargs)

    monkeypatch.setattr(MemoryServer, "serve_forever", serve_briefly)
    monkeypatch.setattr(cli.signal, "signal", lambda *a: None)
    code, out, err = down_mem("serve")
    assert code == 0, err
    assert "spool replayed: 1 file(s), 2 rows inserted" in out
    assert Spool(down_cfg.spool_dir).pending() == []
