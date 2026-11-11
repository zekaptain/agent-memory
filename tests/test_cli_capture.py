"""mem event, mem ingest, mem spool status, and the server's startup drain."""

from __future__ import annotations

import io
import json
import threading

import pytest

from agent_memory.adapters.droid import DroidAdapter
from agent_memory.capture import ids
from agent_memory.capture.record import Record, now
from agent_memory.capture.spool import Spool
from agent_memory.cli import main
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


def kept_raw(client: MemoryClient, harness_session_id: str) -> list[object]:
    """The hook payload stored with each event of a Droid session, None where there is none."""
    sid = ids.session_id("factory-droid", harness_session_id)
    rows = client.query(
        f"SELECT payload::JSON::VARCHAR FROM events WHERE session_id = {sql_literal(sid)} "
        "ORDER BY seq"
    )[1]
    return [(json.loads(row[0]) if row[0] else {}).get("raw") for row in rows]


def serve_then_stop(mem: Mem, monkeypatch) -> tuple[int, str, str]:
    """Run `mem serve` through its startup and stop it before it waits for clients."""
    import agent_memory.cli as cli

    real_serve_forever = MemoryServer.serve_forever

    def serve_briefly(self, stop, **kwargs):
        stop.set()
        return real_serve_forever(self, stop, **kwargs)

    monkeypatch.setattr(MemoryServer, "serve_forever", serve_briefly)
    monkeypatch.setattr(cli.signal, "signal", lambda *a: None)
    return mem("serve")


def test_event_writes_through_a_live_server_and_says_nothing(live_mem, live_client, monkeypatch):
    for _ in range(2):
        feed(monkeypatch, HOOK)
        # A harness adds a hook's stdout to the agent's context.
        assert live_mem("event", "--harness", "droid") == (0, "", "")
    sid = ids.session_id("factory-droid", "cli-1")
    got = live_client.query(f"SELECT content FROM events WHERE session_id = {sql_literal(sid)}")[1]
    assert got == [("hello",)]


def test_event_spools_when_down_and_still_exits_zero(down_mem, down_cfg, monkeypatch):
    feed(monkeypatch, HOOK)
    code, out, err = down_mem("event", "--harness", "droid")
    assert (code, out) == (0, "")
    assert "spooled to" in err
    assert Spool(down_cfg.spool_dir).status()[0].lines == 1
    code, out, _ = down_mem("spool", "status")
    assert code == 0
    assert "factory-droid" in out


def test_event_never_exits_2_on_bad_input(live_mem, monkeypatch):
    # A harness reads exit 2 from a hook as "block the agent".
    feed(monkeypatch, "not json")
    code, out, err = live_mem("event", "--harness", "droid")
    assert (code, out) == (1, "")
    assert "not JSON" in err
    feed(monkeypatch, ["not", "an", "object"])
    code, out, err = live_mem("event", "--harness", "droid")
    assert (code, out) == (1, "")
    assert "JSON object" in err
    feed(monkeypatch, {"hook_event_name": "Stop"})
    code, out, err = live_mem("event", "--harness", "droid")
    assert (code, out) == (1, "")
    assert "session_id" in err
    feed(monkeypatch, HOOK)
    code, out, err = live_mem("event", "--harness", "letta")
    assert (code, out) == (1, "")
    assert "unknown harness" in err
    feed(monkeypatch, HOOK)
    code, out, err = live_mem("event")
    assert (code, out) == (1, "")
    assert "--harness" in err
    feed(monkeypatch, HOOK)
    assert live_mem("event", "--harness", "droid", "--no-such-option")[0] == 1
    feed(monkeypatch, HOOK)
    assert main(["--port", "not-a-port", "event", "--harness", "droid"]) == 1


def test_event_exits_1_when_nothing_could_be_kept(down_mem, monkeypatch):
    def full_disk(spool, records):
        raise OSError("synthetic full disk")

    monkeypatch.setattr(Spool, "append", full_disk)
    feed(monkeypatch, HOOK)
    code, out, err = down_mem("event", "--harness", "droid")
    assert (code, out) == (1, "")
    assert "nothing could be kept" in err


def test_event_with_nothing_to_record(live_mem, monkeypatch):
    feed(monkeypatch, {**HOOK, "hook_event_name": "Stop"})
    assert live_mem("event", "--harness", "droid") == (0, "", "")


def test_event_records_compaction_instructions_that_are_not_text(
    live_mem, live_cfg, live_client, monkeypatch
):
    hook = {
        "session_id": "cli-compact",
        "hook_event_name": "PreCompact",
        "message_id": "c1",
        "custom_instructions": {"keep": "decisions"},
    }
    feed(monkeypatch, hook)
    assert live_mem("event", "--harness", "droid") == (0, "", "")
    sid = ids.session_id("factory-droid", "cli-compact")
    assert live_client.query(
        f"SELECT kind, content FROM events WHERE session_id = {sql_literal(sid)}"
    )[1] == [("note", '{"keep": "decisions"}')]
    assert not (
        Spool(live_cfg.spool_dir).failed_dir / "factory-droid" / "cli-compact.jsonl"
    ).exists()


def test_event_keeps_the_hook_payload_only_when_asked(live_mem, live_cfg, live_client, monkeypatch):
    token = live_cfg.read_token()
    for hsid, flags in (("cli-raw", ["--keep-raw"]), ("cli-no-raw", [])):
        feed(monkeypatch, {**HOOK, "session_id": hsid, "prompt": f"hello {token}"})
        assert live_mem("event", "--harness", "droid", *flags) == (0, "", "")
    assert kept_raw(live_client, "cli-raw") == [
        {**HOOK, "session_id": "cli-raw", "prompt": "hello [REDACTED:mem_token]"}
    ]
    assert kept_raw(live_client, "cli-no-raw") == [None]


@pytest.mark.parametrize("replayed_by", ["serve", "ingest"])
def test_a_kept_hook_payload_survives_the_spool(down_mem, down_cfg, monkeypatch, replayed_by):
    # Neither replay is told what the hook was asked to keep: the spooled record says so.
    token = down_cfg.read_token()
    feed(monkeypatch, {**HOOK, "session_id": "cli-raw", "prompt": f"hello {token}"})
    assert down_mem("event", "--harness", "droid", "--keep-raw")[0] == 0
    feed(monkeypatch, {**HOOK, "session_id": "cli-no-raw"})
    assert down_mem("event", "--harness", "droid")[0] == 0
    redacted = {**HOOK, "session_id": "cli-raw", "prompt": "hello [REDACTED:mem_token]"}
    spooled = [json.loads(f.path.read_text()) for f in Spool(down_cfg.spool_dir).pending()]
    assert {line["harness_session_id"]: line["raw"] for line in spooled} == {
        "cli-raw": redacted,
        "cli-no-raw": None,
    }
    if replayed_by == "serve":
        code, _, err = serve_then_stop(down_mem, monkeypatch)
        assert code == 0, err
    server = MemoryServer(down_cfg, export_interval_hours=0)
    server.start()
    try:
        if replayed_by == "ingest":
            assert down_mem("ingest", "--spool")[0] == 0
        with MemoryClient.from_config(down_cfg) as client:
            assert kept_raw(client, "cli-raw") == [redacted]
            assert kept_raw(client, "cli-no-raw") == [None]
    finally:
        server.stop()
    assert Spool(down_cfg.spool_dir).pending() == []


def test_event_keeps_a_record_the_ledger_rejects(live_mem, live_cfg, live_client, monkeypatch):
    # No hook payload is meant to produce a record the ledger refuses, so an adapter that
    # returns one stands in for a fault the adapters are not known to have.
    refused = Record(
        agent_id="factory-droid",
        harness_session_id="cli-rejected",
        kind="memory_write",
        ts=now(),
        source_key="bad",
        memory={"content": ""},
    )
    monkeypatch.setattr(DroidAdapter, "from_hook", lambda self, payload: [refused])
    feed(monkeypatch, {**HOOK, "session_id": "cli-rejected"})
    code, out, err = live_mem("event", "--harness", "droid")
    assert (code, out) == (0, "")
    assert "1 record(s) rejected" in err
    spool = Spool(live_cfg.spool_dir)
    kept = spool.failed_dir / "factory-droid" / "cli-rejected.jsonl"
    assert str(kept) in err
    assert kept.read_text().count("\n") == 1
    sid = ids.session_id("factory-droid", "cli-rejected")
    assert live_client.query(f"SELECT kind FROM events WHERE session_id = {sql_literal(sid)}")[
        1
    ] == [("capture_error",)]
    code, out, _ = live_mem("spool", "status")
    assert code == 0
    assert "rejected" in out
    code, out, _ = live_mem("spool", "retry")
    assert code == 0
    assert "returned 1 rejected record(s)" in out
    assert not kept.exists()
    assert [(s.agent_id, s.lines) for s in spool.status()] == [("factory-droid", 1)]
    assert live_mem("spool", "retry") == (0, "no rejected records\n", "")


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
    # A transcript has no hook payload to keep, and a spooled record says what it kept.
    for args in (("--spool",), ("--harness", "droid", "--transcript", str(transcript))):
        code, _, err = live_mem("ingest", *args, "--keep-raw")
        assert code == 2
        assert "No such option" in err


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
    code, out, err = serve_then_stop(down_mem, monkeypatch)
    assert code == 0, err
    assert "spool replayed: 1 file(s), 2 rows inserted" in out
    assert Spool(down_cfg.spool_dir).pending() == []


def test_serve_says_which_step_failed(down_mem, monkeypatch):
    import agent_memory.cli as cli
    import agent_memory.lifecycle as lifecycle

    real_serve_forever = MemoryServer.serve_forever
    real_maintain = lifecycle.maintain_corrections
    stops: list[threading.Event] = []

    def serve(self, stop, **kwargs):
        stops.append(stop)
        return real_serve_forever(self, stop, poll_seconds=0.005, **kwargs)

    def maintain(connection, **kwargs) -> int:
        if not stops:
            return real_maintain(connection, **kwargs)  # the pass made before listening
        stops[0].set()
        raise RuntimeError("ledger busy")

    monkeypatch.setattr(MemoryServer, "serve_forever", serve)
    monkeypatch.setattr(lifecycle, "maintain_corrections", maintain)
    monkeypatch.setattr(cli.signal, "signal", lambda *a: None)
    code, _, err = down_mem("serve")
    assert code == 0, err
    assert "correction maintenance failed: ledger busy" in err
    assert "export failed" not in err
