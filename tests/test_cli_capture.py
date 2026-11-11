"""mem event, mem ingest, mem spool status, and the server's startup drain."""

from __future__ import annotations

import io
import json
import threading
import time
from pathlib import Path

import pytest

from agent_memory.adapters.droid import DroidAdapter
from agent_memory.capture import ids
from agent_memory.capture.record import Record, now
from agent_memory.capture.spool import Spool
from agent_memory.capture.writer import Writer
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
    code, out, err = live_mem("event", "--harness", "vibe")
    assert (code, out) == (1, "")
    assert "no adapter" in err
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


# The transcript, read at the hooks where the harness has written what no hook carries


def turn(n: int, *, outcome: bool = True) -> list[dict]:
    """The entries Droid writes for a turn: the prompt, the reply, and then its outcome."""
    entries: list[dict] = [
        {
            "type": "message",
            "id": f"p{n}",
            "timestamp": f"2026-11-11T10:{n:02d}:01Z",
            "message": {"role": "user", "content": [{"type": "text", "text": f"Prompt {n}."}]},
        },
        {
            "type": "message",
            "id": f"r{n}",
            "timestamp": f"2026-11-11T10:{n:02d}:02Z",
            "message": {"role": "assistant", "content": [{"type": "text", "text": f"Reply {n}."}]},
        },
    ]
    if outcome:
        entries.append(
            {
                "type": "agent_turn_outcome",
                "turnId": f"p{n}",
                "reason": "completed",
                "resultKind": "text",
            }
        )
    return entries


def write_transcript(path: Path, *turns: list[dict]) -> Path:
    # Droid gives a session's first entry no timestamp.
    entries: list[dict] = [{"type": "session_start", "id": path.stem, "title": "Synthetic"}]
    for entries_of_turn in turns:
        entries.extend(entries_of_turn)
    path.write_text("".join(json.dumps(entry) + "\n" for entry in entries))
    return path


def droid_hook(path: Path, event: str, **fields) -> dict:
    return {
        "session_id": path.stem,
        "transcript_path": str(path),
        "cwd": "/synthetic/hooks",
        "hook_event_name": event,
        **fields,
    }


def kinds_of(client: MemoryClient, harness_session_id: str) -> dict[str, int]:
    sid = ids.session_id("factory-droid", harness_session_id)
    return dict(
        client.query(
            f"SELECT kind, count(*) FROM events WHERE session_id = {sql_literal(sid)} GROUP BY kind"
        )[1]
    )


def imports_of(client: MemoryClient, harness_session_id: str) -> list[tuple]:
    """Where each import of a session's transcript began and ended."""
    sid = ids.session_id("factory-droid", harness_session_id)
    return client.query(
        "SELECT cursor_start, cursor_end, range_start, range_end FROM capture_observations "
        f"WHERE session_id = {sql_literal(sid)} AND source_mode = 'transcript_import' "
        "ORDER BY range_end"
    )[1]


def test_a_stop_hook_imports_the_turn_and_the_next_reads_on_from_there(
    live_mem, live_client, tmp_path, monkeypatch
):
    path = write_transcript(tmp_path / f"{tmp_path.name}.jsonl", turn(1, outcome=False))
    feed(monkeypatch, droid_hook(path, "Stop", message_id="r1"))
    assert live_mem("event", "--harness", "droid") == (0, "", "")
    assert kinds_of(live_client, path.stem) == {
        "session_start": 1,
        "user_msg": 1,
        "assistant_msg": 1,
    }
    # Droid writes a turn's outcome after its Stop hook has run, so it comes with the next.
    write_transcript(path, turn(1), turn(2, outcome=False))
    feed(monkeypatch, droid_hook(path, "Stop", message_id="r2"))
    assert live_mem("event", "--harness", "droid") == (0, "", "")
    assert kinds_of(live_client, path.stem) == {
        "session_start": 1,
        "user_msg": 2,
        "assistant_msg": 2,
        "outcome": 1,
    }
    assert imports_of(live_client, path.stem) == [(None, "r1", 1, 3), ("r1", "r2", 4, 6)]
    # A Stop with nothing added since leaves no trace.
    feed(monkeypatch, droid_hook(path, "Stop", message_id="r2"))
    assert live_mem("event", "--harness", "droid") == (0, "", "")
    assert len(imports_of(live_client, path.stem)) == 2


def test_a_session_end_hook_brings_the_outcome_of_the_last_turn(
    live_mem, live_client, tmp_path, monkeypatch
):
    path = write_transcript(tmp_path / f"{tmp_path.name}.jsonl", turn(1, outcome=False))
    feed(monkeypatch, droid_hook(path, "Stop", message_id="r1"))
    assert live_mem("event", "--harness", "droid") == (0, "", "")
    write_transcript(path, turn(1))
    feed(monkeypatch, droid_hook(path, "SessionEnd", message_id="r1", reason="other"))
    assert live_mem("event", "--harness", "droid") == (0, "", "")
    assert kinds_of(live_client, path.stem) == {
        "session_start": 1,
        "user_msg": 1,
        "assistant_msg": 1,
        "outcome": 1,
        "session_end": 1,
    }
    # An outcome has no id of its own, so that import ended on the id of its turn, which is
    # the prompt's too. The next reads on from the outcome, not from the prompt.
    assert imports_of(live_client, path.stem) == [(None, "r1", 1, 3), ("r1", "p1", 4, 4)]
    feed(monkeypatch, droid_hook(path, "SessionEnd", message_id="r1", reason="other"))
    assert live_mem("event", "--harness", "droid") == (0, "", "")
    assert len(imports_of(live_client, path.stem)) == 2
    write_transcript(path, turn(1), turn(2, outcome=False))
    feed(monkeypatch, droid_hook(path, "Stop", message_id="r2"))
    assert live_mem("event", "--harness", "droid") == (0, "", "")
    assert imports_of(live_client, path.stem)[-1] == ("p1", "r2", 5, 6)


def test_a_session_first_seen_through_a_prompt_starts_where_its_transcript_does(
    live_mem, live_client, tmp_path, monkeypatch
):
    # Hooks installed while a session is running: the first record the ledger gets of it
    # is a prompt, and the session's row is written with the time of that hook.
    path = write_transcript(tmp_path / f"{tmp_path.name}.jsonl", turn(1))
    feed(monkeypatch, droid_hook(path, "UserPromptSubmit", prompt="Prompt 2.", message_id="p2"))
    assert live_mem("event", "--harness", "droid") == (0, "", "")
    write_transcript(path, turn(1), turn(2, outcome=False))
    feed(monkeypatch, droid_hook(path, "Stop", message_id="r2"))
    assert live_mem("event", "--harness", "droid") == (0, "", "")
    sid = ids.session_id("factory-droid", path.stem)
    session = f"FROM session_summary WHERE session_id = {sql_literal(sid)}"
    # A transcript's first entry is dated a millisecond before the first that has a time.
    began = "TIMESTAMPTZ '2026-11-11 10:01:00.999+00'"
    assert live_client.query(f"SELECT started_at = {began} {session}")[1] == [(True,)]
    assert live_client.query(f"SELECT recorded_started_at > {began} {session}")[1] == [(True,)]


def test_a_stop_hook_reads_the_whole_transcript_when_the_last_import_cannot_be_placed(
    live_mem, live_client, tmp_path, monkeypatch
):
    path = write_transcript(tmp_path / f"{tmp_path.name}.jsonl", turn(1, outcome=False))
    feed(monkeypatch, droid_hook(path, "Stop", message_id="r1"))
    assert live_mem("event", "--harness", "droid") == (0, "", "")
    # The file no longer holds, at the line that import ended on, the entry it ended on.
    write_transcript(path, turn(2, outcome=False), turn(3, outcome=False))
    feed(monkeypatch, droid_hook(path, "Stop", message_id="r3"))
    assert live_mem("event", "--harness", "droid") == (0, "", "")
    assert kinds_of(live_client, path.stem) == {
        "session_start": 1,
        "user_msg": 3,
        "assistant_msg": 3,
    }
    assert imports_of(live_client, path.stem)[-1] == (None, "r3", 1, 5)


def test_a_stop_hook_reads_on_from_the_import_that_waits_while_the_server_is_down(
    down_mem, down_cfg, tmp_path, monkeypatch
):
    path = write_transcript(tmp_path / "down-turns.jsonl", turn(1))
    spool = Spool(down_cfg.spool_dir)
    feed(monkeypatch, droid_hook(path, "Stop", message_id="r1"))
    code, out, err = down_mem("event", "--harness", "droid")
    assert (code, out) == (0, "")
    assert "spooled to" in err
    assert spool.status()[0].lines == 4
    # With nothing added since, a hook has nothing to keep and nothing to say.
    feed(monkeypatch, droid_hook(path, "Stop", message_id="r1"))
    assert down_mem("event", "--harness", "droid") == (0, "", "")
    assert spool.status()[0].lines == 4
    write_transcript(path, turn(1), turn(2, outcome=False))
    feed(monkeypatch, droid_hook(path, "Stop", message_id="r2"))
    assert down_mem("event", "--harness", "droid")[0] == 0
    # The ledger could not say where the last import ended. The one that waits did, so
    # only the two records of the new turn were read and kept.
    assert spool.status()[0].lines == 6
    server = MemoryServer(down_cfg, export_interval_hours=0)
    server.start()
    try:
        assert down_mem("ingest", "--spool")[0] == 0
        with MemoryClient.from_config(down_cfg) as client:
            assert kinds_of(client, path.stem) == {
                "session_start": 1,
                "user_msg": 2,
                "assistant_msg": 2,
                "outcome": 1,
            }
            assert imports_of(client, path.stem) == [(None, "p1", 1, 4), ("p1", "r2", 5, 6)]
    finally:
        server.stop()


def test_a_hook_whose_transcript_cannot_be_read_exits_1_and_keeps_its_own_event(
    live_mem, live_client, tmp_path, monkeypatch
):
    gone = tmp_path / f"{tmp_path.name}.jsonl"
    feed(monkeypatch, droid_hook(gone, "SessionEnd", message_id="r1", reason="other"))
    code, out, err = live_mem("event", "--harness", "droid")
    assert (code, out) == (1, "")
    assert "transcript" in err
    assert kinds_of(live_client, gone.stem) == {"session_end": 1}
    other = tmp_path / "not-a-transcript.json"
    other.write_text("{}")
    feed(monkeypatch, droid_hook(other, "Stop"))
    code, out, err = live_mem("event", "--harness", "droid")
    assert (code, out) == (1, "")
    assert "transcript" in err


def attributions_of(client: MemoryClient, harness_session_id: str) -> list[tuple[str, object]]:
    """Each event's kind and the `capture_attribution` stored with it, None where there is none."""
    sid = ids.session_id("factory-droid", harness_session_id)
    rows = client.query(
        f"SELECT kind, payload::JSON::VARCHAR FROM events WHERE session_id = {sql_literal(sid)} "
        "ORDER BY ts, kind"
    )[1]
    return [
        (kind, (json.loads(payload) if payload else {}).get("capture_attribution"))
        for kind, payload in rows
    ]


def test_a_transcript_event_keeps_which_message_and_turn_the_adapter_placed_it_in(
    live_mem, live_client, tmp_path
):
    def message(entry_id: str, second: int, role: str, *parts: dict) -> dict:
        return {
            "type": "message",
            "id": entry_id,
            "timestamp": f"2026-11-11T10:01:{second:02d}Z",
            "message": {"role": role, "content": list(parts)},
        }

    def said(text: str) -> dict:
        return {"type": "text", "text": text}

    finished = [
        message("p1", 1, "user", said("Prompt 1.")),
        message(
            "c1",
            2,
            "assistant",
            said("Looking."),
            {"type": "tool_use", "id": "call-1", "name": "Read", "input": {"file_path": "/x"}},
        ),
        message(
            "t1", 3, "user", {"type": "tool_result", "tool_use_id": "call-1", "content": "found"}
        ),
        message("r1", 4, "assistant", said("Reply 1.")),
        {"type": "agent_turn_outcome", "turnId": "p1", "reason": "completed", "resultKind": "text"},
    ]
    path = write_transcript(tmp_path / f"{tmp_path.name}.jsonl", finished, turn(2, outcome=False))
    code, _, err = live_mem("ingest", "--harness", "droid", "--transcript", str(path))
    assert (code, err) == (0, "")

    def in_turn_1(message_id: str, *, has_tool_calls: bool = False) -> dict:
        return {
            "message_id": message_id,
            "turn_id": "p1",
            "turn_complete": True,
            "turn_boundary": "agent_turn_outcome",
            "channel": None,
            "has_tool_calls": has_tool_calls,
        }

    def in_no_turn_yet(message_id: str) -> dict:
        return {
            "message_id": message_id,
            "turn_id": None,
            "turn_complete": False,
            "channel": None,
            "has_tool_calls": False,
        }

    assert attributions_of(live_client, path.stem) == [
        ("session_start", None),
        ("user_msg", in_turn_1("p1")),
        # Every event made from a message says what the message held, the call too.
        ("assistant_msg", in_turn_1("c1", has_tool_calls=True)),
        ("tool_call", in_turn_1("c1", has_tool_calls=True)),
        ("tool_result", in_turn_1("t1")),
        ("assistant_msg", in_turn_1("r1")),
        ("outcome", None),
        # Only an outcome names a turn, and the second turn has none yet.
        ("user_msg", in_no_turn_yet("p2")),
        ("assistant_msg", in_no_turn_yet("r2")),
    ]


def test_an_event_keeps_the_attribution_it_was_first_written_with(
    live_mem, live_client, tmp_path, monkeypatch
):
    # A turn as Droid's hooks deliver it: the prompt, the reply at Stop, the outcome later.
    path = write_transcript(tmp_path / f"{tmp_path.name}.jsonl", turn(1, outcome=False))
    feed(monkeypatch, droid_hook(path, "UserPromptSubmit", message_id="p1", prompt="Prompt 1."))
    assert live_mem("event", "--harness", "droid") == (0, "", "")
    feed(monkeypatch, droid_hook(path, "Stop", message_id="r1"))
    assert live_mem("event", "--harness", "droid") == (0, "", "")
    write_transcript(path, turn(1))
    feed(monkeypatch, droid_hook(path, "SessionEnd", message_id="r1", reason="other"))
    assert live_mem("event", "--harness", "droid") == (0, "", "")
    got = attributions_of(live_client, path.stem)
    # The outcome has arrived and named the turn. The reply's event was written before it
    # and no event is rewritten, so it still names no turn.
    assert ("outcome", None) in got
    assert [stored for kind, stored in got if kind == "assistant_msg"] == [
        {
            "message_id": "r1",
            "turn_id": None,
            "turn_complete": False,
            "channel": None,
            "has_tool_calls": False,
        }
    ]
    # The hook wrote the prompt before the transcript was read, so its event keeps only
    # what the hook payload carried.
    assert [stored for kind, stored in got if kind == "user_msg"] == [{"message_id": "p1"}]
    assert ("session_end", {"message_id": "r1"}) in got


def test_ingest_waits_for_a_write_that_is_under_way(live_mem, live_client, tmp_path, monkeypatch):
    # A hook gives the ledger two seconds. The import of a long session takes longer, and
    # giving up on it puts the whole transcript in the spool.
    monkeypatch.setenv("AGENT_MEMORY_DEADLINE", "0.01")
    write = Writer.write

    def slow_write(self, records):
        time.sleep(0.3)
        return write(self, records)

    monkeypatch.setattr(Writer, "write", slow_write)
    path = write_transcript(tmp_path / f"{tmp_path.name}.jsonl", turn(1))
    code, out, err = live_mem("ingest", "--harness", "droid", "--transcript", str(path))
    assert code == 0, err
    assert "4 event(s)" in out
    assert "spooled" not in out + err
    assert not Spool(live_mem.home / "spool").path_for("factory-droid", path.stem).exists()


def test_ingest_spools_a_transcript_once_while_the_server_is_down(down_mem, down_cfg, tmp_path):
    path = write_transcript(tmp_path / "down-import.jsonl", turn(1))
    for _ in range(2):
        code, out, _ = down_mem("ingest", "--harness", "droid", "--transcript", str(path))
        assert code == 0
        assert "spooled to" in out
    assert Spool(down_cfg.spool_dir).status()[0].lines == 4


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
    assert code == 2
    assert "no adapter" in err
    notes = tmp_path / "notes.txt"
    notes.write_text("")
    code, _, err = live_mem("ingest", "--harness", "droid", "--transcript", str(notes))
    assert code == 1
    assert ".jsonl" in err
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
