"""mem records: capture records taken from a harness extension while its input stays open."""

from __future__ import annotations

import itertools
import json
import os
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Generator
from contextlib import contextmanager
from typing import Any

import pytest

from agent_memory.capture import ids
from agent_memory.capture.spool import Spool
from agent_memory.config import Config
from agent_memory.db import MemoryClient, MemoryServer, sql_literal
from tests.test_cli import Mem

AGENT = "oh-my-pi"
ENVIRONMENT = ("AGENT_MEMORY_HOME", "AGENT_MEMORY_HOST", "AGENT_MEMORY_PORT")
_later = itertools.count()


@pytest.fixture
def live_mem(live_cfg, capsys, monkeypatch) -> Mem:
    for var in ENVIRONMENT:
        monkeypatch.delenv(var, raising=False)
    return Mem(home=live_cfg.home, port=live_cfg.port, capsys=capsys)


@pytest.fixture
def down_mem(down_cfg, capsys, monkeypatch) -> Mem:
    for var in ENVIRONMENT:
        monkeypatch.delenv(var, raising=False)
    return Mem(home=down_cfg.home, port=down_cfg.port, capsys=capsys)


def line(hsid: str, key: str, kind: str = "user_msg", **fields: Any) -> str:
    """A capture record as an extension writes it, which is not how `Record.to_json` does:
    no version, a time in milliseconds, and only the fields it has. Each is later than the
    one made before it."""
    record = {
        "agent_id": AGENT,
        "harness_session_id": hsid,
        "kind": kind,
        "ts": 1795000000000 + next(_later),
        "source_key": key,
        "content": f"{kind} {key}",
        **fields,
    }
    return json.dumps(record)


def feed(monkeypatch, *lines: str, ends: bool = True) -> None:
    """Make stdin a pipe that holds these lines and then ends. Without `ends`, the last
    line has no newline."""
    text = "\n".join(lines) + ("\n" if ends else "")
    reading, writing = os.pipe()

    def fill() -> None:
        with os.fdopen(writing, "wb") as end:
            end.write(text.encode())

    threading.Thread(target=fill, daemon=True).start()
    monkeypatch.setattr("sys.stdin", os.fdopen(reading))


def events(client: MemoryClient, hsid: str, agent: str = AGENT) -> list[tuple[Any, ...]]:
    """Kind, content, event id, and parent event id of a session's events, oldest first."""
    sid = sql_literal(ids.session_id(agent, hsid))
    return client.query(
        "SELECT kind, content, event_id::VARCHAR, parent_event_id::VARCHAR FROM events "
        f"WHERE session_id = {sid} ORDER BY ts, kind"
    )[1]


def until(check: Callable[[], Any], *, seconds: float = 20.0) -> Any:
    end = time.monotonic() + seconds
    while not (found := check()):
        assert time.monotonic() < end, "timed out"
        time.sleep(0.05)
    return found


@contextmanager
def running(cfg: Config, *args: str) -> Generator[subprocess.Popen[bytes]]:
    """`mem records` as an extension keeps it: a process of its own, written to by a pipe."""
    env = {name: value for name, value in os.environ.items() if name not in ENVIRONMENT}
    command = [sys.executable, "-m", "agent_memory.cli", "--home", str(cfg.home)]
    proc = subprocess.Popen(
        [*command, "--port", str(cfg.port), "records", *args],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )
    try:
        yield proc
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.communicate()


def write(proc: subprocess.Popen[bytes], *lines: str) -> None:
    assert proc.stdin is not None
    proc.stdin.write("".join(f"{each}\n" for each in lines).encode())
    proc.stdin.flush()


def test_records_get_the_ids_every_capture_path_gives_them(live_mem, live_client, monkeypatch):
    feed(
        monkeypatch,
        line("ids", "p1", session={"cwd": "/work/demo", "model": "local/one"}),
        line("ids", "call-1", "tool_call", payload={"tool_name": "read"}),
        line("ids", "call-1", "tool_result", parent_key="call-1"),
    )
    # The extension that runs this reads neither stream, and stdout is not its to fill.
    assert live_mem("records") == (0, "", "")
    call = ids.event_id(AGENT, "ids", "tool_call", "call-1")
    assert events(live_client, "ids") == [
        ("user_msg", "user_msg p1", ids.event_id(AGENT, "ids", "user_msg", "p1"), None),
        ("tool_call", "tool_call call-1", call, None),
        (
            "tool_result",
            "tool_result call-1",
            ids.event_id(AGENT, "ids", "tool_result", "call-1"),
            call,
        ),
    ]
    sid = sql_literal(ids.session_id(AGENT, "ids"))
    assert live_client.query(
        f"SELECT agent_id, cwd, project, model FROM sessions WHERE session_id = {sid}"
    )[1] == [(AGENT, "/work/demo", "demo", "local/one")]
    feed(monkeypatch, line("ids", "p1"))
    assert live_mem("records") == (0, "", "")
    assert len(events(live_client, "ids")) == 3


def test_records_reach_the_ledger_while_the_input_is_open(live_cfg, live_client):
    with running(live_cfg) as proc:
        write(proc, line("open", "p1"))
        until(lambda: events(live_client, "open"))
        write(proc, line("open", "p2"))
        until(lambda: len(events(live_client, "open")) == 2)
        assert proc.poll() is None
        out, err = proc.communicate(timeout=20)
        assert (proc.returncode, out, err) == (0, b"", b"")


def test_a_pass_is_one_observation_of_each_session_it_read(live_mem, live_client, monkeypatch):
    feed(
        monkeypatch,
        line("seen", "p1", session={"cwd": "/work/demo", "metadata": {"title": "first"}}),
        line("seen", "r1", "assistant_msg", session={"model": "local/two"}),
        line("seen", "o1", "outcome"),
        line("seen-too", "p1"),
    )
    assert live_mem("records") == (0, "", "")
    sid = sql_literal(ids.session_id(AGENT, "seen"))
    rows = live_client.query(
        "SELECT source_mode, cursor_start, cursor_end, exclusions::JSON::VARCHAR "
        f"FROM capture_observations WHERE session_id = {sid}"
    )[1]
    assert rows == [("record", "p1", "o1", '["no_complete_capture_evidence"]')]
    assert live_client.query(
        "SELECT capture_completeness, model, title, coverage_gap FROM session_summary "
        f"WHERE session_id = {sid}"
    )[1] == [("partial", "local/two", "first", False)]
    other = sql_literal(ids.session_id(AGENT, "seen-too"))
    assert live_client.query(
        f"SELECT count(*) FROM capture_observations WHERE session_id = {other}"
    )[1] == [(1,)]


def test_more_lines_than_a_pass_takes_are_all_delivered(live_mem, live_client, monkeypatch):
    import agent_memory.capture.stream as stream

    monkeypatch.setattr(stream, "PASS_LINES", 2)
    feed(monkeypatch, *(line("many", f"p{n}") for n in range(5)))
    assert live_mem("records") == (0, "", "")
    assert [row[1] for row in events(live_client, "many")] == [f"user_msg p{n}" for n in range(5)]
    sid = sql_literal(ids.session_id(AGENT, "many"))
    assert live_client.query(f"SELECT count(*) FROM capture_observations WHERE session_id = {sid}")[
        1
    ] == [(3,)]


def test_text_is_redacted_before_it_is_written_anywhere(
    live_mem, live_client, down_mem, down_cfg, monkeypatch
):
    token = down_cfg.read_token()
    feed(monkeypatch, line("secret", "p1", content=f"the token is {token}"))
    assert down_mem("records")[0] == 0
    spooled = Spool(down_cfg.spool_dir).path_for(AGENT, "secret").read_text()
    assert token not in spooled
    assert "[REDACTED:mem_token]" in spooled
    feed(monkeypatch, line("secret", "p1", content=f"the token is {live_mem_token(live_mem)}"))
    assert live_mem("records") == (0, "", "")
    assert events(live_client, "secret")[0][1] == "the token is [REDACTED:mem_token]"


def live_mem_token(mem: Mem) -> str:
    return (mem.home / "token").read_text().strip()


def test_records_wait_in_the_spool_when_down_and_replay_adds_each_once(
    down_mem, down_cfg, monkeypatch
):
    lines = [line("down", "p1"), line("down", "r1", "assistant_msg")]
    feed(monkeypatch, *lines)
    code, out, err = down_mem("records")
    assert (code, out) == (0, "")
    assert "ledger not reached" in err
    assert Spool(down_cfg.spool_dir).status()[0].lines == 2
    server = MemoryServer(down_cfg, export_interval_hours=0)
    server.start()
    try:
        assert down_mem("ingest", "--spool")[0] == 0
        # What the extension hands over again, and a second replay, find the events there.
        feed(monkeypatch, *lines)
        assert down_mem("records") == (0, "", "")
        assert down_mem("ingest", "--spool")[0] == 0
        with MemoryClient.from_config(down_cfg) as client:
            assert [row[0] for row in events(client, "down")] == ["user_msg", "assistant_msg"]
    finally:
        server.stop()
    assert Spool(down_cfg.spool_dir).pending() == []


def test_a_stream_that_spooled_replays_before_it_writes_on(down_cfg):
    # A result written ahead of its call would never be linked to it: an event is not
    # rewritten when its parent arrives.
    spool = Spool(down_cfg.spool_dir)
    with running(down_cfg) as proc:
        write(proc, line("behind", "call-1", "tool_call"))
        until(lambda: spool.path_for(AGENT, "behind").exists())
        server = MemoryServer(down_cfg, export_interval_hours=0)
        server.start()
        try:
            write(proc, line("behind", "call-1", "tool_result", parent_key="call-1"))
            with MemoryClient.from_config(down_cfg) as client:
                got = until(lambda: len(found := events(client, "behind")) == 2 and found)
                assert [(row[0], row[3]) for row in got] == [
                    ("tool_call", None),
                    ("tool_result", ids.event_id(AGENT, "behind", "tool_call", "call-1")),
                ]
                until(lambda: spool.pending() == [])
                # Caught up, the stream writes to the ledger without the spool again.
                write(proc, line("behind", "p2"))
                until(lambda: len(events(client, "behind")) == 3)
                assert spool.pending() == []
            out, err = proc.communicate(timeout=20)
        finally:
            server.stop()
    assert (proc.returncode, out) == (0, b"")
    assert b"ledger not reached" in err
    assert b"the spool was replayed" in err


def test_a_line_that_is_not_a_record_is_kept_and_the_lines_after_it_are_taken(
    live_mem, live_cfg, live_client, monkeypatch
):
    token = live_cfg.read_token()
    feed(
        monkeypatch,
        f"cut off before any record {token}",
        line("lines", "p1"),
        line("lines", "p2", "no-such-kind", content=f"said {token}"),
        f'{{"agent_id": "{AGENT}", "harness_session_id": "lines", "cut": "off {token}',
        line("lines", "p3"),
    )
    code, out, err = live_mem("records")
    assert (code, out) == (0, "")
    assert token not in err
    for number in (1, 3, 4):
        assert f"line {number} is not a capture record" in err
    got = events(live_client, "lines")
    assert sorted(row[0] for row in got) == [
        "capture_error",
        "capture_error",
        "user_msg",
        "user_msg",
    ]
    assert {row[1].split(":")[0] for row in got if row[0] == "capture_error"} == {
        "unknown event kind 'no-such-kind'",
        "not JSON",
    }
    failed = Spool(live_cfg.spool_dir).failed_dir
    # Text that is not JSON cannot be redacted field by field, so only its hash is kept.
    kept = (failed / AGENT / "lines.jsonl").read_text().splitlines()
    assert [sorted(json.loads(each)) for each in kept] == [
        ["agent_id", "content", "harness_session_id", "kind", "source_key", "ts"],
        ["unparseable_line_sha256"],
    ]
    assert json.loads(kept[0])["content"] == "said [REDACTED:mem_token]"
    # No session was named or read before the first line, so no session can show it.
    nowhere = (failed / "unknown" / "records.jsonl").read_text()
    assert "unparseable_line_sha256" in nowhere
    assert token not in nowhere


def test_a_last_line_without_its_end_is_taken(live_mem, live_client, monkeypatch):
    # A harness that is killed can leave its last line unfinished or merely unended.
    feed(monkeypatch, line("ending", "p1"), line("ending", "p2"), ends=False)
    assert live_mem("records") == (0, "", "")
    assert [row[1] for row in events(live_client, "ending")] == ["user_msg p1", "user_msg p2"]


def test_a_record_of_an_unknown_agent_is_kept_and_the_rest_is_written(
    live_mem, live_cfg, live_client, monkeypatch
):
    stranger = json.loads(line("stranger", "p1"))
    feed(monkeypatch, json.dumps({**stranger, "agent_id": "nobody"}), line("known", "p1"))
    code, out, err = live_mem("records")
    assert (code, out) == (0, "")
    assert "1 record(s) rejected" in err
    assert "unknown agent 'nobody'" in err
    kept = Spool(live_cfg.spool_dir).failed_dir / "nobody" / "stranger.jsonl"
    assert str(kept) in err
    assert json.loads(kept.read_text())["source_key"] == "p1"
    assert [row[0] for row in events(live_client, "known")] == ["user_msg"]


def test_an_agent_named_with_dots_waits_inside_the_spool(down_mem, down_cfg, monkeypatch):
    # The agent's id comes from the line, and the spool names a directory after it.
    dotted = json.loads(line("dotted", "p1"))
    feed(monkeypatch, json.dumps({**dotted, "agent_id": ".."}))
    assert down_mem("records")[0] == 0
    spool = Spool(down_cfg.spool_dir)
    (waiting,) = spool.pending()
    assert waiting.path == spool.root / "_" / "dotted.jsonl"
    assert not (spool.root.parent / "dotted.jsonl").exists()


def test_at_the_end_of_input_what_the_ledger_did_not_take_in_time_is_spooled(
    down_mem, down_cfg, monkeypatch
):
    import agent_memory.capture.pipeline as pipeline

    class Hangs:
        def __init__(self) -> None:
            time.sleep(3)
            raise RuntimeError("unreachable")

    monkeypatch.setattr(pipeline.MemoryClient, "from_config", staticmethod(lambda cfg: Hangs()))
    monkeypatch.setenv("AGENT_MEMORY_DEADLINE", "0.3")
    feed(monkeypatch, line("slow", "p1"))
    started = time.monotonic()
    code, out, err = down_mem("records")
    assert time.monotonic() - started < 1.5
    assert (code, out) == (0, "")
    assert "timed out waiting for the server" in err
    assert Spool(down_cfg.spool_dir).status()[0].lines == 1


def test_records_exit_1_when_a_line_could_not_be_kept_anywhere(down_mem, monkeypatch):
    def full_disk(spool, records):
        raise OSError("synthetic full disk")

    monkeypatch.setattr(Spool, "append", full_disk)
    feed(monkeypatch, line("lost", "p1"))
    code, out, err = down_mem("records")
    assert (code, out) == (1, "")
    assert "1 line(s) could not be kept" in err


def test_records_spool_only_leaves_the_ledger_alone(live_mem, live_cfg, live_client, monkeypatch):
    feed(monkeypatch, line("spool-only", "p1"))
    assert live_mem("records", "--spool-only") == (0, "", "")
    assert Spool(live_cfg.spool_dir).path_for(AGENT, "spool-only").exists()
    assert events(live_client, "spool-only") == []


def test_a_signal_for_the_harness_does_not_end_the_stream_and_sigterm_spools(live_cfg, live_client):
    spool = Spool(live_cfg.spool_dir)
    with running(live_cfg) as proc:
        write(proc, line("signals", "p1"))
        until(lambda: events(live_client, "signals"))
        # A terminal sends an interrupt to the harness and to every process it started.
        proc.send_signal(signal.SIGINT)
        proc.send_signal(signal.SIGHUP)
        write(proc, line("signals", "p2"))
        until(lambda: len(events(live_client, "signals")) == 2)
        assert proc.poll() is None
        write(proc, line("signals", "p3"))
        time.sleep(0.2)
        proc.send_signal(signal.SIGTERM)
        # The harness has not closed the pipe, and may never.
        assert proc.wait(timeout=20) == 0
        assert proc.communicate() == (b"", b"")
    assert spool.path_for(AGENT, "signals").exists() or len(events(live_client, "signals")) == 3
