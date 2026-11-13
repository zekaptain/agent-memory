"""Judge pipeline against a running server: import, candidates, decide, decision events, CLI."""

from __future__ import annotations

import itertools
import json
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from functools import partial

import pytest
from rich.table import Table

from agent_memory.adapters.droid import DroidAdapter
from agent_memory.capture import ids
from agent_memory.capture.pipeline import deliver
from agent_memory.capture.record import Record
from agent_memory.cli import main
from agent_memory.db import sql_literal
from agent_memory.judge import cli as judge_cli
from agent_memory.judge import pipeline
from agent_memory.judge.drafter import Draft
from agent_memory.judge.lesson import Claim, Lesson
from agent_memory.judge.pipeline import (
    build_candidates,
    decide_items,
    import_sessions,
    purge_run,
)
from agent_memory.judge.provenance import HUMAN, session_callers
from agent_memory.judge.store import JudgeError, RunStore, imports_log, sole_writer
from agent_memory.judge.triggers import fetch_turn_events, segment_turns
from tests.conftest import FIXTURES
from tests.test_adapters import said, seen_by, write


@pytest.fixture(scope="module")
def sessions_dir(tmp_path_factory):
    root = tmp_path_factory.mktemp("sessions")
    project = root / "-Users-tester-code-demo"
    project.mkdir()
    shutil.copy(FIXTURES / "droid_session_judge.jsonl", project / "judge-session-1.jsonl")
    (project / "judge-session-1.settings.json").write_text("{}")
    return root


def test_import_candidates_decide_and_ledger_events(live_cfg, live_client, sessions_dir, capsys):
    first = import_sessions(live_cfg, sessions_dir)
    assert (first.files, first.skipped_files) == (1, 0)
    assert first.inserted > 0
    assert first.errors == []
    second = import_sessions(live_cfg, sessions_dir)
    assert (second.files, second.skipped_files) == (0, 1)  # unchanged file, skipped by mtime

    summary = build_candidates(live_cfg, "t1", n=4, seed=1, clean_share=0.25)
    assert summary.pool == {"T2": 1, "T1": 1, "T3": 1, "T4": 0, "T5": 0, "T0": 1}
    assert summary.chosen == summary.pool
    assert summary.pool_callers == {"human": 4, "agent": 0}
    assert summary.harness_turns == 0
    store = RunStore(live_cfg, "t1")
    assert store.read_run()["questions_version"] == "questions_v4"
    assert store.read_run()["labels_version"] == "labels_v4"
    items = store.read_jsonl(store.items_path)
    assert len(items) == 4
    assert {i["split"] for i in items} <= {"calibration", "held_out"}
    t3 = next(i for i in items if i["stratum"] == "T3")
    projection = store.read_projection(t3["item_id"])
    assert "TOOL RESULT (error): mem: memory.duckdb is locked" in projection
    assert "NEXT MESSAGE (human): No, that's wrong" in projection
    assert t3["flags"] == ["T3", "T4"]
    assert t3["facts"]["tools"] == ["Execute"]
    assert t3["facts"]["repeated_tool_error"] is True
    assert t3["facts"]["tool_error_count"] == 2
    assert t3["projection_hash"]
    assert t3["sidecar"]
    with pytest.raises(JudgeError, match="already has items"):
        build_candidates(live_cfg, "t1", n=4)

    assert decide_items(live_cfg, "t1", ["rules"]).written == {"rules": 4}
    verdicts = store.read_jsonl(store.verdicts_path("rules"))
    assert {v["item_id"] for v in verdicts} == {i["item_id"] for i in items}
    by_item = {v["item_id"]: v for v in verdicts}
    assert by_item[t3["item_id"]]["answers"]["failed"]["value"] == "yes"
    assert by_item[t3["item_id"]]["answers"]["category"]["value"] == "tool_misuse"
    assert "grounded" not in by_item[t3["item_id"]]["answers"]  # no drafts yet
    assert decide_items(live_cfg, "t1", ["rules"]).written == {"rules": 0}  # resumable

    _, rows = live_client.query(
        "SELECT e.content, CAST(e.payload AS JSON)::VARCHAR, s.agent_id, s.project, s.scope "
        "FROM events e JOIN sessions s USING (session_id) WHERE e.kind = 'decision'"
    )
    assert len(rows) == 4
    for content, payload, agent, project, scope in rows:
        assert (agent, project, scope) == ("judge", "judge-experiment", "judge")
        judge = json.loads(payload)["judge"]
        assert judge["run_id"] == "t1"
        assert judge["backend"] == "rules"
        assert judge["questions_version"] == "questions_v4"
        assert judge["projection_hash"]
        assert judge["action"] in {"write_lesson", "review", "discard"}
        assert set(judge["answers"]["failed"]) == {"value", "probabilities", "confidence", "method"}
        for text in (content, payload):
            assert "memory.duckdb is locked" not in text
            assert "No, that's wrong" not in text
    assert decide_items(live_cfg, "t1", ["rules"], record_events=False).written == {"rules": 0}

    # A draft arriving later supersedes the turn-only verdict for that item and adds a
    # second decision event; the other items stay as they were.
    error_line = next(line for line in projection.splitlines() if "TOOL RESULT (error):" in line)
    error_number = int(error_line.split("]", 1)[0].lstrip("["))
    lesson = Lesson(
        rule="Stop mem serve before using Execute to run mem migrate.",
        kind="tool_quirk",
        claims=[
            Claim(text="The mem command reported memory.duckdb is locked.", cited=(error_number,))
        ],
    )
    draft = Draft.from_lesson(
        t3["item_id"], lesson, drafter="ollama:m", model_snapshot="m", latency_ms=1.0
    )
    store.write_jsonl(
        store.drafts_path,
        [draft.to_dict()],
    )
    assert decide_items(live_cfg, "t1", ["rules"]).written == {"rules": 1}
    verdicts = store.read_jsonl(store.verdicts_path("rules"))
    assert len(verdicts) == 4
    redone = next(v for v in verdicts if v["item_id"] == t3["item_id"])
    assert redone["has_draft"] is True
    assert redone["answers"]["grounded"]["value"] == "yes"
    assert redone["claims"] == 1
    assert redone["answers"]["grounded.1"]["value"] == "yes"
    assert redone["answers"]["grounded"]["method"] == "min_over_claims"
    _, payloads = live_client.query(
        "SELECT CAST(payload AS JSON)::VARCHAR FROM events WHERE kind = 'decision'"
    )
    claim_decisions = [
        json.loads(row[0])["judge"]["answers"] for row in payloads if "grounded.1" in row[0]
    ]
    assert len(claim_decisions) == 1
    assert claim_decisions[0]["faithful.1"]["value"] == "yes"
    assert claim_decisions[0]["rule_supported"]["probabilities"] == {"yes": 0.5, "no": 0.5}
    assert all("The mem command reported memory.duckdb" not in row[0] for row in payloads)
    # The rules floor cannot tell whether a lesson is already enforced, so its p(none) sits
    # inside the band and the draft goes to review rather than straight to a write.
    assert redone["action"] == "review"
    assert any(r.startswith("enforced=0.60 within") for r in redone["reasons"])
    _, (count,) = live_client.query(
        "SELECT count(*) FROM events WHERE e.kind = 'decision'".replace("e.", "")
    )
    assert count[0] == 5
    assert decide_items(live_cfg, "t1", ["rules"]).written == {"rules": 0}

    # CLI status on the same run, then purge.
    code = main(
        [
            "--home",
            str(live_cfg.home),
            "--port",
            str(live_cfg.port),
            "judge",
            "status",
            "--run",
            "t1",
        ]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "verdicts rules" in out
    assert "frozen" in out
    assert purge_run(live_cfg, "t1") == 4
    assert not store.projections_dir.exists()


def test_cli_refuses_default_home(capsys, monkeypatch):
    monkeypatch.delenv("AGENT_MEMORY_HOME", raising=False)
    code = main(["--home", "~/.agent-memory", "judge", "status", "--run", "x"])
    err = capsys.readouterr().err
    assert code == 1
    assert "refuses to run against the default home" in err


@pytest.mark.parametrize("command", ["decide", "live"])
def test_judge_requires_initialized_agent_before_work(tmp_path, monkeypatch, capsys, command):
    from agent_memory import db
    from agent_memory.config import Config
    from agent_memory.smoke import free_port

    cfg = Config(home=tmp_path / "legacy-experiment", port=free_port())
    migrations = db.available_migrations()
    with monkeypatch.context() as patch:
        patch.setattr(db, "available_migrations", lambda: [m for m in migrations if m.version < 5])
        assert main(["--home", str(cfg.home), "init"]) == 0
    store = RunStore(cfg, "legacy")
    store.create({"questions_version": "questions_v4", "frozen": {"synthetic": True}})
    before = store.run_json.read_bytes()
    server = db.MemoryServer(cfg, export_interval_hours=0)
    # Simulate a server that was started by the pre-migration package.
    with monkeypatch.context() as patch:
        patch.setattr(db, "available_migrations", lambda: [m for m in migrations if m.version < 5])
        server.start()
    try:
        args = ["--backend", "rules"]
        if command == "live":
            args += ["--since", "2026-01-01"]
        assert (
            main(
                [
                    "--home",
                    str(cfg.home),
                    "--port",
                    str(cfg.port),
                    "judge",
                    command,
                    "--run",
                    "legacy",
                    *args,
                ]
            )
            == 1
        )
        assert f"run mem --home {cfg.home} init" in capsys.readouterr().err
        assert store.run_json.read_bytes() == before
        assert not store.live_dir.exists()
    finally:
        server.stop()
    assert main(["--home", str(cfg.home), "init"]) == 0
    server = db.MemoryServer(cfg, export_interval_hours=0)
    server.start()
    try:
        assert decide_items(cfg, "legacy", ["rules"]).written == {"rules": 0}
    finally:
        server.stop()


def test_cli_decide_without_remote_opt_in(live_cfg, capsys, monkeypatch):
    monkeypatch.delenv("AGENT_MEMORY_JUDGE_ALLOW_REMOTE", raising=False)
    store = RunStore(live_cfg, "t2")
    store.create({"questions_version": "questions_v4"})
    store.write_projection("00000000-0000-0000-0000-000000000001", "state")
    store.write_jsonl(
        store.items_path,
        [
            {
                "item_id": "00000000-0000-0000-0000-000000000001",
                "stratum": "T0",
                "flags": ["T0"],
                "facts": {},
                "turn_event_id": "t",
                "projection_hash": "h",
            }
        ],
    )
    code = main(
        [
            "--home",
            str(live_cfg.home),
            "--port",
            str(live_cfg.port),
            "judge",
            "decide",
            "--run",
            "t2",
            "--backend",
            "jev",
        ]
    )
    err = capsys.readouterr().err
    assert code == 1
    assert "AGENT_MEMORY_JUDGE_ALLOW_REMOTE" in err


def test_a_model_name_between_colons_is_not_printed_as_an_emoji(capsys):
    """Rich reads `:dolphin:` as the name of an emoji, so the backend `llm:dolphin:latest`
    reached the terminal as `llm`, a dolphin, and `latest`, in progress lines and tables."""
    judge_cli._log("llm:dolphin:latest 1/3: review (0.0s)")
    judge_cli._warn("llm:m: item i failed: down")
    table = Table()
    table.add_row("verdicts llm:dolphin:latest")
    judge_cli.console.print(table)
    out, err = capsys.readouterr()
    assert "llm:dolphin:latest 1/3: review (0.0s)" in out
    assert "verdicts llm:dolphin:latest" in out
    assert "llm:m: item i failed: down" in err


def test_import_writes_what_the_ledger_accepts_and_retries_the_file(
    live_cfg, tmp_path, monkeypatch
):
    project = tmp_path / "-Users-tester-code-demo"
    project.mkdir()
    shutil.copy(FIXTURES / "droid_session_judge.jsonl", project / "judge-session-refused.jsonl")
    real = DroidAdapter.from_transcript

    def with_a_refused_record(adapter, path, **options):
        records = real(adapter, path, **options)
        refused = replace(
            records[0],
            kind="memory_write",
            source_key="refused",
            memory={"content": ""},
            observation=None,
        )
        return [*records, refused]

    with monkeypatch.context() as patch:
        patch.setattr(DroidAdapter, "from_transcript", with_a_refused_record)
        first = import_sessions(live_cfg, tmp_path)
    assert first.files == 0
    assert first.inserted > 0
    (error,) = first.errors
    assert "judge-session-refused.jsonl: 1 of" in error
    assert "no content" in error
    # The file was not logged as imported, so the next import reads it again.
    second = import_sessions(live_cfg, tmp_path)
    assert (second.files, second.skipped_files, second.inserted) == (1, 0, 0)
    assert second.errors == []


def test_a_second_candidates_for_one_run_is_refused_at_once(live_cfg, sessions_dir, monkeypatch):
    """Two commands used to project the same turns and each write the run's items."""
    import_sessions(live_cfg, sessions_dir)
    started, release = threading.Event(), threading.Event()
    real = pipeline.project_candidate
    calls = itertools.count()

    def held(*args, **kwargs):
        if next(calls) == 0:
            started.set()
            assert release.wait(30)
        return real(*args, **kwargs)

    monkeypatch.setattr(pipeline, "project_candidate", held)
    build = partial(build_candidates, live_cfg, "t-two-builders", n=4, seed=1)
    with ThreadPoolExecutor(1) as pool:
        first = pool.submit(build)
        assert started.wait(30)
        try:
            with pytest.raises(JudgeError, match="another command is writing"):
                build()
        finally:
            release.set()
        built = first.result(60).items
    assert built == len(RunStore(live_cfg, "t-two-builders").items()) > 0
    with pytest.raises(JudgeError, match="already has items"):
        build()


def test_a_second_import_into_one_home_is_refused_at_once(live_cfg, sessions_dir):
    with (
        sole_writer(imports_log(live_cfg)),
        pytest.raises(JudgeError, match="another command is writing"),
    ):
        import_sessions(live_cfg, sessions_dir)
    assert import_sessions(live_cfg, sessions_dir).errors == []


def test_a_session_another_session_started_is_read_as_an_agents(live_cfg, live_client, tmp_path):
    # A subagent's session as Droid's hooks capture it: its start hook is the first to
    # write the session's row and names the session that called it.
    def start(session_id: str, *, keeps_caller: bool = True, **fields) -> Record:
        payload = {
            "session_id": session_id,
            "hook_event_name": "SessionStart",
            "source": "startup",
            "cwd": "/synthetic/callers",
            "permission_mode": "auto-high",
            **fields,
        }
        (record,) = DroidAdapter().from_hook(payload)
        if keeps_caller:
            return record
        # A row written before the adapter kept the caller in session metadata. The start's
        # event held it then too, in its payload.
        assert record.session is not None
        return replace(
            record, session=replace(record.session, metadata={"permission_mode": "auto-high"})
        )

    called = {"calling_session_id": "callers-parent"}
    records = [
        start("callers-spawned", **called),
        start("callers-earlier", keeps_caller=False, **called),
        start("callers-imported", keeps_caller=False, **called),
        start("callers-own"),
    ]
    written = deliver(live_cfg, records).written
    assert written is not None
    assert (written.sessions, written.events) == (4, 4)
    # The judge imported the transcript of one of them. Its first entry names the caller
    # as well, and its title gives the caller's role, which no hook says.
    project = tmp_path / "-synthetic-callers"
    project.mkdir()
    entries = [
        {
            "type": "session_start",
            "id": "callers-imported",
            "title": "Worker: Build the parser",
            "callingSessionId": "callers-parent",
        },
        {
            "type": "message",
            "id": "callers-imported-1",
            "timestamp": "2026-11-12T10:00:01Z",
            "message": {"role": "user", "content": [{"type": "text", "text": "Build it."}]},
        },
    ]
    (project / "callers-imported.jsonl").write_text(
        "".join(json.dumps(entry) + "\n" for entry in entries)
    )
    imported = import_sessions(live_cfg, tmp_path)
    assert (imported.files, imported.errors) == (1, [])

    callers = session_callers(live_cfg, live_client)
    roles = {"callers-spawned": None, "callers-earlier": None, "callers-imported": "worker"}
    for name, role in roles.items():
        caller = callers[ids.session_id("factory-droid", name)]
        assert (caller.kind, caller.role, caller.calling_session_id) == (
            "agent",
            role,
            "callers-parent",
        )
    assert callers[ids.session_id("factory-droid", "callers-own")] == HUMAN


def test_a_notice_only_the_user_saw_starts_no_turn(live_cfg, live_client, tmp_path):
    # Droid tells the user that the provider refused a request, in a message the agent is
    # not given, and the agent goes on. The import writes a note there, so the turn the
    # prompt began runs on to its outcome.
    refused = "The model provider declined this request again."
    entries = [
        {"type": "session_start", "id": "notice-turn", "timestamp": "2026-11-13T10:00:00Z"},
        said("p1", "user", "Run it.", "2026-11-13T10:00:01Z"),
        seen_by(said("n1", "user", refused, "2026-11-13T10:00:02Z"), "user_only"),
        said("r1", "assistant", "Done.", "2026-11-13T10:00:03Z"),
        {"type": "agent_turn_outcome", "turnId": "p1", "reason": "completed", "resultKind": "text"},
    ]
    records = DroidAdapter().from_transcript(write(tmp_path / "notice-turn.jsonl", entries))
    written = deliver(live_cfg, records).written
    assert written is not None
    assert written.events == 5
    sid = ids.session_id("factory-droid", "notice-turn")
    _, rows = live_client.query(
        f"SELECT kind, content FROM events WHERE session_id = {sql_literal(sid)} ORDER BY ts"
    )
    assert [tuple(row) for row in rows][1:4] == [
        ("user_msg", "Run it."),
        ("note", refused),
        ("assistant_msg", "Done."),
    ]
    (turn,) = segment_turns(fetch_turn_events(live_client, session_ids=[sid]))
    assert [event.kind for event in turn.events] == ["user_msg", "assistant_msg", "outcome"]
    assert (turn.closed_by, turn.next_user) == ("outcome", None)
