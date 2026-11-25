"""Production admission through a real throwaway Quack server and blinded frontends."""

from __future__ import annotations

import io
import json
import os
import re
import select
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Barrier
from uuid import uuid4

import pytest

from agent_memory import endpoints
from agent_memory.admission import ledger, run
from agent_memory.admission.review import ReviewController, review_day
from agent_memory.admission.store import AdmissionStore, write_jsonl
from agent_memory.capture.pipeline import make_redactor
from agent_memory.capture.record import Record, SessionInfo
from agent_memory.capture.writer import Writer
from agent_memory.cli import main
from agent_memory.db import MemoryClient, MemoryServer
from agent_memory.judge import label_frontend
from agent_memory.judge.backends.base import answer_from_probabilities
from agent_memory.judge.drafter import Draft
from agent_memory.judge.label_build import frontend_binary
from agent_memory.judge.label_controller import Action
from agent_memory.judge.label_frontend import PlainFrontend
from agent_memory.judge.lesson import Claim, Lesson
from agent_memory.judge.questions import DEFAULT_THRESHOLDS
from agent_memory.judge.store import JudgeError, RunStore


@pytest.fixture
def admission_live(cfg):
    server = MemoryServer(cfg, export_interval_hours=0)
    server.start()
    try:
        yield cfg
    finally:
        server.stop()


def install_model(cfg):
    store = AdmissionStore(cfg)
    model = {
        "version": "synthetic-v1",
        "backend": "llm:synthetic",
        "embedder": "ollama:synthetic",
        "thresholds": dict(DEFAULT_THRESHOLDS),
        "metrics": {"held_out": {"precision": 1.0, "scored_writes": 20}},
    }
    store.write_approver(model)
    write_jsonl(store.examples_path, [])
    return store


def capture(cfg, *, session="capture", count=1):
    stamp = datetime.now(UTC) - timedelta(minutes=1)
    records = []
    for i in range(count):
        for seq, (kind, content) in enumerate(
            [
                ("user_msg", f"Remember project setup {i}."),
                ("assistant_msg", "The synthetic evidence is available."),
                ("outcome", "completed"),
            ]
        ):
            records.append(
                Record(
                    agent_id="factory-droid",
                    harness_session_id=session,
                    kind=kind,
                    source_key=f"{i}-{kind}",
                    content=content,
                    ts=stamp + timedelta(seconds=i * 3 + seq),
                    seq=i * 3 + seq,
                    session=SessionInfo(project="synthetic-project", started_at=stamp),
                )
            )
    with MemoryClient.from_config(cfg) as client:
        Writer(client, cfg.blob_dir, make_redactor(cfg)).write(records)


class Backend:
    name = "llm:synthetic"
    local = True

    def decide(self, inp, questions):
        answers = {}
        for question in questions:
            selected = (
                "none"
                if question.id in ("enforced", "category")
                else "1"
                if question.type == "score"
                else "yes"
            )
            probs = {o: 0.01 for o in question.options()}
            probs[selected] = 0.99
            answers[question.id] = answer_from_probabilities(
                question,
                probs,
                method="synthetic",
                model_snapshot="test",
                latency_ms=1,
            )
        return answers


@pytest.fixture
def models(monkeypatch):
    class Drafter:
        def draft(self, item_id, state):
            assert "NEXT MESSAGE" not in state
            return Draft.from_lesson(
                item_id,
                Lesson(
                    rule="Use the synthetic project setup.",
                    kind="project_fact",
                    claims=[Claim(text="The synthetic evidence is available.", cited=(2,))],
                ),
                drafter="ollama:synthetic",
                model_snapshot="test",
                latency_ms=1,
            )

    monkeypatch.setattr(run, "make_drafter", lambda spec: Drafter())
    monkeypatch.setattr(run, "make_backend", lambda spec: Backend())
    monkeypatch.setattr(run.approver, "validate_examples", lambda model, examples: examples)
    monkeypatch.setattr(
        endpoints, "post_json", lambda url, body: ({"embeddings": [[1.0, 0.0]]}, 1.0)
    )
    monkeypatch.setattr(
        run.approver,
        "score",
        lambda *args: {
            "action": "write",
            "score": 0.95,
            "reasons": ["synthetic score"],
            "neighbor_vote": 1.0,
            "max_similarity": 1.0,
        },
    )


def query(cfg, sql):
    with MemoryClient.from_config(cfg) as client:
        return client.query(sql)[1]


def test_shadow_provenance_idempotency_and_content_dedup(admission_live, models):
    cfg = admission_live
    store = install_model(cfg)
    capture(cfg, count=2)
    result = run.run_day(cfg, day="2026-11-10", drafter="ollama:synthetic")
    assert result["processed"] == 2
    assert query(cfg, "SELECT status FROM memory_status") == [("quarantined",)]
    assert query(cfg, "SELECT action, actor FROM memory_actions") == [("confirm", "admission")]
    metadata = json.loads(query(cfg, "SELECT metadata::JSON::VARCHAR FROM memories")[0][0])
    assert metadata["lesson"]["kind"] == "project_fact"
    assert metadata["admission"]["score"] == 0.95
    assert metadata["admission"]["approver_version"] == "synthetic-v1"
    assert len(query(cfg, "SELECT * FROM memory_sources")) == 4
    assert {r[0] for r in query(cfg, "SELECT relation FROM memory_sources")} == {"derived_from"}
    events = query(
        cfg,
        "SELECT e.content, e.payload::JSON::VARCHAR, s.agent_id, s.harness_session_id "
        "FROM events e JOIN sessions s USING(session_id) WHERE kind = 'decision'",
    )
    assert len(events) == 2
    for content, payload, agent, session in events:
        assert (agent, session) == ("admission", "admission-2026-11-10")
        assert "The synthetic evidence" not in content + payload
        assert "Use the synthetic project setup" not in content + payload
    before = len(query(cfg, "SELECT * FROM events"))
    assert run.run_day(cfg, day="2026-11-10", drafter="ollama:synthetic")["processed"] == 0
    assert len(query(cfg, "SELECT * FROM events")) == before
    day = store.day("2026-11-10")
    assert len(RunStore.read_jsonl(day.root / "admitted.jsonl")) == 2
    assert all(p.stat().st_mode & 0o777 == 0o600 for p in day.root.rglob("*") if p.is_file())


def test_admission_writes_as_its_own_agent_in_an_observed_session(admission_live, models):
    """Admission used to write as agent judge, in sessions of scope judge that no capture
    observation covered. Migration 0005 registers judge for experiment verdicts only."""
    cfg = admission_live
    install_model(cfg)
    capture(cfg)
    run.run_day(cfg, day="2026-11-10", drafter="ollama:synthetic")
    [(sid, agent, scope, project, completeness, metadata)] = query(
        cfg,
        "SELECT session_id::VARCHAR, agent_id, scope, project, capture_completeness, "
        "metadata::JSON::VARCHAR FROM sessions WHERE harness_session_id = 'admission-2026-11-10'",
    )
    assert (agent, scope, project, completeness) == (
        "admission",
        "admission",
        "synthetic-project",
        "partial",
    )
    assert json.loads(metadata)["via"] == "mem admit"
    assert query(cfg, "SELECT count(*) FROM sessions WHERE agent_id = 'judge'") == [(0,)]
    assert query(
        cfg,
        "SELECT source_mode, adapter_version, exclusions::JSON::VARCHAR "
        f"FROM capture_observations WHERE session_id = '{sid}'",
    ) == [("record", "mem_admit_v1", '["no_complete_capture_evidence"]')]
    [(content, content_bytes, tokens)] = query(
        cfg, "SELECT content, content_bytes, token_estimate FROM events WHERE kind = 'decision'"
    )
    assert content_bytes == len(content.encode())
    assert tokens == content_bytes // 4
    # Its own decisions are never read back as turns to draft lessons from.
    assert run.run_day(cfg, day="2026-11-11", drafter="ollama:synthetic")["processed"] == 0


def test_the_decision_event_keeps_the_judges_action_beside_the_approvers(
    admission_live, models, monkeypatch
):
    """The approver's action used to be written over the judge's, so the event reported an
    action the judge never gave and lost the judge's reasons."""
    cfg = admission_live
    store = install_model(cfg)
    capture(cfg)
    monkeypatch.setattr(
        run.approver,
        "score",
        lambda *args: {"action": "review", "score": 0.6, "reasons": ["below the write cut"]},
    )
    run.run_day(cfg, day="2026-11-10", drafter="ollama:synthetic")
    day = store.day("2026-11-10")
    [item] = RunStore.read_jsonl(day.items_path)
    [verdict] = RunStore.read_jsonl(day.verdicts_path("llm:synthetic"))
    assert (verdict["action"], verdict["reasons"]) == ("write_lesson", ["all thresholds cleared"])
    [(content, payload)] = query(
        cfg, "SELECT content, payload::JSON::VARCHAR FROM events WHERE kind = 'decision'"
    )
    event = json.loads(payload)
    assert (event["judge"]["action"], event["judge"]["reasons"]) == (
        "write_lesson",
        ["all thresholds cleared"],
    )
    assert event["admission"]["action"] == "review"
    assert event["admission"]["reasons"] == ["below the write cut"]
    assert (event["admission"]["approver_version"], event["admission"]["score"]) == (
        "synthetic-v1",
        0.6,
    )
    assert content == (
        "judge llm:synthetic: write_lesson; approver synthetic-v1: review "
        f"for turn {item['turn_event_id']}"
    )
    # The memory's own admission metadata is what it was: the approver's score and reasons.
    metadata = json.loads(query(cfg, "SELECT metadata::JSON::VARCHAR FROM memories")[0][0])
    assert set(metadata["admission"]) == {
        "approver_version",
        "score",
        "reasons",
        "draft_hash",
        "item_id",
        "backend",
        "drafter",
    }


def test_discard_has_no_ledger_writes(admission_live, models, monkeypatch):
    cfg = admission_live
    install_model(cfg)
    capture(cfg)
    monkeypatch.setattr(
        run.approver,
        "score",
        lambda *args: {
            "action": "discard",
            "score": 0.1,
            "reasons": ["low score"],
        },
    )
    before = len(query(cfg, "SELECT * FROM events"))
    assert run.run_day(cfg, day="2026-11-10")["processed"] == 1
    assert query(cfg, "SELECT * FROM memories") == []
    assert len(query(cfg, "SELECT * FROM events")) == before


def test_auto_adds_approval(admission_live, models, monkeypatch):
    cfg = admission_live
    store = install_model(cfg)
    monkeypatch.setattr(run.approver, "can_auto", lambda model, min_writes=20: True)
    store.write_state(mode="auto", approver_version="synthetic-v1")
    capture(cfg)
    run.run_day(cfg, day="2026-11-10")
    assert query(cfg, "SELECT status FROM memory_status") == [("active",)]
    assert query(cfg, "SELECT action, actor FROM memory_actions") == [("approve", "admission")]
    reason = query(cfg, "SELECT reason FROM memory_actions")[0][0]
    assert "synthetic-v1" in reason
    assert "0.95" in reason


def test_old_auto_day_cannot_borrow_new_approver_permission(
    admission_live,
    models,
    monkeypatch,
):
    cfg = admission_live
    store = install_model(cfg)
    store.write_state(mode="auto", approver_version="synthetic-v1")
    monkeypatch.setattr(run.approver, "can_auto", lambda model, min_writes=20: True)
    capture(cfg)
    original = run.write_jsonl

    def interrupt(path, rows, **kwargs):
        if path.name == "scores.jsonl":
            raise OSError("interrupted score checkpoint")
        return original(path, rows, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(run, "write_jsonl", interrupt)
        with pytest.raises(OSError, match="checkpoint"):
            run.run_day(cfg, day="2026-11-10")
    store.write_approver({**store.read_approver(), "version": "synthetic-v2"})
    store.write_state(mode="auto", approver_version="synthetic-v2")
    run.run_day(cfg, day="2026-11-11")
    assert query(cfg, "SELECT status FROM memory_status") == [("quarantined",)]
    assert query(cfg, "SELECT * FROM memory_actions") == []


def test_resume_after_commit_and_late_turn_in_old_session(admission_live, models, monkeypatch):
    cfg = admission_live
    store = install_model(cfg)
    capture(cfg)
    original = run.write_jsonl

    def fail_after_commit(path, rows, **kwargs):
        if path.name == "admitted.jsonl":
            raise OSError("synthetic disk interruption")
        return original(path, rows, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(run, "write_jsonl", fail_after_commit)
        with pytest.raises(OSError, match="disk interruption"):
            run.run_day(cfg, day="2026-11-09")
    assert store.read_state().get("watermark") is None
    assert len(query(cfg, "SELECT * FROM memories")) == 1
    assert run.run_day(cfg, day="2026-11-10")["processed"] == 1
    assert len(query(cfg, "SELECT * FROM memories")) == 1
    assert len(query(cfg, "SELECT * FROM events WHERE kind = 'decision'")) == 1
    capture(cfg, count=2)  # Same session, one previously uncaptured turn.
    assert run.run_day(cfg, day="2026-11-10")["processed"] == 1
    assert len(query(cfg, "SELECT * FROM memories")) == 1


@pytest.mark.parametrize(
    "row_dated", [timedelta(seconds=-30), timedelta(hours=1)], ids=["before the run", "after it"]
)
def test_a_session_is_found_and_dated_by_its_first_event_not_by_its_row(
    admission_live, models, row_dated
):
    cfg = admission_live
    store = install_model(cfg)
    # The first record the ledger gets of the session is a later one, which dates its row:
    # a record that arrives out of order, or one from a clock that ran ahead.
    late = Record("factory-droid", "capture", "note", datetime.now(UTC) + row_dated, "n")
    with MemoryClient.from_config(cfg) as client:
        Writer(client, cfg.blob_dir, make_redactor(cfg)).write([late])
    capture(cfg)
    assert run.run_day(cfg, day="2026-11-10")["processed"] == 1
    [(began, earlier)] = query(
        cfg,
        "SELECT started_at::VARCHAR, started_at < recorded_started_at FROM session_summary "
        "WHERE harness_session_id = 'capture'",
    )
    assert earlier
    [item] = RunStore.read_jsonl(store.day("2026-11-10").items_path)
    assert item["session_started_at"] == began


def test_a_run_is_refused_before_a_day_begins_where_the_server_lacks_the_summary(
    tmp_path, monkeypatch, models
):
    from agent_memory import db
    from agent_memory.config import Config
    from agent_memory.judge.provenance import SUMMARY_MIGRATION
    from agent_memory.smoke import free_port

    cfg = Config(home=tmp_path / "older", port=free_port())
    earlier = [m for m in db.available_migrations() if m.version < SUMMARY_MIGRATION]
    server = MemoryServer(cfg, export_interval_hours=0)
    # A server that a package without the migration started.
    with monkeypatch.context() as patch:
        patch.setattr(db, "available_migrations", lambda: earlier)
        db.initialize_home(cfg)
        server.start()
    try:
        store = install_model(cfg)
        capture(cfg)
        with pytest.raises(JudgeError, match=re.escape(f"mem --home {cfg.home} migrate")):
            run.run_day(cfg, day="2026-11-10")
        assert not store.day("2026-11-10").exists()
    finally:
        server.stop()


@pytest.mark.parametrize(
    ("answer", "status", "action"),
    [
        ("1", "active", "approve"),
        ("2", "retired", "retire"),
        ("3", "quarantined", None),
    ],
)
def test_plain_review_blinded_actions_and_examples(
    admission_live,
    models,
    monkeypatch,
    answer,
    status,
    action,
):
    cfg = admission_live
    store = install_model(cfg)
    capture(cfg)
    run.run_day(cfg, day="2026-11-10")
    monkeypatch.setattr(sys, "stdin", io.StringIO(f"{answer}\n2\n1\n"))
    output = io.StringIO()
    monkeypatch.setattr(sys, "stdout", output)
    assert review_day(cfg, "2026-11-10", plain=True)["reviewed"] == 1
    rendered = output.getvalue()
    assert "Store this exact lesson?" in rendered
    assert "The synthetic evidence is available." in rendered
    for hidden in ("0.95", "synthetic-v1", "ollama:", "llm:", "synthetic score"):
        assert hidden not in rendered
    assert query(cfg, "SELECT status FROM memory_status") == [(status,)]
    actions = query(cfg, "SELECT action, actor FROM memory_actions")
    assert actions == ([(action, "human")] if action else [])
    examples = RunStore.read_jsonl(store.examples_path)
    assert len(examples) == (1 if action else 0)
    if examples:
        assert examples[0]["label"] is (answer == "1")
        assert examples[0]["source"] == actions_source(cfg)
    assert ReviewController(cfg, "2026-11-10").done
    with pytest.raises(JudgeError, match="this sitting has ended"):
        ReviewController(cfg, "2026-11-10").apply(Action("answer", "1"))


def actions_source(cfg):
    return query(cfg, "SELECT action_id::VARCHAR FROM memory_actions ORDER BY created_at DESC")[0][
        0
    ]


def test_review_discard_admits_on_yes_and_corrects(admission_live, models, monkeypatch):
    cfg = admission_live
    store = install_model(cfg)
    capture(cfg)
    monkeypatch.setattr(
        run.approver,
        "score",
        lambda *args: {
            "action": "discard",
            "score": 0.1,
            "reasons": ["low score"],
        },
    )
    run.run_day(cfg, day="2026-11-10")
    assert ReviewController(cfg, "2026-11-10", discards=0).done
    controller = ReviewController(cfg, "2026-11-10", discards=30)
    output = io.StringIO()
    frontend = PlainFrontend(io.StringIO("1\n1\nA [bold] literal note.\n2\n1\n"), output)
    while not controller.done:
        controller.apply(frontend.exchange(controller.screen()))
    assert query(cfg, "SELECT status FROM memory_status") == [("active",)]
    assert query(cfg, "SELECT action, actor FROM memory_actions") == [("approve", "human")]
    assert "A [bold] literal note." in output.getvalue()
    # Reopening a reviewed discard must retire the same memory, not invent an ID.
    controller = ReviewController(cfg, "2026-11-10", again=True)
    for action in (Action("answer", "n"), Action("answer", "continue"), Action("answer", "save")):
        controller.apply(action)
    assert query(cfg, "SELECT status FROM memory_status") == [("retired",)]
    examples = RunStore.read_jsonl(store.examples_path)
    assert [r["label"] for r in examples] == [True, False]
    assert examples[0]["memory_id"] == examples[1]["memory_id"]
    assert examples[0]["split"] == examples[1]["split"]


def test_review_navigation_and_pending_save_recovery(admission_live, models, monkeypatch):
    cfg = admission_live
    store = install_model(cfg)
    capture(cfg)
    run.run_day(cfg, day="2026-11-10")
    controller = ReviewController(cfg, "2026-11-10")
    controller.apply(Action("answer", "n"))
    controller.apply(Action("back"))
    assert controller.screen().selected == "n"
    controller.apply(Action("answer", "y"))
    controller.apply(Action("answer", "continue"))
    controller.apply(Action("redo"))
    controller.apply(Action("answer", "y"))
    controller.apply(Action("answer", "continue"))
    original = ledger.review_memory
    with monkeypatch.context() as patch:
        patch.setattr(
            ledger, "review_memory", lambda *args: (_ for _ in ()).throw(OSError("offline"))
        )
        with pytest.raises(OSError, match="offline"):
            controller.apply(Action("answer", "save"))
    assert query(cfg, "SELECT * FROM memory_actions") == []
    assert ledger.review_memory is original
    assert ReviewController(cfg, "2026-11-10").done
    assert query(cfg, "SELECT action, actor FROM memory_actions") == [("approve", "human")]
    assert len(RunStore.read_jsonl(store.examples_path)) == 1


def test_correct_rejected_review_without_reviving_retired_row(admission_live, models):
    cfg = admission_live
    store = install_model(cfg)
    capture(cfg)
    run.run_day(cfg, day="2026-11-10")
    controller = ReviewController(cfg, "2026-11-10")
    for value in ("n", "continue", "save"):
        controller.apply(Action("answer", value))
    controller = ReviewController(cfg, "2026-11-10", again=True)
    assert controller.screen().selected == "n"
    for value in ("y", "continue", "save"):
        controller.apply(Action("answer", value))
    statuses = [r[0] for r in query(cfg, "SELECT status FROM memory_status")]
    assert sorted(statuses) == ["active", "retired"]
    examples = store.read_examples()
    assert len(examples) == 1
    assert examples[0]["label"] is True


def test_restore_review_recovers_after_ledger_commit(admission_live, models, monkeypatch):
    cfg = admission_live
    store = install_model(cfg)
    capture(cfg)
    run.run_day(cfg, day="2026-11-10")
    controller = ReviewController(cfg, "2026-11-10")
    for value in ("n", "continue", "save"):
        controller.apply(Action("answer", value))
    controller = ReviewController(cfg, "2026-11-10", again=True)
    controller.apply(Action("answer", "y"))
    controller.apply(Action("answer", "continue"))
    original = ledger.admit

    def interrupt(*args, **kwargs):
        original(*args, **kwargs)
        raise OSError("interrupted after commit")

    with monkeypatch.context() as patch:
        patch.setattr(ledger, "admit", interrupt)
        with pytest.raises(OSError, match="after commit"):
            controller.apply(Action("answer", "save"))
    assert ReviewController(cfg, "2026-11-10").done
    assert sorted(r[0] for r in query(cfg, "SELECT status FROM memory_status")) == [
        "active",
        "retired",
    ]
    examples = store.read_examples()
    assert len(examples) == 1
    assert examples[0]["label"] is True
    assert len(query(cfg, "SELECT * FROM memory_actions")) == 2


def test_run_real_transports_redact_before_persistence(admission_live, monkeypatch, capsys):
    """Exercise the real CLI, drafter, judge and embedder with only HTTP mocked."""
    cfg = admission_live
    install_model(cfg)
    capture(cfg)
    calls = []
    secret = "ghp_" + "a" * 36

    def post(url, body, **kwargs):
        calls.append((url, body))
        if url.endswith("/api/embed"):
            assert "NEXT MESSAGE" not in body["input"]
            assert secret not in body["input"]
            return {"embeddings": [[1.0, 0.0]]}, 1.0
        if "You write one lesson" in body["messages"][0]["content"]:
            content = {
                "lesson": {
                    "rule": f"Use the synthetic credential {secret}.",
                    "kind": "environment",
                    "claims": [{"text": "The synthetic evidence is available.", "cited": [2]}],
                }
            }
        else:
            ids = re.findall(r'- id "([^"]+)"', body["messages"][1]["content"])
            content = {
                qid: {
                    "answer": (
                        "none"
                        if qid in ("enforced", "category")
                        else "1"
                        if qid == "generalizable"
                        else "yes"
                    ),
                    "probability": 0.99,
                }
                for qid in ids
            }
        return {"message": {"content": json.dumps(content)}, "model": "synthetic"}, 1.0

    monkeypatch.setattr(endpoints, "post_json", post)
    monkeypatch.setattr(run.approver, "validate_examples", lambda model, examples: examples)
    monkeypatch.setattr(
        run.approver,
        "score",
        lambda *args: {
            "action": "review",
            "score": 0.7,
            "reasons": ["synthetic band"],
        },
    )
    args = ["--home", str(cfg.home), "--port", str(cfg.port), "admit"]
    assert main([*args, "run", "--day", "2026-11-10", "--drafter", "ollama:synthetic"]) == 0
    assert len(calls) == 4  # Draft, turn questions, draft questions and lesson embedding.
    assert query(cfg, "SELECT status FROM memory_status") == [("quarantined",)]
    assert query(cfg, "SELECT type FROM memories") == [("fact",)]
    assert "[REDACTED:github_token]" in query(cfg, "SELECT content FROM memories")[0][0]
    for path in (cfg.home / "admission").rglob("*"):
        if path.is_file():
            assert secret.encode() not in path.read_bytes()
    assert secret not in capsys.readouterr().out


def test_model_error_does_not_log_private_reply(admission_live, models, monkeypatch, capsys):
    cfg = admission_live
    install_model(cfg)
    capture(cfg)

    class BadDrafter:
        def draft(self, *args):
            raise JudgeError("judge returned no JSON: PRIVATE_SESSION_REPLY")

    monkeypatch.setattr(run, "make_drafter", lambda spec: BadDrafter())
    args = ["--home", str(cfg.home), "--port", str(cfg.port), "admit", "run"]
    assert main(args) == 1
    output = capsys.readouterr()
    assert "PRIVATE_SESSION_REPLY" not in output.err
    assert "rerun mem admit run" in output.err
    assert query(cfg, "SELECT * FROM memories") == []


@pytest.mark.parametrize(
    ("command", "args"),
    [
        ("import-labels", ["--from", "/nonexistent"]),
        ("fit", []),
        ("status", []),
        ("run", []),
        ("review", []),
        ("enable", ["--auto"]),
    ],
)
def test_every_command_requires_the_admission_agent_before_work(
    admission_live,
    monkeypatch,
    capsys,
    command,
    args,
):
    cfg = admission_live
    original = MemoryClient.query

    def missing_agent(client, sql):
        return (
            (["agent_id"], [])
            if sql == "SELECT agent_id FROM agents WHERE agent_id = 'admission'"
            else original(client, sql)
        )

    monkeypatch.setattr(MemoryClient, "query", missing_agent)
    assert main(["--home", str(cfg.home), "--port", str(cfg.port), "admit", command, *args]) == 1
    assert "stop the server" in capsys.readouterr().err
    assert not (cfg.home / "admission").exists()


def test_jev_refused_without_optin(admission_live, monkeypatch):
    cfg = admission_live
    store = install_model(cfg)
    model = store.read_approver()
    store.write_approver({**model, "backend": "jev:typesafe/jev-1.14"})
    monkeypatch.delenv("AGENT_MEMORY_JUDGE_ALLOW_REMOTE", raising=False)
    with pytest.raises(JudgeError, match="AGENT_MEMORY_JUDGE_ALLOW_REMOTE"):
        run.run_day(cfg)
    assert not (store.root / "days").exists()


def test_judge_never_imports_admission():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import agent_memory.judge.cli; import sys; "
            "assert not any(k.startswith('agent_memory.admission') for k in sys.modules)",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_full_cli_label_import_fit_enable_run_and_review(
    admission_live,
    tmp_path,
    monkeypatch,
    capsys,
):
    """Real fitted scoring and auto gates, not the production-path unit-test stubs."""
    from agent_memory.admission.embed import DEFAULT_EMBEDDER
    from tests.test_admission_fit import BACKEND, _freeze, _mock_embed, _row
    from tests.test_judge import structured_draft, synthetic_run

    cfg = admission_live
    source = synthetic_run(tmp_path, n=20)
    _freeze(source)
    before = {p: p.read_bytes() for p in source.root.rglob("*") if p.is_file()}
    _mock_embed(monkeypatch)
    args = ["--home", str(cfg.home), "--port", str(cfg.port), "admit"]
    assert main([*args, "import-labels", "--from", str(source.root), "--backend", BACKEND]) == 0
    assert main([*args, "fit", "--backend", BACKEND]) == 0
    assert main([*args, "enable", "--auto"]) == 1  # Fewer than 20 held-out examples.
    assert "at least 20 scored" in capsys.readouterr().err
    assert all(p.read_bytes() == body for p, body in before.items())
    store = AdmissionStore(cfg)
    rows = [_row(i, bool(i % 2), "calibration" if i < 30 else "held_out") for i in range(80)]
    write_jsonl(store.examples_path, rows)
    assert main([*args, "fit", "--backend", BACKEND]) == 0
    assert store.read_approver()["metrics"]["held_out"]["scored_writes"] == 25
    assert main([*args, "enable", "--auto"]) == 0
    assert main([*args, "status"]) == 0
    assert store.read_state()["mode"] == "auto"
    backend = Backend()
    backend.name = BACKEND
    monkeypatch.setattr(run, "make_backend", lambda spec: backend)

    class Drafter:
        def draft(self, iid, state):
            return structured_draft(iid)

    monkeypatch.setattr(run, "make_drafter", lambda spec: Drafter())
    capture(cfg)
    assert main([*args, "run", "--day", "2026-11-10"]) == 0
    assert query(cfg, "SELECT status FROM memory_status") == [("active",)]
    assert query(cfg, "SELECT actor FROM memory_actions") == [("admission",)]
    assert main([*args, "run", "--day", "2026-11-10"]) == 0
    assert len(query(cfg, "SELECT * FROM memory_actions")) == 1
    controller = ReviewController(cfg, "2026-11-10")
    for value in ("n", "continue", "save"):
        controller.apply(Action("answer", value))
    assert query(cfg, "SELECT status FROM memory_status") == [("retired",)]
    assert store.read_state()["mode"] == "shadow"  # New labels disarm the stale model.
    assert main([*args, "enable", "--auto"]) == 1
    assert "examples changed" in capsys.readouterr().err
    assert main([*args, "enable", "--shadow"]) == 0
    # Unclear corrections revoke previous training labels without changing ledger status.
    controller = ReviewController(cfg, "2026-11-10", again=True)
    for value in ("u", "continue", "save"):
        controller.apply(Action("answer", value))
    production = [r for r in store.read_examples() if r.get("source_type") == "review"]
    assert len(production) == 1
    assert production[0]["label"] is None
    assert production[0]["embedder"] == DEFAULT_EMBEDDER
    assert query(cfg, "SELECT status FROM memory_status") == [("retired",)]
    assert main([*args, "fit", "--backend", BACKEND]) == 0
    assert store.read_approver()["counts"]["calibration"] == 30


def test_cli_enable_adjustable_minimum_and_missing_fit(admission_live, capsys):
    from agent_memory.admission.embed import DEFAULT_EMBEDDER
    from agent_memory.admission.fit import fit_approver
    from tests.test_admission_fit import BACKEND, _row

    cfg = admission_live
    args = ["--home", str(cfg.home), "--port", str(cfg.port), "admit"]
    assert main([*args, "run"]) == 1
    assert "import v4 human labels" in capsys.readouterr().err
    store = AdmissionStore(cfg)
    write_jsonl(
        store.examples_path,
        [_row(i, True, "calibration" if i < 20 else "held_out") for i in range(39)],
    )
    fit_approver(cfg, backend=BACKEND, embedder=DEFAULT_EMBEDDER)
    assert main([*args, "enable", "--auto"]) == 1
    assert main([*args, "enable", "--auto", "--min-writes", "5"]) == 0


def test_go_review_session_drives_same_screens(admission_live, models, monkeypatch):
    cfg = admission_live
    install_model(cfg)
    capture(cfg)
    run.run_day(cfg, day="2026-11-10")
    screens, closed = [], []

    class Frontend:
        def __init__(self, binary):
            pass

        def exchange(self, screen):
            screens.append(screen.payload())
            step = screen.key.partition(":")[2]
            return Action("answer", {"store": "y", "note": "continue", "summary": "save"}[step])

        def finish(self, text):
            assert text == "Reviewed 1, skipped 0."

        def close(self):
            closed.append(True)

    monkeypatch.setattr(label_frontend, "GoFrontend", Frontend)
    monkeypatch.setattr(label_frontend, "automatic_plain", lambda: False)
    monkeypatch.setattr(label_frontend, "frontend_binary", lambda **kwargs: Path("/helper"))
    assert review_day(cfg, "2026-11-10")["reviewed"] == 1
    assert closed == [True]
    assert [s["subject"] for s in screens] == ["Draft A", "Notes", "Summary"]
    assert "synthetic-v1" not in json.dumps(screens)
    assert query(cfg, "SELECT action, actor FROM memory_actions") == [("approve", "human")]


@pytest.mark.parametrize("width", [60, 140])
def test_real_go_admission_screen(admission_live, models, tmp_path, width):
    """Send an actual admission screen across the unmodified Python/Go protocol."""
    import fcntl
    import pty
    import shutil
    import struct
    import termios

    if shutil.which("go") is None:
        pytest.skip("real admission UI round trip needs Go")
    cfg = admission_live
    install_model(cfg)
    capture(cfg)
    run.run_day(cfg, day="2026-11-10")
    screen = ReviewController(cfg, "2026-11-10").screen()
    binary = frontend_binary()
    path = tmp_path / "screen.json"
    path.write_text(json.dumps(screen.payload()))
    body = """
import fcntl, json, sys, termios
from pathlib import Path
from agent_memory.judge.label_controller import Action, DisplayDraft, Screen
from agent_memory.judge.label_frontend import GoFrontend
from agent_memory.judge.label_questions import Option
fcntl.ioctl(sys.stdin.fileno(), termios.TIOCSCTTY, 0)
data = json.loads(Path(sys.argv[2]).read_text())
data["drafts"] = [DisplayDraft(**d) for d in data["drafts"]]
data["options"] = [Option(**o) for o in data["options"]]
frontend = GoFrontend(Path(sys.argv[1]))
try:
    assert frontend.exchange(Screen(**data)) == Action("answer", "y")
    frontend.finish("Review screen accepted.")
finally:
    frontend.close()
print("ADMISSION_UI_OK", flush=True)
"""
    master, slave = pty.openpty()
    fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", 40, width, 0, 0))
    process = subprocess.Popen(
        [sys.executable, "-c", body, str(binary), str(path)],
        stdin=slave,
        stdout=slave,
        stderr=slave,
        start_new_session=True,
        env={**os.environ, "TERM": "xterm-256color", "NO_COLOR": "1"},
    )
    os.close(slave)
    output, submitted = bytearray(), False
    deadline = time.monotonic() + 15
    try:
        while time.monotonic() < deadline:
            if not select.select([master], [], [], 0.2)[0]:
                if process.poll() is not None:
                    break
                continue
            try:
                part = os.read(master, 65536)
            except OSError:
                break
            if not part:
                break
            output.extend(part)
            if not submitted and b"Store this exact lesson?" in output:
                os.write(master, b"\r")
                submitted = True
        assert submitted, output.decode(errors="replace")
        assert process.wait(timeout=3) == 0, output.decode(errors="replace")
        assert b"ADMISSION_UI_OK" in output
        assert b"Rule:" in output
        assert b"synthetic-v1" not in output
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=3)
        os.close(master)


@pytest.mark.parametrize(
    ("command", "args"),
    [
        ("import", ["--sessions-dir", "/tmp"]),
        ("candidates", ["--run", "x"]),
        ("draft", ["--run", "x"]),
        ("decide", ["--run", "x", "--backend", "rules"]),
        ("label", ["--run", "x", "--plain"]),
        ("report", ["--run", "x"]),
        ("live", ["--run", "x", "--since", "2026-01-01"]),
        ("status", ["--run", "x"]),
        ("purge", ["--run", "x"]),
    ],
)
def test_experiment_home_guard_remains(command, args, capsys):
    assert main(["--home", str(Path.home() / ".agent-memory"), "judge", command, *args]) == 1
    assert "refuses to run against the default home" in capsys.readouterr().err


def test_concurrent_admission_and_remember_all_land(admission_live):
    from agent_memory.mcp_server.tools import MemoryTools

    cfg = admission_live
    scope = f"concurrent-admit-{uuid4()}"
    capture(cfg, session="admit-race", count=1)
    with MemoryClient.from_config(cfg) as client:
        _, rows = client.query(
            "SELECT event_id::VARCHAR FROM events WHERE kind = 'outcome' LIMIT 1"
        )
        turn_event_id = str(rows[0][0])

    admits = 4
    remembers = 4
    barrier = Barrier(admits + remembers)
    created_at = datetime.now(UTC).isoformat()

    def run_admit(index: int) -> tuple[dict, str]:
        token = f"admitword{index}k"
        item = {
            "item_id": f"race-item-{index}",
            "turn_event_id": turn_event_id,
            "project": scope,
            "sidecar": [],
            "stratum": "ordinary",
            "flags": [],
            "projection_version": "projection_v1",
            "projection_hash": f"ph-{index}",
        }
        draft = {
            "drafter": "synthetic",
            "rule": f"Always verify {token} before deploying.",
            "kind": "project_fact",
            "claims": [{"text": f"Check {token}.", "cited": []}],
        }
        verdict = {
            "run_id": "race-run",
            "item_id": f"race-item-{index}",
            "backend": "llm:synthetic",
            "questions_version": "questions_v2",
            "projection_version": "projection_v1",
            "projection_hash": f"ph-{index}",
            "drafter": "synthetic",
            "draft_hash": f"dh-{index}",
            "action": "write_lesson",
            "reasons": [],
            "answers": {},
        }
        result = {
            "action": "write",
            "approver_version": "synthetic-v1",
            "score": 0.95,
            "reasons": ["passes"],
        }
        barrier.wait(timeout=10)
        admitted = ledger.admit(
            cfg, "2026-11-24", created_at, item, draft, verdict, result, auto=True
        )
        return admitted, token

    def run_remember(index: int) -> tuple[dict, str]:
        tool = MemoryTools(cfg, "factory-droid", project=scope, harness_session_id=str(uuid4()))
        try:
            tool._client_or_raise()
            token = f"alongsideadmit{index}m"
            barrier.wait(timeout=10)
            return tool.remember(f"note {token}", "fact", scope=scope), token
        finally:
            tool.close()

    with ThreadPoolExecutor(max_workers=admits + remembers) as pool:
        admit_futs = [pool.submit(run_admit, i) for i in range(admits)]
        rem_futs = [pool.submit(run_remember, i) for i in range(remembers)]
        admit_results = [f.result() for f in admit_futs]
        rem_results = [f.result() for f in rem_futs]

    assert all(r["memory_id"] for r, _ in admit_results)
    assert all(not r["queued"] for r, _ in rem_results)
    checker = MemoryTools(cfg, "factory-droid", project=scope, harness_session_id=str(uuid4()))
    try:
        for admitted, token in admit_results:
            found = [item["memory_id"] for item in checker.recall(token, scopes=[scope])["items"]]
            assert admitted["memory_id"] in found
        for written, token in rem_results:
            found = [item["memory_id"] for item in checker.recall(token, scopes=[scope])["items"]]
            assert written["memory_id"] in found
    finally:
        checker.close()


def _capture_turns(cfg, turns: list[tuple[str, str]], *, session: str = "exact-session") -> None:
    stamp = datetime.now(UTC) - timedelta(minutes=2)
    records = []
    for i, (user_text, assistant_text) in enumerate(turns):
        for seq, (kind, content) in enumerate(
            [
                ("user_msg", user_text),
                ("assistant_msg", assistant_text),
                ("outcome", "completed"),
            ]
        ):
            records.append(
                Record(
                    agent_id="factory-droid",
                    harness_session_id=session,
                    kind=kind,
                    source_key=f"{i}-{kind}",
                    content=content,
                    ts=stamp + timedelta(seconds=i * 3 + seq),
                    seq=i * 3 + seq,
                    session=SessionInfo(project="exact-project", started_at=stamp),
                )
            )
    with MemoryClient.from_config(cfg) as client:
        Writer(client, cfg.blob_dir, make_redactor(cfg)).write(records)


def _install_rule_drafter(monkeypatch, rules_by_marker: dict[str, Lesson]) -> None:
    class RuleDrafter:
        def draft(self, item_id, state):
            for marker, lesson in rules_by_marker.items():
                if marker in state:
                    return Draft.from_lesson(
                        item_id,
                        lesson,
                        drafter="ollama:synthetic",
                        model_snapshot="test",
                        latency_ms=1,
                    )
            raise AssertionError(f"unexpected state: {state!r}")

    monkeypatch.setattr(run, "make_drafter", lambda spec: RuleDrafter())


@pytest.mark.parametrize("mode", ["shadow", "auto"])
def test_exact_values_stored_on_lesson_and_memory_without_changing_action(
    admission_live, models, monkeypatch, capsys, mode
):
    cfg = admission_live
    store = install_model(cfg)
    if mode == "auto":
        monkeypatch.setattr(run.approver, "can_auto", lambda model, min_writes=20: True)
        store.write_state(mode="auto", approver_version="synthetic-v1", min_writes=20)
    _capture_turns(
        cfg,
        [
            ("MARKER_PASS check port 9494.", "Server binds 127.0.0.1:9494 in config.py."),
            ("MARKER_FAIL check port 9494.", "Server binds 127.0.0.1:9494 in config.py."),
        ],
    )
    _install_rule_drafter(
        monkeypatch,
        {
            "MARKER_PASS": Lesson(
                rule="Bind 127.0.0.1:9494 in config.py.",
                kind="environment",
                claims=[Claim(text="The server binds 127.0.0.1:9494.", cited=(2,))],
            ),
            "MARKER_FAIL": Lesson(
                rule="Bind 127.0.0.1:8888 in missing_module.py.",
                kind="environment",
                claims=[Claim(text="The server binds a port.", cited=(2,))],
            ),
        },
    )
    args = ["--home", str(cfg.home), "--port", str(cfg.port), "admit"]
    assert main([*args, "run", "--day", "2026-11-10", "--drafter", "ollama:synthetic"]) == 0
    out = capsys.readouterr().out
    assert "1 of 2 lesson(s) failed exact-value check." in out

    # Actions are unchanged in both shadow and auto modes.
    expected_status = "active" if mode == "auto" else "quarantined"
    statuses = [r[0] for r in query(cfg, "SELECT status FROM memory_status")]
    assert statuses == [expected_status, expected_status]
    if mode == "auto":
        actions = query(cfg, "SELECT action, actor FROM memory_actions")
        assert actions == [("approve", "admission"), ("approve", "admission")]

    # Stored with the lesson in drafts.jsonl.
    day = store.day("2026-11-10")
    drafts = RunStore.read_jsonl(day.drafts_path)
    by_rule = {row["rule"]: row["exact_values"] for row in drafts}
    assert by_rule["Bind 127.0.0.1:9494 in config.py."]["ok"] is True
    assert by_rule["Bind 127.0.0.1:9494 in config.py."]["missing"] == []
    failed = by_rule["Bind 127.0.0.1:8888 in missing_module.py."]
    assert failed["ok"] is False
    assert failed["missing"] == ["127.0.0.1:8888", "missing_module.py"]

    # Readable from the memories the lessons became, and kept out of decision events.
    memories = {
        content: json.loads(meta)
        for content, meta in query(cfg, "SELECT content, metadata::JSON::VARCHAR FROM memories")
    }
    assert memories["Bind 127.0.0.1:9494 in config.py."]["lesson"]["exact_values"]["ok"] is True
    mem_failed = memories["Bind 127.0.0.1:8888 in missing_module.py."]["lesson"]["exact_values"]
    assert mem_failed["ok"] is False
    assert mem_failed["missing"] == ["127.0.0.1:8888", "missing_module.py"]
    for (payload,) in query(cfg, "SELECT payload::JSON::VARCHAR FROM events WHERE kind='decision'"):
        assert "missing_module.py" not in payload
        assert "exact_values" not in payload

    # Day report records how many lessons carried values missing from their cited evidence.
    assert main([*args, "report", "--day", "2026-11-10"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert len(report["days"]) == 1
    assert report["days"][0]["lessons"] == 2
    assert report["days"][0]["exact_values"] == {
        "checked": 2,
        "passed": 1,
        "failed": 1,
        "redacted_cited": 0,
        "missing_rate": 0.5,
    }


def test_exact_values_hidden_in_blind_admission_review(admission_live, models, monkeypatch):
    cfg = admission_live
    install_model(cfg)
    _capture_turns(
        cfg,
        [("MARKER_BLIND check config.", "Server uses port 9494.")],
    )
    _install_rule_drafter(
        monkeypatch,
        {
            "MARKER_BLIND": Lesson(
                rule="Run cargo on port 8888 via src/phantom.py.",
                kind="environment",
                claims=[Claim(text="The server uses a port.", cited=(2,))],
            ),
        },
    )
    run.run_day(cfg, day="2026-11-10", drafter="ollama:synthetic")
    controller = ReviewController(cfg, "2026-11-10")
    for answer in ("n", "continue"):
        screen = controller.screen()
        blob = " ".join(
            [
                screen.subject,
                screen.metadata,
                screen.evidence,
                screen.projection,
                screen.question,
                screen.notice,
                screen.guide,
                *(d.lesson + " " + d.cited_evidence for d in screen.drafts),
            ]
        ).lower()
        assert "exact_values" not in blob
        assert "exact-value" not in blob
        assert "missing:" not in blob
        assert "redacted_cited" not in blob
        assert "failed check" not in blob
        controller.apply(Action("answer", answer))

    # PlainFrontend renders no exact-value check output either.
    buf = io.StringIO()
    plain_controller = ReviewController(cfg, "2026-11-10")
    frontend = PlainFrontend(io.StringIO("q\n"), buf)
    while not plain_controller.done:
        plain_controller.apply(frontend.exchange(plain_controller.screen()))
    rendered = buf.getvalue().lower()
    assert "exact_values" not in rendered
    assert "exact-value" not in rendered
    assert "redacted_cited" not in rendered


def test_exact_values_report_retirement_rate_over_examples_and_imports(
    admission_live, models, monkeypatch, capsys, tmp_path
):
    from agent_memory.admission.examples import import_labels
    from tests.test_judge import synthetic_run

    cfg = admission_live
    install_model(cfg)
    _capture_turns(
        cfg,
        [
            ("M_FAIL_1 check config.", "Port is 9494."),
            ("M_FAIL_2 check config.", "Port is 9494."),
            ("M_FAIL_3 check config.", "Port is 9494."),
            ("M_PASS_1 check config.", "Port is 9494."),
            ("M_PASS_2 check config.", "Port is 9494."),
        ],
    )
    _install_rule_drafter(
        monkeypatch,
        {
            "M_FAIL_1": Lesson(
                rule="Use port 8001 in app_one.py.",
                kind="environment",
                claims=[Claim(text="Port is configured.", cited=(2,))],
            ),
            "M_FAIL_2": Lesson(
                rule="Use port 8002 in app_two.py.",
                kind="environment",
                claims=[Claim(text="Port is configured.", cited=(2,))],
            ),
            "M_FAIL_3": Lesson(
                rule="Use port 8003 in app_three.py.",
                kind="environment",
                claims=[Claim(text="Port is configured.", cited=(2,))],
            ),
            "M_PASS_1": Lesson(
                rule="Use port 9494 for local server one.",
                kind="environment",
                claims=[Claim(text="Port is 9494.", cited=(2,))],
            ),
            "M_PASS_2": Lesson(
                rule="Keep port 9494 for local server two.",
                kind="environment",
                claims=[Claim(text="Port is 9494.", cited=(2,))],
            ),
        },
    )
    run.run_day(cfg, day="2026-11-10", drafter="ollama:synthetic")

    # Rate all 5 lessons: 2 of 3 failing lessons retired, 1 approved; 0 of 2 passing retired.
    decisions = {
        "app_one.py": "n",
        "app_two.py": "n",
        "app_three.py": "y",
        "local server one": "y",
        "local server two": "y",
    }
    controller = ReviewController(cfg, "2026-11-10")
    while not controller.done:
        lesson_text = controller.screen().drafts[0].lesson
        choice = next(v for key, v in decisions.items() if key in lesson_text)
        for action in (choice, "continue", "save"):
            controller.apply(Action("answer", action))

    args = ["--home", str(cfg.home), "--port", str(cfg.port), "admit"]
    assert main([*args, "report"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["totals"]["lessons"] == 5
    assert report["totals"]["exact_values"]["failed"] == 3
    assert report["examples"]["total"] == 5
    assert report["examples"]["checked"] == 5
    assert report["examples"]["unchecked"] == 0
    assert report["examples"]["failed"] == {
        "n": 3,
        "retired": 2,
        "approved": 1,
        "retired_rate": pytest.approx(2 / 3),
    }
    assert report["examples"]["passed"] == {
        "n": 2,
        "retired": 0,
        "approved": 2,
        "retired_rate": 0.0,
    }

    # Revising an answer updates the rate without double counting; Unclear withdraws it.
    controller = ReviewController(cfg, "2026-11-10", again=True)
    while not controller.done:
        lesson_text = controller.screen().drafts[0].lesson
        if "app_three.py" in lesson_text:
            for action in ("u", "continue", "save"):
                controller.apply(Action("answer", action))
        else:
            controller.apply(Action("skip"))
    assert main([*args, "report"]) == 0
    updated = json.loads(capsys.readouterr().out)
    assert updated["examples"]["failed"] == {
        "n": 2,
        "retired": 2,
        "approved": 0,
        "retired_rate": 1.0,
    }

    # Structured frozen v4 imports also carry exact_values and deduplicate across backends.
    exp = synthetic_run(tmp_path / "exp-run", n=1)
    iid = exp.items()[0]["item_id"]
    exp.write_projection(iid, "Coding-agent turn.\n[1] ASSISTANT (final): Uses port 9494.\n")
    structured = Draft.from_lesson(
        iid,
        Lesson(
            rule="Uses port 7777.",
            kind="environment",
            claims=[Claim(text="Uses a port.", cited=(1,))],
        ),
        drafter="ollama:m",
        model_snapshot="m",
        latency_ms=1.0,
    ).to_dict()
    exp.write_jsonl(exp.drafts_path, [structured])
    from agent_memory.judge.drafter import draft_hash

    dhash = draft_hash(structured["lesson"])
    labels = exp.read_jsonl(exp.labels_path)
    labels[0]["drafts"]["ollama:m"]["draft_hash"] = dhash
    labels[0]["drafts"]["ollama:m"]["store_exact"] = False
    exp.write_jsonl(exp.labels_path, labels)
    for backend in ("llm:ref", "rules"):
        verdicts = exp.read_jsonl(exp.verdicts_path(backend))
        verdicts[0]["draft_hash"] = dhash
        exp.write_jsonl(exp.verdicts_path(backend), verdicts)
    exp.update_run(
        frozen={
            "frozen_at": "2026-01-01T00:00:00+00:00",
            "backends": {
                "llm:ref": {"thresholds": dict(DEFAULT_THRESHOLDS)},
                "rules": {"thresholds": dict(DEFAULT_THRESHOLDS)},
            },
        }
    )
    import_labels(cfg, exp.root, backend="llm:ref", embedder="ollama:synthetic")
    import_labels(cfg, exp.root, backend="rules", embedder="ollama:synthetic")
    assert main([*args, "report"]) == 0
    combined = json.loads(capsys.readouterr().out)
    # Counted once despite two imported backends: 2 review fails + 1 imported fail = 3.
    assert combined["examples"]["failed"] == {
        "n": 3,
        "retired": 3,
        "approved": 0,
        "retired_rate": 1.0,
    }


def test_exact_values_redaction_marker_in_cited_lines_end_to_end(
    admission_live, models, monkeypatch, capsys
):
    cfg = admission_live
    store = install_model(cfg)
    _capture_turns(
        cfg,
        [
            (
                "MARKER_REDACT configure auth.",
                "Export api_key=synthetic_unredacted_token_123456 before running pytest.",
            ),
        ],
    )
    _install_rule_drafter(
        monkeypatch,
        {
            "MARKER_REDACT": Lesson(
                rule="Set api_key=placeholder_val before running pytest.",
                kind="environment",
                claims=[Claim(text="Export api_key before running pytest.", cited=(2,))],
            ),
        },
    )
    args = ["--home", str(cfg.home), "--port", str(cfg.port), "admit"]
    assert main([*args, "run", "--day", "2026-11-10", "--drafter", "ollama:synthetic"]) == 0
    assert "0 of 1 lesson(s) failed exact-value check." in capsys.readouterr().out

    day = store.day("2026-11-10")
    draft = RunStore.read_jsonl(day.drafts_path)[0]
    assert draft["exact_values"]["redacted_cited"] is True
    assert draft["exact_values"]["missing"] == []
    assert draft["exact_values"]["ok"] is True

    meta = json.loads(query(cfg, "SELECT metadata::JSON::VARCHAR FROM memories")[0][0])
    assert meta["lesson"]["exact_values"]["redacted_cited"] is True
    assert meta["lesson"]["exact_values"]["ok"] is True

    assert main([*args, "report", "--day", "2026-11-10"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["days"][0]["exact_values"] == {
        "checked": 1,
        "passed": 1,
        "failed": 0,
        "redacted_cited": 1,
        "missing_rate": 0.0,
    }


def test_exact_values_report_backfills_legacy_rows_and_preserves_examples_hash(
    admission_live, models, monkeypatch, capsys
):
    from agent_memory.admission import approver

    cfg = admission_live
    store = install_model(cfg)
    _capture_turns(cfg, [("M_LEGACY check port.", "Port is 9494.")])
    _install_rule_drafter(
        monkeypatch,
        {
            "M_LEGACY": Lesson(
                rule="Use port 8123 in legacy_app.py.",
                kind="environment",
                claims=[Claim(text="Port is configured.", cited=(2,))],
            ),
        },
    )
    run.run_day(cfg, day="2026-11-10", drafter="ollama:synthetic")
    controller = ReviewController(cfg, "2026-11-10")
    for action in ("n", "continue", "save"):
        controller.apply(Action("answer", action))

    # exact_values must not alter approver.examples_hash.
    examples = store.read_examples()
    stripped_examples = [{k: v for k, v in row.items() if k != "exact_values"} for row in examples]
    assert approver.examples_hash(examples) == approver.examples_hash(stripped_examples)

    # Strip exact_values from drafts.jsonl and examples.jsonl to simulate pre-MEM-9 data.
    day = store.day("2026-11-10")
    stripped_drafts = [
        {k: v for k, v in row.items() if k != "exact_values"}
        for row in RunStore.read_jsonl(day.drafts_path)
    ]
    write_jsonl(day.drafts_path, stripped_drafts)
    write_jsonl(store.examples_path, stripped_examples)

    args = ["--home", str(cfg.home), "--port", str(cfg.port), "admit"]
    assert main([*args, "report", "--day", "2026-11-10"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["days"][0]["exact_values"]["failed"] == 1
    assert report["examples"]["failed"] == {
        "n": 1,
        "retired": 1,
        "approved": 0,
        "retired_rate": 1.0,
    }
    # Read-only: report does not rewrite drafts.jsonl or examples.jsonl.
    assert "exact_values" not in RunStore.read_jsonl(day.drafts_path)[0]
    assert "exact_values" not in store.read_examples()[0]

    # Missing day fails cleanly.
    assert main([*args, "report", "--day", "2026-01-01"]) == 1
    assert "no admission day 2026-01-01" in capsys.readouterr().err
