"""Synthetic regressions for judge accounting, calibration, provenance, and bounded work."""

from __future__ import annotations

import copy
from dataclasses import replace
from typing import Any

import pytest

from agent_memory.judge import metrics, pipeline, report, triggers
from agent_memory.judge.backends.rules import RulesBackend
from agent_memory.judge.drafter import draft_hash
from agent_memory.judge.provenance import Caller
from agent_memory.judge.questions import TURN_QUESTIONS
from agent_memory.judge.store import JudgeError
from tests.test_judge import _rows, as_questions_version, synthetic_run


@pytest.mark.parametrize("first", ["draft", "decline", "failed"])
def test_the_turn_is_charged_only_to_first_persisted_verdict(tmp_path, monkeypatch, first):
    store = synthetic_run(tmp_path, n=2)
    items = store.read_jsonl(store.items_path)
    item_id = items[1]["item_id"]
    # Keep a completed item: a failure on a later item's first draft is retryable.
    path = store.verdicts_path("rules")
    store.write_jsonl(path, store.read_jsonl(path)[:1])
    drafts = store.read_jsonl(store.drafts_path)[:1]
    drafts += [
        {"item_id": item_id, "drafter": "a", "lesson": None if first == "decline" else "A"},
        {"item_id": item_id, "drafter": "b", "lesson": "B"},
        {"item_id": item_id, "drafter": "c", "lesson": None},
    ]
    store.write_jsonl(store.drafts_path, drafts)
    calls = []
    records = []
    real_decide = RulesBackend.decide

    def paid(self, inp, questions):
        asked_turn = questions is TURN_QUESTIONS
        calls.append((inp.item_id, asked_turn, inp.draft))
        if first == "failed" and inp.draft == "A":
            raise JudgeError("synthetic draft-question failure")
        answers = real_decide(self, inp, questions)
        return {
            key: replace(
                answer,
                latency_ms=12.0 if asked_turn else 3.0,
                cost_usd=0.003 if asked_turn else 0.001,
            )
            for key, answer in answers.items()
        }

    monkeypatch.setattr(RulesBackend, "decide", paid)
    monkeypatch.setattr(pipeline, "_record", lambda cfg, batch, log: records.extend(batch))
    monkeypatch.setattr(pipeline, "require_judge_agent", lambda cfg: None)
    written = pipeline.decide_items(store.cfg, store.run_id, ["rules"], store=store).written
    assert written == {"rules": 2 if first == "failed" else 3}
    rows = [row for row in store.read_jsonl(path) if row["item_id"] == item_id]
    assert [row["stage_a_reused"] for row in rows] == [False] + [True] * (len(rows) - 1)
    assert sum(asked_turn for _, asked_turn, _ in calls) == 1
    turn_ids = {q.id for q in TURN_QUESTIONS}
    for index, row in enumerate(rows):
        for key in turn_ids:
            assert row["answers"][key]["latency_ms"] == (0.0 if index else 12.0)
            assert row["answers"][key]["cost_usd"] == (0.0 if index else 0.003)
        draft_count = len(set(row["answers"]) - turn_ids)
        assert row["latency_ms"] == (0.0 if index else 72.0) + draft_count * 3.0
        assert row["cost_usd"] == pytest.approx((0.0 if index else 0.018) + draft_count * 0.001)
        if row["declined"]:
            assert row["action"] == "discard"
            assert draft_count == 0

    # Decision records carry the corrected totals, but never state or lesson text.
    assert len(records) == len(rows)
    assert len({record.source_key for record in records}) == len(rows)
    for record, row in zip(records, rows, strict=True):
        judge = record.payload["judge"]
        assert judge["latency_ms"] == row["latency_ms"]
        assert judge["cost_usd"] == row["cost_usd"]
        assert set(judge["answers"]["failed"]) == {"value", "probabilities", "confidence", "method"}
        assert "SENTINEL_STATE_TEXT" not in record.to_json()
        assert "lesson" not in judge

    # Retry the failed arm and add another arm; neither pays for the turn questions again.
    first = "draft"
    calls.clear()
    records.clear()
    store.write_jsonl(
        store.drafts_path, [{"item_id": item_id, "drafter": "d", "lesson": "D"}], append=True
    )
    written = pipeline.decide_items(store.cfg, store.run_id, ["rules"], store=store).written
    assert written["rules"] == (2 if len(rows) == 2 else 1)
    assert not any(asked_turn for _, asked_turn, _ in calls)
    resumed = store.read_jsonl(path)[-written["rules"] :]
    assert all(row["stage_a_reused"] for row in resumed)
    assert all(record.payload["judge"]["latency_ms"] == 18.0 for record in records)
    assert all(record.payload["judge"]["cost_usd"] == pytest.approx(0.006) for record in records)
    calls.clear()
    again = pipeline.decide_items(store.cfg, store.run_id, ["rules"], store=store)
    assert again.written == {"rules": 0}
    assert calls == []


@pytest.mark.parametrize("missing", ["null", "gate", "backend"])
def test_frozen_report_never_fits_a_missing_cascade_gate(tmp_path, monkeypatch, missing):
    store = synthetic_run(tmp_path)
    rules = {"thresholds": {"failed": 0.5}, "gate": None}
    if missing == "gate":
        rules.pop("gate")
    frozen = {
        "frozen_at": "2026-01-01T00:00:00+00:00",
        "reference": "llm:ref",
        "backends": {} if missing == "backend" else {"rules": rules},
    }
    store.update_run(frozen=frozen)
    before = store.run_json.read_bytes()

    def unexpected(*args):
        pytest.fail("a frozen report must not fit a gate on any split")

    monkeypatch.setattr(report, "choose_gate", unexpected)
    monkeypatch.setattr(report, "best_threshold", unexpected)
    text = report.build_report(store)
    assert "### Cascade to" not in text
    assert "(3) cascade" not in text
    assert "(1) within" in text
    assert "## Held-out split (8 items)" in text
    assert store.run_json.read_bytes() == before


def test_freeze_without_calibration_overlap_leaves_cascade_unavailable(tmp_path, monkeypatch):
    store = synthetic_run(tmp_path)
    held_out = {
        row["item_id"] for row in store.read_jsonl(store.items_path) if row["split"] == "held_out"
    }
    path = store.verdicts_path("rules")
    store.write_jsonl(path, [r for r in store.read_jsonl(path) if r["item_id"] in held_out])

    def unexpected(*args):
        pytest.fail("only held-out rows overlap; no gate may be fitted")

    monkeypatch.setattr(report, "choose_gate", unexpected)
    text = report.build_report(store, freeze=True)
    assert store.read_run()["frozen"]["backends"]["rules"]["gate"] is None
    assert "### Cascade to" not in text
    assert "(3) cascade" not in text


@pytest.mark.parametrize("version", ["questions_v1", "questions_v2", "questions_v3"])
def test_existing_frozen_reports_keep_their_cuts_and_output(tmp_path, monkeypatch, version):
    store = synthetic_run(tmp_path, v1=version == "questions_v1")
    as_questions_version(store, version)
    monkeypatch.setattr(report, "utc_now_iso", lambda: "2026-01-01T00:00:00+00:00")
    expected = report.build_report(store, freeze=True)
    paths = [store.run_json, store.items_path, store.labels_path, store.verdicts_path("rules")]
    before = [path.read_bytes() for path in paths]

    def unexpected(*args):
        pytest.fail("stored historical cuts and gates must not be recomputed")

    monkeypatch.setattr(report, "best_threshold", unexpected)
    monkeypatch.setattr(report, "choose_gate", unexpected)
    assert report.build_report(store) == expected
    assert [path.read_bytes() for path in paths] == before


@pytest.mark.parametrize(
    ("probs", "labels", "expected"),
    [
        ([0.8, 0.9], [False, False], (1.0, 1.0)),
        ([0.1, 0.2], [True, True], (0.0, 1.0)),
        ([0.9, 0.9], [False, False], (1.0, 1.0)),
        ([0.1, 0.1], [True, True], (0.0, 1.0)),
        ([1.0, 1.0], [False, False], (1.01, 1.0)),
        ([0.0, 0.0], [True, True], (0.0, 1.0)),
        ([0.9, 0.9], [False, True], (0.5, 0.5)),
        ([0.1, 0.1], [False, True], (0.5, 0.5)),
        ([0.2, 0.6, 0.7, 0.9], [False, False, True, True], (0.65, 1.0)),
        ([], [], (0.5, 0.0)),
    ],
)
def test_best_threshold_includes_boundaries_and_keeps_ties(probs, labels, expected):
    assert metrics.best_threshold(probs, labels) == pytest.approx(expected)


def test_all_negative_boundary_survives_freeze_rounding(tmp_path):
    store = synthetic_run(tmp_path, n=1)
    path = store.verdicts_path("rules")
    rows = store.read_jsonl(path)
    rows[0]["answers"]["failed"]["probabilities"] = {"yes": 1.0, "no": 0.0}
    store.write_jsonl(path, rows)
    report.build_report(store, freeze=True)
    cut = store.read_run()["frozen"]["backends"]["rules"]["thresholds"]["failed"]
    assert cut > 1.0


@pytest.mark.parametrize("change", ["text", "decline", "missing_hash", "added", "removed"])
@pytest.mark.parametrize("preferred", ["a", "b", "same", "neither"])
def test_drafter_notes_exclude_stale_ratings_and_comparisons(change, preferred):
    rows: list[dict[str, Any]] = [
        {"drafter": name, "lesson": name.upper(), "latency_ms": 10.0, "cost_usd": 0.01}
        for name in ("a", "b")
    ]
    label = {
        "worth_remembering": False,
        "preferred": preferred,
        "drafts": {
            row["drafter"]: {
                "draft_hash": draft_hash(row["lesson"]),
                "declined": False,
                "grounded": True,
                "faithful": True,
                "generalizable": 1,
                "enforced_by": "none",
            }
            for row in rows
        },
    }
    stale_rows, stale_label = copy.deepcopy(rows), copy.deepcopy(label)
    if change == "text":
        stale_rows[0]["lesson"] = "Rewritten A"
    elif change == "decline":
        stale_rows[0]["lesson"] = None
    elif change == "missing_hash":
        stale_label["drafts"]["a"].pop("draft_hash")
    elif change == "added":
        stale_rows.append({"drafter": "c", "lesson": "C"})
    else:
        stale_label["drafts"]["removed"] = {"draft_hash": draft_hash("Removed")}
    text = "\n".join(
        report.drafter_notes(
            {"good": label, "stale": stale_label}, {"good": rows, "stale": stale_rows}
        )
    )
    assert "`preferred` is over the 1 item(s)" in text
    a = next(line for line in text.splitlines() if line.startswith("| a |"))
    cells = [cell.strip() for cell in a.split("|")[1:-1]]
    assert cells[1] == "2"  # Draft production/costs are not discarded with stale labels.
    assert cells[4] == ("2" if change in ("added", "removed") else "1")
    assert cells[-1] == "0.0200"
    if change == "decline":
        assert cells[3] == "1.00"  # Item-level worth_remembering remains valid.


def test_stale_only_comparison_has_no_drafter_acceptance_conclusion():
    label = {
        "preferred": "a",
        "drafts": {
            name: {"draft_hash": draft_hash("old"), "grounded": True, "faithful": True}
            for name in ("a", "b")
        },
    }
    text = "\n".join(
        report.drafter_notes(
            {"i": label},
            {"i": [{"drafter": name, "lesson": "new"} for name in ("a", "b")]},
        )
    )
    assert "Criterion 5" not in text


def test_next_user_linking_is_linear_and_keeps_session_provenance(monkeypatch):
    script = [
        ("user_msg", "First"),
        ("assistant_msg", "Work"),
        ("outcome", "completed"),
        *[("user_msg", "Background task completed. task_id: synthetic") for _ in range(30)],
        ("user_msg", "# Follow-up Instructions\nNo, fix it."),
        ("user_msg", "Interrupted"),
    ]
    rows = _rows("human", script) + _rows("agent", script)
    calls = []
    real = triggers.message_provenance

    def counted(text, caller):
        calls.append((text, caller))
        return real(text, caller)

    monkeypatch.setattr(triggers, "message_provenance", counted)
    turns = triggers.segment_turns(rows, {"agent": Caller("agent", "worker", "parent")})
    linking_calls = len(calls)
    for sid in ("human", "agent"):
        session = [turn for turn in turns if turn.session_id == sid]
        for index, turn in enumerate(session):
            later = next(
                (t.user_event for t in session[index + 1 :] if t.opener_provenance != "harness"),
                None,
            )
            assert turn.next_user is later
        assert session[0].closed_by == "outcome"
        assert session[0].next_provenance == "parent_agent"
        assert session[-1].next_user is None
    assert linking_calls <= len(turns)


def test_completed_resume_checks_expected_keys_only_once(tmp_path, monkeypatch):
    store = synthetic_run(tmp_path, n=20)
    original = pipeline.expected_verdicts(
        store.read_jsonl(store.items_path), store.drafts_by_item()
    )

    class CountedKeys(dict):
        visits = 0

        def __iter__(self):
            for key in super().__iter__():
                self.visits += 1
                yield key

    expected = CountedKeys(original)
    monkeypatch.setattr(pipeline, "expected_verdicts", lambda items, drafts: expected)

    def unexpected(*args):
        pytest.fail("completed resume must not read projections or make model calls")

    monkeypatch.setattr(store, "read_projection", unexpected)
    monkeypatch.setattr(RulesBackend, "decide", unexpected)
    assert pipeline.decide_items(
        store.cfg, store.run_id, ["rules"], store=store, record_events=False
    ).written == {"rules": 0}
    assert expected.visits == len(expected)


@pytest.mark.parametrize(
    "pairs",
    [
        [],
        [(0.9, 0.1), (0.5, 0.5), (0.1, 0.9)],
        [(float("nan"), 0.5), (float("inf"), 0.5), (0.5, float("nan"))],
    ],
)
def test_paired_discrimination_accepts_one_pass_input_without_changing_values(pairs):
    expected = (
        sum(1.0 if f > c else 0.5 if f == c else 0.0 for f, c in pairs) / len(pairs)
        if pairs
        else None
    )
    assert metrics.paired_discrimination(iter(pairs)) == expected


def test_project_pairs_are_lazy_and_preserve_order_and_missing_labels(tmp_path):
    store = synthetic_run(tmp_path, n=10)
    _, scored = report.load_scored(store)
    rows = list(report.per_item(scored["rules"]).values())
    rows[0].label["failed"] = None
    rows[1].verdict.row["answers"].pop("failed")
    valid = [
        r for r in rows if r.label.get("failed") is not None and r.verdict.p("failed") is not None
    ]
    expected = []
    for project in dict.fromkeys(row.item["project"] for row in valid):
        failed = [
            r.verdict.p("failed")
            for r in valid
            if r.item["project"] == project
            and r.label.get("failed") is True
            and r.verdict.p("failed") is not None
        ]
        clean = [
            r.verdict.p("failed")
            for r in valid
            if r.item["project"] == project
            and r.label.get("failed") is False
            and r.verdict.p("failed") is not None
        ]
        expected.extend((f, c) for f in failed for c in clean)
    pairs = report._pairs(rows)
    assert iter(pairs) is pairs
    assert list(pairs) == expected
