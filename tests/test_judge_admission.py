"""v4 exact-rule admission, independent human truth, and historical reproduction."""

from __future__ import annotations

import hashlib
import io
import json
import re
from dataclasses import replace

import pytest

from agent_memory.judge import pipeline, report
from agent_memory.judge.backends import http
from agent_memory.judge.backends.llm import LlmJudgeBackend
from agent_memory.judge.backends.rules import RulesBackend
from agent_memory.judge.drafter import Draft
from agent_memory.judge.label_controller import Action, LabelController
from agent_memory.judge.label_frontend import PlainFrontend
from agent_memory.judge.labeling import assemble, missing_steps, order_drafts, raw_answers
from agent_memory.judge.lesson import Claim, Lesson
from agent_memory.judge.questions import (
    DEFAULT_THRESHOLDS,
    QUESTIONS_VERSION,
    STAGE_A,
    WRITE_GATES,
    write_gates,
)
from agent_memory.judge.store import JudgeError
from agent_memory.judge.verdict import Verdict, derive_action, judge
from tests.test_judge import structured_draft, synthetic_run


@pytest.mark.parametrize(
    ("support", "cut", "expected"),
    [
        ("no", 0.5, "discard"),
        ("yes", 0.5, "write_lesson"),
        ("unclear", 0.0, "review"),
        ("missing", 0.0, "review"),
        ("invalid", 0.0, "review"),
    ],
)
def test_true_claims_do_not_justify_an_unsupported_rule(monkeypatch, support, cut, expected):
    draft = Draft.from_lesson(
        "i",
        Lesson(
            rule="Always disable database locking before migrations.",
            kind="process",
            claims=[
                Claim(text="The server was running.", cited=(1,)),
                Claim(text="The database was locked.", cited=(2,)),
            ],
        ),
        drafter="synthetic",
        model_snapshot="synthetic",
        latency_ms=0.0,
    ).to_dict()
    calls = []

    def post(url, body, **kwargs):
        prompt = body["messages"][-1]["content"]
        calls.append(prompt)
        ids = re.findall(r'- id "([^"]+)"', prompt)
        data = {}
        for key in ids:
            value = {
                "failed": "no",
                "incomplete": "no",
                "category": "none",
                "generalizable": "1",
                "enforced": "none",
            }.get(key, "yes")
            probability = 0.95
            if key == "rule_supported":
                if support == "missing":
                    continue
                value = "maybe" if support == "invalid" else "no" if support == "no" else "yes"
                probability = 0.5 if support == "unclear" else 0.95
            data[key] = {"answer": value, "probability": probability}
        return {"model": "synthetic", "message": {"content": json.dumps(data)}}, 20.0

    monkeypatch.setattr(http, "post_json", post)
    item = {
        "item_id": "i",
        "facts": {},
        "flags": [],
        "stratum": "T0",
        "projection_hash": "hash",
        "turn_event_id": "turn",
    }
    judged = judge(
        LlmJudgeBackend("synthetic", think=False),
        item,
        "[1] The server was running.\n[2] The database was locked.",
        draft,
        {"rule_supported": cut},
        run_id="synthetic",
    )
    verdict = judged.row
    assert len(calls) == 2
    assert "DRAFT LESSON" not in calls[0]
    assert draft["lesson"] in calls[1]
    assert "EXACT rule" in calls[1]
    assert "scope and conditions" in calls[1]
    assert "causal leap or overgeneralization" in calls[1]
    assert verdict["answers"]["grounded"]["probabilities"]["yes"] == 0.95
    assert verdict["answers"]["faithful"]["probabilities"]["yes"] == 0.95
    assert verdict["answers"]["failed"]["value"] == "no"  # Failure is not required.
    assert verdict["action"] == expected
    assert verdict["latency_ms"] == pytest.approx(40.0)
    entry = judged.decision_record(item, "2026-01-01T00:00:00+00:00")
    assert entry.payload is not None
    payload = entry.payload["judge"]
    assert payload["questions_version"] == "questions_v4"
    assert payload["claims"] == 2
    assert payload["answers"]["rule_supported"] == {
        k: verdict["answers"]["rule_supported"][k]
        for k in ("value", "probabilities", "confidence", "method")
    }
    assert "Always disable" not in entry.to_json()
    assert "The server was running" not in entry.to_json()
    if support in ("missing", "invalid"):
        assert judged.p("rule_supported") is None
        assert verdict["reasons"] == ["rule_supported: unanswered"]


def test_absent_gate_never_writes_even_with_zero_threshold_and_band():
    probabilities = dict.fromkeys(WRITE_GATES, 0.99)
    probabilities.pop("rule_supported")
    result = derive_action(probabilities, {"rule_supported": 0.0, "band": 0.0})
    assert result.action == "discard"
    assert result.reasons == ["rule_supported: no answer"]
    for version in ("questions_v2", "questions_v3"):
        assert derive_action(probabilities, gates=write_gates(version)).action == "write_lesson"


def stage_b(rows):
    return report.evaluate_stage_b("rules", "ollama:m", rows, DEFAULT_THRESHOLDS, WRITE_GATES)


def test_valuable_turn_with_bad_draft_is_not_a_correct_write(tmp_path):
    store = synthetic_run(tmp_path, n=4)
    _, scored = report.load_scored(store)
    row = list(scored["rules"].values())[3]
    assert row.label["worth_remembering"] is True
    row.draft["store_exact"] = False
    result = stage_b([row])
    assert result.actions["write_lesson"] == 1
    assert result.write_precision == 0.0
    assert result.admission_scored == result.writes_scored == 1
    assert result.miss_rate is None  # No human-approved exact draft to miss.
    # Historical scoring deliberately used the opportunity label instead.
    v3 = "questions_v3"
    historical = replace(row, verdict=Verdict({**row.verdict.row, "questions_version": v3}))
    old = report.evaluate_stage_b(
        "rules", "ollama:m", [historical], DEFAULT_THRESHOLDS, write_gates(v3)
    )
    assert old.write_precision == 1.0
    row.label["worth_remembering"] = False
    row.draft["store_exact"] = True
    assert stage_b([row]).write_precision == 1.0  # The labels have distinct subjects.


@pytest.mark.parametrize("missing", ["absent", "null", "unknown", "hash", "current_draft"])
def test_missing_or_stale_admission_is_unscored_not_correct_or_negative(tmp_path, missing):
    store = synthetic_run(tmp_path, n=4)
    labels = store.read_jsonl(store.labels_path)
    entry = labels[3]["drafts"]["ollama:m"]
    if missing == "absent":
        entry.pop("store_exact")
    elif missing == "null":
        entry["store_exact"] = None
    elif missing == "unknown":
        entry["store_exact"] = None
        entry["unknown"] = ["store_exact"]
    elif missing == "hash":
        entry["draft_hash"] = "stale"
    else:
        drafts = store.read_jsonl(store.drafts_path)
        drafts[3]["lesson"] = "A different exact rule."
        store.write_jsonl(store.drafts_path, drafts)
    store.write_jsonl(store.labels_path, labels)
    _, scored = report.load_scored(store)
    row = list(scored["rules"].values())[3]
    result = stage_b([row])
    assert result.actions["write_lesson"] == 1
    assert result.write_precision is result.miss_rate is None
    assert result.admission_scored == result.writes_scored == 0
    assert result.unscored == 1
    assert report.should_write(row.label, row.draft) is None
    text = report.build_report(store)
    assert "Write decisions vs `store_exact`" in text
    assert "admission scored | writes scored | unscored" in text


def test_precision_and_misses_exclude_unknowns_but_costs_do_not(tmp_path):
    store = synthetic_run(tmp_path, n=4)
    _, scored = report.load_scored(store)
    rows = list(scored["rules"].values())
    for row in rows:
        row.draft["store_exact"] = None
    rows[0].draft["store_exact"] = True  # Discard: a missed approved draft.
    rows[1].draft["store_exact"] = False  # Write: an explicitly bad draft.
    result = stage_b(rows)
    assert result.write_precision == 0.0
    assert result.miss_rate == 1.0
    assert result.admission_scored == 2
    assert result.writes_scored == 1
    assert result.unscored == 2


def test_rule_thresholds_fit_calibration_only_and_remain_frozen(tmp_path, monkeypatch):
    store = synthetic_run(tmp_path)
    labels = store.read_jsonl(store.labels_path)
    verdicts = store.read_jsonl(store.verdicts_path("rules"))
    for i, (label, verdict) in enumerate(zip(labels, verdicts, strict=True)):
        label["drafts"]["ollama:m"]["rule_supported"] = (
            False if i == 0 else True if i == 1 else None
        )
        p = 0.6 if i == 0 else 0.7 if i == 1 else 0.99
        verdict["answers"]["rule_supported"]["probabilities"] = {"yes": p, "no": 1 - p}
        if i >= 12:
            label["drafts"]["ollama:m"]["rule_supported"] = False
    store.write_jsonl(store.labels_path, labels)
    store.write_jsonl(store.verdicts_path("rules"), verdicts)
    text = report.build_report(store)
    assert "Held-out split (8 items)" not in text
    report.build_report(store, freeze=True)
    frozen_bytes = store.run_json.read_bytes()
    assert store.read_run()["frozen"]["backends"]["rules"]["thresholds"]["rule_supported"] == 0.65
    for label in labels[12:]:
        label["drafts"]["ollama:m"]["rule_supported"] = True
        label["drafts"]["ollama:m"]["store_exact"] = None
    store.write_jsonl(store.labels_path, labels)

    def never(*args, **kwargs):
        pytest.fail("frozen report must not refit")

    monkeypatch.setattr(report, "best_threshold", never)
    monkeypatch.setattr(report, "choose_gate", never)
    text = report.build_report(store)
    assert "Held-out split (8 items)" in text
    assert "rule_supported ≥ 0.65" in text
    assert store.run_json.read_bytes() == frozen_bytes


def test_multi_draft_costs_include_rule_question_and_charge_stage_a_once(tmp_path, monkeypatch):
    store = synthetic_run(tmp_path, n=1)
    store.verdicts_path("rules").unlink()
    item = store.read_jsonl(store.items_path)[0]
    drafts = [structured_draft(item["item_id"], drafter=name).to_dict() for name in ("a", "b")]
    store.write_jsonl(store.drafts_path, drafts)
    real = RulesBackend.decide

    def paid(self, inp, questions):
        return {
            key: replace(a, latency_ms=10.0, cost_usd=0.01)
            for key, a in real(self, inp, questions).items()
        }

    monkeypatch.setattr(RulesBackend, "decide", paid)
    pipeline.decide_items(store.cfg, store.run_id, ["rules"], store=store, record_events=False)
    rows = store.read_jsonl(store.verdicts_path("rules"))
    assert len(rows) == 2
    assert [r["stage_a_reused"] for r in rows] == [False, True]
    a_cost = len(STAGE_A) * 0.01
    b_cost = 6 * 0.01  # One grounded/faithful pair plus four whole-draft questions.
    assert sum(r["cost_usd"] for r in rows) == pytest.approx(a_cost + 2 * b_cost)
    assert all(r["answers"]["rule_supported"]["cost_usd"] == 0.01 for r in rows)
    assert all(r["answers"]["grounded"]["cost_usd"] == 0.0 for r in rows)
    scored = {(r["item_id"], r["drafter"]): report.Scored(item, {}, Verdict(r)) for r in rows}
    unique = list(report.per_item(scored).values())
    a = report.evaluate_stage_a("rules", unique, DEFAULT_THRESHOLDS, category_on_failed=True)
    b = stage_b(list(scored.values()))
    assert a.total_cost_usd == pytest.approx(a_cost + 2 * b_cost)
    assert b.total_cost_usd == pytest.approx(2 * b_cost)
    assert b.admission_scored == 0
    assert b.unscored == 2


@pytest.mark.parametrize(
    ("version", "digest"),
    [
        ("questions_v1", "79ebc878d11c7e80e9b8a0546d91a8c20b50fa509de4b85bf857b3c3a14c1534"),
        ("questions_v2", "907da8c1cf2eec6562d23e62c3360e04ba6a7b7ca500a7eeee5524c0e3284733"),
        ("questions_v3", "a1edfaf6aaf334fe31b95e1b8f17cb2c83b18f1adb43568e83a3c6220e641fec"),
    ],
)
def test_historical_report_matches_pre_v4_bytes(tmp_path, monkeypatch, version, digest):
    """Hashes captured from the actual pre-v4 working tree using these synthetic fixtures."""
    store = synthetic_run(tmp_path, v1=version == "questions_v1")
    store.update_run(questions_version=version)
    labels = store.read_jsonl(store.labels_path)
    for label in labels:
        if "drafts" in label:
            label["labels_version"] = "labels_v2"
            for entry in label["drafts"].values():
                entry.pop("rule_supported")
                entry.pop("store_exact")
    store.write_jsonl(store.labels_path, labels)
    for name in ("rules", "llm:ref"):
        path = store.verdicts_path(name)
        rows = store.read_jsonl(path)
        for row in rows:
            row["questions_version"] = version
            row["answers"].pop("rule_supported", None)
        store.write_jsonl(path, rows)
    monkeypatch.setattr(report, "utc_now_iso", lambda: "2026-01-01T00:00:00+00:00")
    text = report.build_report(store, freeze=True)
    assert hashlib.sha256(text.encode()).hexdigest() == digest
    paths = [
        store.run_json,
        store.labels_path,
        store.items_path,
        store.verdicts_path("rules"),
        store.verdicts_path("llm:ref"),
    ]
    before = [p.read_bytes() for p in paths]
    assert report.build_report(store) == text
    for command in (
        lambda: pipeline.draft_items(store.cfg, store.run_id, "ollama:synthetic"),
        lambda: pipeline.decide_items(store.cfg, store.run_id, ["rules"]),
        lambda: LabelController(store),
        lambda: LabelController(store, review=True),
    ):
        with pytest.raises(JudgeError, match=version):
            command()
    assert [p.read_bytes() for p in paths] == before


def test_unknown_new_labels_resume_without_backfill_and_edit_with_history(tmp_path):
    store = synthetic_run(tmp_path, n=1)
    store.labels_path.unlink()
    controller = LabelController(store)
    for value in ("n", "y", "1", "y", "y"):
        controller.apply(Action("answer", value))
    assert controller.step == "A.rule_supported"
    screen = controller.screen()
    assert screen.subject == "Draft A"
    assert screen.drafts[0].lesson == "Lesson 0."
    assert screen.drafts[0].focus == ""
    assert "scope and conditions" in screen.question
    # Exercise the plain interface rather than bypassing its numbered options.
    output = io.StringIO()
    plain = PlainFrontend(io.StringIO("3\n"), output)
    controller.apply(plain.exchange(screen))
    assert "Draft A" in output.getvalue()
    assert "Lesson 0." in output.getvalue()
    controller.apply(Action("back"))
    assert controller.screen().selected == "u"
    controller.apply(Action("answer", "u"))
    controller.apply(Action("answer", "1"))
    controller.apply(Action("answer", "1"))
    assert controller.step == "A.store_exact"
    assert controller.screen().question == "Would you store this exact draft?"
    assert controller.screen().subject == "Draft A"
    assert controller.screen().drafts[0].lesson == "Lesson 0."
    controller.apply(Action("answer", "u"))
    controller.apply(Action("answer", "continue"))
    assert "Draft A: Store this exact draft: Unclear" in controller.screen().evidence
    controller.apply(Action("answer", "save"))
    saved = store.read_jsonl(store.labels_path)[0]
    assert saved["labels_version"] == "labels_v4"
    entry = saved["drafts"]["ollama:m"]
    assert entry["rule_supported"] is entry["store_exact"] is None
    assert entry["unknown"] == ["rule_supported", "store_exact"]
    assert LabelController(store).done
    # A backfill adds only missing questions, not an already-answered Unclear.
    entry.pop("rule_supported")
    entry["unknown"].remove("rule_supported")
    store.write_jsonl(store.labels_path, [saved])
    backfill = LabelController(store)
    assert backfill.plan == ["A.rule_supported"]
    backfill.apply(Action("answer", "n"))
    backfill.apply(Action("answer", "save"))
    review = LabelController(store, item_ids=[saved["item_id"]])
    while review.step != "summary":
        value = "continue" if review.step == "note" else review.screen().selected
        if review.step == "A.rule_supported":
            value = "y"
        review.apply(Action("answer", value))
    review.apply(Action("answer", "save"))
    entry = store.read_jsonl(store.labels_path)[0]
    assert entry["history"][-1]["changed"]["drafts.ollama:m.rule_supported"] is False
    assert entry["drafts"]["ollama:m"]["store_exact"] is None


def test_unknown_label_hash_invalidation_and_mid_label_draft_change(tmp_path):
    store = synthetic_run(tmp_path, n=1)
    rows = store.read_jsonl(store.drafts_path)
    views = order_drafts(rows[0]["item_id"], rows, 1)
    label = assemble({"A.rule_supported": "u", "A.store_exact": "u"}, views, False)
    assert raw_answers(label, views)["A.store_exact"] == "u"
    changed = order_drafts(rows[0]["item_id"], [{**rows[0], "lesson": "Changed."}], 1)
    assert "A.store_exact" in missing_steps(label, changed, False)
    controller = LabelController(store, item_ids=[rows[0]["item_id"]])
    while controller.step != "summary":
        controller.apply(
            Action(
                "answer", "continue" if controller.step == "note" else controller.screen().selected
            )
        )
    before = store.labels_path.read_bytes()
    rows[0]["lesson"] = "Changed while open."
    store.write_jsonl(store.drafts_path, rows)
    with pytest.raises(JudgeError, match="Drafts changed during labeling"):
        controller.apply(Action("answer", "save"))
    assert store.labels_path.read_bytes() == before


def test_drafter_unknown_ratings_have_independent_denominators(tmp_path):
    store = synthetic_run(tmp_path, n=2)
    labels = {r["item_id"]: r for r in store.read_jsonl(store.labels_path)}
    for label in labels.values():
        entry = label["drafts"]["ollama:m"]
        entry["rule_supported"] = entry["store_exact"] = entry["faithful"] = None
    text = "\n".join(
        report.drafter_notes(labels, store.read_drafts(), questions_version=QUESTIONS_VERSION)
    )
    assert text.count("n/a (0 scored)") == 3
    assert "unknown" not in text


@pytest.mark.parametrize("change", ["text", "added", "removed"])
def test_hash_backfill_reasks_comparison_instead_of_reendorsing_old_preference(change):
    rows = [{"item_id": "i", "drafter": name, "lesson": name.upper()} for name in ("a", "b", "c")]
    views = order_drafts("i", rows, 1)
    label = assemble({"preferred": "same", "A.store_exact": "y"}, views, False)
    assert raw_answers(label, views)["preferred"] == "same"
    if change == "text":
        rows[0]["lesson"] = "A different exact rule."
    elif change == "added":
        rows.append({"item_id": "i", "drafter": "d", "lesson": "D"})
    else:
        rows.pop()
    current = order_drafts("i", rows, 1)
    assert "preferred" not in raw_answers(label, current)
    assert "preferred" in missing_steps(label, current, False)
