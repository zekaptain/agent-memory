"""Synthetic admission fitting, provenance, isolation and privacy regressions."""

from __future__ import annotations

import copy
import json
import math
from typing import Any

import pytest

from agent_memory import endpoints
from agent_memory.admission import AdmissionError, approver
from agent_memory.admission.embed import DEFAULT_EMBEDDER, make_embedder
from agent_memory.admission.examples import example_for, import_labels
from agent_memory.admission.fit import fit_approver
from agent_memory.admission.store import AdmissionStore, memory_split, write_jsonl
from agent_memory.config import Config
from agent_memory.endpoints import EgressError
from agent_memory.judge.drafter import draft_hash
from agent_memory.judge.questions import DEFAULT_THRESHOLDS
from tests.test_judge import structured_draft, synthetic_run

BACKEND = "llm:ref"


def _mock_embed(monkeypatch):
    calls = []

    def post(url, body, **kwargs):
        calls.append((url, body))
        return {"embeddings": [[1.0, 0.0, 0.0]]}, 1.0

    monkeypatch.setattr(endpoints, "post_json", post)
    return calls


def _freeze(store):
    store.update_run(
        frozen={
            "frozen_at": "2026-01-01T00:00:00+00:00",
            "backends": {BACKEND: {"thresholds": dict(DEFAULT_THRESHOLDS)}},
        }
    )


def _row(i: int, label: bool, split="calibration", *, item_id=None) -> dict[str, Any]:
    p = 0.95 if label else 0.1
    answers = {
        gate: {"probabilities": {"yes": p, "no": 1 - p}}
        for gate in ("grounded", "faithful", "rule_supported")
    }
    answers["generalizable"] = {"probabilities": {"0": 0.05, "1": 0.9, "2": 0.05}}
    answers["enforced"] = {"probabilities": {"none": 0.95, "tooling": 0.05}}
    iid = item_id or f"item-{i}"
    return example_for(
        {
            "rule": f"Lesson {i}.",
            "kind": "project_fact" if label else "process",
            "claims": [{"text": "A synthetic claim.", "cited": [1]}],
        },
        {"item_id": iid, "caller": "human"},
        {
            "run_id": "synthetic",
            "backend": BACKEND,
            "answers": answers,
            "thresholds": DEFAULT_THRESHOLDS,
            "projection_hash": f"projection-{i}",
            "draft_hash": f"draft-{i}",
        },
        [1.0, 0.0, 0.0] if label else [0.0, 1.0, 0.0],
        source=f"exp/synthetic/{iid}/drafter-{i}",
        label=label,
        split=split,
    )


def _fit(tmp_path, rows=None):
    cfg = Config(home=tmp_path / "production")
    store = AdmissionStore(cfg)
    rows = (
        rows
        if rows is not None
        else [_row(i, bool(i % 2), "calibration" if i < 30 else "held_out") for i in range(80)]
    )
    write_jsonl(store.examples_path, rows)
    model = fit_approver(cfg, backend=BACKEND, embedder=DEFAULT_EMBEDDER)
    return cfg, store, model, rows


def test_import_requires_frozen_v4_and_imports_only_exact_draft_labels(tmp_path, monkeypatch):
    source = synthetic_run(tmp_path)
    cfg = Config(home=tmp_path / "production")
    calls = _mock_embed(monkeypatch)
    with pytest.raises(AdmissionError, match="frozen"):
        import_labels(cfg, source.root, backend=BACKEND, embedder=DEFAULT_EMBEDDER)
    assert calls == []
    _freeze(source)
    assert import_labels(cfg, source.root, backend=BACKEND, embedder=DEFAULT_EMBEDDER) == 20
    store = AdmissionStore(cfg)
    rows = store.read_examples()
    assert len(rows) == 20
    assert sum(row["label"] for row in rows) == 3
    assert all(type(row["label"]) is bool for row in rows)
    assert all(set(row["gates"]) == {BACKEND} for row in rows)
    assert all(row["human_labels"]["rule_supported"] == row["rule_supported"] for row in rows)
    assert all(row["judge_thresholds"][BACKEND]["rule_supported"] == 0.5 for row in rows)
    assert {row["split"] for row in rows} == {"calibration", "held_out"}
    assert "SENTINEL_STATE_TEXT" not in store.examples_path.read_text()
    assert all("SENTINEL_STATE_TEXT" not in str(body) for _, body in calls)
    assert all(url.endswith("/api/embed") for url, _ in calls)
    assert import_labels(cfg, source.run_json, backend=BACKEND, embedder=DEFAULT_EMBEDDER) == 0
    assert len(calls) == 20  # Unchanged lessons reuse embeddings, including label corrections.
    source.update_run(questions_version="questions_v3")
    with pytest.raises(AdmissionError, match="questions_v4"):
        import_labels(cfg, source.root, backend=BACKEND, embedder=DEFAULT_EMBEDDER)


@pytest.mark.parametrize(
    ("target", "field", "value"),
    [
        ("label", "draft_hash", "stale"),
        ("label", "store_exact", None),
        ("label", "store_exact", 1),
        ("verdict", "draft_hash", "stale"),
        ("verdict", "projection_hash", "stale"),
        ("verdict", "questions_version", "questions_v3"),
        ("draft", "declined", True),
    ],
)
def test_import_skips_stale_unclear_declined_and_non_boolean_labels(
    tmp_path, monkeypatch, target, field, value
):
    source = synthetic_run(tmp_path, n=1)
    _freeze(source)
    calls = _mock_embed(monkeypatch)
    if target == "label":
        rows = source.read_jsonl(source.labels_path)
        rows[0]["worth_remembering"] = True
        rows[0]["drafts"]["ollama:m"][field] = value
        source.write_jsonl(source.labels_path, rows)
    else:
        path = source.drafts_path if target == "draft" else source.verdicts_path(BACKEND)
        rows = source.read_jsonl(path)
        rows[0][field] = value
        source.write_jsonl(path, rows)
    cfg = Config(home=tmp_path / "production")
    with pytest.raises(AdmissionError, match="no matching"):
        import_labels(cfg, source.root, backend=BACKEND, embedder=DEFAULT_EMBEDDER)
    assert calls == []
    assert not (cfg.home / "admission" / "examples.jsonl").exists()


def test_reimport_corrections_replace_labels_and_remove_obsolete_examples(tmp_path, monkeypatch):
    source = synthetic_run(tmp_path, n=1)
    _freeze(source)
    calls = _mock_embed(monkeypatch)
    cfg = Config(home=tmp_path / "production")
    import_labels(cfg, source.root, backend=BACKEND, embedder=DEFAULT_EMBEDDER)
    store = AdmissionStore(cfg)
    labels = source.read_jsonl(source.labels_path)
    labels[0]["drafts"]["ollama:m"]["store_exact"] = True
    source.write_jsonl(source.labels_path, labels)
    assert import_labels(cfg, source.root, backend=BACKEND, embedder=DEFAULT_EMBEDDER) == 1
    assert store.read_examples()[0]["label"] is True
    assert len(calls) == 1
    store.write_state(mode="auto", approver_version="old")
    labels[0]["drafts"]["ollama:m"]["store_exact"] = None
    source.write_jsonl(source.labels_path, labels)
    with pytest.raises(AdmissionError, match="no matching"):
        import_labels(cfg, source.root, backend=BACKEND, embedder=DEFAULT_EMBEDDER)
    assert store.read_examples() == []
    assert store.read_state()["mode"] == "shadow"


def test_import_arms_do_not_overwrite_other_backend_features(tmp_path, monkeypatch):
    source = synthetic_run(tmp_path, n=1)
    _freeze(source)
    source.update_run(
        frozen={
            "backends": {
                BACKEND: {"thresholds": DEFAULT_THRESHOLDS},
                "rules": {"thresholds": DEFAULT_THRESHOLDS},
            }
        }
    )
    _mock_embed(monkeypatch)
    cfg = Config(home=tmp_path / "production")
    import_labels(cfg, source.root, backend=BACKEND, embedder=DEFAULT_EMBEDDER)
    import_labels(cfg, source.root, backend="rules", embedder=DEFAULT_EMBEDDER)
    rows = AdmissionStore(cfg).read_examples()
    assert len(rows) == 2
    assert len(approver.select_examples(rows, BACKEND, DEFAULT_EMBEDDER)) == 1
    assert len(approver.select_examples(rows, "rules", DEFAULT_EMBEDDER)) == 1


def test_import_structured_fields_and_model_identity(tmp_path, monkeypatch):
    source = synthetic_run(tmp_path, n=1)
    _freeze(source)
    calls = _mock_embed(monkeypatch)
    draft = structured_draft(source.read_jsonl(source.items_path)[0]["item_id"]).to_dict()
    source.write_jsonl(source.drafts_path, [draft])
    labels = source.read_jsonl(source.labels_path)
    labels[0]["drafts"]["ollama:m"]["draft_hash"] = draft_hash(draft["lesson"])
    source.write_jsonl(source.labels_path, labels)
    verdicts = source.read_jsonl(source.verdicts_path(BACKEND))
    verdicts[0]["draft_hash"] = draft_hash(draft["lesson"])
    source.write_jsonl(source.verdicts_path(BACKEND), verdicts)
    cfg = Config(home=tmp_path / "production")
    with pytest.raises(AdmissionError, match="no frozen thresholds"):
        import_labels(cfg, source.root, backend="other", embedder=DEFAULT_EMBEDDER)
    assert import_labels(cfg, source.root, backend=BACKEND, embedder=DEFAULT_EMBEDDER) == 1
    example = AdmissionStore(cfg).read_examples()[0]
    assert example["rule"] == draft["rule"]
    assert example["kind"] == draft["kind"]
    assert example["claims"] == [claim["text"] for claim in draft["claims"]]
    assert all(claim["text"] in calls[0][1]["input"] for claim in draft["claims"])
    draft["rule"] = "A silently changed rule."
    source.write_jsonl(source.drafts_path, [draft])
    with pytest.raises(AdmissionError, match="no longer match"):
        import_labels(cfg, source.root, backend=BACKEND, embedder=DEFAULT_EMBEDDER)


def test_fit_deterministic_heldout_is_never_training_or_cut_data(tmp_path):
    cfg, store, model, rows = _fit(tmp_path)
    assert fit_approver(cfg, backend=BACKEND, embedder=DEFAULT_EMBEDDER) == model
    assert model["counts"] == {"calibration": 30, "held_out": 50, "positive_calibration": 15}
    assert model["optimizer"]["steps"] == 1200
    assert len(model["weights"]) == len(model["features"]) == len(model["means"])
    assert model["metrics"]["held_out"]["precision"] == 1
    assert model["metrics"]["held_out"]["scored_writes"] == 25
    assert approver.can_auto(model)
    assert not approver.can_auto(model, min_writes=26)
    serialized = store.approver_path.read_text()
    assert "A synthetic claim." not in serialized
    assert "Lesson 1." not in serialized
    for row in rows[30:]:
        row["label"] = not row["label"]
        row["kind"] = "held-out-only-kind"
        row["caller"] = "held-out-only-caller"
        row["embedding"] = [0, 0, 1]
        row["gates"][BACKEND] = {}
    write_jsonl(store.examples_path, rows)
    changed = fit_approver(cfg, backend=BACKEND, embedder=DEFAULT_EMBEDDER)
    for key in (
        "weights",
        "means",
        "scales",
        "intercept",
        "p_hi",
        "p_lo",
        "thresholds",
        "features",
        "calibration_hash",
        "calibration_ids",
    ):
        assert changed[key] == model[key], key
    assert "kind:held-out-only-kind" not in changed["features"]
    assert not approver.can_auto(changed)


def test_neighbor_features_exclude_self_siblings_and_all_heldout(tmp_path):
    first = _row(0, True, item_id="shared")
    sibling = _row(1, True, item_id="shared")
    sibling["group_id"] = "exp:another-run:shared"
    negative = _row(2, False)
    negative["embedding"] = [0.8, 0.6, 0]
    held = _row(3, True, "held_out")
    vote, similarity = approver.neighbor_features(first, [first, sibling, negative, held])
    assert vote == 0
    assert similarity == pytest.approx(0.8)
    assert approver.neighbor_features(first, [first, sibling, held]) == (0.5, 0.0)
    cal = [_row(i, i % 2 == 0) for i in range(1, 11)]
    for row in cal:
        row["embedding"] = [1, 0, 0]
    assert approver.neighbor_features(first, cal) == approver.neighbor_features(first, cal[::-1])
    # Eight, not ten, equally similar examples contribute.
    nearest = sorted(cal, key=lambda row: row["id"])[:8]
    assert approver.neighbor_features(first, cal)[0] == sum(row["label"] for row in nearest) / 8


def test_same_item_in_different_experiments_cannot_cross_splits():
    first = _row(0, True, item_id="shared")
    second = _row(1, True, "held_out", item_id="shared")
    second["group_id"] = "exp:another-run:shared"
    with pytest.raises(AdmissionError, match="cannot span"):
        approver.select_examples([first, second], BACKEND, DEFAULT_EMBEDDER)


@pytest.mark.parametrize("label", [True, False, None])
def test_production_feedback_replaces_experiment_training_influence(label):
    original = _row(0, True, item_id="shared")
    review = _row(1, False, item_id="shared")
    review.update(memory_id="reviewed-memory", source_type="review", label=label)
    selected = approver.select_examples([original, review], BACKEND, DEFAULT_EMBEDDER)
    assert [r["label"] for r in selected] == ([] if label is None else [label])
    assert all(r["split"] == memory_split("reviewed-memory") for r in selected)
    review["embedding"] = original["embedding"]
    assert approver.neighbor_features(original, [review]) == (0.5, 0.0)


def test_scoring_guardrails_and_stale_or_corrupt_models(tmp_path):
    _, store, model, _ = _fit(tmp_path)
    examples = store.read_examples()
    draft = _row(999, True, "production")
    draft["label"] = None
    assert approver.score(model, draft, examples)["action"] == "write"
    forced_high = approver.seal(
        {
            **model,
            "weights": [0.0] * len(model["weights"]),
            "intercept": 100,
            "p_hi": 0.8,
            "p_lo": 0.2,
        }
    )
    assert approver.score(forced_high, draft, examples)["action"] == "write"
    unsupported = copy.deepcopy(draft)
    unsupported["gates"][BACKEND]["rule_supported"] = 0
    assert approver.score(forced_high, unsupported, examples)["action"] == "discard"
    missing = copy.deepcopy(draft)
    missing["gates"][BACKEND].pop("rule_supported")
    missing["gates"][BACKEND]["grounded"] = 0.5  # Judge's other uncertainty cannot hide absence.
    assert approver.score(forced_high, missing, examples)["action"] == "review"
    distant = {**draft, "embedding": [0, 0, 1]}
    result = approver.score(forced_high, distant, examples)
    assert result["score"] > 0.99
    assert result["action"] == "review"
    assert result["max_similarity"] == 0
    assert "similarity" in result["reasons"][0]
    corrupt = copy.deepcopy(model)
    corrupt["weights"][0] += 1
    with pytest.raises(AdmissionError, match="hash mismatch"):
        approver.score(corrupt, draft, examples)
    assert not approver.can_auto(corrupt)
    stale = copy.deepcopy(examples)
    stale[0]["draft_hash"] = "changed"
    with pytest.raises(AdmissionError, match="examples changed"):
        approver.score(model, draft, stale)
    examples[0]["label"] = not examples[0]["label"]
    with pytest.raises(AdmissionError, match="examples changed"):
        approver.score(model, draft, examples)


def test_only_calibration_cuts_meet_precision_and_miss_budget(tmp_path):
    _, store, model, _ = _fit(tmp_path)
    rows = store.read_examples()
    calibration = [row for row in rows if row["split"] == "calibration"]
    results = [approver.score(model, row, rows) for row in calibration]
    assert model["metrics"]["calibration"]["precision"] >= 0.85
    assert model["metrics"]["calibration"]["miss"] <= model["frozen_judge_miss"]
    assert model["p_lo"] <= model["p_hi"]
    for cut in sorted({0.0, *(result["score"] for result in results)}):
        eligible = [
            row["label"]
            for row, result in zip(calibration, results, strict=True)
            if row["judge_action"] != "discard"
            and result["max_similarity"] >= model["similarity_floor"]
            and result["score"] >= cut
        ]
        if cut < model["p_hi"]:
            assert not eligible or sum(eligible) / len(eligible) < 0.85
    assert set(model["metrics"]["held_out"]) >= {"precision", "miss", "coverage", "brier", "auroc"}


def test_latest_review_overrides_without_duplicate_influence_and_keeps_split(tmp_path):
    cfg = Config(home=tmp_path / "production")
    store = AdmissionStore(cfg)
    first = {**_row(1, True), "memory_id": "memory-one", "source": "action-1", "id": "action-1"}
    final = {**first, "source": "action-2", "id": "action-2", "label": False, "split": "held_out"}
    write_jsonl(store.examples_path, [first, first, final, final])
    effective = store.read_examples()
    assert len(effective) == 1
    assert effective[0]["label"] is False
    assert effective[0]["split"] == memory_split("memory-one")
    assert effective[0]["group_id"] == "memory:memory-one"
    rows = [_row(i, bool(i % 2)) for i in range(10)]
    write_jsonl(store.examples_path, [*rows, first, first, final])
    model = fit_approver(cfg, backend=BACKEND, embedder=DEFAULT_EMBEDDER)
    write_jsonl(store.examples_path, [*rows, final])
    deduplicated = fit_approver(cfg, backend=BACKEND, embedder=DEFAULT_EMBEDDER)
    assert deduplicated == model
    unclear = {**final, "label": None, "source": "action-3", "id": "action-3"}
    write_jsonl(store.examples_path, [unclear], append=True)
    assert store.read_examples()[-1]["label"] is None


def test_latest_review_is_not_replaced_by_replayed_older_actions(tmp_path):
    store = AdmissionStore(Config(home=tmp_path / "production"))
    first = {
        **_row(1, True),
        "memory_id": "memory",
        "id": "action-old",
        "source": "action-old",
        "reviewed_at": "2026-01-01T00:00:00+00:00",
    }
    newer = {
        **first,
        "id": "action-new",
        "source": "action-new",
        "label": False,
        "reviewed_at": "2026-01-02T00:00:00+00:00",
    }
    write_jsonl(store.examples_path, [newer, first, newer, first])
    assert store.read_examples()[0]["label"] is False
    # Without timestamps, duplicate retries also keep the unique action insertion order.
    first.pop("reviewed_at")
    newer.pop("reviewed_at")
    write_jsonl(store.examples_path, [first, newer, first])
    assert store.read_examples()[0]["label"] is False


def test_private_files_atomic_append_and_day_store(tmp_path):
    cfg = Config(home=tmp_path / "production")
    store = AdmissionStore(cfg)
    assert store.read_state() == {"mode": "shadow", "watermark": None, "approver_version": None}
    assert store.read_approver() == {}
    store.write_state(watermark="2026-01-01T00:00:00Z")
    store.write_approver({"private": "data"})
    write_jsonl(store.examples_path, [_row(0, True)])
    store.examples_path.chmod(0o644)
    write_jsonl(store.examples_path, [_row(1, False)], append=True)
    store.examples_path.write_text(store.examples_path.read_text().rstrip("\n"))
    write_jsonl(store.examples_path, [_row(2, True)], append=True)
    assert len(store.read_examples()) == 3
    day = store.day("2026-01-01")
    day.create({"questions_version": "questions_v4"})
    day.write_projection("item", "PRIVATE PROJECTION")
    day.write_jsonl(day.drafts_path, [{"item_id": "item"}])
    day.update_run(done=True)
    for path in [cfg.home, store.root, *(store.root.rglob("*"))]:
        assert path.stat().st_mode & 0o777 == (0o700 if path.is_dir() else 0o600), path
    with pytest.raises(AdmissionError, match="YYYY-MM-DD"):
        store.day("../escape")
    with pytest.raises(AdmissionError, match="YYYY-MM-DD"):
        store.day("2026-02-30")
    store.examples_path.write_text('{"torn":')
    with pytest.raises(AdmissionError, match="invalid admission JSONL"):
        store.read_examples()
    with pytest.raises(AdmissionError, match="invalid admission JSONL"):
        write_jsonl(store.examples_path, [_row(2, True)], append=True)


def test_embedder_refuses_remote_and_never_includes_projection(monkeypatch):
    monkeypatch.delenv("AGENT_MEMORY_JUDGE_ALLOW_REMOTE", raising=False)
    monkeypatch.setattr(endpoints, "OLLAMA_HOST", "https://not-local.invalid")
    calls = _mock_embed(monkeypatch)
    with pytest.raises(EgressError, match="non-loopback"):
        make_embedder()
    assert calls == []
    monkeypatch.setattr(endpoints, "OLLAMA_HOST", "http://localhost:11434")
    embed = make_embedder()
    assert embed.name == DEFAULT_EMBEDDER
    assert embed.embed(
        {
            "rule": "Keep the lesson.",
            "kind": "process",
            "claims": [{"text": "Supported claim.", "cited": [1], "projection": "NOT SENT"}],
            "projection": "NOT SENT",
            "state": "NOT SENT",
        }
    ) == [1, 0, 0]
    assert "NOT SENT" not in json.dumps(calls)
    assert "Supported claim." in calls[0][1]["input"]


@pytest.mark.parametrize("vector", [[], [math.nan], [math.inf], ["text"], [True]])
def test_embedder_rejects_invalid_vectors(monkeypatch, vector):
    monkeypatch.setattr(endpoints, "OLLAMA_HOST", "http://localhost:11434")
    monkeypatch.setattr(
        endpoints, "post_json", lambda *args, **kwargs: ({"embeddings": [vector]}, 0)
    )
    with pytest.raises(AdmissionError, match="embedding"):
        make_embedder().embed({"rule": "A rule."})


def test_redacts_before_embedding_and_persistence(tmp_path, monkeypatch):
    source = synthetic_run(tmp_path, n=1)
    _freeze(source)
    cfg = Config(home=tmp_path / "production")
    cfg.home.mkdir(mode=0o700)
    secret = "private-ledger-bearer-value"
    cfg.token_path.write_text(secret)
    cfg.token_path.chmod(0o600)
    drafts = source.read_jsonl(source.drafts_path)
    drafts[0]["lesson"] = f"Never disclose {secret}."
    source.write_jsonl(source.drafts_path, drafts)
    labels = source.read_jsonl(source.labels_path)
    labels[0]["drafts"]["ollama:m"]["draft_hash"] = draft_hash(drafts[0]["lesson"])
    source.write_jsonl(source.labels_path, labels)
    verdicts = source.read_jsonl(source.verdicts_path(BACKEND))
    verdicts[0]["draft_hash"] = draft_hash(drafts[0]["lesson"])
    source.write_jsonl(source.verdicts_path(BACKEND), verdicts)
    calls = _mock_embed(monkeypatch)
    import_labels(cfg, source.root, backend=BACKEND, embedder=DEFAULT_EMBEDDER)
    assert secret not in json.dumps(calls)
    assert secret not in AdmissionStore(cfg).examples_path.read_text()


def test_no_labels_or_bad_inputs_disable_previously_enabled_auto(tmp_path):
    cfg, store, model, rows = _fit(tmp_path)
    store.write_state(mode="auto", approver_version=model["version"])
    write_jsonl(store.examples_path, [])
    with pytest.raises(AdmissionError, match="no human-labeled calibration"):
        fit_approver(cfg, backend=BACKEND, embedder=DEFAULT_EMBEDDER)
    assert store.read_state()["mode"] == "shadow"
    assert store.read_state()["approver_version"] is None
    assert not approver.can_auto({})
    write_jsonl(store.examples_path, rows)
    rows[0]["embedding"] = [1, 0]
    write_jsonl(store.examples_path, rows)
    with pytest.raises(AdmissionError, match="dimensions"):
        fit_approver(cfg, backend=BACKEND, embedder=DEFAULT_EMBEDDER)
    assert store.read_state()["mode"] == "shadow"
    for row in rows:
        row["embedding"] = [1, 0, 0]
    rows[0]["judge_thresholds"][BACKEND]["faithful"] = 0.8
    write_jsonl(store.examples_path, rows)
    with pytest.raises(AdmissionError, match="conflicting frozen"):
        fit_approver(cfg, backend=BACKEND, embedder=DEFAULT_EMBEDDER)


def test_cross_split_siblings_and_bad_json_fail_closed(tmp_path):
    first = _row(0, True, item_id="same")
    held = _row(1, True, "held_out", item_id="same")
    with pytest.raises(AdmissionError, match="cannot span"):
        _fit(tmp_path, [first, held])
    store = AdmissionStore(Config(home=tmp_path / "other"))
    store.state_path.write_text("[]")
    with pytest.raises(AdmissionError, match="expected an admission object"):
        store.read_state()
    target = tmp_path / "must-not-overwrite"
    target.write_text("unchanged")
    store.approver_path.symlink_to(target)
    with pytest.raises(AdmissionError, match="symbolic links"):
        store.write_approver({"changed": True})
    assert target.read_text() == "unchanged"


def test_no_heldout_writes_never_enables_auto_even_with_lower_requested_minimum(tmp_path):
    _, _, model, _ = _fit(tmp_path, [_row(i, True) for i in range(20)])
    assert model["metrics"]["held_out"]["scored_writes"] == 0
    assert not approver.can_auto(model, min_writes=1)
    _, _, model, _ = _fit(
        tmp_path, [_row(i, True, "calibration" if i < 20 else "held_out") for i in range(39)]
    )
    assert model["metrics"]["held_out"]["scored_writes"] == 19
    assert not approver.can_auto(model)
    assert approver.can_auto(model, min_writes=1)


@pytest.mark.parametrize(
    ("precision", "writes", "allowed"),
    [(0.8, 20, True), (0.799999, 20, False), (1.0, 19, False), (None, 0, False)],
)
def test_auto_precision_and_count_boundary(tmp_path, precision, writes, allowed):
    _, _, model, _ = _fit(tmp_path)
    model["metrics"]["held_out"].update(precision=precision, scored_writes=writes)
    assert approver.can_auto(approver.seal(model)) is allowed
