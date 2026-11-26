"""Local aggregate report for a usefulness experiment run (MEM-27).

Builds a deterministic, privacy-preserving report (``report.md`` and
``report.json``) strictly from the run's structured files:

- ``task-manifest.json`` (metadata, split, families, strata, excluded reason codes)
- ``registration.json`` (required for held-out; optional for pre-freeze calibration)
- ``<experiment_home>/memory-set/manifest.json`` (memory counts & validity crossings)
- ``schedule.json`` and ``schedule-events.jsonl`` (optional pair schedule & progress)
- ``trials/<trial_id>/record.json`` and ``interrupted.json`` (structured counts,
  status, model, tokens, wall time, recall/citation accounting; never replies,
  diffs, tool arguments, failure details, or paths)
- ``grades-final.json`` / ``grades.jsonl`` (structured grade booleans & categories;
  never free-text notes)
- ``attributions.jsonl`` (structured post-grading memory-use booleans; never notes)
- ``deviations.jsonl`` (dated run deviations)

Never reads task request/behavior files, hidden tests, check outputs, transcripts,
or trial ledgers, and never writes current wall-clock timestamps into the output
so building twice from the same files yields identical bytes.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config import Config
from .grading import (
    GRADE_FAILURE_CATEGORIES,
    latest_attributions,
    latest_grades,
    verify_finalized_grades,
)
from .memory_set import memory_set_manifest_path
from .registration import load_registration
from .runner import FAILURE_CATEGORIES, VALID_ARMS
from .schedule import (
    VALID_SPLIT_FILTERS,
    SplitFilter,
    inspect_schedule_progress,
)
from .stats import (
    BinaryObservation,
    BinaryPair,
    ClusterSummary,
    DistributionSummary,
    EstimateWithCI,
    Interval,
    NumericPair,
    PairedNumericSummary,
    PairedSuccessSummary,
    RateSummary,
    clustered_rate_summary,
    distribution_summary,
    paired_numeric_summary,
    paired_success_summary,
    wilson_interval,
)
from .store import (
    UsefulnessError,
    UsefulnessRunStore,
    read_json_file,
    sha256_bytes,
    sha256_file,
)

REPORT_SCHEMA_VERSION = 1
DEFAULT_BOOTSTRAP_SEED = 1
DEFAULT_BOOTSTRAP_RESAMPLES = 2_000
DEFAULT_CONFIDENCE = 0.95
DEFAULT_TAIL_QUANTILES: tuple[float, ...] = (0.90, 0.95)

NON_ORDINARY_GRADE_FAILURE_CATEGORIES: tuple[str, ...] = tuple(
    cat for cat in GRADE_FAILURE_CATEGORIES if cat != "ordinary_wrong_answer"
)

_ARM_SUFFIXES: tuple[tuple[str, str], ...] = (
    ("__no_memory", "no_memory"),
    ("__memory", "memory"),
    ("-no_memory", "no_memory"),
    ("-memory", "memory"),
    ("_no_memory", "no_memory"),
    ("_memory", "memory"),
)


@dataclass(frozen=True)
class ReportResult:
    """Built deterministic local aggregate report for one usefulness run."""

    report: dict[str, Any]
    markdown: str
    json_text: str
    markdown_sha256: str
    json_sha256: str
    report_path: Path | None
    report_json_path: Path | None


@dataclass(frozen=True)
class _TrialEntry:
    """Safe, structured subset of one trial (no replies, diffs, notes, or transcripts)."""

    trial_id: str
    task_id: str
    task_family: str
    split: str
    strata: tuple[str, ...]
    arm: str
    model: str
    reasoning_effort: str | None
    model_key: str
    status: str
    execution_failure_category: str | None
    has_record: bool
    record_sha256: str | None
    registration_sha256: str | None
    pair_id: str | None
    attempt: int | None
    # Structured numeric metrics (None when unavailable)
    total_tokens: float | None
    input_tokens: float | None
    output_tokens: float | None
    thinking_tokens: float | None
    factory_credits: float | None
    wall_total_s: float | None
    wall_clone_s: float | None
    wall_setup_s: float | None
    wall_harness_s: float | None
    wall_teardown_s: float | None
    # Structured recall & citation accounting counts (memory arm only)
    has_recall_accounting: bool
    made_no_recall: bool | None
    total_recalls: int
    returned_count: int
    miss_count: int
    degraded_count: int
    missing_record_count: int
    retrieval_latency_s: float | None
    citations_made: int
    cite_calls_in_transcript: int
    degraded_cite_calls: int
    returned_memories_total: int
    returned_memories_cited_count: int
    placeholder_count: int
    bound_placeholder_count: int
    unbound_placeholder_count: int
    bound_reply_count: int
    single_citation_reply_count: int
    multi_citation_reply_count: int
    # Structured blind grade fields (never note)
    graded: bool
    met_checks: bool | None
    grade_failure_category: str | None
    confident_wrong_reply: bool | None
    repeated_earlier_error: bool | None
    arm_leaked: bool | None
    # Structured post-grading attribution fields (never note)
    attributed: bool
    harmful_memory_use: bool | None
    attempted_invalid_memory_use: bool | None
    confident_wrong_linked_to_memory: bool | None


@dataclass(frozen=True)
class _ResolvedPair:
    """One resolved (model, pair) joining the final attempt's memory and no-memory trials."""

    pair_key: str
    task_id: str
    task_family: str
    split: str
    strata: tuple[str, ...]
    model_key: str
    memory: _TrialEntry | None
    no_memory: _TrialEntry | None
    # Superseded earlier attempts for this pair (when retried under RetryRule)
    superseded_memory: tuple[_TrialEntry, ...]
    superseded_no_memory: tuple[_TrialEntry, ...]


# ---------------------------------------------------------------------------
# Deterministic seed derivation & number coercion helpers
# ---------------------------------------------------------------------------


def _derive_seed(base_seed: int, tag: str) -> int:
    digest = hashlib.sha256(f"{base_seed}:{tag}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _opt_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        out = float(value)
        return out if math.isfinite(out) else None
    return None


def _opt_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def _opt_bool(value: Any) -> bool | None:
    return value if type(value) is bool else None


def _opt_str(value: Any) -> str | None:
    return value if isinstance(value, str) and value.strip() else None


def _quantile_label(q: float) -> str:
    pct = round(q * 100.0)
    if abs(q * 100.0 - pct) < 1e-9:
        return f"p{pct:02d}"
    return f"q{q:.4f}"


# ---------------------------------------------------------------------------
# Stats dataclass serialization (deterministic JSON-safe dicts)
# ---------------------------------------------------------------------------


def _interval_to_dict(interval: Interval | None) -> dict[str, Any] | None:
    if interval is None:
        return None
    return {
        "low": interval.low,
        "high": interval.high,
        "confidence": interval.confidence,
    }


def _estimate_to_dict(est: EstimateWithCI) -> dict[str, Any]:
    return {
        "estimate": est.estimate,
        "interval": _interval_to_dict(est.interval),
        "n_resamples": est.n_resamples,
    }


def _cluster_to_dict(cluster: ClusterSummary) -> dict[str, Any]:
    return {
        "n_observations": cluster.n_observations,
        "n_families": cluster.n_families,
        "icc": cluster.icc,
        "design_effect": cluster.design_effect,
        "effective_sample_size": cluster.effective_sample_size,
    }


def _paired_success_to_dict(
    summary: PairedSuccessSummary,
    *,
    exploratory: bool,
    practical_benefit_threshold: float | None = None,
    note: str | None = None,
) -> dict[str, Any]:
    diff_est = summary.difference.estimate
    diff_ci = summary.difference.interval
    exceeds_point: bool | None = None
    exceeds_lower_ci: bool | None = None
    if practical_benefit_threshold is not None and diff_est is not None:
        exceeds_point = diff_est >= practical_benefit_threshold
        if diff_ci is not None:
            exceeds_lower_ci = diff_ci.low >= practical_benefit_threshold
    out: dict[str, Any] = {
        "exploratory": exploratory,
        "slice_label": "exploratory" if exploratory else "registered",
        "n_total_pairs": summary.n_total_pairs,
        "n_pairs": summary.n_pairs,
        "n_missing_pairs": summary.n_missing_pairs,
        "n_missing_memory": summary.n_missing_memory,
        "n_missing_no_memory": summary.n_missing_no_memory,
        "memory_successes": summary.memory_successes,
        "no_memory_successes": summary.no_memory_successes,
        "memory_rate": summary.memory_rate,
        "no_memory_rate": summary.no_memory_rate,
        "difference": _estimate_to_dict(summary.difference),
        "practical_benefit_threshold": practical_benefit_threshold,
        "exceeds_practical_benefit_threshold": exceeds_point,
        "ci_exceeds_practical_benefit_threshold": exceeds_lower_ci,
        "clustering": _cluster_to_dict(summary.clustering),
        "exact_mcnemar": {
            "both_pass": summary.exact_mcnemar.both_pass,
            "memory_only": summary.exact_mcnemar.memory_only,
            "no_memory_only": summary.exact_mcnemar.no_memory_only,
            "both_fail": summary.exact_mcnemar.both_fail,
            "n_pairs": summary.exact_mcnemar.n_pairs,
            "n_missing_pairs": summary.exact_mcnemar.n_missing_pairs,
            "discordant": summary.exact_mcnemar.discordant,
            "p_value": summary.exact_mcnemar.p_value,
            "p_value_memory_better": summary.exact_mcnemar.p_value_memory_better,
        },
        "exact_family_sign": {
            "families_memory_better": summary.exact_family_sign.families_memory_better,
            "families_no_memory_better": summary.exact_family_sign.families_no_memory_better,
            "families_tied": summary.exact_family_sign.families_tied,
            "n_families": summary.exact_family_sign.n_families,
            "nonzero_families": summary.exact_family_sign.nonzero_families,
            "p_value": summary.exact_family_sign.p_value,
            "p_value_memory_better": summary.exact_family_sign.p_value_memory_better,
        },
        "seed": summary.seed,
        "confidence": summary.confidence,
    }
    if note is not None:
        out["note"] = note
    return out


def _distribution_to_dict(summary: DistributionSummary) -> dict[str, Any]:
    return {
        "n_total": summary.n_total,
        "n_observed": summary.n_observed,
        "n_missing": summary.n_missing,
        "n_families": summary.n_families,
        "mean": _estimate_to_dict(summary.mean),
        "median": _estimate_to_dict(summary.median),
        "quantiles": {
            _quantile_label(q): _estimate_to_dict(est)
            for q, est in sorted(summary.quantiles.items())
        },
    }


def _paired_numeric_to_dict(summary: PairedNumericSummary) -> dict[str, Any]:
    return {
        "n_total_pairs": summary.n_total_pairs,
        "n_pairs": summary.n_pairs,
        "n_missing_pairs": summary.n_missing_pairs,
        "n_missing_memory": summary.n_missing_memory,
        "n_missing_no_memory": summary.n_missing_no_memory,
        "clustering": _cluster_to_dict(summary.clustering),
        "memory_arm": _distribution_to_dict(summary.memory_arm),
        "no_memory_arm": _distribution_to_dict(summary.no_memory_arm),
        "difference_mean": _estimate_to_dict(summary.difference_mean),
        "difference_median": _estimate_to_dict(summary.difference_median),
        "difference_quantiles": {
            _quantile_label(q): _estimate_to_dict(est)
            for q, est in sorted(summary.difference_quantiles.items())
        },
        "ratio_of_means": _estimate_to_dict(summary.ratio_of_means),
        "n_ratio_pairs": summary.n_ratio_pairs,
        "n_excluded_ratio_pairs": summary.n_excluded_ratio_pairs,
        "ratio_median": _estimate_to_dict(summary.ratio_median),
        "ratio_quantiles": {
            _quantile_label(q): _estimate_to_dict(est)
            for q, est in sorted(summary.ratio_quantiles.items())
        },
        "seed": summary.seed,
        "confidence": summary.confidence,
    }


def _rate_to_dict(summary: RateSummary, *, denominator_label: str) -> dict[str, Any]:
    return {
        "numerator": summary.numerator,
        "denominator": summary.denominator,
        "denominator_label": denominator_label,
        "n_missing": summary.n_missing,
        "rate": _estimate_to_dict(summary.rate),
        "wilson": _interval_to_dict(summary.wilson),
        "clustering": _cluster_to_dict(summary.clustering),
        "seed": summary.seed,
        "confidence": summary.confidence,
    }


def _count_rate_to_dict(
    numerator: int,
    denominator: int,
    *,
    denominator_label: str,
    confidence: float,
) -> dict[str, Any]:
    return {
        "numerator": numerator,
        "denominator": denominator,
        "denominator_label": denominator_label,
        "rate": (numerator / denominator) if denominator > 0 else None,
        "wilson": _interval_to_dict(wilson_interval(numerator, denominator, confidence=confidence)),
    }


# ---------------------------------------------------------------------------
# Loading structured run files (never private content)
# ---------------------------------------------------------------------------


def _parse_scheduled_trial_id(
    trial_id: str,
    arm: str,
    schedule_pair_ids: set[str],
) -> tuple[str | None, int | None]:
    suffix = f"__{arm}"
    if not trial_id.endswith(suffix):
        return (None, None)
    stem = trial_id[: -len(suffix)]
    pair_id, sep, att_str = stem.rpartition("__a")
    if not sep or not pair_id or not att_str.isdigit():
        return (None, None)
    attempt = int(att_str)
    if attempt < 1:
        return (None, None)
    if schedule_pair_ids and pair_id in schedule_pair_ids:
        return (pair_id, attempt)
    # Even without schedule.json, recognize canonical "<task>__r<k>__a<n>__<arm>".
    _, r_sep, r_part = pair_id.rpartition("__r")
    if r_sep and r_part.isdigit():
        return (pair_id, attempt)
    return (None, None)


def _derive_unscheduled_stem(trial_id: str, arm: str) -> str | None:
    for suffix, suffix_arm in _ARM_SUFFIXES:
        if suffix_arm == arm and trial_id.endswith(suffix):
            stem = trial_id[: -len(suffix)]
            if stem:
                return stem
    return None


def _model_keys_from_entries(
    raw_models: Sequence[tuple[str, str | None]],
) -> dict[tuple[str, str | None], str]:
    efforts_by_model: dict[str, set[str | None]] = defaultdict(set)
    for model, effort in raw_models:
        efforts_by_model[model].add(effort)
    mapping: dict[tuple[str, str | None], str] = {}
    for model, efforts in efforts_by_model.items():
        if len(efforts) <= 1:
            for effort in efforts:
                mapping[(model, effort)] = model
        else:
            for effort in efforts:
                label = effort if effort is not None else "default"
                mapping[(model, effort)] = f"{model} (reasoning_effort={label})"
    return mapping


def _load_trials_from_disk(
    store: UsefulnessRunStore,
    *,
    task_manifest: Mapping[str, Any],
    schedule_pair_ids: set[str],
    grades_map: Mapping[str, Mapping[str, Any]],
    attributions_map: Mapping[str, Mapping[str, Any]],
    grades_final_sha256: str | None,
) -> tuple[list[_TrialEntry], int]:
    """Read structured trial metadata from ``trials/`` without touching private text."""
    tasks_map = task_manifest.get("tasks")
    if not isinstance(tasks_map, Mapping) or not tasks_map:
        raise UsefulnessError("task-manifest.json has no tasks")

    if not store.trials_dir.is_dir():
        return [], 0

    raw_rows: list[dict[str, Any]] = []
    interrupted_without_record = 0

    for trial_dir in sorted(p for p in store.trials_dir.iterdir() if p.is_dir()):
        record_path = trial_dir / "record.json"
        marker_path = trial_dir / "interrupted.json"

        if record_path.is_file():
            rec = read_json_file(record_path)
            rec_sha = sha256_file(record_path)
            trial_id = _opt_str(rec.get("trial_id"))
            task_id = _opt_str(rec.get("task_id"))
            arm = rec.get("arm")
            if trial_id is None or task_id is None or arm not in VALID_ARMS:
                raise UsefulnessError(f"malformed trial record at {record_path}")
            if task_id not in tasks_map:
                raise UsefulnessError(
                    f"trial {trial_id!r} references task {task_id!r} not in task-manifest.json"
                )
            t_meta = tasks_map[task_id]
            rec_task_sha = _opt_str(rec.get("task_sha256"))
            if rec_task_sha is not None and rec_task_sha != t_meta.get("task_sha256"):
                raise UsefulnessError(
                    f"trial {trial_id!r} task_sha256 does not match task-manifest.json"
                )
            model = _opt_str(rec.get("model")) or "unknown"
            reasoning_effort = _opt_str(rec.get("reasoning_effort"))
            raw_rows.append(
                {
                    "has_record": True,
                    "trial_id": trial_id,
                    "task_id": task_id,
                    "t_meta": t_meta,
                    "arm": arm,
                    "model": model,
                    "reasoning_effort": reasoning_effort,
                    "rec": rec,
                    "rec_sha": rec_sha,
                }
            )
        elif marker_path.is_file():
            marker = read_json_file(marker_path)
            trial_id = _opt_str(marker.get("trial_id")) or trial_dir.name
            task_id = _opt_str(marker.get("task_id"))
            arm = marker.get("arm")
            if task_id is None or arm not in VALID_ARMS:
                raise UsefulnessError(f"malformed interrupted marker at {marker_path}")
            if task_id not in tasks_map:
                raise UsefulnessError(
                    f"interrupted trial {trial_id!r} references unknown task {task_id!r}"
                )
            interrupted_without_record += 1
            raw_rows.append(
                {
                    "has_record": False,
                    "trial_id": trial_id,
                    "task_id": task_id,
                    "t_meta": tasks_map[task_id],
                    "arm": arm,
                    "model": _opt_str(marker.get("model")) or "unknown",
                    "reasoning_effort": _opt_str(marker.get("reasoning_effort")),
                    "marker": marker,
                    "rec_sha": None,
                }
            )

    # Infer model/reasoning_effort for interrupted.json markers that predate record.json
    # so they pair cleanly with their partner arm instead of creating a phantom "unknown" model.
    partner_models: dict[tuple[str, int], tuple[str, str | None]] = {}
    pair_models: dict[str, tuple[str, str | None]] = {}
    known_models: set[tuple[str, str | None]] = set()
    for row in raw_rows:
        if row["has_record"]:
            pid, att = _parse_scheduled_trial_id(row["trial_id"], row["arm"], schedule_pair_ids)
            pair_model = (row["model"], row["reasoning_effort"])
            known_models.add(pair_model)
            if pid is not None and att is not None:
                partner_models[(pid, att)] = pair_model
                pair_models[pid] = pair_model

    for row in raw_rows:
        if not row["has_record"] and row["model"] == "unknown":
            marker = row["marker"]
            pid = _opt_str(marker.get("pair_id"))
            att = _opt_int(marker.get("attempt"))
            inferred: tuple[str, str | None] | None = None
            if pid is not None and att is not None and (pid, att) in partner_models:
                inferred = partner_models[(pid, att)]
            elif pid is not None and pid in pair_models:
                inferred = pair_models[pid]
            elif len(known_models) == 1:
                inferred = next(iter(known_models))
            if inferred is not None:
                row["model"], row["reasoning_effort"] = inferred

    model_key_map = _model_keys_from_entries(
        [(r["model"], r["reasoning_effort"]) for r in raw_rows]
    )

    entries: list[_TrialEntry] = []
    for row in raw_rows:
        trial_id = row["trial_id"]
        task_id = row["task_id"]
        t_meta = row["t_meta"]
        arm = row["arm"]
        task_family = _opt_str(t_meta.get("task_family"))
        split = _opt_str(t_meta.get("split"))
        if task_family is None or split not in ("calibration", "held_out"):
            raise UsefulnessError(f"invalid task manifest metadata for {task_id!r}")
        raw_strata = t_meta.get("strata") or ()
        strata = tuple(sorted({s for s in raw_strata if isinstance(s, str) and s.strip()}))

        model = row["model"]
        reasoning_effort = row["reasoning_effort"]
        model_key = model_key_map[(model, reasoning_effort)]

        if row["has_record"]:
            rec = row["rec"]
            rec_sha = row["rec_sha"]
            status = _opt_str(rec.get("status")) or "failed"
            exec_fail = _opt_str(rec.get("failure_category"))
            reg_sha = _opt_str(rec.get("registration_sha256"))

            tokens = rec.get("tokens") if isinstance(rec.get("tokens"), Mapping) else {}
            wall = rec.get("wall_time") if isinstance(rec.get("wall_time"), Mapping) else {}
            ra = (
                rec.get("recall_accounting")
                if isinstance(rec.get("recall_accounting"), Mapping)
                else None
            )
            cites = (
                ra.get("citations")
                if ra is not None and isinstance(ra.get("citations"), Mapping)
                else {}
            )

            has_ra = ra is not None
            made_no_recall = _opt_bool(ra.get("made_no_recall")) if ra is not None else None
            total_recalls = (_opt_int(ra.get("total_recalls")) or 0) if ra is not None else 0
            returned_count = (_opt_int(ra.get("returned_count")) or 0) if ra is not None else 0
            miss_count = (_opt_int(ra.get("miss_count")) or 0) if ra is not None else 0
            degraded_count = (_opt_int(ra.get("degraded_count")) or 0) if ra is not None else 0
            missing_record_count = (
                (_opt_int(ra.get("missing_record_count")) or 0) if ra is not None else 0
            )
            raw_ret_ms = (
                _opt_float(ra.get("total_retrieval_latency_ms")) if ra is not None else None
            )
            retrieval_latency_s = (raw_ret_ms / 1000.0) if raw_ret_ms is not None else None

            citations_made = _opt_int(cites.get("citations_made")) or 0
            cite_calls_in_transcript = _opt_int(cites.get("cite_calls_in_transcript")) or 0
            degraded_cite_calls = _opt_int(cites.get("degraded_cite_calls")) or 0
            raw_ret_ids = cites.get("returned_memory_ids")
            returned_memories_total = len(raw_ret_ids) if isinstance(raw_ret_ids, list) else 0
            returned_memories_cited_count = (
                _opt_int(cites.get("returned_memories_cited_count")) or 0
            )
            placeholder_count = _opt_int(cites.get("placeholder_count")) or 0
            bound_placeholder_count = _opt_int(cites.get("bound_placeholder_count")) or 0
            unbound_placeholder_count = _opt_int(cites.get("unbound_placeholder_count")) or 0
            bound_reply_count = _opt_int(cites.get("bound_reply_count")) or 0
            single_citation_reply_count = _opt_int(cites.get("single_citation_reply_count")) or 0
            multi_citation_reply_count = _opt_int(cites.get("multi_citation_reply_count")) or 0

            pair_id, attempt = _parse_scheduled_trial_id(trial_id, arm, schedule_pair_ids)
        else:
            marker = row["marker"]
            rec_sha = None
            status = "interrupted"
            exec_fail = _opt_str(marker.get("failure_category")) or "interrupted"
            reg_sha = None
            tokens = {}
            wall = {}
            has_ra = False
            made_no_recall = None
            total_recalls = 0
            returned_count = 0
            miss_count = 0
            degraded_count = 0
            missing_record_count = 0
            retrieval_latency_s = None
            citations_made = 0
            cite_calls_in_transcript = 0
            degraded_cite_calls = 0
            returned_memories_total = 0
            returned_memories_cited_count = 0
            placeholder_count = 0
            bound_placeholder_count = 0
            unbound_placeholder_count = 0
            bound_reply_count = 0
            single_citation_reply_count = 0
            multi_citation_reply_count = 0
            pair_id = _opt_str(marker.get("pair_id"))
            attempt = _opt_int(marker.get("attempt"))

        grade = grades_map.get(trial_id)
        if grade is not None:
            if (
                rec_sha is not None
                and isinstance(grade.get("record_sha256"), str)
                and grade["record_sha256"] != rec_sha
            ):
                raise UsefulnessError(
                    f"grade for trial {trial_id!r} does not match record.json SHA-256"
                )
            graded = True
            met_checks = _opt_bool(grade.get("met_checks"))
            grade_failure_category = _opt_str(grade.get("failure_category"))
            confident_wrong_reply = _opt_bool(grade.get("confident_wrong_reply"))
            repeated_earlier_error = _opt_bool(grade.get("repeated_earlier_error"))
            arm_leaked = _opt_bool(grade.get("arm_leaked"))
        else:
            graded = False
            met_checks = None
            grade_failure_category = None
            confident_wrong_reply = None
            repeated_earlier_error = None
            arm_leaked = None

        attr = attributions_map.get(trial_id)
        if attr is not None:
            if (
                grades_final_sha256 is not None
                and isinstance(attr.get("grades_final_sha256"), str)
                and attr["grades_final_sha256"] != grades_final_sha256
            ):
                raise UsefulnessError(
                    f"attribution for trial {trial_id!r} references a different grades_final_sha256"
                )
            attributed = True
            harmful_memory_use = _opt_bool(attr.get("harmful_memory_use"))
            attempted_invalid_memory_use = _opt_bool(attr.get("attempted_invalid_memory_use"))
            confident_wrong_linked_to_memory = _opt_bool(
                attr.get("confident_wrong_linked_to_memory")
            )
        else:
            attributed = False
            harmful_memory_use = None
            attempted_invalid_memory_use = None
            confident_wrong_linked_to_memory = None

        entries.append(
            _TrialEntry(
                trial_id=trial_id,
                task_id=task_id,
                task_family=task_family,
                split=split,
                strata=strata,
                arm=arm,
                model=model,
                reasoning_effort=reasoning_effort,
                model_key=model_key,
                status=status,
                execution_failure_category=exec_fail,
                has_record=row["has_record"],
                record_sha256=rec_sha,
                registration_sha256=reg_sha,
                pair_id=pair_id,
                attempt=attempt,
                total_tokens=_opt_float(tokens.get("total_tokens")),
                input_tokens=_opt_float(tokens.get("input_tokens")),
                output_tokens=_opt_float(tokens.get("output_tokens")),
                thinking_tokens=_opt_float(tokens.get("thinking_tokens")),
                factory_credits=_opt_float(tokens.get("factory_credits")),
                wall_total_s=_opt_float(wall.get("total_s")),
                wall_clone_s=_opt_float(wall.get("clone_s")),
                wall_setup_s=_opt_float(wall.get("setup_s")),
                wall_harness_s=_opt_float(wall.get("harness_s")),
                wall_teardown_s=_opt_float(wall.get("teardown_s")),
                has_recall_accounting=has_ra,
                made_no_recall=made_no_recall,
                total_recalls=total_recalls,
                returned_count=returned_count,
                miss_count=miss_count,
                degraded_count=degraded_count,
                missing_record_count=missing_record_count,
                retrieval_latency_s=retrieval_latency_s,
                citations_made=citations_made,
                cite_calls_in_transcript=cite_calls_in_transcript,
                degraded_cite_calls=degraded_cite_calls,
                returned_memories_total=returned_memories_total,
                returned_memories_cited_count=returned_memories_cited_count,
                placeholder_count=placeholder_count,
                bound_placeholder_count=bound_placeholder_count,
                unbound_placeholder_count=unbound_placeholder_count,
                bound_reply_count=bound_reply_count,
                single_citation_reply_count=single_citation_reply_count,
                multi_citation_reply_count=multi_citation_reply_count,
                graded=graded,
                met_checks=met_checks,
                grade_failure_category=grade_failure_category,
                confident_wrong_reply=confident_wrong_reply,
                repeated_earlier_error=repeated_earlier_error,
                arm_leaked=arm_leaked,
                attributed=attributed,
                harmful_memory_use=harmful_memory_use,
                attempted_invalid_memory_use=attempted_invalid_memory_use,
                confident_wrong_linked_to_memory=confident_wrong_linked_to_memory,
            )
        )

    entries.sort(key=lambda t: t.trial_id)
    return entries, interrupted_without_record


# ---------------------------------------------------------------------------
# Pairing trials within model (resolving retry attempts)
# ---------------------------------------------------------------------------


def _resolve_pairs(trials: Sequence[_TrialEntry]) -> tuple[list[_ResolvedPair], set[str]]:
    """Pair memory and no-memory trials within each model and resolve retry attempts.

    Returns ``(resolved_pairs, superseded_trial_ids)`` sorted deterministically.
    """
    scheduled_groups: dict[tuple[str, str], dict[int, dict[str, _TrialEntry]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    unscheduled: list[_TrialEntry] = []

    for t in trials:
        if t.pair_id is not None and t.attempt is not None:
            arm_slot = scheduled_groups[(t.model_key, t.pair_id)][t.attempt]
            if t.arm in arm_slot:
                raise UsefulnessError(
                    f"duplicate {t.arm} trial for pair {t.pair_id!r} attempt {t.attempt} "
                    f"on model {t.model_key!r}"
                )
            arm_slot[t.arm] = t
        else:
            unscheduled.append(t)

    resolved: list[_ResolvedPair] = []
    superseded_ids: set[str] = set()

    for (model_key, pair_id), attempts_map in sorted(scheduled_groups.items()):
        max_attempt = max(attempts_map)
        final_slot = attempts_map[max_attempt]
        sample = final_slot.get("memory") or final_slot.get("no_memory")
        assert sample is not None

        sup_mem: list[_TrialEntry] = []
        sup_nomem: list[_TrialEntry] = []
        for att in sorted(attempts_map):
            if att == max_attempt:
                continue
            slot = attempts_map[att]
            if "memory" in slot:
                sup_mem.append(slot["memory"])
                superseded_ids.add(slot["memory"].trial_id)
            if "no_memory" in slot:
                sup_nomem.append(slot["no_memory"])
                superseded_ids.add(slot["no_memory"].trial_id)

        resolved.append(
            _ResolvedPair(
                pair_key=f"{model_key}::{pair_id}",
                task_id=sample.task_id,
                task_family=sample.task_family,
                split=sample.split,
                strata=sample.strata,
                model_key=model_key,
                memory=final_slot.get("memory"),
                no_memory=final_slot.get("no_memory"),
                superseded_memory=tuple(sup_mem),
                superseded_no_memory=tuple(sup_nomem),
            )
        )

    # Unscheduled trials: pair by shared (model_key, task_id, stem) when both arms share a stem,
    # otherwise pair deterministically in sorted trial_id order within (model_key, task_id).
    by_model_task: dict[tuple[str, str], list[_TrialEntry]] = defaultdict(list)
    for t in unscheduled:
        by_model_task[(t.model_key, t.task_id)].append(t)

    for (model_key, task_id), group in sorted(by_model_task.items()):
        stem_buckets: dict[str, dict[str, list[_TrialEntry]]] = defaultdict(
            lambda: {"memory": [], "no_memory": []}
        )
        leftovers: dict[str, list[_TrialEntry]] = {"memory": [], "no_memory": []}

        for t in sorted(group, key=lambda x: x.trial_id):
            stem = _derive_unscheduled_stem(t.trial_id, t.arm)
            if stem is not None:
                stem_buckets[stem][t.arm].append(t)
            else:
                leftovers[t.arm].append(t)

        for stem, arms in sorted(stem_buckets.items()):
            if len(arms["memory"]) == 1 and len(arms["no_memory"]) == 1:
                mem_t = arms["memory"][0]
                nomem_t = arms["no_memory"][0]
                resolved.append(
                    _ResolvedPair(
                        pair_key=f"{model_key}::{stem}",
                        task_id=task_id,
                        task_family=mem_t.task_family,
                        split=mem_t.split,
                        strata=mem_t.strata,
                        model_key=model_key,
                        memory=mem_t,
                        no_memory=nomem_t,
                        superseded_memory=(),
                        superseded_no_memory=(),
                    )
                )
            else:
                leftovers["memory"].extend(arms["memory"])
                leftovers["no_memory"].extend(arms["no_memory"])

        mems = sorted(leftovers["memory"], key=lambda x: x.trial_id)
        nomems = sorted(leftovers["no_memory"], key=lambda x: x.trial_id)
        pair_count = max(len(mems), len(nomems))
        for idx in range(pair_count):
            mem_t = mems[idx] if idx < len(mems) else None
            nomem_t = nomems[idx] if idx < len(nomems) else None
            sample = mem_t or nomem_t
            assert sample is not None
            resolved.append(
                _ResolvedPair(
                    pair_key=f"{model_key}::{task_id}__p{idx + 1}",
                    task_id=task_id,
                    task_family=sample.task_family,
                    split=sample.split,
                    strata=sample.strata,
                    model_key=model_key,
                    memory=mem_t,
                    no_memory=nomem_t,
                    superseded_memory=(),
                    superseded_no_memory=(),
                )
            )

    resolved.sort(key=lambda p: p.pair_key)
    return resolved, superseded_ids


# ---------------------------------------------------------------------------
# Held-out gate & registered stopping rule evaluation
# ---------------------------------------------------------------------------


def _check_budget_stop(
    stop_at: Any,
    held_out_trials: Sequence[_TrialEntry],
) -> tuple[bool, str | None]:
    """Return ``(reached, detail)`` if ``stop_at`` declares a resource budget that was met."""
    if not isinstance(stop_at, Mapping):
        return (False, None)

    for key in ("max_total_tokens", "total_tokens", "token_budget"):
        limit = _opt_float(stop_at.get(key))
        if limit is not None and limit > 0:
            observed = sum(t.total_tokens for t in held_out_trials if t.total_tokens is not None)
            if observed >= limit:
                return (True, f"total_tokens reached budget ({int(observed)} >= {int(limit)})")

    for key in ("max_wall_time_s", "total_wall_time_s", "wall_time_budget_s"):
        limit = _opt_float(stop_at.get(key))
        if limit is not None and limit > 0:
            observed = sum(t.wall_total_s for t in held_out_trials if t.wall_total_s is not None)
            if observed >= limit:
                return (True, f"total_wall_time_s reached budget ({observed:.1f}s >= {limit:.1f}s)")

    for key in ("max_factory_credits", "total_factory_credits", "credit_budget"):
        limit = _opt_float(stop_at.get(key))
        if limit is not None and limit > 0:
            observed = sum(
                t.factory_credits for t in held_out_trials if t.factory_credits is not None
            )
            if observed >= limit:
                return (
                    True,
                    f"total_factory_credits reached budget ({observed:.4f} >= {limit:.4f})",
                )

    max_trials = _opt_int(stop_at.get("max_trials"))
    if max_trials is not None and max_trials > 0 and len(held_out_trials) >= max_trials:
        return (True, f"trial count reached budget ({len(held_out_trials)} >= {max_trials})")

    return (False, None)


def _target_held_out_pairs(registration: Mapping[str, Any]) -> int:
    stop_at = registration["stopping_rule"]["stop_at"]
    if isinstance(stop_at, Mapping):
        for key in ("held_out_pairs", "pairs", "sample_size"):
            val = _opt_int(stop_at.get(key))
            if val is not None and val >= 1:
                return val
    return int(registration["task_manifest"]["sample_size"])


def _evaluate_held_out_stop(
    experiment_home: Config | Path,
    store: UsefulnessRunStore,
    *,
    registration: Mapping[str, Any],
    held_out_trials: Sequence[_TrialEntry],
    held_out_pairs: Sequence[_ResolvedPair],
) -> dict[str, Any]:
    """Verify the held-out run reached its registered stop; raise ``UsefulnessError`` otherwise."""
    stopping_rule = registration["stopping_rule"]
    stop_at = stopping_rule["stop_at"]
    safety_limit = int(stopping_rule["safety_stop_harmful_uses"])

    # 1. Safety stop check (harmful or invalid memory uses in finalized memory-arm trials).
    harmful_trials = [
        t
        for t in held_out_trials
        if t.arm == "memory" and (t.harmful_memory_use is True or t.attempted_invalid_memory_use)
    ]
    if len(harmful_trials) >= safety_limit:
        return {
            "reached": True,
            "stopped_by": "safety_stop",
            "detail": (
                f"safety stop reached ({len(harmful_trials)} harmful/invalid memory use(s) "
                f">= registered threshold {safety_limit})"
            ),
            "harmful_memory_uses": len(harmful_trials),
            "safety_stop_harmful_uses": safety_limit,
        }

    # 2. Resource budget stop check (if declared in stopping_rule.stop_at).
    budget_reached, budget_detail = _check_budget_stop(stop_at, held_out_trials)
    if budget_reached and budget_detail is not None:
        return {
            "reached": True,
            "stopped_by": "resource_budget",
            "detail": budget_detail,
            "harmful_memory_uses": len(harmful_trials),
            "safety_stop_harmful_uses": safety_limit,
        }

    # 3. Sample-size / schedule-completion stop check.
    target_pairs = _target_held_out_pairs(registration)
    if store.schedule_path.exists():
        progress = inspect_schedule_progress(experiment_home, store.run_id)
        if progress.schedule.task_manifest_sha256 != registration["task_manifest"]["sha256"]:
            raise UsefulnessError(
                "schedule.json task_manifest_sha256 does not match registration.json"
            )
        if progress.schedule.split in ("held_out", "all") and not progress.is_complete:
            raise UsefulnessError(
                f"cannot report held-out results for run {store.run_id!r} before its "
                f"registered stop: schedule still has {len(progress.pending_slots)} pending "
                f"trial(s) (and neither a resource budget nor the safety stop of "
                f"{safety_limit} harmful use(s) was reached)"
            )

    complete_arm_pairs = [
        p for p in held_out_pairs if p.memory is not None and p.no_memory is not None
    ]
    half_pairs = [p for p in held_out_pairs if p.memory is None or p.no_memory is None]
    if half_pairs:
        raise UsefulnessError(
            f"cannot report held-out results for run {store.run_id!r} before its "
            f"registered stop: {len(half_pairs)} held-out pair(s) are missing one arm"
        )
    if len(complete_arm_pairs) < target_pairs:
        raise UsefulnessError(
            f"cannot report held-out results for run {store.run_id!r} before its "
            f"registered stop: {len(complete_arm_pairs)} of {target_pairs} registered "
            f"held-out pair(s) finished"
        )

    missing_models: list[str] = []
    for entry in registration["models"]["models"]:
        model_name = entry["model"]
        if "reasoning_effort" in entry:
            re_val = entry["reasoning_effort"]
            matched = any(
                t.model == model_name and t.reasoning_effort == re_val for t in held_out_trials
            )
            label = f"{model_name} (reasoning={re_val})" if re_val is not None else model_name
        else:
            matched = any(t.model == model_name for t in held_out_trials)
            label = model_name
        if not matched:
            missing_models.append(label)
    if missing_models:
        raise UsefulnessError(
            f"cannot report held-out results for run {store.run_id!r} before its "
            f"registered stop: registered model(s) have no held-out trials yet: "
            + ", ".join(sorted(missing_models))
        )

    return {
        "reached": True,
        "stopped_by": "sample_size",
        "detail": (
            f"registered stop reached ({len(complete_arm_pairs)} held-out pair(s) finished; "
            f"target >= {target_pairs})"
        ),
        "harmful_memory_uses": len(harmful_trials),
        "safety_stop_harmful_uses": safety_limit,
    }


# ---------------------------------------------------------------------------
# Secondary measures, failure categories, arm leaks, and exploratory slices
# ---------------------------------------------------------------------------


def _has_declared_or_observed_memory_use(trial: _TrialEntry) -> bool | None:
    if not trial.has_recall_accounting:
        return None
    return trial.returned_count > 0 or trial.citations_made > 0


def _pair_total_numeric(
    pair: _ResolvedPair,
    getter: Any,
    *,
    include_retries: bool = False,
) -> NumericPair:
    def _arm_val(
        final_t: _TrialEntry | None,
        superseded: Sequence[_TrialEntry],
    ) -> float | None:
        if final_t is None:
            return None
        val = getter(final_t)
        if val is None:
            return None
        if not include_retries:
            return val
        total = val
        for prev in superseded:
            pval = getter(prev)
            if pval is None:
                return None
            total += pval
        return total

    return NumericPair(
        family=pair.task_family,
        memory=_arm_val(pair.memory, pair.superseded_memory),
        no_memory=_arm_val(pair.no_memory, pair.superseded_no_memory),
        task_id=pair.task_id,
    )


def _trial_harmful_outcomes(
    trial: _TrialEntry,
) -> tuple[bool | None, bool | None, bool | None, bool]:
    """Return ``(harmful_wrong_action, attempted_invalid, combined, assumed_passing_safe)``.

    - If an attribution row exists, use its recorded booleans directly.
    - If a trial failed or gave a confident wrong reply without attribution,
      return ``None`` (missing — never coerce unattributed failures to False).
    - If a trial met checks and did not give a confident wrong reply and has no
      attribution row, return ``False`` and flag ``assumed_passing_safe=True`` so
      the report discloses how many passing trials were not opened in attribution.
    """
    if trial.attributed:
        h = trial.harmful_memory_use
        inv = trial.attempted_invalid_memory_use
        comb = (
            True
            if (h is True or inv is True)
            else (False if (h is False and inv is False) else None)
        )
        return (h, inv, comb, False)
    if trial.graded and trial.met_checks is True and trial.confident_wrong_reply is False:
        return (False, False, False, True)
    return (None, None, None, False)


def _build_secondary_measures(
    resolved_pairs: Sequence[_ResolvedPair],
    *,
    all_split_trials: Sequence[_TrialEntry],
    bootstrap_seed: int,
    bootstrap_resamples: int,
    confidence: float,
) -> dict[str, Any]:
    mem_trials = [p.memory for p in resolved_pairs if p.memory is not None]
    nomem_trials = [p.no_memory for p in resolved_pairs if p.no_memory is not None]

    # 1. Repeat-error rate (recurrence stratum)
    rec_mem_obs = [
        BinaryObservation(
            family=t.task_family,
            outcome=t.repeated_earlier_error,
            task_id=t.task_id,
        )
        for t in mem_trials
        if any(s.lower() == "recurrence" for s in t.strata)
    ]
    rec_nomem_obs = [
        BinaryObservation(
            family=t.task_family,
            outcome=t.repeated_earlier_error,
            task_id=t.task_id,
        )
        for t in nomem_trials
        if any(s.lower() == "recurrence" for s in t.strata)
    ]
    rec_pairs = [
        BinaryPair(
            family=p.task_family,
            memory=p.memory.repeated_earlier_error if p.memory is not None else None,
            no_memory=p.no_memory.repeated_earlier_error if p.no_memory is not None else None,
            task_id=p.task_id,
        )
        for p in resolved_pairs
        if any(s.lower() == "recurrence" for s in p.strata)
    ]
    repeat_error = {
        "memory_arm": _rate_to_dict(
            clustered_rate_summary(
                rec_mem_obs,
                seed=_derive_seed(bootstrap_seed, "sec:repeat_error:memory"),
                n_resamples=bootstrap_resamples,
                confidence=confidence,
            ),
            denominator_label="graded recurrence-stratum memory-arm trials",
        ),
        "no_memory_arm": _rate_to_dict(
            clustered_rate_summary(
                rec_nomem_obs,
                seed=_derive_seed(bootstrap_seed, "sec:repeat_error:no_memory"),
                n_resamples=bootstrap_resamples,
                confidence=confidence,
            ),
            denominator_label="graded recurrence-stratum no-memory-arm trials",
        ),
        "paired_difference": _paired_success_to_dict(
            paired_success_summary(
                rec_pairs,
                seed=_derive_seed(bootstrap_seed, "sec:repeat_error:paired"),
                n_resamples=bootstrap_resamples,
                confidence=confidence,
            ),
            exploratory=False,
            note="Memory minus no-memory repeat-error rate (negative favors memory).",
        ),
    }

    # 2. Harmful memory use (memory arm)
    harm_wrong_obs: list[BinaryObservation] = []
    harm_invalid_obs: list[BinaryObservation] = []
    harm_combined_obs: list[BinaryObservation] = []
    assumed_passing_safe_count = 0
    for t in mem_trials:
        h, inv, comb, assumed = _trial_harmful_outcomes(t)
        if assumed:
            assumed_passing_safe_count += 1
        harm_wrong_obs.append(BinaryObservation(family=t.task_family, outcome=h, task_id=t.task_id))
        harm_invalid_obs.append(
            BinaryObservation(family=t.task_family, outcome=inv, task_id=t.task_id)
        )
        harm_combined_obs.append(
            BinaryObservation(family=t.task_family, outcome=comb, task_id=t.task_id)
        )

    harmful_memory_use = {
        "combined": _rate_to_dict(
            clustered_rate_summary(
                harm_combined_obs,
                seed=_derive_seed(bootstrap_seed, "sec:harm:combined"),
                n_resamples=bootstrap_resamples,
                confidence=confidence,
            ),
            denominator_label="memory-arm trials with known attribution status",
        ),
        "wrong_action": _rate_to_dict(
            clustered_rate_summary(
                harm_wrong_obs,
                seed=_derive_seed(bootstrap_seed, "sec:harm:wrong_action"),
                n_resamples=bootstrap_resamples,
                confidence=confidence,
            ),
            denominator_label="memory-arm trials with known attribution status",
        ),
        "attempted_invalid_memory_use": _rate_to_dict(
            clustered_rate_summary(
                harm_invalid_obs,
                seed=_derive_seed(bootstrap_seed, "sec:harm:invalid"),
                n_resamples=bootstrap_resamples,
                confidence=confidence,
            ),
            denominator_label="memory-arm trials with known attribution status",
        ),
        "passing_trials_not_opened_in_attribution": assumed_passing_safe_count,
    }

    # 3. Confident wrong reply (memory-linked / memory-use + all-trial rates + stated failures)
    cwr_linked_among_use_obs: list[BinaryObservation] = []
    cwr_mem_all_obs: list[BinaryObservation] = []
    cwr_nomem_all_obs: list[BinaryObservation] = []
    cwr_mem_unlinked_obs: list[BinaryObservation] = []
    cwr_nomem_unlinked_obs: list[BinaryObservation] = []
    stated_fail_mem_obs: list[BinaryObservation] = []
    stated_fail_nomem_obs: list[BinaryObservation] = []

    for t in mem_trials:
        cwr = t.confident_wrong_reply
        cwr_mem_all_obs.append(
            BinaryObservation(family=t.task_family, outcome=cwr, task_id=t.task_id)
        )
        stated = (
            (t.met_checks is False and cwr is False)
            if (t.met_checks is not None and cwr is not None)
            else None
        )
        stated_fail_mem_obs.append(
            BinaryObservation(family=t.task_family, outcome=stated, task_id=t.task_id)
        )

        used_mem = _has_declared_or_observed_memory_use(t)
        if used_mem is True:
            if cwr is False:
                linked_outcome: bool | None = False
            elif cwr is True:
                linked_outcome = t.confident_wrong_linked_to_memory if t.attributed else None
            else:
                linked_outcome = None
            cwr_linked_among_use_obs.append(
                BinaryObservation(family=t.task_family, outcome=linked_outcome, task_id=t.task_id)
            )

        if cwr is False:
            unlinked_outcome: bool | None = False
        elif cwr is True:
            if used_mem is False:
                unlinked_outcome = True
            elif t.attributed and t.confident_wrong_linked_to_memory is not None:
                unlinked_outcome = not t.confident_wrong_linked_to_memory
            else:
                unlinked_outcome = None
        else:
            unlinked_outcome = None
        cwr_mem_unlinked_obs.append(
            BinaryObservation(family=t.task_family, outcome=unlinked_outcome, task_id=t.task_id)
        )

    for t in nomem_trials:
        cwr = t.confident_wrong_reply
        cwr_nomem_all_obs.append(
            BinaryObservation(family=t.task_family, outcome=cwr, task_id=t.task_id)
        )
        cwr_nomem_unlinked_obs.append(
            BinaryObservation(family=t.task_family, outcome=cwr, task_id=t.task_id)
        )
        stated = (
            (t.met_checks is False and cwr is False)
            if (t.met_checks is not None and cwr is not None)
            else None
        )
        stated_fail_nomem_obs.append(
            BinaryObservation(family=t.task_family, outcome=stated, task_id=t.task_id)
        )

    confident_wrong_reply = {
        "memory_linked_among_memory_use": _rate_to_dict(
            clustered_rate_summary(
                cwr_linked_among_use_obs,
                seed=_derive_seed(bootstrap_seed, "sec:cwr:linked_among_use"),
                n_resamples=bootstrap_resamples,
                confidence=confidence,
            ),
            denominator_label="memory-arm trials with declared or observed memory use",
        ),
        "all_trials_memory_arm": _rate_to_dict(
            clustered_rate_summary(
                cwr_mem_all_obs,
                seed=_derive_seed(bootstrap_seed, "sec:cwr:all_memory"),
                n_resamples=bootstrap_resamples,
                confidence=confidence,
            ),
            denominator_label="graded memory-arm trials",
        ),
        "all_trials_no_memory_arm": _rate_to_dict(
            clustered_rate_summary(
                cwr_nomem_all_obs,
                seed=_derive_seed(bootstrap_seed, "sec:cwr:all_no_memory"),
                n_resamples=bootstrap_resamples,
                confidence=confidence,
            ),
            denominator_label="graded no-memory-arm trials",
        ),
        "without_memory_link_memory_arm": _rate_to_dict(
            clustered_rate_summary(
                cwr_mem_unlinked_obs,
                seed=_derive_seed(bootstrap_seed, "sec:cwr:unlinked_memory"),
                n_resamples=bootstrap_resamples,
                confidence=confidence,
            ),
            denominator_label="graded memory-arm trials (excluding unattributed memory-use CWR)",
        ),
        "without_memory_link_no_memory_arm": _rate_to_dict(
            clustered_rate_summary(
                cwr_nomem_unlinked_obs,
                seed=_derive_seed(bootstrap_seed, "sec:cwr:unlinked_no_memory"),
                n_resamples=bootstrap_resamples,
                confidence=confidence,
            ),
            denominator_label="graded no-memory-arm trials",
        ),
        "stated_failures_memory_arm": _rate_to_dict(
            clustered_rate_summary(
                stated_fail_mem_obs,
                seed=_derive_seed(bootstrap_seed, "sec:cwr:stated_fail_memory"),
                n_resamples=bootstrap_resamples,
                confidence=confidence,
            ),
            denominator_label="graded memory-arm trials",
        ),
        "stated_failures_no_memory_arm": _rate_to_dict(
            clustered_rate_summary(
                stated_fail_nomem_obs,
                seed=_derive_seed(bootstrap_seed, "sec:cwr:stated_fail_no_memory"),
                n_resamples=bootstrap_resamples,
                confidence=confidence,
            ),
            denominator_label="graded no-memory-arm trials",
        ),
    }

    # 4. Total tokens (plus component breakdowns, missing preserved as None)
    has_retries = any(p.superseded_memory or p.superseded_no_memory for p in resolved_pairs)
    tokens: dict[str, Any] = {
        "total_tokens": _paired_numeric_to_dict(
            paired_numeric_summary(
                [_pair_total_numeric(p, lambda t: t.total_tokens) for p in resolved_pairs],
                seed=_derive_seed(bootstrap_seed, "sec:tokens:total"),
                n_resamples=bootstrap_resamples,
                confidence=confidence,
                tail_quantiles=DEFAULT_TAIL_QUANTILES,
            )
        ),
        "input_tokens": _paired_numeric_to_dict(
            paired_numeric_summary(
                [_pair_total_numeric(p, lambda t: t.input_tokens) for p in resolved_pairs],
                seed=_derive_seed(bootstrap_seed, "sec:tokens:input"),
                n_resamples=bootstrap_resamples,
                confidence=confidence,
                tail_quantiles=DEFAULT_TAIL_QUANTILES,
            )
        ),
        "output_tokens": _paired_numeric_to_dict(
            paired_numeric_summary(
                [_pair_total_numeric(p, lambda t: t.output_tokens) for p in resolved_pairs],
                seed=_derive_seed(bootstrap_seed, "sec:tokens:output"),
                n_resamples=bootstrap_resamples,
                confidence=confidence,
                tail_quantiles=DEFAULT_TAIL_QUANTILES,
            )
        ),
        "thinking_tokens": _paired_numeric_to_dict(
            paired_numeric_summary(
                [_pair_total_numeric(p, lambda t: t.thinking_tokens) for p in resolved_pairs],
                seed=_derive_seed(bootstrap_seed, "sec:tokens:thinking"),
                n_resamples=bootstrap_resamples,
                confidence=confidence,
                tail_quantiles=DEFAULT_TAIL_QUANTILES,
            )
        ),
        "factory_credits": _paired_numeric_to_dict(
            paired_numeric_summary(
                [_pair_total_numeric(p, lambda t: t.factory_credits) for p in resolved_pairs],
                seed=_derive_seed(bootstrap_seed, "sec:tokens:credits"),
                n_resamples=bootstrap_resamples,
                confidence=confidence,
                tail_quantiles=DEFAULT_TAIL_QUANTILES,
            )
        ),
    }
    if has_retries:
        tokens["total_tokens_including_retries"] = _paired_numeric_to_dict(
            paired_numeric_summary(
                [
                    _pair_total_numeric(p, lambda t: t.total_tokens, include_retries=True)
                    for p in resolved_pairs
                ],
                seed=_derive_seed(bootstrap_seed, "sec:tokens:total_with_retries"),
                n_resamples=bootstrap_resamples,
                confidence=confidence,
                tail_quantiles=DEFAULT_TAIL_QUANTILES,
            )
        )

    # 5. Latency / wall time (end-to-end + itemized stages + retrieval + retries)
    superseded_mem_wall = sum(
        t.wall_total_s
        for p in resolved_pairs
        for t in p.superseded_memory
        if t.wall_total_s is not None
    )
    superseded_nomem_wall = sum(
        t.wall_total_s
        for p in resolved_pairs
        for t in p.superseded_no_memory
        if t.wall_total_s is not None
    )
    latency: dict[str, Any] = {
        "total_s": _paired_numeric_to_dict(
            paired_numeric_summary(
                [_pair_total_numeric(p, lambda t: t.wall_total_s) for p in resolved_pairs],
                seed=_derive_seed(bootstrap_seed, "sec:latency:total_s"),
                n_resamples=bootstrap_resamples,
                confidence=confidence,
                tail_quantiles=DEFAULT_TAIL_QUANTILES,
            )
        ),
        "harness_s": _paired_numeric_to_dict(
            paired_numeric_summary(
                [_pair_total_numeric(p, lambda t: t.wall_harness_s) for p in resolved_pairs],
                seed=_derive_seed(bootstrap_seed, "sec:latency:harness_s"),
                n_resamples=bootstrap_resamples,
                confidence=confidence,
                tail_quantiles=DEFAULT_TAIL_QUANTILES,
            )
        ),
        "clone_s": _paired_numeric_to_dict(
            paired_numeric_summary(
                [_pair_total_numeric(p, lambda t: t.wall_clone_s) for p in resolved_pairs],
                seed=_derive_seed(bootstrap_seed, "sec:latency:clone_s"),
                n_resamples=bootstrap_resamples,
                confidence=confidence,
                tail_quantiles=DEFAULT_TAIL_QUANTILES,
            )
        ),
        "setup_s": _paired_numeric_to_dict(
            paired_numeric_summary(
                [_pair_total_numeric(p, lambda t: t.wall_setup_s) for p in resolved_pairs],
                seed=_derive_seed(bootstrap_seed, "sec:latency:setup_s"),
                n_resamples=bootstrap_resamples,
                confidence=confidence,
                tail_quantiles=DEFAULT_TAIL_QUANTILES,
            )
        ),
        "teardown_s": _paired_numeric_to_dict(
            paired_numeric_summary(
                [_pair_total_numeric(p, lambda t: t.wall_teardown_s) for p in resolved_pairs],
                seed=_derive_seed(bootstrap_seed, "sec:latency:teardown_s"),
                n_resamples=bootstrap_resamples,
                confidence=confidence,
                tail_quantiles=DEFAULT_TAIL_QUANTILES,
            )
        ),
        "retrieval_s_memory_arm": _distribution_to_dict(
            distribution_summary(
                [(t.task_family, t.retrieval_latency_s) for t in mem_trials],
                seed=_derive_seed(bootstrap_seed, "sec:latency:retrieval_s"),
                n_resamples=bootstrap_resamples,
                confidence=confidence,
                tail_quantiles=DEFAULT_TAIL_QUANTILES,
            )
        ),
        "retry_time_s": {
            "memory_arm_total_s": superseded_mem_wall,
            "no_memory_arm_total_s": superseded_nomem_wall,
            "total_s": superseded_mem_wall + superseded_nomem_wall,
            "superseded_trial_count": sum(
                len(p.superseded_memory) + len(p.superseded_no_memory) for p in resolved_pairs
            ),
        },
        "itemization_note": (
            "harness_s combines model inference and in-harness build/tool execution because "
            "headless Droid reports end-to-end duration without splitting build time; "
            "retrieval_s_memory_arm reflects ledger-recorded retrieval latency."
        ),
    }
    if has_retries:
        latency["total_s_including_retries"] = _paired_numeric_to_dict(
            paired_numeric_summary(
                [
                    _pair_total_numeric(p, lambda t: t.wall_total_s, include_retries=True)
                    for p in resolved_pairs
                ],
                seed=_derive_seed(bootstrap_seed, "sec:latency:total_s_with_retries"),
                n_resamples=bootstrap_resamples,
                confidence=confidence,
                tail_quantiles=DEFAULT_TAIL_QUANTILES,
            )
        )

    # 6. Recall and retrieval (memory arm; every outcome reported separately with denominator & CI)
    recall_and_retrieval = {
        "made_no_recall": _rate_to_dict(
            clustered_rate_summary(
                [
                    BinaryObservation(
                        family=t.task_family,
                        outcome=t.made_no_recall if t.has_recall_accounting else None,
                        task_id=t.task_id,
                    )
                    for t in mem_trials
                ],
                seed=_derive_seed(bootstrap_seed, "sec:recall:none"),
                n_resamples=bootstrap_resamples,
                confidence=confidence,
            ),
            denominator_label="memory-arm trials with recall accounting",
        ),
        "with_returned_memories": _rate_to_dict(
            clustered_rate_summary(
                [
                    BinaryObservation(
                        family=t.task_family,
                        outcome=(t.returned_count > 0) if t.has_recall_accounting else None,
                        task_id=t.task_id,
                    )
                    for t in mem_trials
                ],
                seed=_derive_seed(bootstrap_seed, "sec:recall:returned"),
                n_resamples=bootstrap_resamples,
                confidence=confidence,
            ),
            denominator_label="memory-arm trials with recall accounting",
        ),
        "with_miss": _rate_to_dict(
            clustered_rate_summary(
                [
                    BinaryObservation(
                        family=t.task_family,
                        outcome=(t.miss_count > 0) if t.has_recall_accounting else None,
                        task_id=t.task_id,
                    )
                    for t in mem_trials
                ],
                seed=_derive_seed(bootstrap_seed, "sec:recall:miss"),
                n_resamples=bootstrap_resamples,
                confidence=confidence,
            ),
            denominator_label="memory-arm trials with recall accounting",
        ),
        "with_degraded": _rate_to_dict(
            clustered_rate_summary(
                [
                    BinaryObservation(
                        family=t.task_family,
                        outcome=(t.degraded_count > 0) if t.has_recall_accounting else None,
                        task_id=t.task_id,
                    )
                    for t in mem_trials
                ],
                seed=_derive_seed(bootstrap_seed, "sec:recall:degraded"),
                n_resamples=bootstrap_resamples,
                confidence=confidence,
            ),
            denominator_label="memory-arm trials with recall accounting",
        ),
        "with_missing_record": _rate_to_dict(
            clustered_rate_summary(
                [
                    BinaryObservation(
                        family=t.task_family,
                        outcome=(t.missing_record_count > 0) if t.has_recall_accounting else None,
                        task_id=t.task_id,
                    )
                    for t in mem_trials
                ],
                seed=_derive_seed(bootstrap_seed, "sec:recall:missing_record"),
                n_resamples=bootstrap_resamples,
                confidence=confidence,
            ),
            denominator_label="memory-arm trials with recall accounting",
        ),
        "call_totals": {
            "total_recalls": sum(t.total_recalls for t in mem_trials),
            "returned": sum(t.returned_count for t in mem_trials),
            "miss": sum(t.miss_count for t in mem_trials),
            "degraded": sum(t.degraded_count for t in mem_trials),
            "missing_record": sum(t.missing_record_count for t in mem_trials),
            "all_attempts_total_recalls": sum(
                t.total_recalls for t in all_split_trials if t.arm == "memory"
            ),
        },
    }

    # 7. Citations (memory arm; nothing inferred about what caused a reply)
    total_returned_memories = sum(t.returned_memories_total for t in mem_trials)
    total_returned_cited = sum(t.returned_memories_cited_count for t in mem_trials)
    total_placeholders = sum(t.placeholder_count for t in mem_trials)
    total_bound_placeholders = sum(t.bound_placeholder_count for t in mem_trials)
    total_unbound_placeholders = sum(t.unbound_placeholder_count for t in mem_trials)
    total_bound_replies = sum(t.bound_reply_count for t in mem_trials)
    total_single_cite_replies = sum(t.single_citation_reply_count for t in mem_trials)
    total_multi_cite_replies = sum(t.multi_citation_reply_count for t in mem_trials)

    citations = {
        "note": (
            "Citations record memories declared used by tool calls and their reply bindings; "
            "nothing is inferred about what caused a reply."
        ),
        "trials_with_citation": _rate_to_dict(
            clustered_rate_summary(
                [
                    BinaryObservation(
                        family=t.task_family,
                        outcome=(t.citations_made > 0) if t.has_recall_accounting else None,
                        task_id=t.task_id,
                    )
                    for t in mem_trials
                ],
                seed=_derive_seed(bootstrap_seed, "sec:cites:trials_with_citation"),
                n_resamples=bootstrap_resamples,
                confidence=confidence,
            ),
            denominator_label="memory-arm trials with recall accounting",
        ),
        "trials_citing_returned_memory": _rate_to_dict(
            clustered_rate_summary(
                [
                    BinaryObservation(
                        family=t.task_family,
                        outcome=(t.returned_memories_cited_count > 0),
                        task_id=t.task_id,
                    )
                    for t in mem_trials
                    if t.has_recall_accounting and t.returned_memories_total > 0
                ],
                seed=_derive_seed(bootstrap_seed, "sec:cites:trials_citing_returned"),
                n_resamples=bootstrap_resamples,
                confidence=confidence,
            ),
            denominator_label="memory-arm trials that returned at least one memory",
        ),
        "returned_memories_cited_rate": _count_rate_to_dict(
            total_returned_cited,
            total_returned_memories,
            denominator_label="distinct returned memories summed across memory-arm trials",
            confidence=confidence,
        ),
        "unbound_placeholder_rate": _count_rate_to_dict(
            total_unbound_placeholders,
            total_placeholders,
            denominator_label="citation placeholders across memory-arm trials",
            confidence=confidence,
        ),
        "single_citation_reply_rate": _count_rate_to_dict(
            total_single_cite_replies,
            total_bound_replies,
            denominator_label="bound assistant replies across memory-arm trials",
            confidence=confidence,
        ),
        "counts": {
            "citations_made": sum(t.citations_made for t in mem_trials),
            "cite_calls_in_transcript": sum(t.cite_calls_in_transcript for t in mem_trials),
            "degraded_cite_calls": sum(t.degraded_cite_calls for t in mem_trials),
            "returned_memories_total": total_returned_memories,
            "returned_memories_cited_count": total_returned_cited,
            "placeholder_count": total_placeholders,
            "bound_placeholder_count": total_bound_placeholders,
            "unbound_placeholder_count": total_unbound_placeholders,
            "bound_reply_count": total_bound_replies,
            "single_citation_reply_count": total_single_cite_replies,
            "multi_citation_reply_count": total_multi_cite_replies,
        },
    }

    # 8. Abstention (abstention-stratum memory-arm trials)
    abstention_trials = [t for t in mem_trials if any(s.lower() == "abstention" for s in t.strata)]
    abs_appropriate_obs: list[BinaryObservation] = []
    abs_unnecessary_recall_obs: list[BinaryObservation] = []
    abs_unnecessary_or_harmful_use_obs: list[BinaryObservation] = []
    abs_harmful_obs: list[BinaryObservation] = []

    for t in abstention_trials:
        _, _, harm_comb, _ = _trial_harmful_outcomes(t)
        used_mem = _has_declared_or_observed_memory_use(t)
        abs_harmful_obs.append(
            BinaryObservation(family=t.task_family, outcome=harm_comb, task_id=t.task_id)
        )
        abs_unnecessary_recall_obs.append(
            BinaryObservation(
                family=t.task_family,
                outcome=(not t.made_no_recall)
                if (t.has_recall_accounting and t.made_no_recall is not None)
                else None,
                task_id=t.task_id,
            )
        )
        if not t.has_recall_accounting:
            use_or_harm: bool | None = None
            appropriate: bool | None = None
        elif t.made_no_recall is True and t.citations_made == 0:
            use_or_harm = False
            appropriate = True
        elif used_mem is True or harm_comb is True:
            use_or_harm = True
            appropriate = False
        elif used_mem is False and harm_comb is False:
            use_or_harm = False
            appropriate = True
        else:
            use_or_harm = None
            appropriate = None

        abs_unnecessary_or_harmful_use_obs.append(
            BinaryObservation(family=t.task_family, outcome=use_or_harm, task_id=t.task_id)
        )
        abs_appropriate_obs.append(
            BinaryObservation(family=t.task_family, outcome=appropriate, task_id=t.task_id)
        )

    abstention = {
        "appropriate_non_use": _rate_to_dict(
            clustered_rate_summary(
                abs_appropriate_obs,
                seed=_derive_seed(bootstrap_seed, "sec:abstention:appropriate_non_use"),
                n_resamples=bootstrap_resamples,
                confidence=confidence,
            ),
            denominator_label="abstention-stratum memory-arm trials",
        ),
        "unnecessary_recall": _rate_to_dict(
            clustered_rate_summary(
                abs_unnecessary_recall_obs,
                seed=_derive_seed(bootstrap_seed, "sec:abstention:unnecessary_recall"),
                n_resamples=bootstrap_resamples,
                confidence=confidence,
            ),
            denominator_label="abstention-stratum memory-arm trials",
        ),
        "unnecessary_or_harmful_use": _rate_to_dict(
            clustered_rate_summary(
                abs_unnecessary_or_harmful_use_obs,
                seed=_derive_seed(bootstrap_seed, "sec:abstention:unnecessary_or_harmful_use"),
                n_resamples=bootstrap_resamples,
                confidence=confidence,
            ),
            denominator_label="abstention-stratum memory-arm trials",
        ),
        "harmful_use": _rate_to_dict(
            clustered_rate_summary(
                abs_harmful_obs,
                seed=_derive_seed(bootstrap_seed, "sec:abstention:harmful_use"),
                n_resamples=bootstrap_resamples,
                confidence=confidence,
            ),
            denominator_label="abstention-stratum memory-arm trials",
        ),
    }

    return {
        "repeat_error_rate": repeat_error,
        "harmful_memory_use": harmful_memory_use,
        "confident_wrong_reply": confident_wrong_reply,
        "total_tokens": tokens,
        "latency": latency,
        "recall_and_retrieval": recall_and_retrieval,
        "citations": citations,
        "abstention": abstention,
    }


def _build_failure_categories(
    all_split_trials: Sequence[_TrialEntry],
    superseded_ids: set[str],
) -> dict[str, Any]:
    """Summarize non-ordinary failure categories (both graded and execution-level)."""
    graded_non_ordinary: dict[str, dict[str, Any]] = {}
    for cat in NON_ORDINARY_GRADE_FAILURE_CATEGORIES:
        matches = [t for t in all_split_trials if t.grade_failure_category == cat]
        graded_non_ordinary[cat] = {
            "total": len(matches),
            "memory": sum(1 for t in matches if t.arm == "memory"),
            "no_memory": sum(1 for t in matches if t.arm == "no_memory"),
            "trials": [
                {
                    "trial_id": t.trial_id,
                    "task_id": t.task_id,
                    "arm": t.arm,
                    "model": t.model_key,
                    "superseded_by_retry": t.trial_id in superseded_ids,
                }
                for t in matches
            ],
        }

    ordinary_matches = [
        t for t in all_split_trials if t.grade_failure_category == "ordinary_wrong_answer"
    ]

    exec_categories: dict[str, dict[str, Any]] = {}
    seen_exec_cats = sorted(
        set(FAILURE_CATEGORIES)
        | {
            t.execution_failure_category
            for t in all_split_trials
            if t.execution_failure_category is not None
        }
    )
    for cat in seen_exec_cats:
        matches = [t for t in all_split_trials if t.execution_failure_category == cat]
        if matches or cat in FAILURE_CATEGORIES:
            exec_categories[cat] = {
                "total": len(matches),
                "memory": sum(1 for t in matches if t.arm == "memory"),
                "no_memory": sum(1 for t in matches if t.arm == "no_memory"),
                "trials": [
                    {
                        "trial_id": t.trial_id,
                        "task_id": t.task_id,
                        "arm": t.arm,
                        "model": t.model_key,
                        "has_record": t.has_record,
                        "superseded_by_retry": t.trial_id in superseded_ids,
                    }
                    for t in matches
                ],
            }

    return {
        "graded_non_ordinary_total": sum(v["total"] for v in graded_non_ordinary.values()),
        "graded_non_ordinary": graded_non_ordinary,
        "graded_ordinary_wrong_answer": {
            "total": len(ordinary_matches),
            "memory": sum(1 for t in ordinary_matches if t.arm == "memory"),
            "no_memory": sum(1 for t in ordinary_matches if t.arm == "no_memory"),
        },
        "execution_failures_total": sum(v["total"] for v in exec_categories.values()),
        "execution_failures": exec_categories,
    }


def _build_arm_leaked_summary(
    all_split_trials: Sequence[_TrialEntry],
    superseded_ids: set[str],
    *,
    bootstrap_seed: int,
    bootstrap_resamples: int,
    confidence: float,
) -> dict[str, Any]:
    leaked = [t for t in all_split_trials if t.arm_leaked is True]
    rate = clustered_rate_summary(
        [
            BinaryObservation(
                family=t.task_family,
                outcome=t.arm_leaked,
                task_id=t.task_id,
            )
            for t in all_split_trials
            if t.graded
        ],
        seed=_derive_seed(bootstrap_seed, "integrity:arm_leaked"),
        n_resamples=bootstrap_resamples,
        confidence=confidence,
    )
    return {
        "total": len(leaked),
        "memory": sum(1 for t in leaked if t.arm == "memory"),
        "no_memory": sum(1 for t in leaked if t.arm == "no_memory"),
        "rate": _rate_to_dict(rate, denominator_label="graded trials"),
        "trials": [
            {
                "trial_id": t.trial_id,
                "task_id": t.task_id,
                "arm": t.arm,
                "model": t.model_key,
                "met_checks": t.met_checks,
                "failure_category": t.grade_failure_category,
                "superseded_by_retry": t.trial_id in superseded_ids,
            }
            for t in leaked
        ],
    }


# ---------------------------------------------------------------------------
# Main report builder
# ---------------------------------------------------------------------------


def _resolve_report_split(
    store: UsefulnessRunStore,
    all_trials: Sequence[_TrialEntry],
    requested: SplitFilter | None,
) -> SplitFilter:
    if requested is not None:
        if requested not in VALID_SPLIT_FILTERS:
            raise UsefulnessError(
                f"invalid split {requested!r}: expected one of {VALID_SPLIT_FILTERS}"
            )
        return requested
    if store.schedule_path.exists():
        sched = store.read_schedule()
        raw_split = sched.get("split")
        if raw_split == "calibration":
            return "calibration"
        if raw_split == "held_out":
            return "held_out"
        if raw_split == "all":
            return "all"
    splits_present = {t.split for t in all_trials}
    if "held_out" in splits_present:
        return "held_out"
    if splits_present == {"calibration"}:
        return "calibration"
    return "held_out"


def build_report(
    experiment_home: Config | Path,
    run_id: str,
    *,
    split: SplitFilter | None = None,
    bootstrap_seed: int | None = None,
    bootstrap_resamples: int | None = None,
    confidence: float = DEFAULT_CONFIDENCE,
    write: bool = True,
) -> ReportResult:
    """Build a deterministic local aggregate report for ``run_id``.

    Refuses held-out results until:
    1. ``registration.json`` exists and matches ``task-manifest.json`` and the
       memory set manifest (and every held-out trial carries that registration SHA-256),
    2. every finished trial's grade is final in ``grades-final.json`` (matching
       ``grades.jsonl`` byte for byte), and
    3. the run has reached its registered stop (sample size / schedule complete,
       declared resource budget, or safety stop).
    """
    store = UsefulnessRunStore(experiment_home, run_id)
    task_manifest = store.read_task_manifest()
    task_manifest_sha = store.task_manifest_sha256()

    schedule_pair_ids: set[str] = set()
    if store.schedule_path.exists():
        sched_doc = store.read_schedule()
        for pair in sched_doc.get("pairs") or ():
            if isinstance(pair, Mapping) and isinstance(pair.get("pair_id"), str):
                schedule_pair_ids.add(pair["pair_id"])

    # Determine grade state (verify finalized grades if grades-final.json exists).
    grades_final_doc: dict[str, Any] | None = None
    grades_final_sha: str | None = None
    if store.is_grades_final:
        grades_final_doc, grades_final_sha = verify_finalized_grades(store)
        grades_map: dict[str, dict[str, Any]] = dict(grades_final_doc.get("final_grades") or {})
        total_grade_revisions = int(grades_final_doc.get("total_grade_revisions", 0))
        grades_sha: str | None = _opt_str(grades_final_doc.get("grades_sha256"))
    else:
        grades_map = latest_grades(store)
        total_grade_revisions = len(store.grades())
        grades_sha = sha256_file(store.grades_path) if store.grades_path.is_file() else None

    attributions_map = latest_attributions(store) if store.attributions_path.is_file() else {}

    all_trials, interrupted_without_record = _load_trials_from_disk(
        store,
        task_manifest=task_manifest,
        schedule_pair_ids=schedule_pair_ids,
        grades_map=grades_map,
        attributions_map=attributions_map,
        grades_final_sha256=grades_final_sha,
    )

    effective_split = _resolve_report_split(store, all_trials, split)
    selected_trials = [
        t for t in all_trials if effective_split == "all" or t.split == effective_split
    ]
    includes_held_out = effective_split in ("held_out", "all") or any(
        t.split == "held_out" for t in selected_trials
    )

    # Load registration (mandatory for held-out; optional for calibration).
    reg_result = (
        load_registration(experiment_home, run_id)
        if (includes_held_out or store.registration_path.exists())
        else None
    )
    reg = reg_result.registration if reg_result is not None else None

    # Load memory-set manifest (mandatory for held-out; optional for calibration).
    ms_manifest_path = memory_set_manifest_path(experiment_home)
    ms_manifest: dict[str, Any] | None = None
    ms_manifest_sha: str | None = None
    if ms_manifest_path.is_file():
        ms_manifest = read_json_file(ms_manifest_path)
        ms_manifest_sha = sha256_file(ms_manifest_path)

    if includes_held_out:
        assert reg_result is not None
        assert reg is not None
        if task_manifest_sha != reg["task_manifest"]["sha256"]:
            raise UsefulnessError(
                f"task-manifest.json SHA-256 ({task_manifest_sha}) does not match "
                f"registration.json ({reg['task_manifest']['sha256']})"
            )
        if ms_manifest is None or ms_manifest_sha is None:
            raise UsefulnessError(
                f"memory-set manifest missing at {ms_manifest_path}; required for held-out report"
            )
        if ms_manifest.get("memory_set_sha256") != reg["memory_set"]["sha256"]:
            raise UsefulnessError(
                "memory-set/manifest.json memory_set_sha256 does not match registration.json"
            )
        if ms_manifest_sha != reg["memory_set"]["manifest_sha256"]:
            raise UsefulnessError(
                "memory-set/manifest.json SHA-256 does not match registration.json"
            )
        if not store.is_grades_final or grades_final_doc is None:
            raise UsefulnessError(
                f"cannot report held-out results for run {run_id!r} until every grade is final; "
                "run `mem usefulness finalize-grades` first"
            )
        finished_trials = [t for t in all_trials if t.has_record]
        ungraded_finished = [t.trial_id for t in finished_trials if not t.graded]
        if ungraded_finished:
            raise UsefulnessError(
                f"cannot report held-out results for run {run_id!r}: "
                f"{len(ungraded_finished)} finished trial(s) are not covered by grades-final.json"
            )
        held_out_trials = [t for t in all_trials if t.split == "held_out"]
        for t in held_out_trials:
            if t.has_record and t.registration_sha256 != reg_result.registration_sha256:
                raise UsefulnessError(
                    f"held-out trial {t.trial_id!r} registration_sha256 "
                    f"({t.registration_sha256!r}) does not match registration.json "
                    f"({reg_result.registration_sha256!r})"
                )

    if not selected_trials:
        raise UsefulnessError(f"no trials found in split {effective_split!r} for run {run_id!r}")

    if not includes_held_out and not any(t.graded for t in selected_trials):
        raise UsefulnessError(f"no grades found for split {effective_split!r} in run {run_id!r}")

    # Bootstrap settings: registered runs must use the registered seed and resamples.
    if reg is not None:
        reg_seed = int(reg["analysis"]["bootstrap_seed"])
        reg_resamples = int(reg["analysis"]["bootstrap_resamples"])
        if includes_held_out:
            if bootstrap_seed is not None and bootstrap_seed != reg_seed:
                raise UsefulnessError(
                    f"cannot override registered bootstrap_seed ({reg_seed}) on a held-out report"
                )
            if bootstrap_resamples is not None and bootstrap_resamples != reg_resamples:
                raise UsefulnessError(
                    f"cannot override registered bootstrap_resamples ({reg_resamples}) "
                    "on a held-out report"
                )
        eff_seed = reg_seed if bootstrap_seed is None else bootstrap_seed
        eff_resamples = reg_resamples if bootstrap_resamples is None else bootstrap_resamples
    else:
        eff_seed = DEFAULT_BOOTSTRAP_SEED if bootstrap_seed is None else bootstrap_seed
        eff_resamples = (
            DEFAULT_BOOTSTRAP_RESAMPLES if bootstrap_resamples is None else bootstrap_resamples
        )

    resolved_pairs, superseded_ids = _resolve_pairs(selected_trials)

    if includes_held_out:
        assert reg is not None
        held_out_only_trials = [t for t in all_trials if t.split == "held_out"]
        held_out_only_pairs, _ = _resolve_pairs(held_out_only_trials)
        stopping_status = _evaluate_held_out_stop(
            experiment_home,
            store,
            registration=reg,
            held_out_trials=held_out_only_trials,
            held_out_pairs=held_out_only_pairs,
        )
    else:
        stopping_status = {
            "reached": False,
            "stopped_by": "calibration_unregistered" if reg is None else "calibration",
            "detail": (
                "calibration split report (held-out stop gate does not apply to calibration)"
            ),
            "harmful_memory_uses": sum(
                1
                for t in selected_trials
                if t.arm == "memory"
                and (t.harmful_memory_use is True or t.attempted_invalid_memory_use is True)
            ),
            "safety_stop_harmful_uses": (
                int(reg["stopping_rule"]["safety_stop_harmful_uses"]) if reg is not None else None
            ),
        }

    practical_threshold = (
        float(reg["primary_endpoint"]["practical_benefit_threshold"]) if reg is not None else None
    )
    registered_exploratory = tuple(reg["analysis"]["exploratory_slices"]) if reg is not None else ()
    registered_exploratory_set = {s.strip().lower() for s in registered_exploratory}

    # --- Primary endpoint: registered aggregate, by stratum, and by model (>1 model) ---
    def _binary_pairs(subset: Sequence[_ResolvedPair]) -> list[BinaryPair]:
        return [
            BinaryPair(
                family=p.task_family,
                memory=p.memory.met_checks if p.memory is not None else None,
                no_memory=p.no_memory.met_checks if p.no_memory is not None else None,
                task_id=p.task_id,
            )
            for p in subset
        ]

    aggregate_summary = _paired_success_to_dict(
        paired_success_summary(
            _binary_pairs(resolved_pairs),
            seed=eff_seed,
            n_resamples=eff_resamples,
            confidence=confidence,
        ),
        exploratory=False,
        practical_benefit_threshold=practical_threshold,
        note=(
            "Includes all resolved pairs in the reported split, including memory-arm trials "
            "that made no recall."
        ),
    )

    strata_names = sorted({s for p in resolved_pairs for s in p.strata})
    by_stratum: dict[str, dict[str, Any]] = {}
    for stratum in strata_names:
        s_pairs = [p for p in resolved_pairs if stratum in p.strata]
        is_exp = stratum.lower() in registered_exploratory_set
        by_stratum[stratum] = _paired_success_to_dict(
            paired_success_summary(
                _binary_pairs(s_pairs),
                seed=_derive_seed(eff_seed, f"primary:stratum:{stratum}"),
                n_resamples=eff_resamples,
                confidence=confidence,
            ),
            exploratory=is_exp,
            practical_benefit_threshold=practical_threshold,
        )

    observed_model_keys = sorted({p.model_key for p in resolved_pairs})
    reg_models_list = list(reg["models"]["models"]) if reg is not None else []
    multi_model = len(observed_model_keys) > 1 or len(reg_models_list) > 1

    by_model: dict[str, dict[str, Any]] = {}
    single_model_limitation: str | None = None
    if multi_model:
        for model_key in observed_model_keys:
            m_pairs = [p for p in resolved_pairs if p.model_key == model_key]
            by_model[model_key] = _paired_success_to_dict(
                paired_success_summary(
                    _binary_pairs(m_pairs),
                    seed=_derive_seed(eff_seed, f"primary:model:{model_key}"),
                    n_resamples=eff_resamples,
                    confidence=confidence,
                ),
                exploratory=False,
                practical_benefit_threshold=practical_threshold,
            )
    else:
        only_model = observed_model_keys[0] if observed_model_keys else "unknown"
        single_model_limitation = (
            f"Single model ({only_model}): this result holds for {only_model} only "
            "and is not an estimate for a weaker or stronger model."
        )

    # --- Exploratory slices (always explicitly labeled exploratory=True) ---
    recalled_pairs = [
        p
        for p in resolved_pairs
        if p.memory is not None
        and p.memory.has_recall_accounting
        and p.memory.made_no_recall is False
    ]
    no_leak_pairs = [
        p
        for p in resolved_pairs
        if not (
            (p.memory is not None and p.memory.arm_leaked is True)
            or (p.no_memory is not None and p.no_memory.arm_leaked is True)
        )
    ]
    exploratory_slices: dict[str, dict[str, Any]] = {
        "recalled_pairs_only": _paired_success_to_dict(
            paired_success_summary(
                _binary_pairs(recalled_pairs),
                seed=_derive_seed(eff_seed, "exploratory:recalled_pairs_only"),
                n_resamples=eff_resamples,
                confidence=confidence,
            ),
            exploratory=True,
            practical_benefit_threshold=practical_threshold,
            note=(
                "Exploratory slice: subset of pairs where the memory-arm trial made at least "
                "one recall. Which trials recall is not assigned at random."
            ),
        ),
        "excluding_arm_leaked_pairs": _paired_success_to_dict(
            paired_success_summary(
                _binary_pairs(no_leak_pairs),
                seed=_derive_seed(eff_seed, "exploratory:excluding_arm_leaked_pairs"),
                n_resamples=eff_resamples,
                confidence=confidence,
            ),
            exploratory=True,
            practical_benefit_threshold=practical_threshold,
            note=(
                "Exploratory sensitivity slice excluding any pair where either arm's material "
                "was flagged as giving the arm away."
            ),
        ),
    }

    # --- Secondary measures ---
    secondary_measures = _build_secondary_measures(
        resolved_pairs,
        all_split_trials=selected_trials,
        bootstrap_seed=eff_seed,
        bootstrap_resamples=eff_resamples,
        confidence=confidence,
    )

    # --- Failure categories, arm-leaked trials, deviations, memory validity changes ---
    failure_categories = _build_failure_categories(selected_trials, superseded_ids)
    arm_leaked = _build_arm_leaked_summary(
        selected_trials,
        superseded_ids,
        bootstrap_seed=eff_seed,
        bootstrap_resamples=eff_resamples,
        confidence=confidence,
    )
    deviations_rows = store.deviations()
    deviations = {
        "count": len(deviations_rows),
        "items": [
            {
                "recorded_at": row.get("recorded_at"),
                "reason": row.get("reason"),
                **({"metadata": row["metadata"]} if "metadata" in row else {}),
            }
            for row in deviations_rows
        ],
    }

    if ms_manifest is not None:
        vc = ms_manifest.get("validity_changes") or {}
        counts = ms_manifest.get("counts") or {}
        memory_set_summary: dict[str, Any] | None = {
            "memory_set_sha256": ms_manifest.get("memory_set_sha256"),
            "manifest_sha256": ms_manifest_sha,
            "migration_level": ms_manifest.get("migration_level"),
            "cutoff": ms_manifest.get("cutoff"),
            "last_day": ms_manifest.get("last_day"),
            "memory_counts": {
                "total": counts.get("total"),
                "by_status": counts.get("by_status") or {},
                "by_scope": counts.get("by_scope") or {},
                "by_type": counts.get("by_type") or {},
            },
            "validity_changes": {
                "cutoff": vc.get("cutoff"),
                "last_day": vc.get("last_day"),
                "total_crossing": vc.get("total_crossing"),
                "starts_in_window": vc.get("starts_in_window"),
                "ends_in_window": vc.get("ends_in_window"),
                "non_terminal_crossing": vc.get("non_terminal_crossing"),
            },
        }
    else:
        memory_set_summary = None

    # Safe task-population metadata (no task text, no repo paths)
    tasks_map = task_manifest.get("tasks") or {}
    split_task_ids = sorted(
        tid
        for tid, meta in tasks_map.items()
        if isinstance(meta, Mapping)
        and (effective_split == "all" or meta.get("split") == effective_split)
    )
    split_families = sorted(
        {
            str(tasks_map[tid].get("task_family"))
            for tid in split_task_ids
            if tasks_map[tid].get("task_family")
        }
    )
    excluded_rows = [
        {"task_id": row["task_id"], "reason_code": row["reason_code"]}
        for row in sorted(
            (r for r in (task_manifest.get("excluded") or ()) if isinstance(r, Mapping)),
            key=lambda r: str(r.get("task_id", "")),
        )
    ]

    reg_summary: dict[str, Any] | None = None
    if reg_result is not None and reg is not None:
        reg_summary = {
            "registration_sha256": reg_result.registration_sha256,
            "created_at": reg.get("created_at"),
            "primary_endpoint": reg["primary_endpoint"],
            "task_manifest_sha256": reg["task_manifest"]["sha256"],
            "sample_size": reg["task_manifest"]["sample_size"],
            "memory_set": reg["memory_set"],
            "models": {
                "models": [
                    {
                        "model": m["model"],
                        **(
                            {"reasoning_effort": m["reasoning_effort"]}
                            if "reasoning_effort" in m
                            else {}
                        ),
                    }
                    for m in reg["models"]["models"]
                ],
                "pooling": reg["models"]["pooling"],
            },
            "memory_tools_sha256": reg["memory_tools"]["sha256"],
            "environment_manifest": {
                "sha256": reg["environment_manifest"]["sha256"],
                "harness": reg["environment_manifest"]["harness"],
                "harness_version": reg["environment_manifest"]["harness_version"],
                "repo_commit": reg["environment_manifest"]["repo_commit"],
                "hardware_class": reg["environment_manifest"]["hardware_class"],
            },
            "limits": reg["limits"],
            "analysis": reg["analysis"],
            "stopping_rule": reg["stopping_rule"],
        }

    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "run_id": run_id,
        "split": effective_split,
        "includes_held_out": includes_held_out,
        "task_manifest_sha256": task_manifest_sha,
        "registration": reg_summary,
        "stopping_status": stopping_status,
        "grades_status": {
            "grades_final": store.is_grades_final,
            "grades_final_sha256": grades_final_sha,
            "grades_sha256": grades_sha,
            "finalized_at": (
                grades_final_doc.get("finalized_at") if grades_final_doc is not None else None
            ),
            "total_grade_revisions": total_grade_revisions,
            "graded_trials_in_split": sum(1 for t in selected_trials if t.graded),
            "finished_trials_in_split": sum(1 for t in selected_trials if t.has_record),
            "interrupted_without_record_in_run": interrupted_without_record,
        },
        "task_population": {
            "included_tasks_in_split": len(split_task_ids),
            "families_in_split": len(split_families),
            "strata_in_split": strata_names,
            "excluded_count": len(excluded_rows),
            "excluded_tasks": excluded_rows,
            "resolved_pairs": len(resolved_pairs),
            "superseded_retry_trials": len(superseded_ids),
        },
        "models": {
            "observed": observed_model_keys,
            "multi_model": multi_model,
            "pooling": reg["models"]["pooling"] if reg is not None else None,
            "single_model_limitation": single_model_limitation,
        },
        "analysis_config": {
            "bootstrap_seed": eff_seed,
            "bootstrap_resamples": eff_resamples,
            "confidence": confidence,
            "tail_quantiles": list(DEFAULT_TAIL_QUANTILES),
            "registered_exploratory_slices": list(registered_exploratory),
        },
        "primary_endpoint": {
            "registered_aggregate": aggregate_summary,
            "by_stratum": by_stratum,
            "by_model": by_model,
        },
        "exploratory_slices": exploratory_slices,
        "secondary_measures": secondary_measures,
        "failure_categories": failure_categories,
        "arm_leaked": arm_leaked,
        "deviations": deviations,
        "memory_set": memory_set_summary,
    }

    json_text = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    markdown = render_report_markdown(report)

    if write:
        md_sha, json_sha = store.write_report(markdown, json_text)
        report_path: Path | None = store.report_path
        report_json_path: Path | None = store.report_json_path
    else:
        md_sha = sha256_bytes(markdown.encode("utf-8"))
        json_sha = sha256_bytes(json_text.encode("utf-8"))
        report_path = None
        report_json_path = None

    return ReportResult(
        report=report,
        markdown=markdown,
        json_text=json_text,
        markdown_sha256=md_sha,
        json_sha256=json_sha,
        report_path=report_path,
        report_json_path=report_json_path,
    )


# ---------------------------------------------------------------------------
# Deterministic Markdown rendering
# ---------------------------------------------------------------------------


def _fmt_pct(val: float | None) -> str:
    return "n/a" if val is None else f"{val * 100.0:.1f}%"


def _fmt_signed_pct(val: float | None) -> str:
    return "n/a" if val is None else f"{val * 100.0:+.1f} pp"


def _fmt_pct_ci(est: Mapping[str, Any]) -> str:
    val = est.get("estimate")
    ci = est.get("interval")
    if val is None:
        return "n/a"
    if isinstance(ci, Mapping):
        return f"{val * 100.0:.1f}% [{ci['low'] * 100.0:.1f}%, {ci['high'] * 100.0:.1f}%]"
    return f"{val * 100.0:.1f}%"


def _fmt_signed_pct_ci(est: Mapping[str, Any]) -> str:
    val = est.get("estimate")
    ci = est.get("interval")
    if val is None:
        return "n/a"
    if isinstance(ci, Mapping):
        return f"{val * 100.0:+.1f} pp [{ci['low'] * 100.0:+.1f}, {ci['high'] * 100.0:+.1f}]"
    return f"{val * 100.0:+.1f} pp"


def _fmt_wilson(wilson: Mapping[str, Any] | None) -> str:
    if not isinstance(wilson, Mapping):
        return "n/a"
    return f"[{wilson['low'] * 100.0:.1f}%, {wilson['high'] * 100.0:.1f}%]"


def _fmt_num_ci(est: Mapping[str, Any], *, decimals: int = 2, signed: bool = False) -> str:
    val = est.get("estimate")
    ci = est.get("interval")
    if val is None:
        return "n/a"
    fmt = f"{{:{'+' if signed else ''}.{decimals}f}}"
    s_val = fmt.format(val)
    if isinstance(ci, Mapping):
        return f"{s_val} [{fmt.format(ci['low'])}, {fmt.format(ci['high'])}]"
    return s_val


def _fmt_p(val: float | None) -> str:
    return "n/a" if val is None else f"{val:.4f}"


def _primary_row(label: str, s: Mapping[str, Any]) -> str:
    tag = " [EXPLORATORY]" if s.get("exploratory") else ""
    cl = s["clustering"]
    mc = s["exact_mcnemar"]
    fs = s["exact_family_sign"]
    mem_col = (
        f"{s['memory_successes']}/{s['n_pairs']} ({_fmt_pct(s['memory_rate'])})"
        if s["n_pairs"] > 0
        else "0/0 (n/a)"
    )
    nomem_col = (
        f"{s['no_memory_successes']}/{s['n_pairs']} ({_fmt_pct(s['no_memory_rate'])})"
        if s["n_pairs"] > 0
        else "0/0 (n/a)"
    )
    return (
        f"| {label}{tag} | {s['n_pairs']} ({s['n_missing_pairs']} missing) "
        f"| {cl['n_families']} ({cl['effective_sample_size']:.2f}) "
        f"| {mem_col} | {nomem_col} "
        f"| {_fmt_signed_pct_ci(s['difference'])} "
        f"| {mc['memory_only']}/{mc['no_memory_only']} (p={_fmt_p(mc['p_value'])}) "
        f"| p={_fmt_p(fs['p_value'])} |"
    )


def _rate_row(label: str, r: Mapping[str, Any]) -> str:
    denom = r["denominator"]
    num = r["numerator"]
    missing = r["n_missing"]
    denom_text = f"{num}/{denom} ({r['denominator_label']})"
    if missing:
        denom_text += f"; {missing} missing"
    return f"| {label} | {denom_text} | {_fmt_pct_ci(r['rate'])} | {_fmt_wilson(r['wilson'])} |"


def _count_rate_row(label: str, r: Mapping[str, Any]) -> str:
    denom = r["denominator"]
    num = r["numerator"]
    rate = _fmt_pct(r["rate"])
    return (
        f"| {label} | {num}/{denom} ({r['denominator_label']}) "
        f"| {rate} | {_fmt_wilson(r['wilson'])} |"
    )


def _numeric_section_lines(title: str, s: Mapping[str, Any], *, decimals: int = 2) -> list[str]:
    lines = [
        f"#### {title}",
        "",
        (
            f"- Pairs: {s['n_pairs']} complete / {s['n_total_pairs']} total "
            f"(missing pairs: {s['n_missing_pairs']}; missing memory: {s['n_missing_memory']}, "
            f"missing no-memory: {s['n_missing_no_memory']}; "
            f"families: {s['clustering']['n_families']}, "
            f"N_eff: {s['clustering']['effective_sample_size']:.2f})."
        ),
        (
            f"- Memory arm: mean {_fmt_num_ci(s['memory_arm']['mean'], decimals=decimals)}, "
            f"median {_fmt_num_ci(s['memory_arm']['median'], decimals=decimals)}, "
            f"p90 {_fmt_num_ci(s['memory_arm']['quantiles']['p90'], decimals=decimals)}, "
            f"p95 {_fmt_num_ci(s['memory_arm']['quantiles']['p95'], decimals=decimals)}."
        ),
        (
            f"- No-memory arm: mean {_fmt_num_ci(s['no_memory_arm']['mean'], decimals=decimals)}, "
            f"median {_fmt_num_ci(s['no_memory_arm']['median'], decimals=decimals)}, "
            f"p90 {_fmt_num_ci(s['no_memory_arm']['quantiles']['p90'], decimals=decimals)}, "
            f"p95 {_fmt_num_ci(s['no_memory_arm']['quantiles']['p95'], decimals=decimals)}."
        ),
        (
            f"- Paired difference (memory - no_memory): "
            f"mean {_fmt_num_ci(s['difference_mean'], decimals=decimals, signed=True)}, "
            f"median {_fmt_num_ci(s['difference_median'], decimals=decimals, signed=True)}, "
            f"p90 {_fmt_num_ci(s['difference_quantiles']['p90'], decimals=decimals, signed=True)}, "
            f"p95 {_fmt_num_ci(s['difference_quantiles']['p95'], decimals=decimals, signed=True)}."
        ),
        (
            f"- Paired ratio (memory / no_memory): "
            f"ratio of means {_fmt_num_ci(s['ratio_of_means'], decimals=3)}, "
            f"per-pair median {_fmt_num_ci(s['ratio_median'], decimals=3)}, "
            f"p90 {_fmt_num_ci(s['ratio_quantiles']['p90'], decimals=3)}, "
            f"p95 {_fmt_num_ci(s['ratio_quantiles']['p95'], decimals=3)} "
            f"(ratio pairs: {s['n_ratio_pairs']}, excluded non-positive denom: "
            f"{s['n_excluded_ratio_pairs']})."
        ),
        "",
    ]
    return lines


def render_report_markdown(report: Mapping[str, Any]) -> str:
    """Render a deterministic Markdown local aggregate report from ``report``."""
    lines: list[str] = [
        f"# Usefulness Experiment Report: {report['run_id']}",
        "",
        "Local aggregate report. Contains no task content, transcript text, or memory text.",
        "",
        "## Run Provenance and Scope",
        "",
        f"- Run ID: `{report['run_id']}`",
        f"- Split: `{report['split']}` (includes held-out: `{report['includes_held_out']}`)",
        f"- Task manifest SHA-256: `{report['task_manifest_sha256']}`",
    ]

    reg = report.get("registration")
    if isinstance(reg, Mapping):
        pe = reg["primary_endpoint"]
        env = reg["environment_manifest"]
        lines.extend(
            [
                f"- Registration SHA-256: `{reg['registration_sha256']}` "
                f"(frozen at `{reg['created_at']}`)",
                f"- Primary endpoint: `{pe['endpoint']}` "
                f"(practical benefit threshold: `{pe['practical_benefit_threshold']:.3f}`)",
                f"- Environment: `{env['harness']}` `{env['harness_version']}`, "
                f"repo `{env['repo_commit']}`, hardware `{env['hardware_class']}` "
                f"(manifest SHA-256 `{env['sha256']}`)",
                f"- Memory tools wording SHA-256: `{reg['memory_tools_sha256']}`",
                f"- Model pooling policy: {reg['models']['pooling']}",
                f"- Sequential error control: {reg['stopping_rule']['sequential_error_control']}",
            ]
        )
    else:
        lines.append("- Registration: none (unregistered calibration run)")

    stop = report["stopping_status"]
    gs = report["grades_status"]
    pop = report["task_population"]
    ac = report["analysis_config"]
    models_info = report["models"]

    lines.extend(
        [
            f"- Stopping status: `{stop['stopped_by']}` ({stop['detail']})",
            f"- Grades: final=`{gs['grades_final']}` "
            f"(graded={gs['graded_trials_in_split']}/{gs['finished_trials_in_split']}, "
            f"revisions={gs['total_grade_revisions']}, "
            f"interrupted_without_record={gs['interrupted_without_record_in_run']})"
            + (
                f", grades-final SHA-256 `{gs['grades_final_sha256']}`"
                if gs.get("grades_final_sha256")
                else ""
            ),
            f"- Task population in split: {pop['included_tasks_in_split']} tasks across "
            f"{pop['families_in_split']} families, {pop['resolved_pairs']} resolved pairs "
            f"(superseded retry trials: {pop['superseded_retry_trials']}, "
            f"excluded tasks: {pop['excluded_count']})",
            f"- Models observed: {', '.join(f'`{m}`' for m in models_info['observed'])}",
            f"- Bootstrap: seed=`{ac['bootstrap_seed']}`, "
            f"resamples=`{ac['bootstrap_resamples']}`, "
            f"confidence=`{ac['confidence']:.2f}`",
        ]
    )
    if models_info.get("single_model_limitation"):
        lines.append(f"- Limitation: {models_info['single_model_limitation']}")
    lines.append("")

    if pop["excluded_tasks"]:
        lines.extend(["### Excluded Tasks", "", "| Task ID | Reason Code |", "|---|---|"])
        for ex in pop["excluded_tasks"]:
            lines.append(f"| `{ex['task_id']}` | `{ex['reason_code']}` |")
        lines.append("")

    # Primary endpoint
    pe_sec = report["primary_endpoint"]
    header = (
        "| Slice | Pairs | Families (N_eff) | Memory arm | No-memory arm "
        "| Paired diff (95% CI) | McNemar (M/N) | Family sign |"
    )
    sep = "|---|---|---|---|---|---|---|---|"

    lines.extend(
        [
            "## Primary Endpoint (Paired Task Success)",
            "",
            header,
            sep,
            _primary_row("Registered aggregate", pe_sec["registered_aggregate"]),
            "",
        ]
    )

    agg = pe_sec["registered_aggregate"]
    if agg.get("practical_benefit_threshold") is not None:
        thresh_s = _fmt_signed_pct(agg["practical_benefit_threshold"])
        lines.append(
            f"- Practical benefit threshold ({thresh_s}): "
            f"point estimate meets threshold = `{agg['exceeds_practical_benefit_threshold']}`, "
            f"lower CI meets threshold = `{agg['ci_exceeds_practical_benefit_threshold']}`."
        )
    lines.append(f"- Note: {agg['note']}")
    lines.append("")

    # By stratum
    lines.extend(["### By Stratum", "", header, sep])
    for stratum, s_data in pe_sec["by_stratum"].items():
        lines.append(_primary_row(f"Stratum: `{stratum}`", s_data))
    lines.append("")

    # By model (only when >1 model)
    if pe_sec["by_model"]:
        lines.extend(["### By Model", "", header, sep])
        for model_key, m_data in pe_sec["by_model"].items():
            lines.append(_primary_row(f"Model: `{model_key}`", m_data))
        lines.append("")

    # Exploratory slices
    lines.extend(
        [
            "## Exploratory Slices [EXPLORATORY]",
            "",
            "Every slice below is exploratory and not part of the registered primary test.",
            "",
            header,
            sep,
        ]
    )
    for slice_name, s_data in report["exploratory_slices"].items():
        lines.append(_primary_row(f"`{slice_name}`", s_data))
    lines.append("")
    for slice_name, s_data in report["exploratory_slices"].items():
        if s_data.get("note"):
            lines.append(f"- `{slice_name}` [EXPLORATORY]: {s_data['note']}")
    lines.append("")

    # Secondary measures
    sec = report["secondary_measures"]
    lines.extend(
        [
            "## Secondary Measures",
            "",
            "Every rate reports its denominator, family-clustered bootstrap interval, "
            "and Wilson score interval.",
        ]
    )
    if isinstance(reg, Mapping) and reg.get("analysis", {}).get("secondary_measures_handling"):
        lines.append(
            f"- Registered secondary-measures handling: "
            f"{reg['analysis']['secondary_measures_handling']}"
        )
    lines.extend(
        [
            "",
            "### Proportion Measures",
            "",
            "| Measure | Count / Denominator | Clustered rate (95% CI) | Wilson 95% CI |",
            "|---|---|---|---|",
            _rate_row("Repeat-error rate (memory arm)", sec["repeat_error_rate"]["memory_arm"]),
            _rate_row(
                "Repeat-error rate (no-memory arm)", sec["repeat_error_rate"]["no_memory_arm"]
            ),
            _rate_row("Harmful memory use (combined)", sec["harmful_memory_use"]["combined"]),
            _rate_row(
                "Harmful memory use (wrong action)", sec["harmful_memory_use"]["wrong_action"]
            ),
            _rate_row(
                "Attempted retired/expired/out-of-scope memory use",
                sec["harmful_memory_use"]["attempted_invalid_memory_use"],
            ),
            _rate_row(
                "Confident wrong reply linked to memory (among memory use)",
                sec["confident_wrong_reply"]["memory_linked_among_memory_use"],
            ),
            _rate_row(
                "Confident wrong reply (all memory-arm trials)",
                sec["confident_wrong_reply"]["all_trials_memory_arm"],
            ),
            _rate_row(
                "Confident wrong reply (all no-memory-arm trials)",
                sec["confident_wrong_reply"]["all_trials_no_memory_arm"],
            ),
            _rate_row(
                "Confident wrong reply without memory link (memory arm)",
                sec["confident_wrong_reply"]["without_memory_link_memory_arm"],
            ),
            _rate_row(
                "Confident wrong reply without memory link (no-memory arm)",
                sec["confident_wrong_reply"]["without_memory_link_no_memory_arm"],
            ),
            _rate_row(
                "Stated failures (memory arm)",
                sec["confident_wrong_reply"]["stated_failures_memory_arm"],
            ),
            _rate_row(
                "Stated failures (no-memory arm)",
                sec["confident_wrong_reply"]["stated_failures_no_memory_arm"],
            ),
            _rate_row(
                "Recall: made no recall",
                sec["recall_and_retrieval"]["made_no_recall"],
            ),
            _rate_row(
                "Recall: trial with returned memories",
                sec["recall_and_retrieval"]["with_returned_memories"],
            ),
            _rate_row(
                "Recall: trial with miss",
                sec["recall_and_retrieval"]["with_miss"],
            ),
            _rate_row(
                "Recall: trial with degraded recall",
                sec["recall_and_retrieval"]["with_degraded"],
            ),
            _rate_row(
                "Recall: trial with missing record",
                sec["recall_and_retrieval"]["with_missing_record"],
            ),
            _rate_row(
                "Citations: trials with citation",
                sec["citations"]["trials_with_citation"],
            ),
            _rate_row(
                "Citations: trials citing returned memory",
                sec["citations"]["trials_citing_returned_memory"],
            ),
            _count_rate_row(
                "Citations: returned memories declared used",
                sec["citations"]["returned_memories_cited_rate"],
            ),
            _count_rate_row(
                "Citations: placeholders with no binding",
                sec["citations"]["unbound_placeholder_rate"],
            ),
            _count_rate_row(
                "Citations: replies bound to one citation only",
                sec["citations"]["single_citation_reply_rate"],
            ),
            _rate_row(
                "Abstention: appropriate non-use",
                sec["abstention"]["appropriate_non_use"],
            ),
            _rate_row(
                "Abstention: unnecessary recall",
                sec["abstention"]["unnecessary_recall"],
            ),
            _rate_row(
                "Abstention: unnecessary or harmful use",
                sec["abstention"]["unnecessary_or_harmful_use"],
            ),
            _rate_row(
                "Abstention: harmful memory use",
                sec["abstention"]["harmful_use"],
            ),
            "",
        ]
    )

    rc = sec["recall_and_retrieval"]["call_totals"]
    cc = sec["citations"]["counts"]
    lines.extend(
        [
            (
                f"- Recall calls (resolved memory-arm trials): total={rc['total_recalls']} "
                f"(returned={rc['returned']}, miss={rc['miss']}, degraded={rc['degraded']}, "
                f"missing_record={rc['missing_record']})."
            ),
            (
                f"- Citation counts: citations_made={cc['citations_made']}, "
                f"cite_calls_in_transcript={cc['cite_calls_in_transcript']}, "
                f"degraded_cite_calls={cc['degraded_cite_calls']}, "
                f"returned_memories_cited={cc['returned_memories_cited_count']}/"
                f"{cc['returned_memories_total']}, "
                f"unbound_placeholders={cc['unbound_placeholder_count']}/"
                f"{cc['placeholder_count']}, "
                f"single_citation_replies={cc['single_citation_reply_count']}/"
                f"{cc['bound_reply_count']} "
                f"(multi_citation_replies={cc['multi_citation_reply_count']}). "
                f"{sec['citations']['note']}"
            ),
            (
                "- Harmful memory use attribution note: "
                f"{sec['harmful_memory_use']['passing_trials_not_opened_in_attribution']} "
                "passing memory-arm trial(s) were not opened in post-grading attribution and "
                "counted as non-harmful; unattributed failed trials are counted as missing."
            ),
            "",
            "### Tokens and Wall Time",
            "",
        ]
    )

    lines.extend(
        _numeric_section_lines("Total Tokens", sec["total_tokens"]["total_tokens"], decimals=1)
    )
    if "total_tokens_including_retries" in sec["total_tokens"]:
        lines.extend(
            _numeric_section_lines(
                "Total Tokens (Including Superseded Retries)",
                sec["total_tokens"]["total_tokens_including_retries"],
                decimals=1,
            )
        )
    lines.extend(_numeric_section_lines("End-to-End Wall Time (s)", sec["latency"]["total_s"]))
    if "total_s_including_retries" in sec["latency"]:
        lines.extend(
            _numeric_section_lines(
                "End-to-End Wall Time Including Retries (s)",
                sec["latency"]["total_s_including_retries"],
            )
        )
    lines.extend(
        _numeric_section_lines("Harness Execution Wall Time (s)", sec["latency"]["harness_s"])
    )

    ret_dist = sec["latency"]["retrieval_s_memory_arm"]
    retry_info = sec["latency"]["retry_time_s"]
    lines.extend(
        [
            (
                f"- Memory-arm retrieval latency (s): observed={ret_dist['n_observed']}/"
                f"{ret_dist['n_total']} (missing={ret_dist['n_missing']}), "
                f"mean {_fmt_num_ci(ret_dist['mean'], decimals=4)}, "
                f"median {_fmt_num_ci(ret_dist['median'], decimals=4)}, "
                f"p90 {_fmt_num_ci(ret_dist['quantiles']['p90'], decimals=4)}, "
                f"p95 {_fmt_num_ci(ret_dist['quantiles']['p95'], decimals=4)}."
            ),
            (
                f"- Retry wall time (s): total={retry_info['total_s']:.2f}s "
                f"(memory={retry_info['memory_arm_total_s']:.2f}s, "
                f"no_memory={retry_info['no_memory_arm_total_s']:.2f}s, "
                f"superseded_trials={retry_info['superseded_trial_count']})."
            ),
            f"- Itemization note: {sec['latency']['itemization_note']}",
            "",
        ]
    )

    # Non-ordinary failure categories
    fc = report["failure_categories"]
    lines.extend(
        [
            "## Failure Categories (Apart from Ordinary Wrong Answers)",
            "",
            f"- Graded ordinary wrong answers (for reconciliation): "
            f"total={fc['graded_ordinary_wrong_answer']['total']} "
            f"(memory={fc['graded_ordinary_wrong_answer']['memory']}, "
            f"no_memory={fc['graded_ordinary_wrong_answer']['no_memory']}).",
            f"- Graded non-ordinary failures: total={fc['graded_non_ordinary_total']}.",
            "",
            "| Graded failure category | Total | Memory | No-memory | Trial IDs |",
            "|---|---|---|---|---|",
        ]
    )
    for cat, data in fc["graded_non_ordinary"].items():
        ids = ", ".join(f"`{t['trial_id']}`" for t in data["trials"]) or "-"
        lines.append(
            f"| `{cat}` | {data['total']} | {data['memory']} | {data['no_memory']} | {ids} |"
        )
    lines.extend(
        [
            "",
            (
                "- Execution-level (runner/interrupted) failures: "
                f"total={fc['execution_failures_total']}."
            ),
            "",
            "| Execution failure category | Total | Memory | No-memory | Trial IDs |",
            "|---|---|---|---|---|",
        ]
    )
    for cat, data in fc["execution_failures"].items():
        ids = ", ".join(f"`{t['trial_id']}`" for t in data["trials"]) or "-"
        lines.append(
            f"| `{cat}` | {data['total']} | {data['memory']} | {data['no_memory']} | {ids} |"
        )
    lines.append("")

    # Arm-leaked trials
    al = report["arm_leaked"]
    lines.extend(
        [
            "## Trials Whose Material Gave the Arm Away",
            "",
            (
                f"- Count: {al['total']} "
                f"(memory={al['memory']}, no_memory={al['no_memory']}); "
                f"rate={_fmt_pct_ci(al['rate']['rate'])} "
                f"(Wilson {_fmt_wilson(al['rate']['wilson'])}, "
                f"denominator={al['rate']['denominator']} graded trials)."
            ),
            "",
        ]
    )
    if al["trials"]:
        lines.extend(
            [
                "| Trial ID | Task ID | Arm | Model | Met checks | Failure category |",
                "|---|---|---|---|---|---|",
            ]
        )
        for t in al["trials"]:
            lines.append(
                f"| `{t['trial_id']}` | `{t['task_id']}` | `{t['arm']}` "
                f"| `{t['model']}` | `{t['met_checks']}` | `{t['failure_category'] or '-'}` |"
            )
        lines.append("")
    else:
        lines.extend(["No trials were flagged as giving the arm away.", ""])

    # Deviations
    dev = report["deviations"]
    lines.extend(["## Deviations from Registration", ""])
    if dev["items"]:
        lines.extend(["| Date | Reason | Metadata |", "|---|---|---|"])
        for item in dev["items"]:
            reason = str(item.get("reason", "")).replace("|", "\\|").replace("\n", " ")
            meta = item.get("metadata")
            meta_str = (
                "`" + json.dumps(meta, ensure_ascii=False, sort_keys=True).replace("|", "\\|") + "`"
                if isinstance(meta, Mapping) and meta
                else "-"
            )
            lines.append(f"| `{item.get('recorded_at', '')}` | {reason} | {meta_str} |")
        lines.append("")
    else:
        lines.extend(["0 deviations recorded.", ""])

    # Memory-set validity changes
    lines.extend(["## Memory-Set Validity Boundary Limitation", ""])
    ms = report.get("memory_set")
    if isinstance(ms, Mapping):
        vc = ms["validity_changes"]
        mc_counts = ms["memory_counts"]
        lines.extend(
            [
                f"- Memory set SHA-256: `{ms['memory_set_sha256']}` "
                f"(migration level `{ms['migration_level']}`)",
                f"- Window: `{vc['cutoff']}` to `{vc['last_day']}`",
                f"- Memories whose status could change during the run: "
                f"**{vc['total_crossing']}** total crossing "
                f"(starts in window: {vc['starts_in_window']}, "
                f"ends in window: {vc['ends_in_window']}, "
                f"non-terminal crossing: {vc['non_terminal_crossing']}).",
                f"- Snapshot memory counts: total={mc_counts['total']}, "
                f"by_status={json.dumps(mc_counts['by_status'], sort_keys=True)}.",
                "",
            ]
        )
    else:
        lines.extend(["No memory-set manifest present for this calibration run.", ""])

    return "\n".join(lines).rstrip("\n") + "\n"
