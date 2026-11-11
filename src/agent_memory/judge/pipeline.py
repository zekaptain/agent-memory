"""The judge pipeline: import transcripts, pick candidates, draft, decide, and the live loop.

Each step is resumable and writes JSONL under the run directory. Verdicts also land in the
experiment ledger as `decision` events so the ledger itself can answer "what did the judge
say about this turn, with which model, at what cost". State text never enters those events.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb

from ..adapters.base import UnsupportedTranscript
from ..adapters.droid import DroidAdapter
from ..capture.pipeline import deliver, make_redactor
from ..capture.record import Record, SessionInfo, now
from ..capture.writer import Writer
from ..config import Config
from ..db import AgentMemoryError, MemoryClient, sql_literal
from .backends import Answer, DecisionBackend, JudgeInput, make_backend
from .drafter import DRAFTER_VERSION, draft_hash, make_drafter
from .labeling import LABELS_VERSION
from .lesson import lesson_from_row
from .projection import PROJECTION_VERSION, drafter_rendering, project_candidate
from .provenance import PROVENANCE_VERSION, session_callers
from .questions import (
    DEFAULT_THRESHOLDS,
    QUESTIONS_VERSION,
    STAGE_A,
    aggregate_claims,
    derive_action,
    gate_probabilities,
    stage_b_for,
    thresholds_from_run,
)
from .store import JudgeError, RunStore, ensure_experiment_home, imports_log, utc_now_iso
from .triggers import (
    CALLER_KINDS,
    STRATUM_PRIORITY,
    TRIGGERS_VERSION,
    Candidate,
    candidates_from_turns,
    fetch_turn_events,
    sample_candidates,
    segment_turns,
)

Log = Callable[[str], None]
"""`log` takes progress lines; `warn` takes problems the user has to act on (a skipped item, a
spooled batch). A caller with one channel passes only `log` and gets both on it."""
JUDGE_AGENT = "judge"
CALIBRATION_SHARE = 0.6
EVENT_BATCH = 25


def _quiet(_: str) -> None:
    return None


def require_judge_agent(cfg: Config) -> None:
    """Fail before model calls or imports when an experiment home needs migration."""
    with MemoryClient.from_config(cfg) as client:
        _, rows = client.query(
            f"SELECT agent_id FROM agents WHERE agent_id = {sql_literal(JUDGE_AGENT)}"
        )
    if not rows:
        raise JudgeError(f"run mem --home {cfg.home} init")


def require_current_questions(run: Mapping[str, Any]) -> None:
    """Drafting and judging write rows in the current question set; a run built under an older
    one keeps its verdicts and labels as they are and is read-only from here on."""
    version = run.get("questions_version")
    if version != QUESTIONS_VERSION:
        raise JudgeError(
            f"run {run.get('run_id')!r} was built with {version}; this code drafts and judges "
            f"with {QUESTIONS_VERSION}. Its verdicts and labels stay as they are: start a new run."
        )


# Import


@dataclass
class ImportSummary:
    files: int = 0
    skipped_files: int = 0
    records: int = 0
    inserted: int = 0
    skipped_rows: int = 0
    errors: list[str] = field(default_factory=list)


def iter_transcripts(
    sessions_dir: Path, *, project: str | None = None, since: datetime | None = None
) -> list[Path]:
    """Droid keeps `<sessions_dir>/<project slug>/<session id>.jsonl`."""
    root = sessions_dir.expanduser()
    if not root.is_dir():
        raise JudgeError(f"{root} is not a directory")
    out = []
    for path in sorted(root.rglob("*.jsonl")):
        if project and path.parent.name != project:
            continue
        if since is not None and datetime.fromtimestamp(path.stat().st_mtime, UTC) < since:
            continue
        out.append(path)
    return out


def import_sessions(
    cfg: Config,
    sessions_dir: Path,
    *,
    limit: int | None = None,
    project: str | None = None,
    since: datetime | None = None,
    log: Log = _quiet,
) -> ImportSummary:
    """Write every transcript straight to the experiment ledger. Unchanged files are skipped."""
    ensure_experiment_home(cfg)
    summary = ImportSummary()
    paths = iter_transcripts(sessions_dir, project=project, since=since)
    seen = {(row["path"], row["mtime"]) for row in RunStore.read_jsonl(imports_log(cfg))}
    adapter = DroidAdapter()
    with MemoryClient.from_config(cfg) as client:
        writer = Writer(client, cfg.blob_dir, make_redactor(cfg))
        for path in paths:
            if limit is not None and summary.files >= limit:
                break
            mtime = path.stat().st_mtime
            if (str(path), mtime) in seen:
                summary.skipped_files += 1
                continue
            try:
                records = adapter.from_transcript(path)
            except UnsupportedTranscript:
                summary.skipped_files += 1
                continue
            if not records:
                summary.skipped_files += 1
                continue
            try:
                result = writer.write(records)
            except (AgentMemoryError, duckdb.Error) as exc:
                # Reported once, grouped, by the caller; a line per file would repeat it.
                summary.errors.append(f"{path.name}: {type(exc).__name__}: {exc}")
                continue
            summary.records += len(records)
            summary.inserted += result.inserted
            summary.skipped_rows += result.skipped
            for warning in result.warnings:
                log(f"{path.name}: {warning}")
            if result.rejected:
                # Left out of the imports log, so the next import tries the file again.
                summary.errors.append(
                    f"{path.name}: {len(result.rejected)} of {len(records)} record(s) rejected "
                    f"(the rest were written): {result.rejected[0].reason}"
                )
                continue
            summary.files += 1
            RunStore.write_jsonl(
                imports_log(cfg),
                [
                    {
                        "path": str(path),
                        "mtime": mtime,
                        "records": len(records),
                        "inserted": result.inserted,
                        "imported_at": utc_now_iso(),
                    }
                ],
                append=True,
            )
            log(f"imported {path.parent.name}/{path.name}: {result.inserted} rows")
    return summary


# Candidates


@dataclass
class CandidateSummary:
    pool: dict[str, int]
    chosen: dict[str, int]
    items: int
    redactions: int
    pool_callers: dict[str, int] = field(default_factory=dict)
    chosen_callers: dict[str, int] = field(default_factory=dict)
    harness_turns: int = 0


def assign_splits(
    candidates: Sequence[Candidate], *, seed: int, calibration_share: float = CALIBRATION_SHARE
) -> dict[str, str]:
    """Calibration or held-out, stratified by trigger stratum, seeded."""
    rng = random.Random(seed * 7919 + 1)
    split: dict[str, str] = {}
    for stratum in STRATUM_PRIORITY:
        for caller in CALLER_KINDS:
            group = sorted(
                (c for c in candidates if c.stratum == stratum and c.caller == caller),
                key=lambda c: c.item_id,
            )
            rng.shuffle(group)
            n_cal = round(len(group) * calibration_share)
            for i, c in enumerate(group):
                split[c.item_id] = "calibration" if i < n_cal else "held_out"
    return split


def _session_ids_since(client: MemoryClient, since: datetime) -> list[str]:
    stamp = sql_literal(since.astimezone(UTC).isoformat())
    _, rows = client.query(
        "SELECT session_id::VARCHAR FROM sessions "
        f"WHERE agent_id = 'factory-droid' AND started_at >= {stamp}::TIMESTAMPTZ"
    )
    return [str(r[0]) for r in rows]


def build_candidates(
    cfg: Config,
    run_id: str,
    *,
    n: int | None = 200,
    seed: int = 1,
    clean_share: float = 0.25,
    human_share: float = 0.5,
    store: RunStore | None = None,
    session_ids: Sequence[str] | None = None,
    exclude_item_ids: Sequence[str] = (),
    extra_params: dict[str, Any] | None = None,
    log: Log = _quiet,
) -> CandidateSummary:
    """Segment turns, apply triggers, sample, project, and write items.jsonl."""
    ensure_experiment_home(cfg)
    store = store or RunStore(cfg, run_id)
    if store.exists() and store.items_path.exists():
        raise JudgeError(f"run {run_id!r} already has items; use a new run id or purge it")
    redactor = make_redactor(cfg)
    excluded = set(exclude_item_ids)
    with MemoryClient.from_config(cfg) as client:
        callers = session_callers(cfg, client)
        rows = fetch_turn_events(client, session_ids=session_ids, blob_dir=cfg.blob_dir)
        turns = segment_turns(rows, callers)
        harness_turns = sum(1 for t in turns if t.opener_provenance == "harness")
        pool = [c for c in candidates_from_turns(turns) if c.item_id not in excluded]
        chosen = (
            sample_candidates(
                pool, n=n, seed=seed, clean_share=clean_share, human_share=human_share
            )
            if n is not None
            else sorted(pool, key=lambda c: c.item_id)
        )
        if not store.exists():
            store.create(
                {
                    "triggers_version": TRIGGERS_VERSION,
                    "provenance_version": PROVENANCE_VERSION,
                    "projection_version": PROJECTION_VERSION,
                    "questions_version": QUESTIONS_VERSION,
                    "labels_version": LABELS_VERSION,
                    "n": n,
                    "seed": seed,
                    "clean_share": clean_share,
                    "human_share": human_share,
                    "thresholds": dict(DEFAULT_THRESHOLDS),
                    "turns_seen": len(turns),
                    "harness_turns": harness_turns,
                    "sessions_by_caller": _count_values(c.kind for c in callers.values()),
                    "pool": _count(pool),
                    "pool_callers": _count_values(c.caller for c in pool),
                    **(extra_params or {}),
                }
            )
        splits = assign_splits(chosen, seed=seed)
        items = []
        redactions = 0
        for index, cand in enumerate(chosen, start=1):
            projection = project_candidate(client, cand, redactor, cfg.blob_dir)
            store.write_projection(cand.item_id, projection.text)
            redactions += projection.redactions
            items.append(
                {
                    **cand.to_dict(),
                    "split": splits[cand.item_id],
                    "projection_version": PROJECTION_VERSION,
                    "projection_hash": projection.hash,
                    "projection_chars": len(projection.text),
                    "redactions": projection.redactions,
                    "sidecar": projection.sidecar,
                }
            )
            if index % 25 == 0:
                log(f"projected {index}/{len(chosen)}")
    store.write_jsonl(store.items_path, items)
    return CandidateSummary(
        pool=_count(pool),
        chosen=_count(chosen),
        items=len(items),
        redactions=redactions,
        pool_callers=_count_values(c.caller for c in pool),
        chosen_callers=_count_values(c.caller for c in chosen),
        harness_turns=harness_turns,
    )


def _count(candidates: Sequence[Candidate]) -> dict[str, int]:
    counts = dict.fromkeys(STRATUM_PRIORITY, 0)
    for c in candidates:
        counts[c.stratum] += 1
    return counts


def _count_values(values: Iterable[str]) -> dict[str, int]:
    counts = dict.fromkeys(CALLER_KINDS, 0)
    for v in values:
        counts[v] = counts.get(v, 0) + 1
    return counts


# Draft


@dataclass
class DraftSummary:
    written: int = 0
    declined: int = 0
    failed: int = 0


def draft_items(
    cfg: Config,
    run_id: str,
    drafter_spec: str,
    *,
    limit: int | None = None,
    store: RunStore | None = None,
    log: Log = _quiet,
    warn: Log | None = None,
) -> DraftSummary:
    """One draft per item for this drafter; other drafters' rows are left alone. Resumable."""
    warn = warn or log
    ensure_experiment_home(cfg)
    store = store or RunStore(cfg, run_id)
    run = store.read_run()
    require_current_questions(run)
    existing = store.read_jsonl(store.drafts_path)
    versions = {str(row.get("drafter_version") or "drafter_v2") for row in existing}
    incompatible = sorted(versions - {DRAFTER_VERSION})
    if incompatible:
        raise JudgeError(
            f"run {run_id} holds {', '.join(incompatible)} drafts; {DRAFTER_VERSION} arms "
            "cannot be mixed in. Start a new run."
        )
    drafter = make_drafter(drafter_spec)
    done = {row["item_id"] for row in existing if row.get("drafter") == drafter.name}
    todo = [row for row in store.read_jsonl(store.items_path) if row["item_id"] not in done]
    if limit is not None:
        todo = todo[:limit]
    summary = DraftSummary()
    snapshot = ""
    for index, item in enumerate(todo, start=1):
        state = drafter_rendering(store.read_projection(item["item_id"]))
        try:
            draft = drafter.draft(item["item_id"], state)
        except JudgeError as exc:
            # A drafter that has never produced anything in this run is probably
            # misconfigured; once it has, a bad item is skipped and the next run retries it.
            if not done and summary.written == 0:
                raise
            summary.failed += 1
            warn(f"draft {index}/{len(todo)} failed, skipped (rerun to retry): {exc}")
            continue
        store.write_jsonl(store.drafts_path, [draft.to_dict()], append=True)
        summary.written += 1
        summary.declined += draft.declined
        snapshot = draft.model_snapshot or snapshot
        what = "declined" if draft.declined else "drafted"
        log(f"{what} {index}/{len(todo)} ({draft.latency_ms / 1000:.1f}s)")
    drafters = dict(run.get("drafters") or {})
    previous = drafters.get(drafter.name) or {}
    drafters[drafter.name] = {
        "drafter_version": DRAFTER_VERSION,
        "model_snapshot": snapshot or previous.get("model_snapshot", ""),
        "drafts": len(done) + summary.written,
        "declined": int(previous.get("declined", 0)) + summary.declined,
        "drafted_at": utc_now_iso(),
    }
    store.update_run(drafter=drafter.name, drafter_version=DRAFTER_VERSION, drafters=drafters)
    return summary


# Decide


def _judge_input(item: dict[str, Any], state: str) -> JudgeInput:
    return JudgeInput(
        item_id=item["item_id"], state=state, facts=item["facts"], flags=tuple(item["flags"])
    )


def stage_a_from_verdict(verdict: Mapping[str, Any]) -> dict[str, Answer] | None:
    """The Stage A answers a stored verdict carries, or None when it predates a question. The
    copies cost nothing, so the row they land in reports only what it paid for."""
    answers = verdict.get("answers") or {}
    if any(q.id not in answers for q in STAGE_A):
        return None
    return {
        q.id: replace(Answer.from_dict(answers[q.id]), latency_ms=0.0, cost_usd=0.0)
        for q in STAGE_A
    }


def judge_draft(
    backend: DecisionBackend,
    item: dict[str, Any],
    state: str,
    draft: Mapping[str, Any] | None,
    thresholds: dict[str, float],
    stage_a: Mapping[str, Answer],
    *,
    run_id: str,
    stage_a_reused: bool = False,
) -> dict[str, Any]:
    """One verdict row: the item's Stage A answers plus Stage B on this draft. A declined draft
    gets no Stage B and is always discarded; an item with no draft yet gets Stage A alone."""
    answers: dict[str, Answer] = dict(stage_a)
    text = draft.get("lesson") if draft else None
    lesson = lesson_from_row(draft) if draft else None
    claims = tuple(lesson.claims) if lesson else ()
    declined = draft is not None and text is None
    if text:
        answers.update(
            backend.decide(
                replace(_judge_input(item, state), draft=text, claims=claims), stage_b_for(lesson)
            )
        )
        answers.update(aggregate_claims(answers))
    probabilities = {k: a.probabilities for k, a in answers.items()}
    derived = derive_action(
        gate_probabilities(probabilities, methods={k: a.method for k, a in answers.items()}),
        thresholds,
        declined=declined,
    )
    snapshots = sorted({a.model_snapshot for a in answers.values() if a.model_snapshot})
    return {
        "run_id": run_id,
        "item_id": item["item_id"],
        "backend": backend.name,
        "model_snapshot": ", ".join(snapshots),
        "questions_version": QUESTIONS_VERSION,
        "projection_version": item.get("projection_version", PROJECTION_VERSION),
        "projection_hash": item["projection_hash"],
        "drafter": draft.get("drafter") if draft else None,
        "has_draft": bool(text),
        "claims": len(claims),
        "declined": declined,
        "draft_hash": draft_hash(text),
        "stage_a_reused": stage_a_reused,
        "answers": {k: a.to_dict() for k, a in answers.items()},
        "latency_ms": sum(a.latency_ms for a in answers.values()),
        "cost_usd": sum(a.cost_usd for a in answers.values()),
        "action": derived.action,
        "reasons": derived.reasons,
        "thresholds": thresholds,
        "decided_at": utc_now_iso(),
    }


def judge_item(
    backend: DecisionBackend,
    item: dict[str, Any],
    state: str,
    draft: Mapping[str, Any] | None,
    thresholds: dict[str, float],
    *,
    run_id: str,
) -> dict[str, Any]:
    """Stage A on the projection alone, then Stage B on the draft if there is one."""
    stage_a = backend.decide(_judge_input(item, state), STAGE_A)
    return judge_draft(backend, item, state, draft, thresholds, stage_a, run_id=run_id)


def decision_record(
    verdict: dict[str, Any],
    item: dict[str, Any],
    run_created: str,
    *,
    harness_session_id: str | None = None,
    project: str = "judge-experiment",
) -> Record:
    """The ledger copy of a verdict: distributions and provenance, never the state text."""
    answers = {
        qid: {k: a[k] for k in ("value", "probabilities", "confidence", "method")}
        for qid, a in verdict["answers"].items()
    }
    payload = {
        "judge": {
            key: verdict.get(key)
            for key in (
                "run_id",
                "item_id",
                "backend",
                "model_snapshot",
                "questions_version",
                "projection_version",
                "projection_hash",
                "drafter",
                "draft_hash",
                "declined",
                "has_draft",
                "claims",
                "stage_a_reused",
                "latency_ms",
                "cost_usd",
                "action",
                "reasons",
                "thresholds",
            )
        }
        | {
            "triggers_version": TRIGGERS_VERSION,
            "stratum": item["stratum"],
            "flags": item["flags"],
            "turn_event_id": item["turn_event_id"],
            "answers": answers,
        }
    }
    return Record(
        agent_id=JUDGE_AGENT,
        harness_session_id=harness_session_id or f"judge-{verdict['run_id']}",
        kind="decision",
        ts=now(),
        # Keyed on every input, so judging the same turn again with a draft (or a new
        # projection, or another drafter's draft) appends a new event instead of silently
        # keeping the old one.
        source_key="/".join(
            (
                verdict["backend"],
                verdict["item_id"],
                verdict["questions_version"],
                verdict["projection_hash"][:16],
                (verdict.get("draft_hash") or "nodraft")[:16],
                verdict.get("drafter") or "nodrafter",
            )
        ),
        content=f"judge {verdict['backend']}: {verdict['action']} for turn {item['turn_event_id']}",
        payload=payload,
        session=SessionInfo(
            project=project,
            scope="judge",
            started_at=datetime.fromisoformat(run_created),
            metadata={"run_id": verdict["run_id"]},
        ),
    )


def expected_verdicts(
    items: Sequence[dict[str, Any]], drafts: Mapping[str, Sequence[Mapping[str, Any]]]
) -> dict[tuple[str, str | None], tuple[str, str | None]]:
    """(item, drafter) -> (projection hash, draft hash) for every verdict row a run should hold:
    one per draft, or one draftless row for an item nothing has drafted yet."""
    expected: dict[tuple[str, str | None], tuple[str, str | None]] = {}
    for item in items:
        rows = drafts.get(item["item_id"]) or []
        if not rows:
            expected[(item["item_id"], None)] = (item["projection_hash"], None)
        for row in rows:
            # Keyed exactly as judge_draft records it, so an untagged legacy draft (no
            # drafter) still matches its own verdict instead of being judged on every run.
            key = (item["item_id"], row.get("drafter"))
            expected[key] = (item["projection_hash"], draft_hash(row.get("lesson")))
    return expected


def decide_items(
    cfg: Config,
    run_id: str,
    backend_specs: Sequence[str],
    *,
    limit: int | None = None,
    store: RunStore | None = None,
    record_events: bool = True,
    log: Log = _quiet,
    warn: Log | None = None,
) -> dict[str, int]:
    """Run each backend over every (item, draft) without a current verdict. Stage A is judged
    once per item and copied into every row of that item. Returns rows written per backend."""
    warn = warn or log
    ensure_experiment_home(cfg)
    store = store or RunStore(cfg, run_id)
    run = store.read_run()
    require_current_questions(run)
    items = store.read_jsonl(store.items_path)
    drafts = store.read_drafts()
    if record_events:
        require_judge_agent(cfg)
    expected = expected_verdicts(items, drafts)
    written: dict[str, int] = {}
    for spec in backend_specs:
        backend = make_backend(spec)
        thresholds = thresholds_from_run(run, backend.name)
        path = store.verdicts_path(backend.name)
        existing = store.read_jsonl(path)
        # A verdict is final unless its item has since gained a draft or a new projection, in
        # which case it is superseded and judged again.
        kept = [
            v
            for v in existing
            if (v["item_id"], v.get("drafter")) in expected
            and (v["projection_hash"], v.get("draft_hash"))
            == expected[(v["item_id"], v.get("drafter"))]
        ]
        if len(kept) != len(existing):
            store.write_jsonl(path, kept)
        done = {(v["item_id"], v.get("drafter")) for v in kept}
        stage_a_cache: dict[str, Mapping[str, Answer]] = {}
        for v in kept:
            copied = stage_a_from_verdict(v)
            if copied is not None and v["item_id"] not in stage_a_cache:
                stage_a_cache[v["item_id"]] = copied
        pending_items = {key[0] for key in expected if key not in done}
        todo = [i for i in items if i["item_id"] in pending_items]
        if limit is not None:
            todo = todo[:limit]
        pending_records: list[Record] = []
        count = 0
        failed = 0
        for index, item in enumerate(todo, start=1):
            state = store.read_projection(item["item_id"])
            waiting: list[Mapping[str, Any] | None] = [
                d
                for d in drafts.get(item["item_id"]) or []
                if (item["item_id"], d.get("drafter")) not in done
            ]
            if not drafts.get(item["item_id"]):
                waiting = [None]
            stage_a = stage_a_cache.get(item["item_id"])
            reused = stage_a is not None
            try:
                if stage_a is None:
                    stage_a = backend.decide(_judge_input(item, state), STAGE_A)
            except JudgeError as exc:
                # A backend that has never produced a verdict in this run is probably
                # misconfigured, so fail fast. Once it has, a bad item is skipped and the
                # next decide picks it up again.
                if not kept and count == 0:
                    raise
                failed += 1
                warn(f"{backend.name}: item {item['item_id']} failed: {exc}")
                continue
            for draft in waiting:
                try:
                    verdict = judge_draft(
                        backend,
                        item,
                        state,
                        draft,
                        thresholds,
                        stage_a,
                        run_id=run_id,
                        stage_a_reused=reused,
                    )
                except JudgeError as exc:
                    if not kept and count == 0:
                        raise
                    failed += 1
                    who = f" draft {draft.get('drafter')}" if draft else ""
                    warn(f"{backend.name}: item {item['item_id']}{who} failed: {exc}")
                    continue
                store.write_jsonl(path, [verdict], append=True)
                count += 1
                # Later drafts of this item copy these answers instead of paying again.
                stage_a = stage_a_from_verdict(verdict) or stage_a
                stage_a_cache[item["item_id"]] = stage_a
                reused = True
                if record_events:
                    pending_records.append(decision_record(verdict, item, run["created_at"]))
                    if len(pending_records) >= EVENT_BATCH:
                        _record(cfg, pending_records, warn)
                        pending_records = []
                who = f" [{draft.get('drafter')}]" if draft else ""
                log(
                    f"{backend.name} {index}/{len(todo)}{who}: {verdict['action']} "
                    f"({verdict['latency_ms'] / 1000:.1f}s)"
                )
        if pending_records:
            _record(cfg, pending_records, warn)
        if failed:
            warn(f"{backend.name}: {failed} verdict(s) failed and were not written; rerun to retry")
        written[backend.name] = count
    return written


def _record(cfg: Config, records: list[Record], warn: Log) -> None:
    delivery = deliver(cfg, records, deadline=60.0)
    if delivery.written is None:
        warn(
            f"ledger unavailable; {len(records)} decision event(s) spooled, replayed when "
            "`mem serve` next starts on this home"
        )
    elif delivery.written.rejected:
        rejected = delivery.written.rejected
        warn(
            f"the ledger rejected {len(rejected)} decision event(s), kept in "
            f"{', '.join(str(p) for p in delivery.failed_paths)}: {rejected[0].reason}"
        )


# Live


@dataclass
class LiveSummary:
    day: str
    imported: ImportSummary
    candidates: CandidateSummary | None
    drafted: int
    decided: dict[str, int]


def live_step(
    cfg: Config,
    run_id: str,
    *,
    sessions_dir: Path,
    since: datetime,
    backends: Sequence[str],
    drafter_specs: Sequence[str] = (),
    day: str | None = None,
    log: Log = _quiet,
    warn: Log | None = None,
) -> LiveSummary:
    """One day of the live week: import new sessions, judge their turns with frozen thresholds."""
    warn = warn or log
    ensure_experiment_home(cfg)
    parent = RunStore(cfg, run_id)
    run = parent.read_run()
    require_current_questions(run)
    if not run.get("frozen"):
        raise JudgeError("thresholds are not frozen; run `mem judge report --freeze` first")
    require_judge_agent(cfg)
    day = day or datetime.now(UTC).strftime("%Y-%m-%d")
    imported = import_sessions(cfg, sessions_dir, since=since, log=log)
    child = parent.child(day)
    known = {row["item_id"] for row in parent.read_jsonl(parent.items_path)}
    for other in sorted(parent.live_dir.glob("*/items.jsonl")) if parent.live_dir.exists() else []:
        known.update(row["item_id"] for row in RunStore.read_jsonl(other))
    candidates = None
    if not child.items_path.exists():
        with MemoryClient.from_config(cfg) as client:
            session_ids = _session_ids_since(client, since)
        if not child.exists():
            child.create(
                {
                    "parent_run": run_id,
                    "live_day": day,
                    "since": since.isoformat(),
                    "frozen": run["frozen"],
                    "thresholds": run.get("thresholds", dict(DEFAULT_THRESHOLDS)),
                    "questions_version": run.get("questions_version", QUESTIONS_VERSION),
                }
            )
        candidates = build_candidates(
            cfg,
            run_id,
            n=None,
            seed=run.get("seed", 1),
            store=child,
            session_ids=session_ids,
            exclude_item_ids=sorted(known),
            log=log,
        )
    drafted = 0
    for spec in drafter_specs:
        drafted += draft_items(cfg, run_id, spec, store=child, log=log, warn=warn).written
    decided = decide_items(cfg, run_id, backends, store=child, log=log, warn=warn)
    return LiveSummary(
        day=day, imported=imported, candidates=candidates, drafted=drafted, decided=decided
    )


def purge_run(cfg: Config, run_id: str) -> int:
    """Delete every projection file for the run and its live days. Verdicts and labels stay."""
    ensure_experiment_home(cfg)
    store = RunStore(cfg, run_id)
    removed = store.purge_projections()
    if store.live_dir.exists():
        for day in sorted(p for p in store.live_dir.iterdir() if p.is_dir()):
            removed += store.child(day.name).purge_projections()
    return removed
