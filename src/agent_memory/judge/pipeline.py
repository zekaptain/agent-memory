"""The judge pipeline: import transcripts, pick candidates, draft, decide, and the live loop.

Each step is resumable and writes JSONL under the run directory. Verdicts also land in the
experiment ledger as `decision` events so the ledger itself can answer "what did the judge
say about this turn, with which model, at what cost". No projection text enters those events.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb

from ..adapters.base import UnsupportedTranscript
from ..adapters.droid import DroidAdapter
from ..capture.pipeline import deliver, make_redactor
from ..capture.record import Record
from ..capture.writer import Writer
from ..config import Config
from ..db import AgentMemoryError, MemoryClient, sql_literal
from .backends import make_backend
from .drafter import DRAFTER_VERSION, draft_hash, make_drafter
from .labeling import LABELS_VERSION
from .projection import PROJECTION_VERSION, drafter_rendering, project_candidate
from .provenance import PROVENANCE_VERSION, require_summary, session_callers
from .questions import DEFAULT_THRESHOLDS, QUESTIONS_VERSION, thresholds_from_run
from .store import (
    JudgeError,
    RunStore,
    ServerUnreachable,
    ensure_experiment_home,
    imports_log,
    sole_writer,
    utc_now_iso,
)
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
from .verdict import JUDGE_AGENT, TurnAnswers, Verdict, ask_turn, judge

Log = Callable[[str], None]
"""`log` takes progress lines; `warn` takes problems the user has to act on (a skipped item, a
spooled batch). A caller with one channel passes only `log` and gets both on it."""
CALIBRATION_SHARE = 0.6
EVENT_BATCH = 25
# A command stops after this many failures in a row that are not the items' own.
STOP_AFTER = 3


def quiet(_: str) -> None:
    return None


class _Failures:
    """When the failures of one drafter or judge end a command.

    A failed item is skipped, and the next command asks it again. The command stops after
    `STOP_AFTER` failures in a row when the fault is not the items': the model server could
    not be reached each time, or this drafter or judge has written nothing in the run yet.
    So an item that fails every time cannot stop a run, and neither a server that is not
    there nor an unfit model can use a run up."""

    def __init__(self, name: str, *, wrote_before: bool) -> None:
        self._name = name
        self._wrote = wrote_before
        self._in_a_row = 0
        self._unreached = 0

    def answered(self) -> None:
        """The model server answered, usably or not."""
        self._unreached = 0

    def wrote(self) -> None:
        self._wrote = True
        self._in_a_row = self._unreached = 0

    def failed(self, exc: JudgeError) -> str | None:
        """Count a failure. Returns the error the command stops with, or None to go on."""
        self._in_a_row += 1
        self._unreached = self._unreached + 1 if isinstance(exc, ServerUnreachable) else 0
        if self._unreached >= STOP_AFTER:
            why = f"the model server could not be reached {STOP_AFTER} times in a row"
        elif not self._wrote and self._in_a_row >= STOP_AFTER:
            why = f"{STOP_AFTER} failures in a row and nothing written yet"
        else:
            return None
        return f"{self._name}: stopped, {why}: {exc}"


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
    log: Log = quiet,
) -> ImportSummary:
    """Write every transcript straight to the experiment ledger. Unchanged files are skipped."""
    ensure_experiment_home(cfg)
    summary = ImportSummary()
    paths = iter_transcripts(sessions_dir, project=project, since=since)
    adapter = DroidAdapter()
    with MemoryClient.from_config(cfg) as client, sole_writer(imports_log(cfg)):
        seen = {(row["path"], row["mtime"]) for row in RunStore.read_jsonl(imports_log(cfg))}
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
    """Droid's sessions that began at or after `since`. A session's row can carry a later
    time than its first event, so the start is the summary's."""
    stamp = sql_literal(since.astimezone(UTC).isoformat())
    _, rows = client.query(
        "SELECT session_id::VARCHAR FROM session_summary "
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
    log: Log = quiet,
) -> CandidateSummary:
    """Segment turns, apply triggers, sample, project, and write items.jsonl."""
    ensure_experiment_home(cfg)
    store = store or RunStore(cfg, run_id)
    _require_no_items(store)
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
                }
            )
        splits = assign_splits(chosen, seed=seed)
        items = []
        redactions = 0
        with sole_writer(store.items_path):
            # Asked again now that no other command can be building this run's items.
            _require_no_items(store)
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
            store.write_items(items)
    return CandidateSummary(
        pool=_count(pool),
        chosen=_count(chosen),
        items=len(items),
        redactions=redactions,
        pool_callers=_count_values(c.caller for c in pool),
        chosen_callers=_count_values(c.caller for c in chosen),
        harness_turns=harness_turns,
    )


def _require_no_items(store: RunStore) -> None:
    if store.has_items():
        raise JudgeError(f"run {store.run_id!r} already has items; use a new run id or purge it")


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
    log: Log = quiet,
    warn: Log | None = None,
) -> DraftSummary:
    """One draft per item for this drafter; other drafters' rows are left alone. Resumable."""
    warn = warn or log
    ensure_experiment_home(cfg)
    store = store or RunStore(cfg, run_id)
    run = store.read_run()
    require_current_questions(run)
    with sole_writer(store.drafts_path):
        existing = store.drafts()
        versions = {str(row.get("drafter_version") or "drafter_v2") for row in existing}
        incompatible = sorted(versions - {DRAFTER_VERSION})
        if incompatible:
            raise JudgeError(
                f"run {run_id} holds {', '.join(incompatible)} drafts; {DRAFTER_VERSION} drafts "
                "cannot be mixed in. Start a new run."
            )
        drafter = make_drafter(drafter_spec)
        done = {row["item_id"] for row in existing if row.get("drafter") == drafter.name}
        todo = [row for row in store.items() if row["item_id"] not in done]
        if limit is not None:
            todo = todo[:limit]
        summary = DraftSummary()
        snapshot = ""
        failures = _Failures(drafter.name, wrote_before=bool(done))
        stopped: str | None = None
        for index, item in enumerate(todo, start=1):
            state = drafter_rendering(store.read_projection(item["item_id"]))
            try:
                draft = drafter.draft(item["item_id"], state)
            except JudgeError as exc:
                summary.failed += 1
                warn(f"draft {index}/{len(todo)} failed, skipped (rerun to retry): {exc}")
                stopped = failures.failed(exc)
                if stopped:
                    break
                continue
            store.add_draft(draft.to_dict())
            failures.wrote()
            summary.written += 1
            summary.declined += draft.declined
            snapshot = draft.model_snapshot or snapshot
            what = "declined" if draft.declined else "drafted"
            log(f"{what} {index}/{len(todo)} ({draft.latency_ms / 1000:.1f}s)")
        # A command that stops still accounts for the drafts it wrote before it did.
        if summary.written or not stopped:
            drafters = dict(run.get("drafters") or {})
            previous = drafters.get(drafter.name) or {}
            drafters[drafter.name] = {
                "drafter_version": DRAFTER_VERSION,
                "model_snapshot": snapshot or previous.get("model_snapshot", ""),
                "drafts": len(done) + summary.written,
                "declined": int(previous.get("declined", 0)) + summary.declined,
                "drafted_at": utc_now_iso(),
            }
            store.update_run(
                drafter=drafter.name, drafter_version=DRAFTER_VERSION, drafters=drafters
            )
        if stopped:
            raise JudgeError(stopped)
    return summary


# Decide


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
            # Keyed exactly as `judge` records it, so an untagged legacy draft (no
            # drafter) still matches its own verdict instead of being judged on every run.
            key = (item["item_id"], row.get("drafter"))
            expected[key] = (item["projection_hash"], draft_hash(row.get("lesson")))
    return expected


@dataclass
class DecideSummary:
    """Per judge: the verdicts written, and the failures that left a verdict unwritten."""

    written: dict[str, int] = field(default_factory=dict)
    failed: dict[str, int] = field(default_factory=dict)


def decide_items(
    cfg: Config,
    run_id: str,
    backend_specs: Sequence[str],
    *,
    limit: int | None = None,
    store: RunStore | None = None,
    record_events: bool = True,
    log: Log = quiet,
    warn: Log | None = None,
) -> DecideSummary:
    """Ask each judge about every (item, draft) without a current verdict. The turn questions
    are asked once per item and copied into every row of that item."""
    warn = warn or log
    ensure_experiment_home(cfg)
    store = store or RunStore(cfg, run_id)
    run = store.read_run()
    require_current_questions(run)
    items = store.items()
    drafts = store.drafts_by_item()
    if record_events:
        require_judge_agent(cfg)
    expected = expected_verdicts(items, drafts)
    # A judge named twice is run once. Every verdict file this command will write is held
    # before any model is called, so a second command is refused before it costs anything.
    backends = list({b.name: b for b in map(make_backend, backend_specs)}.values())
    summary = DecideSummary()
    with ExitStack() as held:
        for backend in backends:
            held.enter_context(sole_writer(store.verdicts_path(backend.name)))
        for backend in backends:
            thresholds = thresholds_from_run(run, backend.name)
            existing = store.verdicts(backend.name)
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
                store.replace_verdicts(backend.name, kept)
            done = {(v["item_id"], v.get("drafter")) for v in kept}
            turns: dict[str, TurnAnswers] = {}
            for v in kept:
                carried = Verdict(v).turn_answers()
                if carried is not None and v["item_id"] not in turns:
                    turns[v["item_id"]] = carried
            pending_items = {key[0] for key in expected if key not in done}
            todo = [i for i in items if i["item_id"] in pending_items]
            if limit is not None:
                todo = todo[:limit]
            pending_records: list[Record] = []
            count = 0
            failed = 0
            failures = _Failures(backend.name, wrote_before=bool(kept))
            stopped: str | None = None
            for index, item in enumerate(todo, start=1):
                state = store.read_projection(item["item_id"])
                waiting: list[Mapping[str, Any] | None] = [
                    d
                    for d in drafts.get(item["item_id"]) or []
                    if (item["item_id"], d.get("drafter")) not in done
                ]
                if not drafts.get(item["item_id"]):
                    waiting = [None]
                turn = turns.get(item["item_id"])
                try:
                    if turn is None:
                        turn = ask_turn(backend, item, state)
                        failures.answered()
                except JudgeError as exc:
                    # The item is skipped, and the next decide asks it again.
                    failed += 1
                    warn(f"{backend.name}: item {item['item_id']} failed: {exc}")
                    stopped = failures.failed(exc)
                    if stopped:
                        break
                    continue
                for draft in waiting:
                    try:
                        verdict = judge(
                            backend, item, state, draft, thresholds, run_id=run_id, turn=turn
                        )
                    except JudgeError as exc:
                        failed += 1
                        who = f" draft {draft.get('drafter')}" if draft else ""
                        warn(f"{backend.name}: item {item['item_id']}{who} failed: {exc}")
                        stopped = failures.failed(exc)
                        if stopped:
                            break
                        continue
                    store.add_verdict(backend.name, verdict.row)
                    failures.wrote()
                    count += 1
                    # Later drafts of this item copy these answers instead of paying again.
                    turn = verdict.turn_answers() or replace(turn, reused=True)
                    turns[item["item_id"]] = turn
                    if record_events:
                        pending_records.append(verdict.decision_record(item, run["created_at"]))
                        if len(pending_records) >= EVENT_BATCH:
                            _record(cfg, pending_records, warn)
                            pending_records = []
                    who = f" [{draft.get('drafter')}]" if draft else ""
                    log(
                        f"{backend.name} {index}/{len(todo)}{who}: {verdict.recorded.action} "
                        f"({verdict.latency_ms / 1000:.1f}s)"
                    )
                if stopped:
                    break
            # A command that stops still records the verdicts it wrote before it did.
            if pending_records:
                _record(cfg, pending_records, warn)
            if stopped:
                raise JudgeError(stopped)
            if failed:
                warn(
                    f"{backend.name}: {failed} verdict(s) failed and were not written; "
                    "rerun to retry"
                )
            summary.written[backend.name] = count
            summary.failed[backend.name] = failed
    return summary


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
    failed: int = 0


def live_step(
    cfg: Config,
    run_id: str,
    *,
    sessions_dir: Path,
    since: datetime,
    backends: Sequence[str],
    drafter_specs: Sequence[str] = (),
    day: str | None = None,
    log: Log = quiet,
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
    child = parent.live_day(day)
    if not child.has_items():
        # The day's sessions and their callers are read through the summary.
        with MemoryClient.from_config(cfg) as client:
            require_summary(cfg, client)
    imported = import_sessions(cfg, sessions_dir, since=since, log=log)
    known = {row["item_id"] for row in parent.items()}
    for other in parent.live_days():
        known.update(row["item_id"] for row in other.items())
    candidates = None
    if not child.has_items():
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
    drafted = failed = 0
    for spec in drafter_specs:
        drafts = draft_items(cfg, run_id, spec, store=child, log=log, warn=warn)
        drafted += drafts.written
        failed += drafts.failed
    decided = decide_items(cfg, run_id, backends, store=child, log=log, warn=warn)
    return LiveSummary(
        day=day,
        imported=imported,
        candidates=candidates,
        drafted=drafted,
        decided=decided.written,
        failed=failed + sum(decided.failed.values()),
    )


def purge_run(cfg: Config, run_id: str) -> int:
    """Delete every projection file for the run and its live days. Verdicts and labels stay."""
    ensure_experiment_home(cfg)
    store = RunStore(cfg, run_id)
    return store.purge_projections() + sum(day.purge_projections() for day in store.live_days())
