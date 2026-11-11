"""Resumable daily admission from captured production events, never transcript imports."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from functools import partial
from typing import Any

from .. import endpoints
from ..capture.pipeline import make_redactor
from ..config import Config
from ..db import MemoryClient, sql_literal
from ..judge.backends import make_backend
from ..judge.backends.base import DecisionBackend, require_remote
from ..judge.drafter import make_drafter
from ..judge.pipeline import Log, quiet
from ..judge.projection import PROJECTION_VERSION, project_candidate
from ..judge.provenance import session_callers
from ..judge.questions import QUESTIONS_VERSION
from ..judge.store import JudgeError, RunStore, utc_now_iso
from ..judge.triggers import candidates_from_turns, fetch_turn_events, segment_turns
from ..judge.verdict import JUDGE_AGENT, ask_turn, judge
from . import approver, ledger
from .embed import make_embedder
from .examples import example_for
from .store import AdmissionStore, write_jsonl

DEFAULT_DRAFTER = "ollama:gemma4:26b"


def _local_drafter(spec: str) -> Any:
    # Admission deliberately does not offer the hosted experiment drafter arm.
    if spec.startswith("ollama:"):
        host = endpoints.OLLAMA_HOST
    elif spec.startswith("llamacpp:"):
        host = endpoints.LLAMACPP_HOST
    else:
        raise JudgeError("admission needs a local ollama: or llamacpp: drafter")
    if not endpoints.is_loopback(host):
        raise JudgeError("admission drafting requires a loopback model host")
    return make_drafter(spec)


def _backend(spec: str) -> DecisionBackend:
    backend = make_backend(spec)
    if not backend.local:
        require_remote("The admission judge")
    return backend


def _model_call(call: Callable[[], Any], subject: str) -> Any:
    try:
        return call()
    except (JudgeError, endpoints.EgressError):
        # Model/HTTP errors can quote a private reply or request. Keep those out of
        # terminal output (and scheduler logs); completed stage files remain intact.
        raise JudgeError(
            f"{subject} failed; check the configured model service and egress settings, "
            "then rerun mem admit run to resume"
        ) from None


def _discover(cfg: Config, store: AdmissionStore, day: RunStore, cutoff: str) -> None:
    state = store.read_state()
    watermark = state.get("watermark")
    since = (
        f" AND (s.started_at >= {sql_literal(watermark)}::TIMESTAMPTZ "
        "OR EXISTS (SELECT 1 FROM events changed WHERE changed.session_id = s.session_id "
        f"AND changed.recorded_at >= {sql_literal(watermark)}::TIMESTAMPTZ))"
        if watermark
        else ""
    )
    known = set()
    for path in (store.root / "days").glob("*/items.jsonl"):
        known.update(row["item_id"] for row in RunStore.read_jsonl(path))
    with MemoryClient.from_config(cfg) as client:
        _, rows = client.query(
            "SELECT s.session_id::VARCHAR FROM sessions s WHERE s.agent_id NOT IN "
            f"({sql_literal(JUDGE_AGENT)}, {sql_literal(ledger.ADMISSION_AGENT)}) "
            f"AND s.started_at <= {sql_literal(cutoff)}::TIMESTAMPTZ{since}"
        )
        turns = segment_turns(
            fetch_turn_events(client, session_ids=[str(r[0]) for r in rows], blob_dir=cfg.blob_dir),
            session_callers(cfg, client),
        )
        redactor = make_redactor(cfg)
        for candidate in sorted(candidates_from_turns(turns), key=lambda c: c.item_id):
            if candidate.item_id in known:
                continue
            projection = project_candidate(client, candidate, redactor, cfg.blob_dir)
            day.write_projection(candidate.item_id, projection.text)
            write_jsonl(
                day.items_path,
                [
                    {
                        **candidate.to_dict(),
                        "projection_version": PROJECTION_VERSION,
                        "projection_hash": projection.hash,
                        "sidecar": projection.sidecar,
                    }
                ],
                append=True,
            )


def _process(cfg: Config, store: AdmissionStore, day: RunStore, log: Log) -> int:
    run = day.read_run()
    model = run["approver"]
    drafter = _local_drafter(run["drafter"])
    backend = _backend(model["backend"])
    if backend.name != model["backend"]:
        raise JudgeError("backend identity does not match the fitted approver")
    embedder = make_embedder(model["embedder"])
    examples = RunStore.read_jsonl(day.root / "examples.jsonl")
    drafts = {row["item_id"]: row for row in RunStore.read_jsonl(day.drafts_path)}
    verdicts = {row["item_id"]: row for row in RunStore.read_jsonl(day.verdicts_path(backend.name))}
    scores_path = day.root / "scores.jsonl"
    scores = {row["item_id"]: row for row in RunStore.read_jsonl(scores_path)}
    done = {row["item_id"] for row in RunStore.read_jsonl(day.root / "admitted.jsonl")}
    count = 0
    for item in RunStore.read_jsonl(day.items_path):
        iid = item["item_id"]
        if iid in done:
            continue
        projection = day.read_projection(iid)
        if iid not in drafts:
            from ..judge.projection import drafter_rendering

            draft = _model_call(
                partial(drafter.draft, iid, drafter_rendering(projection)), "Admission drafting"
            ).to_dict()
            draft, _ = make_redactor(cfg).value(draft)
            write_jsonl(day.drafts_path, [draft], append=True)
            drafts[iid] = draft
        draft = drafts[iid]
        if iid not in verdicts:
            turn = _model_call(
                partial(ask_turn, backend, item, projection), "Admission turn judging"
            )
            judged = _model_call(
                partial(
                    judge,
                    backend,
                    item,
                    projection,
                    draft,
                    model["thresholds"],
                    run_id=day.run_id,
                    turn=turn,
                ),
                "Admission draft judging",
            )
            write_jsonl(day.verdicts_path(backend.name), [judged.row], append=True)
            verdicts[iid] = judged.row
        verdict = verdicts[iid]
        if iid not in scores:
            embedding = (
                _model_call(partial(embedder.embed, draft), "Admission embedding")
                if not draft["declined"]
                else []
            )
            if draft["declined"]:
                result = {"action": "discard", "score": 0.0, "reasons": ["drafter declined"]}
            else:
                example = example_for(
                    draft,
                    item,
                    verdict,
                    embedding,
                    source=f"{day.run_id}/{iid}",
                    label=None,
                    split="production",
                )
                example["embedder"] = model["embedder"]
                result = approver.score(model, example, examples)
            result = {
                **result,
                "item_id": iid,
                "embedding": embedding,
                "approver_version": model["version"],
                "backend": backend.name,
                "drafter": draft["drafter"],
                "draft_hash": verdict["draft_hash"],
            }
            write_jsonl(scores_path, [result], append=True)
            scores[iid] = result
        result = ledger.admit(
            cfg,
            day.run_id,
            run["created_at"],
            item,
            draft,
            verdict,
            scores[iid],
            auto=(
                run["mode"] == "auto"
                and store.read_state().get("mode") == "auto"
                and store.read_state().get("approver_version") == model["version"]
            ),
        )
        write_jsonl(day.root / "admitted.jsonl", [result], append=True)
        done.add(iid)
        count += 1
        log(f"processed {len(done)} lesson(s)")
    return count


def run_day(
    cfg: Config,
    *,
    day: str | None = None,
    drafter: str = DEFAULT_DRAFTER,
    backend: str | None = None,
    log: Log = quiet,
) -> dict[str, Any]:
    """Finish older days first, then discover and process today's captured turns.

    Each day pins the model, neighbors and mode. Partial model or ledger failures leave
    finished stages intact. The watermark advances only after all pending work succeeds.
    """
    store = AdmissionStore(cfg)
    model = store.read_approver()
    if not model:
        raise JudgeError(
            "no admission approver; import v4 human labels and run mem admit fit first"
        )
    if backend is not None and make_backend(backend).name != model["backend"]:
        raise JudgeError("backend does not match approver.json; fit a separate approver first")
    _local_drafter(drafter)
    _backend(model["backend"])  # Check remote opt-in before creating any day files.
    approver.validate_examples(model, store.read_examples())
    state = store.read_state()
    if state.get("mode") == "auto" and not approver.can_auto(
        model, min_writes=state.get("min_writes", 20)
    ):
        raise JudgeError("auto admission is below the held-out bar; enable --shadow and refit")
    day = day or datetime.now(UTC).strftime("%Y-%m-%d")
    target = store.day(day)
    count = 0
    for path in sorted((store.root / "days").glob("*/run.json")):
        if path.parent != target.root:
            old = store.day(path.parent.name)
            items = RunStore.read_jsonl(old.items_path)
            done = {r["item_id"] for r in RunStore.read_jsonl(old.root / "admitted.jsonl")}
            if any(i["item_id"] not in done for i in items):
                count += _process(cfg, store, old, log)
    cutoff = utc_now_iso()
    if not target.exists():
        target.create(
            {
                "questions_version": QUESTIONS_VERSION,
                "approver": model,
                "drafter": drafter,
                "mode": state.get("mode", "shadow"),
            }
        )
        write_jsonl(target.root / "examples.jsonl", store.read_examples())
    else:
        run = target.read_run()
        if backend is not None and make_backend(backend).name != run["approver"]["backend"]:
            raise JudgeError("this day is pinned to a different backend; resume without --backend")
    _discover(cfg, store, target, cutoff)
    count += _process(cfg, store, target, log)
    store.write_state(watermark=cutoff)
    return {"day": day, "processed": count, "mode": target.read_run()["mode"]}
