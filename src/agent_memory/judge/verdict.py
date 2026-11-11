"""A verdict: what one judge answered about one draft of one turn.

`ask_turn` and `judge` build one; `Verdict` reads one back from its stored row, which it never
copies or changes. The turn questions are asked once per turn, and their answers are copied
at no cost into the verdict of every later draft of that turn.

The action is derived, not judged: `derive_action` reads the write gates of a question set
against thresholds. Gates travel as `{gate: probability of clearing}`. A gate the judge was
asked but did not usably answer is None and sends the verdict to review; a gate that is left
out was never asked and cannot clear.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Any, Literal

from ..capture.record import Record, SessionInfo, now
from .backends.base import Answer, DecisionBackend, JudgeInput, answer_from_probabilities
from .drafter import draft_hash
from .lesson import lesson_from_row
from .projection import PROJECTION_VERSION
from .questions import (
    BY_ID,
    DEFAULT_THRESHOLDS,
    QUESTIONS_VERSION,
    STAGE_A,
    WRITE_GATES,
    stage_b_for,
    unanswered,
    write_gates,
)
from .store import utc_now_iso
from .triggers import TRIGGERS_VERSION

# The registered agent every verdict is recorded under.
JUDGE_AGENT = "judge"

Action = Literal["write_lesson", "review", "discard"]

_TURN_IDS = frozenset(q.id for q in STAGE_A)
# A questions_v1 run asked `grounded` under the name `supported`.
_ALIAS = {"grounded": "supported", "supported": "grounded"}
# What the ledger copy of a verdict may hold: nothing that quotes the state or the lesson.
_LEDGER_KEYS = (
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
_LEDGER_ANSWER_KEYS = ("value", "probabilities", "confidence", "method")


@dataclass(frozen=True)
class Derived:
    action: Action
    reasons: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class TurnAnswers:
    """One judge's answers to the turn questions. `reused` marks copies taken from an earlier
    verdict of the turn, which has already paid for them."""

    answers: Mapping[str, Answer]
    reused: bool = False


def derive_action(
    p_true: Mapping[str, float | None],
    thresholds: Mapping[str, float] | None = None,
    *,
    gates: Sequence[str] = WRITE_GATES,
    declined: bool = False,
) -> Derived:
    """write_lesson when every gate clears its threshold; review when any gate sits inside the
    uncertainty band or was left unanswered (None); discard otherwise, and always for a draft
    the drafter declined. A gate missing from `p_true` was never asked: it cannot clear."""
    if declined:
        return Derived("discard", ["drafter declined: nothing worth keeping"])
    t = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    band = t["band"]
    reasons = []
    uncertain = False
    all_above = True
    for key in gates:
        if key not in p_true:
            reasons.append(f"{key}: no answer")
            all_above = False
            continue
        p = p_true[key]
        if p is None:
            # A placeholder is no evidence either way, so no threshold may read it.
            uncertain = True
            reasons.append(f"{key}: unanswered")
        elif key == "rule_supported" and p == 0.5:
            uncertain = True
            reasons.append("rule_supported=0.50: no direction of support")
        elif abs(p - t[key]) < band:
            uncertain = True
            reasons.append(f"{key}={p:.2f} within {band:.2f} of {t[key]:.2f}")
        elif p < t[key]:
            all_above = False
            reasons.append(f"{key}={p:.2f} below {t[key]:.2f}")
    if uncertain:
        return Derived("review", reasons)
    if all_above:
        return Derived("write_lesson", reasons or ["all thresholds cleared"])
    return Derived("discard", reasons)


def _clearing(qid: str, probabilities: Mapping[str, float]) -> float | None:
    """The probability that a question clears: p(yes) for a noul question, the mass on levels
    1 and 2 for `generalizable`, and p(none) for `enforced`. None when the distribution has
    no reading for it."""
    if qid == "generalizable":
        if "1" not in probabilities and "2" not in probabilities:
            return None
        return float(probabilities.get("1", 0.0)) + float(probabilities.get("2", 0.0))
    if qid == "enforced":
        return probabilities.get("none")
    return probabilities.get("yes")


def _stored(answers: Mapping[str, Any], qid: str) -> Mapping[str, Any] | None:
    return answers.get(qid) or answers.get(_ALIAS.get(qid, ""))


def _left_unanswered(stored: Mapping[str, Any]) -> bool:
    return unanswered(str(stored.get("method", "")))


def _gates(answers: Mapping[str, Any], names: Sequence[str]) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for gate in names:
        stored = _stored(answers, gate)
        if not stored:
            continue
        if _left_unanswered(stored):
            out[gate] = None
        elif (p := _clearing(gate, stored["probabilities"])) is not None:
            out[gate] = p
    return out


@dataclass(frozen=True)
class Verdict:
    """One stored verdict. Every reading goes to `row` as it is now: nothing is cached, and
    the row is the form that is written and read back."""

    row: Mapping[str, Any]

    @property
    def item_id(self) -> str:
        return self.row["item_id"]

    @property
    def backend(self) -> str:
        return self.row["backend"]

    @property
    def questions_version(self) -> str | None:
        return self.row.get("questions_version")

    @property
    def projection_hash(self) -> str | None:
        return self.row.get("projection_hash")

    @property
    def drafter(self) -> str | None:
        return self.row.get("drafter")

    @property
    def draft_hash(self) -> str | None:
        return self.row.get("draft_hash")

    @property
    def declined(self) -> bool:
        """The drafter declined to write a lesson, so there was no draft to ask about."""
        return bool(self.row.get("declined"))

    @property
    def recorded(self) -> Derived:
        """The action derived when the verdict was judged, at the thresholds of that day."""
        return Derived(self.row["action"], list(self.row["reasons"]))

    @property
    def latency_ms(self) -> float:
        return self.row.get("latency_ms", 0.0)

    @property
    def cost_usd(self) -> float:
        return self.row.get("cost_usd", 0.0)

    @property
    def draft_latency_ms(self) -> float:
        """What the draft questions took: the row's latency without the turn's answers."""
        return sum(float(a.get("latency_ms", 0.0)) for a in self._draft_answers())

    @property
    def draft_cost_usd(self) -> float:
        return sum(float(a.get("cost_usd", 0.0)) for a in self._draft_answers())

    def _draft_answers(self) -> list[Mapping[str, Any]]:
        answers = self.row.get("answers") or {}
        return [answer for qid, answer in answers.items() if qid not in _TURN_IDS]

    def _answered(self, qid: str) -> Mapping[str, Any] | None:
        """The stored answer to a question. None when the verdict holds none, or holds only
        the placeholder for a question left unanswered, which nothing may score."""
        stored = _stored(self.row.get("answers") or {}, qid)
        return None if not stored or _left_unanswered(stored) else stored

    def p(self, qid: str) -> float | None:
        """The probability the judge gave that a question clears."""
        answer = self._answered(qid)
        return _clearing(qid, answer["probabilities"]) if answer else None

    def confidence(self, qid: str) -> float:
        answer = self._answered(qid)
        return float(answer["confidence"]) if answer else 0.0

    def value(self, qid: str) -> str | None:
        """The judge's choice as the text a label holds: a score is its nearest level."""
        answer = self._answered(qid)
        if not answer:
            return None
        value = answer["value"]
        return str(round(value)) if isinstance(value, float) else str(value)

    def gates(self, names: Sequence[str] | None = None) -> dict[str, float | None]:
        """The write gates as `derive_action` reads them: None for a gate the judge left
        unanswered, no entry for one the verdict was never asked. `names` defaults to the
        gates of the question set the verdict was judged with."""
        if names is None:
            names = write_gates(self.questions_version)
        return _gates(self.row.get("answers") or {}, names)

    def action_at(self, thresholds: Mapping[str, float] | None = None) -> Derived:
        """The action these answers lead to at `thresholds`, by the write rule of the question
        set the verdict was judged with."""
        names = write_gates(self.questions_version)
        return derive_action(self.gates(names), thresholds, gates=names, declined=self.declined)

    def turn_answers(self) -> TurnAnswers | None:
        """The turn's answers for the next draft of the same turn: copies that cost nothing,
        so the row they land in reports only what it paid for. None when the verdict predates
        a turn question."""
        answers = self.row.get("answers") or {}
        if any(q.id not in answers for q in STAGE_A):
            return None
        return TurnAnswers(
            {
                q.id: replace(Answer.from_dict(answers[q.id]), latency_ms=0.0, cost_usd=0.0)
                for q in STAGE_A
            },
            reused=True,
        )

    def decision_record(
        self,
        item: Mapping[str, Any],
        run_created: str,
        *,
        harness_session_id: str | None = None,
        project: str = "judge-experiment",
    ) -> Record:
        """The ledger copy of the verdict: distributions and provenance, never the state or
        the lesson text."""
        row = self.row
        answers = {
            qid: {key: answer[key] for key in _LEDGER_ANSWER_KEYS}
            for qid, answer in row["answers"].items()
        }
        payload = {
            "judge": {key: row.get(key) for key in _LEDGER_KEYS}
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
            harness_session_id=harness_session_id or f"judge-{row['run_id']}",
            kind="decision",
            ts=now(),
            # Keyed on every input, so judging the same turn again with a draft (or a new
            # projection, or another drafter's draft) appends a new event instead of silently
            # keeping the old one.
            source_key="/".join(
                (
                    row["backend"],
                    row["item_id"],
                    row["questions_version"],
                    row["projection_hash"][:16],
                    (row.get("draft_hash") or "nodraft")[:16],
                    row.get("drafter") or "nodrafter",
                )
            ),
            content=f"judge {row['backend']}: {row['action']} for turn {item['turn_event_id']}",
            payload=payload,
            session=SessionInfo(
                project=project,
                scope="judge",
                started_at=datetime.fromisoformat(run_created),
                metadata={"run_id": row["run_id"]},
            ),
        )


def _judge_input(item: Mapping[str, Any], state: str) -> JudgeInput:
    return JudgeInput(
        item_id=item["item_id"], state=state, facts=item["facts"], flags=tuple(item["flags"])
    )


def ask_turn(backend: DecisionBackend, item: Mapping[str, Any], state: str) -> TurnAnswers:
    """Ask the turn questions, from the projection alone. Raises what the backend raises."""
    return TurnAnswers(backend.decide(_judge_input(item, state), STAGE_A))


def _claim_gates(answers: Mapping[str, Answer]) -> dict[str, Answer]:
    """The whole-draft `grounded` and `faithful` of a draft judged claim by claim. They cost
    nothing: every claim must clear, so each is its weakest claim. One claim the judge left
    unanswered leaves the whole gate unanswered."""
    out = {}
    for key in ("grounded", "faithful"):
        claims = [answer for qid, answer in answers.items() if qid.startswith(key + ".")]
        if not claims:
            continue
        if any(unanswered(answer.method) for answer in claims):
            out[key] = answer_from_probabilities(BY_ID[key], {}, method="min_over_claims_missing")
            continue
        p = min(float(answer.probabilities["yes"]) for answer in claims)
        out[key] = answer_from_probabilities(
            BY_ID[key], {"yes": p, "no": 1 - p}, method="min_over_claims"
        )
    return out


def judge(
    backend: DecisionBackend,
    item: Mapping[str, Any],
    state: str,
    draft: Mapping[str, Any] | None,
    thresholds: Mapping[str, float],
    *,
    run_id: str,
    turn: TurnAnswers | None = None,
) -> Verdict:
    """Judge one draft of a turn in the current question set. `turn` holds answers the turn
    already has; without it the turn questions are asked first. A declined draft gets no draft
    questions and is always discarded; a turn with no draft yet gets its turn answers alone.
    Raises what the backend raises, and then no verdict exists."""
    if turn is None:
        turn = ask_turn(backend, item, state)
    answers: dict[str, Answer] = dict(turn.answers)
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
        answers.update(_claim_gates(answers))
    stored = {qid: answer.to_dict() for qid, answer in answers.items()}
    derived = derive_action(_gates(stored, WRITE_GATES), thresholds, declined=declined)
    snapshots = sorted({a.model_snapshot for a in answers.values() if a.model_snapshot})
    return Verdict(
        {
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
            "stage_a_reused": turn.reused,
            "answers": stored,
            "latency_ms": sum(a.latency_ms for a in answers.values()),
            "cost_usd": sum(a.cost_usd for a in answers.values()),
            "action": derived.action,
            "reasons": derived.reasons,
            "thresholds": thresholds,
            "decided_at": utc_now_iso(),
        }
    )
