"""Turn segmentation over the ledger and the deterministic failure triggers (`triggers_v2`).

A turn is a `user_msg` and everything up to the next `outcome` event in the same session.
Triggers read only what the ledger already records: the outcome reason, tool-result error
flags, and the next message. They never read the agent's own claims.

Every user message is tagged with who wrote it (see `provenance`). Harness-injected messages
open turns, so the agent's reaction to them is segmented, but those turns are not candidates
and a harness message never counts as "the next message". A correction from a person is T4;
the same words from a parent agent are T5.
"""

from __future__ import annotations

import json
import random
import re
import uuid
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..capture.ids import NAMESPACE
from ..db import MemoryClient
from .provenance import HUMAN, Caller, Provenance, message_provenance, strip_harness_blocks

TRIGGERS_VERSION = "triggers_v2"

# T1 cancelled, T2 error or provider failure, T3 tool error, T4 human correction,
# T5 parent-agent correction, T0 clean.
PROVIDER_ERRORS = frozenset(
    {"error", "model_provider_unavailable", "model_rate_limited", "model_authentication_failed"}
)
STRATUM_PRIORITY = ("T2", "T1", "T3", "T4", "T5", "T0")
TRIGGERED_STRATA = ("T1", "T3", "T4", "T5")
CALLER_KINDS = ("human", "agent")
SNIPPET_CHARS = 500

# Matched case-insensitively against the first 500 characters of the user's next message.
CORRECTION_LEXICON: tuple[str, ...] = (
    r"^\s*no\b",
    r"\bthat'?s (?:not|wrong|incorrect)\b",
    r"\b(?:is|was) wrong\b",
    r"\brevert\b",
    r"\bundo\b",
    r"\bdidn'?t (?:work|ask|want|do|run|fix)\b",
    r"\bstill (?:fail|broken|not|doesn'?t|wrong)",
    r"\bnot what i (?:asked|meant|wanted)\b",
    r"\byou (?:broke|missed|ignored|skipped|forgot|removed|deleted)\b",
    r"\bwhy did you\b",
    r"\bstop\b",
    r"\bdon'?t do that\b",
    r"\bthat broke\b",
    r"\bgo back\b",
)
_CORRECTION = re.compile("|".join(f"(?:{p})" for p in CORRECTION_LEXICON), re.IGNORECASE)

_TURN_KINDS = ("user_msg", "assistant_msg", "tool_call", "tool_result", "outcome")

# Only user messages and outcomes carry a snippet; everything else is structure. Tool-call
# payloads hold whole file contents, so only the two fields segmentation needs are fetched.
# User snippets are taken after the harness blocks are removed, since a reminder alone can be
# longer than the snippet; a message truncated inside a block is finished from its blob.
HARNESS_BLOCK_SQL = "'<system-(reminder|notification)>.*?</system-(reminder|notification)>'"
EVENTS_SQL = (
    "SELECT e.session_id::VARCHAR, s.project, e.event_id::VARCHAR, e.kind, e.seq, "
    "e.ts::VARCHAR, "
    "CASE WHEN e.kind = 'user_msg' THEN left(trim(regexp_replace(e.content, "
    + HARNESS_BLOCK_SQL
    + ", '', 'gs')), {snippet}) WHEN e.kind = 'outcome' THEN left(e.content, {snippet}) END "
    "AS snippet, "
    "CASE WHEN e.kind = 'tool_call' THEN json_object('tool_name', "
    "json_extract_string(CAST(e.payload AS JSON), '$.tool_name')) "
    "WHEN e.kind = 'tool_result' THEN json_object('is_error', "
    "json_extract(CAST(e.payload AS JSON), '$.is_error')) END::VARCHAR AS payload, "
    "e.parent_event_id::VARCHAR, s.started_at::VARCHAR, "
    "CASE WHEN e.kind = 'user_msg' AND e.truncated THEN e.blob_ref END AS blob_ref "
    "FROM events e JOIN sessions s USING (session_id) "
    "WHERE e.kind IN ({kinds}){extra} "
    "ORDER BY e.session_id, e.ts, e.seq, e.recorded_at"
)


@dataclass(frozen=True)
class TurnEvent:
    event_id: str
    kind: str
    seq: int
    ts: str
    snippet: str | None = None
    payload: dict[str, Any] | None = None
    parent_event_id: str | None = None


@dataclass
class Turn:
    session_id: str
    project: str | None
    session_started_at: str | None
    events: list[TurnEvent] = field(default_factory=list)
    outcome_reason: str | None = None
    next_user: TurnEvent | None = None
    caller: Caller = HUMAN

    @property
    def user_event(self) -> TurnEvent:
        return self.events[0]

    @property
    def turn_id(self) -> str:
        return self.user_event.event_id

    @property
    def opener_provenance(self) -> Provenance:
        return message_provenance(self.user_event.snippet, self.caller.kind)

    @property
    def next_provenance(self) -> Provenance | None:
        if self.next_user is None:
            return None
        return message_provenance(self.next_user.snippet, self.caller.kind)

    def of_kind(self, kind: str) -> list[TurnEvent]:
        return [e for e in self.events if e.kind == kind]


def parse_payload(raw: Any) -> dict[str, Any] | None:
    if raw is None:
        return None
    if isinstance(raw, dict):
        return raw
    try:
        value = json.loads(raw)
    except (TypeError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def fetch_turn_events(
    client: MemoryClient,
    *,
    session_ids: Sequence[str] | None = None,
    blob_dir: Path | None = None,
) -> list[tuple[str, str | None, str | None, TurnEvent]]:
    """Compact event rows for segmentation: (session_id, project, session_started_at, event)."""
    from ..db import sql_literal

    extra = ""
    if session_ids is not None:
        if not session_ids:
            return []
        extra = " AND e.session_id IN (" + ", ".join(sql_literal(s) for s in session_ids) + ")"
    kinds = ", ".join(f"'{k}'" for k in _TURN_KINDS)
    _, rows = client.query(EVENTS_SQL.format(kinds=kinds, extra=extra, snippet=SNIPPET_CHARS))
    out = []
    for sid, project, eid, kind, seq, ts, snippet, payload, parent, started, blob_ref in rows:
        if blob_ref and blob_dir is not None and _needs_blob(snippet):
            snippet = user_snippet(read_blob(blob_dir, str(blob_ref)))
        event = TurnEvent(
            event_id=str(eid),
            kind=str(kind),
            seq=int(seq or 0),
            ts=str(ts),
            snippet=snippet,
            payload=parse_payload(payload),
            parent_event_id=str(parent) if parent else None,
        )
        out.append((str(sid), project, str(started) if started else None, event))
    return out


def _needs_blob(snippet: str | None) -> bool:
    """The inline 4 KiB ended inside a harness block, so the message itself is in the blob."""
    return not snippet or snippet.lstrip().startswith("<system-")


def read_blob(blob_dir: Path, blob_ref: str) -> str | None:
    try:
        return (blob_dir / blob_ref).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def user_snippet(content: str | None) -> str:
    body, _ = strip_harness_blocks(content)
    return body[:SNIPPET_CHARS]


def segment_turns(
    rows: Iterable[tuple[str, str | None, str | None, TurnEvent]],
    callers: Mapping[str, Caller] | None = None,
) -> list[Turn]:
    """Group ordered events into turns. A turn closes at its outcome or at the next user message.
    The next message of a turn is the next one a person or a parent agent wrote."""
    turns: list[Turn] = []
    by_session: dict[str, list[Turn]] = {}
    current: Turn | None = None
    current_sid: str | None = None
    for sid, project, started, event in rows:
        if sid != current_sid:
            current = None
            current_sid = sid
        if event.kind == "user_msg":
            current = Turn(
                session_id=sid,
                project=project,
                session_started_at=started,
                caller=(callers or {}).get(sid, HUMAN),
            )
            current.events.append(event)
            turns.append(current)
            by_session.setdefault(sid, []).append(current)
            continue
        if current is None:
            continue  # tool traffic before the first user message
        if event.kind == "outcome":
            if current.outcome_reason is None:
                current.events.append(event)
                current.outcome_reason = (event.snippet or "").strip() or None
            continue
        if current.outcome_reason is None:
            current.events.append(event)
    for session_turns in by_session.values():
        for i, turn in enumerate(session_turns):
            turn.next_user = next(
                (
                    later.user_event
                    for later in session_turns[i + 1 :]
                    if later.opener_provenance != "harness"
                ),
                None,
            )
    return turns


def is_correction(text: str | None) -> bool:
    return bool(text) and _CORRECTION.search(text[:SNIPPET_CHARS]) is not None


def tool_error_facts(turn: Turn) -> dict[str, Any]:
    """Counts the rules backend and the projection share: errors, repeats, tools used."""
    calls = {e.event_id: e for e in turn.of_kind("tool_call")}
    tools_used: list[str] = []
    for call in calls.values():
        name = (call.payload or {}).get("tool_name")
        if name and name not in tools_used:
            tools_used.append(str(name))
    error_tools: list[str] = []
    ok_after_error = False
    last_result_ok: bool | None = None
    for result in turn.of_kind("tool_result"):
        is_error = bool((result.payload or {}).get("is_error"))
        parent = calls.get(result.parent_event_id or "")
        name = str((parent.payload or {}).get("tool_name") or "tool") if parent else "tool"
        if is_error:
            error_tools.append(name)
            ok_after_error = False
        elif error_tools:
            ok_after_error = True
        last_result_ok = not is_error
    repeated = any(error_tools.count(t) >= 2 for t in set(error_tools))
    return {
        "tool_count": len(calls),
        "tool_error_count": len(error_tools),
        "error_tools": error_tools,
        "repeated_tool_error": repeated,
        "recovered_after_error": ok_after_error,
        "last_result_ok": last_result_ok,
        "tools": tools_used,
    }


def trigger_flags(turn: Turn) -> list[str]:
    flags = []
    reason = turn.outcome_reason
    if reason == "cancelled":
        flags.append("T1")
    if reason in PROVIDER_ERRORS:
        flags.append("T2")
    if any((e.payload or {}).get("is_error") for e in turn.of_kind("tool_result")):
        flags.append("T3")
    if turn.next_user is not None and is_correction(turn.next_user.snippet):
        flags.append("T4" if turn.next_provenance == "human" else "T5")
    if not flags and reason == "completed":
        flags.append("T0")
    return flags


def stratum_of(flags: Sequence[str]) -> str | None:
    for s in STRATUM_PRIORITY:
        if s in flags:
            return s
    return None


@dataclass(frozen=True)
class Candidate:
    item_id: str
    session_id: str
    project: str | None
    session_started_at: str | None
    turn_event_id: str
    event_ids: list[str]
    next_user_event_id: str | None
    outcome_reason: str | None
    flags: list[str]
    stratum: str
    facts: dict[str, Any]
    caller: str = "human"  # who started the session: human, or agent
    caller_role: str | None = None  # the spawned droid's role when caller is agent
    opener_provenance: str = "human"
    next_provenance: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def item_id_for(session_id: str, turn_event_id: str) -> str:
    return str(uuid.uuid5(NAMESPACE, f"judge/{session_id}/{turn_event_id}"))


def candidates_from_turns(turns: Iterable[Turn]) -> list[Candidate]:
    """Every turn a person or a parent agent opened that has an outcome or a trigger and did
    some work. Order is stable."""
    out = []
    for turn in turns:
        if not turn.of_kind("assistant_msg") and not turn.of_kind("tool_call"):
            continue
        if turn.opener_provenance == "harness":
            continue
        flags = trigger_flags(turn)
        stratum = stratum_of(flags)
        if stratum is None:
            continue
        out.append(
            Candidate(
                item_id=item_id_for(turn.session_id, turn.turn_id),
                session_id=turn.session_id,
                project=turn.project,
                session_started_at=turn.session_started_at,
                turn_event_id=turn.turn_id,
                event_ids=[e.event_id for e in turn.events],
                next_user_event_id=turn.next_user.event_id if turn.next_user else None,
                outcome_reason=turn.outcome_reason,
                flags=flags,
                stratum=stratum,
                facts={**tool_error_facts(turn), "caller": turn.caller.label},
                caller=turn.caller.kind,
                caller_role=turn.caller.role,
                opener_provenance=turn.opener_provenance,
                next_provenance=turn.next_provenance,
            )
        )
    return out


def sample_candidates(
    candidates: Sequence[Candidate],
    *,
    n: int,
    seed: int,
    clean_share: float = 0.25,
    human_share: float = 0.5,
) -> list[Candidate]:
    """Seeded and stratified twice: `human_share` of n from sessions a person started and the
    rest from sessions an agent started (either side back-fills the other when short), then
    within each side every T2, the triggered budget split across T1/T3/T4/T5, and
    `clean_share` from clean controls. Short strata are back-filled from the others."""
    rng = random.Random(seed)
    human = [c for c in candidates if c.caller == "human"]
    agent = [c for c in candidates if c.caller != "human"]
    n_human = min(round(n * human_share), len(human))
    n_agent = min(n - n_human, len(agent))
    n_human = min(n - n_agent, len(human))
    return _sample_strata(human, n_human, rng, clean_share) + _sample_strata(
        agent, n_agent, rng, clean_share
    )


def _sample_strata(
    candidates: Sequence[Candidate], n: int, rng: random.Random, clean_share: float
) -> list[Candidate]:
    if n <= 0:
        return []
    pools: dict[str, list[Candidate]] = {s: [] for s in STRATUM_PRIORITY}
    for c in candidates:
        pools[c.stratum].append(c)
    for pool in pools.values():
        pool.sort(key=lambda c: c.item_id)
        rng.shuffle(pool)
    n_clean = min(round(n * clean_share), len(pools["T0"]))
    n_triggered = n - n_clean
    chosen: list[Candidate] = pools["T2"][:n_triggered]
    remaining = n_triggered - len(chosen)
    rest = {s: pools[s] for s in TRIGGERED_STRATA}
    while remaining > 0 and any(rest.values()):
        for s in TRIGGERED_STRATA:
            if remaining <= 0:
                break
            if rest[s]:
                chosen.append(rest[s].pop())
                remaining -= 1
    # Triggered strata ran dry: fill from clean turns so the sample keeps its size.
    chosen += pools["T0"][: n_clean + remaining]
    return chosen
