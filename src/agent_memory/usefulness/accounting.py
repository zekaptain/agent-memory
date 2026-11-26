"""Recall and citation accounting for memory-arm usefulness trials (MEM-19).

Combines a finished memory-arm trial's transcript with its per-trial ledger home:
- Imports the trial transcript through the capture path (``from_transcript`` +
  ``deliver``) with ``AGENT_MEMORY_MCP_SERVER_NAME`` matched to the trial's
  configured memory server so cross-session attribution binds citations.
- Counts recalls per trial and classifies each recall as:
  ``returned`` (ledger retrieval with ``returned > 0``),
  ``miss`` (ledger retrieval with ``returned == 0``),
  ``degraded`` (tool returned ``degraded: true`` without the ledger), or
  ``missing_record`` (transcript shows a recall that the ledger does not hold,
  never counted as a miss).
- Counts citations made, returned memories declared used, placeholders with no
  binding, and actual assistant replies bound to one citation only.
"""

from __future__ import annotations

import json
import os
import uuid
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import duckdb

from ..adapters import HARNESS_NAMES, get_adapter
from ..capture.pipeline import IMPORT_DEADLINE, deliver
from ..capture.record import Record
from ..config import Config
from ..db import MemoryClient, open_direct, sql_literal
from .store import UsefulnessError

RecallOutcome = Literal["returned", "miss", "degraded", "missing_record"]
RecallSource = Literal["transcript_and_ledger", "transcript", "transcript_only", "ledger_only"]
TranscriptImportStatus = Literal["imported", "empty", "missing", "failed", "skipped"]

DEFAULT_SERVER_NAME = "agent-memory"
_RECALL_OPERATIONS = ("recall", "graph")
_CHUNK_SIZE = 500

type LedgerSource = MemoryClient | duckdb.DuckDBPyConnection


@dataclass(frozen=True)
class TrialLedgerBaseline:
    """IDs already present in a trial home before the harness runs.

    A trial home is copied from a historical memory set, so pre-existing
    sessions, retrievals, context links, and placeholders must be excluded
    when accounting for what *this* trial did.
    """

    session_ids: frozenset[str] = frozenset()
    retrieval_ids: frozenset[str] = frozenset()
    link_ids: frozenset[str] = frozenset()
    placeholder_event_ids: frozenset[str] = frozenset()

    @classmethod
    def empty(cls) -> TrialLedgerBaseline:
        return cls()


@dataclass(frozen=True)
class TranscriptImportSummary:
    """Result of importing a memory-arm trial's transcript into its trial home."""

    status: TranscriptImportStatus
    harness: str | None
    transcript_path: str | None
    records_parsed: int
    inserted: int
    skipped: int
    spooled: int
    failed_lines: int
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "harness": self.harness,
            "transcript_path": self.transcript_path,
            "records_parsed": self.records_parsed,
            "inserted": self.inserted,
            "skipped": self.skipped,
            "spooled": self.spooled,
            "failed_lines": self.failed_lines,
            "error": self.error,
        }


@dataclass(frozen=True)
class RecallCallRecord:
    """Accounting entry for one recall observed in the transcript and/or trial ledger."""

    outcome: RecallOutcome
    source: RecallSource
    operation: str | None
    call_id: str | None
    retrieval_id: str | None
    returned_count: int | None
    returned_memory_ids: tuple[str, ...]
    strategy: str | None
    latency_ms: float | None
    detail: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "outcome": self.outcome,
            "source": self.source,
            "operation": self.operation,
            "call_id": self.call_id,
            "retrieval_id": self.retrieval_id,
            "returned_count": self.returned_count,
            "returned_memory_ids": list(self.returned_memory_ids),
            "strategy": self.strategy,
            "latency_ms": self.latency_ms,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class CitationRecord:
    """One citation group in the trial ledger and its binding state."""

    original_answer_event_id: str
    is_placeholder: bool
    bound_answer_event_id: str | None
    binding_reason: str | None
    retrieval_ids: tuple[str, ...]
    cited_memory_ids: tuple[str, ...]
    context_event_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_answer_event_id": self.original_answer_event_id,
            "is_placeholder": self.is_placeholder,
            "bound_answer_event_id": self.bound_answer_event_id,
            "binding_reason": self.binding_reason,
            "retrieval_ids": list(self.retrieval_ids),
            "cited_memory_ids": list(self.cited_memory_ids),
            "context_event_ids": list(self.context_event_ids),
        }


@dataclass(frozen=True)
class CitationAccounting:
    """Citation, placeholder-binding, and single-citation-reply counts for one trial."""

    citations_made: int
    cite_calls_in_transcript: int
    degraded_cite_calls: int
    citation_links: int
    retrieval_links: int
    cited_memory_links: int
    context_event_links: int
    returned_memory_ids: tuple[str, ...]
    cited_memory_ids: tuple[str, ...]
    returned_memories_cited_count: int
    returned_memories_cited_ids: tuple[str, ...]
    placeholder_count: int
    bound_placeholder_count: int
    unbound_placeholder_count: int
    unbound_placeholder_ids: tuple[str, ...]
    bound_reply_count: int
    single_citation_reply_count: int
    single_citation_reply_ids: tuple[str, ...]
    multi_citation_reply_count: int
    multi_citation_reply_ids: tuple[str, ...]
    citations: tuple[CitationRecord, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "citations_made": self.citations_made,
            "cite_calls_in_transcript": self.cite_calls_in_transcript,
            "degraded_cite_calls": self.degraded_cite_calls,
            "citation_links": self.citation_links,
            "retrieval_links": self.retrieval_links,
            "cited_memory_links": self.cited_memory_links,
            "context_event_links": self.context_event_links,
            "returned_memory_ids": list(self.returned_memory_ids),
            "cited_memory_ids": list(self.cited_memory_ids),
            "returned_memories_cited_count": self.returned_memories_cited_count,
            "returned_memories_cited_ids": list(self.returned_memories_cited_ids),
            "placeholder_count": self.placeholder_count,
            "bound_placeholder_count": self.bound_placeholder_count,
            "unbound_placeholder_count": self.unbound_placeholder_count,
            "unbound_placeholder_ids": list(self.unbound_placeholder_ids),
            "bound_reply_count": self.bound_reply_count,
            "single_citation_reply_count": self.single_citation_reply_count,
            "single_citation_reply_ids": list(self.single_citation_reply_ids),
            "multi_citation_reply_count": self.multi_citation_reply_count,
            "multi_citation_reply_ids": list(self.multi_citation_reply_ids),
            "citations": [c.to_dict() for c in self.citations],
        }


@dataclass(frozen=True)
class RecallAccounting:
    """Summary of what recall and citations did in one memory-arm trial."""

    made_no_recall: bool
    total_recalls: int
    returned_count: int
    miss_count: int
    degraded_count: int
    missing_record_count: int
    total_retrieval_latency_ms: float | None
    trial_session_ids: tuple[str, ...]
    recalls: tuple[RecallCallRecord, ...]
    citations: CitationAccounting
    transcript_import: TranscriptImportSummary

    def to_dict(self) -> dict[str, Any]:
        return {
            "made_no_recall": self.made_no_recall,
            "total_recalls": self.total_recalls,
            "returned_count": self.returned_count,
            "miss_count": self.miss_count,
            "degraded_count": self.degraded_count,
            "missing_record_count": self.missing_record_count,
            "total_retrieval_latency_ms": self.total_retrieval_latency_ms,
            "trial_session_ids": list(self.trial_session_ids),
            "recalls": [r.to_dict() for r in self.recalls],
            "citations": self.citations.to_dict(),
            "transcript_import": self.transcript_import.to_dict(),
        }


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------


def _rows(source: LedgerSource, sql: str) -> list[dict[str, Any]]:
    if isinstance(source, MemoryClient):
        cols, rows = source.query(sql)
    else:
        cursor = source.execute(sql)
        cols = [d[0] for d in cursor.description or ()]
        rows = cursor.fetchall()
    return [dict(zip(cols, row, strict=True)) for row in rows]


def _uuid(value: Any) -> str | None:
    if value is None:
        return None
    try:
        return str(uuid.UUID(str(value)))
    except (ValueError, TypeError):
        return None


def is_memory_tool_call(
    payload: Mapping[str, Any] | None,
    operation: str,
    *,
    server_name: str = DEFAULT_SERVER_NAME,
) -> bool:
    """Return True if ``payload`` invokes ``operation`` on ``server_name``."""
    if not isinstance(payload, Mapping):
        return False
    name = payload.get("tool_name")
    if payload.get("mcp_server") == server_name and name == operation:
        return True
    return name in {
        f"{server_name}.{operation}",
        f"{server_name}__{operation}",
        f"{server_name}___{operation}",
        f"mcp__{server_name}__{operation}",
        f"tools.{server_name}.{operation}",
    }


def _recall_operation(
    payload: Mapping[str, Any] | None,
    *,
    server_name: str = DEFAULT_SERVER_NAME,
) -> str | None:
    for op in _RECALL_OPERATIONS:
        if is_memory_tool_call(payload, op, server_name=server_name):
            return op
    return None


def _decode_tool_result_dict(value: Any) -> dict[str, Any] | None:
    """Unwrap structured MCP tool results or their text/JSON envelope."""
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            parsed = json.loads(text)
        except (ValueError, TypeError):
            return None
        return _decode_tool_result_dict(parsed)
    if isinstance(value, list):
        candidates: list[dict[str, Any]] = []
        for part in value:
            decoded = _decode_tool_result_dict(part)
            if decoded is not None:
                candidates.append(decoded)
        if len(candidates) == 1:
            return candidates[0]
        for candidate in candidates:
            if any(k in candidate for k in ("degraded", "retrieval_id", "answer_event_id")):
                return candidate
        return candidates[0] if candidates else None
    if not isinstance(value, dict):
        return None
    if any(k in value for k in ("degraded", "retrieval_id", "answer_event_id")):
        return value
    for key in ("structuredContent", "structured_content", "content", "text"):
        if key in value:
            decoded = _decode_tool_result_dict(value.get(key))
            if decoded is not None:
                return decoded
    return None


# ---------------------------------------------------------------------------
# Baseline capture & transcript import
# ---------------------------------------------------------------------------


def capture_trial_baseline(source: LedgerSource) -> TrialLedgerBaseline:
    """Snapshot pre-trial IDs from a freshly cloned trial home before the harness runs."""
    sessions = frozenset(
        str(row["session_id"])
        for row in _rows(source, "SELECT session_id::VARCHAR AS session_id FROM sessions")
    )
    retrievals = frozenset(
        str(row["retrieval_id"])
        for row in _rows(source, "SELECT retrieval_id::VARCHAR AS retrieval_id FROM retrievals")
    )
    links = frozenset(
        str(row["link_id"])
        for row in _rows(source, "SELECT link_id::VARCHAR AS link_id FROM context_links")
    )
    # Typed-column filter matches capture/attribution.py so we never scan full events.payload.
    placeholders = frozenset(
        str(row["event_id"])
        for row in _rows(
            source,
            "SELECT event_id::VARCHAR AS event_id FROM events "
            "WHERE kind = 'assistant_msg' AND content IS NULL "
            "AND coalesce(try_cast(payload.placeholder AS BOOLEAN), false)",
        )
    )
    return TrialLedgerBaseline(
        session_ids=sessions,
        retrieval_ids=retrievals,
        link_ids=links,
        placeholder_event_ids=placeholders,
    )


def import_trial_transcript(
    cfg: Config,
    transcript_path: Path | None,
    *,
    harness: str = "droid",
    memory_server_name: str = DEFAULT_SERVER_NAME,
) -> tuple[TranscriptImportSummary, list[Record]]:
    """Import a memory-arm trial's transcript into its trial home via the capture path.

    Sets ``AGENT_MEMORY_MCP_SERVER_NAME`` temporarily so attribution matches
    ``memory_server_name`` during both transcript parsing and answer binding.
    """
    harness_name = harness if harness in HARNESS_NAMES else "droid"
    if transcript_path is None or not transcript_path.is_file():
        return (
            TranscriptImportSummary(
                status="missing",
                harness=harness_name,
                transcript_path=str(transcript_path) if transcript_path is not None else None,
                records_parsed=0,
                inserted=0,
                skipped=0,
                spooled=0,
                failed_lines=0,
                error="transcript file not found",
            ),
            [],
        )

    prev_server = os.environ.get("AGENT_MEMORY_MCP_SERVER_NAME")
    os.environ["AGENT_MEMORY_MCP_SERVER_NAME"] = memory_server_name
    try:
        try:
            adapter = get_adapter(harness_name)
            records = list(adapter.from_transcript(transcript_path))
        except Exception as exc:
            return (
                TranscriptImportSummary(
                    status="failed",
                    harness=harness_name,
                    transcript_path=str(transcript_path),
                    records_parsed=0,
                    inserted=0,
                    skipped=0,
                    spooled=0,
                    failed_lines=0,
                    error=f"{type(exc).__name__}: {exc}",
                ),
                [],
            )

        if not records:
            return (
                TranscriptImportSummary(
                    status="empty",
                    harness=harness_name,
                    transcript_path=str(transcript_path),
                    records_parsed=0,
                    inserted=0,
                    skipped=0,
                    spooled=0,
                    failed_lines=0,
                ),
                [],
            )

        try:
            delivery = deliver(cfg, records, spool_only=False, deadline=IMPORT_DEADLINE)
        except Exception as exc:
            return (
                TranscriptImportSummary(
                    status="failed",
                    harness=harness_name,
                    transcript_path=str(transcript_path),
                    records_parsed=len(records),
                    inserted=0,
                    skipped=0,
                    spooled=0,
                    failed_lines=0,
                    error=f"{type(exc).__name__}: {exc}",
                ),
                records,
            )

        written = delivery.written
        inserted = written.inserted if written is not None else 0
        skipped = written.skipped if written is not None else 0
        spooled_count = len(delivery.spooled)
        rejected_count = (
            len(written.rejected) if written is not None else len(delivery.failed_paths)
        )
        status: TranscriptImportStatus = (
            "imported"
            if written is not None and spooled_count == 0 and not delivery.error
            else "failed"
        )
        return (
            TranscriptImportSummary(
                status=status,
                harness=harness_name,
                transcript_path=str(transcript_path),
                records_parsed=len(records),
                inserted=inserted,
                skipped=skipped,
                spooled=spooled_count,
                failed_lines=rejected_count,
                error=delivery.error,
            ),
            records,
        )
    finally:
        if prev_server is None:
            os.environ.pop("AGENT_MEMORY_MCP_SERVER_NAME", None)
        else:
            os.environ["AGENT_MEMORY_MCP_SERVER_NAME"] = prev_server


# ---------------------------------------------------------------------------
# Recall and citation accounting
# ---------------------------------------------------------------------------


def _extract_transcript_tool_pairs_from_records(
    records: Sequence[Record],
) -> list[tuple[str | None, Mapping[str, Any] | None, bool, Any, bool]]:
    """Return tool call/result pairs from parsed capture Records.

    Each entry is ``(call_id, call_payload, has_result, result_content, result_is_error)``.
    """
    results_by_parent: dict[str, Record] = {}
    for rec in records:
        if rec.kind == "tool_result" and rec.parent_key:
            results_by_parent[rec.parent_key] = rec

    entries: list[tuple[str | None, Mapping[str, Any] | None, bool, Any, bool]] = []
    for rec in records:
        if rec.kind != "tool_call":
            continue
        result = results_by_parent.get(rec.source_key)
        if result is None:
            entries.append((rec.source_key, rec.payload, False, None, False))
        else:
            is_error = bool((result.payload or {}).get("is_error"))
            entries.append((rec.source_key, rec.payload, True, result.content, is_error))
    return entries


def _extract_transcript_tool_pairs_from_ledger(
    source: LedgerSource,
    trial_session_ids: Sequence[str],
    *,
    blob_dir: Path | None = None,
) -> list[tuple[str | None, Mapping[str, Any] | None, bool, Any, bool]]:
    """Fallback when no transcript file/records were passed: read imported tool events."""
    if not trial_session_ids:
        return []
    session_literals = ", ".join(sql_literal(sid) for sid in trial_session_ids)
    rows = _rows(
        source,
        "SELECT event_id::VARCHAR AS event_id, kind, parent_event_id::VARCHAR AS parent_event_id, "
        "content, truncated, blob_ref, payload::JSON::VARCHAR AS payload_json "
        f"FROM events WHERE session_id IN ({session_literals}) "
        "AND kind IN ('tool_call', 'tool_result') "
        "ORDER BY ts, recorded_at, seq, event_id",
    )
    results_by_parent: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row["kind"] == "tool_result" and row["parent_event_id"]:
            results_by_parent[str(row["parent_event_id"])] = row

    entries: list[tuple[str | None, Mapping[str, Any] | None, bool, Any, bool]] = []
    for row in rows:
        if row["kind"] != "tool_call":
            continue
        call_id = str(row["event_id"])
        call_payload = json.loads(row["payload_json"]) if row["payload_json"] else {}
        res = results_by_parent.get(call_id)
        if res is None:
            entries.append((call_id, call_payload, False, None, False))
            continue
        res_payload = json.loads(res["payload_json"]) if res["payload_json"] else {}
        content = res["content"]
        if res.get("truncated") and res.get("blob_ref") and blob_dir is not None:
            blob_path = blob_dir / str(res["blob_ref"])
            if blob_path.is_file():
                content = blob_path.read_text(encoding="utf-8", errors="replace")
        entries.append((call_id, call_payload, True, content, bool(res_payload.get("is_error"))))
    return entries


def _lookup_events_by_id(
    source: LedgerSource,
    event_ids: Sequence[str],
) -> dict[str, dict[str, Any]]:
    found: dict[str, dict[str, Any]] = {}
    ordered = sorted(set(event_ids))
    for start in range(0, len(ordered), _CHUNK_SIZE):
        literals = ", ".join(sql_literal(eid) for eid in ordered[start : start + _CHUNK_SIZE])
        for row in _rows(
            source,
            "SELECT event_id::VARCHAR AS event_id, kind, "
            "coalesce(try_cast(payload.placeholder AS BOOLEAN), false) AS placeholder, "
            "coalesce(try_cast(payload.part AS VARCHAR) = 'thinking', false) AS thinking "
            f"FROM events WHERE event_id IN ({literals})",
        ):
            found[str(row["event_id"])] = row
    return found


def _is_actual_reply(event: Mapping[str, Any] | None) -> bool:
    return (
        event is not None
        and event["kind"] == "assistant_msg"
        and not bool(event["placeholder"])
        and not bool(event["thinking"])
    )


def _account_from_source(
    source: LedgerSource,
    *,
    transcript_path: Path | None,
    transcript_records: Sequence[Record] | None,
    harness: str,
    memory_server_name: str,
    baseline: TrialLedgerBaseline,
    transcript_import: TranscriptImportSummary | None,
    blob_dir: Path | None,
) -> RecallAccounting:
    # 1. Trial-scoped sessions.
    all_sessions = [
        str(row["session_id"])
        for row in _rows(source, "SELECT session_id::VARCHAR AS session_id FROM sessions")
    ]
    trial_session_ids = tuple(
        sorted(sid for sid in all_sessions if sid not in baseline.session_ids)
    )

    # 2. Trial-scoped retrievals and retrieval_items.
    retrieval_rows = _rows(
        source,
        "SELECT retrieval_id::VARCHAR AS retrieval_id, session_id::VARCHAR AS session_id, "
        "strategy, returned, latency_ms "
        "FROM retrievals ORDER BY created_at, retrieval_id",
    )
    trial_retrievals: dict[str, dict[str, Any]] = {}
    for row in retrieval_rows:
        rid = str(row["retrieval_id"])
        if rid not in baseline.retrieval_ids:
            trial_retrievals[rid] = row

    trial_retrieval_items: dict[str, list[str]] = defaultdict(list)
    ordered_rids = sorted(trial_retrievals)
    for start in range(0, len(ordered_rids), _CHUNK_SIZE):
        literals = ", ".join(sql_literal(rid) for rid in ordered_rids[start : start + _CHUNK_SIZE])
        for row in _rows(
            source,
            "SELECT retrieval_id::VARCHAR AS retrieval_id, memory_id::VARCHAR AS memory_id "
            f"FROM retrieval_items WHERE retrieval_id IN ({literals}) "
            "ORDER BY retrieval_id, rank",
        ):
            trial_retrieval_items[str(row["retrieval_id"])].append(str(row["memory_id"]))

    # 3. Transcript tool calls (prefer passed records, then transcript_path, then imported events).
    parsed_records: Sequence[Record] | None = transcript_records
    if parsed_records is None and transcript_path is not None and transcript_path.is_file():
        harness_name = harness if harness in HARNESS_NAMES else "droid"
        parsed_records = get_adapter(harness_name).from_transcript(transcript_path)

    if parsed_records is not None:
        tool_entries = _extract_transcript_tool_pairs_from_records(parsed_records)
    else:
        tool_entries = _extract_transcript_tool_pairs_from_ledger(
            source, trial_session_ids, blob_dir=blob_dir
        )

    # 4. Reconcile transcript recall calls against trial retrievals.
    recalls: list[RecallCallRecord] = []
    matched_retrieval_ids: set[str] = set()
    cite_calls_in_transcript = 0
    degraded_cite_calls = 0

    for call_id, payload, has_result, result_content, is_error in tool_entries:
        if is_memory_tool_call(payload, "cite", server_name=memory_server_name):
            cite_calls_in_transcript += 1
            decoded_cite = _decode_tool_result_dict(result_content) if has_result else None
            if decoded_cite is not None and decoded_cite.get("degraded") is True:
                degraded_cite_calls += 1
            continue

        op = _recall_operation(payload, server_name=memory_server_name)
        if op is None:
            continue

        if not has_result:
            recalls.append(
                RecallCallRecord(
                    outcome="missing_record",
                    source="transcript_only",
                    operation=op,
                    call_id=call_id,
                    retrieval_id=None,
                    returned_count=None,
                    returned_memory_ids=(),
                    strategy=None,
                    latency_ms=None,
                    detail="missing_tool_result",
                )
            )
            continue

        decoded = _decode_tool_result_dict(result_content)
        if decoded is not None and decoded.get("degraded") is True:
            recalls.append(
                RecallCallRecord(
                    outcome="degraded",
                    source="transcript",
                    operation=op,
                    call_id=call_id,
                    retrieval_id=None,
                    returned_count=None,
                    returned_memory_ids=(),
                    strategy=None,
                    latency_ms=None,
                    detail="degraded_tool_result",
                )
            )
            continue

        rid = _uuid(decoded.get("retrieval_id")) if decoded is not None else None
        if rid is not None and rid in trial_retrievals:
            matched_retrieval_ids.add(rid)
            row = trial_retrievals[rid]
            ret_count = int(row["returned"])
            outcome: RecallOutcome = "returned" if ret_count > 0 else "miss"
            latency = float(row["latency_ms"]) if row["latency_ms"] is not None else None
            recalls.append(
                RecallCallRecord(
                    outcome=outcome,
                    source="transcript_and_ledger",
                    operation=op,
                    call_id=call_id,
                    retrieval_id=rid,
                    returned_count=ret_count,
                    returned_memory_ids=tuple(trial_retrieval_items.get(rid, ())),
                    strategy=str(row["strategy"]) if row["strategy"] is not None else None,
                    latency_ms=latency,
                )
            )
            continue

        # Any recall shown in the transcript without a matching trial retrieval
        # in the ledger is a missing_record, never a miss.
        if rid is not None:
            detail = "retrieval_not_in_ledger"
        elif is_error:
            detail = "tool_error"
        else:
            detail = "no_retrieval_id"
        recalls.append(
            RecallCallRecord(
                outcome="missing_record",
                source="transcript_only",
                operation=op,
                call_id=call_id,
                retrieval_id=rid,
                returned_count=None,
                returned_memory_ids=(),
                strategy=None,
                latency_ms=None,
                detail=detail,
            )
        )

    # 5. Include any trial ledger retrievals not paired with a transcript call.
    for rid, row in trial_retrievals.items():
        if rid in matched_retrieval_ids:
            continue
        ret_count = int(row["returned"])
        outcome = "returned" if ret_count > 0 else "miss"
        strategy_str = str(row["strategy"]) if row["strategy"] is not None else None
        op_guess = (
            "graph"
            if strategy_str
            and ("graph" in strategy_str.lower() or "relation" in strategy_str.lower())
            else "recall"
        )
        latency = float(row["latency_ms"]) if row["latency_ms"] is not None else None
        recalls.append(
            RecallCallRecord(
                outcome=outcome,
                source="ledger_only",
                operation=op_guess,
                call_id=None,
                retrieval_id=rid,
                returned_count=ret_count,
                returned_memory_ids=tuple(trial_retrieval_items.get(rid, ())),
                strategy=strategy_str,
                latency_ms=latency,
            )
        )

    # 6. Trial-scoped context_links, placeholders, and effective_answer_bindings.
    link_rows = _rows(
        source,
        "SELECT link_id::VARCHAR AS link_id, answer_event_id::VARCHAR AS answer_event_id, "
        "link_type, retrieval_id::VARCHAR AS retrieval_id, memory_id::VARCHAR AS memory_id, "
        "context_event_id::VARCHAR AS context_event_id "
        "FROM context_links ORDER BY created_at, link_id",
    )
    trial_links = [row for row in link_rows if str(row["link_id"]) not in baseline.link_ids]

    placeholder_rows = _rows(
        source,
        "SELECT event_id::VARCHAR AS event_id FROM events "
        "WHERE kind = 'assistant_msg' AND content IS NULL "
        "AND coalesce(try_cast(payload.placeholder AS BOOLEAN), false)",
    )
    trial_placeholders: set[str] = {
        str(row["event_id"])
        for row in placeholder_rows
        if str(row["event_id"]) not in baseline.placeholder_event_ids
    }

    # Also verify event metadata for any answer_event_id directly referenced by trial links.
    link_answer_ids = {str(row["answer_event_id"]) for row in trial_links}
    direct_events = _lookup_events_by_id(source, list(link_answer_ids))
    for eid, ev in direct_events.items():
        if (
            ev["kind"] == "assistant_msg"
            and bool(ev["placeholder"])
            and eid not in baseline.placeholder_event_ids
        ):
            trial_placeholders.add(eid)

    # Query effective_answer_bindings for all trial placeholders.
    bindings_by_placeholder: dict[str, tuple[str | None, str | None]] = {}
    ordered_placeholders = sorted(trial_placeholders)
    for start in range(0, len(ordered_placeholders), _CHUNK_SIZE):
        literals = ", ".join(
            sql_literal(pid) for pid in ordered_placeholders[start : start + _CHUNK_SIZE]
        )
        for row in _rows(
            source,
            "SELECT placeholder_event_id::VARCHAR AS placeholder_id, "
            "answer_event_id::VARCHAR AS answer_id, reason_code "
            f"FROM effective_answer_bindings WHERE placeholder_event_id IN ({literals})",
        ):
            answer_id = str(row["answer_id"]) if row["answer_id"] is not None else None
            reason = str(row["reason_code"]) if row["reason_code"] is not None else None
            bindings_by_placeholder[str(row["placeholder_id"])] = (answer_id, reason)

    bound_target_ids = [aid for aid, _ in bindings_by_placeholder.values() if aid is not None]
    reply_events = _lookup_events_by_id(source, bound_target_ids)
    reply_events.update(direct_events)

    # Group trial context_links by original answer_event_id (plus any placeholder with 0 links).
    links_by_orig: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for link in trial_links:
        links_by_orig[str(link["answer_event_id"])].append(link)
    for pid in ordered_placeholders:
        links_by_orig.setdefault(pid, [])

    citation_records: list[CitationRecord] = []
    citations_per_reply: dict[str, list[str]] = defaultdict(list)
    unbound_placeholder_ids: list[str] = []
    bound_placeholder_count = 0

    for orig_id in sorted(links_by_orig):
        group = links_by_orig[orig_id]
        is_placeholder = orig_id in trial_placeholders
        bound_answer_id: str | None
        binding_reason: str | None
        if is_placeholder:
            if orig_id in bindings_by_placeholder:
                candidate_aid, binding_reason = bindings_by_placeholder[orig_id]
                if candidate_aid is not None and _is_actual_reply(reply_events.get(candidate_aid)):
                    bound_answer_id = candidate_aid
                else:
                    bound_answer_id = None
                    if candidate_aid is not None:
                        binding_reason = "invalid_answer_target"
            else:
                bound_answer_id = None
                binding_reason = "no_binding"
            if bound_answer_id is None:
                unbound_placeholder_ids.append(orig_id)
            else:
                bound_placeholder_count += 1
                citations_per_reply[bound_answer_id].append(orig_id)
        else:
            if _is_actual_reply(reply_events.get(orig_id)):
                bound_answer_id = orig_id
                binding_reason = "direct_answer_id"
                citations_per_reply[bound_answer_id].append(orig_id)
            else:
                bound_answer_id = None
                binding_reason = "invalid_answer_target"

        retrieval_ids = tuple(
            sorted(
                {
                    str(link["retrieval_id"])
                    for link in group
                    if link["link_type"] == "retrieval" and link["retrieval_id"] is not None
                }
            )
        )
        cited_mids = tuple(
            sorted(
                {
                    str(link["memory_id"])
                    for link in group
                    if link["link_type"] == "cited_memory" and link["memory_id"] is not None
                }
            )
        )
        context_eids = tuple(
            sorted(
                {
                    str(link["context_event_id"])
                    for link in group
                    if link["link_type"] == "context_event" and link["context_event_id"] is not None
                }
            )
        )
        citation_records.append(
            CitationRecord(
                original_answer_event_id=orig_id,
                is_placeholder=is_placeholder,
                bound_answer_event_id=bound_answer_id,
                binding_reason=binding_reason,
                retrieval_ids=retrieval_ids,
                cited_memory_ids=cited_mids,
                context_event_ids=context_eids,
            )
        )

    returned_memory_ids = tuple(
        sorted({mid for mids in trial_retrieval_items.values() for mid in mids})
    )
    cited_memory_ids = tuple(
        sorted(
            {
                str(link["memory_id"])
                for link in trial_links
                if link["link_type"] == "cited_memory" and link["memory_id"] is not None
            }
        )
    )
    returned_memories_cited_ids = tuple(sorted(set(returned_memory_ids) & set(cited_memory_ids)))
    single_citation_reply_ids = tuple(
        sorted(aid for aid, origs in citations_per_reply.items() if len(origs) == 1)
    )
    multi_citation_reply_ids = tuple(
        sorted(aid for aid, origs in citations_per_reply.items() if len(origs) > 1)
    )

    citations = CitationAccounting(
        citations_made=len(citation_records),
        cite_calls_in_transcript=cite_calls_in_transcript,
        degraded_cite_calls=degraded_cite_calls,
        citation_links=len(trial_links),
        retrieval_links=sum(1 for link in trial_links if link["link_type"] == "retrieval"),
        cited_memory_links=sum(1 for link in trial_links if link["link_type"] == "cited_memory"),
        context_event_links=sum(1 for link in trial_links if link["link_type"] == "context_event"),
        returned_memory_ids=returned_memory_ids,
        cited_memory_ids=cited_memory_ids,
        returned_memories_cited_count=len(returned_memories_cited_ids),
        returned_memories_cited_ids=returned_memories_cited_ids,
        placeholder_count=bound_placeholder_count + len(unbound_placeholder_ids),
        bound_placeholder_count=bound_placeholder_count,
        unbound_placeholder_count=len(unbound_placeholder_ids),
        unbound_placeholder_ids=tuple(sorted(unbound_placeholder_ids)),
        bound_reply_count=len(citations_per_reply),
        single_citation_reply_count=len(single_citation_reply_ids),
        single_citation_reply_ids=single_citation_reply_ids,
        multi_citation_reply_count=len(multi_citation_reply_ids),
        multi_citation_reply_ids=multi_citation_reply_ids,
        citations=tuple(citation_records),
    )

    returned_count = sum(1 for r in recalls if r.outcome == "returned")
    miss_count = sum(1 for r in recalls if r.outcome == "miss")
    degraded_count = sum(1 for r in recalls if r.outcome == "degraded")
    missing_record_count = sum(1 for r in recalls if r.outcome == "missing_record")
    latencies = [r.latency_ms for r in recalls if r.latency_ms is not None]
    total_latency_ms = round(sum(latencies), 6) if latencies else None

    effective_import = transcript_import or TranscriptImportSummary(
        status="skipped",
        harness=harness if harness in HARNESS_NAMES else "droid",
        transcript_path=str(transcript_path) if transcript_path is not None else None,
        records_parsed=len(parsed_records) if parsed_records is not None else 0,
        inserted=0,
        skipped=0,
        spooled=0,
        failed_lines=0,
    )

    return RecallAccounting(
        made_no_recall=len(recalls) == 0,
        total_recalls=len(recalls),
        returned_count=returned_count,
        miss_count=miss_count,
        degraded_count=degraded_count,
        missing_record_count=missing_record_count,
        total_retrieval_latency_ms=total_latency_ms,
        trial_session_ids=trial_session_ids,
        recalls=tuple(recalls),
        citations=citations,
        transcript_import=effective_import,
    )


def account_trial_recall(
    source: LedgerSource | Config | Path,
    *,
    transcript_path: Path | None = None,
    transcript_records: Sequence[Record] | None = None,
    harness: str = "droid",
    memory_server_name: str = DEFAULT_SERVER_NAME,
    baseline: TrialLedgerBaseline | None = None,
    transcript_import: TranscriptImportSummary | None = None,
) -> RecallAccounting:
    """Count what recall and citations did in a memory-arm trial.

    ``source`` may be a live ``MemoryClient``, an open DuckDB connection, a
    ``Config``, or the path to a stopped trial home directory (opened read-only).
    """
    effective_baseline = baseline if baseline is not None else TrialLedgerBaseline.empty()

    if isinstance(source, (Config, Path)):
        home_path = source.home if isinstance(source, Config) else source
        db_path = home_path / "memory.duckdb"
        if not db_path.is_file():
            raise UsefulnessError(f"trial home database not found: {db_path}")
        con = open_direct(db_path, read_only=True)
        try:
            return _account_from_source(
                con,
                transcript_path=transcript_path,
                transcript_records=transcript_records,
                harness=harness,
                memory_server_name=memory_server_name,
                baseline=effective_baseline,
                transcript_import=transcript_import,
                blob_dir=home_path / "blobs",
            )
        finally:
            con.close()

    blob_dir: Path | None = None
    return _account_from_source(
        source,
        transcript_path=transcript_path,
        transcript_records=transcript_records,
        harness=harness,
        memory_server_name=memory_server_name,
        baseline=effective_baseline,
        transcript_import=transcript_import,
        blob_dir=blob_dir,
    )
