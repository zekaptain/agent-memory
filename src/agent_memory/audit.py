"""Read-only ledger diagnostics and conservative reconstruction of legacy replacements.

No content, payloads, review reasons, tokens, or user-supplied metadata are returned.
The legacy helper proves historical acceptance, not current eligibility: a later
quarantine or retirement must never resurrect an accepted predecessor.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any, Literal

import duckdb

from .db import MemoryClient

type AuditSource = MemoryClient | duckdb.DuckDBPyConnection
type Row = dict[str, Any]


@dataclass(frozen=True)
class Finding:
    code: str
    entity_type: str
    entity_id: str
    related_ids: tuple[str, ...] = ()
    severity: Literal["error", "warning"] = "error"


@dataclass(frozen=True)
class LegacyAcceptance:
    """Evidence for an edge, not a synthetic ``approve`` memory action.

    ``trusted_at_creation`` uses the replacement row and legacy non-tool default.
    ``explicit_approval`` additionally names the existing approval action.
    Both require valid endpoints and an eligible, unambiguous predecessor.
    """

    predecessor_id: str
    replacement_id: str
    accepted_at: datetime
    method: Literal["trusted_at_creation", "explicit_approval"]
    evidence_action_id: str | None

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "accepted_at": self.accepted_at.astimezone(UTC).isoformat()}


@dataclass(frozen=True)
class LegacyPreflight:
    """Migration input; callers must abort on *any* blocker before writing anything."""

    acceptances: tuple[LegacyAcceptance, ...]
    blockers: tuple[Finding, ...]

    @property
    def ok(self) -> bool:
        return not self.blockers


@dataclass(frozen=True)
class AuditReport:
    schema_versions: tuple[int, ...] = ()
    findings: tuple[Finding, ...] = ()
    legacy_acceptances: tuple[LegacyAcceptance, ...] = ()

    @property
    def exit_code(self) -> int:
        """0: clean; 1: findings (including coverage warnings); 2: incomplete audit."""
        if any(f.code in {"audit_read_failed", "schema_missing_columns"} for f in self.findings):
            return 2
        return int(bool(self.findings))

    @property
    def status(self) -> str:
        return ("clean", "findings", "error")[self.exit_code]

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "status": self.status,
            "exit_code": self.exit_code,
            "schema_versions": list(self.schema_versions),
            "summary": {
                "errors": sum(f.severity == "error" for f in self.findings),
                "warnings": sum(f.severity == "warning" for f in self.findings),
            },
            "findings": [{**asdict(f), "related_ids": list(f.related_ids)} for f in self.findings],
            "legacy_acceptances": [a.to_dict() for a in self.legacy_acceptances],
        }


def read_failure() -> AuditReport:
    """A fixed diagnostic: database/connection exception text may contain secrets."""
    return AuditReport(findings=(Finding("audit_read_failed", "audit", "check"),))


_LEGACY_COLUMNS = {
    "memories": {
        "memory_id",
        "scope",
        "trust_level",
        "supersedes",
        "valid_from",
        "valid_to",
        "expires_at",
        "created_at",
    },
    "memory_actions": {"action_id", "memory_id", "action", "created_at"},
}
_AUDIT_COLUMNS = {
    **_LEGACY_COLUMNS,
    "sessions": {"session_id", "capture_completeness"},
    "events": {"event_id", "session_id", "kind", "parent_event_id", "payload"},
    "retrievals": {"retrieval_id", "session_id", "request_event_id", "returned"},
    "retrieval_items": {"retrieval_id", "memory_id", "rank", "status_at_recall"},
    "context_links": {"answer_event_id", "link_type", "retrieval_id"},
}
_LATER_COLUMNS = {
    "memory_replacements": {
        "predecessor_id",
        "successor_id",
        "evidence_action_id",
        "evidence_event_id",
    },
    "capture_observations": {"observation_id", "session_id"},
    "answer_bindings": {"binding_id", "placeholder_event_id", "answer_event_id"},
}
_COVERAGE_COLUMNS = {
    "session_summary": {
        "session_id",
        "capture_completeness",
        "recorded_capture_completeness",
        "coverage_observation_id",
        "coverage_gap",
        "coverage_from",
        "started_at",
        "coverage_parse_errors",
        "coverage_exclusions",
    }
}
_BINDING_COLUMNS = {
    "effective_answer_bindings": {"binding_id", "placeholder_event_id", "answer_event_id"}
}


def _rows(source: AuditSource, sql: str) -> list[Row]:
    if isinstance(source, MemoryClient):
        columns, rows = source.query(sql)
    else:
        cursor = source.execute(sql)
        columns = [d[0] for d in cursor.description]
        rows = cursor.fetchall()
    return [dict(zip(columns, row, strict=True)) for row in rows]


def _schema(source: AuditSource) -> dict[str, set[str]]:
    tables: dict[str, set[str]] = defaultdict(set)
    for row in _rows(
        source,
        "SELECT table_name, column_name FROM duckdb_columns() "
        "WHERE database_name = current_database() AND schema_name = 'main'",
    ):
        tables[row["table_name"]].add(row["column_name"])
    return tables


def _missing(schema: dict[str, set[str]], required: dict[str, set[str]]) -> tuple[Finding, ...]:
    # Only fixed application table/column names enter the report, never arbitrary schema text.
    return tuple(
        Finding(
            "schema_missing_columns",
            "table",
            table,
            tuple(sorted(columns - schema.get(table, set()))),
        )
        for table, columns in sorted(required.items())
        if not columns <= schema.get(table, set())
    )


def _ordered(findings: list[Finding]) -> tuple[Finding, ...]:
    return tuple(
        sorted(set(findings), key=lambda f: (f.code, f.entity_type, f.entity_id, f.related_ids))
    )


def legacy_preflight(source: AuditSource, *, as_of: datetime | None = None) -> LegacyPreflight:
    """Read legacy rows without migrations or writes; return evidence and blockers.

    This also works inside a migration's existing transaction. It neither starts nor
    commits one. No dependency on status views or later lifecycle tables is required.
    Multiple claimants, cycles, cross-scope edges, inactive predecessors, and ordering
    that cannot be proved from legacy review history block automatic reconstruction.
    A future validity boundary alone is not an acceptance event.
    """
    missing = _missing(_schema(source), _LEGACY_COLUMNS)
    if missing:
        return LegacyPreflight((), missing)
    return _legacy(source, as_of=as_of or datetime.now(UTC))


def _legacy(source: AuditSource, *, as_of: datetime) -> LegacyPreflight:
    memories = {
        row["memory_id"]: row
        for row in _rows(
            source,
            "SELECT memory_id::VARCHAR AS memory_id, scope, trust_level, "
            "supersedes::VARCHAR AS supersedes, valid_from, valid_to, expires_at, created_at "
            "FROM memories",
        )
    }
    reviews: dict[str, list[Row]] = defaultdict(list)
    for row in _rows(
        source,
        "SELECT action_id::VARCHAR AS action_id, memory_id::VARCHAR AS memory_id, "
        "action, created_at, rowid AS action_order FROM memory_actions "
        "WHERE action IN ('approve', 'quarantine', 'retire') ORDER BY created_at, rowid",
    ):
        reviews[row["memory_id"]].append(row)
    children: dict[str, list[str]] = defaultdict(list)
    invalid = set()
    findings = []
    for mid, row in memories.items():
        start = row["valid_from"]
        if start is not None and any(
            row[bound] is not None and row[bound] <= start for bound in ("valid_to", "expires_at")
        ):
            invalid.add(mid)
            findings.append(Finding("invalid_validity_interval", "memory", mid))
        if any(a["created_at"] < row["created_at"] for a in reviews[mid]):
            invalid.add(mid)
            findings.append(Finding("legacy_review_before_creation", "memory", mid))
        if row["supersedes"]:
            children[row["supersedes"]].append(mid)
    cycles = _cycle_members(memories)
    acceptances = []
    for mid, row in sorted(memories.items()):
        predecessor_id = row["supersedes"]
        if predecessor_id is None:
            continue
        related = (predecessor_id,)
        reasons = []
        predecessor = memories.get(predecessor_id)
        if predecessor is None:
            reasons.append("legacy_predecessor_missing")
        elif row["scope"] != predecessor["scope"]:
            reasons.append("legacy_replacement_cross_scope")
        if len(children[predecessor_id]) > 1:
            reasons.append("legacy_replacement_multiple")
        if mid in cycles:
            reasons.append("legacy_replacement_cycle")
        if mid in invalid or predecessor_id in invalid:
            reasons.append("legacy_acceptance_ambiguous")
        if not reasons and predecessor is not None:
            candidate = _acceptance_candidate(row, reviews[mid], as_of)
            if candidate is None:
                reasons.append("legacy_acceptance_ambiguous")
            else:
                at, method, evidence = candidate
                # rowid orders actions within one table, not a memory creation against
                # its predecessor's review. Do not invent a cross-table tie breaker.
                predecessor_reviews = reviews[predecessor_id]
                if evidence is not None:
                    approval = next(a for a in reviews[mid] if a["action_id"] == evidence)
                    predecessor_reviews = [
                        a
                        for a in predecessor_reviews
                        if (a["created_at"], a["action_order"]) < (at, approval["action_order"])
                    ]
                if evidence is None and any(a["created_at"] == at for a in predecessor_reviews):
                    reasons.append("legacy_review_order_ambiguous")
                elif not _eligible(predecessor, predecessor_reviews, at):
                    reasons.append("legacy_replacement_stale")
                else:
                    acceptances.append(LegacyAcceptance(predecessor_id, mid, at, method, evidence))
        findings.extend(Finding(code, "memory", mid, related) for code in reasons)
    acceptances.sort(key=lambda a: (a.accepted_at, a.predecessor_id, a.replacement_id))
    return LegacyPreflight(tuple(acceptances), _ordered(findings))


def _valid_at(memory: Row, at: datetime) -> bool:
    return (
        memory["created_at"] <= at
        and (memory["valid_from"] is None or memory["valid_from"] <= at)
        and all(memory[b] is None or at < memory[b] for b in ("valid_to", "expires_at"))
    )


def _eligible(memory: Row, actions: list[Row], at: datetime) -> bool:
    if not _valid_at(memory, at):
        return False
    before = [a for a in actions if a["created_at"] <= at]
    if any(a["action"] == "retire" for a in before):
        return False
    reviews = [a for a in before if a["action"] in {"approve", "quarantine"}]
    if reviews:
        return reviews[-1]["action"] == "approve"
    return memory["trust_level"] in {"user", "agent", "imported"}


def _acceptance_candidate(
    memory: Row, actions: list[Row], as_of: datetime
) -> tuple[datetime, Literal["trusted_at_creation", "explicit_approval"], str | None] | None:
    created = memory["created_at"]
    if created <= as_of and _eligible(memory, actions, created):
        approvals = [a for a in actions if a["created_at"] <= created and a["action"] == "approve"]
        if approvals:
            return created, "explicit_approval", approvals[0]["action_id"]
        return created, "trusted_at_creation", None
    for index, review in enumerate(actions):
        at = review["created_at"]
        if (
            review["action"] == "approve"
            and created <= at <= as_of
            and _eligible(memory, actions[: index + 1], at)
        ):
            return at, "explicit_approval", review["action_id"]
    return None


def _cycle_members(memories: dict[str, Row]) -> set[str]:
    """Walk the functional graph iteratively, including very long legacy chains."""
    done: set[str] = set()
    cycles: set[str] = set()
    for start in memories:
        path: dict[str, int] = {}
        node = start
        while node in memories and node not in done:
            if node in path:
                cycles.update(list(path)[path[node] :])
                break
            path[node] = len(path)
            node = memories[node]["supersedes"]
        done.update(path)
    return cycles


def check(source: AuditSource) -> AuditReport:
    """Inspect only ledger columns, adapting to 0001 or additive later schemas.

    Reads are observational, not a transaction-wide snapshot over Quack. Re-run a
    stopped ``--direct`` audit for a stable migration preflight. Session flags and
    plausible event shapes are assertions, not verified full-capture evidence.
    """
    schema = _schema(source)
    versions = (
        tuple(
            int(r["version"])
            for r in _rows(source, "SELECT version FROM schema_migrations ORDER BY version")
        )
        if "version" in schema.get("schema_migrations", set())
        else ()
    )
    missing = _missing(schema, _AUDIT_COLUMNS)
    missing += _missing(schema, {t: cols for t, cols in _LATER_COLUMNS.items() if t in schema})
    if "memory_replacements" in schema:
        missing += _missing(schema, {"memory_actions": {"related_memory_id"}})
    if "capture_observations" in schema:
        missing += _missing(schema, _COVERAGE_COLUMNS)
    if "answer_bindings" in schema:
        missing += _missing(schema, _BINDING_COLUMNS)
    if missing:
        return AuditReport(versions, missing)
    legacy = (
        LegacyPreflight((), _replacement_findings(source))
        if "memory_replacements" in schema
        else _legacy(source, as_of=datetime.now(UTC))
    )
    findings = list(legacy.blockers)
    events = {
        row["event_id"]: row
        for row in _rows(
            source,
            "SELECT event_id::VARCHAR AS event_id, session_id::VARCHAR AS session_id, kind, "
            "parent_event_id::VARCHAR AS parent_event_id, "
            "coalesce(try_cast(json_extract_string(payload::JSON, '$.placeholder') AS BOOLEAN), "
            "false) AS placeholder, "
            "coalesce(json_extract_string(payload::JSON, '$.part'), '') = 'thinking' "
            "AS thinking FROM events",
        )
    }
    resolved = _resolved_answers(source, events, findings) if "answer_bindings" in schema else {}
    _capture_findings(source, events, findings, resolved, observed="capture_observations" in schema)
    _retrieval_findings(source, events, findings, resolved)
    return AuditReport(versions, _ordered(findings), legacy.acceptances)


def _replacement_findings(source: AuditSource) -> tuple[Finding, ...]:
    """Later schemas distinguish proposal intent from durable accepted edges."""
    memories = {
        r["memory_id"]: r
        for r in _rows(
            source,
            "SELECT memory_id::VARCHAR AS memory_id, supersedes::VARCHAR AS supersedes, scope, "
            "valid_from, valid_to, expires_at FROM memories",
        )
    }
    proposals = {
        (r["predecessor_id"], r["successor_id"])
        for r in _rows(
            source,
            "SELECT related_memory_id::VARCHAR AS predecessor_id, "
            "memory_id::VARCHAR AS successor_id "
            "FROM memory_actions WHERE action = 'propose_correction'",
        )
    }
    accepted = {
        r["predecessor_id"]: r
        for r in _rows(
            source,
            "SELECT predecessor_id::VARCHAR AS predecessor_id, "
            "successor_id::VARCHAR AS successor_id, "
            "evidence_action_id::VARCHAR AS evidence_action_id, "
            "evidence_event_id::VARCHAR AS evidence_event_id FROM memory_replacements",
        )
    }
    findings = []
    for mid, memory in memories.items():
        start = memory["valid_from"]
        if start is not None and any(
            memory[b] is not None and memory[b] <= start for b in ("valid_to", "expires_at")
        ):
            findings.append(Finding("invalid_validity_interval", "memory", mid))
        predecessor = memory["supersedes"]
        if predecessor is not None and (predecessor, mid) not in proposals:
            findings.append(Finding("legacy_acceptance_ambiguous", "memory", mid, (predecessor,)))
    for predecessor, successor in sorted(proposals):
        parent, child = memories.get(predecessor), memories.get(successor)
        if parent is None or child is None:
            findings.append(
                Finding("replacement_endpoint_missing", "memory", successor, (predecessor,))
            )
        elif parent["scope"] != child["scope"]:
            findings.append(Finding("replacement_cross_scope", "memory", successor, (predecessor,)))
        current = accepted.get(predecessor)
        if current is not None and current["successor_id"] != successor:
            findings.append(
                Finding(
                    "replacement_proposal_stale", "memory", successor, (predecessor,), "warning"
                )
            )
    for predecessor, edge in accepted.items():
        successor = edge["successor_id"]
        if (predecessor, successor) not in proposals:
            findings.append(
                Finding("replacement_proposal_missing", "memory", successor, (predecessor,))
            )
        if edge["evidence_action_id"] is None and edge["evidence_event_id"] is None:
            findings.append(
                Finding("replacement_evidence_missing", "memory", successor, (predecessor,))
            )
    graph = {mid: {"supersedes": accepted.get(mid, {}).get("successor_id")} for mid in memories}
    for mid in _cycle_members(graph):
        findings.append(Finding("replacement_cycle", "memory", mid))
    return _ordered(findings)


def _actual_answer(event: Row | None) -> bool:
    return (
        event is not None
        and event["kind"] == "assistant_msg"
        and not event["placeholder"]
        and not event["thinking"]
    )


def _resolved_answers(
    source: AuditSource, events: dict[str, Row], findings: list[Finding]
) -> dict[str, str]:
    resolved = {}
    for binding in _rows(
        source,
        "SELECT binding_id::VARCHAR AS binding_id, "
        "placeholder_event_id::VARCHAR AS placeholder_id, "
        "answer_event_id::VARCHAR AS answer_id FROM effective_answer_bindings",
    ):
        placeholder, answer = (
            events.get(binding["placeholder_id"]),
            events.get(binding["answer_id"]),
        )
        if binding["answer_id"] is None:
            continue
        if (
            placeholder is not None
            and placeholder["placeholder"]
            and _actual_answer(answer)
            and answer is not None
            and placeholder["session_id"] == answer["session_id"]
        ):
            resolved[binding["placeholder_id"]] = binding["answer_id"]
        else:
            findings.append(Finding("answer_binding_invalid", "binding", binding["binding_id"]))
    return resolved


def _capture_findings(
    source: AuditSource,
    events: dict[str, Row],
    findings: list[Finding],
    resolved: dict[str, str],
    *,
    observed: bool,
) -> None:
    by_session: dict[str, list[Row]] = defaultdict(list)
    results = set()
    for eid, event in events.items():
        sid = event["session_id"]
        by_session[sid].append(event)
        if event["kind"] == "assistant_msg" and event["placeholder"] and eid not in resolved:
            findings.append(Finding("cite_placeholder_unresolved", "event", eid, (sid,), "warning"))
        if event["kind"] == "capture_error":
            findings.append(Finding("capture_error_recorded", "event", eid, (sid,), "warning"))
        if event["kind"] == "tool_result":
            parent = events.get(event["parent_event_id"])
            if parent is None or parent["kind"] != "tool_call" or parent["session_id"] != sid:
                findings.append(
                    Finding("capture_tool_result_unlinked", "event", eid, (sid,), "warning")
                )
            else:
                results.add(parent["event_id"])
    for eid, event in events.items():
        if event["kind"] == "tool_call" and eid not in results:
            findings.append(
                Finding(
                    "capture_tool_result_missing", "event", eid, (event["session_id"],), "warning"
                )
            )
    session_sql = (
        "SELECT session_id::VARCHAR AS session_id, capture_completeness, "
        "recorded_capture_completeness, coverage_observation_id::VARCHAR AS observation_id, "
        "coalesce(coverage_gap, true) OR coverage_from > started_at AS coverage_gap, "
        "coalesce(json_array_length(coverage_parse_errors::JSON), 0) > 0 AS parse_errors, "
        "coalesce(json_array_length(coverage_exclusions::JSON), 0) > 0 AS exclusions "
        "FROM session_summary"
        if observed
        else "SELECT session_id::VARCHAR AS session_id, capture_completeness FROM sessions"
    )
    for session in _rows(source, session_sql):
        sid = session["session_id"]
        completeness = session["capture_completeness"]
        verified = (
            observed
            and completeness == "full"
            and not any(session[f] for f in ("coverage_gap", "parse_errors", "exclusions"))
        )
        if observed and completeness == "unverified_legacy_assertion":
            completeness = session["recorded_capture_completeness"]
        code = {
            "full": "capture_full_unverified",
            "partial": "capture_partial",
            "transcript_import": "capture_transcript_unverified",
        }.get(completeness, "capture_coverage_unverified")
        if observed and completeness == "transcript_import" and session["observation_id"]:
            code = "capture_transcript_partial"
        if not verified:
            findings.append(Finding(code, "session", sid, severity="warning"))
        if observed:
            related = (session["observation_id"],) if session["observation_id"] else ()
            for flag, reason in (
                ("coverage_gap", "capture_coverage_gap"),
                ("parse_errors", "capture_parse_errors"),
                ("exclusions", "capture_exclusions"),
            ):
                if session[flag]:
                    findings.append(Finding(reason, "session", sid, related, "warning"))
        session_events = by_session[sid]
        if not session_events:
            findings.append(Finding("capture_empty_session", "session", sid, severity="warning"))
        user = any(e["kind"] == "user_msg" for e in session_events)
        assistant = any(_actual_answer(e) for e in session_events)
        if user and not assistant:
            findings.append(
                Finding("capture_missing_assistant", "session", sid, severity="warning")
            )
        if assistant and not user:
            findings.append(Finding("capture_missing_user", "session", sid, severity="warning"))


def _retrieval_findings(
    source: AuditSource, events: dict[str, Row], findings: list[Finding], resolved: dict[str, str]
) -> None:
    items: dict[str, list[Row]] = defaultdict(list)
    for item in _rows(
        source,
        "SELECT retrieval_id::VARCHAR AS retrieval_id, memory_id::VARCHAR AS memory_id, "
        "rank, status_at_recall FROM retrieval_items",
    ):
        items[item["retrieval_id"]].append(item)
    links: dict[str, list[str]] = defaultdict(list)
    for link in _rows(
        source,
        "SELECT retrieval_id::VARCHAR AS retrieval_id, answer_event_id::VARCHAR AS answer_event_id "
        "FROM context_links WHERE link_type = 'retrieval'",
    ):
        eid = link["answer_event_id"]
        links[link["retrieval_id"]].append(resolved.get(eid, eid))
    for retrieval in _rows(
        source,
        "SELECT retrieval_id::VARCHAR AS retrieval_id, session_id::VARCHAR AS session_id, "
        "request_event_id::VARCHAR AS request_event_id, returned FROM retrievals",
    ):
        rid, sid = retrieval["retrieval_id"], retrieval["session_id"]
        request = events.get(retrieval["request_event_id"])
        if request is None:
            findings.append(Finding("retrieval_request_missing", "retrieval", rid, (sid,)))
        elif request["kind"] != "memory_recall" or request["session_id"] != sid:
            findings.append(Finding("retrieval_request_invalid", "retrieval", rid, (sid,)))
        attributed = any(
            answer is not None and _actual_answer(answer) and answer["session_id"] == sid
            for answer in (events.get(eid) for eid in links[rid])
        )
        if not attributed:
            findings.append(
                Finding("retrieval_missing_attribution", "retrieval", rid, (sid,), "warning")
            )
        if retrieval["returned"] != len(items[rid]):
            findings.append(Finding("retrieval_item_count_mismatch", "retrieval", rid))
        for item in items[rid]:
            if item["status_at_recall"] is None:
                findings.append(
                    Finding("retrieval_status_missing", "retrieval", rid, (item["memory_id"],))
                )
