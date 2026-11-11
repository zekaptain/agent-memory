"""Observed capture boundaries, independent of a harness's capabilities."""

from __future__ import annotations

import hashlib
import json
import uuid
from collections.abc import Iterable, Sequence
from dataclasses import asdict
from typing import Any

from ..db import MemoryClient, sql_literal
from . import ids
from .record import Record, parse_ts

POLICY_VERSION = "capture_observations_v1"


def fingerprint(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, default=str, ensure_ascii=False).encode()
    ).hexdigest()


def observation_rows(
    records: Sequence[Record], *, batch: Sequence[Record] | None = None
) -> list[dict[str, Any]]:
    """Build rows from already-redacted records, with stable import identities.

    `batch` is the delivery the records were taken from. A session that the batch observed
    as a whole stays covered by that observation, however the batch is split for writing.
    """
    found: dict[str, dict[str, Any]] = {}
    observed_sessions = {
        (rec.agent_id, rec.harness_session_id)
        for rec in (records if batch is None else batch)
        if rec.observation is not None
    }
    for rec in records:
        observation: dict[str, Any] | None = rec.observation
        if observation is None:
            hook = (rec.payload or {}).get("hook")
            if (
                (rec.agent_id, rec.harness_session_id) in observed_sessions
                and not hook
                and not rec.memory
                and not rec.action
            ):
                continue
            # A queued tool write names its own observation. One spooled by an earlier
            # version does not, and is recognised by the memory or action it carries.
            mode = "hook" if hook else "mcp" if rec.memory or rec.action else "record"
            observation = {
                "source_mode": mode,
                "adapter_version": f"{rec.agent_id}_records_v1",
                "source_fingerprint": fingerprint(
                    [rec.agent_id, rec.harness_session_id, rec.kind, rec.source_key]
                ),
                "covered_from": rec.ts,
                "covered_through": rec.ts,
                "cursor_start": rec.source_key,
                "cursor_end": rec.source_key,
                "exclusions": ["isolated_hook"] if hook else ["no_complete_capture_evidence"],
                "session_metadata": asdict(rec.session) if rec.session else {},
            }
        metadata = dict(observation.get("session_metadata") or {})
        if metadata.get("started_at") is not None:
            metadata["started_at"] = parse_ts(metadata["started_at"]).isoformat()
        row = {
            "session_id": ids.session_id(rec.agent_id, rec.harness_session_id),
            "source_mode": observation["source_mode"],
            "adapter_version": observation["adapter_version"],
            "policy_version": POLICY_VERSION,
            "source_fingerprint": observation["source_fingerprint"],
            "covered_from": observation.get("covered_from"),
            "covered_through": observation.get("covered_through"),
            "cursor_start": observation.get("cursor_start"),
            "cursor_end": observation.get("cursor_end"),
            "range_start": observation.get("range_start"),
            "range_end": observation.get("range_end"),
            "parse_errors": observation.get("parse_errors", []),
            "exclusions": observation.get("exclusions", []),
            "complete_supported": observation.get("complete_supported", False),
            "session_metadata": metadata,
            "attribution_evidence": observation.get("attribution_evidence", []),
        }
        # Some source entries lack timestamps. Their inferred wall time must
        # not turn repeated imports into new observations of identical bytes.
        identity = {
            key: value
            for key, value in row.items()
            if key not in ("covered_from", "covered_through")
        }
        oid = str(
            uuid.uuid5(uuid.NAMESPACE_URL, "agent-memory/observation/" + fingerprint(identity))
        )
        found[oid] = {"observation_id": oid, **row}
    return list(found.values())


def last_import_end(
    client: MemoryClient, agent_id: str, harness_session_id: str
) -> tuple[str, int] | None:
    """The entry and the line at which the latest import of a session's transcript ended,
    or None when the ledger holds no import of it."""
    session = sql_literal(ids.session_id(agent_id, harness_session_id))
    _, rows = client.query(
        "SELECT cursor_end, range_end FROM capture_observations "
        f"WHERE session_id = {session} AND source_mode = 'transcript_import' "
        "AND cursor_end IS NOT NULL AND range_end IS NOT NULL "
        "ORDER BY recorded_at DESC, range_end DESC LIMIT 1"
    )
    return (str(rows[0][0]), int(rows[0][1])) if rows else None


def last_import_waiting(lines: Iterable[str]) -> tuple[str, int] | None:
    """As `last_import_end`, read from the lines of a session's records that wait in the
    spool, oldest first."""
    end = None
    for line in lines:
        # Few records carry an observation, and a session's spool can hold tens of megabytes.
        if '"transcript_import"' not in line:
            continue
        try:
            observation = Record.from_json(line).observation or {}
            if observation.get("source_mode") == "transcript_import":
                end = (str(observation["cursor_end"]), int(observation["range_end"]))
        except (KeyError, TypeError, ValueError):
            continue
    return end


def pending_observations(
    client: MemoryClient, records: Sequence[Record], *, batch: Sequence[Record] | None = None
) -> list[dict[str, Any]]:
    """Pre-read deterministic identities in bounded batches before staged insertion."""
    rows = observation_rows(records, batch=batch)
    existing: set[str] = set()
    for start in range(0, len(rows), 500):
        wanted = ", ".join(sql_literal(row["observation_id"]) for row in rows[start : start + 500])
        _, have = client.query(
            "SELECT observation_id::VARCHAR FROM capture_observations "
            f"WHERE observation_id IN ({wanted})"
        )
        existing.update(str(row[0]) for row in have)
    return [row for row in rows if row["observation_id"] not in existing]
