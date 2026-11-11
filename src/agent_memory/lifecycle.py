"""Shared append-only memory writes, correction proposals and bounded due work.

Plans only prepare rows. Database triggers validate and accept admissible proposals
inside the same transaction as the write/approval; a competing acceptance cannot
leave half a correction committed. Reads here are hints, never the concurrency guard.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any
from uuid import NAMESPACE_URL, uuid5

import duckdb

from .capture.record import parse_ts
from .db import AgentMemoryError, MemoryClient, sql_literal

REVIEW_ACTIONS = frozenset({"approve", "quarantine", "retire", "confirm", "flag_contradiction"})
TERMINAL = frozenset({"retired", "superseded", "expired", "deduplicated"})
# The agent a human at the `mem` CLI writes as.
HUMAN_AGENT = "human"


class LifecycleError(AgentMemoryError, ValueError):
    """A terminal validation/conflict error, never a reason to queue a network retry."""


def check_asker(agent_id: str, action: str) -> None:
    """Refuse a review action that the agent asking for it may not ask for.

    Approving a memory is a human's decision: an agent that could approve what it wrote
    from tool output would hold the key to its own quarantine. The approvals that a write
    brings with it are not asked for and do not come through here: an entity merge keeps
    the approval of the relation it rewrites, and admission in auto mode approves a lesson
    on the approver's score.
    """
    if action == "approve" and agent_id != HUMAN_AGENT:
        raise LifecycleError("only a human approves a memory; ask the user to run `mem review`")


def stable_id(seed: str, purpose: str) -> str:
    return str(uuid5(NAMESPACE_URL, f"agent-memory:lifecycle:{seed}:{purpose}"))


def validate_interval(spec: Mapping[str, Any]) -> None:
    try:
        start, end, expiry = (
            parse_ts(spec[k]) if spec.get(k) is not None else None
            for k in ("valid_from", "valid_to", "expires_at")
        )
    except (ValueError, TypeError) as exc:
        raise LifecycleError("invalid memory interval: use ISO-8601 timestamps") from exc
    if start is not None and any(stop is not None and start >= stop for stop in (end, expiry)):
        raise LifecycleError(
            "invalid memory interval: valid_from must be earlier than valid_to and expires_at"
        )


def is_validation_error(exc: Exception) -> bool:
    return isinstance(
        exc,
        (
            ValueError,
            LifecycleError,
            duckdb.ConstraintException,
            duckdb.ConversionException,
            duckdb.BinderException,
            duckdb.InvalidInputException,
            duckdb.TransactionException,
        ),
    )


class LifecyclePlan:
    """Build an ordered, idempotent write set shared by MCP, replay and graph merges."""

    def __init__(self, client: MemoryClient) -> None:
        self.client = client
        self.rows: list[tuple[str, list[dict[str, Any]]]] = []
        self.memories: dict[str, dict[str, Any]] = {}
        self.aliases: dict[str, str] = {}
        self.actions: set[str] = set()
        self.sources: set[tuple[str, str, str]] = set()
        self.proposals: set[tuple[str, str]] = set()

    def resolve(self, memory_id: str) -> str:
        """Resolve only explicit dedup links. Supersession is NEVER an identity alias."""
        seen: set[str] = set()
        while memory_id not in seen:
            seen.add(memory_id)
            if memory_id in self.aliases:
                memory_id = self.aliases[memory_id]
                continue
            _, rows = self.client.query(
                "SELECT canonical_id::VARCHAR FROM memory_deduplications "
                f"WHERE memory_id = {sql_literal(memory_id)}"
            )
            if not rows:
                return memory_id
            self.aliases[memory_id] = str(rows[0][0])
            memory_id = str(rows[0][0])
        raise LifecycleError("invalid deduplication cycle; inspect the ledger before retrying")

    def memory(self, memory_id: str) -> dict[str, Any]:
        if memory_id not in self.memories:
            columns, rows = self.client.query(
                "SELECT m.* EXCLUDE(metadata), m.metadata::JSON::VARCHAR AS metadata, st.status "
                "FROM memories m JOIN memory_status st USING (memory_id) "
                f"WHERE memory_id = {sql_literal(memory_id)}"
            )
            if not rows:
                raise LifecycleError(f"unknown memory {memory_id}; replay its write first")
            row = dict(zip(columns, rows[0], strict=True))
            row["memory_id"] = memory_id
            row["metadata"] = json.loads(row["metadata"]) if row["metadata"] else None
            self.memories[memory_id] = row
        return self.memories[memory_id]

    def _action(self, row: dict[str, Any]) -> None:
        aid = row["action_id"]
        if aid in self.actions:
            return
        _, existing = self.client.query(
            f"SELECT 1 FROM memory_actions WHERE action_id = {sql_literal(aid)}"
        )
        self.actions.add(aid)
        if not existing:
            self.rows.append(("memory_actions", [row]))

    def add_sources(self, memory_id: str, sources: Sequence[Mapping[str, Any]]) -> None:
        _, existing = self.client.query(
            "SELECT event_id::VARCHAR, relation FROM memory_sources "
            f"WHERE memory_id = {sql_literal(memory_id)}"
        )
        self.sources.update((memory_id, str(eid), str(rel)) for eid, rel in existing)
        rows = []
        for source in sources:
            key = (memory_id, str(source["event_id"]), str(source["relation"]))
            if key not in self.sources:
                self.sources.add(key)
                rows.append({"memory_id": memory_id, "event_id": key[1], "relation": key[2]})
        if rows:
            self.rows.append(("memory_sources", rows))

    def propose(
        self,
        predecessor: str,
        successor: str,
        *,
        event_id: str,
        actor: str,
        session_id: str | None,
        reason: str,
        intent: Mapping[str, Any] | None = None,
    ) -> None:
        predecessor, successor = self.resolve(predecessor), self.resolve(successor)
        if predecessor == successor:
            raise LifecycleError("self correction is invalid; supply a different replacement")
        old, new = self.memory(predecessor), self.memory(successor)
        if old["scope"] != new["scope"]:
            raise LifecycleError("cross-scope correction is invalid; use the predecessor's scope")
        _, accepted = self.client.query(
            "SELECT successor_id::VARCHAR FROM memory_replacements "
            f"WHERE predecessor_id = {sql_literal(predecessor)}"
        )
        target = str(accepted[0][0]) if accepted else None
        if target and target != successor:
            raise LifecycleError(
                f"stale correction conflict: {predecessor} was replaced by {target}; "
                "target that accepted successor"
            )
        if new.get("status") in TERMINAL:
            raise LifecycleError(
                f"replacement {successor} is terminal; create an explicit new correction"
            )
        pair = (predecessor, successor)
        _, proposed = self.client.query(
            "SELECT 1 FROM memory_actions WHERE action = 'propose_correction' "
            f"AND related_memory_id = {sql_literal(predecessor)} "
            f"AND memory_id = {sql_literal(successor)} LIMIT 1"
        )
        if pair in self.proposals or proposed:
            return
        self.proposals.add(pair)
        self._action(
            {
                "action_id": stable_id(event_id, f"propose:{predecessor}:{successor}"),
                "memory_id": successor,
                "action": "propose_correction",
                "actor": actor,
                "session_id": session_id,
                "related_memory_id": predecessor,
                "event_id": event_id,
                "reason": reason,
                "metadata": {
                    "correction": {
                        k: v.isoformat() if hasattr(v, "isoformat") else v
                        for k, v in (intent or {}).items()
                        if k in ("valid_from", "valid_to", "expires_at", "trust_level")
                    }
                },
            }
        )

    def remember(
        self,
        row: dict[str, Any],
        *,
        event_id: str,
        actor: str,
        sources: Sequence[Mapping[str, Any]],
        preserve_identity: bool = False,
        deduplicate: bool = True,
        carry_approval_from: str | None = None,
    ) -> tuple[str, bool]:
        validate_interval(row)
        row = dict(row)
        requested_id = row["memory_id"]
        predecessor = row.get("supersedes")
        if predecessor:
            row["supersedes"] = predecessor = self.resolve(str(predecessor))
            old = self.memory(predecessor)
            if old["scope"] != row["scope"]:
                raise LifecycleError("cross-scope correction is invalid")
        # A replay keeps its originally returned ledger identity, even when it
        # now points at an existing canonical memory.
        _, existing_id = self.client.query(
            f"SELECT 1 FROM memories WHERE memory_id = {sql_literal(requested_id)}"
        )
        canonical = self.resolve(requested_id) if existing_id else None
        graph = (row.get("metadata") or {}).get("graph")
        if canonical is None and deduplicate:
            column = "key" if graph else "content_hash"
            for mid, known in self.memories.items():
                if known["scope"] == row["scope"] and known.get(column) == row[column]:
                    canonical = self.resolve(mid)
            if canonical is None:
                _, duplicates = self.client.query(
                    "SELECT m.memory_id::VARCHAR FROM memories m "
                    f"WHERE m.scope = {sql_literal(row['scope'])} "
                    f"AND m.{column} = {sql_literal(row[column])} "
                    "AND NOT EXISTS (SELECT 1 FROM memory_deduplications d "
                    "WHERE d.memory_id = m.memory_id) "
                    "ORDER BY m.created_at DESC, m.rowid DESC LIMIT 1"
                )
                if duplicates:
                    canonical = str(duplicates[0][0])
        canonical = canonical or requested_id
        # An explicit correction may restore historical *content*, but must
        # do it through a fresh identity, never by approving a terminal row.
        if (
            canonical != requested_id
            and predecessor
            and not existing_id
            and self.memory(canonical).get("status") in TERMINAL
        ):
            canonical = requested_id
        duplicate = canonical != requested_id
        if predecessor == canonical:
            raise LifecycleError("self correction after deduplication; replacement is unchanged")
        if not existing_id and (not duplicate or preserve_identity):
            # One statement per memory allows queued corrections to reference
            # earlier queued memories in this same atomic delivery.
            # A queued duplicate is an identity alias, not a second correction.
            # Its original request (including supersedes) remains in its event
            # and the canonical proposal; do not manufacture a competing edge.
            persisted = {**row, "supersedes": None} if duplicate else row
            self.rows.append(("memories", [persisted]))
            self.memories[requested_id] = persisted
        if duplicate:
            self.memory(canonical)
            if preserve_identity and not existing_id:
                self.rows.append(
                    (
                        "memory_deduplications",
                        [
                            {
                                "memory_id": requested_id,
                                "canonical_id": canonical,
                                "event_id": event_id,
                            }
                        ],
                    )
                )
                self.aliases[requested_id] = canonical
            self._action(
                {
                    "action_id": stable_id(event_id, f"confirm:{canonical}"),
                    "memory_id": canonical,
                    "action": "confirm",
                    "actor": actor,
                    "session_id": row.get("created_by_session"),
                    "event_id": event_id,
                    "reason": "relate() with the same subject, predicate, and object"
                    if graph
                    else "remember() with identical content",
                }
            )
        self.add_sources(canonical, sources)
        if carry_approval_from:
            source = self.memory(self.resolve(carry_approval_from))
            if source.get("status") != "active":
                raise LifecycleError("merge approval requires an active source relation")
            if not duplicate and row["trust_level"] == "tool":
                self.review(
                    {
                        "action_id": stable_id(event_id, f"merge-approve:{canonical}"),
                        "memory_id": canonical,
                        "action": "approve",
                        "actor": actor,
                        "session_id": row.get("created_by_session"),
                        "event_id": event_id,
                        "related_memory_id": carry_approval_from,
                        "reason": f"approval carried from active relation {carry_approval_from}",
                    }
                )
        if predecessor:
            self.propose(
                predecessor,
                canonical,
                event_id=event_id,
                actor=actor,
                session_id=row.get("created_by_session"),
                reason="requested correction",
                intent=row,
            )
        if graph and graph.get("aliases"):
            self._action(
                {
                    "action_id": stable_id(event_id, f"aliases:{canonical}"),
                    "memory_id": canonical,
                    "action": "merge_lineage",
                    "actor": actor,
                    "session_id": row.get("created_by_session"),
                    "event_id": event_id,
                    "related_memory_id": predecessor,
                    "metadata": {"aliases": graph["aliases"]},
                    "reason": "source relation aliases retained at merge",
                }
            )
        return canonical, duplicate

    def review(self, row: dict[str, Any]) -> str:
        row = dict(row)
        if row["action"] not in REVIEW_ACTIONS:
            raise LifecycleError("action must be one of " + ", ".join(sorted(REVIEW_ACTIONS)))
        mid = row["memory_id"] = self.resolve(row["memory_id"])
        self.memory(mid)
        if row.get("related_memory_id"):
            row["related_memory_id"] = self.resolve(row["related_memory_id"])
            self.memory(row["related_memory_id"])
        self._action(row)
        return mid


def maintain_corrections(
    connection: MemoryClient | duckdb.DuckDBPyConnection, *, limit: int = 128
) -> int:
    """Apply at most one indexed batch of already-authorized due corrections.

    Backlogged successors stay pending, so even an unfinished batch cannot
    expose both versions. The acceptance PK arbitrates concurrent maintainers.
    """
    limit = max(1, min(int(limit), 1024))

    def query(sql: str) -> list[tuple]:
        return (
            connection.query(sql)[1]
            if isinstance(connection, MemoryClient)
            else connection.execute(sql).fetchall()
        )

    for attempt in range(2):
        # No memories/status/reviews query on the common no-due-work path.
        rows = query(
            "SELECT work_id::VARCHAR, predecessor_id::VARCHAR, successor_id::VARCHAR, "
            "proposal_id::VARCHAR FROM pending_memory_corrections "
            f"WHERE due_at <= now() ORDER BY due_at, work_id LIMIT {limit}"
        )
        if not rows:
            return 0
        wanted = ", ".join(sql_literal(row[0]) for row in rows)
        admissible = {
            wid: (event, approval, method)
            for wid, event, approval, method in query(
                "SELECT work_id::VARCHAR, evidence_event_id::VARCHAR, "
                "authorization_action_id::VARCHAR, method "
                f"FROM admissible_memory_corrections WHERE work_id IN ({wanted})"
            )
        }
        acceptances, closures = [], []
        predecessors: set[str] = set()
        successors: set[str] = set()
        for wid, old, new, proposal in rows:
            if wid not in admissible:
                closures.append(
                    {
                        "work_id": wid,
                        "reason": "authorization withdrawn or successor ineligible",
                    }
                )
                continue
            # A batch cannot supersede another row's successor. Defer chains
            # and competing proposals to the next bounded pass, in stable order.
            if old in predecessors or old in successors or new in predecessors:
                continue
            predecessors.add(old)
            successors.add(new)
            event, approval, method = admissible[wid]
            acceptances.append(
                {
                    "predecessor_id": old,
                    "successor_id": new,
                    "proposal_id": proposal,
                    "evidence_event_id": event,
                    "evidence_action_id": approval,
                    "method": method,
                }
            )
        pairs = [
            ("memory_replacements", acceptances),
            ("memory_correction_closures", closures),
        ]
        try:
            if isinstance(connection, MemoryClient):
                connection.insert(pairs)
            else:
                connection.execute("BEGIN")
                try:
                    for table, batch in pairs:
                        if not batch:
                            continue
                        columns = ", ".join(batch[0])
                        placeholders = "(" + ", ".join("?" for _ in batch[0]) + ")"
                        connection.execute(
                            f"INSERT INTO {table} ({columns}) VALUES "
                            + ", ".join(placeholders for _ in batch),
                            [value for row in batch for value in row.values()],
                        )
                    connection.execute("COMMIT")
                except Exception:
                    connection.execute("ROLLBACK")
                    raise
            return len(acceptances) + len(closures)
        except duckdb.ConstraintException as exc:
            if "Duplicate key" not in str(exc) or attempt:
                raise
        except duckdb.InvalidInputException as exc:
            if "invalid acceptance" not in str(exc) or attempt:
                raise
        # Re-read once after a concurrent acceptance/withdrawal. The failed
        # transaction left every member of the batch unchanged.
    return 0
