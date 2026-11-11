"""Entity merges: one name folded into another as append-only history."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from ..capture import blobs
from ..capture.record import parse_ts
from ..capture.redact import Redactor
from ..db import MemoryClient
from ..lifecycle import LifecyclePlan, maintain_corrections, stable_id
from .model import Relation, normalize_entity
from .sql import ROW, relations_sql, sources_sql


@dataclass(frozen=True)
class EntityMerge:
    """The decision that `alias` is another name for `canonical` in some scopes. Its text is
    already redacted. `retrieval_id` names the recall the merge was proposed from, so the
    merge event says what its proposer was shown."""

    alias: str
    canonical: str
    reason: str
    actor: str
    scopes: Sequence[str]
    canonical_type: str | None = None
    retrieval_id: str | None = None

    def __post_init__(self) -> None:
        if not self.alias_norm or not self.canonical_norm:
            raise ValueError("alias and canonical are required")
        if self.alias_norm == self.canonical_norm:
            raise ValueError("alias and canonical normalize to the same entity")
        if not self.reason.strip():
            raise ValueError("reason is required")

    @property
    def alias_norm(self) -> str:
        return normalize_entity(self.alias)

    @property
    def canonical_norm(self) -> str:
        return normalize_entity(self.canonical)

    @property
    def request(self) -> dict[str, Any]:
        """Payload of the merge's `memory_write` event."""
        return {
            "merge": {
                "alias": self.alias,
                "alias_norm": self.alias_norm,
                "canonical": self.canonical,
                "canonical_norm": self.canonical_norm,
                "reason": self.reason,
                "actor": self.actor,
                "scopes": self.scopes,
                **({"retrieval_id": self.retrieval_id} if self.retrieval_id else {}),
            }
        }


def apply_merge(
    client: MemoryClient,
    merge: EntityMerge,
    write_event: Callable[[str, dict[str, Any]], dict[str, Any]],
    redactor: Redactor,
) -> dict[str, Any]:
    """Fold the alias into the canonical name in one write.

    Every active relation that mentions the alias is superseded by the same relation with
    the canonical name in its place, or retired when that relation already exists. A relation
    between the two names is retired too, since the merge makes them one entity. Nothing is
    rewritten: the alias stays in the new rows' metadata and the old rows stay readable.

    `write_event` builds the `memory_write` event row from its content and payload; the
    caller owns the session. Returns the relations superseded and retired, with the event's
    id when there were any. Nothing is written when no relation mentions the alias.
    """
    maintain_corrections(client)
    _, rows = client.query(relations_sql(merge.scopes, [merge.alias_norm]))
    old: list[tuple[dict[str, Any], Relation, Relation | None]] = []
    for row in rows:
        record = dict(zip(ROW, row, strict=True))
        relation = Relation.from_metadata(json.loads(record["metadata"]))
        if relation is None:
            continue
        ends = {relation.subject_norm, relation.object_norm}
        # Renamed, a relation between the two names would point an entity at itself.
        renamed = (
            None
            if ends == {merge.alias_norm, merge.canonical_norm}
            else relation.renamed(merge.alias_norm, merge.canonical, merge.canonical_type)
        )
        old.append((record, relation, renamed))
    done: dict[str, Any] = {"superseded": [], "retired": []}
    if not old:
        return done
    _, source_rows = client.query(sources_sql([rec["memory_id"] for rec, _, _ in old]))
    sources: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for mid, eid, rel in source_rows:
        sources[str(mid)].append((str(eid), str(rel)))
    event = write_event(f"merge entity {merge.alias} into {merge.canonical}", merge.request)
    session_id, event_id = event["session_id"], event["event_id"]
    plan = LifecyclePlan(client)

    def retire(mid: str, into: str | None) -> None:
        plan.review(
            {
                "action_id": stable_id(event_id, f"merge-retire:{mid}"),
                "memory_id": mid,
                "action": "retire",
                "actor": merge.actor,
                "session_id": session_id,
                "event_id": event_id,
                "related_memory_id": into,
                "reason": merge.reason,
            }
        )
        done["retired"].append({"memory_id": mid, "into": into})

    for record, relation, renamed in old:
        mid = record["memory_id"]
        if renamed is None:
            retire(mid, None)
            continue
        content = redactor.text(_rename_in_text(record["content"], relation, renamed))[0]
        target, duplicate = plan.remember(
            {
                "memory_id": client.new_ids()[0],
                "scope": redactor.text(record["scope"])[0],
                "type": record["type"],
                "key": renamed.key,
                "content": content,
                "content_hash": blobs.content_hash(content),
                "confidence": float(record["confidence"]),
                "importance": float(record["importance"]),
                "trust_level": record["trust_level"],
                "supersedes": mid,
                "valid_from": _ts(record["valid_from"]),
                "valid_to": _ts(record["valid_to"]),
                "expires_at": _ts(record["expires_at"]),
                "created_by_session": session_id,
                "metadata": redactor.value(renamed.metadata())[0],
            },
            event_id=event_id,
            actor=merge.actor,
            carry_approval_from=mid,
            sources=[
                {"event_id": event_id, "relation": "derived_from"},
                *({"event_id": eid, "relation": rel} for eid, rel in sources[mid]),
            ],
        )
        if duplicate:
            retire(mid, target)
        else:
            done["superseded"].append({"memory_id": mid, "by": target, "key": renamed.key})
    client.insert([("events", [event]), *plan.rows])
    done["event_id"] = event_id
    return done


def _rename_in_text(content: str, old: Relation, new: Relation) -> str:
    """Carry a merge into the memory's prose when the old name appears in it; otherwise
    the new relation's own sentence is the content."""
    text = content
    changed = False
    for before, after in ((old.subject, new.subject), (old.object, new.object)):
        if before == after:
            continue
        # A name stands alone: not inside a word, nor inside a longer name such as `al-bot`
        # or `db.py`, since entity names keep `-`, `/` and an inner `.`.
        pattern = re.compile(
            rf"(?<![\w\-/])(?<!\w\.){re.escape(before)}(?![\w\-/])(?!\.\w)", re.IGNORECASE
        )
        if pattern.search(text):
            # A function, so a backslash in the new name is not read as a group reference.
            text = pattern.sub(lambda _match, name=after: name, text)
            changed = True
    return text if changed else new.sentence


def _ts(value: str | None) -> datetime | None:
    return None if value is None else parse_ts(value)
