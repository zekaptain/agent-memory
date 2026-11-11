"""Relations as memories: one `fact` row per subject-predicate-object edge.

A relation's identity is its normalized key `subject|predicate|object`. The display forms,
entity types, and any surface forms merged into an endpoint live in `metadata.graph`.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, replace
from typing import Any

GRAPH_VERSION = 1
KEY_SEPARATOR = "|"

_NON_WORD = re.compile(r"[^\w\s.\-/]+", re.UNICODE)
_SPACES = re.compile(r"\s+")


def normalize_entity(name: str) -> str:
    """The identity of an entity: NFKC, casefolded, punctuation dropped, spaces collapsed.

    `|` is punctuation, so a normalized name can never contain the key separator."""
    text = unicodedata.normalize("NFKC", name or "").casefold()
    text = _NON_WORD.sub(" ", text).replace("_", " ")
    return _SPACES.sub(" ", text).strip(" .-/")


@dataclass(frozen=True)
class Relation:
    subject: str
    predicate: str
    object: str
    subject_type: str | None = None
    object_type: str | None = None
    # Surface forms that entity resolution merged into subject or object.
    aliases: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "subject", _clean(self.subject))
        object.__setattr__(self, "predicate", _clean(self.predicate))
        object.__setattr__(self, "object", _clean(self.object))
        object.__setattr__(self, "subject_type", _clean_type(self.subject_type))
        object.__setattr__(self, "object_type", _clean_type(self.object_type))
        if not self.subject_norm or not self.predicate_norm or not self.object_norm:
            raise ValueError("subject, predicate, and object must each contain a word")
        if self.subject_norm == self.object_norm:
            raise ValueError("subject and object must be different entities")

    @property
    def subject_norm(self) -> str:
        return normalize_entity(self.subject)

    @property
    def predicate_norm(self) -> str:
        return normalize_entity(self.predicate)

    @property
    def object_norm(self) -> str:
        return normalize_entity(self.object)

    @property
    def key(self) -> str:
        return KEY_SEPARATOR.join((self.subject_norm, self.predicate_norm, self.object_norm))

    @property
    def sentence(self) -> str:
        return f"{self.subject} {self.predicate} {self.object}"

    @property
    def triple(self) -> str:
        return f"({self.subject}) --[{self.predicate}]--> ({self.object})"

    def metadata(self) -> dict[str, Any]:
        return {
            "graph": {
                "v": GRAPH_VERSION,
                "subject": self.subject,
                "subject_norm": self.subject_norm,
                "subject_type": self.subject_type,
                "predicate": self.predicate,
                "predicate_norm": self.predicate_norm,
                "object": self.object,
                "object_norm": self.object_norm,
                "object_type": self.object_type,
                "aliases": list(self.aliases),
            }
        }

    @classmethod
    def from_metadata(cls, metadata: Any) -> Relation | None:
        """The relation stored in a memory's metadata, or None for a non-graph memory."""
        graph = metadata.get("graph") if isinstance(metadata, dict) else None
        if not isinstance(graph, dict):
            return None
        try:
            return cls(
                subject=str(graph["subject"]),
                predicate=str(graph["predicate"]),
                object=str(graph["object"]),
                subject_type=graph.get("subject_type"),
                object_type=graph.get("object_type"),
                aliases=tuple(str(a) for a in graph.get("aliases") or ()),
            )
        except (KeyError, TypeError, ValueError):
            return None

    def renamed(self, old_norm: str, canonical: str, canonical_type: str | None = None) -> Relation:
        """This relation with every endpoint that normalizes to old_norm replaced by canonical.
        The old display form is kept in aliases."""
        changes: dict[str, Any] = {}
        merged: list[str] = list(self.aliases)
        if self.subject_norm == old_norm:
            merged.append(self.subject)
            changes["subject"] = canonical
            changes["subject_type"] = canonical_type or self.subject_type
        if self.object_norm == old_norm:
            merged.append(self.object)
            changes["object"] = canonical
            changes["object_type"] = canonical_type or self.object_type
        if not changes:
            return self
        canonical_norm = normalize_entity(canonical)
        changes["aliases"] = tuple(
            dict.fromkeys(a for a in merged if normalize_entity(a) != canonical_norm)
        )
        return replace(self, **changes)


def _clean(value: Any) -> str:
    return _SPACES.sub(" ", str(value or "")).strip()


def _clean_type(value: Any) -> str | None:
    text = _clean(value).upper().replace(" ", "_")
    return text or None
