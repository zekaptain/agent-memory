"""Entity resolution: cluster the surface forms in a scope with a local model, then record
each merge as supersedes/retire history through MemoryTools.merge_entity.

The model only proposes. Over-merging (a mission folded into its programme) and silently
dropped names are the two documented failure modes of model-driven resolution, so proposals
are shown before they are applied, and any name the model leaves out keeps its own cluster.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from .. import endpoints
from ..db import AgentMemoryError
from .model import Relation, normalize_entity
from .sql import ROW

DEFAULT_MODEL = "qwen3:8b"
BLOCK_SIZE = 60
MAX_SENTENCES = 3
MAX_TOKENS = 4096
UNTYPED = "UNTYPED"

_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


class ResolveError(AgentMemoryError):
    """Entity resolution cannot proceed for an expected reason."""


@dataclass(frozen=True)
class Entity:
    norm: str
    name: str
    entity_type: str | None
    mentions: int
    sentences: tuple[str, ...]
    forms: tuple[str, ...]


@dataclass(frozen=True)
class Cluster:
    canonical: str
    aliases: tuple[str, ...]


@dataclass(frozen=True)
class Merge:
    alias: str
    canonical: str
    entity_type: str | None

    @property
    def alias_norm(self) -> str:
        return normalize_entity(self.alias)

    @property
    def canonical_norm(self) -> str:
        return normalize_entity(self.canonical)


def relations_from_rows(rows: Sequence[tuple]) -> list[Relation]:
    """Relations from graph SQL rows (see sql.ROW); rows without graph metadata are skipped."""
    out = []
    for row in rows:
        record = dict(zip(ROW, row, strict=True))
        relation = Relation.from_metadata(json.loads(record["metadata"]))
        if relation is not None:
            out.append(relation)
    return out


def collect_entities(relations: Sequence[Relation]) -> list[Entity]:
    """Every endpoint, keyed by normalized name, with its commonest display form and type
    and a few sentences that mention it for the model to disambiguate with."""
    forms: dict[str, Counter[str]] = defaultdict(Counter)
    types: dict[str, Counter[str]] = defaultdict(Counter)
    sentences: dict[str, list[str]] = defaultdict(list)
    for relation in relations:
        for name, etype in (
            (relation.subject, relation.subject_type),
            (relation.object, relation.object_type),
        ):
            norm = normalize_entity(name)
            forms[norm][name] += 1
            if etype:
                types[norm][etype] += 1
            if len(sentences[norm]) < MAX_SENTENCES:
                sentences[norm].append(relation.sentence)
    return sorted(
        (
            Entity(
                norm=norm,
                name=forms[norm].most_common(1)[0][0],
                entity_type=types[norm].most_common(1)[0][0] if types[norm] else None,
                mentions=sum(forms[norm].values()),
                sentences=tuple(sentences[norm]),
                forms=tuple(forms[norm]),
            )
            for norm in forms
        ),
        key=lambda e: e.norm,
    )


def blocks(
    entities: Sequence[Entity], block_size: int = BLOCK_SIZE
) -> list[tuple[str, list[Entity]]]:
    """Group by entity type, then chunk, so the model arbitrates within small blocks."""
    by_type: dict[str, list[Entity]] = defaultdict(list)
    for entity in entities:
        by_type[entity.entity_type or UNTYPED].append(entity)
    out: list[tuple[str, list[Entity]]] = []
    for etype in sorted(by_type):
        members = by_type[etype]
        for start in range(0, len(members), max(1, block_size)):
            out.append((etype, members[start : start + block_size]))
    return out


def cluster_prompt(entity_type: str, entities: Sequence[Entity]) -> str:
    listed = "\n".join(
        f"- {e.name}: " + ("; ".join(e.sentences) or "(no context)") for e in entities
    )
    kind = "" if entity_type == UNTYPED else f"{entity_type} "
    return (
        f"Below are {kind}entities from an AI coding agent's memory of a software project. "
        "Some are different surface forms of the same real-world entity: an abbreviation and "
        "its expansion, a first name and a full name, a file path and the module it holds.\n\n"
        f"<entities>\n{listed}\n</entities>\n\n"
        "Cluster them. Each input name must appear in exactly one cluster's aliases. Entities "
        "that are genuinely distinct get their own single-element cluster. Use the context "
        "after each name to avoid merging entities that merely share a name; a specific thing "
        "is not the same entity as the broader thing it belongs to. The canonical name must "
        "be one of the listed names, in its most complete and unambiguous form.\n\n"
        'Return exactly this JSON: {"clusters": [{"canonical": "<name>", '
        '"aliases": ["<name>", ...]}, ...]}'
    )


def parse_clusters(text: str, names: Sequence[str]) -> list[Cluster]:
    """The model's clusters, repaired: names it dropped become singletons, names it invented
    are ignored, a name it placed twice stays in the first cluster, and a canonical that is
    not in its own cluster is replaced by the longest alias."""
    data = _json_object(text)
    raw = data.get("clusters")
    if not isinstance(raw, list):
        raise ResolveError("model JSON has no clusters list")
    by_norm: dict[str, str] = {}
    for name in names:
        by_norm.setdefault(normalize_entity(name), name)
    seen: set[str] = set()
    out: list[Cluster] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        aliases: list[str] = []
        for alias in entry.get("aliases") or []:
            norm = normalize_entity(str(alias))
            if norm in by_norm and norm not in seen:
                seen.add(norm)
                aliases.append(by_norm[norm])
        if not aliases:
            continue
        canonical = by_norm.get(normalize_entity(str(entry.get("canonical") or "")))
        if canonical not in aliases:
            canonical = max(aliases, key=len)
        out.append(Cluster(canonical, tuple(aliases)))
    for norm, name in by_norm.items():
        if norm not in seen:
            out.append(Cluster(name, (name,)))
    return out


def plan_merges(clusters: Sequence[Cluster], entity_type: str | None) -> list[Merge]:
    return [
        Merge(alias=alias, canonical=cluster.canonical, entity_type=entity_type)
        for cluster in clusters
        for alias in cluster.aliases
        if normalize_entity(alias) != normalize_entity(cluster.canonical)
    ]


class Resolver(Protocol):
    name: str

    def clusters(self, entity_type: str, entities: Sequence[Entity]) -> list[Cluster]: ...


class OllamaResolver:
    """Clusters with Ollama; non-loopback hosts require the shared egress opt-in."""

    def __init__(self, model: str = DEFAULT_MODEL, *, host: str = endpoints.OLLAMA_HOST) -> None:
        self.model = model
        self.host = host
        self.name = f"ollama:{model}"

    def clusters(self, entity_type: str, entities: Sequence[Entity]) -> list[Cluster]:
        body: dict[str, Any] = {
            "model": self.model,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0, "num_predict": MAX_TOKENS},
            "messages": [{"role": "user", "content": cluster_prompt(entity_type, entities)}],
        }
        try:
            response, _ = endpoints.post_json(f"{self.host}/api/chat", body)
        except AgentMemoryError as exc:
            raise ResolveError(str(exc)) from None
        content = str((response.get("message") or {}).get("content") or "")
        if response.get("done_reason") == "length":
            raise ResolveError("model ran out of tokens before finishing the clusters")
        return parse_clusters(content, [e.name for e in entities])


def propose(
    relations: Sequence[Relation],
    resolver: Resolver,
    *,
    entity_type: str | None = None,
    block_size: int = BLOCK_SIZE,
) -> list[Merge]:
    """Merges the resolver proposes over the entities in these relations. Blocks with one
    entity need no model call."""
    entities = collect_entities(relations)
    if entity_type:
        wanted = entity_type.upper()
        entities = [e for e in entities if (e.entity_type or UNTYPED) == wanted]
    merges: list[Merge] = []
    for etype, members in blocks(entities, block_size):
        if len(members) < 2:
            continue
        clusters = resolver.clusters(etype, members)
        merges.extend(plan_merges(clusters, None if etype == UNTYPED else etype))
    return merges


def _json_object(text: str) -> dict[str, Any]:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = _JSON_OBJECT.search(text)
        if not match:
            raise ResolveError(f"model returned no JSON: {text[:200]!r}") from None
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise ResolveError(f"model returned unparsable JSON ({exc.msg})") from None
    if not isinstance(data, dict):
        raise ResolveError("model JSON is not an object")
    return data
