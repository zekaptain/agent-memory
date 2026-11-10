"""Local Ollama lesson embeddings; projection text is never an input."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from .. import endpoints
from ..capture.redact import Redactor
from . import AdmissionError

DEFAULT_EMBEDDER = "ollama:nomic-embed-text"


def validate_vector(value: Any) -> list[float]:
    if not isinstance(value, (list, tuple)) or not value:
        raise AdmissionError("embedding must be a nonempty numeric vector")
    try:
        if any(isinstance(x, bool) or not isinstance(x, (int, float)) for x in value):
            raise ValueError
        vector = [float(x) for x in value]
        if not all(math.isfinite(x) for x in vector):
            raise ValueError
    except (ValueError, OverflowError):
        raise AdmissionError("embedding must contain only finite numbers") from None
    return vector


def lesson_fields(draft: Mapping[str, Any]) -> dict[str, Any]:
    """A deliberately narrow allowlist shared by embeddings and persisted examples."""
    rule = draft.get("rule") or draft.get("lesson")
    if not isinstance(rule, str) or not rule.strip():
        raise AdmissionError("admission requires a non-declined draft rule")
    claims = draft.get("claims") or []
    if not isinstance(claims, list):
        raise AdmissionError("draft claims must be a list")
    texts = []
    for claim in claims:
        text = claim.get("text") if isinstance(claim, Mapping) else claim
        if not isinstance(text, str) or not text.strip():
            raise AdmissionError("draft claims must contain text")
        texts.append(text)
    kind = draft.get("kind") or "unknown"
    if not isinstance(kind, str):
        raise AdmissionError("draft kind must be text")
    clean, _ = Redactor().value({"rule": rule, "kind": kind, "claims": texts})
    return clean


class Embedder:
    def __init__(self, spec: str = DEFAULT_EMBEDDER) -> None:
        provider, separator, model = spec.partition(":")
        if provider != "ollama" or not separator or not model.strip():
            raise AdmissionError("embedder must be ollama:<model>")
        self.name = spec
        self.model = model
        self.url = endpoints.OLLAMA_HOST.rstrip("/") + "/api/embed"
        endpoints.require_egress(self.url, "Admission embedding")

    def embed(self, draft: Mapping[str, Any]) -> list[float]:
        fields = lesson_fields(draft)
        text = "\n".join([f"Rule: {fields['rule']}", f"Kind: {fields['kind']}", *fields["claims"]])
        endpoints.require_egress(self.url, "Admission embedding")
        try:
            response, _ = endpoints.post_json(
                self.url, {"model": self.model, "input": text, "truncate": False}
            )
        except endpoints.EgressError:
            # HTTP errors can echo the request body. Do not expose lesson text in errors.
            raise AdmissionError("admission embedding request failed") from None
        vectors = response.get("embeddings")
        if not isinstance(vectors, list) or len(vectors) != 1:
            raise AdmissionError("embedding response must contain exactly one vector")
        return validate_vector(vectors[0])


def make_embedder(spec: str = DEFAULT_EMBEDDER) -> Embedder:
    return Embedder(spec)
