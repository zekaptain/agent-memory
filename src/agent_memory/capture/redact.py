"""Secret redaction, applied before anything is staged for the ledger or written to the spool."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass, replace
from typing import Any

from .record import Record, SessionInfo


@dataclass(frozen=True)
class Rule:
    name: str
    pattern: re.Pattern[str]
    # Groups to keep around the redaction marker, as a template; None replaces the whole match.
    template: str | None = None

    def apply(self, text: str) -> tuple[str, int]:
        marker = f"[REDACTED:{self.name}]"
        repl = marker if self.template is None else self.template.replace("{R}", marker)
        count = 0

        def replace_match(match: re.Match[str]) -> str:
            nonlocal count
            replacement = match.expand(repl)
            count += replacement != match.group()
            return replacement

        return self.pattern.sub(replace_match, text), count


def _rule(name: str, pattern: str, template: str | None = None, flags: int = 0) -> Rule:
    return Rule(name, re.compile(pattern, flags), template)


# Order matters where prefixes overlap (anthropic before openai).
RULES: tuple[Rule, ...] = (
    _rule(
        "private_key_block",
        r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----",
    ),
    _rule("aws_access_key", r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    _rule(
        "aws_secret",
        r"(?i)\b(aws[_-]?secret[_-]?access[_-]?key)\b\s*[:=]\s*['\"]?[A-Za-z0-9/+=]{40}\b",
        r"\1={R}",
    ),
    _rule("github_token", r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{36,}\b"),
    _rule("github_token", r"\bgithub_pat_[A-Za-z0-9_]{22,}\b"),
    _rule("anthropic_key", r"\bsk-ant-[A-Za-z0-9_-]{20,}\b"),
    _rule("openai_key", r"\bsk-(?:proj-)?[A-Za-z0-9_-]{20,}\b"),
    _rule("slack_token", r"\bxox[abprs]-[A-Za-z0-9-]{10,}\b"),
    _rule("google_api_key", r"\bAIza[0-9A-Za-z_-]{35}\b"),
    _rule("jwt", r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b"),
    _rule("bearer_header", r"(?i)\b(bearer)\s+[A-Za-z0-9._~+/-]{16,}=*", r"\1 {R}"),
    _rule("url_credentials", r"\b([a-z][a-z0-9+.-]*://)[^\s/:@]+:[^\s/@]+@", r"\1{R}@"),
    _rule(
        "generic_assignment",
        r"(?i)\b([A-Za-z0-9_-]*(?:api[_-]?key|secret|token|password|passwd|pwd))\b"
        r"\s*[:=]\s*['\"]?([^\s'\"`,;]{8,})",
        r"\1={R}",
    ),
)


class Redactor:
    """Applies the rule set plus any exact secrets (the server token) to text and payloads."""

    def __init__(self, extra_secrets: Iterable[str] = (), rules: Iterable[Rule] = RULES) -> None:
        self.rules = list(rules)
        for secret in extra_secrets:
            secret = secret.strip()
            if secret:
                self.rules.insert(0, _rule("mem_token", re.escape(secret)))

    def text(self, value: str) -> tuple[str, int]:
        total = 0
        for rule in self.rules:
            value, n = rule.apply(value)
            total += n
        return value, total

    def value(self, obj: Any) -> tuple[Any, int]:
        """Redact every string inside a JSON-like structure, keeping its shape."""
        count = 0

        def walk(node: Any) -> Any:
            nonlocal count
            if isinstance(node, str):
                out, n = self.text(node)
                count += n
                return out
            if isinstance(node, dict):
                out = {}
                for key, value in node.items():
                    clean_key = walk(key)
                    clean_value = walk(value)
                    # Use the same assignment rules for structured credentials,
                    # retaining their order, names, and minimum value lengths.
                    if isinstance(key, str) and isinstance(value, str) and clean_value == value:
                        for rule in self.rules:
                            if rule.name in ("aws_secret", "generic_assignment") and (
                                rule.pattern.match(f"{key}={value}")
                            ):
                                marker = f"[REDACTED:{rule.name}]"
                                count += value != marker
                                clean_value = marker
                                break
                    out[clean_key] = clean_value
                return out
            if isinstance(node, list):
                return [walk(v) for v in node]
            return node

        return walk(obj), count

    def record(self, rec: Record) -> Record:
        """Sanitize every content copy without changing capture identities or references."""
        fields, count = self.value(
            {
                "content": rec.content,
                "payload": rec.payload,
                "session": asdict(rec.session) if rec.session else None,
                "memory": rec.memory,
                "action": rec.action,
                "raw": rec.raw,
                "observation": rec.observation,
                "attribution": rec.attribution,
            }
        )
        if fields["session"] is not None:
            fields["session"] = SessionInfo(**fields["session"])
        return replace(rec, **fields, redactions=rec.redactions + count)


def redactor_for(token: str | None) -> Redactor:
    return Redactor(extra_secrets=[token] if token else [])
