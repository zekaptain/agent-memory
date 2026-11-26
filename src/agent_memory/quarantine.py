"""Deterministic write-time content screen for invisible characters and prompt injection.

Every memory write passes through `screen_memory` before it becomes active. A hit does not
refuse, drop, or rewrite the memory: the content is stored as written and the memory starts
quarantined with a human-readable reason so a human can approve or retire it in `mem review`.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

QUARANTINE_ACTOR = "ledger"

_ALLOWED_CONTROLS = frozenset({"\t", "\n", "\r"})
_MAX_CODEPOINTS_IN_REASON = 4
_GRAPH_TEXT_FIELDS = ("subject", "predicate", "object", "subject_type", "object_type")

# Short, audited list of prompt-injection / meta-instruction patterns directed at a recalling
# agent. Kept specific enough that ordinary coding preferences, procedures, code snippets, and
# configuration text pass without false hits.
INJECTION_PATTERNS: tuple[tuple[str, str, re.Pattern[str]], ...] = (
    (
        "instruction_override",
        "instruction override",
        re.compile(
            r"\b(?:ignore|disregard|forget|override)\s+(?:all\s+|any\s+)?"
            r"(?:the\s+)?(?:(?:previous|prior|above|earlier|preceding|your|system|safety|security)"
            r"\s+){1,3}(?:instructions?|prompts?|rules?|directions?|guidelines?|policies)\b"
            r"|\b(?:ignore|disregard|forget)\s+(?:everything|all)\s+(?:above|before|prior)\b"
            r"|\b(?:new\s+system\s+instructions?|your\s+new\s+instructions?)\s*(?::|\bare\b)",
            re.IGNORECASE,
        ),
    ),
    (
        "role_hijack",
        "role hijack",
        re.compile(
            r"\byou\s+are\s+now\s+(?:a\s+|an\s+|acting\s+as\s+|in\s+)?"
            r"(?:unrestricted|unfiltered|jailbroken|developer\s+mode|dan\b|system\b|root\b|admin\b"
            r"|different\s+assistant)"
            r"|\bfrom\s+now\s+on\s*,?\s*you\s+(?:are\s+(?:a|an|in|no\s+longer)"
            r"|must\s+(?:ignore|disregard|obey|follow\s+only))"
            r"|\benter\s+(?:developer|jailbreak|dan|unrestricted|god)\s+mode\b"
            r"|\bdo\s+not\s+follow\s+(?:your\s+|the\s+)?(?:system\s+|safety\s+)?"
            r"(?:instructions?|rules?|guidelines?|policies)\b"
            r"|\bbypass\s+(?:your\s+|the\s+)?(?:safety|security|guardrail|approval|quarantine"
            r"|policy|content)\s+(?:checks?|filters?|rules?|restrictions?|controls?)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "prompt_delimiter",
        "prompt delimiter",
        re.compile(
            r"<\|(?:im_start|im_end|system|assistant|user|endoftext)\|>"
            r"|\[/?INST\]|<<\s*/?SYS\s*>>"
            r"|<\s*(system|system-reminder|system-notification|instructions|developer)\s*>"
            r"[\s\S]{1,400}<\s*/\s*\1\s*>"
            r"|(?:^|\n)\s*<\s*(?:system|instructions|developer|assistant|tool_call|function_call)"
            r"\s*>"
            r"|(?:^|\n)\s*###\s*(?:System|Instruction|Assistant)\s*:",
            re.IGNORECASE,
        ),
    ),
    (
        "conceal_from_user",
        "concealment directive",
        re.compile(
            r"\b(?:do\s+not|don't|never)\s+(?:tell|inform|mention\s+(?:this\s+|that\s+|it\s+)?to"
            r"|reveal\s+(?:this\s+|that\s+|it\s+)?to|show|alert|notify)\s+the\s+(?:user|human)\b"
            r"|\b(?:hide|conceal|omit|keep\s+(?:this|that|it)\s+(?:secret|hidden))\s+"
            r"(?:(?:this|that|it|the\s+following|these)\s+)?from\s+the\s+(?:user|human)\b"
            r"|\b(?:silently|secretly|without\s+telling\s+the\s+user"
            r"|without\s+the\s+user\s+knowing)\s+"
            r"(?:run|execute|send|post|upload|exfiltrate|read|fetch|curl|approve|modify|edit"
            r"|delete)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "secret_exfiltration",
        "secret exfiltration directive",
        re.compile(
            r"\b(?:curl|wget|fetch|post|send|upload|exfiltrate)\b[^\n]{0,120}"
            r"(?:(?:^|[\s\"'@=/~])(?:\.env\b|\.ssh/|\.aws/credentials\b|\.netrc\b|"
            r"/etc/passwd\b|/etc/shadow\b|\.agent-memory/token\b)"
            r"|\b(?:id_rsa|id_ed25519|api[_-]?key|access[_-]?token|secret[_-]?key|"
            r"auth[_-]?token|credentials|printenv)\b)"
            r"|\b(?:reveal|dump|output|print|leak|show)\b[^\n]{0,50}\b"
            r"(?:your\s+)?(?:system\s+prompt|hidden\s+instructions?|developer\s+prompt)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "recall_trigger",
        "recall-time execution directive",
        re.compile(
            r"\b(?:whenever|when|every\s+time|once|as\s+soon\s+as)\s+(?:you\s+)?"
            r"(?:recall|read|see|load|receive)\s+this(?:\s+(?:memory|note|message|text))?\s*,?"
            r"\s*(?:you\s+)?(?:must|should|will|always|immediately)?\s*"
            r"(?:run|execute|call|invoke|send|fetch|curl|read|write|edit|delete|approve|output"
            r"|say|reply|print)\b"
            r"|\b(?:on|upon)\s+recall\s*[:,]?\s*"
            r"(?:run|execute|call|invoke|send|fetch|curl|approve|output|say|reply)\b"
            r"|\b(?:immediately|automatically)\s+(?:run|execute|call|invoke|approve)\b[^\n]{0,80}"
            r"\b(?:without\s+asking|without\s+confirmation|without\s+permission|without\s+user)\b",
            re.IGNORECASE,
        ),
    ),
)


@dataclass(frozen=True)
class Finding:
    """One reason a memory's text must wait in quarantine for human approval."""

    category: Literal["invisible_chars", "instruction_like"]
    rule: str
    detail: str
    field: str = "content"


def is_invisible_or_bidi(ch: str) -> bool:
    """Whether `ch` is an invisible, control, format, bidi, or tag character."""
    if ch in _ALLOWED_CONTROLS:
        return False
    cp = ord(ch)
    if cp == 0x034F or 0xE0000 <= cp <= 0xE007F or 0xE0100 <= cp <= 0xE01EF:
        return True
    return unicodedata.category(ch) in ("Cf", "Cc")


def invisible_codepoints(text: str) -> tuple[int, ...]:
    """Distinct invisible, direction-changing, or control codepoints in order of appearance."""
    seen: set[int] = set()
    found: list[int] = []
    for ch in text:
        if is_invisible_or_bidi(ch):
            cp = ord(ch)
            if cp not in seen:
                seen.add(cp)
                found.append(cp)
    return tuple(found)


def _codepoint_label(cp: int) -> str:
    name = unicodedata.name(chr(cp), "CONTROL/FORMAT")
    return f"U+{cp:04X} {name}"


def screen_text(text: str, *, field: str = "content") -> tuple[Finding, ...]:
    """Scan one text field for invisible/bidi characters and instruction-like patterns."""
    if not text:
        return ()
    findings: list[Finding] = []
    cps = invisible_codepoints(text)
    if cps:
        shown = ", ".join(_codepoint_label(cp) for cp in cps[:_MAX_CODEPOINTS_IN_REASON])
        if len(cps) > _MAX_CODEPOINTS_IN_REASON:
            shown += f" (+{len(cps) - _MAX_CODEPOINTS_IN_REASON} more)"
        findings.append(
            Finding(
                category="invisible_chars",
                rule="invisible_unicode",
                detail=shown,
                field=field,
            )
        )
    for rule, label, pattern in INJECTION_PATTERNS:
        if pattern.search(text):
            findings.append(
                Finding(
                    category="instruction_like",
                    rule=rule,
                    detail=label,
                    field=field,
                )
            )
    return tuple(findings)


def screen_memory(row: Mapping[str, Any]) -> tuple[Finding, ...]:
    """Scan a memory row's content (and relation triple fields, when present)."""
    fields: list[tuple[str, str]] = []
    content = row.get("content")
    if isinstance(content, str) and content:
        fields.append(("content", content))
    key = row.get("key")
    if isinstance(key, str) and key:
        fields.append(("key", key))
    metadata = row.get("metadata")
    graph = metadata.get("graph") if isinstance(metadata, Mapping) else None
    if isinstance(graph, Mapping):
        for name in _GRAPH_TEXT_FIELDS:
            value = graph.get(name)
            if isinstance(value, str) and value:
                fields.append((f"graph.{name}", value))
    seen: set[tuple[str, str, str]] = set()
    findings: list[Finding] = []
    for field_name, text in fields:
        for finding in screen_text(text, field=field_name):
            dedupe_key = (finding.category, finding.rule, finding.detail)
            if dedupe_key not in seen:
                seen.add(dedupe_key)
                findings.append(finding)
    return tuple(findings)


def format_reason(findings: Sequence[Finding]) -> str:
    """Human-readable quarantine reason recorded on the review action and shown in `mem review`."""
    invisible = [f for f in findings if f.category == "invisible_chars"]
    instructions = [f for f in findings if f.category == "instruction_like"]
    parts: list[str] = []
    if invisible:
        details = "; ".join(
            f"{f.detail}" + (f" [in {f.field}]" if f.field != "content" else "") for f in invisible
        )
        parts.append(f"invisible or direction-changing characters ({details})")
    if instructions:
        details = ", ".join(
            f"{f.detail}" + (f" [in {f.field}]" if f.field != "content" else "")
            for f in instructions
        )
        parts.append(f"instruction-like text ({details})")
    return "; ".join(parts)


def screen_metadata(findings: Sequence[Finding]) -> dict[str, Any]:
    """Structured metadata stored alongside the automatic quarantine action."""
    return {
        "content_screen": {
            "rules": list(dict.fromkeys(f.rule for f in findings)),
            "fields": list(dict.fromkeys(f.field for f in findings)),
            "findings": [
                {
                    "category": f.category,
                    "rule": f.rule,
                    "detail": f.detail,
                    "field": f.field,
                }
                for f in findings
            ],
        }
    }
