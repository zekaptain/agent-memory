"""Deterministic check that exact values in a lesson's rule appear in its cited lines."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

EXACT_VALUES_VERSION = "exact_values_v1"

_ENTRY = re.compile(r"^\[(\d+)\] ", re.MULTILINE)
_PROJECTION_SCAFFOLD_TAIL = re.compile(
    r"(?:\n\.\.\. \d+ tool call\(s\) omitted \.\.\.|\n\.\.\. \[projection cut\])+$"
)
REDACTION_MARKER = re.compile(r"\[REDACTED:[A-Za-z0-9_]+\]")

_BACKTICK = re.compile(r"`([^`\r\n]+)`")
_DOUBLE_QUOTED = re.compile(r'"([^"\r\n]{2,80})"')
_URL = re.compile(r"https?://[^\s<>\"'`),;\]]+")
_ASSIGNMENT = re.compile(r"(?<![A-Za-z0-9_./-])([A-Za-z_][A-Za-z0-9_.-]*=[^\s,.;:)\"'\]]+)")
_ROOTED_PATH = re.compile(r"(?<![A-Za-z0-9_.-])(?:~|\.{1,2})?/[^\s<>\"'`,;)\]]+")
_REL_PATH = re.compile(r"(?<![A-Za-z0-9_./-])(?:[A-Za-z0-9_.@-]+/)+[A-Za-z0-9_.@-]*")
_DOTFILE = re.compile(r"(?<![A-Za-z0-9_./-])\.[A-Za-z][A-Za-z0-9_.-]{1,}")
_COLON_SPEC = re.compile(
    r"(?<![A-Za-z0-9_./:-])[A-Za-z0-9_.-]+:[A-Za-z0-9_./-]+(?::[A-Za-z0-9_./-]+)*"
)
_LONG_FLAG = re.compile(r"(?<![A-Za-z0-9_-])--[A-Za-z][A-Za-z0-9_-]*(?:=[^\s,.;:)\"'\]]+)?")
_SHORT_FLAG = re.compile(r"(?<!\S)-[A-Za-z](?=\s|$|[,.;:)\]])")
_SHELL_VAR = re.compile(r"\$(?:\{[A-Za-z_][A-Za-z0-9_]*\}|[A-Za-z_][A-Za-z0-9_]*)")
_DOTTED_IDENT = re.compile(
    r"(?<![A-Za-z0-9_./-])[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_-]*)+"
)
_SNAKE_IDENT = re.compile(r"(?<![A-Za-z0-9_])[A-Za-z0-9]*_[A-Za-z0-9_]+")
_CAMEL_IDENT = re.compile(
    r"(?<![A-Za-z0-9_])(?:[A-Za-z]*[a-z][A-Z][A-Za-z0-9]*|[A-Z]{2,}[a-z][A-Za-z0-9]*)"
)
_ALNUM_CODE = re.compile(r"(?<![A-Za-z0-9_-])[A-Za-z][A-Za-z0-9_-]*\d[A-Za-z0-9_-]*")
_HEX_OCT_BIN = re.compile(r"(?<![A-Za-z0-9_])0[xXoObB][0-9A-Fa-f_]+")
_ISO_DATE = re.compile(r"(?<![A-Za-z0-9_-])\d{4}-\d{2}-\d{2}(?![A-Za-z0-9_-])")
_NUMBER = re.compile(r"(?<![A-Za-z0-9_])\d+(?:\.\d+)*(?!\d|\.\d|(?:st|nd|rd|th|ST|ND|RD|TH)\b)")
_CLI_COMMAND = re.compile(
    r"\b(?:uv|mem|git|pytest|ruff|pyrefly|bun|npm|pnpm|yarn|pip|pip3|cargo|docker|podman|"
    r"kubectl|curl|wget|psql|sqlite3|duckdb|launchctl|systemctl|chmod|chown|ssh|scp|rsync|"
    r"jq|rg|gh)\b"
)

_ABBREVIATIONS = frozenset({"e.g", "i.e", "etc", "vs"})
_EXT_SUFFIX = re.compile(r"\.[A-Za-z][A-Za-z0-9]{0,7}$")


def _strip_trailing_punct(token: str) -> str:
    while token and token[-1] in ".,;:!?]}'\"":
        token = token[:-1]
    while token.endswith(")") and token.count(")") > token.count("("):
        token = token[:-1]
    return token


def _is_path(token: str) -> bool:
    if not token or token == "/":
        return False
    if token.startswith(("/", "~/", "./", "../", ".")):
        return True
    if token.endswith("/") and len(token) > 1:
        return True
    if token.count("/") >= 2:
        return True
    last = token.rsplit("/", 1)[-1]
    return bool(_EXT_SUFFIX.search(last))


def full_cited_entries(projection: str, cited: Sequence[int]) -> list[tuple[int, str]]:
    """Return full (untruncated) bodies of cited numbered entries, without `[n] ` prefixes."""
    wanted = set(cited)
    if not wanted:
        return []
    starts = [(int(m.group(1)), m.start(), m.end()) for m in _ENTRY.finditer(projection)]
    out: list[tuple[int, str]] = []
    for i, (number, _start, body_start) in enumerate(starts):
        if number not in wanted:
            continue
        end = starts[i + 1][1] if i + 1 < len(starts) else len(projection)
        body = _PROJECTION_SCAFFOLD_TAIL.sub("", projection[body_start:end].rstrip()).strip()
        out.append((number, body))
    return out


def extract_values(rule: str) -> list[str]:
    """Extract exact values (commands, paths, identifiers, numbers) from a lesson rule."""
    if not rule or not rule.strip():
        return []
    buf = list(rule)
    found: list[tuple[int, str]] = []

    def mask(start: int, end: int) -> None:
        for idx in range(start, end):
            buf[idx] = " "

    def current() -> str:
        return "".join(buf)

    for match in REDACTION_MARKER.finditer(rule):
        mask(match.start(), match.end())

    def collect(
        pattern: re.Pattern[str],
        *,
        group: int = 0,
        strip_punct: bool = True,
        predicate: Any = None,
    ) -> None:
        for match in pattern.finditer(current()):
            raw = match.group(group).strip()
            value = _strip_trailing_punct(raw) if strip_punct else raw
            if not value or not re.search(r"[A-Za-z0-9]", value):
                continue
            if predicate is not None and not predicate(value):
                continue
            found.append((match.start(), value))
            mask(match.start(), match.end())

    collect(_BACKTICK, group=1, strip_punct=False)
    collect(_DOUBLE_QUOTED, group=1, strip_punct=False)
    collect(_URL)
    collect(_ASSIGNMENT, group=1)
    collect(_ROOTED_PATH)
    collect(_REL_PATH, predicate=_is_path)
    collect(_DOTFILE, predicate=lambda v: len(v) >= 3)
    collect(_COLON_SPEC)
    collect(_LONG_FLAG)
    collect(_SHORT_FLAG, strip_punct=False)
    collect(_SHELL_VAR, strip_punct=False)
    collect(_DOTTED_IDENT, predicate=lambda v: v.lower() not in _ABBREVIATIONS)
    collect(_SNAKE_IDENT, predicate=lambda v: bool(re.search(r"[A-Za-z]", v)))
    collect(_CAMEL_IDENT)
    collect(_ALNUM_CODE)
    collect(_HEX_OCT_BIN)
    collect(_ISO_DATE, strip_punct=False)
    collect(_NUMBER)
    collect(_CLI_COMMAND, strip_punct=False)

    found.sort(key=lambda item: item[0])
    seen: set[str] = set()
    ordered: list[str] = []
    for _, value in found:
        if value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


def _match_pattern(value: str) -> re.Pattern[str]:
    parts = value.split()
    body = r"\s+".join(re.escape(part) for part in parts)
    prefix = ""
    if value[0].isdigit():
        prefix = r"(?<![A-Za-z0-9])"
    elif value[0].isalpha() or value[0] == "_":
        prefix = r"(?<![A-Za-z0-9_])"
    elif value.startswith("-"):
        prefix = r"(?<![A-Za-z0-9_-])"
    elif value.startswith("."):
        prefix = r"(?<![A-Za-z0-9_])"

    suffix = ""
    if value[-1].isdigit():
        suffix = r"(?![A-Za-z0-9])"
    elif value[-1].isalpha() or value[-1] == "_":
        suffix = r"(?![A-Za-z0-9_-])" if value.startswith("-") else r"(?![A-Za-z0-9_])"

    return re.compile(prefix + body + suffix)


def _value_in_text(value: str, cited_text: str) -> bool:
    if not cited_text:
        return False
    pattern = _match_pattern(value)
    if pattern.search(cited_text):
        return True
    if '\\"' in cited_text or "\\\\" in cited_text:
        unescaped = cited_text.replace('\\"', '"').replace("\\\\", "\\")
        if pattern.search(unescaped):
            return True
    return False


def check_exact_values(
    rule: str,
    cited: Sequence[int],
    projection: str,
) -> dict[str, Any]:
    """Check whether exact values in `rule` appear in the cited lines of `projection`.

    Deterministic and requires no model call. A redaction marker (`[REDACTED:...]`) in
    any cited line prevents missing values from failing the check.
    """
    entries = full_cited_entries(projection, cited)
    cited_text = "\n".join(body for _, body in entries)
    redacted_cited = bool(REDACTION_MARKER.search(cited_text))
    values = extract_values(rule)
    missing = (
        []
        if redacted_cited
        else [value for value in values if not _value_in_text(value, cited_text)]
    )
    return {
        "version": EXACT_VALUES_VERSION,
        "ok": not missing,
        "values": values,
        "missing": missing,
        "redacted_cited": redacted_cited,
    }


def cited_from_draft(draft: Mapping[str, Any]) -> list[int]:
    claims = draft.get("claims")
    if isinstance(claims, list):
        numbers: set[int] = set()
        for claim in claims:
            if isinstance(claim, Mapping):
                for entry in claim.get("cited") or ():
                    try:
                        number = int(entry)
                    except (TypeError, ValueError):
                        continue
                    if number > 0:
                        numbers.add(number)
        return sorted(numbers)
    cited = draft.get("cited")
    if isinstance(cited, list):
        numbers = set()
        for entry in cited:
            try:
                number = int(entry)
            except (TypeError, ValueError):
                continue
            if number > 0:
                numbers.add(number)
        return sorted(numbers)
    return []


def check_draft(draft: Mapping[str, Any], projection: str) -> dict[str, Any]:
    """Run the exact-values check on a non-declined draft mapping."""
    rule = str(draft.get("rule") or draft.get("lesson") or "")
    return check_exact_values(rule, cited_from_draft(draft), projection)
