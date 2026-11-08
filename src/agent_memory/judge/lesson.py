"""Shared structured lessons, sampler schema, and numbered projection evidence."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Annotated, Any, Literal

from pydantic import AfterValidator, BaseModel, BeforeValidator, ConfigDict, Field, ValidationError

from .store import JudgeError

LESSON_KINDS = {
    "project_fact": "where things are, conventions, commands, how the project is run",
    "tool_quirk": "how a tool, CLI, or API behaves, and what trips it",
    "process": "how to work: verify before claiming, read before editing, run the tests",
    "environment": "a fact about this machine or setup: ports, paths, installed versions",
    "user_preference": "how this person wants things done or reported",
}
MAX_CLAIMS = 4
MAX_TEXT_CHARS = 300
EXCERPT_CHARS = 320
_ENTRY = re.compile(r"^\[(\d+)\] ", re.MULTILINE)


def _sentence(value: str) -> str:
    text = " ".join(value.split())[:MAX_TEXT_CHARS]
    if not text:
        raise ValueError("text must not be empty")
    return text


def _lines(value: Any) -> tuple[int, ...]:
    values = value if isinstance(value, (list, tuple)) else [value]
    cited: list[int] = []
    for entry in values:
        try:
            number = int(str(entry).strip().strip("[]"))
        except ValueError:
            continue
        if number > 0 and number not in cited:
            cited.append(number)
    return tuple(cited)


Sentence = Annotated[str, AfterValidator(_sentence)]
Lines = Annotated[tuple[int, ...], BeforeValidator(_lines)]


class Claim(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    text: Sentence
    cited: Lines


class Lesson(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    rule: Sentence
    kind: Literal["project_fact", "tool_quirk", "process", "environment", "user_preference"]
    claims: Annotated[list[Claim], Field(min_length=1, max_length=MAX_CLAIMS)]


class DraftReply(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    lesson: Lesson | None


def sampler_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Use validation's own schema, without annotations unsupported by some grammars."""

    def strip(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                key: strip(v) for key, v in value.items() if key not in ("title", "description")
            }
        if isinstance(value, list):
            return [strip(v) for v in value]
        return value

    return strip(model.model_json_schema())


LESSON_SCHEMA = sampler_schema(DraftReply)


_FIELDS = frozenset(name for model in (DraftReply, Lesson, Claim) for name in model.model_fields)


def _details(exc: ValidationError) -> str:
    """Locations and messages only. A key the reply invented is model output too, so it is
    reported as unexpected rather than repeated."""
    parts = []
    for error in exc.errors(include_input=False):
        loc = ".".join(
            str(part) if isinstance(part, int) or part in _FIELDS else "[unexpected key]"
            for part in error["loc"]
        )
        parts.append(f"{loc or 'reply'}: {error['msg']}")
    return "; ".join(parts)


def parse_lesson(data: Any) -> Lesson | None:
    try:
        return DraftReply.model_validate(data).lesson
    except ValidationError as exc:
        raise JudgeError(f"drafter reply rejected: {_details(exc)}") from None


def render_lesson(lesson: Lesson) -> str:
    lines = [f"Rule: {lesson.rule}", f"Kind: {lesson.kind}", "Claims:"]
    for index, claim in enumerate(lesson.claims, 1):
        cited = ", ".join(map(str, claim.cited)) if claim.cited else "no citation"
        lines.append(f"{index}. {claim.text} [{cited}]")
    return "\n".join(lines)


def lesson_from_row(row: Mapping[str, Any]) -> Lesson | None:
    if "claims" not in row or row.get("declined") or ("lesson" in row and row["lesson"] is None):
        return None
    try:
        return Lesson.model_validate({key: row.get(key) for key in ("rule", "kind", "claims")})
    except ValidationError as exc:
        raise JudgeError(
            f"draft row for item {row.get('item_id')!r} by {row.get('drafter')!r} is not a "
            f"valid lesson: {_details(exc)}"
        ) from None


def entry_numbers(projection: str) -> set[int]:
    return {int(match.group(1)) for match in _ENTRY.finditer(projection)}


def cited_entries(projection: str, cited: Sequence[int]) -> list[tuple[int, str]]:
    """The numbered projection entries cited, each cut to a short excerpt."""
    starts = [(int(m.group(1)), m.start()) for m in _ENTRY.finditer(projection)]
    out = []
    for i, (number, start) in enumerate(starts):
        if number not in cited:
            continue
        end = starts[i + 1][1] if i + 1 < len(starts) else len(projection)
        body = projection[start:end].strip()
        if len(body) > EXCERPT_CHARS:
            body = body[:EXCERPT_CHARS].rstrip() + " …"
        out.append((number, body))
    return out
