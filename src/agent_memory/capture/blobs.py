"""Content-addressed storage for event text too large to keep inline."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

BLOB_THRESHOLD = 16 * 1024
INLINE_BYTES = 4 * 1024


@dataclass(frozen=True)
class Stored:
    content: str | None
    content_hash: str | None
    content_bytes: int | None
    truncated: bool
    blob_ref: str | None


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def store(blob_dir: Path, text: str | None) -> Stored:
    """Hash the text; spill it to blob_dir when it exceeds BLOB_THRESHOLD bytes."""
    if text is None:
        return Stored(None, None, None, False, None)
    data = text.encode()
    digest = hashlib.sha256(data).hexdigest()
    if len(data) <= BLOB_THRESHOLD:
        return Stored(text, digest, len(data), False, None)
    rel = Path(digest[:2]) / f"{digest}.txt"
    target = blob_dir / rel
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(f".{target.name}.{os.getpid()}.tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        os.replace(tmp, target)
    inline = data[:INLINE_BYTES].decode(errors="ignore")
    return Stored(inline, digest, len(data), True, rel.as_posix())
