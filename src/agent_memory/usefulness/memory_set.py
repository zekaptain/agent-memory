"""Memory set creation, manifest hashing, verification, and per-trial home cloning.

A memory set is an immutable snapshot of a stopped home's database file and blobs
copied once into ``<experiment_home>/memory-set/`` with a write-once manifest.
Each memory-arm trial receives its own fresh trial home copied from the memory set,
leaving the memory set untouched and keeping every trial's writes readable after
the run.
"""

from __future__ import annotations

import os
import re
import shutil
import time
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import duckdb

from ..config import Config
from ..db import AgentMemoryError, new_token, open_direct
from ..judge.store import lock, private_dir, utc_now_iso
from .store import (
    UsefulnessError,
    ensure_experiment_home,
    read_json_file,
    sha256_file,
    sha256_json,
    write_once_json,
)

MEMORY_SET_MANIFEST_SCHEMA_VERSION = 1
_DATE_ONLY = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_TERMINAL_STATUSES = frozenset({"retired", "superseded", "deduplicated"})


@dataclass(frozen=True)
class MemorySetResult:
    """Result of creating a memory set from a stopped home."""

    memory_set_dir: Path
    manifest_path: Path
    manifest: dict[str, Any]
    manifest_sha256: str
    copy_seconds: float


@dataclass(frozen=True)
class TrialHomeCopy:
    """Result of cloning a memory set into one fresh trial home."""

    trial_home: Path
    memory_set_sha256: str
    database_bytes: int
    blobs_bytes: int
    total_bytes: int
    copy_seconds: float
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "trial_home": str(self.trial_home),
            "memory_set_sha256": self.memory_set_sha256,
            "database_bytes": self.database_bytes,
            "blobs_bytes": self.blobs_bytes,
            "total_bytes": self.total_bytes,
            "copy_seconds": self.copy_seconds,
            "created_at": self.created_at,
        }


@dataclass(frozen=True)
class DiskEstimate:
    """Baseline disk and copy-time estimate for memory-arm trial homes."""

    memory_set_bytes: int
    trial_count: int
    baseline_total_bytes: int
    copy_seconds_per_trial: float | None = None
    estimated_total_copy_seconds: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "memory_set_bytes": self.memory_set_bytes,
            "trial_count": self.trial_count,
            "baseline_total_bytes": self.baseline_total_bytes,
            "copy_seconds_per_trial": self.copy_seconds_per_trial,
            "estimated_total_copy_seconds": self.estimated_total_copy_seconds,
        }


def memory_set_dir(experiment_home: Config | Path) -> Path:
    """Path to ``<experiment_home>/memory-set`` after enforcing the home guard."""
    return ensure_experiment_home(experiment_home) / "memory-set"


def memory_set_manifest_path(experiment_home: Config | Path) -> Path:
    """Path to ``<experiment_home>/memory-set/manifest.json``."""
    return memory_set_dir(experiment_home) / "manifest.json"


def _parse_utc_boundary(value: str | datetime, *, label: str, end_of_day: bool) -> datetime:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            raise UsefulnessError(f"{label} must be timezone-aware, not naive")
        return value.astimezone(UTC)
    if not isinstance(value, str) or not value.strip():
        raise UsefulnessError(f"{label} must be an ISO-8601 date or timezone-aware timestamp")
    text = value.strip()
    if _DATE_ONLY.fullmatch(text):
        parsed_date = date.fromisoformat(text)
        if end_of_day:
            return datetime(
                parsed_date.year,
                parsed_date.month,
                parsed_date.day,
                23,
                59,
                59,
                999999,
                tzinfo=UTC,
            )
        return datetime(parsed_date.year, parsed_date.month, parsed_date.day, tzinfo=UTC)
    normalized = text[:-1] + "+00:00" if text.endswith(("Z", "z")) else text
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise UsefulnessError(f"{label} is not a valid ISO-8601 date/timestamp: {value!r}") from exc
    if parsed.tzinfo is None:
        raise UsefulnessError(
            f"{label} timestamp must include a timezone offset or Z (or be YYYY-MM-DD): {value!r}"
        )
    return parsed.astimezone(UTC)


def _format_utc(dt: datetime) -> str:
    return dt.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _assert_not_nested(a: Path, b: Path, *, a_name: str, b_name: str) -> None:
    if a == b or a.is_relative_to(b) or b.is_relative_to(a):
        raise UsefulnessError(f"{a_name} ({a}) and {b_name} ({b}) must be separate directories")


def _assert_source_stopped(source_home: Path) -> Path:
    """Verify ``source_home`` holds a database that no server is serving."""
    if not source_home.exists() or not source_home.is_dir():
        raise UsefulnessError(f"source home does not exist or is not a directory: {source_home}")
    source_db = source_home / "memory.duckdb"
    if source_db.is_symlink():
        raise UsefulnessError(f"source database must not be a symlink: {source_db}")
    if not source_db.is_file():
        raise UsefulnessError(f"no database at {source_db}; run `mem init` first")
    wal = source_home / "memory.duckdb.wal"
    if wal.exists() and wal.stat().st_size > 0:
        raise UsefulnessError(
            f"source home {source_home} has an uncheckpointed WAL file ({wal.name}); "
            "stop the server cleanly before making a memory set"
        )
    try:
        con = duckdb.connect(str(source_db), read_only=True)
    except (duckdb.IOException, duckdb.ConnectionException) as exc:
        raise UsefulnessError(
            f"source home {source_home} is currently served or locked ({exc}); "
            "stop `mem serve` before making a memory set"
        ) from None
    else:
        con.close()
    return source_db


def _copy_private_file(src: Path, dst: Path) -> int:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    os.chmod(dst, 0o600)
    return dst.stat().st_size


def _copy_blobs(src_blobs: Path, dst_blobs: Path) -> dict[str, dict[str, Any]]:
    """Copy every regular file from ``src_blobs`` into ``dst_blobs`` and hash each copy."""
    private_dir(dst_blobs)
    if not src_blobs.exists():
        return {}
    if src_blobs.is_symlink() or not src_blobs.is_dir():
        raise UsefulnessError(f"blobs path must be a regular directory: {src_blobs}")

    files: dict[str, dict[str, Any]] = {}
    for root, dirs, filenames in os.walk(src_blobs, followlinks=False):
        root_path = Path(root)
        dirs.sort()
        filenames.sort()
        for dname in dirs:
            dpath = root_path / dname
            if dpath.is_symlink():
                raise UsefulnessError(f"blobs directory must not contain symlinks: {dpath}")
        for fname in filenames:
            fpath = root_path / fname
            if fpath.is_symlink() or not fpath.is_file():
                raise UsefulnessError(f"blobs entry must be a regular file: {fpath}")
            rel = fpath.relative_to(src_blobs)
            rel_posix = rel.as_posix()
            dst_path = dst_blobs / rel
            if dst_path.parent != dst_blobs:
                private_dir(dst_path.parent)
            size = _copy_private_file(fpath, dst_path)
            files[rel_posix] = {
                "bytes": size,
                "sha256": sha256_file(dst_path),
            }
    return dict(sorted(files.items()))


def _scan_blobs(blobs_dir: Path) -> dict[str, dict[str, Any]]:
    """Hash every regular file in ``blobs_dir``, refusing symlinks or non-regular files."""
    if not blobs_dir.exists():
        return {}
    if blobs_dir.is_symlink() or not blobs_dir.is_dir():
        raise UsefulnessError(f"memory set blobs path is not a regular directory: {blobs_dir}")

    files: dict[str, dict[str, Any]] = {}
    for root, dirs, filenames in os.walk(blobs_dir, followlinks=False):
        root_path = Path(root)
        dirs.sort()
        filenames.sort()
        for dname in dirs:
            dpath = root_path / dname
            if dpath.is_symlink():
                raise UsefulnessError(f"memory set blobs contains symlink: {dpath}")
        for fname in filenames:
            fpath = root_path / fname
            if fpath.is_symlink() or not fpath.is_file():
                raise UsefulnessError(f"memory set blobs contains non-regular file: {fpath}")
            rel_posix = fpath.relative_to(blobs_dir).as_posix()
            files[rel_posix] = {
                "bytes": fpath.stat().st_size,
                "sha256": sha256_file(fpath),
            }
    return dict(sorted(files.items()))


def _inspect_ledger(
    db_path: Path, *, cutoff: datetime, last_day: datetime
) -> tuple[int, list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Read migration level, memory counts, and validity-boundary counts from ``db_path``."""
    try:
        con = open_direct(db_path, read_only=True)
    except (AgentMemoryError, duckdb.Error) as exc:
        raise UsefulnessError(f"could not inspect copied database {db_path}: {exc}") from None
    try:
        try:
            mig_rows = con.execute(
                "SELECT version, name, checksum FROM schema_migrations ORDER BY version"
            ).fetchall()
        except duckdb.Error as exc:
            raise UsefulnessError(
                f"database {db_path} is missing schema_migrations "
                f"(not an initialized ledger): {exc}"
            ) from None
        if not mig_rows:
            raise UsefulnessError(f"database {db_path} has no applied migrations")
        applied_migrations = [
            {"version": int(v), "name": str(n), "checksum": str(c)} for v, n, c in mig_rows
        ]
        migration_level = max(int(v) for v, _, _ in mig_rows)

        total_row = con.execute("SELECT count(*) FROM memories").fetchone()
        total = int(total_row[0]) if total_row is not None else 0
        by_status = {
            str(status): int(n)
            for status, n in con.execute(
                "SELECT status, count(*) FROM memory_status GROUP BY 1 ORDER BY 1"
            ).fetchall()
        }
        by_scope = {
            str(scope): int(n)
            for scope, n in con.execute(
                "SELECT scope, count(*) FROM memories GROUP BY 1 ORDER BY 1"
            ).fetchall()
        }
        by_type = {
            str(mtype): int(n)
            for mtype, n in con.execute(
                "SELECT type, count(*) FROM memories GROUP BY 1 ORDER BY 1"
            ).fetchall()
        }
        active_by_scope = {
            str(scope): int(n)
            for scope, n in con.execute(
                "SELECT m.scope, count(*) FROM memories m "
                "JOIN memory_status s USING (memory_id) "
                "WHERE s.status = 'active' GROUP BY 1 ORDER BY 1"
            ).fetchall()
        }
        active_by_type = {
            str(mtype): int(n)
            for mtype, n in con.execute(
                "SELECT m.type, count(*) FROM memories m "
                "JOIN memory_status s USING (memory_id) "
                "WHERE s.status = 'active' GROUP BY 1 ORDER BY 1"
            ).fetchall()
        }
        counts: dict[str, Any] = {
            "total": total,
            "by_status": by_status,
            "by_scope": by_scope,
            "by_type": by_type,
            "active_by_scope": active_by_scope,
            "active_by_type": active_by_type,
        }

        validity_rows = con.execute(
            """
            SELECT s.status, m.valid_from, m.valid_to, m.expires_at
            FROM memories m
            JOIN memory_status s USING (memory_id)
            WHERE m.valid_from IS NOT NULL
               OR m.valid_to IS NOT NULL
               OR m.expires_at IS NOT NULL
            """
        ).fetchall()
    finally:
        con.close()

    starts_in_window = 0
    ends_in_window = 0
    total_crossing = 0
    non_terminal_crossing = 0
    for status, valid_from, valid_to, expires_at in validity_rows:
        vf = valid_from.astimezone(UTC) if isinstance(valid_from, datetime) else None
        vt = valid_to.astimezone(UTC) if isinstance(valid_to, datetime) else None
        ex = expires_at.astimezone(UTC) if isinstance(expires_at, datetime) else None

        ends_candidates = [dt for dt in (vt, ex) if dt is not None]
        effective_end = min(ends_candidates) if ends_candidates else None

        starts = vf is not None and cutoff <= vf <= last_day
        ends = effective_end is not None and cutoff <= effective_end <= last_day
        if starts:
            starts_in_window += 1
        if ends:
            ends_in_window += 1
        if starts or ends:
            total_crossing += 1
            if str(status) not in _TERMINAL_STATUSES:
                non_terminal_crossing += 1

    validity_changes: dict[str, Any] = {
        "cutoff": _format_utc(cutoff),
        "last_day": _format_utc(last_day),
        "total_crossing": total_crossing,
        "starts_in_window": starts_in_window,
        "ends_in_window": ends_in_window,
        "non_terminal_crossing": non_terminal_crossing,
    }
    return migration_level, applied_migrations, counts, validity_changes


def _combined_memory_set_sha256(
    *,
    db_sha256: str,
    blobs_sha256: str,
    migration_level: int,
    cutoff: str,
) -> str:
    return sha256_json(
        {
            "database_sha256": db_sha256,
            "blobs_sha256": blobs_sha256,
            "migration_level": migration_level,
            "cutoff": cutoff,
        }
    )


def make_memory_set(
    source_home: Config | Path,
    experiment_home: Config | Path,
    *,
    cutoff: str | datetime,
    last_day: str | datetime,
    created_at: str | None = None,
) -> MemorySetResult:
    """Copy a stopped home's database and blobs into ``experiment_home/memory-set`` once.

    Refuses the default home as ``experiment_home``, refuses a source home that a server
    is serving, refuses overwriting an existing memory set, and writes ``manifest.json``
    once with per-file hashes, migration level, memory counts, and validity-boundary counts.
    """
    exp_home = ensure_experiment_home(experiment_home)
    src_home = (source_home.home if isinstance(source_home, Config) else source_home).expanduser()
    src_home = src_home.resolve()
    _assert_not_nested(src_home, exp_home, a_name="source home", b_name="experiment home")

    cutoff_dt = _parse_utc_boundary(cutoff, label="cutoff", end_of_day=False)
    last_day_dt = _parse_utc_boundary(last_day, label="last_day", end_of_day=True)
    if last_day_dt < cutoff_dt:
        raise UsefulnessError(
            f"last_day ({_format_utc(last_day_dt)}) must not be earlier than "
            f"cutoff ({_format_utc(cutoff_dt)})"
        )

    private_dir(exp_home)
    target_dir = exp_home / "memory-set"
    manifest_file = target_dir / "manifest.json"
    lock_file = exp_home / "memory-set.lock"

    with lock(lock_file):
        if manifest_file.exists() or (target_dir / "memory.duckdb").exists():
            raise UsefulnessError(
                f"memory set already exists at {target_dir} and cannot be overwritten"
            )
        source_db = _assert_source_stopped(src_home)

        staging = exp_home / f".memory-set.staging.{os.getpid()}"
        if staging.exists():
            shutil.rmtree(staging)
        private_dir(staging)
        try:
            t0 = time.perf_counter()
            staged_db = staging / "memory.duckdb"
            db_bytes = _copy_private_file(source_db, staged_db)
            blob_files = _copy_blobs(src_home / "blobs", staging / "blobs")
            copy_seconds = time.perf_counter() - t0

            db_sha256 = sha256_file(staged_db)
            blobs_sha256 = sha256_json(blob_files)
            blobs_total_bytes = sum(int(entry["bytes"]) for entry in blob_files.values())

            migration_level, applied_migrations, counts, validity_changes = _inspect_ledger(
                staged_db, cutoff=cutoff_dt, last_day=last_day_dt
            )

            cutoff_iso = _format_utc(cutoff_dt)
            last_day_iso = _format_utc(last_day_dt)
            combined_sha256 = _combined_memory_set_sha256(
                db_sha256=db_sha256,
                blobs_sha256=blobs_sha256,
                migration_level=migration_level,
                cutoff=cutoff_iso,
            )
            manifest: dict[str, Any] = {
                "schema_version": MEMORY_SET_MANIFEST_SCHEMA_VERSION,
                "created_at": created_at if created_at is not None else utc_now_iso(),
                "cutoff": cutoff_iso,
                "last_day": last_day_iso,
                "migration_level": migration_level,
                "applied_migrations": applied_migrations,
                "database": {
                    "path": "memory.duckdb",
                    "bytes": db_bytes,
                    "sha256": db_sha256,
                },
                "blobs": {
                    "file_count": len(blob_files),
                    "total_bytes": blobs_total_bytes,
                    "sha256": blobs_sha256,
                    "files": blob_files,
                },
                "total_bytes": db_bytes + blobs_total_bytes,
                "memory_set_sha256": combined_sha256,
                "counts": counts,
                "validity_changes": validity_changes,
            }

            if target_dir.exists():
                if any(target_dir.iterdir()):
                    raise UsefulnessError(
                        f"memory set directory {target_dir} is not empty; refusing to overwrite"
                    )
                target_dir.rmdir()
            os.replace(staging, target_dir)
            manifest_sha256 = write_once_json(manifest_file, manifest)
        except Exception:
            if staging.exists():
                shutil.rmtree(staging, ignore_errors=True)
            raise

    return MemorySetResult(
        memory_set_dir=target_dir,
        manifest_path=manifest_file,
        manifest=manifest,
        manifest_sha256=manifest_sha256,
        copy_seconds=copy_seconds,
    )


def verify_memory_set(experiment_home: Config | Path) -> dict[str, Any]:
    """Recompute the database and blob hashes of a memory set and verify against its manifest."""
    ms_dir = memory_set_dir(experiment_home)
    manifest_file = ms_dir / "manifest.json"
    manifest = read_json_file(manifest_file)
    if manifest.get("schema_version") != MEMORY_SET_MANIFEST_SCHEMA_VERSION:
        raise UsefulnessError(
            f"unsupported memory-set manifest schema_version: {manifest.get('schema_version')!r}"
        )

    db_info = manifest.get("database")
    blobs_info = manifest.get("blobs")
    if not isinstance(db_info, Mapping) or not isinstance(blobs_info, Mapping):
        raise UsefulnessError(f"malformed memory-set manifest at {manifest_file}")

    db_path = ms_dir / "memory.duckdb"
    if db_path.is_symlink() or not db_path.is_file():
        raise UsefulnessError(f"memory set database missing or invalid: {db_path}")
    actual_db_sha = sha256_file(db_path)
    expected_db_sha = str(db_info.get("sha256") or "")
    if actual_db_sha != expected_db_sha:
        raise UsefulnessError(
            f"memory set database hash mismatch: expected {expected_db_sha}, got {actual_db_sha}"
        )

    actual_blob_files = _scan_blobs(ms_dir / "blobs")
    expected_blob_files = blobs_info.get("files")
    if not isinstance(expected_blob_files, Mapping):
        raise UsefulnessError(f"malformed blobs entry in {manifest_file}")
    if actual_blob_files != dict(expected_blob_files):
        raise UsefulnessError("memory set blobs do not match manifest")
    actual_blobs_sha = sha256_json(actual_blob_files)
    if actual_blobs_sha != str(blobs_info.get("sha256") or ""):
        raise UsefulnessError("memory set blobs digest mismatch")

    expected_combined = _combined_memory_set_sha256(
        db_sha256=actual_db_sha,
        blobs_sha256=actual_blobs_sha,
        migration_level=int(manifest["migration_level"]),
        cutoff=str(manifest["cutoff"]),
    )
    if expected_combined != str(manifest.get("memory_set_sha256") or ""):
        raise UsefulnessError("memory set combined hash mismatch")

    return manifest


def create_trial_home(
    experiment_home: Config | Path,
    trial_home: Config | Path,
    *,
    verify: bool = True,
) -> TrialHomeCopy:
    """Clone the memory set into a fresh, writable trial home with its own server token.

    Never writes to the memory set itself. The resulting trial home is ready for
    ``mem serve`` and stays readable after the trial finishes.
    """
    ms_dir = memory_set_dir(experiment_home)
    dest_home = ensure_experiment_home(trial_home)
    _assert_not_nested(ms_dir, dest_home, a_name="memory set", b_name="trial home")

    manifest = (
        verify_memory_set(experiment_home) if verify else read_json_file(ms_dir / "manifest.json")
    )

    if dest_home.exists() and any(dest_home.iterdir()):
        raise UsefulnessError(
            f"trial home {dest_home} already exists and is not empty; each trial needs a fresh home"
        )

    private_dir(dest_home)
    for sub in ("blobs", "export", "spool"):
        private_dir(dest_home / sub)

    token_path = dest_home / "token"
    fd = os.open(token_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(new_token() + "\n")

    t0 = time.perf_counter()
    db_bytes = _copy_private_file(ms_dir / "memory.duckdb", dest_home / "memory.duckdb")
    copied_blobs = _copy_blobs(ms_dir / "blobs", dest_home / "blobs")
    copy_seconds = time.perf_counter() - t0
    blobs_bytes = sum(int(entry["bytes"]) for entry in copied_blobs.values())

    return TrialHomeCopy(
        trial_home=dest_home,
        memory_set_sha256=str(manifest["memory_set_sha256"]),
        database_bytes=db_bytes,
        blobs_bytes=blobs_bytes,
        total_bytes=db_bytes + blobs_bytes,
        copy_seconds=copy_seconds,
        created_at=utc_now_iso(),
    )


def estimate_run_disk_usage(
    manifest: Mapping[str, Any],
    trial_count: int,
    *,
    copy_seconds_per_trial: float | None = None,
) -> DiskEstimate:
    """Estimate baseline disk space (and optional copy time) for ``trial_count`` memory trials."""
    if isinstance(trial_count, bool) or trial_count < 0:
        raise UsefulnessError(f"trial_count must be a non-negative integer, got {trial_count!r}")
    if "total_bytes" in manifest:
        ms_bytes = int(manifest["total_bytes"])
    else:
        db_bytes = int(manifest["database"]["bytes"])
        blobs_bytes = int(manifest["blobs"]["total_bytes"])
        ms_bytes = db_bytes + blobs_bytes
    if ms_bytes < 0:
        raise UsefulnessError(f"invalid memory set byte size: {ms_bytes}")
    total_copy = (
        copy_seconds_per_trial * trial_count if copy_seconds_per_trial is not None else None
    )
    return DiskEstimate(
        memory_set_bytes=ms_bytes,
        trial_count=trial_count,
        baseline_total_bytes=ms_bytes * trial_count,
        copy_seconds_per_trial=copy_seconds_per_trial,
        estimated_total_copy_seconds=total_copy,
    )
