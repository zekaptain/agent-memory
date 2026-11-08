"""Build the packaged Go frontend once per source/toolchain/platform into a private cache."""

from __future__ import annotations

import fcntl
import hashlib
import os
import platform
import shutil
import stat
import subprocess
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path

from .store import JudgeError

PROTOCOL_VERSION = 1
MAX_FRAME = 8 * 1024 * 1024
SOURCE = Path(__file__).with_name("label_ui")


def cache_root() -> Path:
    if sys.platform == "darwin":
        base = Path.home() / "Library" / "Caches"
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache")))
        if not base.is_absolute():
            base = Path.home() / ".cache"
    return base / "agent-memory" / "label-ui"


def private_directory(path: Path) -> None:
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    info = path.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid():
        raise JudgeError("Unsafe label UI cache. Use --plain or repair the cache permissions.")
    path.chmod(0o700)


def source_key(source: Path, toolchain: str) -> str:
    digest = hashlib.sha256()
    for value in (
        str(PROTOCOL_VERSION),
        sys.platform,
        platform.machine(),
        toolchain,
    ):
        digest.update(value.encode() + b"\0")
    files = sorted([*source.glob("*.go"), source / "go.mod", source / "go.sum"])
    if not files or any(not path.is_file() for path in files):
        raise JudgeError("Label UI sources are missing. Reinstall agent-memory or use --plain.")
    for path in files:
        digest.update(path.name.encode() + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()


def frontend_binary(
    *,
    source: Path = SOURCE,
    root: Path | None = None,
    progress: Callable[[str], None] | None = None,
) -> Path:
    go = shutil.which("go")
    if go is None:
        raise JudgeError("Label UI needs Go. Install Go, then retry, or use judge label --plain.")
    env = {
        **os.environ,
        "GOTOOLCHAIN": "local",
        "GOWORK": "off",
        "GOFLAGS": "",
        "GOENV": "off",
        "CGO_ENABLED": "0",
        "GOOS": sys.platform,
        "GOARCH": {"x86_64": "amd64", "aarch64": "arm64"}.get(
            platform.machine(), platform.machine()
        ),
    }
    try:
        version = subprocess.run(
            [go, "version"], capture_output=True, check=True, timeout=15, env=env
        ).stdout.decode("utf-8")
        key = source_key(source, version)
        root = root or cache_root()
        private_directory(root.parent)
        private_directory(root)
        directory = root / key
        private_directory(directory)
        # O_NOFOLLOW refuses lock-file symlinks. flock is released even on interruption.
        fd = os.open(directory / "build.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, "rb") as lock:
            if os.fstat(lock.fileno()).st_uid != os.getuid():
                raise JudgeError("Unsafe label UI build lock. Repair the cache or use --plain.")
            os.fchmod(lock.fileno(), 0o600)
            fcntl.flock(lock, fcntl.LOCK_EX)
            binary = directory / "label-ui"
            if binary.exists():
                info = binary.lstat()
                if (
                    stat.S_ISREG(info.st_mode)
                    and info.st_uid == os.getuid()
                    and stat.S_IMODE(info.st_mode) == 0o700
                ):
                    return binary
                raise JudgeError("Unsafe cached label UI. Repair the cache or use --plain.")
            if progress:
                progress("Building label UI. First-time dependency downloads may need network.")
            fd, temporary = tempfile.mkstemp(prefix=".build-", dir=directory)
            os.close(fd)
            try:
                subprocess.run(
                    [
                        go,
                        "build",
                        "-mod=readonly",
                        "-buildvcs=false",
                        "-trimpath",
                        "-o",
                        temporary,
                        ".",
                    ],
                    cwd=source,
                    env=env,
                    capture_output=True,
                    check=True,
                    timeout=300,
                )
                os.chmod(temporary, 0o700)
                os.replace(temporary, binary)
            finally:
                Path(temporary).unlink(missing_ok=True)
            return binary
    except (OSError, subprocess.SubprocessError, UnicodeError):
        raise JudgeError(
            "Could not build the label UI. Check Go and network access for dependencies, "
            "then retry, or use judge label --plain."
        ) from None
