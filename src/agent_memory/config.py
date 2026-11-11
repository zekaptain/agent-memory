"""Paths and connection settings, resolved from the environment and from what `mem` was
given before its subcommand."""

from __future__ import annotations

import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import typer

DEFAULT_HOME = Path("~/.agent-memory")
DEFAULT_PORT = 9494
LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


class ConfigError(Exception):
    """The environment or on-disk layout is not usable."""


@dataclass(frozen=True)
class Config:
    home: Path
    host: str = "localhost"
    port: int = DEFAULT_PORT

    @property
    def db_path(self) -> Path:
        return self.home / "memory.duckdb"

    @property
    def token_path(self) -> Path:
        return self.home / "token"

    @property
    def export_dir(self) -> Path:
        return self.home / "export"

    @property
    def spool_dir(self) -> Path:
        return self.home / "spool"

    @property
    def blob_dir(self) -> Path:
        return self.home / "blobs"

    @property
    def uri(self) -> str:
        host = f"[{self.host}]" if ":" in self.host else self.host
        return f"quack:{host}:{self.port}"

    def read_token(self) -> str:
        try:
            mode = self.token_path.stat().st_mode
        except FileNotFoundError:
            raise ConfigError(f"No token at {self.token_path}. Run `mem init` first.") from None
        if mode & (stat.S_IRWXG | stat.S_IRWXO):
            raise ConfigError(
                f"{self.token_path} is accessible by other users. Run: chmod 600 {self.token_path}"
            )
        token = self.token_path.read_text().strip()
        if len(token) < 16:
            raise ConfigError(f"{self.token_path} does not contain a valid token.")
        return token


def load_config(home: str | Path | None = None, port: int | None = None) -> Config:
    home_path = Path(home or os.environ.get("AGENT_MEMORY_HOME") or DEFAULT_HOME).expanduser()
    host = os.environ.get("AGENT_MEMORY_HOST", "localhost")
    if host not in LOCAL_HOSTS:
        raise ConfigError(
            f"AGENT_MEMORY_HOST={host!r} is not a local address. The server only listens locally."
        )
    if port is None:
        raw = os.environ.get("AGENT_MEMORY_PORT", str(DEFAULT_PORT))
        try:
            port = int(raw)
        except ValueError:
            raise ConfigError(f"AGENT_MEMORY_PORT={raw!r} is not a port number.") from None
    return Config(home=home_path, host=host, port=port)


@dataclass(frozen=True)
class RootOptions:
    """What `mem` was given before its subcommand. The root callback puts it on the context."""

    home: Path | None = None
    port: int | None = None


def command_config(ctx: typer.Context) -> Config:
    """The configuration of the `mem` command that `ctx` belongs to.

    Each command resolves it, not the root callback, so `mem <cmd> --help` works even when
    the environment is misconfigured. A context without the root options is refused: the
    command would otherwise run against the default home."""
    opts = ctx.obj
    if not isinstance(opts, RootOptions):
        raise RuntimeError("mem's root callback did not run")
    return load_config(home=opts.home, port=opts.port)
