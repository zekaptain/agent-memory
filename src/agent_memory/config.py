"""Paths and connection settings, resolved from the environment."""

from __future__ import annotations

import os
import stat
from dataclasses import dataclass
from pathlib import Path

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
