from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from agent_memory.config import Config, ConfigError, load_config


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for var in ("AGENT_MEMORY_HOME", "AGENT_MEMORY_HOST", "AGENT_MEMORY_PORT"):
        monkeypatch.delenv(var, raising=False)


def test_defaults():
    cfg = load_config()
    assert cfg.home == Path("~/.agent-memory").expanduser()
    assert cfg.uri == "quack:localhost:9494"
    assert cfg.db_path == cfg.home / "memory.duckdb"


def test_environment_overrides(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_MEMORY_HOME", str(tmp_path))
    monkeypatch.setenv("AGENT_MEMORY_HOST", "127.0.0.1")
    monkeypatch.setenv("AGENT_MEMORY_PORT", "9999")
    cfg = load_config()
    assert (cfg.home, cfg.uri) == (tmp_path, "quack:127.0.0.1:9999")


def test_arguments_beat_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_MEMORY_HOME", "/nope")
    monkeypatch.setenv("AGENT_MEMORY_PORT", "1")
    cfg = load_config(home=tmp_path, port=1234)
    assert (cfg.home, cfg.port) == (tmp_path, 1234)


def test_ipv6_loopback_is_bracketed(monkeypatch):
    monkeypatch.setenv("AGENT_MEMORY_HOST", "::1")
    assert load_config().uri == "quack:[::1]:9494"


@pytest.mark.parametrize("host", ["0.0.0.0", "example.com", "192.168.1.5"])
def test_non_local_hosts_are_refused(monkeypatch, host):
    monkeypatch.setenv("AGENT_MEMORY_HOST", host)
    with pytest.raises(ConfigError, match="not a local address"):
        load_config()


def test_bad_port(monkeypatch):
    monkeypatch.setenv("AGENT_MEMORY_PORT", "ninety")
    with pytest.raises(ConfigError, match="not a port number"):
        load_config()


def test_token_checks(tmp_path):
    cfg = Config(home=tmp_path)
    with pytest.raises(ConfigError, match="mem init"):
        cfg.read_token()
    cfg.token_path.write_text("short\n")
    os.chmod(cfg.token_path, 0o600)
    with pytest.raises(ConfigError, match="valid token"):
        cfg.read_token()
    cfg.token_path.write_text("x" * 40 + "\n")
    assert cfg.read_token() == "x" * 40
    os.chmod(cfg.token_path, 0o644)
    with pytest.raises(ConfigError, match="chmod 600"):
        cfg.read_token()


def test_init_creates_private_home_and_token(cfg):
    assert stat.S_IMODE(cfg.home.stat().st_mode) == 0o700
    assert stat.S_IMODE(cfg.token_path.stat().st_mode) == 0o600
    for directory in (cfg.export_dir, cfg.spool_dir, cfg.blob_dir):
        assert directory.is_dir()
