from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from agent_memory.cli import main, render_table
from agent_memory.db import ALL_TABLES
from agent_memory.smoke import free_port


@dataclass(frozen=True)
class Mem:
    home: Path
    port: int
    capsys: pytest.CaptureFixture[str]

    def __call__(self, *args: str) -> tuple[int, str, str]:
        code = main(["--home", str(self.home), "--port", str(self.port), *args])
        out = self.capsys.readouterr()
        return code, out.out, out.err


@pytest.fixture
def mem(tmp_path, monkeypatch, capsys) -> Mem:
    for var in ("AGENT_MEMORY_HOME", "AGENT_MEMORY_HOST", "AGENT_MEMORY_PORT"):
        monkeypatch.delenv(var, raising=False)
    return Mem(home=tmp_path / "home", port=free_port(), capsys=capsys)


def test_version_and_help(mem):
    code, out, _ = mem("--version")
    assert (code, out.strip()) == (0, "agent-memory 0.1.0")
    code, out, _ = mem("--help")
    assert code == 0
    for command in ("init", "migrate", "serve", "sql", "export", "smoke"):
        assert command in out


def test_subcommand_help_works_with_a_bad_environment(mem, monkeypatch):
    monkeypatch.setenv("AGENT_MEMORY_HOST", "example.com")
    code, out, _ = mem("sql", "--help")
    assert code == 0
    assert "--direct" in out


def test_bad_environment_is_reported(mem, monkeypatch):
    monkeypatch.setenv("AGENT_MEMORY_HOST", "example.com")
    code, _, err = mem("init")
    assert code == 1
    assert err.startswith("mem: ")
    assert "not a local address" in err


def test_write_requires_direct(mem):
    code, _, err = mem("sql", "--write", "SELECT 1")
    assert code == 2
    assert "--direct" in err


def test_init_then_reinit(mem):
    code, out, _ = mem("init")
    assert code == 0
    assert "created" in out
    assert "0001_init" in out
    code, out, _ = mem("init")
    assert code == 0
    assert "kept existing" in out
    assert "up to date" in out


def test_migrate_status(mem):
    mem("init")
    code, out, _ = mem("migrate", "--status")
    assert code == 0
    assert "0001_init" in out
    assert "applied" in out


def test_direct_sql_as_json(mem):
    mem("init")
    code, out, _ = mem(
        "sql", "--direct", "--json", "SELECT agent_id FROM agents ORDER BY 1 LIMIT 2"
    )
    assert code == 0
    assert json.loads(out) == [{"agent_id": "admission"}, {"agent_id": "amp"}]


def test_direct_sql_is_read_only_by_default(mem):
    mem("init")
    code, _, err = mem(
        "sql",
        "--direct",
        "INSERT INTO sessions (session_id, agent_id, capture_completeness, started_at) "
        "VALUES (uuidv7(), 'human', 'full', now())",
    )
    assert code == 1
    assert "read-only" in err.lower()


def test_direct_export(mem):
    mem("init")
    code, _, _ = mem("export", "--direct")
    assert code == 0
    latest = mem.home / "export" / "latest"
    assert sorted(p.stem for p in latest.iterdir()) == sorted(ALL_TABLES)
    assert not list((mem.home / "export").glob(".tmp-*"))


def test_sql_without_server_fails_cleanly(mem):
    mem("init")
    code, _, err = mem("sql", "SELECT 1")
    assert code == 1
    assert "Is `mem serve` running?" in err


def test_commands_before_init_explain_what_to_do(mem):
    code, _, err = mem("sql", "--direct", "SELECT 1")
    assert code == 1
    assert "mem init" in err


def test_render_table():
    out = render_table(["a", "b"], [(1, None), ("x\ny", "z" * 10)], max_width=5)
    assert out.splitlines() == [
        "a    | b    ",
        "-----+------",
        "1    | NULL ",
        "x\\ny | zzzz…",
        "(2 rows)",
    ]
    assert render_table([], [], 5) == "OK"
