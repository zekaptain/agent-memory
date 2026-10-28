from __future__ import annotations

import duckdb
import pytest

from agent_memory import db
from agent_memory.db import AgentMemoryError, Migration, available_migrations, initialize_home


def test_package_ships_migration_0001():
    labels = [m.label for m in available_migrations()]
    assert labels[0] == "0001_init"


def test_fresh_database_records_applied_migrations(con):
    rows = con.execute("SELECT version, name, checksum FROM schema_migrations").fetchall()
    assert rows == [(m.version, m.name, m.checksum) for m in available_migrations()]


def test_reinitializing_changes_nothing(cfg):
    token = cfg.token_path.read_text()
    result = initialize_home(cfg)
    assert result.applied == []
    assert not result.created_token
    assert cfg.token_path.read_text() == token


def test_edited_migration_is_refused(con, monkeypatch):
    edited = [Migration(m.version, m.name, m.sql + "\n-- edited\n") for m in available_migrations()]
    monkeypatch.setattr(db, "available_migrations", lambda: edited)
    with pytest.raises(AgentMemoryError, match="changed after it was applied"):
        db.migrate(con)


def test_unknown_applied_migration_is_refused(con, monkeypatch):
    monkeypatch.setattr(db, "available_migrations", lambda: [])
    with pytest.raises(AgentMemoryError, match="does not know about"):
        db.migrate(con)


def test_failed_migration_leaves_no_trace(con, monkeypatch):
    broken = Migration(2, "broken", "CREATE TABLE half_done (x INTEGER);\nSELECT error('boom');\n")
    shipped = available_migrations()
    monkeypatch.setattr(db, "available_migrations", lambda: [*shipped, broken])
    with pytest.raises(duckdb.Error, match="boom"):
        db.migrate(con)
    assert (
        con.execute(
            "SELECT count(*) FROM duckdb_tables() WHERE table_name = 'half_done'"
        ).fetchone()[0]
        == 0
    )
    assert [r[0] for r in con.execute("SELECT version FROM schema_migrations").fetchall()] == [1]


def test_new_migration_applies_and_guards_survive_alter(con, monkeypatch):
    extra = Migration(2, "extra", "ALTER TABLE events ADD COLUMN extra VARCHAR;\n")
    shipped = available_migrations()
    monkeypatch.setattr(db, "available_migrations", lambda: [*shipped, extra])
    assert db.migrate(con) == [extra]
    assert db.migrate(con) == []
    con.execute(
        "INSERT INTO sessions (session_id, agent_id, capture_completeness, started_at) "
        "VALUES (uuidv7(), 'human', 'full', now())"
    )
    with pytest.raises(duckdb.Error, match="sessions is append-only"):
        con.execute("UPDATE sessions SET model = 'x'")
