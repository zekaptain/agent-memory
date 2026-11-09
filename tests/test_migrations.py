from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import uuid4

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
    broken = Migration(
        max(m.version for m in available_migrations()) + 1,
        "broken",
        "CREATE TABLE half_done (x INTEGER);\nSELECT error('boom');\n",
    )
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
    assert [r[0] for r in con.execute("SELECT version FROM schema_migrations").fetchall()] == [
        m.version for m in shipped
    ]


def test_new_migration_applies_and_guards_survive_alter(con, monkeypatch):
    extra = Migration(
        max(m.version for m in available_migrations()) + 1,
        "extra",
        "ALTER TABLE events ADD COLUMN extra VARCHAR;\n",
    )
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


@pytest.fixture
def legacy():
    with duckdb.connect(":memory:") as con:
        con.execute("LOAD fts")
        con.execute(db._BOOTSTRAP)
        migration = available_migrations()[0]
        con.execute(migration.sql)
        con.execute(
            "INSERT INTO schema_migrations(version, name, checksum) VALUES (?, ?, ?)",
            [migration.version, migration.name, migration.checksum],
        )
        yield con


def legacy_memory(con, *, supersedes=None, trust="agent", day=0):
    mid = str(uuid4())
    con.execute(
        "INSERT INTO memories "
        "(memory_id, scope, type, content, content_hash, trust_level, supersedes, created_at) "
        "VALUES (?, 'legacy', 'fact', 'synthetic', ?, ?, ?, ?)",
        [mid, mid, trust, supersedes, datetime(2025, 1, 1, tzinfo=UTC) + timedelta(days=day)],
    )
    return mid


def test_legacy_reconstruction_keeps_evidence_and_never_revives(legacy):
    old = legacy_memory(legacy)
    new = legacy_memory(legacy, supersedes=old, day=1)
    legacy.execute(
        "INSERT INTO memory_actions(action_id, memory_id, action, actor, created_at) "
        "VALUES (?, ?, 'retire', 'test', ?)",
        [str(uuid4()), new, datetime(2025, 1, 3, tzinfo=UTC)],
    )
    db.migrate(legacy)
    assert legacy.execute(
        "SELECT predecessor_id::VARCHAR, successor_id::VARCHAR, method FROM memory_replacements"
    ).fetchall() == [(old, new, "legacy_trusted_at_creation")]
    statuses = dict(
        legacy.execute("SELECT memory_id::VARCHAR, status FROM memory_status").fetchall()
    )
    assert statuses == {old: "superseded", new: "retired"}
    assert (
        legacy.execute("SELECT count(*) FROM memory_actions WHERE action = 'approve'").fetchone()[0]
        == 0
    )
    assert (
        legacy.execute(
            "SELECT json_extract_string(metadata::JSON, '$.legacy_evidence.memory_id') "
            "FROM memory_actions WHERE action = 'propose_correction'"
        ).fetchone()[0]
        == new
    )
    assert db.migrate(legacy) == []


def test_ambiguous_legacy_migration_blocks_before_new_schema(legacy):
    old = legacy_memory(legacy)
    legacy_memory(legacy, supersedes=old, trust="tool", day=1)
    with pytest.raises(AgentMemoryError, match="legacy_acceptance_ambiguous"):
        db.migrate(legacy)
    assert legacy.execute("SELECT version FROM schema_migrations").fetchall() == [(1,)]
    assert (
        legacy.execute(
            "SELECT count(*) FROM duckdb_tables() WHERE table_name = 'memory_replacements'"
        ).fetchone()[0]
        == 0
    )
    assert legacy.execute("SELECT count(*) FROM memories").fetchone()[0] == 2


def test_explicit_legacy_approval_is_named_in_acceptance(legacy):
    old = legacy_memory(legacy)
    new = legacy_memory(legacy, supersedes=old, trust="tool", day=1)
    approval = str(uuid4())
    legacy.execute(
        "INSERT INTO memory_actions(action_id, memory_id, action, actor, created_at) "
        "VALUES (?, ?, 'approve', 'test', ?)",
        [approval, new, datetime(2025, 1, 3, tzinfo=UTC)],
    )
    db.migrate(legacy)
    assert legacy.execute(
        "SELECT evidence_action_id::VARCHAR, method FROM memory_replacements"
    ).fetchall() == [(approval, "legacy_explicit_approval")]
