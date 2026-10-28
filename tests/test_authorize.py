"""The mem_authorize macro, evaluated directly. End-to-end enforcement is in test_server."""

from __future__ import annotations

import pytest

ALLOWED = [
    "SELECT 1",
    "  select * from events",
    "SELECT 1;",
    "SELECT 1 ;  ",
    "WITH x AS (SELECT 1 AS a) SELECT a FROM x",
    "FROM events LIMIT 5",
    "DESCRIBE events",
    "SUMMARIZE memories",
    "EXPLAIN SELECT * FROM why_answer",
    "BEGIN TRANSACTION",
    "COMMIT",
    "ROLLBACK",
    "INSERT INTO events (event_id) SELECT event_id FROM stage_events_1",
    "INSERT INTO main.memories SELECT * FROM stage",
    "insert into context_links select * from s",
    # Keywords and separators inside literals and quoted identifiers are data.
    "SELECT 'DROP TABLE events; -- gone' AS note",
    "SELECT 'it''s; DELETE FROM events' AS note",
    'SELECT "update" FROM (SELECT 1 AS "update")',
    'SELECT 1 AS "a""; drop"',
    "SELECT ';'",
    "SELECT * FROM events WHERE content LIKE '%quack_serve(%'",
]

DENIED = [
    "UPDATE events SET content = ''",
    "DELETE FROM events",
    "DROP TABLE events",
    "CREATE TABLE t (x INT)",
    "CREATE OR REPLACE MACRO mem_authorize(a, b) AS true",
    "ALTER TABLE events ADD COLUMN x INT",
    "TRUNCATE events",
    "ATTACH 'other.duckdb' AS o",
    "DETACH mem",
    "COPY events TO '/tmp/x.csv'",
    "EXPORT DATABASE '/tmp/x'",
    "INSTALL httpfs",
    "LOAD httpfs",
    "PRAGMA database_list",
    "SET enable_external_access = true",
    "RESET lock_configuration",
    "CALL quack_stop('quack:localhost:9494')",
    "CHECKPOINT",
    "VACUUM",
    "USE memory",
    "PREPARE p AS SELECT 1",
    "EXECUTE p",
    "SELECT 1; SELECT 2",
    "SELECT 1; DROP TABLE events",
    "SELECT 1 -- trailing comment",
    "SELECT /* inline */ 1",
    "SELECT $$dollar$$",
    "SELECT $tag$x$tag$",
    "SELECT E'escaped\\n'",
    "WITH d AS (DELETE FROM events RETURNING *) SELECT * FROM d",
    "EXPLAIN ANALYZE DELETE FROM events",
    "SELECT * FROM quack_active_connections()",
    "SELECT * FROM query('DROP TABLE events')",
    "SELECT * FROM query_table('events')",
    "INSERT INTO agents SELECT * FROM agents",
    "INSERT INTO schema_migrations SELECT * FROM schema_migrations",
    "INSERT INTO _append_only_guard VALUES ('x')",
    "INSERT INTO events_shadow SELECT * FROM events",
    "INSERT INTO fts_main_memories.docs SELECT 1",
    "SELECT 1 UNION ALL SELECT 2; INSERT INTO agents SELECT * FROM agents",
    "MERGE INTO events USING s ON true WHEN MATCHED THEN DELETE",
    "",
    "   ",
    "GRANT ALL ON events TO public",
]


def authorize(con, sql: str) -> bool:
    return con.execute("SELECT mem_authorize('session', ?)", [sql]).fetchone()[0]


@pytest.mark.parametrize("sql", ALLOWED)
def test_allowed(con, sql):
    assert authorize(con, sql) is True


@pytest.mark.parametrize("sql", DENIED)
def test_denied(con, sql):
    assert authorize(con, sql) is False
