from __future__ import annotations

import itertools
from datetime import timedelta

import duckdb
import pytest

from agent_memory.db import ALL_TABLES
from tests.support import Chain, insert_rows, new_id, scalar


def memory_row(chain: Chain, **overrides) -> dict:
    row = {
        "memory_id": new_id(),
        "scope": "global",
        "type": "fact",
        "content": "Ponds have reeds",
        "content_hash": "h",
        "trust_level": "agent",
        "created_by_session": chain.session_id,
    }
    row.update(overrides)
    return row


def action_row(memory_id: str, action: str, **overrides) -> dict:
    row = {"action_id": new_id(), "memory_id": memory_id, "action": action, "actor": "test"}
    row.update(overrides)
    return row


def status(con: duckdb.DuckDBPyConnection, memory_id: str) -> str:
    return scalar(con, "SELECT status FROM memory_status WHERE memory_id = ?", [memory_id])


def last_column(con: duckdb.DuckDBPyConnection, table: str) -> str:
    return scalar(
        con,
        "SELECT column_name FROM duckdb_columns() "
        "WHERE database_name = current_database() AND schema_name = 'main' AND table_name = ? "
        "ORDER BY column_index DESC LIMIT 1",
        [table],
    )


# Vocabulary


def test_agents_are_seeded_with_capture_tiers(con):
    rows = con.execute("SELECT agent_id, capture_tier FROM agents ORDER BY agent_id").fetchall()
    assert rows == [
        ("amp", "B"),
        ("factory-droid", "A"),
        ("human", "A"),
        ("jetbrains-junie", "C"),
        ("mistral-vibe", "A"),
        ("oh-my-pi", "A"),
    ]


def test_event_kinds_include_compaction_and_capture_error(con):
    kinds = {r[0] for r in con.execute("SELECT kind FROM event_kinds").fetchall()}
    assert {"compaction", "capture_error", "memory_recall", "assistant_msg"} <= kinds


# Append-only


@pytest.fixture
def every_table_has_rows(con, chain):
    insert_rows(con, {"memory_actions": [action_row(chain.memory_id, "confirm")]})
    for table in ALL_TABLES:
        assert con.execute(f"SELECT count(*) FROM {table}").fetchone()[0] > 0, table


def fingerprint(con: duckdb.DuckDBPyConnection, table: str) -> str:
    return scalar(
        con, f"SELECT md5(string_agg(t::VARCHAR, '|' ORDER BY t::VARCHAR)) FROM {table} t"
    )


# DuckDB checks foreign keys before AFTER triggers run, so rows that other
# tables reference fail with a foreign-key error first. Either way the change
# is rejected; the guard-specific tests below use unreferenced rows.
REJECTED = "is append-only|still referenced by a foreign key"


@pytest.mark.parametrize("table", ALL_TABLES)
def test_update_is_rejected_and_rows_are_unchanged(con, every_table_has_rows, table):
    col = last_column(con, table)
    before = fingerprint(con, table)
    with pytest.raises(duckdb.Error, match=REJECTED):
        con.execute(f'UPDATE {table} SET "{col}" = "{col}"')
    assert fingerprint(con, table) == before


@pytest.mark.parametrize("table", ALL_TABLES)
def test_delete_is_rejected_and_rows_are_unchanged(con, every_table_has_rows, table):
    before = fingerprint(con, table)
    with pytest.raises(duckdb.Error, match=REJECTED):
        con.execute(f"DELETE FROM {table}")
    assert fingerprint(con, table) == before


def test_guard_rejects_changes_to_unreferenced_rows(con, chain):
    with pytest.raises(duckdb.Error, match="agents is append-only"):
        con.execute("UPDATE agents SET notes = 'x' WHERE agent_id = 'human'")
    with pytest.raises(duckdb.Error, match="agents is append-only"):
        con.execute("DELETE FROM agents WHERE agent_id = 'human'")
    with pytest.raises(duckdb.Error, match="events is append-only"):
        con.execute("UPDATE events SET content = 'x' WHERE event_id = ?", [chain.end_event])
    with pytest.raises(duckdb.Error, match="events is append-only"):
        con.execute("DELETE FROM events WHERE event_id = ?", [chain.end_event])


def test_zero_row_update_and_delete_are_harmless(con, chain):
    con.execute("UPDATE events SET content = content WHERE false")
    con.execute("DELETE FROM events WHERE false")


def test_rejected_update_aborts_the_whole_transaction(con, chain):
    con.execute("BEGIN TRANSACTION")
    insert_rows(
        con,
        {
            "events": [
                {
                    "event_id": new_id(),
                    "session_id": chain.session_id,
                    "seq": 9,
                    "kind": "note",
                    "ts": chain.now,
                }
            ]
        },
    )
    with pytest.raises(duckdb.Error, match="events is append-only"):
        con.execute("UPDATE events SET content = 'rewritten' WHERE seq = 9")
    con.execute("ROLLBACK")
    assert con.execute("SELECT count(*) FROM events WHERE seq = 9").fetchone()[0] == 0
    assert con.execute("SELECT count(*) FROM events WHERE content = 'rewritten'").fetchone()[0] == 0


# Integrity


@pytest.mark.parametrize(
    ("table", "overrides", "message"),
    [
        ("sessions", {"agent_id": "letta-code"}, "foreign key"),
        ("sessions", {"capture_completeness": "most"}, "CHECK"),
        ("events", {"kind": "not_a_kind"}, "foreign key"),
        ("events", {"session_id": "00000000-0000-0000-0000-000000000000"}, "foreign key"),
        ("events", {"role": "narrator"}, "CHECK"),
        ("memories", {"type": "rumor"}, "foreign key"),
        ("memories", {"trust_level": "anonymous"}, "CHECK"),
        ("memories", {"confidence": 1.5}, "CHECK"),
        ("memories", {"supersedes": "00000000-0000-0000-0000-000000000000"}, "foreign key"),
        ("memory_actions", {"action": "delete"}, "foreign key"),
        ("retrieval_items", {"status_at_recall": "maybe"}, "CHECK"),
        ("context_links", {"retrieval_id": None}, "CHECK"),
    ],
)
def test_constraints_reject_bad_rows(con, chain, table, overrides, message):
    ranks = itertools.count(100)

    def valid() -> dict:
        return {
            "sessions": lambda: {
                "session_id": new_id(),
                "agent_id": "amp",
                "capture_completeness": "partial",
                "started_at": chain.now,
            },
            "events": lambda: {
                "event_id": new_id(),
                "session_id": chain.session_id,
                "seq": 50,
                "kind": "note",
                "ts": chain.now,
            },
            "memories": lambda: memory_row(chain),
            "memory_actions": lambda: action_row(chain.memory_id, "confirm"),
            "retrieval_items": lambda: {
                "retrieval_id": chain.retrieval_id,
                "rank": next(ranks),
                "memory_id": chain.memory_id,
                "injected": False,
            },
            "context_links": lambda: {
                "answer_event_id": chain.answer_event,
                "link_type": "retrieval",
                "retrieval_id": chain.retrieval_id,
            },
        }[table]()

    insert_rows(con, {table: [valid()]})
    with pytest.raises(duckdb.Error, match=message):
        insert_rows(con, {table: [{**valid(), **overrides}]})


def test_events_can_reference_a_parent_event(con, chain):
    insert_rows(
        con,
        {
            "events": [
                {
                    "event_id": new_id(),
                    "session_id": chain.session_id,
                    "seq": 5,
                    "kind": "tool_result",
                    "parent_event_id": chain.answer_event,
                    "ts": chain.now,
                }
            ]
        },
    )


# Derived memory status


def test_user_memory_is_active(con, chain):
    assert status(con, chain.memory_id) == "active"


def test_tool_memory_starts_quarantined_and_approval_activates_it(con, chain):
    row = memory_row(chain, trust_level="tool")
    insert_rows(con, {"memories": [row]})
    assert status(con, row["memory_id"]) == "quarantined"
    insert_rows(con, {"memory_actions": [action_row(row["memory_id"], "approve")]})
    assert status(con, row["memory_id"]) == "active"


def test_latest_review_wins_within_one_transaction(con, chain):
    con.execute("BEGIN TRANSACTION")
    insert_rows(
        con,
        {
            "memory_actions": [
                action_row(chain.memory_id, "approve"),
                action_row(chain.memory_id, "quarantine"),
            ]
        },
    )
    con.execute("COMMIT")
    assert status(con, chain.memory_id) == "quarantined"


def test_latest_review_is_by_time_not_insertion_order(con, chain):
    later = chain.now + timedelta(minutes=5)
    insert_rows(
        con,
        {
            "memory_actions": [
                action_row(chain.memory_id, "quarantine", created_at=later),
                action_row(chain.memory_id, "approve", created_at=chain.now),
            ]
        },
    )
    assert status(con, chain.memory_id) == "quarantined"


def test_retire_beats_a_later_approval(con, chain):
    insert_rows(
        con,
        {
            "memory_actions": [
                action_row(chain.memory_id, "retire", created_at=chain.now),
                action_row(chain.memory_id, "approve", created_at=chain.now + timedelta(hours=1)),
            ]
        },
    )
    assert status(con, chain.memory_id) == "retired"


def test_superseding_memory_replaces_the_old_one(con, chain):
    new = memory_row(chain, supersedes=chain.memory_id, content="Ducks prefer lakes")
    insert_rows(con, {"memories": [new]})
    assert status(con, chain.memory_id) == "superseded"
    assert status(con, new["memory_id"]) == "active"
    active = {
        r[0] for r in con.execute("SELECT memory_id::VARCHAR FROM active_memories").fetchall()
    }
    assert active == {new["memory_id"]}


def test_retired_takes_precedence_over_superseded(con, chain):
    insert_rows(
        con,
        {
            "memories": [memory_row(chain, supersedes=chain.memory_id)],
            "memory_actions": [action_row(chain.memory_id, "retire")],
        },
    )
    assert status(con, chain.memory_id) == "retired"


def test_expiry(con, chain):
    past = memory_row(chain, expires_at=chain.now - timedelta(days=1))
    future = memory_row(chain, expires_at=chain.now + timedelta(days=1))
    insert_rows(con, {"memories": [past, future]})
    assert status(con, past["memory_id"]) == "expired"
    assert status(con, future["memory_id"]) == "active"


def test_memory_history_merges_creation_and_actions(con, chain):
    new = memory_row(chain, supersedes=chain.memory_id)
    insert_rows(
        con,
        {
            "memories": [new],
            "memory_actions": [
                action_row(
                    chain.memory_id,
                    "flag_contradiction",
                    related_memory_id=new["memory_id"],
                    reason="lakes vs ponds",
                )
            ],
        },
    )
    rows = con.execute(
        "SELECT memory_id::VARCHAR, event, actor FROM memory_history ORDER BY ts, event"
    ).fetchall()
    assert (chain.memory_id, "created", "factory-droid") in rows
    assert (new["memory_id"], "created_superseding", "factory-droid") in rows
    assert (chain.memory_id, "flag_contradiction", "test") in rows


# Audit views


def test_why_answer_walks_back_to_the_user_statement(con, chain):
    rows = con.execute(
        "SELECT via, cited, memory_id::VARCHAR, source_kind, source_event_id::VARCHAR, "
        "source_agent_id, status_at_recall, memory_status_now "
        "FROM why_answer WHERE answer_event_id = ? ORDER BY via",
        [chain.answer_event],
    ).fetchall()
    assert rows == [
        (
            "cited_memory",
            True,
            chain.memory_id,
            "user_msg",
            chain.user_event,
            "factory-droid",
            None,
            "active",
        ),
        (
            "retrieval",
            True,
            chain.memory_id,
            "user_msg",
            chain.user_event,
            "factory-droid",
            "active",
            "active",
        ),
    ]


def test_why_answer_shows_status_change_since_recall(con, chain):
    insert_rows(con, {"memory_actions": [action_row(chain.memory_id, "retire")]})
    row = con.execute(
        "SELECT status_at_recall, memory_status_now FROM why_answer "
        "WHERE answer_event_id = ? AND via = 'retrieval'",
        [chain.answer_event],
    ).fetchone()
    assert row == ("active", "retired")


def test_why_answer_includes_context_events(con, chain):
    ctx = new_id()
    insert_rows(
        con,
        {
            "events": [
                {
                    "event_id": ctx,
                    "session_id": chain.session_id,
                    "seq": 0,
                    "kind": "system_context",
                    "content": "AGENTS.md",
                    "ts": chain.now,
                }
            ],
            "context_links": [
                {
                    "answer_event_id": chain.answer_event,
                    "link_type": "context_event",
                    "context_event_id": ctx,
                }
            ],
        },
    )
    row = con.execute(
        "SELECT source_event_id::VARCHAR, source_kind, source_content, memory_id FROM why_answer "
        "WHERE answer_event_id = ? AND via = 'context_event'",
        [chain.answer_event],
    ).fetchone()
    assert row == (ctx, "system_context", "AGENTS.md", None)


def test_session_summary(con, chain):
    row = con.execute(
        "SELECT agent_id, capture_tier, event_count, answers, recalls, ended_at IS NOT NULL "
        "FROM session_summary WHERE session_id = ?",
        [chain.session_id],
    ).fetchone()
    assert row == ("factory-droid", "A", 4, 1, 1, True)


def test_full_text_search_indexes_new_memories(con, chain):
    late = memory_row(chain, content="Herons stalk the shallows", key="birds")
    insert_rows(con, {"memories": [late]})

    def search(term: str) -> set[str]:
        return {
            r[0]
            for r in con.execute(
                "SELECT memory_id::VARCHAR FROM ("
                "SELECT *, fts_main_memories.match_bm25(memory_id, ?) AS score FROM memories"
                ") WHERE score IS NOT NULL",
                [term],
            ).fetchall()
        }

    assert search("ponds") == {chain.memory_id}
    assert search("herons") == {late["memory_id"]}
    assert search("birds") == {late["memory_id"]}


# Replay safety


def test_one_ledger_session_per_harness_session(con, chain):
    row = {
        "session_id": new_id(),
        "agent_id": "factory-droid",
        "harness_session_id": "abc123",
        "capture_completeness": "full",
        "started_at": chain.now,
    }
    insert_rows(con, {"sessions": [row]})
    with pytest.raises(duckdb.Error, match=r"UNIQUE|Duplicate"):
        insert_rows(con, {"sessions": [{**row, "session_id": new_id()}]})
    # A different harness may reuse the same id, and unknown ids never collide.
    insert_rows(con, {"sessions": [{**row, "session_id": new_id(), "agent_id": "amp"}]})
    for _ in range(2):
        insert_rows(
            con,
            {
                "sessions": [
                    {
                        **row,
                        "session_id": new_id(),
                        "agent_id": "oh-my-pi",
                        "harness_session_id": None,
                    }
                ]
            },
        )


def test_context_links_have_a_replayable_identity(con, chain):
    link = {
        "link_id": new_id(),
        "answer_event_id": chain.answer_event,
        "link_type": "cited_memory",
        "memory_id": chain.memory_id,
    }
    insert_rows(con, {"context_links": [link]})
    with pytest.raises(duckdb.Error, match=r"PRIMARY KEY|Duplicate"):
        insert_rows(con, {"context_links": [link]})
    insert_rows(con, {"context_links": [{k: v for k, v in link.items() if k != "link_id"}]})
    assert scalar(con, "SELECT count(*) FROM context_links WHERE link_id IS NULL") == 0
