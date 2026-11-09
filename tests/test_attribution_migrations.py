"""Migrate synthetic old evidence without rewriting events or retrieval snapshots."""

from __future__ import annotations

from datetime import UTC, datetime

from agent_memory import db
from agent_memory.config import Config
from agent_memory.smoke import free_port, synthetic_chain
from tests.support import insert_rows, new_id


def test_old_capture_and_miss_rows_survive_new_views(tmp_path, monkeypatch):
    cfg = Config(home=tmp_path / "legacy", port=free_port())
    migrations = db.available_migrations()
    with monkeypatch.context() as patch:
        patch.setattr(db, "available_migrations", lambda: migrations[:1])
        db.initialize_home(cfg)
    con = db.open_direct(cfg.db_path)
    try:
        sid, user, request, answer, end, memory, retrieval = [new_id() for _ in range(7)]
        rows = synthetic_chain(
            [sid, user, request, answer, end, memory, retrieval], datetime.now(UTC)
        )
        rows["retrievals"][0]["returned"] = 0
        rows["retrieval_items"] = []
        for event in rows["events"]:
            if event["event_id"] == answer:
                event["content"] = None
                event["payload"] = {"placeholder": True}
        insert_rows(con, rows)
        tables = (
            "sessions",
            "events",
            "memories",
            "retrievals",
            "retrieval_items",
            "context_links",
        )
        before = {
            table: con.execute(f"SELECT t::VARCHAR FROM {table} t ORDER BY t::VARCHAR").fetchall()
            for table in tables
        }
        assert con.execute(
            "SELECT count(*) FROM why_answer WHERE via = 'retrieval'"
        ).fetchone() == (0,)
        db.migrate(con)
        after = {
            table: con.execute(f"SELECT t::VARCHAR FROM {table} t ORDER BY t::VARCHAR").fetchall()
            for table in tables
        }
        assert before == after
        assert con.execute(
            "SELECT returned, rank, memory_id, status_at_recall FROM why_answer "
            "WHERE via = 'retrieval' AND answer_event_id = ?",
            [answer],
        ).fetchall() == [(0, None, None, None)]
        assert con.execute(
            "SELECT capture_completeness, recorded_capture_completeness, answers "
            "FROM session_summary WHERE session_id = ?",
            [sid],
        ).fetchone() == ("unverified_legacy_assertion", "full", 0)
        assert db.migrate(con) == []
    finally:
        con.close()
