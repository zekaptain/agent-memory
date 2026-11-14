"""Synthetic citation/import order regressions through the real capture and MCP paths."""

from __future__ import annotations

import json
import re

import pytest

from agent_memory.adapters.droid import DroidAdapter
from agent_memory.capture import attribution, ids
from agent_memory.capture.attribution import reconcile_answers
from agent_memory.capture.pipeline import deliver
from agent_memory.db import MemoryClient, MemoryServer, sql_literal
from agent_memory.mcp_server import MemoryTools
from tests.support import new_id


def entries(placeholder, *, duplicate=False, outcome=True):
    data = [
        {"type": "session_start", "id": "s", "timestamp": "2026-01-01T00:00:00Z"},
        {
            "type": "message",
            "id": "call-message",
            "timestamp": "2026-01-01T00:01:00Z",
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "cite-call",
                        "name": "agent-memory__cite",
                        "input": {},
                    }
                ],
            },
        },
        {
            "type": "message",
            "id": "result-message",
            "timestamp": "2026-01-01T00:02:00Z",
            "message": {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "cite-call",
                        "content": json.dumps({"answer_event_id": placeholder, "links": 1}),
                    }
                ],
            },
        },
        {
            "type": "message",
            "id": "reply",
            "timestamp": "2026-01-01T00:03:00Z",
            "message": {"role": "assistant", "content": "Synthetic answer."},
        },
    ]
    if duplicate:
        data.append(
            {
                "type": "message",
                "id": "other-reply",
                "timestamp": "2026-01-01T00:04:00Z",
                "message": {"role": "assistant", "content": "Another possible answer."},
            }
        )
    if outcome:
        data.append({"type": "agent_turn_outcome", "turnId": "turn-1", "resultKind": "text"})
    return data


def ingest(path, data, cfg, **options):
    path.write_text("\n".join(json.dumps(row) for row in data) + "\n")
    delivery = deliver(cfg, DroidAdapter().from_transcript(path, **options), deadline=10)
    assert delivery.written, delivery.error


def binding(client, placeholder):
    return client.query(
        "SELECT answer_event_id::VARCHAR, reason_code FROM effective_answer_bindings "
        f"WHERE placeholder_event_id = {sql_literal(placeholder)}"
    )[1]


@pytest.mark.parametrize("graph", [False, True])
def test_empty_retrieval_remains_in_answer_audit(live_cfg, live_client, graph):
    tools = MemoryTools(live_cfg, "factory-droid", harness_session_id=f"miss-{graph}")
    try:
        result = tools.graph("no such synthetic entity") if graph else tools.recall("zzemptyzz")
        assert result["items"] == []
        assert not result["degraded"]
        cited = tools.cite(retrieval_id=result["retrieval_id"])
        _, rows = live_client.query(
            "SELECT returned, memory_id, rank, recall_outcome FROM why_answer "
            f"WHERE answer_event_id = {sql_literal(cited['answer_event_id'])}"
        )
        assert rows == [(0, None, None, "miss")]
        _, counts = live_client.query(
            "SELECT answers FROM session_summary "
            f"WHERE session_id = {sql_literal(tools.session_id)}"
        )
        assert counts == [(0,)]
    finally:
        tools.close()


def test_citation_before_import_and_repeated_import(live_cfg, live_client, tmp_path):
    path = tmp_path / "cite-before.jsonl"
    tools = MemoryTools(live_cfg, "factory-droid", harness_session_id=path.stem)
    try:
        recall = tools.recall("zznonezz")
        placeholder = tools.cite(retrieval_id=recall["retrieval_id"])["answer_event_id"]
        data = entries(placeholder)
        ingest(path, data, live_cfg)
        ingest(path, data, live_cfg)
        answer = ids.event_id("factory-droid", path.stem, "assistant_msg", "reply")
        assert binding(live_client, placeholder) == [(answer, "unique_captured_turn")]
        _, rows = live_client.query(
            "SELECT placeholder_event_id::VARCHAR, original_answer_event_id::VARCHAR, returned "
            f"FROM why_answer WHERE answer_event_id = {sql_literal(answer)}"
        )
        assert rows == [(placeholder, placeholder, 0)]
        assert reconcile_answers(live_client, [tools.session_id]) == 0
    finally:
        tools.close()


def test_incomplete_turn_stays_unresolved_then_incremental_import_binds(
    live_cfg, live_client, tmp_path
):
    path = tmp_path / "incremental-cite.jsonl"
    tools = MemoryTools(live_cfg, "factory-droid", harness_session_id=path.stem)
    try:
        result = tools.recall("zznonezz")
        placeholder = tools.cite(retrieval_id=result["retrieval_id"])["answer_event_id"]
        ingest(path, entries(placeholder, outcome=False), live_cfg)
        assert binding(live_client, placeholder) == []
        ingest(path, entries(placeholder), live_cfg, since_key="reply")
        assert binding(live_client, placeholder)[0][0] == ids.event_id(
            "factory-droid", path.stem, "assistant_msg", "reply"
        )
    finally:
        tools.close()


def test_new_ambiguous_evidence_corrects_binding_without_rewriting_history(
    live_cfg, live_client, tmp_path
):
    path = tmp_path / "ambiguous-cite.jsonl"
    tools = MemoryTools(live_cfg, "factory-droid", harness_session_id=path.stem)
    try:
        recall = tools.recall("zznonezz")
        placeholder = tools.cite(retrieval_id=recall["retrieval_id"])["answer_event_id"]
        ingest(path, entries(placeholder), live_cfg)
        assert binding(live_client, placeholder)[0][0]
        ingest(path, entries(placeholder, duplicate=True), live_cfg)
        assert binding(live_client, placeholder) == [(None, "ambiguous_captured_turn")]
        _, rows = live_client.query(
            "SELECT count(*) FROM answer_bindings "
            f"WHERE placeholder_event_id = {sql_literal(placeholder)}"
        )
        assert rows == [(2,)]
    finally:
        tools.close()


def test_import_before_citation_resolves_from_persisted_evidence(
    live_cfg, live_client, tmp_path, monkeypatch
):
    from tests.support import new_id

    path = tmp_path / "import-before-cite.jsonl"
    tools = MemoryTools(live_cfg, "factory-droid", harness_session_id=path.stem)
    placeholder = new_id()
    try:
        ingest(path, entries(placeholder), live_cfg)
        assert binding(live_client, placeholder) == []
        original = tools._event

        def event(client, kind, content, payload):
            row = original(client, kind, content, payload)
            if kind == "assistant_msg":
                row["event_id"] = placeholder
            return row

        monkeypatch.setattr(tools, "_event", event)
        recall = tools.recall("zznonezz")
        assert tools.cite(retrieval_id=recall["retrieval_id"])["answer_event_id"] == placeholder
        assert binding(live_client, placeholder)[0][0] == ids.event_id(
            "factory-droid", path.stem, "assistant_msg", "reply"
        )
    finally:
        tools.close()


def test_explicit_answer_id_is_honored(live_cfg, live_client, tmp_path):
    from tests.support import new_id

    path = tmp_path / "explicit-answer.jsonl"
    tools = MemoryTools(live_cfg, "factory-droid", harness_session_id=path.stem)
    try:
        ingest(path, entries(new_id(), outcome=False), live_cfg)
        actual = ids.event_id("factory-droid", path.stem, "assistant_msg", "reply")
        recall = tools.recall("zznonezz")
        assert (
            tools.cite(retrieval_id=recall["retrieval_id"], answer_event_id=actual)[
                "answer_event_id"
            ]
            == actual
        )
        _, rows = live_client.query(
            "SELECT original_answer_event_id::VARCHAR, placeholder_event_id FROM why_answer "
            f"WHERE answer_event_id = {sql_literal(actual)}"
        )
        assert rows == [(actual, None)]
    finally:
        tools.close()


def test_captured_recall_is_distinct_from_merely_returned(live_cfg, live_client, tmp_path):
    path = tmp_path / "captured-recall.jsonl"
    tools = MemoryTools(live_cfg, "factory-droid", harness_session_id=path.stem)
    try:
        recall = tools.recall("zznonezz")
        placeholder = tools.cite(retrieval_id=recall["retrieval_id"])["answer_event_id"]
        data = entries(placeholder)
        data[1:1] = [
            {
                "type": "message",
                "id": "recall-call-message",
                "timestamp": "2026-01-01T00:00:10Z",
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "recall-call",
                            "name": "agent-memory__recall",
                            "input": {},
                        }
                    ],
                },
            },
            {
                "type": "message",
                "id": "recall-result-message",
                "timestamp": "2026-01-01T00:00:20Z",
                "message": {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "recall-call",
                            "content": json.dumps(
                                {"retrieval_id": recall["retrieval_id"], "items": []}
                            ),
                        }
                    ],
                },
            },
        ]
        ingest(path, data, live_cfg)
        actual = ids.event_id("factory-droid", path.stem, "assistant_msg", "reply")
        _, rows = live_client.query(
            "SELECT returned, captured_in_context, declared_used FROM why_answer "
            f"WHERE answer_event_id = {sql_literal(actual)}"
        )
        assert rows == [(0, True, False)]
    finally:
        tools.close()


@pytest.mark.parametrize("name", ["cite", "other_server__cite", "other.cite"])
def test_unrelated_tool_cannot_create_or_invalidate_binding(live_cfg, live_client, tmp_path, name):
    path = tmp_path / ("unrelated-" + name.replace(".", "-") + ".jsonl")
    tools = MemoryTools(live_cfg, "factory-droid", harness_session_id=path.stem)
    try:
        recall = tools.recall("zznonezz")
        placeholder = tools.cite(retrieval_id=recall["retrieval_id"])["answer_event_id"]
        fake = entries(placeholder)
        fake[1]["message"]["content"][0]["name"] = name
        fake[1]["message"]["content"][0]["id"] = "unrelated-call"
        fake[2]["message"]["content"][0]["tool_use_id"] = "unrelated-call"
        fake[1]["id"] = "unrelated-call-message"
        fake[2]["id"] = "unrelated-result-message"
        ingest(path, fake, live_cfg)
        assert binding(live_client, placeholder) == []
        ingest(path, entries(placeholder), live_cfg)
        actual = ids.event_id("factory-droid", path.stem, "assistant_msg", "reply")
        assert binding(live_client, placeholder) == [(actual, "unique_captured_turn")]
        fake[3]["id"] = "wrong-reply"
        ingest(path, fake, live_cfg)
        assert binding(live_client, placeholder) == [(actual, "unique_captured_turn")]
    finally:
        tools.close()


def test_explicit_reply_id_before_import_preserves_citation(live_cfg, live_client, tmp_path):
    from tests.support import new_id

    path = tmp_path / "explicit-future.jsonl"
    tools = MemoryTools(live_cfg, "factory-droid", harness_session_id=path.stem)
    actual = ids.event_id("factory-droid", path.stem, "assistant_msg", "reply")
    try:
        recall = tools.recall("zznonezz")
        citation = tools.cite(retrieval_id=recall["retrieval_id"], answer_event_id=actual)
        assert not citation["degraded"]
        assert citation["answer_event_id"] == actual
        ingest(path, entries(new_id(), outcome=False), live_cfg)
        assert binding(live_client, citation["placeholder_event_id"]) == [
            (actual, "explicit_answer_id")
        ]
        _, rows = live_client.query(
            f"SELECT returned FROM why_answer WHERE answer_event_id = {sql_literal(actual)}"
        )
        assert rows == [(0,)]
    finally:
        tools.close()


def test_new_prompt_does_not_bind_an_interrupted_prior_turn(live_cfg, live_client, tmp_path):
    path = tmp_path / "interrupted-cite.jsonl"
    tools = MemoryTools(live_cfg, "factory-droid", harness_session_id=path.stem)
    try:
        recall = tools.recall("zznonezz")
        placeholder = tools.cite(retrieval_id=recall["retrieval_id"])["answer_event_id"]
        data = entries(placeholder)
        data.insert(
            3,
            {
                "type": "message",
                "id": "new-prompt",
                "timestamp": "2026-01-01T00:02:30Z",
                "message": {"role": "user", "content": "Different synthetic request."},
            },
        )
        ingest(path, data, live_cfg)
        assert binding(live_client, placeholder) == []
    finally:
        tools.close()


@pytest.fixture
def fresh(cfg):
    """A server on a ledger nothing has cited from yet, which the module's shared one is not."""
    server = MemoryServer(cfg, export_interval_hours=0)
    server.start()
    try:
        with MemoryClient.from_config(cfg) as client:
            yield cfg, client
    finally:
        server.stop()


def asked(monkeypatch) -> list[str]:
    """The SQL every client asks the server from here on."""
    seen: list[str] = []
    ask = MemoryClient.query

    def query(self, sql):
        seen.append(sql)
        return ask(self, sql)

    monkeypatch.setattr(MemoryClient, "query", query)
    return seen


def uncited(name):
    """A session with a reply and no citation."""
    return [row for row in entries(new_id()) if row.get("id") in ("s", "reply")] + [
        {"type": "agent_turn_outcome", "turnId": f"turn-{name}", "resultKind": "text"}
    ]


def test_a_write_opens_no_payload_when_no_citation_waits(fresh, tmp_path, monkeypatch):
    """Binding runs after every write. It read the payload of every event to find
    placeholders, which took seconds on a ledger of millions of events, so every hook there
    missed its deadline for a check that had never found a row."""
    cfg, _ = fresh
    ingest(tmp_path / "before.jsonl", uncited("before"), cfg)
    seen = asked(monkeypatch)
    ingest(tmp_path / "after.jsonl", uncited("after"), cfg)
    assert seen
    assert [sql for sql in seen if "payload" in sql] == []


def test_a_write_opens_only_the_payload_of_placeholders_still_unbound(fresh, tmp_path, monkeypatch):
    cfg, client = fresh
    path = tmp_path / "cited.jsonl"
    ingest(path, uncited("cited"), cfg)
    reply = ids.event_id("factory-droid", path.stem, "assistant_msg", "reply")
    tools = MemoryTools(cfg, "factory-droid", harness_session_id="citing")
    try:
        recall = tools.recall("zznonezz")
        named = tools.cite(retrieval_id=recall["retrieval_id"], answer_event_id=reply)
        assert named["answer_event_id"] == reply
        assert "placeholder_event_id" not in named
        placeholder = tools.cite(retrieval_id=recall["retrieval_id"])["answer_event_id"]
    finally:
        tools.close()
    seen = asked(monkeypatch)
    ingest(tmp_path / "elsewhere.jsonl", uncited("elsewhere"), cfg)
    opened = [sql for sql in seen if "payload" in sql]
    assert opened
    looked_up: set[str] = set()
    for sql in opened:
        by_id = re.search(r" FROM events WHERE event_id IN \(([^)]*)\)$", sql)
        assert by_id, sql
        looked_up.update(by_id.group(1).replace("'", "").split(", "))
    assert looked_up == {placeholder}
    assert binding(client, placeholder) == []


@pytest.mark.parametrize("side", ["placeholder", "reply"])
def test_a_later_write_to_either_session_finishes_an_explicit_binding(
    live_cfg, live_client, tmp_path, monkeypatch, side
):
    path = tmp_path / f"finish-{side}.jsonl"
    reply = ids.event_id("factory-droid", path.stem, "assistant_msg", "reply")
    tools = MemoryTools(live_cfg, "factory-droid", harness_session_id=f"citing-{side}")
    try:
        recall = tools.recall("zznonezz")
        cited = tools.cite(retrieval_id=recall["retrieval_id"], answer_event_id=reply)
        placeholder = cited["placeholder_event_id"]
        with monkeypatch.context() as patch:
            # The reply is committed and the binding step after it does not run.
            patch.setattr("agent_memory.capture.writer.reconcile_answers", lambda *a, **k: 0)
            ingest(path, entries(new_id(), outcome=False), live_cfg)
        assert binding(live_client, placeholder) == []
        elsewhere = ids.session_id("factory-droid", f"elsewhere-{side}")
        assert reconcile_answers(live_client, [elsewhere]) == 0
        assert binding(live_client, placeholder) == []
        session = (
            tools.session_id
            if side == "placeholder"
            else ids.session_id("factory-droid", path.stem)
        )
        assert reconcile_answers(live_client, [session]) == 1
        assert binding(live_client, placeholder) == [(reply, "explicit_answer_id")]
        assert reconcile_answers(live_client, [session]) == 0
    finally:
        tools.close()


def test_a_placeholder_is_not_bound_as_the_reply_another_names(live_cfg, live_client, monkeypatch):
    tools = MemoryTools(live_cfg, "factory-droid", harness_session_id="names-a-placeholder")
    named = new_id()
    try:
        recall = tools.recall("zznonezz")
        first = tools.cite(retrieval_id=recall["retrieval_id"], answer_event_id=named)
        original = tools._event

        def event(client, kind, content, payload):
            row = original(client, kind, content, payload)
            if kind == "assistant_msg":
                row["event_id"] = named
            return row

        monkeypatch.setattr(tools, "_event", event)
        second = tools.cite(retrieval_id=recall["retrieval_id"])
        assert not second["degraded"], second
        assert second["answer_event_id"] == named
        assert binding(live_client, first["placeholder_event_id"]) == []
    finally:
        tools.close()


def test_more_placeholders_wait_than_one_lookup_holds(live_cfg, live_client, tmp_path, monkeypatch):
    monkeypatch.setattr(attribution, "LOOKUP_IDS", 2)
    path = tmp_path / "many-waiting.jsonl"
    replies = [
        ids.event_id("factory-droid", path.stem, "assistant_msg", key)
        for key in ("reply", "other-reply")
    ]
    tools = MemoryTools(live_cfg, "factory-droid", harness_session_id="many-citing")
    try:
        recall = tools.recall("zznonezz")
        placeholders = [
            tools.cite(retrieval_id=recall["retrieval_id"], answer_event_id=mid)[
                "placeholder_event_id"
            ]
            for mid in (*replies, new_id())
        ]
        ingest(path, entries(new_id(), duplicate=True, outcome=False), live_cfg)
        assert [binding(live_client, mid) for mid in placeholders] == [
            [(replies[0], "explicit_answer_id")],
            [(replies[1], "explicit_answer_id")],
            [],
        ]
    finally:
        tools.close()
