from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta

import duckdb
import pytest

from agent_memory.audit import check, legacy_preflight
from agent_memory.cli import main
from agent_memory.config import Config
from agent_memory.db import MemoryClient, available_migrations, open_direct
from agent_memory.mcp_server.tools import MemoryTools
from agent_memory.smoke import free_port, synthetic_chain
from tests.support import insert_rows, new_id

WHEN = datetime(2025, 1, 1, tzinfo=UTC)
PRIVATE = "private transcript sentinel password=do-not-print-this"


@pytest.fixture
def old_cfg(tmp_path) -> Config:
    """An actual 0001 database, without running today's migration runner."""
    cfg = Config(home=tmp_path / "old-schema", port=free_port())
    cfg.home.mkdir()
    with duckdb.connect(str(cfg.db_path)) as con:
        con.execute("LOAD fts")
        con.execute(
            "CREATE TABLE schema_migrations (version INTEGER PRIMARY KEY, name VARCHAR, "
            "checksum VARCHAR, applied_at TIMESTAMPTZ DEFAULT now())"
        )
        migration = next(m for m in available_migrations() if m.version == 1)
        con.execute(migration.sql)
        con.execute(
            "INSERT INTO schema_migrations (version, name, checksum) VALUES (?, ?, ?)",
            [migration.version, migration.name, migration.checksum],
        )
    return cfg


@pytest.fixture
def old_con(old_cfg) -> Iterator[duckdb.DuckDBPyConnection]:
    with open_direct(old_cfg.db_path) as con:
        yield con


def memory(con, *, supersedes=None, trust="agent", created_at=WHEN, **extra) -> str:
    mid = new_id()
    insert_rows(
        con,
        {
            "memories": [
                {
                    "memory_id": mid,
                    "scope": "test",
                    "type": "fact",
                    "content": PRIVATE,
                    "content_hash": mid,
                    "trust_level": trust,
                    "supersedes": supersedes,
                    "created_at": created_at,
                    **extra,
                }
            ]
        },
    )
    return mid


def action(con, mid, name, at) -> str:
    aid = new_id()
    insert_rows(
        con,
        {
            "memory_actions": [
                {
                    "action_id": aid,
                    "memory_id": mid,
                    "action": name,
                    "actor": PRIVATE,
                    "reason": PRIVATE,
                    "created_at": at,
                }
            ]
        },
    )
    return aid


def codes(report) -> set[str]:
    return {f.code for f in report.findings}


def test_direct_old_schema_is_deterministic_and_never_initializes(old_cfg, capsys):
    before = hashlib.sha256(old_cfg.db_path.read_bytes()).hexdigest()
    argv = ["--home", str(old_cfg.home), "audit", "check", "--direct", "--json"]
    assert main(argv) == 0
    first = capsys.readouterr()
    report = json.loads(first.out)
    assert report["status"] == "clean"
    assert report["schema_versions"] == [1]
    assert report["findings"] == []
    assert main(argv) == 0
    assert capsys.readouterr().out == first.out
    assert hashlib.sha256(old_cfg.db_path.read_bytes()).hexdigest() == before
    assert sorted(p.name for p in old_cfg.home.iterdir()) == ["memory.duckdb"]


def test_supported_legacy_acceptance_records_do_not_invent_approvals(old_con):
    predecessor = memory(old_con)
    replacement = memory(old_con, supersedes=predecessor, created_at=WHEN + timedelta(days=1))
    # Later quarantine, expiry, or retirement must not erase a historical acceptance.
    action(old_con, replacement, "quarantine", WHEN + timedelta(days=2))
    action(old_con, replacement, "retire", WHEN + timedelta(days=3))
    result = legacy_preflight(old_con)
    assert result.ok
    assert len(result.acceptances) == 1
    accepted = result.acceptances[0]
    assert (accepted.predecessor_id, accepted.replacement_id) == (predecessor, replacement)
    assert accepted.method == "trusted_at_creation"
    assert accepted.evidence_action_id is None
    assert accepted.accepted_at == WHEN + timedelta(days=1)
    assert "legacy_acceptance_ambiguous" not in codes(check(old_con))


def test_tool_replacement_requires_actual_valid_approval(old_con):
    predecessor = memory(old_con)
    replacement = memory(
        old_con, supersedes=predecessor, trust="tool", created_at=WHEN + timedelta(days=1)
    )
    action(old_con, replacement, "confirm", WHEN + timedelta(days=2))
    before = legacy_preflight(old_con)
    assert not before.ok
    assert before.acceptances == ()
    assert "legacy_acceptance_ambiguous" in {f.code for f in before.blockers}
    approval = action(old_con, replacement, "approve", WHEN + timedelta(days=3))
    after = legacy_preflight(old_con)
    assert after.ok
    assert after.acceptances[0].method == "explicit_approval"
    assert after.acceptances[0].evidence_action_id == approval
    assert after.acceptances[0].accepted_at == WHEN + timedelta(days=3)


@pytest.mark.parametrize(
    "extra",
    [
        {"valid_from": WHEN + timedelta(days=5)},
        {"expires_at": WHEN + timedelta(hours=12)},
        {"valid_to": WHEN + timedelta(hours=12)},
    ],
)
def test_ineligible_replacements_are_ambiguous_even_with_approval(old_con, extra):
    predecessor = memory(old_con)
    replacement = memory(
        old_con, supersedes=predecessor, created_at=WHEN + timedelta(days=1), **extra
    )
    action(old_con, replacement, "approve", WHEN + timedelta(days=2))
    preflight = legacy_preflight(old_con)
    assert not preflight.ok
    assert not preflight.acceptances


@pytest.mark.parametrize("bound", ["valid_to", "expires_at"])
def test_invalid_intervals_block_preflight_even_without_supersedes(old_con, bound):
    mid = memory(old_con, valid_from=WHEN, **{bound: WHEN})
    report = check(old_con)
    assert "invalid_validity_interval" in codes(report)
    assert mid in {f.entity_id for f in report.findings}
    assert not legacy_preflight(old_con).ok


@pytest.mark.parametrize("review", ["retire", "quarantine"])
def test_replacement_of_inactive_predecessor_is_stale(old_con, review):
    predecessor = memory(old_con)
    action(old_con, predecessor, review, WHEN + timedelta(hours=1))
    replacement = memory(old_con, supersedes=predecessor, created_at=WHEN + timedelta(days=1))
    report = legacy_preflight(old_con)
    assert not report.ok
    assert not report.acceptances
    assert any(
        f.code == "legacy_replacement_stale" and f.entity_id == replacement for f in report.blockers
    )


def test_cross_scope_and_multiple_replacements_are_blockers(old_con):
    predecessor = memory(old_con)
    memory(old_con, supersedes=predecessor, scope="other", created_at=WHEN + timedelta(days=1))
    memory(old_con, supersedes=predecessor, created_at=WHEN + timedelta(days=2))
    result = legacy_preflight(old_con)
    assert not result.acceptances
    assert {"legacy_replacement_cross_scope", "legacy_replacement_multiple"} <= {
        f.code for f in result.blockers
    }


def test_same_time_predecessor_review_is_not_an_invented_order(old_con):
    predecessor = memory(old_con, trust="tool")
    action(old_con, predecessor, "approve", WHEN + timedelta(days=1))
    memory(old_con, supersedes=predecessor, created_at=WHEN + timedelta(days=1))
    result = legacy_preflight(old_con)
    assert not result.ok
    assert not result.acceptances
    assert "legacy_review_order_ambiguous" in {f.code for f in result.blockers}


def test_cyclic_replacements_on_damaged_legacy_schema_are_reported(old_con):
    # Normal FKs prevent cycles; a damaged/imported old schema must still be diagnosable.
    with duckdb.connect(":memory:") as damaged:
        columns = old_con.execute("DESCRIBE memories").fetchall()
        definitions = ", ".join(f'"{r[0]}" {r[1]}' for r in columns)
        damaged.execute(f"CREATE TABLE memories ({definitions})")
        damaged.execute(
            "CREATE TABLE memory_actions (action_id UUID, memory_id UUID, action VARCHAR, "
            "created_at TIMESTAMPTZ)"
        )
        first, second = new_id(), new_id()
        insert_rows(
            damaged,
            {
                "memories": [
                    {
                        "memory_id": mid,
                        "scope": "test",
                        "trust_level": "agent",
                        "supersedes": other,
                        "created_at": WHEN,
                    }
                    for mid, other in [(first, second), (second, first)]
                ]
            },
        )
        result = legacy_preflight(damaged)
    assert not result.acceptances
    assert "legacy_replacement_cycle" in {f.code for f in result.blockers}


def test_capture_assertions_and_payload_placeholders_are_not_proof(old_con):
    ids = [new_id() for _ in range(7)]
    rows = synthetic_chain(ids, WHEN, content=PRIVATE)
    rows["sessions"][0]["metadata"] = {"verified": True, "secret": PRIVATE}
    rows["events"][2]["payload"] = {"placeholder": True, "private": PRIVATE}
    # Even nonempty content does not resolve the explicit payload marker.
    rows["events"][2]["content"] = PRIVATE
    insert_rows(old_con, rows)
    report = check(old_con)
    assert {
        "capture_full_unverified",
        "capture_missing_assistant",
        "cite_placeholder_unresolved",
        "retrieval_missing_attribution",
    } <= codes(report)
    assert PRIVATE not in json.dumps(report.to_dict())
    assert ids[3] in {f.entity_id for f in report.findings}
    assert ids[6] in {f.entity_id for f in report.findings}


def test_partial_capture_and_orphan_result_are_explicit_gaps(old_con):
    sid, eid = new_id(), new_id()
    insert_rows(
        old_con,
        {
            "sessions": [
                {
                    "session_id": sid,
                    "agent_id": "mistral-vibe",
                    "started_at": WHEN,
                    "capture_completeness": "partial",
                }
            ],
            "events": [
                {
                    "event_id": eid,
                    "session_id": sid,
                    "seq": 400,  # Best-effort sequence numbers are not coverage proof.
                    "kind": "tool_result",
                    "ts": WHEN,
                    "content": PRIVATE,
                }
            ],
        },
    )
    report = check(old_con)
    assert {"capture_partial", "capture_tool_result_unlinked"} <= codes(report)
    assert not any("sequence" in code for code in codes(report))


def test_real_answer_and_retrieval_link_satisfy_attribution(old_con):
    ids = [new_id() for _ in range(7)]
    insert_rows(old_con, synthetic_chain(ids, WHEN))
    report = check(old_con)
    assert "retrieval_missing_attribution" not in codes(report)
    assert "cite_placeholder_unresolved" not in codes(report)
    assert "capture_full_unverified" in codes(report)


def test_empty_retrieval_still_needs_attribution_and_request(old_con):
    sid, rid = new_id(), new_id()
    insert_rows(
        old_con,
        {
            "sessions": [
                {
                    "session_id": sid,
                    "agent_id": "human",
                    "started_at": WHEN,
                    "capture_completeness": "partial",
                }
            ],
            "retrievals": [
                {"retrieval_id": rid, "session_id": sid, "strategy": "recent_v1", "returned": 0}
            ],
        },
    )
    assert {"retrieval_missing_attribution", "retrieval_request_missing"} <= codes(check(old_con))


def test_unknown_schema_does_not_report_clean(tmp_path, capsys):
    cfg = Config(home=tmp_path)
    duckdb.connect(str(cfg.db_path)).close()
    code = main(["--home", str(cfg.home), "audit", "check", "--direct", "--json"])
    output = capsys.readouterr()
    assert code == 2
    report = json.loads(output.out)
    assert report["status"] == "error"
    assert "schema_missing_columns" in {f["code"] for f in report["findings"]}


def test_quack_audit_uses_only_reads_and_never_direct(live_cfg, live_client, monkeypatch, capsys):
    ids = [new_id() for _ in range(7)]
    live_client.insert(synthetic_chain(ids, WHEN, content=PRIVATE))
    statements = []
    original = MemoryClient.query

    def query(self, sql):
        statements.append(sql)
        return original(self, sql)

    def forbidden(*args, **kwargs):
        pytest.fail("normal audit must not open the database file")

    monkeypatch.setattr(MemoryClient, "query", query)
    monkeypatch.setattr("agent_memory.cli.open_direct", forbidden)
    argv = ["--home", str(live_cfg.home), "--port", str(live_cfg.port), "audit", "check", "--json"]
    code = main(argv)
    output = capsys.readouterr()
    assert code == 1
    report = json.loads(output.out)
    assert report["status"] == "findings"
    assert PRIVATE not in output.out + output.err
    assert statements
    assert all(sql.lstrip().startswith("SELECT") for sql in statements)
    assert "capture_full_unverified" in {f["code"] for f in report["findings"]}


def test_direct_flag_always_opens_read_only(old_cfg, monkeypatch, capsys):
    seen = []

    def opened(path, *, read_only=False):
        seen.append(read_only)
        return open_direct(path, read_only=read_only)

    monkeypatch.setattr("agent_memory.cli.open_direct", opened)
    assert main(["--home", str(old_cfg.home), "audit", "check", "--direct"]) == 0
    assert "clean" in capsys.readouterr().out
    assert seen == [True]


def test_cli_connection_errors_are_structured_and_never_echo_details(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("AGENT_MEMORY_HOST", PRIVATE)
    code = main(["--home", str(tmp_path / "absent"), "audit", "check", "--json"])
    output = capsys.readouterr()
    assert code == 2
    report = json.loads(output.out)
    assert report["status"] == "error"
    assert report["findings"][0]["code"] == "audit_read_failed"
    assert PRIVATE not in output.out + output.err
    assert not (tmp_path / "absent").exists()


def test_audit_help_does_not_resolve_configuration(monkeypatch, capsys):
    monkeypatch.setenv("AGENT_MEMORY_HOST", "not-loopback")
    assert main(["audit", "check", "--help"]) == 0
    output = capsys.readouterr().out
    assert "--direct" in output
    assert "--json" in output


def test_null_start_is_not_fabricated_from_creation_time(old_con):
    memory(old_con, valid_to=WHEN - timedelta(days=1))
    assert "invalid_validity_interval" not in codes(check(old_con))


def test_validity_boundary_requires_an_actual_acceptance_event(old_con):
    predecessor = memory(old_con)
    replacement = memory(
        old_con,
        supersedes=predecessor,
        trust="tool",
        created_at=WHEN + timedelta(days=1),
        valid_from=WHEN + timedelta(days=3),
        valid_to=WHEN + timedelta(days=5),
    )
    action(old_con, replacement, "approve", WHEN + timedelta(days=2))
    assert not legacy_preflight(old_con).ok
    evidence = action(old_con, replacement, "approve", WHEN + timedelta(days=3))
    result = legacy_preflight(old_con)
    assert result.ok
    assert result.acceptances[0].evidence_action_id == evidence
    assert result.acceptances[0].accepted_at == WHEN + timedelta(days=3)


@pytest.mark.parametrize("bound", ["valid_to", "expires_at"])
def test_closed_predecessor_validity_is_stale(old_con, bound):
    predecessor = memory(old_con, **{bound: WHEN + timedelta(days=1)})
    memory(old_con, supersedes=predecessor, created_at=WHEN + timedelta(days=1))
    result = legacy_preflight(old_con)
    assert not result.acceptances
    assert "legacy_replacement_stale" in {f.code for f in result.blockers}


def test_backdated_predecessor_approval_cannot_prove_acceptance(old_con):
    predecessor = memory(old_con, trust="tool")
    action(old_con, predecessor, "approve", WHEN - timedelta(days=1))
    memory(old_con, supersedes=predecessor, created_at=WHEN + timedelta(days=1))
    result = legacy_preflight(old_con)
    assert not result.acceptances
    assert "legacy_review_before_creation" in {f.code for f in result.blockers}


@pytest.mark.parametrize("last_review", ["approve", "quarantine"])
def test_same_time_replacement_reviews_follow_rowid(old_con, last_review):
    predecessor = memory(old_con)
    replacement = memory(
        old_con, trust="tool", supersedes=predecessor, created_at=WHEN + timedelta(days=1)
    )
    first_review = "quarantine" if last_review == "approve" else "approve"
    first_id = action(old_con, replacement, first_review, WHEN + timedelta(days=2))
    last_id = action(old_con, replacement, last_review, WHEN + timedelta(days=2))
    result = legacy_preflight(old_con)
    # Approval proves historical acceptance even if a later same-time review withdraws it.
    assert result.ok
    assert result.acceptances[0].evidence_action_id == (
        last_id if last_review == "approve" else first_id
    )


def test_preflight_does_not_own_its_callers_transaction(old_con):
    old_con.execute("BEGIN")
    predecessor = memory(old_con)
    memory(old_con, supersedes=predecessor, created_at=WHEN + timedelta(days=1))
    assert len(legacy_preflight(old_con).acceptances) == 1
    old_con.execute("ROLLBACK")
    assert old_con.execute("SELECT count(*) FROM memories").fetchone() == (0,)


def test_additive_schema_introspection_does_not_require_latest_views(old_con):
    old_con.execute("ALTER TABLE memories ADD COLUMN extra_audit_metadata VARIANT")
    old_con.execute("CREATE TABLE future_audit_table (private_text VARCHAR)")
    old_con.execute("INSERT INTO future_audit_table VALUES (?)", [PRIVATE])
    old_con.execute("INSERT INTO schema_migrations (version, name) VALUES (9999, 'future')")
    report = check(old_con)
    assert report.schema_versions == (1, 9999)
    assert report.exit_code == 0
    assert PRIVATE not in json.dumps(report.to_dict())


def test_cross_session_link_is_not_answer_attribution(old_con):
    first, second = [[new_id() for _ in range(7)] for _ in range(2)]
    rows = synthetic_chain(first, WHEN)
    rows.pop("context_links")
    insert_rows(old_con, rows)
    insert_rows(old_con, synthetic_chain(second, WHEN))
    insert_rows(
        old_con,
        {
            "context_links": [
                {
                    "answer_event_id": second[3],
                    "link_type": "retrieval",
                    "retrieval_id": first[6],
                }
            ]
        },
    )
    missing = {
        f.entity_id for f in check(old_con).findings if f.code == "retrieval_missing_attribution"
    }
    assert first[6] in missing
    assert second[6] not in missing


def test_retrieval_item_count_and_status_are_diagnosed(old_con):
    ids = [new_id() for _ in range(7)]
    rows = synthetic_chain(ids, WHEN)
    rows["retrievals"][0]["returned"] = 2
    rows["retrieval_items"][0]["status_at_recall"] = None
    insert_rows(old_con, rows)
    assert {"retrieval_item_count_mismatch", "retrieval_status_missing"} <= codes(check(old_con))


def test_actual_mcp_cite_placeholder_over_quack(live_cfg, live_client, capsys):
    tools = MemoryTools(live_cfg, "factory-droid", harness_session_id=new_id(), project="test")
    try:
        recalled = tools.recall("", scopes=["test-audit-no-memories"])
        assert not recalled["degraded"]
        cited = tools.cite(retrieval_id=recalled["retrieval_id"])
        assert not cited["degraded"]
    finally:
        tools.close()
    argv = ["--home", str(live_cfg.home), "--port", str(live_cfg.port), "audit", "check", "--json"]
    assert main(argv) == 1
    report = json.loads(capsys.readouterr().out)
    assert any(
        f["code"] == "cite_placeholder_unresolved" and f["entity_id"] == cited["answer_event_id"]
        for f in report["findings"]
    )
    assert any(
        f["code"] == "retrieval_missing_attribution" and f["entity_id"] == recalled["retrieval_id"]
        for f in report["findings"]
    )


def test_unreachable_server_never_falls_back_to_direct(down_cfg, monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        pytest.fail("an unreachable server does not authorize direct access")

    monkeypatch.setattr("agent_memory.cli.open_direct", forbidden)
    argv = ["--home", str(down_cfg.home), "--port", str(down_cfg.port), "audit", "check", "--json"]
    assert main(argv) == 2
    output = capsys.readouterr()
    assert json.loads(output.out)["findings"][0]["code"] == "audit_read_failed"
    assert down_cfg.read_token() not in output.out + output.err


def test_nonempty_direct_report_is_read_only_and_secret_free(old_cfg, capsys):
    with open_direct(old_cfg.db_path) as con:
        predecessor = memory(con)
        memory(con, trust="tool", supersedes=predecessor, created_at=WHEN + timedelta(days=1))
    before = hashlib.sha256(old_cfg.db_path.read_bytes()).hexdigest()
    argv = ["--home", str(old_cfg.home), "audit", "check", "--direct"]
    assert main(argv) == 1
    output = capsys.readouterr()
    assert "legacy_acceptance_ambiguous" in output.out
    assert PRIVATE not in output.out + output.err
    assert hashlib.sha256(old_cfg.db_path.read_bytes()).hexdigest() == before


def test_explicit_approval_orders_predecessor_reviews_in_same_action_table(old_con):
    predecessor = memory(old_con, trust="tool")
    replacement = memory(
        old_con, supersedes=predecessor, trust="tool", created_at=WHEN + timedelta(days=1)
    )
    action(old_con, predecessor, "approve", WHEN + timedelta(days=2))
    approval = action(old_con, replacement, "approve", WHEN + timedelta(days=2))
    action(old_con, predecessor, "quarantine", WHEN + timedelta(days=2))
    result = legacy_preflight(old_con)
    assert result.ok
    assert result.acceptances[0].evidence_action_id == approval


def test_earliest_same_time_approval_is_the_acceptance_evidence(old_con):
    predecessor = memory(old_con)
    at = WHEN + timedelta(days=1)
    replacement = memory(old_con, supersedes=predecessor, trust="tool", created_at=at)
    approval = action(old_con, replacement, "approve", at)
    action(old_con, predecessor, "quarantine", at)
    action(old_con, replacement, "approve", at)
    result = legacy_preflight(old_con)
    assert result.ok
    assert result.acceptances[0].evidence_action_id == approval


def test_thinking_is_not_an_answer_or_attribution(old_con):
    rows = synthetic_chain([new_id() for _ in range(7)], WHEN)
    rows["events"][2]["payload"] = {"part": "thinking"}
    insert_rows(old_con, rows)
    assert {"capture_missing_assistant", "retrieval_missing_attribution"} <= codes(check(old_con))


def test_later_schema_binding_resolves_then_withdraws_placeholder(con):
    ids = [new_id() for _ in range(7)]
    placeholder = new_id()
    rows = synthetic_chain(ids, WHEN)
    rows["events"].append(
        {
            "event_id": placeholder,
            "session_id": ids[0],
            "kind": "assistant_msg",
            "payload": {"placeholder": True},
            "ts": WHEN,
            "seq": 8,
        }
    )
    for link in rows["context_links"]:
        link["answer_event_id"] = placeholder
    insert_rows(con, rows)
    binding = new_id()
    insert_rows(
        con,
        {
            "answer_bindings": [
                {
                    "binding_id": binding,
                    "placeholder_event_id": placeholder,
                    "answer_event_id": ids[3],
                    "root_placeholder_id": placeholder,
                    "evidence": {"private": PRIVATE},
                    "resolution_method": "manual",
                    "reason_code": "synthetic",
                }
            ]
        },
    )
    report = check(con)
    assert "cite_placeholder_unresolved" not in codes(report)
    assert "retrieval_missing_attribution" not in codes(report)
    assert PRIVATE not in json.dumps(report.to_dict())
    insert_rows(
        con,
        {
            "answer_bindings": [
                {
                    "binding_id": new_id(),
                    "placeholder_event_id": placeholder,
                    "answer_event_id": None,
                    "supersedes_binding_id": binding,
                    "evidence": {},
                    "resolution_method": "manual",
                    "reason_code": "withdrawn",
                }
            ]
        },
    )
    assert {"cite_placeholder_unresolved", "retrieval_missing_attribution"} <= codes(check(con))


@pytest.mark.parametrize("start_gap", [False, True])
def test_later_verified_coverage_must_cover_session_start(con, start_gap):
    ids = [new_id() for _ in range(7)]
    insert_rows(con, synthetic_chain(ids, WHEN))
    insert_rows(
        con,
        {
            "capture_observations": [
                {
                    "observation_id": new_id(),
                    "session_id": ids[0],
                    "source_mode": "supported_capture",
                    "adapter_version": "test",
                    "policy_version": "test",
                    "source_fingerprint": "synthetic",
                    "covered_from": WHEN + timedelta(seconds=int(start_gap)),
                    "covered_through": WHEN + timedelta(minutes=1),
                    "parse_errors": [],
                    "exclusions": [],
                    "complete_supported": True,
                }
            ]
        },
    )
    report = check(con)
    assert ("capture_full_unverified" in codes(report)) == start_gap
    assert ("capture_coverage_gap" in codes(report)) == start_gap


def test_later_observation_diagnostics_never_print_error_or_exclusion_text(con):
    ids = [new_id() for _ in range(7)]
    insert_rows(con, synthetic_chain(ids, WHEN))
    insert_rows(
        con,
        {
            "capture_observations": [
                {
                    "observation_id": new_id(),
                    "session_id": ids[0],
                    "source_mode": "transcript_import",
                    "adapter_version": "test",
                    "policy_version": "test",
                    "source_fingerprint": PRIVATE,
                    "covered_from": WHEN,
                    "covered_through": WHEN,
                    "parse_errors": [PRIVATE],
                    "exclusions": [PRIVATE],
                }
            ]
        },
    )
    report = check(con)
    assert {"capture_coverage_gap", "capture_parse_errors", "capture_exclusions"} <= codes(report)
    assert PRIVATE not in json.dumps(report.to_dict())


@pytest.mark.parametrize("trust", ["tool", "agent"])
def test_native_proposals_are_not_reconstructed_as_legacy(con, trust):
    ids = [new_id() for _ in range(7)]
    insert_rows(con, synthetic_chain(ids, WHEN))
    successor = memory(
        con, trust=trust, supersedes=ids[5], scope="global", created_at=datetime.now(UTC)
    )
    assert "legacy_acceptance_ambiguous" in codes(check(con))
    insert_rows(
        con,
        {
            "memory_actions": [
                {
                    "action_id": new_id(),
                    "memory_id": successor,
                    "related_memory_id": ids[5],
                    "event_id": ids[2],
                    "action": "propose_correction",
                    "actor": PRIVATE,
                }
            ]
        },
    )
    report = check(con)
    assert "legacy_acceptance_ambiguous" not in codes(report)
    assert report.legacy_acceptances == ()
    assert not any(f.severity == "error" for f in report.findings)
