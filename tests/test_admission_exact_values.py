"""Unit tests for deterministic exact-value verification on admission lessons."""

from __future__ import annotations

import pytest

from agent_memory import endpoints
from agent_memory.admission.exact_values import (
    EXACT_VALUES_VERSION,
    check_draft,
    check_exact_values,
    extract_values,
    full_cited_entries,
)


def test_check_is_deterministic_and_makes_no_model_call(monkeypatch):
    def fail_http(*args, **kwargs):
        raise AssertionError("exact-value check must never make an HTTP/model call")

    monkeypatch.setattr(endpoints, "post_json", fail_http)
    projection = (
        "Coding-agent turn.\n"
        "[1] REQUEST (human): Stop mem serve before editing src/agent_memory/db.py.\n"
        '[2] TOOL CALL Execute: {"command": "uv run pytest --tb=short"}\n'
    )
    rule = (
        "Stop `mem serve` before editing src/agent_memory/db.py and run uv run pytest --tb=short."
    )
    first = check_exact_values(rule, [1, 2], projection)
    second = check_exact_values(rule, [1, 2], projection)
    assert first == second
    assert first["version"] == EXACT_VALUES_VERSION
    assert first["ok"] is True
    assert first["missing"] == []
    assert first["redacted_cited"] is False
    assert first["values"] == [
        "mem serve",
        "src/agent_memory/db.py",
        "uv",
        "pytest",
        "--tb=short",
    ]


def test_extracts_identifiers_numbers_paths_and_commands():
    rule = (
        "Set AGENT_MEMORY_JUDGE_ALLOW_REMOTE=1, call LifecyclePlan.remember on "
        "127.0.0.1:9494 with deduplicate=True, edit ~/.factory/hooks.json and "
        "src/agent_memory/admission/ledger.py, pass --shadow (not -i) to `mem admit enable`, "
        "and keep Python 3.12+ lines under 100 chars (e.g. for MEM-9 on 2026-11-10)."
    )
    values = extract_values(rule)
    for expected in (
        "AGENT_MEMORY_JUDGE_ALLOW_REMOTE=1",
        "LifecyclePlan.remember",
        "127.0.0.1:9494",
        "deduplicate=True",
        "~/.factory/hooks.json",
        "src/agent_memory/admission/ledger.py",
        "--shadow",
        "-i",
        "mem admit enable",
        "3.12",
        "100",
        "MEM-9",
        "2026-11-10",
    ):
        assert expected in values
    assert "e.g" not in values
    assert "e.g." not in values


def test_avoids_false_positives_on_plain_english_and_slash_phrases():
    rule = "Read-only checks leave the 2nd calibration/held_out split and yes/no answers alone."
    values = extract_values(rule)
    assert values == ["held_out"]


def test_detects_missing_exact_values_and_checks_only_cited_lines():
    projection = (
        "Coding-agent turn.\n"
        "[1] REQUEST (human): Check port 9494 and src/agent_memory/config.py.\n"
        '[2] TOOL CALL Execute: {"command": "uv run pytest"}\n'
        "[3] ASSISTANT (final): Done with port 9495 and src/agent_memory/db.py.\n"
    )
    # Citing [1] only: 9495 and src/agent_memory/db.py only appear in uncited [3].
    result = check_exact_values(
        "Use port 9494 and port 9495 with src/agent_memory/db.py.",
        [1],
        projection,
    )
    assert result["ok"] is False
    assert result["missing"] == ["9495", "src/agent_memory/db.py"]
    assert result["redacted_cited"] is False


def test_entry_number_prefix_and_scaffold_tail_do_not_satisfy_numbers():
    projection = (
        "Coding-agent turn.\n"
        '[2] TOOL CALL Execute: {"command": "git status"}\n'
        "... 4 tool call(s) omitted ...\n"
        "[3] OUTCOME: completed\n"
    )
    entries = full_cited_entries(projection, [2])
    assert entries == [(2, 'TOOL CALL Execute: {"command": "git status"}')]
    result = check_exact_values("Expect exit code 2 and retry 4 times.", [2], projection)
    assert result["ok"] is False
    assert result["missing"] == ["2", "4"]


def test_reads_full_untruncated_cited_entry_past_excerpt_limit():
    padding = "x" * 500
    projection = f"Coding-agent turn.\n[1] TOOL RESULT (ok): {padding} port=9494 file=config.py\n"
    result = check_exact_values("Use port=9494 in config.py.", [1], projection)
    assert result["ok"] is True
    assert result["missing"] == []


def test_boundary_matching_rejects_partial_number_and_flag_matches():
    projection = (
        "Coding-agent turn.\n"
        '[1] TOOL CALL Execute: {"command": "github --shadow-mode on 2026-11-10"}\n'
    )
    result = check_exact_values("Run git with --shadow and exit code 2.", [1], projection)
    assert result["ok"] is False
    assert result["missing"] == ["git", "--shadow", "2"]


def test_matches_json_escaped_quotes_in_tool_call_lines():
    projection = (
        'Coding-agent turn.\n[1] TOOL CALL Execute: {"command": "git commit -m \\"fix: bug\\""}\n'
    )
    result = check_exact_values('Run `git commit -m "fix: bug"`.', [1], projection)
    assert result["ok"] is True
    assert result["missing"] == []


def test_redaction_marker_in_cited_lines_does_not_count_as_missing():
    projection = (
        "Coding-agent turn.\n"
        '[1] TOOL CALL Execute: {"command": "API_KEY=[REDACTED:generic_assignment] pytest"}\n'
        "[2] TOOL RESULT (ok): passed on port 9494\n"
    )
    result = check_exact_values(
        "Set API_KEY=placeholder_token before running pytest.",
        [1],
        projection,
    )
    assert result["redacted_cited"] is True
    assert result["missing"] == []
    assert result["ok"] is True


def test_redaction_marker_only_in_uncited_line_still_catches_missing_value():
    projection = (
        "Coding-agent turn.\n"
        '[1] TOOL CALL Execute: {"command": "API_KEY=[REDACTED:aws_secret]"}\n'
        "[2] TOOL RESULT (ok): server listens on port 9494\n"
    )
    result = check_exact_values("Server listens on port 8080.", [2], projection)
    assert result["redacted_cited"] is False
    assert result["missing"] == ["8080"]
    assert result["ok"] is False


def test_redaction_marker_in_rule_is_not_treated_as_exact_value():
    projection = "Coding-agent turn.\n[1] ASSISTANT (final): Done.\n"
    result = check_exact_values(
        "Use the synthetic credential [REDACTED:github_token].",
        [1],
        projection,
    )
    assert result["values"] == []
    assert result["missing"] == []
    assert result["ok"] is True


def test_check_draft_collects_citations_across_all_claims():
    projection = (
        "Coding-agent turn.\n"
        "[1] REQUEST (human): Check `mem serve`.\n"
        "[2] TOOL RESULT (ok): listening on 127.0.0.1:9494\n"
    )
    draft = {
        "rule": "Start `mem serve` on 127.0.0.1:9494.",
        "kind": "environment",
        "claims": [
            {"text": "Starts mem serve.", "cited": [1]},
            {"text": "Binds 127.0.0.1:9494.", "cited": [2]},
        ],
    }
    result = check_draft(draft, projection)
    assert result["ok"] is True
    assert result["values"] == ["mem serve", "127.0.0.1:9494"]


@pytest.mark.parametrize(
    ("rule", "cited_line", "ok"),
    [
        ("Migration 0007 registers admission.", "Applied 0007_admission.sql", True),
        ("Migration 0008 registers admission.", "Applied 0007_admission.sql", False),
        ("Use Python 3.12.", "Using Python 3.12.8", True),
        ("Use Python 3.13.", "Using Python 3.12.8", False),
        ("Keep .env out of the repo.", "Added /repo/.env to gitignore", True),
        ("Use sql_literal for SQL.", "Call my_sql_literal(x)", False),
    ],
)
def test_exact_value_matching_cases(rule, cited_line, ok):
    projection = f"Coding-agent turn.\n[1] ASSISTANT (final): {cited_line}\n"
    assert check_exact_values(rule, [1], projection)["ok"] is ok
