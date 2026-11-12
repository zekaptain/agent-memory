"""`mem review`: what waits for a human, and the human's approval or retirement of it."""

from __future__ import annotations

import io
from uuid import uuid4

import pytest

from agent_memory.capture.spool import Spool
from agent_memory.db import sql_literal
from agent_memory.mcp_server.tools import MemoryTools
from tests.test_cli import Mem


@pytest.fixture
def mem(live_cfg, capsys, monkeypatch) -> Mem:
    for var in ("AGENT_MEMORY_HOME", "AGENT_MEMORY_HOST", "AGENT_MEMORY_PORT"):
        monkeypatch.delenv(var, raising=False)
    return Mem(home=live_cfg.home, port=live_cfg.port, capsys=capsys)


@pytest.fixture
def scope() -> str:
    """The module shares one ledger, so each test reads and writes a scope of its own."""
    return f"review-{uuid4()}"


@pytest.fixture
def agent(live_cfg, scope):
    tools = MemoryTools(live_cfg, "factory-droid", project=scope)
    yield tools
    tools.close()


class Terminal(io.StringIO):
    """Standard input while a human types at a terminal."""

    def isatty(self) -> bool:
        return True


def typed(monkeypatch, text: str) -> None:
    monkeypatch.setattr("sys.stdin", Terminal(text))


def piped(monkeypatch, text: str) -> None:
    monkeypatch.setattr("sys.stdin", io.StringIO(text))


def from_tool(agent, content: str = "pytest testpaths is tests/") -> str:
    """A memory an agent wrote from tool output, which starts quarantined."""
    return agent.remember(content, "fact", trust_level="tool")["memory_id"]


def status(client, memory_id: str) -> str:
    return client.query(
        f"SELECT status FROM memory_status WHERE memory_id = {sql_literal(memory_id)}"
    )[1][0][0]


def reviews(client, memory_id: str, action: str) -> list[tuple]:
    """Who recorded each review action of one kind on a memory, in which agent's session,
    how that session reached the ledger, and why."""
    return client.query(
        "SELECT a.actor, s.agent_id, json_extract_string(s.metadata::JSON, '$.via'), a.reason "
        "FROM memory_actions a JOIN sessions s USING (session_id) "
        f"WHERE a.memory_id = {sql_literal(memory_id)} AND a.action = {sql_literal(action)}"
    )[1]


def blocks(out: str) -> dict[str, str]:
    """What `mem review list` printed about each memory, by the id that heads its block."""
    return {block.splitlines()[0]: block for block in out.strip().split("\n\n")[:-1]}


def test_review_is_a_mem_command(mem):
    code, out, _ = mem("review", "--help")
    assert code == 0
    for command in ("list", "approve", "retire"):
        assert command in out
    assert "review" in mem("--help")[1]


def test_list_shows_each_waiting_memory_and_why(mem, agent, scope):
    unapproved = from_tool(agent)
    doubted = agent.remember("Deploys happen on Fridays", "fact")["memory_id"]
    agent.review("quarantine", doubted, "contradicts the runbook")
    agent.remember("uv is the package manager", "fact")
    old = agent.remember("The API listens on port 8000", "fact")["memory_id"]
    correction = agent.remember(
        "The API listens on port 9000", "fact", supersedes=old, trust_level="tool"
    )["memory_id"]

    code, out, _ = mem("review", "list", "--scope", scope)
    assert code == 0
    listed = blocks(out)
    assert list(listed) == [unapproved, doubted, correction]
    assert "pytest testpaths is tests/" in listed[unapproved]
    assert "written by factory-droid" in listed[unapproved]
    assert "never approved" in listed[unapproved]
    assert "factory-droid quarantined it: contradicts the runbook" in listed[doubted]
    assert f"approving it replaces {old}" in listed[correction]
    assert "The API listens on port 8000" in listed[correction]
    assert "uv is the package manager" not in out
    assert out.strip().endswith("3 memories await approval")


def test_list_says_when_nothing_waits(mem, agent, scope):
    agent.remember("uv is the package manager", "fact")
    assert mem("review", "list", "--scope", scope) == (0, "nothing awaits approval\n", "")


def test_a_human_at_a_terminal_approves(mem, agent, scope, live_client, monkeypatch):
    waiting = from_tool(agent)
    typed(monkeypatch, "y\n")
    code, out, _ = mem("review", "approve", waiting, "--reason", "checked pyproject.toml")
    assert code == 0
    # The human is shown what they are asked about before they answer.
    assert out.index("pytest testpaths is tests/") < out.index("Approve this memory?")
    assert f"approved {waiting}: now active" in out
    assert reviews(live_client, waiting, "approve") == [
        ("human", "human", "mem review", "checked pyproject.toml")
    ]
    recalled = agent.recall("pytest testpaths", scopes=[scope])["items"]
    assert [item["memory_id"] for item in recalled] == [waiting]
    assert mem("review", "list", "--scope", scope)[1] == "nothing awaits approval\n"
    assert mem("audit", "check")[0] == 0


def test_approving_an_active_memory_accepts_the_correction_a_tool_asked_for(
    mem, agent, scope, live_client, monkeypatch
):
    old = agent.remember("original", "fact")["memory_id"]
    canonical = agent.remember("canonical", "fact")["memory_id"]
    asked = agent.remember("canonical", "fact", supersedes=old, trust_level="tool")
    assert asked["memory_id"] == canonical
    listed = blocks(mem("review", "list", "--scope", scope)[1])
    assert list(listed) == [canonical]
    assert f"approving it replaces {old}" in listed[canonical]

    typed(monkeypatch, "yes\n")
    assert mem("review", "approve", canonical, "--reason", "the correction is right")[0] == 0
    assert [status(live_client, m) for m in (old, canonical)] == ["superseded", "active"]
    assert mem("review", "list", "--scope", scope)[1] == "nothing awaits approval\n"


def test_approve_refuses_an_answer_that_was_not_typed_at_a_terminal(
    mem, agent, live_client, monkeypatch
):
    waiting = from_tool(agent)
    piped(monkeypatch, "y\n")
    code, out, err = mem("review", "approve", waiting, "--reason", "checked")
    assert (code, out) == (1, "")
    assert "terminal" in err
    assert status(live_client, waiting) == "quarantined"
    assert reviews(live_client, waiting, "approve") == []


@pytest.mark.parametrize("answer", ["n\n", "\n", "", "yep\n"])
def test_anything_but_yes_leaves_the_memory_waiting(mem, agent, live_client, monkeypatch, answer):
    waiting = from_tool(agent)
    typed(monkeypatch, answer)
    code, _, err = mem("review", "approve", waiting, "--reason", "checked")
    assert code == 1
    assert "not approved" in err
    assert status(live_client, waiting) == "quarantined"
    assert reviews(live_client, waiting, "approve") == []


def test_approve_takes_only_a_memory_that_waits(mem, agent, live_client, monkeypatch):
    active = agent.remember("uv is the package manager", "fact")["memory_id"]
    retired = from_tool(agent)
    agent.review("retire", retired, "wrong")
    for memory_id, message in [
        (active, f"memory {active} is active"),
        (retired, f"memory {retired} is retired"),
        (str(uuid4()), "no memory"),
    ]:
        typed(monkeypatch, "y\n")
        code, _, err = mem("review", "approve", memory_id, "--reason", "checked")
        assert code == 1
        assert message in err
    assert reviews(live_client, active, "approve") == []
    assert reviews(live_client, retired, "approve") == []
    typed(monkeypatch, "y\n")
    assert mem("review", "approve", "the-first-one", "--reason", "checked")[0] == 2


def test_retire_withdraws_a_memory_without_a_terminal(mem, agent, scope, live_client, monkeypatch):
    waiting = from_tool(agent)
    piped(monkeypatch, "")
    code, out, _ = mem("review", "retire", waiting, "--reason", "the test paths moved")
    assert code == 0
    assert f"retired {waiting}: now retired" in out
    assert reviews(live_client, waiting, "retire") == [
        ("human", "human", "mem review", "the test paths moved")
    ]
    assert mem("review", "list", "--scope", scope)[1] == "nothing awaits approval\n"


def test_a_reason_is_required(mem, agent, live_client, monkeypatch):
    waiting = from_tool(agent)
    for command in ("approve", "retire"):
        for reason in ([], ["--reason", " "]):
            typed(monkeypatch, "y\n")
            code, _, err = mem("review", command, waiting, *reason)
            assert code == 2
            assert "--reason" in err
    assert status(live_client, waiting) == "quarantined"


def test_review_needs_the_server_to_show_what_waits(down_cfg, capsys, monkeypatch):
    mem = Mem(home=down_cfg.home, port=down_cfg.port, capsys=capsys)
    code, _, err = mem("review", "list")
    assert code == 1
    assert "Is `mem serve` running?" in err
    # An approval is of a memory the human was shown, so none is queued unseen.
    typed(monkeypatch, "y\n")
    code, _, err = mem("review", "approve", str(uuid4()), "--reason", "checked")
    assert code == 1
    assert "Is `mem serve` running?" in err
    assert Spool(down_cfg.spool_dir).status() == []
