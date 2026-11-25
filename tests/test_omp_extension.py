"""`mem extension` installer and end-to-end omp TypeScript extension -> `mem records` -> ledger."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

from agent_memory.db import MemoryClient
from agent_memory.extension import EXTENSION_DIR_NAME, MANAGED_FILES, SOURCE_FILES
from agent_memory.smoke import free_port
from tests.test_cli import Mem

THIS_MEM = Path(sys.executable).parent / "mem"
BUN_BIN = shutil.which("bun")
OMP_BIN = shutil.which("omp")


@pytest.fixture
def mem(tmp_path, capsys, monkeypatch) -> Mem:
    for var in ("AGENT_MEMORY_HOME", "AGENT_MEMORY_HOST", "AGENT_MEMORY_PORT"):
        monkeypatch.delenv(var, raising=False)
    return Mem(home=tmp_path / "home", port=free_port(), capsys=capsys)


@pytest.fixture
def omp_extensions(tmp_path) -> Path:
    agent_dir = tmp_path / "omp" / "agent"
    agent_dir.mkdir(parents=True)
    ext_dir = agent_dir / "extensions"
    ext_dir.mkdir()
    return ext_dir


def _run_ts_file(
    tmp_path: Path, name: str, script: str, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    assert BUN_BIN is not None
    driver = tmp_path / name
    driver.write_text(script, encoding="utf-8")
    run_env = {**os.environ, **(env or {})}
    return subprocess.run(
        [BUN_BIN, str(driver)],
        capture_output=True,
        text=True,
        env=run_env,
        timeout=30,
        check=False,
    )


def _wait_for_rows(cfg, query: str, predicate, timeout: float = 10.0):
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        with MemoryClient.from_config(cfg) as client:
            _, rows = client.query(query)
        last = rows
        if predicate(rows):
            return rows
        time.sleep(0.15)
    raise AssertionError(f"timed out waiting for {query!r}; last={last!r}")


# --- `mem extension` install / status / uninstall ---


def test_install_status_and_uninstall_lifecycle(mem, omp_extensions):
    target = omp_extensions / EXTENSION_DIR_NAME
    code, out, _ = mem("extension", "status", "--dir", str(omp_extensions))
    assert code == 1
    assert "not installed" in out

    code, out, err = mem("extension", "install", "--dir", str(omp_extensions))
    assert code == 0, err
    assert target.is_dir()
    assert sorted(p.name for p in target.iterdir()) == sorted(MANAGED_FILES)
    assert (target / "config.json").stat().st_mode & 0o777 == 0o600
    assert "installed omp capture extension" in out
    assert "mem init" in err  # ledger home is not initialized yet

    cfg = json.loads((target / "config.json").read_text())
    assert cfg["managed_by"] == "agent-memory"
    assert cfg["mem"] == str(THIS_MEM.absolute())
    assert cfg["home"] == str(mem.home.absolute())
    assert cfg["port"] == mem.port

    code, out, err = mem("extension", "status", "--dir", str(omp_extensions))
    assert code == 0, err
    for name in MANAGED_FILES:
        assert name in out
    assert "installed" in out

    # Re-running install changes nothing and preserves file inodes.
    inodes_before = {name: (target / name).stat().st_ino for name in MANAGED_FILES}
    code, out, _ = mem("extension", "install", "--dir", str(omp_extensions))
    assert code == 0
    assert "nothing to change" in out
    assert {name: (target / name).stat().st_ino for name in MANAGED_FILES} == inodes_before

    # Changing port causes status to report differs, and install updates config.json only.
    moved = Mem(home=mem.home, port=mem.port + 1, capsys=mem.capsys)
    code, out, _ = moved("extension", "status", "--dir", str(omp_extensions))
    assert code == 1
    assert "differs" in out
    code, out, _ = moved("extension", "install", "--dir", str(omp_extensions))
    assert code == 0
    assert "updated" in out
    assert json.loads((target / "config.json").read_text())["port"] == moved.port
    assert moved("extension", "status", "--dir", str(omp_extensions))[0] == 0

    # Uninstall removes agent-memory/ and leaves omp_extensions itself intact.
    code, out, err = moved("extension", "uninstall", "--dir", str(omp_extensions))
    assert code == 0, err
    assert not target.exists()
    assert omp_extensions.is_dir()
    assert "removed" in out

    # Re-running uninstall is a clean no-op.
    code, out, _ = moved("extension", "uninstall", "--dir", str(omp_extensions))
    assert code == 0
    assert "not installed" in out


def test_install_and_uninstall_leave_other_extensions_untouched(mem, omp_extensions):
    sibling_file = omp_extensions / "my-guard.ts"
    sibling_file.write_text("export default function() {}\n")
    sibling_dir = omp_extensions / "custom-ext"
    sibling_dir.mkdir()
    (sibling_dir / "index.ts").write_text("// user extension\n")

    code, out, _ = mem("extension", "install", "--dir", str(omp_extensions))
    assert code == 0
    assert "2 other extensions were left alone" in out
    assert sibling_file.read_text() == "export default function() {}\n"
    assert (sibling_dir / "index.ts").read_text() == "// user extension\n"

    code, out, _ = mem("extension", "uninstall", "--dir", str(omp_extensions))
    assert code == 0
    assert "2 other extensions were left alone" in out
    assert sibling_file.read_text() == "export default function() {}\n"
    assert (sibling_dir / "index.ts").read_text() == "// user extension\n"


def test_install_creates_missing_extensions_dir_when_parent_exists_and_refuses_when_not(
    mem, tmp_path
):
    omp_agent = tmp_path / "omp" / "agent"
    omp_agent.mkdir(parents=True)
    ext_dir = omp_agent / "extensions"
    assert not ext_dir.exists()

    code, _, err = mem("extension", "install", "--dir", str(ext_dir))
    assert code == 0, err
    assert (ext_dir / EXTENSION_DIR_NAME / "index.ts").is_file()

    missing_parent = tmp_path / "does-not-exist" / "extensions"
    code, _, err = mem("extension", "install", "--dir", str(missing_parent))
    assert code == 1
    assert "does not exist" in err


def test_refuses_foreign_directory_symlink_or_extra_user_files(mem, omp_extensions):
    target = omp_extensions / EXTENSION_DIR_NAME

    # 1. Foreign directory without managed config.json
    target.mkdir()
    (target / "index.ts").write_text("// user's own extension\n")
    assert mem("extension", "install", "--dir", str(omp_extensions))[0] == 1
    assert mem("extension", "status", "--dir", str(omp_extensions))[0] == 1
    assert mem("extension", "uninstall", "--dir", str(omp_extensions))[0] == 1
    assert (target / "index.ts").read_text() == "// user's own extension\n"
    shutil.rmtree(target)

    # 2. Regular file or symlink in place of agent-memory/
    target.write_text("not a dir\n")
    assert mem("extension", "install", "--dir", str(omp_extensions))[0] == 1
    assert mem("extension", "uninstall", "--dir", str(omp_extensions))[0] == 1
    target.unlink()

    # 3. Managed install with an unexpected user file added inside it
    assert mem("extension", "install", "--dir", str(omp_extensions))[0] == 0
    user_note = target / "notes.txt"
    user_note.write_text("keep me\n")
    assert mem("extension", "install", "--dir", str(omp_extensions))[0] == 1
    assert mem("extension", "status", "--dir", str(omp_extensions))[0] == 1
    assert mem("extension", "uninstall", "--dir", str(omp_extensions))[0] == 1
    assert user_note.read_text() == "keep me\n"


def test_status_detects_out_of_date_source_file(mem, omp_extensions):
    assert mem("extension", "install", "--dir", str(omp_extensions))[0] == 0
    mapper = omp_extensions / EXTENSION_DIR_NAME / "mapper.ts"
    mapper.write_text(mapper.read_text() + "\n// local edit\n")

    code, out, _ = mem("extension", "status", "--dir", str(omp_extensions))
    assert code == 1
    assert "out of date" in out


def test_install_warnings_and_mem_validation(mem, omp_extensions, down_cfg, monkeypatch):
    bad_mem = omp_extensions / "not-executable"
    bad_mem.write_text("echo nope\n")
    code, _, err = mem("extension", "install", "--dir", str(omp_extensions), "--mem", str(bad_mem))
    assert code == 1
    assert "cannot" in err or "not a command" in err

    initialized = Mem(home=down_cfg.home, port=down_cfg.port, capsys=mem.capsys)
    code, _, err = initialized("extension", "install", "--dir", str(omp_extensions))
    assert code == 0
    assert err == ""

    monkeypatch.setenv("AGENT_MEMORY_HOST", "127.0.0.1")
    code, _, err = initialized("extension", "install", "--dir", str(omp_extensions))
    assert code == 0
    assert "AGENT_MEMORY_HOST" in err


# --- End-to-end: installed TypeScript extension -> `mem records` -> real DuckDB ledger ---


@pytest.mark.skipif(BUN_BIN is None, reason="bun is required to run the omp extension")
def test_extension_to_ledger_full_session_and_replay_idempotency(
    live_cfg, tmp_path, omp_extensions, capsys
):
    mem_cli = Mem(home=live_cfg.home, port=live_cfg.port, capsys=capsys)
    code, _, err = mem_cli(
        "extension", "install", "--dir", str(omp_extensions), "--mem", str(THIS_MEM)
    )
    assert code == 0, err
    installed_index = omp_extensions / EXTENSION_DIR_NAME / "index.ts"

    driver = f"""
import agentMemoryExtension from {json.dumps(str(installed_index))};

class FakeSM {{
  sessionId = "01a210a3-b133-705e-b720-09908b69e115";
  sessionFile = "/tmp/omp/s1.jsonl";
  entries = [
    {{ type: "model_change", id: "11111111", timestamp: "2026-11-25T03:00:00.000Z" }},
    {{ type: "thinking_level_change", id: "22222222", timestamp: "2026-11-25T03:00:00.001Z" }},
  ];
  getSessionId() {{ return this.sessionId; }}
  getSessionFile() {{ return this.sessionFile; }}
  getEntries() {{ return this.entries; }}
  getLeafId() {{ return this.entries[this.entries.length - 1]?.id ?? null; }}
  getHeader() {{ return {{ id: this.sessionId, timestamp: "2026-11-25T03:00:00.000Z" }}; }}
}}

const handlers = new Map();
const pi = {{
  pi: {{ VERSION: "19.1.1" }},
  on(event, fn) {{
    const list = handlers.get(event) ?? [];
    list.push(fn);
    handlers.set(event, list);
  }},
}};
agentMemoryExtension(pi);

const sm = new FakeSM();
const ctx = {{
  cwd: "/tmp/demo-repo",
  mode: "print",
  hasUI: false,
  agent: {{ kind: "main", id: "Main", name: "main", depth: 0 }},
  model: {{ provider: "ollama", id: "gemma4:26b" }},
  sessionManager: sm,
}};
const emit = (ev, payload) => {{
  for (const fn of handlers.get(ev) ?? []) fn(payload, ctx);
}};

emit("session_start", {{ type: "session_start" }});

const userMsg = {{
  role: "user",
  content: [{{ type: "text", text: "Read note.txt and echo hello" }}],
  attribution: "user",
  timestamp: 1795575570874,
}};
sm.entries.push({{ type: "message", id: "5f67c9db", message: userMsg }});
emit("message_end", {{ type: "message_end", message: userMsg }});

emit("tool_call", {{
  type: "tool_call",
  toolName: "read",
  toolCallId: "call_read_1",
  input: {{ path: "note.txt" }},
}});
const step1 = {{
  role: "assistant",
  content: [
    {{ type: "toolCall", id: "call_read_1", name: "read", arguments: {{ path: "note.txt" }} }},
  ],
  stopReason: "toolUse",
  model: "gemma4:26b",
  provider: "ollama",
  responseId: "resp_1",
  timestamp: 1795575570898,
}};
sm.entries.push({{ type: "message", id: "6fb5082f", message: step1 }});
const tr1 = {{
  role: "toolResult",
  toolCallId: "call_read_1",
  toolName: "read",
  content: [{{ type: "text", text: "hello from note" }}],
  isError: false,
  timestamp: 1795575573108,
}};
sm.entries.push({{ type: "message", id: "97886fde", message: tr1 }});
emit("message_end", {{ type: "message_end", message: tr1 }});
emit("turn_end", {{ type: "turn_end", turnIndex: 0, message: step1 }});

emit("tool_call", {{
  type: "tool_call",
  toolName: "mcp__rec_echo_back",
  toolCallId: "call_mcp_2",
  input: {{ text: "hello" }},
}});
const tr2 = {{
  role: "toolResult",
  toolCallId: "call_mcp_2",
  toolName: "mcp__rec_echo_back",
  content: [{{ type: "text", text: "echo: hello" }}],
  details: {{ serverName: "rec", mcpToolName: "echo_back" }},
  isError: false,
  timestamp: 1795575573500,
}};
sm.entries.push({{ type: "message", id: "bd326b50", message: tr2 }});
emit("message_end", {{ type: "message_end", message: tr2 }});

const finalReply = {{
  role: "assistant",
  content: [{{ type: "text", text: "done" }}],
  stopReason: "stop",
  model: "gemma4:26b",
  provider: "ollama",
  responseId: "resp_2",
  timestamp: 1795575573903,
  completedAt: 1795575574144,
}};
sm.entries.push({{ type: "message", id: "21a15f54", message: finalReply }});
emit("turn_end", {{ type: "turn_end", turnIndex: 1, message: finalReply }});
emit("session_stop", {{
  type: "session_stop",
  session_id: sm.sessionId,
  turn_id: 1,
  last_assistant_message: finalReply,
}});
emit("agent_end", {{ type: "agent_end" }});

sm.entries.push({{ type: "compaction", id: "7512dd62", timestamp: "2026-11-25T03:03:10.654Z" }});
emit("session_compact", {{
  type: "session_compact",
  fromExtension: false,
  compactionEntry: {{
    id: "7512dd62",
    parentId: "21a15f54",
    timestamp: "2026-11-25T03:03:10.654Z",
    summary: "Compacted turn 1",
    tokensBefore: 2000,
    tokensAfter: 800,
    method: "soft",
  }},
}});

sm.entries.push({{ type: "custom", id: "4a40a89a", customType: "session_exit" }});
emit("session_shutdown", {{ type: "session_shutdown" }});

// Wait briefly past the 50ms shutdown close grace window so stdin closes cleanly.
await new Promise((r) => setTimeout(r, 120));
"""

    first = _run_ts_file(tmp_path, "run_full.ts", driver)
    assert first.returncode == 0, first.stderr

    expected_kinds = {
        "session_start",
        "user_msg",
        "tool_call",
        "tool_result",
        "assistant_msg",
        "outcome",
        "compaction",
        "session_end",
    }
    sid = "01a210a3-b133-705e-b720-09908b69e115"
    rows = _wait_for_rows(
        live_cfg,
        "SELECT e.kind, e.content, e.parent_event_id FROM events e "
        "JOIN sessions s USING (session_id) "
        f"WHERE s.harness_session_id = '{sid}' ORDER BY e.ts, e.kind",
        lambda r: {row[0] for row in r} >= expected_kinds and len(r) == 10,
    )
    assert len(rows) == 10

    with MemoryClient.from_config(live_cfg) as client:
        _, sess = client.query(
            "SELECT agent_id, harness_session_id, model, project, harness_version, "
            "answers, compactions, capture_errors, ended_at IS NOT NULL, calling_session_id "
            f"FROM session_summary WHERE harness_session_id = '{sid}'"
        )
        assert sess == [
            (
                "oh-my-pi",
                sid,
                "ollama/gemma4:26b",
                "demo-repo",
                "19.1.1",
                1,
                1,
                0,
                True,
                None,
            )
        ]
        # Both tool_result rows must link to their tool_call parent_event_id.
        _, linked = client.query(
            "SELECT c.kind, r.kind FROM events r "
            "JOIN events c ON r.parent_event_id = c.event_id "
            "JOIN sessions s ON r.session_id = s.session_id "
            f"WHERE s.harness_session_id = '{sid}' AND r.kind = 'tool_result'"
        )
        assert linked == [("tool_call", "tool_result"), ("tool_call", "tool_result")]

    # Replay the exact same session a second time: adds no session and no event.
    second = _run_ts_file(tmp_path, "run_replay.ts", driver)
    assert second.returncode == 0, second.stderr
    time.sleep(0.8)  # allow mem records' 0.5s pass linger + write to complete

    with MemoryClient.from_config(live_cfg) as client:
        _, counts = client.query(
            f"SELECT (SELECT count(*) FROM sessions WHERE harness_session_id = '{sid}'), "
            "  (SELECT count(*) FROM events e JOIN sessions s USING (session_id) "
            f"   WHERE s.harness_session_id = '{sid}')"
        )
        assert counts == [(1, 10)]


@pytest.mark.skipif(BUN_BIN is None, reason="bun is required to run the omp extension")
def test_extension_to_ledger_session_transitions_and_subagents(
    live_cfg, tmp_path, omp_extensions, capsys
):
    mem_cli = Mem(home=live_cfg.home, port=live_cfg.port, capsys=capsys)
    assert (
        mem_cli("extension", "install", "--dir", str(omp_extensions), "--mem", str(THIS_MEM))[0]
        == 0
    )
    installed_index = omp_extensions / EXTENSION_DIR_NAME / "index.ts"

    driver = f"""
import agentMemoryExtension from {json.dumps(str(installed_index))};

function makePi() {{
  const handlers = new Map();
  const pi = {{
    pi: {{ VERSION: "19.1.1" }},
    on(event, fn) {{
      const list = handlers.get(event) ?? [];
      list.push(fn);
      handlers.set(event, list);
    }},
  }};
  agentMemoryExtension(pi);
  return (ev, payload, ctx) => {{
    for (const fn of handlers.get(ev) ?? []) fn(payload, ctx);
  }};
}}

function makeSM(id, file, entries = []) {{
  return {{
    sessionId: id,
    sessionFile: file,
    entries: [...entries],
    getSessionId() {{ return this.sessionId; }},
    getSessionFile() {{ return this.sessionFile; }},
    getEntries() {{ return this.entries; }},
    getLeafId() {{ return this.entries[this.entries.length - 1]?.id ?? null; }},
  }};
}}

const mainEmit = makePi();
const mainSM = makeSM("sid-main", "/tmp/main.jsonl", [{{ type: "model_change", id: "a0000001" }}]);
const mainCtx = {{
  cwd: "/tmp/proj",
  mode: "rpc",
  agent: {{ kind: "main", id: "Main", name: "main", depth: 0 }},
  model: {{ provider: "ollama", id: "gemma4:26b" }},
  sessionManager: mainSM,
}};

mainEmit("session_start", {{}}, mainCtx);
const uMain = {{ role: "user", content: "delegate", timestamp: 1795575818479 }};
mainSM.entries.push({{ type: "message", id: "91b05380", message: uMain }});
mainEmit("message_end", {{ message: uMain }}, mainCtx);

// Spawn depth-1 subagent
mainEmit(
  "before_subagent_spawn",
  {{ agent: "task", invocationKind: "task", spawnKey: "Worker" }},
  mainCtx,
);
const subEmit = makePi();
const subSM = makeSM("sid-sub1", "/tmp/main/Worker.jsonl", [
  {{ type: "model_change", id: "b0000001" }},
]);
const subCtx = {{
  cwd: "/tmp/proj",
  mode: "rpc",
  agent: {{ kind: "sub", id: "Worker", name: "task", depth: 1, parentId: "Main" }},
  model: {{ provider: "ollama", id: "gemma4:26b" }},
  sessionManager: subSM,
}};
subEmit("session_start", {{}}, subCtx);
const uSub = {{
  role: "user",
  content: "sub work",
  attribution: "agent",
  timestamp: 1795575820892,
}};
subSM.entries.push({{ type: "message", id: "fa590a4b", message: uSub }});
subEmit("message_end", {{ message: uSub }}, subCtx);
const aSub = {{
  role: "assistant",
  content: [{{ type: "text", text: "sub ok" }}],
  stopReason: "stop",
  timestamp: 1795575825000,
}};
subSM.entries.push({{ type: "message", id: "d12479b8", message: aSub }});
subEmit("turn_end", {{ turnIndex: 0, message: aSub }}, subCtx);
subEmit("agent_end", {{}}, subCtx);
subEmit("session_shutdown", {{}}, subCtx);

// Main finishes its prompt
const aMain = {{
  role: "assistant",
  content: [{{ type: "text", text: "main ok" }}],
  stopReason: "stop",
  timestamp: 1795575830000,
}};
mainSM.entries.push({{ type: "message", id: "78fb933b", message: aMain }});
mainEmit("turn_end", {{ turnIndex: 0, message: aMain }}, mainCtx);
mainEmit(
  "session_stop",
  {{ session_id: "sid-main", turn_id: 0, last_assistant_message: aMain }},
  mainCtx,
);

// Fork, branch, new, clear, and resume back to sid-main
mainEmit("session_before_branch", {{ reason: "fork", entryId: "78fb933b" }}, mainCtx);
mainSM.sessionId = "sid-fork";
mainSM.sessionFile = "/tmp/fork.jsonl";
mainEmit("session_branch", {{ reason: "fork", previousSessionFile: "/tmp/main.jsonl" }}, mainCtx);

mainEmit("session_before_branch", {{ reason: "branch", entryId: "91b05380" }}, mainCtx);
mainSM.sessionId = "sid-branch";
mainSM.sessionFile = "/tmp/branch.jsonl";
mainEmit("session_branch", {{ reason: "branch", previousSessionFile: "/tmp/fork.jsonl" }}, mainCtx);

mainEmit("session_before_switch", {{ reason: "new" }}, mainCtx);
mainSM.sessionId = "sid-new";
mainSM.sessionFile = "/tmp/new.jsonl";
mainEmit("session_switch", {{ reason: "new", previousSessionFile: "/tmp/branch.jsonl" }}, mainCtx);

mainSM.sessionId = "sid-clear";
mainSM.sessionFile = "/tmp/clear.jsonl";
mainEmit("session_clear", {{}}, mainCtx);

mainEmit(
  "session_before_switch",
  {{ reason: "resume", targetSessionFile: "/tmp/main.jsonl" }},
  mainCtx,
);
mainSM.sessionId = "sid-main";
mainSM.sessionFile = "/tmp/main.jsonl";
mainEmit(
  "session_switch",
  {{ reason: "resume", previousSessionFile: "/tmp/clear.jsonl" }},
  mainCtx,
);

mainEmit("session_shutdown", {{}}, mainCtx);
await new Promise((r) => setTimeout(r, 120));
"""

    res = _run_ts_file(tmp_path, "run_transitions.ts", driver)
    assert res.returncode == 0, res.stderr

    # Expect 6 distinct sessions: sid-main, sid-sub1, sid-fork, sid-branch, sid-new, sid-clear
    # (resume returned to sid-main without creating a 7th session).
    rows = _wait_for_rows(
        live_cfg,
        "SELECT harness_session_id, calling_session_id FROM session_summary "
        "WHERE harness_session_id LIKE 'sid-%' ORDER BY harness_session_id",
        lambda r: len(r) == 6,
    )
    assert rows == [
        ("sid-branch", None),
        ("sid-clear", None),
        ("sid-fork", None),
        ("sid-main", None),
        ("sid-new", None),
        ("sid-sub1", "sid-main"),
    ]

    with MemoryClient.from_config(live_cfg) as client:
        # Subagent events carry payload.subagent so they can be distinguished at the event level.
        _, sub_events = client.query(
            "SELECT e.kind, e.payload.subagent.id::VARCHAR, e.payload.subagent.depth::INTEGER "
            "FROM events e JOIN sessions s USING (session_id) "
            "WHERE s.harness_session_id = 'sid-sub1' ORDER BY e.ts, e.kind"
        )
        assert len(sub_events) == 5  # session_start, user_msg, assistant_msg, outcome, session_end
        assert all(row[1] == "Worker" and row[2] == 1 for row in sub_events)


@pytest.mark.skipif(OMP_BIN is None, reason="omp is not installed in this environment")
def test_real_omp_discovers_installed_extension_without_model_call(live_cfg, tmp_path, capsys):
    """Real omp discovers <cwd>/.omp/extensions/agent-memory/index.ts and records into DuckDB."""
    work = tmp_path / "omp-real-smoke"
    ext_dir = work / ".omp" / "extensions"
    ext_dir.mkdir(parents=True)
    fake_home = tmp_path / "omp-home"
    fake_home.mkdir()
    real_duckdb = Path.home() / ".duckdb"
    if real_duckdb.is_dir():
        (fake_home / ".duckdb").symlink_to(real_duckdb)

    mem_cli = Mem(home=live_cfg.home, port=live_cfg.port, capsys=capsys)
    code, _, err = mem_cli("extension", "install", "--dir", str(ext_dir), "--mem", str(THIS_MEM))
    assert code == 0, err

    # Run real omp in RPC mode with empty stdin: starts session, loads discovered extension,
    # emits session_start + session_shutdown, and exits without making any model call.
    assert OMP_BIN is not None
    env = {
        **os.environ,
        "HOME": str(fake_home),
        "XDG_CONFIG_HOME": str(fake_home / ".config"),
    }
    proc = subprocess.run(
        [OMP_BIN, "--mode", "rpc", "--no-ui", "--no-title"],
        input="",
        capture_output=True,
        text=True,
        cwd=work,
        env=env,
        timeout=20,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr

    rows = _wait_for_rows(
        live_cfg,
        "SELECT s.agent_id, s.harness_version, e.kind "
        "FROM events e JOIN sessions s USING (session_id) "
        "WHERE s.project = 'omp-real-smoke' ORDER BY e.ts, e.kind",
        lambda r: {row[2] for row in r} >= {"session_start", "session_end"},
    )
    assert all(row[0] == "oh-my-pi" for row in rows)
    assert any(row[1] for row in rows)  # omp version from pi.pi.VERSION
    assert {row[2] for row in rows} == {"session_start", "session_end"}
    assert all(f in SOURCE_FILES or f == "config.json" for f in MANAGED_FILES)
