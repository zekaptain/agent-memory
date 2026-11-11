"""`mem hooks`: mem's capture command in Droid's hooks file, and nothing else in it changed."""

from __future__ import annotations

import json
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

from agent_memory.capture.spool import Spool
from agent_memory.smoke import free_port
from tests.test_cli import Mem

# The hook events at which Droid's adapter records something or reads the transcript.
EVENTS = [
    "SessionStart",
    "SessionEnd",
    "UserPromptSubmit",
    "PreCompact",
    "SubagentStop",
    "Notification",
    "Stop",
]
THIS_MEM = Path(sys.executable).parent / "mem"


@pytest.fixture
def mem(tmp_path, capsys, monkeypatch) -> Mem:
    for var in ("AGENT_MEMORY_HOME", "AGENT_MEMORY_HOST", "AGENT_MEMORY_PORT"):
        monkeypatch.delenv(var, raising=False)
    return Mem(home=tmp_path / "home", port=free_port(), capsys=capsys)


@pytest.fixture
def hooks_file(tmp_path) -> Path:
    factory = tmp_path / "factory"
    factory.mkdir()
    return factory / "hooks.json"


def theirs() -> dict:
    """Hooks someone else wrote, in the shapes Droid's file takes."""
    return {
        "SessionStart": [
            {"hooks": [{"type": "command", "command": "/usr/local/bin/probe.py", "timeout": 5}]}
        ],
        "PreToolUse": [
            {
                "matcher": "Execute",
                "commandRegex": "^git ",
                "hooks": [{"type": "command", "command": "/usr/local/bin/audit.sh"}],
            }
        ],
        "UserPromptSubmit": [],
    }


def hook(command: str) -> dict:
    return {"hooks": [{"type": "command", "command": command}]}


def capture(mem: Mem, *flags: str) -> str:
    """The command `mem hooks install` writes for this home and port."""
    words = [str(THIS_MEM), "--home", str(mem.home), "--port", str(mem.port)]
    return shlex.join([*words, "event", "--harness", "droid", *flags])


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def test_install_writes_one_command_at_each_hook_event_capture_needs(mem, hooks_file):
    code, out, err = mem("hooks", "install", "--file", str(hooks_file))
    assert code == 0, err
    assert read(hooks_file) == {event: [hook(capture(mem))] for event in EVENTS}
    assert list(read(hooks_file)) == EVENTS
    assert hooks_file.stat().st_mode & 0o777 == 0o600
    assert "7 hooks" in out
    assert capture(mem) in out
    # A hook runs with Droid's environment, so the command names everything itself.
    assert THIS_MEM.is_absolute()
    assert mem.home.is_absolute()


def test_install_says_so_when_the_home_holds_no_ledger(mem, hooks_file, down_cfg):
    code, _, err = mem("hooks", "install", "--file", str(hooks_file))
    assert code == 0
    assert "mem init" in err
    initialized = Mem(home=down_cfg.home, port=down_cfg.port, capsys=mem.capsys)
    assert initialized("hooks", "install", "--file", str(hooks_file))[2] == ""


def test_install_says_which_setting_a_hook_will_not_see(mem, hooks_file, monkeypatch):
    # The command carries the home and the port. The host has no option to carry it.
    monkeypatch.setenv("AGENT_MEMORY_HOST", "127.0.0.1")
    code, _, err = mem("hooks", "install", "--file", str(hooks_file))
    assert code == 0
    assert "AGENT_MEMORY_HOST" in err


def test_install_keeps_every_hook_that_is_not_mems(mem, hooks_file):
    hooks_file.write_text(json.dumps(theirs()))
    hooks_file.chmod(0o640)
    code, out, _ = mem("hooks", "install", "--file", str(hooks_file))
    assert code == 0
    found = read(hooks_file)
    assert list(found)[:3] == list(theirs())
    assert found["PreToolUse"] == theirs()["PreToolUse"]
    assert found["SessionStart"] == [*theirs()["SessionStart"], hook(capture(mem))]
    assert found["UserPromptSubmit"] == [hook(capture(mem))]
    assert "2 other hooks" in out
    assert hooks_file.stat().st_mode & 0o777 == 0o640


def test_install_again_changes_nothing_and_replaces_what_differs(mem, hooks_file):
    hooks_file.write_text(json.dumps(theirs()))
    assert mem("hooks", "install", "--file", str(hooks_file))[0] == 0
    first, written = hooks_file.read_bytes(), hooks_file.stat().st_ino
    assert mem("hooks", "install", "--file", str(hooks_file))[0] == 0
    # Not written again either: Droid tells its user when the file changes under it.
    assert (hooks_file.read_bytes(), hooks_file.stat().st_ino) == (first, written)
    # For another port, and with a flag: one command at each hook event still, the new one.
    moved = Mem(home=mem.home, port=mem.port + 1, capsys=mem.capsys)
    assert moved("hooks", "install", "--keep-raw", "--file", str(hooks_file))[0] == 0
    found = read(hooks_file)
    assert list(found) == list(json.loads(first))
    assert found["Stop"] == [hook(capture(moved, "--keep-raw"))]
    assert found["SessionStart"] == [
        *theirs()["SessionStart"],
        hook(capture(moved, "--keep-raw")),
    ]


def test_install_takes_mem_event_out_of_hook_events_that_do_not_need_it(mem, hooks_file):
    # Written by hand at hook events where it records nothing, or twice where it does.
    by_hand = "mem event --harness droid"
    audit = {"type": "command", "command": "/usr/local/bin/audit.sh"}
    hooks_file.write_text(
        json.dumps(
            {
                "PostToolUse": [hook(by_hand)],
                "PreToolUse": [{"matcher": "Execute", "hooks": [audit, hook(by_hand)["hooks"][0]]}],
                "Notification": [hook(by_hand), hook(by_hand)],
            }
        )
    )
    code, out, _ = mem("hooks", "status", "--file", str(hooks_file))
    assert code == 1
    lines = {line.split()[0]: line for line in out.splitlines() if line.startswith("  ")}
    assert "repeated 2 times" in lines["Notification"]
    assert "not needed" in lines["PostToolUse"]
    code, out, _ = mem("hooks", "install", "--file", str(hooks_file))
    assert code == 0
    assert "PostToolUse, PreToolUse" in out
    found = read(hooks_file)
    assert "PostToolUse" not in found
    assert found["PreToolUse"] == [{"matcher": "Execute", "hooks": [audit]}]
    assert found["Notification"] == [hook(capture(mem))]


def test_install_can_ask_for_a_note_before_each_tool_call(mem, hooks_file):
    assert mem("hooks", "install", "--record-pre-tool", "--file", str(hooks_file))[0] == 0
    found = read(hooks_file)
    assert list(found) == [*EVENTS, "PreToolUse"]
    assert found["PreToolUse"] == [hook(capture(mem, "--record-pre-tool"))]
    code, out, _ = mem("hooks", "status", "--file", str(hooks_file))
    assert code == 0
    assert "not needed" not in out
    # Installed again without the flag, nothing runs before a tool call.
    assert mem("hooks", "install", "--file", str(hooks_file))[0] == 0
    assert list(read(hooks_file)) == EVENTS


def test_a_longer_command_that_runs_mem_event_is_left_alone(mem, hooks_file):
    # Taking one of these out would take the rest of its command with it.
    longer = [
        "mem event --harness droid || true",
        "cd /somewhere && uv run mem event --harness droid",
        "mem event --harness droid 2>/dev/null",
        "/usr/local/bin/notify.sh; mem event --harness droid",
    ]
    written = {"Notification": [hook(command) for command in longer]}
    hooks_file.write_text(json.dumps(written))
    code, out, err = mem("hooks", "install", "--file", str(hooks_file))
    assert code == 0
    assert read(hooks_file)["Notification"] == [*written["Notification"], hook(capture(mem))]
    assert "4 other hooks" in out
    # It says so, because capture now runs more than once at that hook event.
    for command in longer:
        assert command in err
    code, out, _ = mem("hooks", "status", "--file", str(hooks_file))
    assert code == 0
    for command in longer:
        assert command in out
    assert mem("hooks", "uninstall", "--file", str(hooks_file))[0] == 0
    assert read(hooks_file) == written


def test_install_keeps_what_it_does_not_recognise(mem, hooks_file):
    odd = [
        "a string",
        {"hooks": "not a list"},
        {"matcher": "Execute"},
        {"hooks": [7, {}]},
        # Another command with an `event` of its own, and mem doing something else.
        hook("/usr/local/bin/tracker event --harness droid"),
        hook("mem ingest --spool"),
        hook("mem --home event spool status"),
        hook("mem event --harness droid 'unfinished"),
    ]
    hooks_file.write_text(json.dumps({"Stop": odd, "Later": []}))
    assert mem("hooks", "install", "--file", str(hooks_file))[0] == 0
    assert read(hooks_file)["Stop"] == [*odd, hook(capture(mem))]
    assert read(hooks_file)["Later"] == []
    assert mem("hooks", "uninstall", "--file", str(hooks_file))[0] == 0
    assert read(hooks_file) == {"Stop": odd, "Later": []}


def test_uninstall_leaves_the_file_as_it_was_without_mem(mem, hooks_file):
    hooks_file.write_text(json.dumps(theirs()))
    assert mem("hooks", "install", "--file", str(hooks_file))[0] == 0
    code, out, _ = mem("hooks", "uninstall", "--file", str(hooks_file))
    assert code == 0
    # An event that held nothing before mem's hook holds nothing after, and is not kept.
    assert read(hooks_file) == {event: groups for event, groups in theirs().items() if groups}
    assert list(read(hooks_file)) == ["SessionStart", "PreToolUse"]
    assert "7 hooks" in out
    assert "2 other hooks" in out
    # With nothing of mem's in it the file is not written again.
    hooks_file.write_text(json.dumps(theirs()))
    before = hooks_file.read_bytes()
    code, out, _ = mem("hooks", "uninstall", "--file", str(hooks_file))
    assert (code, hooks_file.read_bytes()) == (0, before)
    assert "no hook" in out
    hooks_file.unlink()
    assert mem("hooks", "uninstall", "--file", str(hooks_file))[0] == 0
    assert not hooks_file.exists()


def test_status_says_at_which_hook_events_capture_runs(mem, hooks_file):
    code, out, _ = mem("hooks", "status", "--file", str(hooks_file))
    assert code == 1
    assert out.count("missing") == 7
    assert mem("hooks", "install", "--keep-raw", "--file", str(hooks_file))[0] == 0
    code, out, _ = mem("hooks", "status", "--file", str(hooks_file))
    assert code == 0
    assert out.count("installed") >= 7
    assert "missing" not in out
    # Read with another port in use, the same file runs something else.
    moved = Mem(home=mem.home, port=mem.port + 1, capsys=mem.capsys)
    code, out, _ = moved("hooks", "status", "--file", str(hooks_file))
    assert code == 1
    assert out.count("differs") == 7
    assert capture(mem, "--keep-raw") in out
    # One hook event short, and one too many.
    found = read(hooks_file)
    del found["Stop"]
    found["PostToolUse"] = [hook(capture(mem))]
    hooks_file.write_text(json.dumps(found))
    code, out, _ = mem("hooks", "status", "--file", str(hooks_file))
    assert code == 1
    lines = {line.split()[0]: line for line in out.splitlines() if line.startswith("  ")}
    assert "missing" in lines["Stop"]
    assert "not needed" in lines["PostToolUse"]
    assert "installed" in lines["SessionEnd"]


@pytest.mark.parametrize(
    "content",
    ["not json", "[]", '{"Stop": {"hooks": []}}'],
    ids=["not JSON", "not an object", "an event that is not a list"],
)
def test_install_leaves_a_file_it_cannot_understand_alone(mem, hooks_file, content):
    hooks_file.write_text(content)
    code, _, err = mem("hooks", "install", "--file", str(hooks_file))
    assert code == 1
    assert str(hooks_file) in err
    assert "nothing was changed" in err
    assert hooks_file.read_text() == content


def test_install_refuses_where_it_would_hide_hooks_or_has_nowhere_to_write(mem, hooks_file):
    # Droid reads the hooks in settings.json only while there is no hooks.json.
    settings = hooks_file.with_name("settings.json")
    settings.write_text(json.dumps({"hooks": theirs()}))
    code, _, err = mem("hooks", "install", "--file", str(hooks_file))
    assert code == 1
    assert "settings.json" in err
    assert not hooks_file.exists()
    settings.write_text(json.dumps({"model": "some-model"}))
    assert mem("hooks", "install", "--file", str(hooks_file))[0] == 0
    nowhere = hooks_file.parent / "absent" / "hooks.json"
    code, _, err = mem("hooks", "install", "--file", str(nowhere))
    assert code == 1
    assert "does not exist" in err


def test_install_does_not_hide_hooks_it_cannot_read(mem, hooks_file):
    # Settings with a comment are not JSON: a mention of hooks is taken for some.
    settings = hooks_file.with_name("settings.json")
    settings.write_text('{\n  // my hooks\n  "hooks": {"Stop": []}\n}\n')
    code, _, err = mem("hooks", "install", "--file", str(hooks_file))
    assert (code, hooks_file.exists()) == (1, False)
    assert "settings.json" in err
    settings.unlink()
    # The file Droid kept its hooks in before still loads while there is no hooks.json.
    before = hooks_file.parent / "hooks" / "hooks.json"
    before.parent.mkdir()
    before.write_text(json.dumps(theirs()))
    code, _, err = mem("hooks", "install", "--file", str(hooks_file))
    assert (code, hooks_file.exists()) == (1, False)
    assert str(before) in err


def test_install_names_the_mem_it_is_told_to(mem, hooks_file, tmp_path):
    code, _, err = mem("hooks", "install", "--file", str(hooks_file), "--mem", "/no/such/mem")
    assert code == 1
    assert "/no/such/mem" in err
    assert not hooks_file.exists()
    other = tmp_path / "bin" / "mem"
    other.parent.mkdir()
    other.write_text("#!/bin/sh\n")
    other.chmod(0o755)
    assert mem("hooks", "install", "--file", str(hooks_file), "--mem", str(other))[0] == 0
    assert shlex.split(read(hooks_file)["Stop"][0]["hooks"][0]["command"])[0] == str(other)
    assert mem("hooks", "status", "--file", str(hooks_file), "--mem", str(other))[0] == 0
    assert mem("hooks", "status", "--file", str(hooks_file))[0] == 1
    # Under another name its hooks would not be found again, to change or to take out.
    renamed = other.with_name("agent-mem")
    other.rename(renamed)
    before = hooks_file.read_bytes()
    code, _, err = mem("hooks", "install", "--file", str(hooks_file), "--mem", str(renamed))
    assert code == 1
    assert "not named mem" in err
    assert hooks_file.read_bytes() == before


def test_install_writes_through_a_link_to_the_file(mem, hooks_file, tmp_path):
    # A hooks file kept elsewhere and linked into place stays a link.
    kept = tmp_path / "dotfiles" / "hooks.json"
    kept.parent.mkdir()
    kept.write_text(json.dumps(theirs()))
    hooks_file.symlink_to(kept)
    assert mem("hooks", "install", "--file", str(hooks_file))[0] == 0
    assert hooks_file.is_symlink()
    assert read(kept)["Stop"] == [hook(capture(mem))]


def test_the_installed_command_runs_as_a_hook(down_cfg, hooks_file, capsys, monkeypatch):
    for var in ("AGENT_MEMORY_HOME", "AGENT_MEMORY_HOST", "AGENT_MEMORY_PORT"):
        monkeypatch.delenv(var, raising=False)
    mem = Mem(home=down_cfg.home, port=down_cfg.port, capsys=capsys)
    assert mem("hooks", "install", "--file", str(hooks_file))[0] == 0
    command = read(hooks_file)["Notification"][0]["hooks"][0]["command"]
    payload = {
        "session_id": "installed-hook",
        "cwd": "/synthetic/hooks",
        "hook_event_name": "Notification",
        "message": "Droid needs your input",
    }
    # As Droid runs it: through a shell, the payload on stdin, somewhere else.
    ran = subprocess.run(
        command,
        shell=True,
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        timeout=60,
        cwd="/",
        check=False,
    )
    assert (ran.returncode, ran.stdout) == (0, "")
    assert Spool(down_cfg.spool_dir).status()[0].lines == 1
