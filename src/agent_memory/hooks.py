"""`mem hooks`: mem's capture command in Droid's hooks file, and nothing else in it changed.
Registered on the main app from cli.py."""

from __future__ import annotations

import json
import os
import re
import shlex
import stat
import sys
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

import typer

from .adapters.droid import DroidAdapter
from .config import Config, command_config
from .db import AgentMemoryError

hooks_app = typer.Typer(
    help="Write mem's hooks into Droid's hooks file, check them, or take them out.",
    no_args_is_help=True,
)

DEFAULT_FILE = Path("~/.factory/hooks.json")

# What makes a shell run more than the one command, or send its output somewhere.
_SHELL_OPERATORS = frozenset("();<>|&")
# `mem event` somewhere in a command, whatever else the command does.
_MENTIONS_EVENT = re.compile(r"\bmem\b[^;&|]*\bevent\b")

# Droid's file: the groups of each hook event, and in each group the hooks it runs.
Hooks = dict[str, list[Any]]

# Droid's documentation says it reads its hooks when a session starts. A session that was
# running when the file changed ran the new hooks ten seconds later (Droid's desktop app,
# 2026-11-11), so neither is promised.
RUNNING_SESSIONS = (
    "A Droid session that is already running may use the change at once or keep the hooks\n"
    "it started with. /hooks in Droid shows the hooks a session runs."
)


class HooksError(AgentMemoryError):
    """`mem hooks` did not do what was asked."""


def runs_event(args: Sequence[str]) -> bool:
    """Whether these arguments to `mem` run `mem event`: whether that is the first word
    which is neither a root option nor the value of one."""
    is_value = False
    for arg in args:
        if is_value:
            is_value = False
        elif arg in ("--home", "--port"):
            is_value = True
        elif not arg.startswith("-"):
            return arg == "event"
    return False


def _capture_words(mem: Path, cfg: Config) -> list[str]:
    """The command a hook runs, before its flags. It names the mem to run, the home, and
    the port, because a hook runs with Droid's environment and working directory."""
    home, port = str(cfg.home.absolute()), str(cfg.port)
    return [str(mem), "--home", home, "--port", port, "event", "--harness", "droid"]


def _is_mems(hook: Any) -> bool:
    """Whether a hook does nothing but run `mem event`. One that runs it inside a longer
    command is someone else's: taking it out would take the rest of the command with it."""
    command = hook.get("command") if isinstance(hook, dict) else None
    if not isinstance(command, str):
        return False
    lexer = shlex.shlex(command, posix=True, punctuation_chars=True)
    lexer.whitespace_split = True
    try:
        words = list(lexer)
    except ValueError:
        return False
    if not words or any(set(word) <= _SHELL_OPERATORS for word in words):
        return False
    return Path(words[0]).name == "mem" and runs_event(words[1:])


def _entries(group: Any) -> list[Any]:
    entries = group.get("hooks") if isinstance(group, dict) else None
    return entries if isinstance(entries, list) else []


def _without_mems(hooks: Hooks, keep: Sequence[str] = ()) -> tuple[Hooks, int]:
    """The hooks without mem's, and how many those were. A group that held nothing else
    goes with its hook, and so does a hook event, unless it is one of `keep`. Everything
    else stays as it was."""
    left: Hooks = {}
    removed = 0
    for event, groups in hooks.items():
        kept: list[Any] = []
        gone = 0
        for group in groups:
            theirs = [hook for hook in _entries(group) if not _is_mems(hook)]
            mine = len(_entries(group)) - len(theirs)
            if not mine:
                kept.append(group)
            elif theirs:
                kept.append({**group, "hooks": theirs})
            gone += mine
        if kept or not gone or event in keep:
            left[event] = kept
        removed += gone
    return left, removed


def _with_mems(hooks: Hooks, command: str, events: Sequence[str]) -> Hooks:
    """The hooks with mem's command once at each of `events` and at no other, after the
    hooks already there."""
    after, _ = _without_mems(hooks, keep=events)
    for event in events:
        after.setdefault(event, []).append({"hooks": [{"type": "command", "command": command}]})
    return after


def _also_runs_event(hook: Any) -> bool:
    """Whether a hook that is not mem's runs `mem event` somewhere in its command."""
    command = hook.get("command") if isinstance(hook, dict) else None
    return isinstance(command, str) and not _is_mems(hook) and bool(_MENTIONS_EVENT.search(command))


def _commands(hooks: Hooks, wanted: Callable[[Any], bool] = _is_mems) -> dict[str, list[str]]:
    """Per hook event, the commands of the hooks wanted: mem's, unless told otherwise."""
    found: dict[str, list[str]] = {}
    for event, groups in hooks.items():
        for hook in (hook for group in groups for hook in _entries(group)):
            if wanted(hook):
                found.setdefault(event, []).append(hook["command"])
    return found


def _others(hooks: Hooks) -> int:
    return sum(
        not _is_mems(hook)
        for groups in hooks.values()
        for group in groups
        for hook in _entries(group)
    )


def _read(path: Path) -> Hooks | None:
    """The hooks in Droid's file, or None when there is no file. A file in another shape
    than the one Droid documents is refused, not guessed at."""
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except (OSError, UnicodeDecodeError) as exc:
        raise HooksError(f"{path} cannot be read ({exc}); nothing was changed") from None
    try:
        hooks = json.loads(text)
    except json.JSONDecodeError as exc:
        raise HooksError(f"{path} is not JSON ({exc}); nothing was changed") from None
    if not isinstance(hooks, dict):
        raise HooksError(f"{path} does not hold an object keyed by hook event; nothing was changed")
    for event, groups in hooks.items():
        if not isinstance(groups, list):
            raise HooksError(f"{path}: {event!r} does not hold a list; nothing was changed")
    return hooks


@dataclass(frozen=True)
class _SettingsHooks:
    """Hook events declared in `settings.json` beside `hooks.json`, without its commands."""

    path: Path
    events: tuple[str, ...] = ()
    unreadable_reason: str | None = None


def _declares_hooks(groups: Any) -> bool:
    """Whether a hook-event value declares at least one hook."""
    if not isinstance(groups, list):
        return bool(groups)
    for group in groups:
        if isinstance(group, dict) and "hooks" in group:
            entries = group["hooks"]
            if isinstance(entries, list):
                if entries:
                    return True
            elif entries:
                return True
        elif group:
            return True
    return False


def _defined_events(hooks: Hooks) -> set[str]:
    """Hook events at which `hooks` declares at least one hook."""
    return {event for event, groups in hooks.items() if _declares_hooks(groups)}


def _legacy_hidden_by(path: Path) -> Path | None:
    """The earlier `hooks/hooks.json`, which Droid reads only while there is no `hooks.json`."""
    legacy = path.parent / "hooks" / "hooks.json"
    return legacy if legacy.exists() else None


def _read_settings_hooks(path: Path) -> _SettingsHooks:
    """Hook events declared in `settings.json` beside `path`, without reading its commands.
    Droid runs a hook event from `settings.json` only when `hooks.json` beside it does not
    define that event."""
    settings = path.with_name("settings.json")
    try:
        text = settings.read_text(encoding="utf-8")
    except FileNotFoundError:
        return _SettingsHooks(path=settings)
    except (OSError, UnicodeDecodeError) as exc:
        return _SettingsHooks(path=settings, unreadable_reason=f"cannot be read ({exc})")
    try:
        declared = json.loads(text)
    except json.JSONDecodeError:
        # Not plain JSON, so it cannot be read here: a mention of hooks is taken for some.
        if '"hooks"' in text:
            return _SettingsHooks(path=settings, unreadable_reason="mentions hooks but is not JSON")
        return _SettingsHooks(path=settings)
    if not isinstance(declared, dict) or not declared.get("hooks"):
        return _SettingsHooks(path=settings)
    raw_hooks = declared["hooks"]
    if not isinstance(raw_hooks, dict):
        return _SettingsHooks(
            path=settings, unreadable_reason="declares hooks in an unrecognised shape"
        )
    events = tuple(
        event
        for event, groups in raw_hooks.items()
        if isinstance(event, str) and _declares_hooks(groups)
    )
    return _SettingsHooks(path=settings, events=events)


def _write(path: Path, hooks: Hooks) -> None:
    """Put the hooks in the file in one step: through a link to it, and with the mode it
    had, or a private one if it is new."""
    target = path.resolve()
    try:
        mode = stat.S_IMODE(target.stat().st_mode)
    except FileNotFoundError:
        mode = 0o600
    text = json.dumps(hooks, indent=2, ensure_ascii=False) + "\n"
    try:
        fd, name = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(text)
            os.chmod(name, mode)
            os.replace(name, target)
        finally:
            Path(name).unlink(missing_ok=True)
    except OSError as exc:
        raise HooksError(f"{path} cannot be written ({exc}); nothing was changed") from None


def _mem(given: Path | None) -> Path:
    """The mem a hook runs: the one given, or the one installed beside this Python."""
    return (given or Path(sys.executable).parent / "mem").expanduser().absolute()


def _counted(count: int, noun: str = "hook") -> str:
    return f"{count} {noun}{'' if count == 1 else 's'}"


def _say_others(hooks: Hooks) -> None:
    count = _others(hooks)
    if count:
        print(f"{_counted(count, 'other hook')} {'was' if count == 1 else 'were'} left alone")


FILE = Annotated[
    Path | None,
    typer.Option(
        "--file", help="Droid's hooks file. Default: ~/.factory/hooks.json.", show_default=False
    ),
]
MEM = Annotated[
    Path | None,
    typer.Option(
        "--mem",
        help="The mem command a hook runs. Default: the one installed beside this Python.",
        show_default=False,
    ),
]


@hooks_app.command()
def install(
    ctx: typer.Context,
    file: FILE = None,
    mem: MEM = None,
    keep_raw: Annotated[
        bool, typer.Option("--keep-raw", help="Keep each full hook payload with its event.")
    ] = False,
    record_pre_tool: Annotated[
        bool,
        typer.Option("--record-pre-tool", help="Also run at PreToolUse and record a note there."),
    ] = False,
) -> None:
    """Write mem's hooks into Droid's hooks file.

    Each runs `mem event` for the home and port in use: once at every hook event capture
    needs and at no other. Every other hook in the file is left as it is. Run it again
    to change the home, the port, or a flag.
    """
    cfg = command_config(ctx)
    path = (file or DEFAULT_FILE).expanduser()
    command_path = _mem(mem)
    if command_path.name != "mem":
        raise HooksError(f"{command_path} is not named mem, so `mem hooks` could not find it again")
    if not (command_path.is_file() and os.access(command_path, os.X_OK)):
        hint = "" if mem else "; name the one a hook should run with --mem"
        raise HooksError(f"{command_path} is not a command that can be run{hint}")
    flags = [
        flag
        for flag, asked in (("--keep-raw", keep_raw), ("--record-pre-tool", record_pre_tool))
        if asked
    ]
    command = shlex.join([*_capture_words(command_path, cfg), *flags])
    events = DroidAdapter(record_pre_tool=record_pre_tool).hook_events

    before = _read(path)
    if before is None:
        if not path.parent.is_dir():
            raise HooksError(
                f"{path.parent} does not exist. Droid reads ~/.factory/hooks.json, and "
                "a project's .factory/hooks.json; name one of them with --file"
            )
        legacy = _legacy_hidden_by(path)
        if legacy:
            raise HooksError(
                f"{legacy} holds hooks that Droid reads only while there is no {path.name} "
                f"beside it, so writing {path} would switch them off. Save them from Droid's "
                "/hooks screen first, which moves them, then run this again"
            )
    after = _with_mems(before or {}, command, events)
    settings_hooks = _read_settings_hooks(path)
    before_defined = _defined_events(before or {})
    after_defined = _defined_events(after)
    added_events = [
        event for event in after if event in after_defined and event not in before_defined
    ]
    if settings_hooks.unreadable_reason is not None:
        if added_events:
            raise HooksError(
                f"{settings_hooks.path} {settings_hooks.unreadable_reason}, so writing {path} "
                "could hide hooks it declares; nothing was changed"
            )
    else:
        newly_hidden = [
            event
            for event in settings_hooks.events
            if event in after_defined and event not in before_defined
        ]
        if newly_hidden:
            noun = "that event" if len(newly_hidden) == 1 else "those events"
            raise HooksError(
                f"{settings_hooks.path} declares hooks at {', '.join(newly_hidden)}, "
                f"which Droid stops running once {path.name} defines {noun}. "
                f"Move them into {path.name} first, then run this again"
            )
    if after == before:
        print(f"{_counted(len(events))} already in {path}; nothing to change")
    else:
        _write(path, after)
        print(f"wrote {_counted(len(events))} to {path}: {', '.join(events)}")
    print(f"each runs: {command}")
    _say_others(after)
    dropped = [event for event in _commands(before or {}) if event not in events]
    if dropped:
        print(f"took mem event out of {', '.join(dropped)}, where it records nothing")
    if after != before:
        print(RUNNING_SESSIONS)
    _warn_also(after)
    if settings_hooks.unreadable_reason is not None:
        _warn(
            f"{settings_hooks.path} {settings_hooks.unreadable_reason}; "
            f"could not check which events {path.name} hides"
        )
    else:
        already_hidden = [
            event
            for event in settings_hooks.events
            if event in after_defined and event in before_defined
        ]
        if already_hidden:
            noun, pron = (
                ("that event", "it") if len(already_hidden) == 1 else ("those events", "them")
            )
            _warn(
                f"{path.name} hides hooks declared in {settings_hooks.path} at "
                f"{', '.join(already_hidden)} (Droid does not run {noun} from settings.json "
                f"while {path.name} defines {pron})"
            )
    if not cfg.db_path.exists():
        _warn(
            f"{cfg.home} holds no ledger yet. A hook keeps what it records in the spool "
            "there until you run `mem init` and `mem serve` for it"
        )
    if os.environ.get("AGENT_MEMORY_HOST", "localhost") != "localhost":
        _warn("AGENT_MEMORY_HOST is set here. A hook sees it only if Droid's environment does")


def _warn(message: str) -> None:
    print(f"mem hooks: {message}", file=sys.stderr)


def _warn_also(hooks: Hooks) -> None:
    for event, commands in _commands(hooks, _also_runs_event).items():
        for command in commands:
            _warn(
                f"a {event} hook also runs mem event, inside a longer command that was left "
                f"alone. Capture runs twice there until you take it out: {command}"
            )


@hooks_app.command()
def uninstall(file: FILE = None) -> None:
    """Take mem's hooks out of Droid's hooks file.

    Every other hook is left as it is, also one that runs `mem event` inside a longer
    command.
    """
    path = (file or DEFAULT_FILE).expanduser()
    before = _read(path)
    if before is None:
        print(f"{path} does not exist; there is no hook to take out")
        return
    after, removed = _without_mems(before)
    if not removed:
        print(f"{path} holds no hook that runs mem event; nothing to change")
    else:
        _write(path, after)
        print(f"took {_counted(removed)} out of {path}")
        print(RUNNING_SESSIONS)
    _say_others(after)
    _warn_also(after)


def _state(commands: list[str], words: list[str]) -> str:
    if not commands:
        return "missing"
    if len(commands) > 1:
        return f"repeated {len(commands)} times"
    return "installed" if shlex.split(commands[0])[: len(words)] == words else "differs"


@hooks_app.command()
def status(ctx: typer.Context, file: FILE = None, mem: MEM = None) -> None:
    """Say at which hook events capture runs.

    Exits 0 when `mem event` runs for the home and port in use, once at every hook event
    capture needs and at no other.
    """
    cfg = command_config(ctx)
    path = (file or DEFAULT_FILE).expanduser()
    hooks = _read(path)
    words = _capture_words(_mem(mem), cfg)
    found = _commands(hooks or {})
    pre_tool = any("--record-pre-tool" in shlex.split(c) for cs in found.values() for c in cs)
    needed = DroidAdapter(record_pre_tool=pre_tool).hook_events
    states = {event: _state(found.get(event, []), words) for event in needed}
    states |= {
        event: "not needed: it records nothing here" for event in found if event not in needed
    }
    also = _commands(hooks or {}, _also_runs_event)

    print(path if hooks is not None else f"{path} (no such file)")
    width = max(len(event) for event in states)
    for event, state in states.items():
        print(f"  {event.ljust(width)}  {state}")
    for command in dict.fromkeys(c for cs in found.values() for c in cs):
        print(f"runs: {command}")
    for event, commands in also.items():
        for command in commands:
            print(f"a {event} hook also runs mem event, inside a longer command: {command}")
    if hooks is not None:
        settings_hooks = _read_settings_hooks(path)
        if settings_hooks.unreadable_reason is not None:
            print(
                f"{settings_hooks.path} {settings_hooks.unreadable_reason}; "
                f"could not check which events {path.name} hides"
            )
        else:
            defined = _defined_events(hooks)
            hidden = [event for event in settings_hooks.events if event in defined]
            if hidden:
                print(
                    f"{path.name} hides hooks declared in {settings_hooks.path} at: "
                    f"{', '.join(hidden)}"
                )
    if any(state != "installed" for state in states.values()):
        print(f"`mem hooks install` writes: {shlex.join(words)}")
        raise typer.Exit(1)
