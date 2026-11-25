"""`mem extension`: install mem's capture extension where omp finds it, check it, or take it
out, changing nothing else in omp's folders. Registered on the main app from cli.py."""

from __future__ import annotations

import json
import os
import shlex
import sys
import tempfile
from pathlib import Path
from typing import Annotated, Any

import typer

from .config import Config, command_config
from .db import AgentMemoryError

extension_app = typer.Typer(
    help="Install mem's capture extension for omp, check it, or take it out.",
    no_args_is_help=True,
)

DEFAULT_DIR = Path("~/.omp/agent/extensions")
EXTENSION_DIR_NAME = "agent-memory"
SOURCE_FILES = ("index.ts", "mapper.ts", "sink.ts")
CONFIG_NAME = "config.json"
MANAGED_FILES = (*SOURCE_FILES, CONFIG_NAME)
MANAGED_BY = "agent-memory"

NEW_SESSION_NOTE = "Start a new omp session to pick up the change."


class ExtensionError(AgentMemoryError):
    """`mem extension` did not do what was asked."""


def _source_dir() -> Path:
    return Path(__file__).resolve().parent / "adapters" / "omp"


def _mem(given: Path | None) -> Path:
    """The mem the extension runs: the one given, or the one installed beside this Python."""
    return (given or Path(sys.executable).parent / "mem").expanduser().absolute()


def _config_payload(mem_path: Path, cfg: Config) -> dict[str, Any]:
    return {
        "managed_by": MANAGED_BY,
        "mem": str(mem_path),
        "home": str(cfg.home),
        "port": cfg.port,
        "files": list(MANAGED_FILES),
    }


def _config_bytes(mem_path: Path, cfg: Config) -> bytes:
    return (json.dumps(_config_payload(mem_path, cfg), indent=2, ensure_ascii=False) + "\n").encode(
        "utf-8"
    )


def _desired_files(mem_path: Path, cfg: Config) -> dict[str, tuple[bytes, int]]:
    src = _source_dir()
    files: dict[str, tuple[bytes, int]] = {}
    for name in SOURCE_FILES:
        path = src / name
        if not path.is_file():
            raise ExtensionError(f"bundled extension file {path} is missing")
        files[name] = (path.read_bytes(), 0o644)
    files[CONFIG_NAME] = (_config_bytes(mem_path, cfg), 0o600)
    return files


def _read_config(path: Path) -> dict[str, Any] | None:
    if path.is_symlink() or not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _ownership_problem(target: Path) -> str | None:
    """Return a reason `target` cannot be modified safely, or None if it is managed by mem."""
    if target.is_symlink() or not target.is_dir():
        return f"{target} exists and is not an agent-memory extension directory"
    cfg = _read_config(target / CONFIG_NAME)
    if cfg is None or cfg.get("managed_by") != MANAGED_BY:
        return f"{target} was not installed by `mem extension` (missing managed {CONFIG_NAME})"
    extras = sorted(entry.name for entry in target.iterdir() if entry.name not in MANAGED_FILES)
    if extras:
        return f"{target} contains files not managed by mem ({', '.join(extras)})"
    bad_entries = sorted(
        entry.name for entry in target.iterdir() if entry.is_symlink() or not entry.is_file()
    )
    if bad_entries:
        return f"{target} contains unexpected non-file entries ({', '.join(bad_entries)})"
    return None


def _write_atomic(path: Path, data: bytes, mode: int) -> None:
    try:
        fd, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(data)
            os.chmod(name, mode)
            os.replace(name, path)
        finally:
            Path(name).unlink(missing_ok=True)
    except OSError as exc:
        raise ExtensionError(f"{path} cannot be written ({exc}); nothing was changed") from None


def _counted(count: int, noun: str = "extension") -> str:
    return f"{count} {noun}{'' if count == 1 else 's'}"


def _other_entries(extensions_dir: Path) -> int:
    if not extensions_dir.is_dir():
        return 0
    return sum(1 for entry in extensions_dir.iterdir() if entry.name != EXTENSION_DIR_NAME)


def _say_others(extensions_dir: Path) -> None:
    count = _other_entries(extensions_dir)
    if count:
        print(f"{_counted(count, 'other extension')} {'was' if count == 1 else 'were'} left alone")


def _warn(message: str) -> None:
    print(f"mem extension: {message}", file=sys.stderr)


def _command_line(mem_path: Path, cfg: Config) -> str:
    return shlex.join([str(mem_path), "--home", str(cfg.home), "--port", str(cfg.port), "records"])


DIR = Annotated[
    Path | None,
    typer.Option(
        "--dir",
        help="omp's extensions directory. Default: ~/.omp/agent/extensions.",
        show_default=False,
    ),
]
MEM = Annotated[
    Path | None,
    typer.Option(
        "--mem",
        help="The mem command the extension runs. Default: the one installed beside this Python.",
        show_default=False,
    ),
]


@extension_app.command()
def install(ctx: typer.Context, dir: DIR = None, mem: MEM = None) -> None:
    """Install mem's capture extension into omp's extensions directory.

    Writes the extension files and configuration into `<dir>/agent-memory/` and leaves
    every other file and folder in omp's directories untouched. Run again to update the
    extension files or change the home, port, or mem path.
    """
    cfg = command_config(ctx)
    extensions_dir = (dir or DEFAULT_DIR).expanduser()
    command_path = _mem(mem)
    if not (command_path.is_file() and os.access(command_path, os.X_OK)):
        hint = "" if mem else "; name the one the extension should run with --mem"
        raise ExtensionError(f"{command_path} is not a command that can be run{hint}")

    if extensions_dir.exists():
        if not extensions_dir.is_dir():
            raise ExtensionError(f"{extensions_dir} exists and is not a directory")
    else:
        if not extensions_dir.parent.is_dir():
            raise ExtensionError(
                f"{extensions_dir.parent} does not exist. omp reads ~/.omp/agent/extensions, "
                "and a project's .omp/extensions; name one with --dir"
            )
        try:
            extensions_dir.mkdir()
        except OSError as exc:
            raise ExtensionError(f"{extensions_dir} cannot be created ({exc})") from None

    target = extensions_dir / EXTENSION_DIR_NAME
    desired = _desired_files(command_path, cfg)

    existed = target.exists() or target.is_symlink()
    if existed:
        problem = _ownership_problem(target)
        if problem:
            raise ExtensionError(f"{problem}; nothing was changed")
        unchanged = all(
            (target / name).is_file() and (target / name).read_bytes() == data
            for name, (data, _) in desired.items()
        )
        if unchanged:
            print(f"already installed in {target}; nothing to change")
            print(f"runs: {_command_line(command_path, cfg)}")
            _say_others(extensions_dir)
            _post_install_warnings(cfg)
            return
        for name, (data, mode) in desired.items():
            path = target / name
            if not path.is_file() or path.read_bytes() != data:
                _write_atomic(path, data, mode)
    else:
        # Stage in a sibling temporary directory and move into place so a failed write
        # never leaves a half-installed extension directory behind.
        staging = Path(tempfile.mkdtemp(dir=extensions_dir, prefix=f".{EXTENSION_DIR_NAME}."))
        try:
            for name, (data, mode) in desired.items():
                dest = staging / name
                dest.write_bytes(data)
                os.chmod(dest, mode)
            os.replace(staging, target)
        except OSError as exc:
            for child in staging.iterdir():
                child.unlink(missing_ok=True)
            staging.rmdir()
            raise ExtensionError(
                f"{target} cannot be written ({exc}); nothing was changed"
            ) from None

    action = "updated" if existed else "installed"
    print(f"{action} omp capture extension in {target}")
    print(f"runs: {_command_line(command_path, cfg)}")
    _say_others(extensions_dir)
    print(NEW_SESSION_NOTE)
    _post_install_warnings(cfg)


def _post_install_warnings(cfg: Config) -> None:
    if not cfg.db_path.exists():
        _warn(
            f"{cfg.home} holds no ledger yet. The extension keeps what it records in the "
            "spool there until you run `mem init` and `mem serve` for it"
        )
    if os.environ.get("AGENT_MEMORY_HOST", "localhost") != "localhost":
        _warn("AGENT_MEMORY_HOST is set here. The extension sees it only if omp's environment does")


@extension_app.command()
def status(ctx: typer.Context, dir: DIR = None, mem: MEM = None) -> None:
    """Report whether mem's omp capture extension is installed and up to date.

    Exits 0 when the extension is installed for the home, port, and mem command in use
    and its files match this version of agent-memory.
    """
    cfg = command_config(ctx)
    extensions_dir = (dir or DEFAULT_DIR).expanduser()
    target = extensions_dir / EXTENSION_DIR_NAME
    command_path = _mem(mem)
    desired = _desired_files(command_path, cfg)

    if not target.exists() and not target.is_symlink():
        print(f"{target} (not installed)")
        print(f"`mem extension install` runs: {_command_line(command_path, cfg)}")
        raise typer.Exit(1)

    problem = _ownership_problem(target)
    if problem:
        print(f"{target} ({problem})")
        raise typer.Exit(1)

    installed_cfg = _read_config(target / CONFIG_NAME) or {}
    file_states: dict[str, str] = {}
    for name, (data, _) in desired.items():
        path = target / name
        if not path.is_file():
            file_states[name] = "missing"
        elif path.read_bytes() == data:
            file_states[name] = "installed"
        else:
            file_states[name] = "differs" if name == CONFIG_NAME else "out of date"

    print(target)
    width = max(len(name) for name in file_states)
    for name, state in file_states.items():
        print(f"  {name.ljust(width)}  {state}")

    if isinstance(installed_cfg.get("mem"), str):
        parts = [installed_cfg["mem"]]
        if isinstance(installed_cfg.get("home"), str):
            parts.extend(["--home", installed_cfg["home"]])
        if isinstance(installed_cfg.get("port"), int):
            parts.extend(["--port", str(installed_cfg["port"])])
        parts.append("records")
        print(f"runs: {shlex.join(parts)}")

    if any(state != "installed" for state in file_states.values()):
        print(f"`mem extension install` writes: {_command_line(command_path, cfg)}")
        raise typer.Exit(1)


@extension_app.command()
def uninstall(dir: DIR = None) -> None:
    """Remove mem's omp capture extension, leaving everything else in omp's folders alone."""
    extensions_dir = (dir or DEFAULT_DIR).expanduser()
    target = extensions_dir / EXTENSION_DIR_NAME
    if not target.exists() and not target.is_symlink():
        print(f"{target} is not installed; nothing to change")
        _say_others(extensions_dir)
        return

    problem = _ownership_problem(target)
    if problem:
        raise ExtensionError(f"{problem}; nothing was changed")

    try:
        for name in MANAGED_FILES:
            (target / name).unlink(missing_ok=True)
        target.rmdir()
    except OSError as exc:
        raise ExtensionError(f"{target} cannot be removed ({exc})") from None

    print(f"removed omp capture extension from {target}")
    _say_others(extensions_dir)
    print(NEW_SESSION_NOTE)
