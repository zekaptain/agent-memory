"""Bounded NDJSON transport and a linear, cursor-free interface for the same controller."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
from typing import IO, Any, TextIO

from .label_build import MAX_FRAME, PROTOCOL_VERSION, frontend_binary
from .label_controller import Action, LabelController, Screen
from .label_questions import display_safe
from .labeling import LabelSummary
from .store import JudgeError, RunStore


def encode_frame(frame: dict[str, Any]) -> bytes:
    data = (json.dumps(frame, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    if len(data) > MAX_FRAME:
        raise JudgeError("Label screen exceeds the protocol limit. Use --plain.")
    return data


def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate field")
        result[key] = value
    return result


def read_action(stream: IO[bytes], request_id: int) -> Action:
    data = stream.readline(MAX_FRAME + 1)
    if not data:
        raise JudgeError("Label UI closed. Earlier saved items are safe. Retry or use --plain.")
    if len(data) > MAX_FRAME or not data.endswith(b"\n"):
        raise JudgeError("Invalid label UI frame size. Retry or use --plain.")
    try:
        frame = json.loads(data, object_pairs_hook=unique_object)
        if (
            not isinstance(frame, dict)
            or set(frame) != {"version", "request_id", "type", "action", "value"}
            or type(frame["version"]) is not int
            or frame["version"] != PROTOCOL_VERSION
            or type(frame["request_id"]) is not int
            or frame["request_id"] != request_id
            or frame["type"] != "action"
            or frame["action"] not in ("answer", "back", "skip", "quit", "redo", "note")
            or not isinstance(frame["value"], str)
        ):
            raise ValueError
        frame["value"].encode("utf-8")
        return Action(frame["action"], frame["value"])
    except (ValueError, TypeError, RecursionError):
        raise JudgeError("Invalid label UI action. Retry or use --plain.") from None


class GoFrontend:
    def __init__(self, binary: Path) -> None:
        self.request_id = 0
        try:
            # Only display data crosses stdin. Neither argv nor env contains session text.
            # The frontend uses /dev/tty for rendering; stdout is exclusively the protocol.
            self.process: subprocess.Popen[bytes] = subprocess.Popen(
                [str(binary)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
        except OSError:
            raise JudgeError("Could not launch label UI. Retry or use --plain.") from None

    def send(self, kind: str, **fields: Any) -> None:
        self.request_id += 1
        data = encode_frame(
            {"version": PROTOCOL_VERSION, "request_id": self.request_id, "type": kind, **fields}
        )
        try:
            assert self.process.stdin is not None
            self.process.stdin.write(data)
            self.process.stdin.flush()
        except OSError:
            raise JudgeError(
                "Label UI disconnected. Earlier saved items are safe. Retry or use --plain."
            ) from None

    def exchange(self, screen: Screen) -> Action:
        self.send("screen", screen=screen.payload())
        assert self.process.stdout is not None
        return read_action(self.process.stdout, self.request_id)

    def finish(self, message: str) -> None:
        self.send("done", message=message)
        try:
            code = self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            raise JudgeError("Label UI did not close. Earlier saved items are safe.") from None
        if code:
            raise JudgeError("Label UI failed while closing. Earlier saved items are safe.")

    def close(self) -> None:
        if self.process.stdin:
            with suppress(OSError):
                self.process.stdin.close()
        try:
            self.process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        if self.process.stdout:
            self.process.stdout.close()


class PlainFrontend:
    def __init__(self, source: TextIO, output: TextIO) -> None:
        self.source, self.output = source, output

    def write(self, text: str) -> None:
        print(display_safe(text), file=self.output, flush=True)

    def line(self) -> str | None:
        line = self.source.readline()
        return line.rstrip("\r\n") if line else None

    def exchange(self, screen: Screen) -> Action:
        self.write(f"\nLabel review | Item {screen.item} of {screen.total} | {screen.subject}")
        self.write(screen.metadata.replace(" · ", "\n"))
        if screen.notice:
            self.write(screen.notice)
        self.write(screen.evidence)
        for draft in screen.drafts:
            focus = f"\n{draft.focus}" if draft.focus else ""
            self.write(
                f"\nDraft {draft.slot}{focus}\n{draft.lesson}"
                f"\nCited evidence\n{draft.cited_evidence}"
            )
        while True:
            self.write("\n" + screen.question)
            for index, option in enumerate(screen.options, 1):
                current = " (current)" if option.id == screen.selected else ""
                self.write(f"{index}. {option.label}{current}")
            commands = [
                *(["back"] if screen.can_back else []),
                "evidence",
                "help",
                *(["redo"] if screen.can_redo else []),
                "skip",
                "quit",
            ]
            self.write("Number or: " + ", ".join(commands))
            value = self.line()
            if value is None or value.lower() in ("quit", "q"):
                return Action("quit")
            lower = value.strip().lower()
            if lower in ("evidence", "p"):
                self.write(screen.projection)
            elif lower in ("help", "?"):
                self.write(screen.guide)
            elif lower in ("back", "b") and screen.can_back:
                return Action("back")
            elif lower in ("redo", "r") and screen.can_redo:
                return Action("redo")
            elif lower in ("skip", "s"):
                return Action("skip")
            elif lower in {str(number) for number in range(1, len(screen.options) + 1)}:
                selected = screen.options[int(lower) - 1].id
                if screen.key.endswith(":note") and selected == "edit":
                    self.write("Note (one line, letters are literal; /cancel cancels):")
                    note = self.line()
                    if note is None:
                        return Action("quit")
                    if note == "/cancel":
                        continue
                    note = note.replace("\t", " ")  # as the Go helper's input reads a tab
                    if display_safe(note) != note or len(note.encode("utf-8")) > 4096:
                        self.write("Use one line of plain text, at most 4096 bytes.")
                        continue
                    return Action("note", note)
                return Action("answer", selected)
            else:
                self.write("Enter a listed number or navigation command.")


def automatic_plain() -> bool:
    return not sys.stdin.isatty() or not sys.stdout.isatty() or os.environ.get("TERM") == "dumb"


def label_session(
    store: RunStore,
    *,
    plain: bool = False,
    limit: int | None = None,
    seed: int = 1,
    item_ids: tuple[str, ...] = (),
    review: bool = False,
    progress: Callable[[str], None] | None = None,
) -> LabelSummary:
    controller = LabelController(store, limit=limit, seed=seed, item_ids=item_ids, review=review)
    frontend: GoFrontend | PlainFrontend | None = None
    try:
        # An empty/resumed run never requires Go or a terminal.
        if not controller.done:
            frontend = (
                PlainFrontend(sys.stdin, sys.stdout)
                if plain or automatic_plain()
                else GoFrontend(frontend_binary(progress=progress))
            )
            while not controller.done:
                controller.apply(frontend.exchange(controller.screen()))
            if isinstance(frontend, GoFrontend):
                frontend.finish(controller.result_text())
    except (KeyboardInterrupt, EOFError):
        controller.stopped = True
    finally:
        if isinstance(frontend, GoFrontend):
            frontend.close()
    print(controller.result_text())
    return controller.summary()
