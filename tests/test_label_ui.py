"""Label controller, blinding, protocol, cache, and real plain entry point regressions."""

from __future__ import annotations

import errno
import fcntl
import io
import json
import os
import pty
import select
import stat
import struct
import subprocess
import sys
import termios
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import pytest

from agent_memory.judge import label_build, label_controller, label_frontend
from agent_memory.judge.label_controller import Action, LabelController
from agent_memory.judge.label_frontend import PlainFrontend, encode_frame, read_action
from agent_memory.judge.label_questions import GUIDE, display_safe
from agent_memory.judge.labeling import SLOTS, missing_steps, order_drafts
from agent_memory.judge.report import build_report
from agent_memory.judge.store import JudgeError
from tests.test_judge import structured_draft, synthetic_run


def fresh(tmp_path: Path, n: int = 2, *, two: bool = True):
    store = synthetic_run(tmp_path, n=n)
    store.labels_path.unlink()
    if two:
        store.write_jsonl(
            store.drafts_path,
            [
                {
                    "item_id": item["item_id"],
                    "drafter": "PRIVATE_DRAFTER",
                    "lesson": "Literal [bold]x [/dim] 界 café",
                    "cited": [1, 9],
                    "probabilities": {"secret": 0.987654321},
                }
                for item in store.read_jsonl(store.items_path)
            ],
            append=True,
        )
    for item in store.read_jsonl(store.items_path):
        store.write_projection(
            item["item_id"],
            "Coding-agent turn.\n[1] REQUEST (human): Run tests\n"
            "[2] TOOL RESULT (error): 3 failed\n[3] OUTCOME: cancelled",
        )
    return store


def answer(controller: LabelController, value: str):
    controller.apply(Action("answer", value))


def to_summary(controller: LabelController):
    while controller.step != "summary":
        screen = controller.screen()
        step = controller.step
        if step == "note":
            value = "continue"
        elif step in ("failed", "worth_remembering"):
            value = "n"
        else:
            value = screen.selected
        answer(controller, value)


def test_blinded_protocol_and_literal_evidence(tmp_path):
    store = fresh(tmp_path, 1)
    controller = LabelController(store)
    frames = []
    while controller.step != "summary":
        screen = controller.screen()
        frames.append(screen.payload())
        value = "continue" if controller.step == "note" else screen.options[0].id
        answer(controller, value)
    frames.append(controller.screen().payload())
    boundary = json.dumps(frames, ensure_ascii=False)
    for hidden in (
        "PRIVATE_DRAFTER",
        "ollama:m",
        "llm:ref",
        "probabilities",
        "0.987654321",
        "drafter",
        "backend",
        "draft_hash",
        "verdict",
        "slot_mapping",
    ):
        assert hidden not in boundary
    assert "Literal [bold]x [/dim] 界 café" in boundary
    assert "Missing citations: [9]" in boundary
    assert {frame["subject"] for frame in frames} >= {
        "Turn",
        "Draft A",
        "Draft B",
        "Compare drafts",
        "Notes",
        "Summary",
    }
    assert "Project documentation" in boundary
    assert "project_docs" not in boundary
    assert "Unclear" in boundary
    assert "Incomplete" in boundary
    assert "parent agent" in GUIDE
    assert "Grounded" in GUIDE
    assert "Faithful" in GUIDE


def test_back_comparison_highlights_and_branch_cleanup(tmp_path):
    controller = LabelController(fresh(tmp_path, 1))
    answer(controller, "y")
    answer(controller, "2")
    answer(controller, "y")
    answer(controller, "3")
    controller.apply(Action("back"))
    assert controller.step == "lesson_kind"
    assert controller.screen().selected == "3"
    controller.apply(Action("back"))
    answer(controller, "n")
    assert "lesson_kind" not in controller.answers
    controller.apply(Action("back"))
    controller.apply(Action("back"))
    controller.apply(Action("back"))
    assert controller.step == "failed"
    assert controller.screen().selected == "y"
    answer(controller, "i")
    assert "category" not in controller.answers
    answer(controller, "n")
    while controller.step != "preferred":
        answer(controller, controller.screen().options[-1].id)
    controller.apply(Action("back"))
    assert controller.step == "B.store_exact"
    assert controller.screen().selected == "u"
    controller.apply(Action("back"))
    assert controller.step == "B.enforced_by"
    assert controller.screen().selected == "4"
    answer(controller, "1")
    answer(controller, "n")
    answer(controller, "b")  # Option B is distinct from the Back action.
    controller.apply(Action("back"))
    assert controller.step == "preferred"
    assert controller.screen().selected == "b"
    answer(controller, "a")
    answer(controller, "continue")
    answer(controller, "save")
    row = controller.store.read_jsonl(controller.store.labels_path)[0]
    assert (row["failed"], row["incomplete"], row["category"]) == (None, True, None)
    assert row["lesson_kind"] is None
    assert row["preferred"] == controller.drafts[0].drafter


@pytest.mark.parametrize("value", ["q", "b", "p", "s", "?", "[bold]note [/dim]"])
def test_notes_are_literal(tmp_path, value):
    controller = LabelController(fresh(tmp_path, 1, two=False))
    to_summary(controller)
    controller.apply(Action("back"))
    controller.apply(Action("note", value))
    assert controller.screen().selected == "continue"
    answer(controller, "continue")
    answer(controller, "save")
    row = controller.store.read_jsonl(controller.store.labels_path)[0]
    assert row["note"] == value


def test_save_failure_retry_and_redo_preserve_earlier_items(tmp_path, monkeypatch):
    store = fresh(tmp_path, 3, two=False)
    controller = LabelController(store)
    to_summary(controller)
    assert controller.labeled == 0
    assert not store.labels_path.exists()
    answer(controller, "save")
    saved = store.labels_path.read_bytes()
    assert controller.screen().notice == "Saved."
    to_summary(controller)
    original = store.write_jsonl

    def fail(path, rows, **kwargs):
        path.write_text("partial")
        raise OSError("PRIVATE_CONTENT must not reach output")

    monkeypatch.setattr(store, "write_jsonl", fail)
    answer(controller, "save")
    assert controller.step == "summary"
    assert controller.labeled == 1
    assert "Save failed" in controller.screen().notice
    assert "Saved" not in controller.screen().notice
    assert "PRIVATE_CONTENT" not in controller.screen().notice
    assert store.labels_path.read_bytes() == saved
    assert not list(store.root.glob(".labels-*.tmp"))
    monkeypatch.setattr(store, "write_jsonl", original)
    controller.apply(Action("redo"))
    assert controller.step == "failed"
    to_summary(controller)
    answer(controller, "save")
    controller.apply(Action("quit"))
    assert len(store.read_jsonl(store.labels_path)) == 2
    assert (controller.summary().labeled, controller.summary().remaining) == (2, 1)
    assert stat.S_IMODE(store.labels_path.stat().st_mode) == 0o600
    # Lost acknowledgement/next launch still finds committed items, never duplicates.
    resumed = LabelController(store)
    assert len(resumed.todo) == 1


def test_backfill_declines_review_history_and_reports(tmp_path):
    store = synthetic_run(tmp_path, n=4)
    items = store.read_jsonl(store.items_path)
    ids = [item["item_id"] for item in items]
    items[2]["facts"]["tool_error_count"] = 1
    store.write_jsonl(store.items_path, items)
    store.write_jsonl(
        store.drafts_path,
        [
            {"item_id": ids[0], "drafter": "second", "lesson": "Second.", "cited": [1]},
            {"item_id": ids[1], "drafter": "second", "lesson": None},
        ],
        append=True,
    )
    controller = LabelController(store, seed=5)
    asked = []
    while not controller.done:
        asked.append(controller.step)
        if controller.step == "summary":
            value = "save"
        elif controller.step == "preferred":
            value = "same"
        else:
            value = controller.screen().options[0].id
        answer(controller, value)
    assert asked.count("preferred") == 2
    assert "failed" not in asked
    assert len(asked) == 12  # 6 new draft + 2 compare + 1 error + 3 saves.
    assert (controller.summary().labeled, controller.summary().changed) == (3, 0)
    rows = {row["item_id"]: row for row in store.read_jsonl(store.labels_path)}
    assert rows[ids[1]]["drafts"]["second"]["declined"]
    assert rows[ids[0]]["drafts"]["ollama:m"]["grounded"] is False
    for item_id, row in rows.items():
        assert not missing_steps(
            row, order_drafts(item_id, store.drafts_by_item()[item_id], 5), False
        )
    assert LabelController(store).done
    review = LabelController(store, item_ids=[ids[3]], review=True)
    assert review.step == "review"
    answer(review, "keep")
    assert review.labeled == 0
    review = LabelController(store, item_ids=[ids[3]], review=True)
    answer(review, "redo")
    assert review.screen().selected == "y"
    to_summary(review)
    answer(review, "save")
    row = store.read_jsonl(store.labels_path)[3]
    assert row["history"][0]["changed"] == {
        "failed": True,
        "category": "tool_misuse",
        "worth_remembering": True,
        "lesson_kind": "process",
    }
    assert row["labeled_at"] == "2026-01-01T00:00:00+00:00"
    assert "4 value(s) changed on 1 item(s)" in build_report(store)


def test_skip_quit_invalid_actions_and_note_validation(tmp_path):
    controller = LabelController(fresh(tmp_path, 3))
    for invalid in (Action("back"), Action("redo"), Action("answer", "bad"), Action("unknown")):
        with pytest.raises(JudgeError):
            controller.apply(invalid)
    controller.apply(Action("skip"))
    controller.apply(Action("quit"))
    assert controller.summary().skipped == 1
    assert not controller.store.labels_path.exists()
    controller = LabelController(controller.store)
    to_summary(controller)
    controller.apply(Action("back"))
    for invalid in ("\x1b[2J", "two\nlines", "tab\there", "界" * 1400, "\u202eevil", "\ud800"):
        with pytest.raises(JudgeError):
            controller.apply(Action("note", invalid))
    with pytest.raises(JudgeError):
        controller.apply(Action("answer", "edit"))


def test_a_tab_in_a_note_never_reaches_a_screen(tmp_path):
    store = fresh(tmp_path, 1, two=False)
    controller = LabelController(store)
    to_summary(controller)
    controller.apply(Action("back"))
    plain = PlainFrontend(io.StringIO("1\nseen\tin plain mode\n"), io.StringIO())
    typed = plain.exchange(controller.screen())
    assert typed == Action("note", "seen in plain mode")
    controller.apply(typed)
    answer(controller, "continue")
    answer(controller, "save")

    # A note saved by an earlier version can hold a tab, and the Go helper refuses a
    # screen whose note is not one clean line.
    (row,) = store.read_jsonl(store.labels_path)
    store.write_jsonl(store.labels_path, [{**row, "note": "kept\tfrom before"}])
    review = LabelController(store, item_ids=[row["item_id"]], review=True)
    answer(review, "redo")
    assert review.screen().note == "kept from before"
    to_summary(review)
    answer(review, "save")
    assert store.read_jsonl(store.labels_path)[0]["note"] == "kept\tfrom before"


def frame(**overrides):
    return encode_frame(
        {
            "version": 2,
            "request_id": 3,
            "type": "action",
            "action": "answer",
            "value": "y",
            **overrides,
        }
    )


@pytest.mark.parametrize(
    "data",
    [
        b"",
        b"{}",
        b"not json\n",
        frame(version=1),
        frame(version=True),
        frame(request_id=2),
        frame(request_id=True),
        frame(action="evil"),
        frame(value=2),
        frame(extra="hidden"),
        b'{"version":1,"version":1}\n',
        b"\xff\n",
        b"[]\n",
        b"[" * 2000 + b"]" * 2000 + b"\n",
        frame().replace(b'"y"', b'"\\ud800"'),
    ],
)
def test_reject_protocol_without_echoing_payload(data):
    with pytest.raises(JudgeError) as error:
        read_action(io.BytesIO(data), 3)
    assert "not json" not in str(error.value)


def test_frame_limit_and_one_action_per_request(monkeypatch):
    assert read_action(io.BytesIO(frame()), 3) == Action("answer", "y")
    monkeypatch.setattr(label_frontend, "MAX_FRAME", 10)
    with pytest.raises(JudgeError, match="limit"):
        encode_frame({"text": "x" * 11})
    with pytest.raises(JudgeError, match="size"):
        read_action(io.BytesIO(b"x" * 11 + b"\n"), 3)


def test_plain_commands_notes_brackets_and_control_safety(tmp_path):
    controller = LabelController(fresh(tmp_path, 1, two=False))
    out = io.StringIO()
    plain = PlainFrontend(io.StringIO("evidence\nhelp\ninvalid\n" + "9" * 5000 + "\n2\n"), out)
    assert plain.exchange(controller.screen()) == Action("answer", "n")
    assert "Labeling guide" in out.getvalue()
    assert "\x1b" not in out.getvalue()
    to_summary(controller)
    controller.apply(Action("back"))
    plain = PlainFrontend(io.StringIO("1\nq\n"), out)
    assert plain.exchange(controller.screen()) == Action("note", "q")
    plain = PlainFrontend(io.StringIO("1\n/cancel\n2\n"), out)
    assert plain.exchange(controller.screen()) == Action("answer", "continue")
    assert display_safe("[bold]x [/dim]") == "[bold]x [/dim]"
    assert display_safe("界 café 👩‍💻") == "界 café 👩‍💻"
    assert display_safe("\x1b]52;c;attack\x07\r\u009b2J\u202eevil") == (
        "\\u001b]52;c;attack\\u0007\\u000d\\u009b2J\\u202eevil"
    )


def test_protocol_two_and_plain_claim_focus(tmp_path):
    store = fresh(tmp_path, 1, two=False)
    item_id = store.read_jsonl(store.items_path)[0]["item_id"]
    store.write_jsonl(
        store.drafts_path,
        [
            structured_draft(
                item_id,
                claims=[
                    {"text": "First claim.", "cited": [1]},
                    {"text": "Second claim.", "cited": [2]},
                ],
            ).to_dict()
        ],
    )
    controller = LabelController(store)
    answer(controller, "n")
    answer(controller, "n")
    answer(controller, "y")
    answer(controller, "y")
    screen = controller.screen()
    assert controller.step == "A.c2.grounded"
    wire = json.loads(
        encode_frame(
            {
                "version": label_build.PROTOCOL_VERSION,
                "request_id": 1,
                "type": "screen",
                "screen": screen.payload(),
            }
        )
    )
    assert wire["version"] == 2
    assert wire["screen"]["drafts"][0]["focus"] == "Claim 2 of 2"
    output = io.StringIO()
    plain = PlainFrontend(io.StringIO("2\n"), output)
    assert plain.exchange(screen) == Action("answer", "n")
    text = output.getvalue()
    assert "Draft A\nClaim 2 of 2\nRule:" in text
    assert "▶ 2. Second claim. [2]" in text
    assert "Cited evidence\n[2] TOOL RESULT (error): 3 failed" in text
    assert "[1] REQUEST" not in text
    assert "\x1b" not in text


def many_drafts(tmp_path: Path, count: int):
    store = fresh(tmp_path, 1, two=False)
    item_id = store.read_jsonl(store.items_path)[0]["item_id"]
    store.write_jsonl(
        store.drafts_path,
        [
            structured_draft(
                item_id,
                drafter=f"PRIVATE_DRAFTER_{i}",
                claims=[
                    {"text": "First claim.", "cited": [1]},
                    {"text": "Second claim.", "cited": [2]},
                ],
            ).to_dict()
            for i in range(count)
        ],
    )
    return store


def selected_answer(controller: LabelController) -> str:
    return {
        "failed": "n",
        "worth_remembering": "n",
        "preferred": controller.drafts[-1].slot.lower(),
        "note": "continue",
        "summary": "save",
    }.get(controller.step, controller.screen().options[0].id)


@pytest.mark.parametrize("count", [3, 4])
def test_three_four_drafts_plain_navigation_and_blinded_frames(tmp_path, count):
    store = many_drafts(tmp_path, count)
    controller = LabelController(store)
    frames, rendered = [], []
    while controller.step != "summary":
        screen = controller.screen()
        frames.append(screen.payload())
        selected = selected_answer(controller)
        index = next(i for i, option in enumerate(screen.options, 1) if option.id == selected)
        output = io.StringIO()
        action = PlainFrontend(io.StringIO(f"{index}\n"), output).exchange(screen)
        assert action == Action("answer", selected)
        rendered.append(output.getvalue())
        if ".c2." in controller.step:
            assert screen.drafts[0].focus == "Claim 2 of 2"
            assert "[2] TOOL RESULT" in screen.drafts[0].cited_evidence
            assert "[1] REQUEST" not in screen.drafts[0].cited_evidence
        controller.apply(action)
    controller.apply(Action("back"))
    controller.apply(Action("back"))
    assert controller.step == "preferred"
    assert controller.screen().selected == SLOTS[count - 1].lower()
    options = {option.id: option.label for option in controller.screen().options}
    assert options["same"] == "All equally"
    assert options["neither"] == "None"
    answer(controller, selected_answer(controller))
    answer(controller, "continue")
    frames.append(controller.screen().payload())
    assert {frame["subject"] for frame in frames} >= {
        *(f"Draft {slot}" for slot in SLOTS[:count]),
        "Compare drafts",
        "Summary",
    }
    for frame in frames:
        if frame["subject"] in ("Compare drafts", "Notes", "Summary"):
            assert [draft["slot"] for draft in frame["drafts"]] == list(SLOTS[:count])
    boundary = json.dumps(frames) + "".join(rendered)
    for hidden in ("PRIVATE_DRAFTER", "draft_hash", "model_snapshot", "probabilities"):
        assert hidden not in boundary
    for slot in SLOTS[:count]:
        assert f"Draft {slot}\nClaim 2 of 2" in "".join(rendered)
    expected = controller.drafts[-1].drafter
    answer(controller, "save")
    row = store.read_jsonl(store.labels_path)[0]
    assert len(row["drafts"]) == count
    assert row["preferred"] == expected
    assert LabelController(store).done


def test_order_drafts_refuses_overflow_instead_of_truncating():
    rows = [{"drafter": f"PRIVATE_DRAFTER_{i}", "lesson": None} for i in range(5)]
    with pytest.raises(JudgeError, match="at most 4 drafts") as error:
        order_drafts("private item", rows, 1)
    assert "PRIVATE" not in str(error.value)
    assert "private item" not in str(error.value)


@pytest.mark.parametrize("plain", [False, True])
def test_draft_overflow_fails_before_opening_either_frontend(tmp_path, monkeypatch, plain):
    store = fresh(tmp_path, 2, two=False)
    items = store.read_jsonl(store.items_path)
    ids = [item["item_id"] for item in items]
    store.write_jsonl(
        store.drafts_path,
        [{"item_id": ids[1], "drafter": f"extra{i}", "lesson": None} for i in range(4)],
        append=True,
    )
    before = store.drafts_path.read_bytes()

    def unexpected(*args, **kwargs):
        pytest.fail("all selected items must be checked before opening a frontend")

    monkeypatch.setattr(label_frontend, "automatic_plain", lambda: False)
    monkeypatch.setattr(label_frontend, "frontend_binary", unexpected)
    monkeypatch.setattr(label_frontend, "PlainFrontend", unexpected)
    with pytest.raises(JudgeError, match="at most 4 drafts"):
        label_frontend.label_session(store, plain=plain, item_ids=tuple(ids))
    assert store.drafts_path.read_bytes() == before
    assert not store.labels_path.exists()


@pytest.mark.parametrize("slots", [[""], ["AB"], ["E"], ["a"], ["A", "A"], list("ABCDE")])
def test_screen_producer_rejects_invalid_draft_slots(tmp_path, slots):
    screen = LabelController(fresh(tmp_path, 1, two=False)).screen()
    drafts = [label_controller.DisplayDraft(slot, "lesson", "evidence") for slot in slots]
    with pytest.raises(JudgeError):
        replace(screen, drafts=drafts)


def test_screen_producer_rejects_unknown_draft_subject(tmp_path):
    screen = LabelController(fresh(tmp_path, 1, two=False)).screen()
    with pytest.raises(JudgeError):
        replace(screen, subject="Draft E")


@pytest.mark.parametrize("count", [3, 4])
@pytest.mark.parametrize("plain", [False, True])
def test_real_plain_entrypoint_three_four_drafts(tmp_path, count, plain):
    store = many_drafts(tmp_path, count)
    controller = LabelController(store)
    inputs = []
    while not controller.done:
        screen = controller.screen()
        selected = selected_answer(controller)
        inputs.append(str(next(i for i, o in enumerate(screen.options, 1) if o.id == selected)))
        answer(controller, selected)
    expected = store.read_jsonl(store.labels_path)[0]
    store.labels_path.unlink()
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from agent_memory.cli import main; sys.exit(main(sys.argv[1:]))",
            "--home",
            str(store.cfg.home),
            "judge",
            "label",
            "--run",
            "syn",
            *(["--plain"] if plain else []),
        ],
        input="\n".join(inputs) + "\n",
        text=True,
        capture_output=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr
    assert "Labeled 1" in result.stdout
    assert "\x1b" not in result.stdout
    for slot in SLOTS[:count]:
        assert f"Draft {slot}\nClaim 2 of 2" in result.stdout
    actual = store.read_jsonl(store.labels_path)[0]
    assert actual["drafts"] == expected["drafts"]
    assert actual["preferred"] == expected["preferred"]


@pytest.mark.parametrize("mode", ["--plain", "pipe", "dumb"])
def test_real_plain_entrypoint(tmp_path, mode):
    store = fresh(tmp_path, 1, two=False)
    env = {**os.environ, "TERM": "dumb" if mode == "dumb" else "xterm-256color"}
    env.pop("FORCE_COLOR", None)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from agent_memory.cli import main; sys.exit(main(sys.argv[1:]))",
            "--home",
            str(store.cfg.home),
            "judge",
            "label",
            "--run",
            "syn",
            *(["--plain"] if mode == "--plain" else []),
        ],
        input="2\n2\n1\n1\n1\n2\n1\n2\n2\n1\n",
        text=True,
        capture_output=True,
        env=env,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr
    assert "Labeled 1" in result.stdout
    assert "\x1b" not in result.stdout + result.stderr
    assert store.read_jsonl(store.labels_path)[0]["drafts"]["ollama:m"]["generalizable"] == 1


@pytest.fixture(scope="module")
def real_go_binary(tmp_path_factory):
    if label_build.shutil.which("go") is None:
        pytest.skip("real label UI round trip needs Go")
    return label_build.frontend_binary(root=tmp_path_factory.mktemp("label-ui") / "cache")


@pytest.mark.parametrize("count", [3, 4])
@pytest.mark.parametrize("stage", ["claim", "comparison", "rule_supported", "store_exact"])
def test_real_go_entrypoint_three_four_drafts(tmp_path, real_go_binary, count, stage):
    """Real Python transport, Go main, and TTY input; no test-only Go protocol."""
    controller = LabelController(many_drafts(tmp_path, count))
    key = "c2.grounded" if stage == "claim" else stage
    target = "preferred" if stage == "comparison" else f"{SLOTS[count - 1]}.{key}"
    while controller.step != target:
        answer(controller, selected_answer(controller))
    screen = controller.screen()
    if stage == "comparison":
        screen = replace(screen, selected=SLOTS[count - 1].lower())
    elif stage in ("rule_supported", "store_exact"):
        screen = replace(screen, selected="u")
    path = tmp_path / "screen.json"
    path.write_text(json.dumps(screen.payload()))
    body = """
import fcntl, json, sys, termios
from pathlib import Path
from agent_memory.judge.label_controller import Action, DisplayDraft, Screen
from agent_memory.judge.label_frontend import GoFrontend
from agent_memory.judge.label_questions import Option
fcntl.ioctl(sys.stdin.fileno(), termios.TIOCSCTTY, 0)
data = json.loads(Path(sys.argv[2]).read_text())
data["drafts"] = [DisplayDraft(**d) for d in data["drafts"]]
data["options"] = [Option(**o) for o in data["options"]]
screen = Screen(**data)
frontend = GoFrontend(Path(sys.argv[1]))
try:
    assert frontend.exchange(screen) == Action("answer", screen.selected)
    frontend.finish("Finished synthetic round trip.")
finally:
    frontend.close()
print("ROUNDTRIP_OK", flush=True)
"""
    master, slave = pty.openpty()
    fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", 40, 100, 0, 0))
    process = subprocess.Popen(
        [sys.executable, "-c", body, str(real_go_binary), str(path)],
        stdin=slave,
        stdout=slave,
        stderr=slave,
        start_new_session=True,
        env={**os.environ, "TERM": "xterm-256color", "NO_COLOR": "1"},
    )
    os.close(slave)
    output, submitted = bytearray(), False
    deadline = time.monotonic() + 15
    try:
        while time.monotonic() < deadline:
            if not select.select([master], [], [], 0.2)[0]:
                if process.poll() is not None:
                    break
                continue
            try:
                part = os.read(master, 65536)
            except OSError as exc:
                if exc.errno != errno.EIO:
                    raise
                break  # Linux reports the last PTY close as EIO.
            if not part:
                break
            output.extend(part)
            marker = "overgeneralization?" if stage == "rule_supported" else screen.question
            if not submitted and marker.encode() in output:
                os.write(master, b"\r")
                submitted = True
        assert submitted, output.decode(errors="replace")
        assert process.wait(timeout=3) == 0, output.decode(errors="replace")
        assert b"ROUNDTRIP_OK" in output
        assert screen.subject.encode() in output
        if stage == "claim":
            assert b"Claim 2 of 2" in output
        elif stage in ("rule_supported", "store_exact"):
            assert b"Rule:" in output
            assert b"Unclear" in output
        assert b"PRIVATE_DRAFTER" not in output
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=3)
        os.close(master)


def fake_go(tmp_path: Path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    for name in ("main.go", "go.mod", "go.sum"):
        (source / name).write_text(name)
    counter = tmp_path / "builds"
    go = tmp_path / "go"
    go.write_text(
        "#!" + sys.executable + "\n"
        "import pathlib, sys, time\n"
        "if sys.argv[1] == 'version': print('go version go1.test test/arch')\n"
        "else:\n"
        f" p=pathlib.Path({str(counter)!r})\n"
        " with p.open('a') as f: f.write('build\\n')\n"
        " time.sleep(.1)\n"
        " pathlib.Path(sys.argv[sys.argv.index('-o')+1]).write_text('binary')\n"
    )
    go.chmod(0o700)
    monkeypatch.setattr(label_build.shutil, "which", lambda _: str(go))
    return source, counter


def test_cold_cache_concurrent_launch_and_source_invalidation(tmp_path, monkeypatch):
    source, counter = fake_go(tmp_path, monkeypatch)
    root = tmp_path / "cache" / "ui"
    progress = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        paths = list(
            pool.map(
                lambda _: label_build.frontend_binary(
                    source=source, root=root, progress=progress.append
                ),
                range(4),
            )
        )
    assert len(set(paths)) == 1
    assert counter.read_text() == "build\n"
    assert len(progress) == 1
    assert stat.S_IMODE(paths[0].stat().st_mode) == 0o700
    label_build.frontend_binary(source=source, root=root)
    assert counter.read_text() == "build\n"
    (source / "main.go").write_text("changed")
    assert label_build.frontend_binary(source=source, root=root) != paths[0]
    assert counter.read_text() == "build\nbuild\n"
    before = label_build.source_key(source, "go1")
    monkeypatch.setattr(label_build, "PROTOCOL_VERSION", 3)
    assert label_build.source_key(source, "go1") != before
    assert label_build.source_key(source, "go2") != before


def test_missing_go_build_failure_and_empty_run(tmp_path, monkeypatch):
    monkeypatch.setattr(label_build.shutil, "which", lambda _: None)
    with pytest.raises(JudgeError, match="--plain"):
        label_build.frontend_binary()
    store = synthetic_run(tmp_path, n=1)
    assert label_frontend.label_session(store).labeled == 0
    source, _ = fake_go(tmp_path, monkeypatch)
    monkeypatch.setattr(
        label_build.subprocess,
        "run",
        lambda *a, **k: (_ for _ in ()).throw(subprocess.CalledProcessError(1, "go")),
    )
    with pytest.raises(JudgeError, match="--plain"):
        label_build.frontend_binary(source=source, root=tmp_path / "cache" / "ui")


def test_atomic_replace_failure_keeps_old_labels(tmp_path, monkeypatch):
    store = synthetic_run(tmp_path, n=1)
    before = store.labels_path.read_bytes()
    monkeypatch.setattr(
        label_controller.os, "replace", lambda *args: (_ for _ in ()).throw(OSError("disk full"))
    )
    with pytest.raises(OSError, match="disk full"):
        label_controller.atomic_labels(store, [])
    assert store.labels_path.read_bytes() == before


def helper_script(tmp_path: Path, body: str) -> Path:
    binary = tmp_path / "helper"
    binary.write_text(f"#!{sys.executable}\n{body}\n")
    binary.chmod(0o700)
    return binary


@pytest.mark.parametrize("response", ["", "malformed", "stale", "duplicate"])
def test_transport_crashes_and_out_of_order_actions(tmp_path, response):
    body = (
        "import json, sys\n"
        "frame=json.loads(sys.stdin.readline())\n"
        "rid=frame['request_id']\n"
        "action={'version':2,'request_id':rid,'type':'action','action':'answer','value':'n'}\n"
    )
    if response == "malformed":
        body += "print('PRIVATE_BAD_FRAME', flush=True)\n"
    elif response == "stale":
        body += "action['request_id']=99\nprint(json.dumps(action), flush=True)\n"
    elif response == "duplicate":
        body += (
            "print(json.dumps(action), flush=True)\n"
            "print(json.dumps(action), flush=True)\n"
            "sys.stdin.readline()\n"
        )
    frontend = label_frontend.GoFrontend(helper_script(tmp_path, body))
    screen = LabelController(fresh(tmp_path / "run", 1)).screen()
    try:
        if response == "duplicate":
            assert frontend.exchange(screen) == Action("answer", "n")
        with pytest.raises(JudgeError) as error:
            frontend.exchange(screen)
        assert "PRIVATE_BAD_FRAME" not in str(error.value)
    finally:
        frontend.close()
    assert frontend.process.poll() is not None


def test_transport_closes_when_frontend_ignores_eof(tmp_path):
    binary = helper_script(tmp_path, "import time\nwhile True: time.sleep(1)")
    frontend = label_frontend.GoFrontend(binary)
    frontend.close()
    assert frontend.process.poll() is not None


def test_no_silent_fallback_on_build_failure(tmp_path, monkeypatch):
    store = fresh(tmp_path, 1)
    monkeypatch.setattr(label_frontend, "automatic_plain", lambda: False)
    monkeypatch.setattr(
        label_frontend,
        "frontend_binary",
        lambda **kwargs: (_ for _ in ()).throw(JudgeError("Missing Go. Use --plain.")),
    )
    with pytest.raises(JudgeError, match="--plain"):
        label_frontend.label_session(store)
    assert not store.labels_path.exists()


def test_cache_refuses_symlink_and_preserves_dependency_files(tmp_path, monkeypatch):
    source, _ = fake_go(tmp_path, monkeypatch)
    before = {path.name: path.read_bytes() for path in source.iterdir()}
    root = tmp_path / "cache" / "ui"
    root.parent.mkdir()
    target = tmp_path / "target"
    target.mkdir()
    root.symlink_to(target, target_is_directory=True)
    with pytest.raises(JudgeError, match="Unsafe"):
        label_build.frontend_binary(source=source, root=root)
    assert {path.name: path.read_bytes() for path in source.iterdir()} == before


@pytest.mark.parametrize("when", ["before_save", "after_save", "interrupt"])
def test_session_cleanup_around_save_acknowledgement(tmp_path, monkeypatch, capsys, when):
    store = fresh(tmp_path, 2, two=False)
    closed = []

    class FailingFrontend:
        def __init__(self, _binary):
            pass

        def exchange(self, screen):
            if when == "before_save" and screen.key.endswith(":summary"):
                raise JudgeError("Frontend disconnected before save.")
            if when == "after_save" and screen.notice == "Saved.":
                raise JudgeError("Frontend disconnected after save.")
            if when == "interrupt":
                raise KeyboardInterrupt
            key = screen.key.partition(":")[2]
            value = {
                "failed": "n",
                "worth_remembering": "n",
                "note": "continue",
                "summary": "save",
            }.get(key, screen.selected)
            return Action("answer", value)

        def finish(self, message):
            raise AssertionError("should not reach completion")

        def close(self):
            closed.append(True)

    monkeypatch.setattr(label_frontend, "GoFrontend", FailingFrontend)
    monkeypatch.setattr(label_frontend, "automatic_plain", lambda: False)
    monkeypatch.setattr(label_frontend, "frontend_binary", lambda **kwargs: tmp_path / "unused")
    if when == "interrupt":
        assert label_frontend.label_session(store).labeled == 0
        assert "Stopped" in capsys.readouterr().out
    else:
        with pytest.raises(JudgeError, match="disconnected"):
            label_frontend.label_session(store)
        assert "Saved" not in capsys.readouterr().out
    assert closed == [True]
    assert len(store.read_jsonl(store.labels_path)) == (1 if when == "after_save" else 0)
