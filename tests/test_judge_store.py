"""Crash recovery for run files, including human labels and verdict rewrites."""

import os
import stat
from concurrent.futures import ThreadPoolExecutor

import pytest

from agent_memory.judge import pipeline
from agent_memory.judge.report import build_report
from agent_memory.judge.store import JudgeError, RunStore
from tests.test_judge import synthetic_run


@pytest.mark.parametrize("tail", [b'{"private":', b'{"private":"\xe2\x82'])
def test_read_jsonl_drops_torn_final_line(tmp_path, tail):
    path = tmp_path / "labels.jsonl"
    path.write_bytes(b'{"item_id":"finished"}\n' + tail)
    assert RunStore.read_jsonl(path) == [{"item_id": "finished"}]


@pytest.mark.parametrize("suffix", [b"\n", b'\n{"item_id":"later"}\n'])
def test_read_jsonl_reports_corrupt_line_without_content(tmp_path, suffix):
    path = tmp_path / "labels.jsonl"
    path.write_bytes(b'{"item_id":"finished"}\nprivate-invalid-json' + suffix)
    with pytest.raises(JudgeError) as error:
        RunStore.read_jsonl(path)
    assert str(path) in str(error.value)
    assert "line 2" in str(error.value)
    assert "private-invalid-json" not in str(error.value)


@pytest.mark.parametrize("filename", ["labels.jsonl", "verdicts.jsonl"])
def test_failed_rewrite_preserves_original(tmp_path, monkeypatch, filename):
    path = tmp_path / filename
    RunStore.write_jsonl(path, [{"item_id": "original"}])
    original = path.read_bytes()

    def crash(*args):
        raise OSError("simulated crash")

    monkeypatch.setattr(os, "replace", crash)
    with pytest.raises(OSError, match="simulated crash"):
        RunStore.write_jsonl(path, [{"item_id": "replacement"}])
    assert path.read_bytes() == original


@pytest.mark.parametrize("finished", [b"", b'{"item_id":"finished"}\n'], ids=["alone", "after"])
@pytest.mark.parametrize(
    "tail",
    [b'{"private":', b'{"private":"\xe2\x82', b'{"private":"' + b"x" * 200_000],
    ids=["cut", "cut inside a character", "longer than a block"],
)
def test_append_drops_a_torn_final_line_instead_of_joining_it(tmp_path, finished, tail):
    path = tmp_path / "verdicts.jsonl"
    path.write_bytes(finished + tail)
    before = RunStore.read_jsonl(path)
    assert RunStore.write_jsonl(path, [{"item_id": "next"}], append=True) == 1
    assert RunStore.read_jsonl(path) == [*before, {"item_id": "next"}]
    assert b"private" not in path.read_bytes()


def test_append_keeps_a_whole_final_row_that_lacks_its_line_ending(tmp_path):
    path = tmp_path / "verdicts.jsonl"
    path.write_bytes(b'{"item_id":"finished"}\n{"item_id":"whole"}')
    RunStore.write_jsonl(path, [{"item_id": "next"}], append=True)
    assert [row["item_id"] for row in RunStore.read_jsonl(path)] == ["finished", "whole", "next"]


def test_decide_resumes_after_a_verdict_row_was_cut_off(tmp_path):
    store = synthetic_run(tmp_path, n=3)
    path = store.verdicts_path("rules")
    before = [row["item_id"] for row in store.read_jsonl(path)]
    data = path.read_bytes()
    path.write_bytes(data[:-40])  # The run was killed while it appended its last row.
    assert [row["item_id"] for row in store.read_jsonl(path)] == before[:-1]

    written = pipeline.decide_items(
        store.cfg, store.run_id, ["rules"], store=store, record_events=False
    )

    assert written == {"rules": 1}
    assert [row["item_id"] for row in store.read_jsonl(path)] == before


def test_jsonl_rewrite_and_append(tmp_path):
    path = tmp_path / "labels.jsonl"
    assert RunStore.write_jsonl(path, [{"n": 1}, {"n": 2}]) == 2
    assert RunStore.write_jsonl(path, [{"n": 3}]) == 1
    assert RunStore.write_jsonl(path, [{"n": 4}], append=True) == 1
    assert RunStore.read_jsonl(path) == [{"n": 3}, {"n": 4}]
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_updates_of_a_run_made_at_once_all_land(tmp_path):
    store = synthetic_run(tmp_path, n=1)
    with ThreadPoolExecutor(8) as pool:
        list(pool.map(lambda i: store.update_run(**{f"field_{i}": i}), range(40)))
    run = store.read_run()
    assert [run.get(f"field_{i}") for i in range(40)] == list(range(40))


def test_run_files_are_private(tmp_path):
    store = synthetic_run(tmp_path, n=1)
    store.update_run(note="changed")
    build_report(store)
    RunStore.write_jsonl(store.root / "appended.jsonl", [{"n": 1}], append=True)
    for path in [store.root, *store.root.rglob("*")]:
        assert stat.S_IMODE(path.stat().st_mode) == (0o700 if path.is_dir() else 0o600), path


def test_a_damaged_run_json_is_reported_without_its_content(tmp_path):
    store = synthetic_run(tmp_path, n=1)
    store.run_json.write_text('{"PRIVATE_CONTENT":')
    with pytest.raises(JudgeError) as error:
        store.read_run()
    assert str(store.run_json) in str(error.value)
    assert "PRIVATE_CONTENT" not in str(error.value)


def test_a_rewrite_reaches_the_disk_before_it_replaces_the_file(tmp_path, monkeypatch):
    order = []
    fsync, replace = os.fsync, os.replace
    monkeypatch.setattr(os, "fsync", lambda fd: (order.append("fsync"), fsync(fd))[1])
    monkeypatch.setattr(os, "replace", lambda a, b: (order.append("replace"), replace(a, b))[1])
    RunStore.write_jsonl(tmp_path / "labels.jsonl", [{"item_id": "a"}])
    assert order == ["fsync", "replace"]
    assert [path.name for path in tmp_path.iterdir()] == ["labels.jsonl"]


def test_a_label_that_names_no_item_is_reported(tmp_path):
    store = synthetic_run(tmp_path, n=1)
    store.write_jsonl(store.labels_path, [{"failed": True, "note": "PRIVATE_CONTENT"}])
    with pytest.raises(JudgeError, match="names no item") as error:
        store.labels()
    assert "PRIVATE_CONTENT" not in str(error.value)


def test_verdict_files_and_live_days_are_listed_in_name_order(tmp_path):
    store = synthetic_run(tmp_path, n=2)
    assert list(store.verdicts_by_file()) == ["llm_ref", "rules"]
    assert store.verdicts_by_file()["llm_ref"] == store.verdicts("llm:ref")
    assert len(store.verdicts("rules")) == 2
    # A file is read whatever it is called, as the copy a person leaves beside one always was.
    copy = store.verdicts_dir / "rules copy.jsonl"
    copy.write_bytes(store.verdicts_path("rules").read_bytes())
    assert list(store.verdicts_by_file()) == ["llm_ref", "rules copy", "rules"]
    assert store.verdicts_by_file()["rules copy"] == store.verdicts("rules")
    copy.unlink()
    assert store.live_days() == []
    for day in ("2026-11-02", "2026-11-01"):
        store.live_day(day).create({"live_day": day})
    (store.live_dir / "stray.txt").write_text("not a day")
    assert [day.root.name for day in store.live_days()] == ["2026-11-01", "2026-11-02"]
    assert RunStore(store.cfg, "none-yet").verdicts_by_file() == {}
