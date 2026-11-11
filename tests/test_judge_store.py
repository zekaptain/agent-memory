"""Crash recovery for run files, including human labels and verdict rewrites."""

import os
import stat

import pytest

from agent_memory.judge import pipeline
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
