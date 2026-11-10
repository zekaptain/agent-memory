"""Crash recovery for run files, including human labels and verdict rewrites."""

import os
import stat

import pytest

from agent_memory.judge.store import JudgeError, RunStore


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


def test_jsonl_rewrite_and_append(tmp_path):
    path = tmp_path / "labels.jsonl"
    assert RunStore.write_jsonl(path, [{"n": 1}, {"n": 2}]) == 2
    assert RunStore.write_jsonl(path, [{"n": 3}]) == 1
    assert RunStore.write_jsonl(path, [{"n": 4}], append=True) == 1
    assert RunStore.read_jsonl(path) == [{"n": 3}, {"n": 4}]
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
