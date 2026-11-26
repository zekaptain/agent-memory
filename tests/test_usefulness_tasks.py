"""Tests for usefulness experiment run files, task schema, manifest, split, and hidden tests."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from agent_memory.config import DEFAULT_HOME, Config
from agent_memory.usefulness import (
    TASK_MANIFEST_SCHEMA_VERSION,
    UsefulnessError,
    UsefulnessRunStore,
    assert_hidden_tests_absent,
    build_task_manifest,
    ensure_experiment_home,
    install_hidden_tests,
    load_task,
    load_verified_task,
    sha256_file,
    split_tasks,
    validate_split,
    verify_task_against_manifest,
)


def _write_task(
    tasks_dir: Path,
    task_id: str,
    *,
    repository: str = "/tmp/example-repo",
    commit: str = "a1b2c3d4e5f60718293a4b5c6d7e8f9012345678",
    project: str = "agent-memory",
    work_done_at: str = "2026-11-10T12:00:00Z",
    task_family: str = "fam-a",
    strata: list[str] | None = None,
    checks: dict[str, Any] | None = None,
    request: str | None = "Fix the broken pagination offset.",
    required_behavior: str | None = "- Returns page 2 without repeating the last row of page 1.\n",
    hidden_tests: dict[str, str] | None = None,
    extra_meta: dict[str, Any] | None = None,
    omit_meta_keys: tuple[str, ...] = (),
) -> Path:
    task_dir = tasks_dir / task_id
    task_dir.mkdir(parents=True, exist_ok=True)
    meta: dict[str, Any] = {
        "task_id": task_id,
        "repository": repository,
        "commit": commit,
        "project": project,
        "work_done_at": work_done_at,
        "task_family": task_family,
        "strata": ["ordinary"] if strata is None else strata,
        "checks": (
            {"commands": [{"name": "pytest", "command": ["uv", "run", "pytest", "-q"]}]}
            if checks is None
            else checks
        ),
    }
    if extra_meta:
        meta.update(extra_meta)
    for key in omit_meta_keys:
        meta.pop(key, None)
    (task_dir / "task.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    if request is not None:
        (task_dir / "request.md").write_text(request, encoding="utf-8")
    if required_behavior is not None:
        (task_dir / "required_behavior.md").write_text(required_behavior, encoding="utf-8")
    if hidden_tests:
        for rel, body in hidden_tests.items():
            dest = task_dir / "hidden_tests" / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(body, encoding="utf-8")
    return task_dir


# ---------------------------------------------------------------------------
# Run store and experiment-home guard
# ---------------------------------------------------------------------------


def test_store_refuses_default_home_and_enforces_write_once_and_append_only(
    tmp_path: Path,
) -> None:
    with pytest.raises(UsefulnessError, match="default home"):
        ensure_experiment_home(DEFAULT_HOME)
    with pytest.raises(UsefulnessError, match="default home"):
        UsefulnessRunStore(Config(home=DEFAULT_HOME), "run-1")

    exp_home = tmp_path / "exp-home"
    store = UsefulnessRunStore(exp_home, "run-1")
    digest = store.write_task_manifest({"schema_version": 1, "tasks": {}})
    assert digest == store.task_manifest_sha256()
    assert store.read_task_manifest() == {"schema_version": 1, "tasks": {}}

    # Write-once: second write is refused and original bytes remain untouched.
    with pytest.raises(UsefulnessError, match="already exists"):
        store.write_task_manifest({"schema_version": 1, "tasks": {"tampered": True}})
    assert store.task_manifest_sha256() == digest

    # Deviations are append-only and keep earlier entries unchanged.
    d1 = store.append_deviation("harness upgraded", date="2026-11-25T10:00:00+00:00")
    d2 = store.append_deviation("pair invalidated under rule 2", metadata={"pair_id": "p3"})
    rows = store.deviations()
    assert rows == [d1, d2]

    with pytest.raises(UsefulnessError, match="bad run id"):
        UsefulnessRunStore(exp_home, "../escape")


# ---------------------------------------------------------------------------
# Task schema validation (refuses any missing part and names it)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("omit_key", "expected_name"),
    [
        ("repository", "repository"),
        ("commit", "commit"),
        ("project", "project"),
        ("work_done_at", "work_done_at"),
        ("task_family", "task_family"),
        ("strata", "strata"),
        ("checks", "checks"),
    ],
)
def test_load_task_refuses_missing_metadata_part_and_names_it(
    tmp_path: Path,
    omit_key: str,
    expected_name: str,
) -> None:
    task_dir = _write_task(tmp_path / "tasks", "t1", omit_meta_keys=(omit_key,))
    with pytest.raises(UsefulnessError, match=expected_name):
        load_task(task_dir)


def test_load_task_refuses_missing_files_and_invalid_fields(tmp_path: Path) -> None:
    tasks_dir = tmp_path / "tasks"

    # Missing task.json.
    bare = tasks_dir / "no-meta"
    bare.mkdir(parents=True)
    with pytest.raises(UsefulnessError, match=r"task\.json"):
        load_task(bare)

    # Missing request.md / blank request.md.
    no_req = _write_task(tasks_dir, "no-req", request=None)
    with pytest.raises(UsefulnessError, match="request"):
        load_task(no_req)
    blank_req = _write_task(tasks_dir, "blank-req", request="   \n")
    with pytest.raises(UsefulnessError, match="request"):
        load_task(blank_req)

    # Missing required_behavior.md.
    no_beh = _write_task(tasks_dir, "no-beh", required_behavior=None)
    with pytest.raises(UsefulnessError, match="required_behavior"):
        load_task(no_beh)

    # Empty checks.commands or missing command name/argv.
    empty_cmds = _write_task(tasks_dir, "empty-cmds", checks={"commands": []})
    with pytest.raises(UsefulnessError, match=r"checks\.commands"):
        load_task(empty_cmds)
    no_cmd_name = _write_task(
        tasks_dir,
        "no-cmd-name",
        checks={"commands": [{"command": ["pytest"]}]},
    )
    with pytest.raises(UsefulnessError, match="name"):
        load_task(no_cmd_name)

    # Floating ref instead of fixed hex commit SHA.
    floating = _write_task(tasks_dir, "floating", commit="main")
    with pytest.raises(UsefulnessError, match="commit"):
        load_task(floating)

    # Naive timestamp without timezone.
    naive_ts = _write_task(tasks_dir, "naive-ts", work_done_at="2026-11-10T12:00:00")
    with pytest.raises(UsefulnessError, match="timezone"):
        load_task(naive_ts)

    # Mismatched task_id inside task.json.
    mismatch = _write_task(tasks_dir, "dir-name", extra_meta={"task_id": "other-name"})
    with pytest.raises(UsefulnessError, match="does not match"):
        load_task(mismatch)

    # Symlink inside hidden_tests is refused.
    sym_task = _write_task(tasks_dir, "sym-task")
    outside = tmp_path / "secret.txt"
    outside.write_text("nope", encoding="utf-8")
    hidden_dir = sym_task / "hidden_tests"
    hidden_dir.mkdir()
    os.symlink(outside, hidden_dir / "test_escape.py")
    with pytest.raises(UsefulnessError, match="symlink"):
        load_task(sym_task)


# ---------------------------------------------------------------------------
# Chronological + family-disjoint split
# ---------------------------------------------------------------------------


def test_split_puts_calibration_before_held_out_and_keeps_families_disjoint(
    tmp_path: Path,
) -> None:
    tasks_dir = tmp_path / "tasks"
    # Family A: Nov 1 and Nov 3; Family B: Nov 2 and Nov 4 (overlaps A!);
    # Family C: Nov 5; Family D: Nov 6.
    # The ONLY valid boundary between A/B and C/D is after Nov 4 (cannot split A from B).
    t1 = load_task(_write_task(tasks_dir, "t1", task_family="fam-a", work_done_at="2026-11-01"))
    t2 = load_task(_write_task(tasks_dir, "t2", task_family="fam-b", work_done_at="2026-11-02"))
    t3 = load_task(_write_task(tasks_dir, "t3", task_family="fam-a", work_done_at="2026-11-03"))
    t4 = load_task(_write_task(tasks_dir, "t4", task_family="fam-b", work_done_at="2026-11-04"))
    t5 = load_task(_write_task(tasks_dir, "t5", task_family="fam-c", work_done_at="2026-11-05"))
    t6 = load_task(_write_task(tasks_dir, "t6", task_family="fam-d", work_done_at="2026-11-06"))

    # Even with target calibration_fraction=0.2, cut cannot separate overlapping fam-a and fam-b.
    split = split_tasks([t1, t2, t3, t4, t5, t6], calibration_fraction=0.2)
    assert set(split.calibration) == {"t1", "t2", "t3", "t4"}
    assert set(split.held_out) == {"t5", "t6"}
    assert set(split.calibration_families).isdisjoint(split.held_out_families)
    assert split.latest_calibration_work_done_at < split.earliest_held_out_work_done_at

    # Explicit split_after works when no family straddles it, and refuses when one does.
    split_explicit = split_tasks(
        [t1, t2, t3, t4, t5, t6],
        split_after="2026-11-05T12:00:00Z",
    )
    assert set(split_explicit.calibration) == {"t1", "t2", "t3", "t4", "t5"}
    assert set(split_explicit.held_out) == {"t6"}

    with pytest.raises(UsefulnessError, match="straddles"):
        split_tasks([t1, t2, t3, t4, t5, t6], split_after="2026-11-02T12:00:00Z")


def test_split_refuses_unsplittable_task_sets_and_bad_explicit_splits(tmp_path: Path) -> None:
    tasks_dir = tmp_path / "tasks"

    # 1. Only one family -> cannot split.
    one_fam = [
        load_task(_write_task(tasks_dir, "s1", task_family="only", work_done_at="2026-11-01")),
        load_task(_write_task(tasks_dir, "s2", task_family="only", work_done_at="2026-11-02")),
    ]
    with pytest.raises(UsefulnessError, match="at least two distinct task families"):
        split_tasks(one_fam)

    # 2. Two overlapping families -> no chronological boundary separates them.
    overlap = [
        load_task(_write_task(tasks_dir, "o1", task_family="fa", work_done_at="2026-11-01")),
        load_task(_write_task(tasks_dir, "o2", task_family="fb", work_done_at="2026-11-02")),
        load_task(_write_task(tasks_dir, "o3", task_family="fa", work_done_at="2026-11-03")),
        load_task(_write_task(tasks_dir, "o4", task_family="fb", work_done_at="2026-11-04")),
    ]
    with pytest.raises(UsefulnessError, match="no chronological family-disjoint boundary"):
        split_tasks(overlap)

    # 3. Tied timestamps across families -> cannot put calibration strictly before held-out.
    tied = [
        load_task(
            _write_task(tasks_dir, "e1", task_family="f1", work_done_at="2026-11-05T09:00:00Z")
        ),
        load_task(
            _write_task(tasks_dir, "e2", task_family="f2", work_done_at="2026-11-05T09:00:00Z")
        ),
    ]
    with pytest.raises(UsefulnessError, match="no chronological family-disjoint boundary"):
        split_tasks(tied)

    # 4. Manual split violating family disjointness or time order is refused.
    a = load_task(_write_task(tasks_dir, "m1", task_family="shared", work_done_at="2026-11-01"))
    b = load_task(_write_task(tasks_dir, "m2", task_family="shared", work_done_at="2026-11-02"))
    c = load_task(_write_task(tasks_dir, "m3", task_family="other", work_done_at="2026-11-03"))
    with pytest.raises(UsefulnessError, match="same task family"):
        validate_split([a, b, c], calibration_ids=["m1"], held_out_ids=["m2", "m3"])

    early = load_task(_write_task(tasks_dir, "v1", task_family="f1", work_done_at="2026-11-01"))
    late = load_task(_write_task(tasks_dir, "v2", task_family="f2", work_done_at="2026-11-10"))
    with pytest.raises(UsefulnessError, match="chronological order"):
        validate_split([early, late], calibration_ids=["v2"], held_out_ids=["v1"])


# ---------------------------------------------------------------------------
# Manifest hashing, exclusion, tamper detection, and write-once persistence
# ---------------------------------------------------------------------------


def test_build_manifest_records_parts_families_excluded_and_detects_any_edit(
    tmp_path: Path,
) -> None:
    exp_home = tmp_path / "exp-home"
    tasks_dir = exp_home / "tasks"

    _write_task(
        tasks_dir,
        "task-cal",
        task_family="fam-cal",
        work_done_at="2026-11-10T10:00:00Z",
        strata=["exact_value", "recurrence"],
        hidden_tests={"tests/test_hidden_cal.py": "def test_secret(): assert 1 == 1\n"},
    )
    _write_task(
        tasks_dir,
        "task-held",
        task_family="fam-held",
        work_done_at="2026-11-12T10:00:00Z",
        strata=["abstention"],
        hidden_tests={"tests/test_hidden_held.py": "def test_abstain(): assert True\n"},
    )
    # Inline-excluded task may lack request.md because that is why it was excluded.
    _write_task(
        tasks_dir,
        "task-excl-inline",
        task_family="fam-bad",
        request=None,
        extra_meta={"exclude_reason": "private_dependency_unavailable"},
    )
    # Explicitly excluded task with no directory on disk.
    manifest = build_task_manifest(
        tasks_dir,
        excluded={"task-excl-missing": "duplicate_of_task_cal"},
        cutoff=datetime(2026, 11, 9, 0, 0, tzinfo=UTC),
    )

    assert manifest["schema_version"] == TASK_MANIFEST_SCHEMA_VERSION
    assert set(manifest["tasks"]) == {"task-cal", "task-held"}
    assert set(manifest["families"]) == {"fam-cal", "fam-held"}
    assert manifest["split"]["calibration"] == ["task-cal"]
    assert manifest["split"]["held_out"] == ["task-held"]
    assert manifest["excluded"] == [
        {
            "task_id": "task-excl-inline",
            "reason_code": "private_dependency_unavailable",
            "task_family": "fam-bad",
        },
        {
            "task_id": "task-excl-missing",
            "reason_code": "duplicate_of_task_cal",
        },
    ]

    # Manifest holds per-part hashes and never leaks private request/behavior/test text.
    raw_manifest = json.dumps(manifest)
    assert "Fix the broken pagination offset." not in raw_manifest
    assert "def test_secret" not in raw_manifest
    expected_parts = {
        "repository_commit",
        "request",
        "checks",
        "required_behavior",
        "hidden_tests",
        "project",
        "work_done_at",
        "task_family",
        "strata",
        "task_json",
    }
    assert set(manifest["tasks"]["task-cal"]["parts"]) == expected_parts

    # Persist write-once in run store.
    store = UsefulnessRunStore(exp_home, "run-alpha")
    manifest_sha = store.write_task_manifest(manifest)
    assert manifest_sha == store.task_manifest_sha256()
    with pytest.raises(UsefulnessError, match="already exists"):
        store.write_task_manifest(manifest)

    # Unmodified tasks verify cleanly.
    loaded_cal, split_cal = load_verified_task(tasks_dir / "task-cal", store.read_task_manifest())
    assert split_cal == "calibration"
    assert loaded_cal.task_id == "task-cal"

    # Cutoff enforcement refuses tasks on or before cutoff.
    with pytest.raises(UsefulnessError, match="on or before cutoff"):
        build_task_manifest(tasks_dir, cutoff="2026-11-10T10:00:00Z")

    # Mutating ANY part causes verification to fail and name the changed part.
    cal_dir = tasks_dir / "task-cal"
    orig_req = (cal_dir / "request.md").read_text(encoding="utf-8")
    (cal_dir / "request.md").write_text(orig_req + "\nExtra hint.\n", encoding="utf-8")
    with pytest.raises(UsefulnessError, match="request"):
        load_verified_task(cal_dir, store.read_task_manifest())
    (cal_dir / "request.md").write_text(orig_req, encoding="utf-8")

    # Mutating a hidden test file is caught.
    hidden_file = cal_dir / "hidden_tests" / "tests" / "test_hidden_cal.py"
    orig_hidden = hidden_file.read_text(encoding="utf-8")
    hidden_file.write_text("def test_secret(): assert 1 == 2\n", encoding="utf-8")
    with pytest.raises(UsefulnessError, match="hidden_tests"):
        load_verified_task(cal_dir, store.read_task_manifest())
    hidden_file.write_text(orig_hidden, encoding="utf-8")

    # Adding a new hidden test file is caught.
    extra_hidden = cal_dir / "hidden_tests" / "tests" / "test_extra.py"
    extra_hidden.write_text("def test_extra(): pass\n", encoding="utf-8")
    with pytest.raises(UsefulnessError, match="hidden_tests"):
        load_verified_task(cal_dir, store.read_task_manifest())
    extra_hidden.unlink()

    # Mutating task.json metadata (e.g. strata) is caught.
    meta_path = cal_dir / "task.json"
    orig_meta = meta_path.read_text(encoding="utf-8")
    meta_obj = json.loads(orig_meta)
    meta_obj["strata"] = ["ordinary"]
    meta_path.write_text(json.dumps(meta_obj, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(UsefulnessError, match="strata"):
        load_verified_task(cal_dir, store.read_task_manifest())
    meta_path.write_text(orig_meta, encoding="utf-8")

    # Unlisted task is refused.
    unlisted = load_task(_write_task(tasks_dir, "task-unlisted", task_family="fam-u"))
    with pytest.raises(UsefulnessError, match="not listed"):
        verify_task_against_manifest(unlisted, store.read_task_manifest())


# ---------------------------------------------------------------------------
# Hidden tests absent during agent work, installed only afterwards
# ---------------------------------------------------------------------------


def test_hidden_tests_absent_during_work_and_installed_afterwards(tmp_path: Path) -> None:
    tasks_dir = tmp_path / "tasks"
    task_dir = _write_task(
        tasks_dir,
        "task-h",
        hidden_tests={
            "tests/hidden/test_edge.py": "def test_edge(): assert True\n",
            "tests/hidden/data.txt": "expected=42\n",
        },
    )
    task = load_task(task_dir)

    clone_dir = tmp_path / "clone"
    (clone_dir / "tests").mkdir(parents=True)
    (clone_dir / "tests" / "test_public.py").write_text("def test_pub(): pass\n", encoding="utf-8")

    # Before agent works: hidden tests are NOT in the clone.
    assert_hidden_tests_absent(task, clone_dir)
    assert not (clone_dir / "tests" / "hidden" / "test_edge.py").exists()

    # If a hidden test somehow exists in the clone before work starts, refuse.
    leaked = clone_dir / "tests" / "hidden" / "test_edge.py"
    leaked.parent.mkdir(parents=True)
    leaked.write_text("leaked", encoding="utf-8")
    with pytest.raises(UsefulnessError, match="must not be in the clone"):
        assert_hidden_tests_absent(task, clone_dir)
    leaked.unlink()

    # After agent finishes: install hidden tests and verify hashes match.
    installed = install_hidden_tests(task, clone_dir, expected_hashes=task.hidden_tests)
    assert len(installed) == 2
    assert (
        sha256_file(clone_dir / "tests" / "hidden" / "test_edge.py")
        == task.hidden_tests["tests/hidden/test_edge.py"]
    )
    assert (
        sha256_file(clone_dir / "tests" / "hidden" / "data.txt")
        == task.hidden_tests["tests/hidden/data.txt"]
    )

    # Re-installing over existing files is refused.
    with pytest.raises(UsefulnessError, match="already exists"):
        install_hidden_tests(task, clone_dir)

    # If the source hidden test was tampered with after loading, install refuses.
    clone2 = tmp_path / "clone2"
    clone2.mkdir()
    (task_dir / "hidden_tests" / "tests" / "hidden" / "data.txt").write_text(
        "expected=99\n", encoding="utf-8"
    )
    with pytest.raises(UsefulnessError, match="changed on disk"):
        install_hidden_tests(task, clone2)
