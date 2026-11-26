from __future__ import annotations

import json
import stat
import subprocess
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from agent_memory import cli
from agent_memory.config import DEFAULT_HOME, Config
from agent_memory.db import MemoryClient, MemoryServer, initialize_home, open_direct
from agent_memory.mcp_server.tools import MemoryTools
from agent_memory.smoke import free_port
from agent_memory.usefulness.memory_set import (
    MEMORY_SET_MANIFEST_SCHEMA_VERSION,
    create_trial_home,
    estimate_run_disk_usage,
    make_memory_set,
    memory_set_dir,
    verify_memory_set,
)
from agent_memory.usefulness.store import UsefulnessError, sha256_file


def _seed_source_home(tmp_path: Path) -> tuple[Config, datetime, datetime]:
    """Build a stopped source home with varied memories, validity windows, and blobs."""
    src_cfg = Config(home=tmp_path / "source-home", port=free_port())
    initialize_home(src_cfg)

    now = datetime.now(UTC).replace(microsecond=0)
    cutoff = now + timedelta(days=1)
    last_day = now + timedelta(days=15)

    server = MemoryServer(src_cfg, export_interval_hours=0)
    server.start()
    try:
        tools = MemoryTools(src_cfg, "factory-droid", project="alpha")
        try:
            # 1. Active global fact (no validity bounds)
            tools.remember(
                content="Always format Python with Ruff at 100 chars.",
                memory_type="preference",
                scope="global",
                trust_level="user",
            )
            # 2. Active project decision expiring INSIDE [cutoff, last_day]
            tools.remember(
                content="Staging endpoint is port 8081 until migration completes.",
                memory_type="decision",
                scope="alpha",
                trust_level="user",
                expires_at=(now + timedelta(days=5)).isoformat(),
            )
            # 3. Pending procedure whose valid_from starts INSIDE [cutoff, last_day]
            tools.remember(
                content="New release checklist takes effect next week.",
                memory_type="procedure",
                scope="alpha",
                trust_level="user",
                valid_from=(now + timedelta(days=3)).isoformat(),
            )
            # 4. Quarantined tool fact with valid_to INSIDE [cutoff, last_day]
            tools.remember(
                content="Temporary CI cache key expires mid-run.",
                memory_type="fact",
                scope="beta",
                trust_level="tool",
                valid_to=(now + timedelta(days=10)).isoformat(),
            )
            # 5. Retired fact whose expires_at is INSIDE [cutoff, last_day]
            #    (counts in total_crossing, excluded from non_terminal_crossing)
            r5 = tools.remember(
                content="Deprecated flag --old-mode.",
                memory_type="fact",
                scope="alpha",
                trust_level="user",
                expires_at=(now + timedelta(days=7)).isoformat(),
            )
            tools.review(
                action="retire",
                memory_id=r5["memory_id"],
                reason="flag removed",
            )
            # 6. Memory whose expires_at is BEFORE cutoff (already expired before window)
            #    with a later valid_to INSIDE window -> effective end is expires_at (< cutoff),
            #    so it must NOT count as ending inside [cutoff, last_day].
            tools.remember(
                content="Already expired note with later valid_to.",
                memory_type="fact",
                scope="beta",
                trust_level="user",
                expires_at=(now - timedelta(hours=2)).isoformat(),
                valid_to=(now + timedelta(days=6)).isoformat(),
            )
            # 7. Active memory expiring AFTER last_day (outside window)
            tools.remember(
                content="Long-lived service architecture note.",
                memory_type="fact",
                scope="global",
                trust_level="user",
                expires_at=(now + timedelta(days=60)).isoformat(),
            )
        finally:
            tools.close()
    finally:
        server.stop()

    # Add synthetic blobs under source_home/blobs/
    blob_a = src_cfg.blob_dir / "ab" / "ab0123456789.txt"
    blob_b = src_cfg.blob_dir / "cd" / "cd9876543210.bin"
    blob_a.parent.mkdir(parents=True, exist_ok=True)
    blob_b.parent.mkdir(parents=True, exist_ok=True)
    blob_a.write_text("synthetic blob content A\n", encoding="utf-8")
    blob_b.write_bytes(b"\x00\x01\x02synthetic-binary-blob")

    return src_cfg, cutoff, last_day


def test_make_memory_set_copies_db_blobs_counts_and_writes_once(tmp_path: Path) -> None:
    src_cfg, cutoff, last_day = _seed_source_home(tmp_path)
    exp_home = tmp_path / "exp-home"

    res = make_memory_set(src_cfg, exp_home, cutoff=cutoff, last_day=last_day)

    assert res.memory_set_dir == memory_set_dir(exp_home)
    assert res.manifest_path.is_file()
    assert res.copy_seconds >= 0.0

    db_copy = res.memory_set_dir / "memory.duckdb"
    assert db_copy.is_file()
    assert stat.S_IMODE(db_copy.stat().st_mode) == 0o600
    assert sha256_file(db_copy) == sha256_file(src_cfg.db_path)

    m = res.manifest
    assert m["schema_version"] == MEMORY_SET_MANIFEST_SCHEMA_VERSION
    assert m["migration_level"] >= 10
    assert len(m["applied_migrations"]) >= 10
    assert m["database"]["sha256"] == sha256_file(db_copy)
    assert m["database"]["bytes"] == db_copy.stat().st_size
    assert m["blobs"]["file_count"] == 2
    assert set(m["blobs"]["files"].keys()) == {"ab/ab0123456789.txt", "cd/cd9876543210.bin"}
    assert m["total_bytes"] == m["database"]["bytes"] + m["blobs"]["total_bytes"]

    # Counts by status, scope, type
    counts = m["counts"]
    assert counts["total"] == 7
    assert counts["by_status"] == {
        "active": 3,
        "expired": 1,
        "pending": 1,
        "quarantined": 1,
        "retired": 1,
    }
    assert counts["by_scope"] == {"alpha": 3, "beta": 2, "global": 2}
    assert counts["by_type"] == {"decision": 1, "fact": 4, "preference": 1, "procedure": 1}
    assert counts["active_by_scope"] == {"alpha": 1, "global": 2}
    assert counts["active_by_type"] == {"decision": 1, "fact": 1, "preference": 1}

    # Validity-boundary crossings in [cutoff, last_day]:
    # #2 expires_at in window (ends, active)
    # #3 valid_from in window (starts, pending)
    # #4 valid_to in window (ends, quarantined)
    # #5 expires_at in window (ends, retired -> excluded from non_terminal_crossing)
    # #6 effective_end = expires_at < cutoff -> NOT in window
    # #7 expires_at > last_day -> NOT in window
    vc = m["validity_changes"]
    assert vc["starts_in_window"] == 1
    assert vc["ends_in_window"] == 3
    assert vc["total_crossing"] == 4
    assert vc["non_terminal_crossing"] == 3

    # Verify passes cleanly
    verified = verify_memory_set(exp_home)
    assert verified["memory_set_sha256"] == m["memory_set_sha256"]

    # Second write is refused
    with pytest.raises(UsefulnessError, match="already exists"):
        make_memory_set(src_cfg, exp_home, cutoff=cutoff, last_day=last_day)


def test_refuses_served_source_home_in_process_and_cross_process(tmp_path: Path) -> None:
    src_cfg = Config(home=tmp_path / "live-source", port=free_port())
    initialize_home(src_cfg)
    exp_home = tmp_path / "exp-home"

    # 1. In-process MemoryServer holding the DB lock
    server = MemoryServer(src_cfg, export_interval_hours=0)
    server.start()
    try:
        with pytest.raises(UsefulnessError, match="served or locked"):
            make_memory_set(src_cfg, exp_home, cutoff="2026-11-26", last_day="2026-12-10")
        assert not (exp_home / "memory-set").exists()
    finally:
        server.stop()

    # 2. Cross-process `mem serve` subprocess holding the DB lock
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "agent_memory.cli",
            "--home",
            str(src_cfg.home),
            "--port",
            str(src_cfg.port),
            "serve",
            "--export-interval-hours",
            "0",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 20.0
        ready = False
        while time.monotonic() < deadline:
            try:
                with MemoryClient.from_config(src_cfg) as client:
                    client.query("SELECT 1")
                ready = True
                break
            except Exception:
                time.sleep(0.15)
        assert ready, "subprocess mem serve did not start in time"

        with pytest.raises(UsefulnessError, match="served or locked"):
            make_memory_set(src_cfg, exp_home, cutoff="2026-11-26", last_day="2026-12-10")
        assert not (exp_home / "memory-set").exists()
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()

    # 3. Non-empty WAL file is also refused
    wal_path = src_cfg.home / "memory.duckdb.wal"
    wal_path.write_bytes(b"uncheckpointed")
    try:
        with pytest.raises(UsefulnessError, match="WAL"):
            make_memory_set(src_cfg, exp_home, cutoff="2026-11-26", last_day="2026-12-10")
    finally:
        wal_path.unlink()


def test_refuses_default_home_and_invalid_inputs(tmp_path: Path) -> None:
    src_cfg = Config(home=tmp_path / "src", port=free_port())
    initialize_home(src_cfg)

    with pytest.raises(UsefulnessError, match="default home"):
        make_memory_set(src_cfg, DEFAULT_HOME, cutoff="2026-11-26", last_day="2026-12-10")

    with pytest.raises(UsefulnessError, match="default home"):
        verify_memory_set(DEFAULT_HOME)

    exp_home = tmp_path / "exp"
    with pytest.raises(UsefulnessError, match="earlier than cutoff"):
        make_memory_set(src_cfg, exp_home, cutoff="2026-12-10", last_day="2026-11-26")

    with pytest.raises(UsefulnessError, match="separate directories"):
        make_memory_set(
            src_cfg, src_cfg.home / "nested", cutoff="2026-11-26", last_day="2026-12-10"
        )

    make_memory_set(src_cfg, exp_home, cutoff="2026-11-26", last_day="2026-12-10")
    with pytest.raises(UsefulnessError, match="default home"):
        create_trial_home(exp_home, DEFAULT_HOME)


def test_trial_home_isolation_persistence_and_memory_set_invariance(tmp_path: Path) -> None:
    src_cfg, cutoff, last_day = _seed_source_home(tmp_path)
    exp_home = tmp_path / "exp-home"
    res = make_memory_set(src_cfg, exp_home, cutoff=cutoff, last_day=last_day)
    manifest_sha_before = sha256_file(res.manifest_path)

    # Clone trial home 1 and run a real server + MCP recall & write against it
    trial1_dir = exp_home / "runs" / "r1" / "trials" / "t1-memory" / "home"
    copy1 = create_trial_home(exp_home, trial1_dir)
    assert copy1.memory_set_sha256 == res.manifest["memory_set_sha256"]
    assert copy1.total_bytes == res.manifest["total_bytes"]
    assert (trial1_dir / "blobs" / "ab" / "ab0123456789.txt").read_text(encoding="utf-8") == (
        "synthetic blob content A\n"
    )

    trial1_cfg = Config(home=trial1_dir, port=free_port())
    server = MemoryServer(trial1_cfg, export_interval_hours=0)
    server.start()
    try:
        tools = MemoryTools(trial1_cfg, "factory-droid", project="alpha")
        try:
            recalled = tools.recall(query="staging endpoint port 8081")
            assert len(recalled["items"]) >= 1
            written = tools.remember(
                content="Trial 1 added a new note during execution.",
                memory_type="observation",
                scope="alpha",
                trust_level="agent",
            )
            added_id = written["memory_id"]
        finally:
            tools.close()
    finally:
        server.stop()

    # 1. Memory set hash is unchanged after the trial
    verified = verify_memory_set(exp_home)
    assert verified["memory_set_sha256"] == res.manifest["memory_set_sha256"]
    assert sha256_file(res.manifest_path) == manifest_sha_before

    # 2. What trial 1 added to its trial home can still be read after the run
    con1 = open_direct(trial1_cfg.db_path, read_only=True)
    try:
        mem_rows = con1.execute(
            "SELECT content FROM memories WHERE memory_id = ?", [added_id]
        ).fetchall()
        assert mem_rows == [("Trial 1 added a new note during execution.",)]
        assert con1.execute("SELECT count(*) FROM retrievals").fetchone() == (1,)
    finally:
        con1.close()

    # 3. Trial 2 copied from the same memory set does not see trial 1's writes
    trial2_dir = exp_home / "runs" / "r1" / "trials" / "t2-memory" / "home"
    create_trial_home(exp_home, trial2_dir)
    con2 = open_direct(trial2_dir / "memory.duckdb", read_only=True)
    try:
        assert con2.execute(
            "SELECT count(*) FROM memories WHERE memory_id = ?", [added_id]
        ).fetchone() == (0,)
        assert con2.execute("SELECT count(*) FROM retrievals").fetchone() == (0,)
    finally:
        con2.close()

    # 4. Reusing a non-empty trial home is refused
    with pytest.raises(UsefulnessError, match="not empty"):
        create_trial_home(exp_home, trial1_dir)


def test_verify_memory_set_detects_tampering(tmp_path: Path) -> None:
    src_cfg, cutoff, last_day = _seed_source_home(tmp_path)
    exp_home = tmp_path / "exp-home"
    res = make_memory_set(src_cfg, exp_home, cutoff=cutoff, last_day=last_day)

    # Extra blob file detected
    extra = res.memory_set_dir / "blobs" / "extra.txt"
    extra.write_text("rogue", encoding="utf-8")
    with pytest.raises(UsefulnessError, match="blobs do not match"):
        verify_memory_set(exp_home)
    extra.unlink()
    verify_memory_set(exp_home)

    # Modified DB byte detected
    db_path = res.memory_set_dir / "memory.duckdb"
    with db_path.open("r+b") as fh:
        b0 = fh.read(1)
        fh.seek(0)
        fh.write(bytes([b0[0] ^ 0xFF]))
    with pytest.raises(UsefulnessError, match="database hash mismatch"):
        verify_memory_set(exp_home)
    with pytest.raises(UsefulnessError, match="database hash mismatch"):
        create_trial_home(exp_home, tmp_path / "trial-after-tamper")


def test_estimate_disk_and_cli_commands(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    src_cfg, cutoff, last_day = _seed_source_home(tmp_path)
    exp_home = tmp_path / "exp-cli"

    code = cli.main(
        [
            "--home",
            str(exp_home),
            "usefulness",
            "memory-set",
            "--source",
            str(src_cfg.home),
            "--cutoff",
            cutoff.isoformat(),
            "--last-day",
            last_day.isoformat(),
            "--trials",
            "40",
            "--json",
        ]
    )
    assert code == 0
    out = json.loads(capsys.readouterr().out)
    assert out["manifest"]["counts"]["total"] == 7
    assert out["disk_estimate"]["trial_count"] == 40
    assert out["disk_estimate"]["baseline_total_bytes"] == out["manifest"]["total_bytes"] * 40

    # Verify via CLI
    code_v = cli.main(["--home", str(exp_home), "usefulness", "verify-memory-set", "--json"])
    assert code_v == 0
    verified = json.loads(capsys.readouterr().out)
    assert verified["memory_set_sha256"] == out["manifest"]["memory_set_sha256"]

    # Refuses default home via CLI
    code_bad = cli.main(
        [
            "--home",
            str(DEFAULT_HOME),
            "usefulness",
            "memory-set",
            "--source",
            str(src_cfg.home),
            "--cutoff",
            "2026-11-26",
            "--last-day",
            "2026-12-10",
        ]
    )
    assert code_bad == 1
    assert "default home" in capsys.readouterr().err

    # Estimate validation
    with pytest.raises(UsefulnessError, match="non-negative"):
        estimate_run_disk_usage(out["manifest"], -1)
    with pytest.raises(UsefulnessError, match="non-negative"):
        estimate_run_disk_usage(out["manifest"], True)
