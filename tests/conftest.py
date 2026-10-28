from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import pytest

from agent_memory.config import Config
from agent_memory.db import MemoryClient, MemoryServer, initialize_home, open_direct
from agent_memory.smoke import free_port, synthetic_chain
from tests.support import Chain, insert_rows, new_id


@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    cfg = Config(home=tmp_path / "home", port=free_port())
    initialize_home(cfg)
    return cfg


@pytest.fixture
def con(cfg: Config) -> Iterator[duckdb.DuckDBPyConnection]:
    con = open_direct(cfg.db_path)
    yield con
    con.close()


@pytest.fixture
def chain(con: duckdb.DuckDBPyConnection) -> Chain:
    """A complete user statement -> memory -> recall -> answer chain, written directly."""
    ids = [new_id() for _ in range(7)]
    now = datetime.now(UTC)
    insert_rows(con, synthetic_chain(ids, now))
    return Chain(*ids, now=now)


@pytest.fixture(scope="module")
def live(tmp_path_factory) -> Iterator[tuple[Config, MemoryServer]]:
    """A running Quack server on a fresh home, shared by a test module."""
    cfg = Config(home=tmp_path_factory.mktemp("live") / "home", port=free_port())
    initialize_home(cfg)
    server = MemoryServer(cfg, export_interval_hours=0)
    server.start()
    yield cfg, server
    server.stop()


@pytest.fixture
def live_cfg(live) -> Config:
    return live[0]


@pytest.fixture
def live_client(live_cfg) -> Iterator[MemoryClient]:
    with MemoryClient.from_config(live_cfg) as client:
        yield client


@pytest.fixture
def down_cfg(tmp_path) -> Config:
    """An initialized home whose port nobody listens on."""
    cfg = Config(home=tmp_path / "down", port=free_port())
    initialize_home(cfg)
    return cfg


FIXTURES = Path(__file__).parent / "fixtures"
