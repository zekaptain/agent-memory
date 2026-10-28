"""Helpers shared by the test modules. Fixtures live in conftest.py."""

from __future__ import annotations

import json
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import duckdb


def new_id() -> str:
    return str(uuid.uuid4())


def scalar(con: duckdb.DuckDBPyConnection, sql: str, params: Sequence[Any] = ()) -> Any:
    row = con.execute(sql, list(params)).fetchone()
    if row is None:
        raise AssertionError(f"no rows from: {sql}")
    return row[0]


def insert_rows(
    con: duckdb.DuckDBPyConnection, rows_by_table: Mapping[str, Sequence[Mapping[str, Any]]]
) -> None:
    for table, rows in rows_by_table.items():
        types = dict(
            con.execute(
                "SELECT column_name, data_type FROM duckdb_columns() "
                "WHERE database_name = current_database() AND schema_name = 'main' "
                "AND table_name = ?",
                [table],
            ).fetchall()
        )
        for row in rows:
            cols = list(row)
            collist = ", ".join(f'"{c}"' for c in cols)
            placeholders = ", ".join(
                "?::JSON::VARIANT" if types[c] == "VARIANT" else "?" for c in cols
            )
            values = [
                json.dumps(row[c]) if types[c] == "VARIANT" and row[c] is not None else row[c]
                for c in cols
            ]
            con.execute(f"INSERT INTO {table} ({collist}) VALUES ({placeholders})", values)


@dataclass(frozen=True)
class Chain:
    session_id: str
    user_event: str
    recall_event: str
    answer_event: str
    end_event: str
    memory_id: str
    retrieval_id: str
    now: datetime
