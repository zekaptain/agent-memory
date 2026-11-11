# One server process owns the database file

Several harnesses, their hooks, and their MCP servers write to one ledger
at the same time, and only one process can open a DuckDB file for
writing. `mem serve` is therefore the only process that opens
`memory.duckdb`. Every client attaches over Quack on localhost with a
bearer token and runs its statements on the server, which checks each one
against the `mem_authorize` allow-list. When the server is down, clients
queue redacted records in the spool. Admin work (migrations, `mem sql
--direct`) opens the file directly and needs the server stopped.

## Consequences

- The design depends on Quack, which first shipped in DuckDB 2.0.0. Its
  protocol and authorization hook may change; `mem smoke` and the server
  tests are the early warning.
- Quack forwards neither bound parameters nor `RETURNING`, so the client
  stages rows in a local temporary table and inserts from it. See
  "Verified DuckDB 2.0.0 behavior" in
  [architecture.md](../architecture.md#verified-duckdb-200-behavior).
- Nothing else may open the file while the server runs, so backups and
  offline analysis use the Parquet export.
