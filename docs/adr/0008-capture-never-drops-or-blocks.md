# Capture never drops a record and never blocks the agent

`mem event` runs inside the harness as a hook. Droid blocks the agent
when a hook exits 2 and can add a hook's stdout to the agent's context,
so a capture problem handled the usual way would change what the agent
does. It must not do that, and it must not silently lose what happened
either. So `mem event` prints nothing on stdout and never exits 2. When
the server is unreachable, the redacted records go to the spool. A record
the ledger rejects is kept under `spool/failed/`, with a `capture_error`
event in its session, while the rest of its batch is written.

## Consequences

- A failing capture is quiet at the harness. It shows up in `mem spool
  status`, in the `status` tool's `spool_rejected_lines`, as
  `capture_error` events, and in `mem audit check`.
- Nothing retries a rejected record on its own. `mem spool retry` returns
  it to the spool once the cause is fixed.
