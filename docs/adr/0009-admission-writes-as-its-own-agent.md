# Admission writes as its own agent, not as the judge

An agent is the ledger's only stored statement of who wrote a row, and
the row keeps it for good. Migration 0005 registered `judge` for
experiment verdicts only. Admission was built later and wrote its
`decision` events, the memories it admitted, and the approver's approvals under
that same agent, so an experiment's verdict and a production admission
could not be told apart by who wrote them, and the note in migration
0005 had become untrue with no way to edit it. Migration 0007 therefore
registers `admission`, and admission writes as it. Keeping one agent and
recording the note as a caveat was the cheaper choice, since it needs no
migration; it was rejected because every later row would have repeated
the confusion.

## Consequences

- The verdict inside an admission's `decision` event is still the judge
  model's. The agent says who recorded it, and the event's payload names
  the judge.
- A home must be migrated to 0007 before `mem admit` runs. Rows that
  admission wrote as `judge` before then stay as they are.
- `mem admit run` passes over the sessions of both agents when it looks
  for turns to learn from.
