# Events stay in the ledger, and only a human's blanking removes text

The ledger only grows, and two things could take content out of it: an
archive for old events, to save space, and the removal of text that
must not be kept, such as a secret that redaction missed or another
party's data. We decided both together on 2026-11-23. Events stay.
Measured to 10 million events, nothing that a hook or an agent waits
for slows with the ledger's size, so an archive would save disk and
nothing else, and it would cost the join from a memory, a recall, or a
reply to its events, which is what the ledger is for. Text that must
not be kept is another matter, because the only other remedy is to
delete the home, which loses the whole record to be rid of one line.
So a human, and never an agent, may blank the text of an event or a
memory. The row stays, with who wrote it, when, and in which session,
and with everything that refers to it, and the ledger records the
blanking: who did it, when, why, and to which rows.

## Consequences

- Of a blanked row the ledger can still say that it happened, who wrote
  it, when, how large its text was, and what refers to it. It can no
  longer say what the text was. The hash of the text goes with the
  text, because a hash confirms a guess at a short secret.
- Space is never a reason to blank. What bounds a home's growth, and
  when this decision is looked at again, is under "Growth" in
  [roadmap.md](../roadmap.md).
- A blanking takes text out and puts none in. A wrong memory is still
  corrected by a superseding memory
  ([ADR 0001](0001-append-only-ledger-with-derived-status.md)).
- Nothing blanks yet, and it cannot be done by hand. A migration is
  applied to every home and cannot name one home's rows. With the guard
  against updates dropped, DuckDB 2.0.0 still refuses to change an
  event's payload or a memory's hash once another row refers to the
  row, and text it does overwrite stays in the database file until the
  space is used again. A blanking therefore writes the database anew
  without the text.
- A blanking reaches what the home holds: the row, its blob, the
  snapshots, and the spool's files. It does not reach a copy of the
  home, or the harness's own transcript. Importing that transcript
  again does not bring the text back, because the writer finds the
  event by its id ([ADR 0003](0003-deterministic-capture-ids.md)).
