# The judge experiment has its own home and never writes lessons to the ledger

The judge experiment exists to find out whether a judge can be trusted to
admit lessons, so nothing it produces may act as a memory before that
question is answered. `mem judge` therefore refuses the default home
(`ensure_experiment_home`) and works in a separate one, which also keeps
its bulk-imported sessions out of the production ledger. Lessons stay in
run files, and a verdict is recorded in the experiment's ledger only as a
`decision` event without lesson text. Admission is the one route from a
lesson to a memory: it lives in `admission/`, reads frozen human labels,
and writes through the lifecycle. Nothing under `judge/` imports it.

## Consequences

- A separate home does not select a separate port. The experiment's
  server and its clients need the same explicit `--home` and `--port`.
- Judge helpers that admission reuses must work without the experiment
  guard, and admission keeps its own run files.
