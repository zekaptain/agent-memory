# Working on the usefulness experiment

This guide applies to changes under `usefulness/`, on top of the root
`AGENTS.md` and `src/agent_memory/AGENTS.md`. The design and build plan
are in `docs/memory-utility-evaluation-spec.md`, and its terms are in
`GLOSSARY.md` under "Usefulness experiment".

- Nothing under `judge/` or `admission/` imports `usefulness/`. Do not
  change `judge/metrics.py` or other helpers that reproduce stored judge
  reports.
- The usefulness experiment adds no table, no migration, and no agent to
  any ledger, and refuses to use the default home as an experiment home
  (`store.ensure_experiment_home`).
- Write-once run files (task manifest, memory-set manifest, registration)
  must go through `store.write_once_json` and cannot be overwritten;
  deviations are append-only (`store.append_jsonl`).
- A task's hidden tests live only under `<task_dir>/hidden_tests/` (never
  in the repository clone while the agent works) and are installed only
  after the agent finishes via `tasks.install_hidden_tests`. Every trial
  must verify its task against the run's task manifest before starting
  (`tasks.load_verified_task` / `tasks.verify_task_against_manifest`).
- Every trial runs in a fresh git clone at `task.commit` with future refs
  stripped and repository-configured memory MCP servers removed in both
  arms. No capture hook runs during a trial, and nothing reaches the
  default home.
- `runner.run_trial` treats a model with no loopback `model_base_url` as
  hosted and refuses to run without `AGENT_MEMORY_JUDGE_ALLOW_REMOTE=1`
  (through `endpoints.require_egress`). Token counts the harness does not
  report are recorded as `None`, never `0`.
- Tests of the trial runner use `ScriptedHarnessAdapter` or a fake local
  executable script with `DroidHarnessAdapter`; no test calls a real
  model.
- Statistics in `stats.py` use the Python standard library only, with no
  new dependency. A missing value is `None`, left out of estimates, and
  counted as missing; never coerce a missing value to zero. Check every
  statistic against a worked example whose answer comes from outside the
  code.
