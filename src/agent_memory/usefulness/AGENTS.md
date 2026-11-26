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
- Write-once run files (task manifest, memory-set manifest, schedule,
  registration, finalized grades `grades-final.json`) must go through
  `store.write_once_json` and cannot be overwritten; deviations,
  schedule execution events, grade revisions (`grades.jsonl`), and
  post-grading attributions (`attributions.jsonl`) are append-only
  (`store.append_jsonl`). Interrupted trials are kept and marked on disk
  (`record.json` with `failure_category="interrupted"` or write-once
  `interrupted.json`), never overwritten in place; pair retries use new
  attempt-scoped trial IDs and follow `schedule.RetryRule` only.
- Blind grading (`grading.BlindGradingController`) and post-grading
  memory-use attribution (`grading.MemoryAttributionController`) drive
  the shared `judge.label_frontend.drive` loop with the `"Trial"`
  subject (on both the Python and Go helper sides, and `--plain`),
  leaving judge labeling and admission review unchanged. Blind grading
  never exposes the arm, `trial_id`, trial paths in check output, memory
  tool calls, or ledger content, and orders trials deterministically from
  a seed without placing a pair's two trials back to back. Grades stay
  correctable with append-only history until `grading.finalize_grades`
  writes `grades-final.json`; memory-use attribution cannot open until
  `grades-final.json` exists and matches `grades.jsonl`, reads trial
  homes read-only, and never alters any grade.
- Held-out tasks cannot start a trial before `registration.json` is
  frozen (`registration.freeze_registration`). Once `registration.json`
  exists in a run, every trial verifies the live task manifest, memory
  set, environment manifest, model/limits, and memory-tool wording
  against it, records `registration_sha256` in `record.json`, and gives
  the memory arm's `mem-mcp` the registered wording via
  `AGENT_MEMORY_MCP_WORDING_FILE` rather than assuming the installed
  defaults.
- A memory set (`memory_set.py`) is copied once from a stopped home
  (never a home served by `mem serve`) and never written to afterwards.
  Every memory-arm trial clones a fresh trial home via
  `memory_set.create_trial_home`, which stays readable after the run.
- A task's hidden tests live only under `<task_dir>/hidden_tests/` (never
  in the repository clone while the agent works) and are installed only
  after the agent finishes via `tasks.install_hidden_tests` inside
  `checks.run_trial_checks`. Check results go to write-once `checks.json`
  (omitting the arm for blind grading) and must never alter `record.json`.
  Every trial must verify its task against the run's task manifest before
  starting (`tasks.load_verified_task` / `tasks.verify_task_against_manifest`).
- Every trial runs in a fresh git clone at `task.commit` with future refs
  stripped and repository-configured memory MCP servers removed in both
  arms. No capture hook runs during a trial, and nothing reaches the
  default home.
- `runner.run_trial` treats a model with no loopback `model_base_url` as
  hosted and refuses to run without `AGENT_MEMORY_JUDGE_ALLOW_REMOTE=1`
  (through `endpoints.require_egress`). Token counts the harness does not
  report are recorded as `None`, never `0`.
- At the end of a memory-arm trial, `runner.run_trial` imports its
  transcript into its trial home via the capture path
  (`accounting.import_trial_transcript`, setting
  `AGENT_MEMORY_MCP_SERVER_NAME` to the trial's configured memory server
  so cross-session attribution binds citations) and counts recalls and
  citations against the pre-trial baseline (`accounting.account_trial_recall`).
  A recall the transcript shows that is absent from the trial's retrievals
  is counted as `missing_record`, never as `miss`.
- Tests of the trial runner use `ScriptedHarnessAdapter` or a fake local
  executable script with `DroidHarnessAdapter`; no test calls a real
  model.
- Statistics in `stats.py` use the Python standard library only, with no
  new dependency. A missing value is `None`, left out of estimates, and
  counted as missing; never coerce a missing value to zero. Check every
  statistic against a worked example whose answer comes from outside the
  code.
- Run reports (`report.py`, `mem usefulness report`) are local aggregates
  built strictly from structured run files (`task-manifest.json`,
  `registration.json`, `memory-set/manifest.json`, `schedule.json`,
  `schedule-events.jsonl`, `record.json`, `interrupted.json`,
  `grades-final.json` / `grades.jsonl`, `attributions.jsonl`,
  `deviations.jsonl`) and never read task content, transcripts, trial
  ledgers, check outputs, diffs, replies, failure details, or free-text
  grade/attribution notes. Reports refuse held-out results until
  `registration.json` matches the live manifests, every finished trial is
  covered by verified `grades-final.json`, and the run has reached its
  registered stop (sample size, resource budget, or safety stop). Reports
  contain no wall-clock generation timestamps and must produce identical
  bytes for `report.md` and `report.json` when rebuilt from the same
  files.
