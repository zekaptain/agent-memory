# Working on the usefulness experiment

This guide applies to changes under `usefulness/`, on top of the root
`AGENTS.md` and `src/agent_memory/AGENTS.md`. The design and build plan
are in `docs/memory-utility-evaluation-spec.md`, and its terms are in
`GLOSSARY.md` under "Usefulness experiment".

- Nothing under `judge/` or `admission/` imports `usefulness/`. Do not
  change `judge/metrics.py` or other helpers that reproduce stored judge
  reports.
- The usefulness experiment adds no table, no migration, and no agent to
  any ledger, and refuses to use the default home as an experiment home.
- Statistics in `stats.py` use the Python standard library only, with no
  new dependency. A missing value is `None`, left out of estimates, and
  counted as missing; never coerce a missing value to zero. Check every
  statistic against a worked example whose answer comes from outside the
  code.
