# Hardening the memory contracts: validation and delivery record

These two sections were moved out of
[logic-hardening-spec.md](../logic-hardening-spec.md) on 2026-11-11. They
record the delivery order, the test plan, and a local timing check from
2026-11-09. They are kept for the record and are not the current contract.

## Validation and delivery

Delivery order is preflight/spec, lifecycle, observations/bindings, judge,
then documentation and cross-system checks. Each phase needs focused
regressions before it is considered complete.

Tests use synthetic old-schema databases and actual MCP/capture/Quack paths.
Cover future validity, approval and withdrawal, competing replacements,
dedupe with new sources, queued IDs and dependent reviews, restart/replay,
coverage enrichment, both citation arrival orders, ambiguity, and misses.
Exercise intermediate states and failures between dependent writes.

Judge tests use mocked HTTP, rendered question subjects, navigation,
unsupported rules over true claims, valuable turns with bad drafts, and
historical report reproduction.

Run pytest, Ruff lint and format checks, Pyrefly, throwaway `mem smoke`,
and labeling-helper Go tests/vet. Measure fresh-process first-call and due
pending-work overhead on realistic synthetic sizes. Do not infer performance
from an unmeasured bounded query.

The usefulness experiment is design-only in
`memory-utility-evaluation-spec.md`. No evaluation harness is part of this
change.

## Local timing check (2026-11-09)

Measured in three fresh Python processes per case on this machine (macOS,
18 CPUs, DuckDB 2.0), with cached extensions and synthetic throwaway homes.
The first-call timer includes client connection, session/observation setup,
due maintenance, ranking, and retrieval provenance. Server startup is
measured separately. Dataset construction and Python imports are excluded.
No model calls or usefulness experiment were run.

Each populated case has 10,000 initial facts and 2,000 accepted corrections.
The backlog cases add 512 authorized, due corrections. Startup processes
128; client setup and ranking each process at most another 128; the final
maintenance measurement processes the remaining 128.

| Case | First call, range (ms) | Maximum server startup (ms) | Maximum final maintenance pass (ms) |
|---|---:|---:|---:|
| Empty ledger, FTS miss | 90.1–92.8 | 44.9 | 4.4 |
| Populated ledger, no pending work | 109.2–110.9 | 58.4 | 4.4 |
| Populated ledger, 512 due corrections, FTS | 181.6–183.0 | 60.7 | 50.4 |
| Same backlog, plus 1,000 relations, graph | 192.1–195.6 | 61.9 | 47.7 |

FTS used default limits and returned eight items in populated cases. Graph
used three hops and limit 100 over a four-way tree, returning 84 relations.
Every case checked that no predecessor and proposed successor were both
active. These are local samples, not a general latency guarantee.

The first implementation checked and committed due corrections one at a
time: the same FTS backlog case took 3,634.6 ms on its first call and
1,779.2 ms for the final 128-item maintenance pass. The shipped implementation
checks the bounded batch once and commits independent replacements together.
It defers conflicting proposals and correction chains rather than weakening
authorization or validity checks.

Validation: the full Python suite passes (679 tests), Ruff lint and formatting
pass, the default Pyrefly check has zero errors, and the throwaway smoke check
passes 16/16. Go tests and vet pass. Pyrefly's optional warning-level check
reports six warnings (unnecessary conversions and a redundant exception
clause), so that stricter warning-level command is not clean.

No live home was migrated. No private transcript, hook installation, model
experiment, hosted call, commit, or push was part of this validation.
