# Hardening the memory contracts

Approved scope: ledger transitions, observed capture, answer attribution, and
versioned judge admission. This document records the contract, not evidence
that a live home has been migrated. No live rollout, hook installation,
private transcript import, model experiment, or hosted call is authorized.

## Correction is a proposal until accepted

`memories.supersedes` proposes a correction. It does not establish acceptance.
An append-only acceptance record commits one replacement per predecessor,
with its proposal, supporting event or review, acceptance time, and method.
Further corrections must target the accepted successor.

Before acceptance, the predecessor remains eligible under its own status.
A newly created correction stays pending, even if its trust would otherwise
allow recall. Acceptance hides the predecessor permanently. Retirement,
expiry, or quarantine of the successor never revives the predecessor.
Restoration requires an explicit new memory or correction.

Tool-origin memories require approval; existing agent/user trust defaults
remain. Eligibility also requires `valid_from <= now`, when set, and
`now < valid_to` and `now < expires_at`, when set. Empty/reversed intervals
are invalid. Pending eligibility has an inspectable reason distinct from
quarantine. Confirmation supplies evidence, not approval or restoration.

An admissible correction is accepted in the same transaction as its write
or approval. Previously authorized future corrections are processed at
startup, maintenance, and before FTS or graph ranking. This only applies
recorded authorization. Indexed pending work and bounded batches keep this
off the full-ledger scan path; a backlog must not expose two versions.

Reject self-replacement, cycles, cross-scope correction, and stale competing
replacement. A unique predecessor constraint settles concurrent acceptance;
the loser receives a conflict, not a queued network retry.

## Shared write semantics

Remember, replay, and graph merge share correction and provenance logic.
Deduplication still processes correction intent and additional sources.
An already-existing replacement uses a proposal in `memory_actions`.

A queued ID remains a valid ledger identity. If replay finds a canonical
duplicate, it records a deduplication link and resolves dependent queued
operations through that link. Ordinary supersession is not an ID alias:
reviewing a predecessor must not review its successor by accident.

Graph merges carry approval from an active relation, including tool-origin
relations. Existing canonical relations keep the incoming source and alias
lineage. No merge silently drops evidence because content already exists.

Validation failures are terminal diagnostics, not outages. Replay remains
idempotent, parent-first, redacted, and bounded in its retries. A partial
failure must not leave a correction half accepted.

## Coverage comes from observations

Append-only `capture_observations` describe source mode, adapter and policy
version, source fingerprint, cursor/range, parse errors, exclusions, and
effective session metadata. Reimporting the same observation is idempotent.
All new persistent fields pass through redaction.

The session row remains unchanged. `session_summary` exposes recorded
completeness separately from observed coverage:

| Evidence | Effective coverage |
|---|---|
| MCP only, even with a harness session ID | Partial |
| Isolated hooks | Partial |
| Verified import | Transcript import, with boundary and exclusions |
| Explicit complete supported capture evidence | Full within its boundary |
| Old tier-based full assertion without evidence | Unverified legacy assertion |

An import covers its recorded boundary, not later activity. Omitted thinking,
unavailable attachments, parse failures, and unsupported hook categories are
not captured. A transcript `user` role does not establish human authorship.

## Attribution preserves uncertainty

Append-only `answer_bindings` connect placeholders to captured replies with
evidence, method/version, and correction history. Effective links preserve
the original placeholder and raw `context_links`.

Explicit answer IDs take priority. Otherwise, the adapter preserves supplied
message/turn/channel IDs; a captured cite result can identify its placeholder
and the unique reply in that captured turn. No timestamp-nearest or
next-assistant heuristic is allowed. Absent or ambiguous evidence leaves the
placeholder unresolved. Arrival order, incremental imports, and retries
must not change that rule.

`why_answer` reports both attribution and raw identities. Its retrieval item
join is outer, so a cited zero-result FTS or graph retrieval produces an
audit row with `returned = 0` and null memory fields. A miss is not an
unavailable recall. Historical `status_at_recall` is immutable.

Use the terms returned, captured in context where observed, and declared
used. These are evidence records, not access to the model's reasoning.
Placeholders do not count as actual captured replies.

### Cross-session bindings, as built (2026-11-09)

MCP need not share the harness session. A captured `cite` result carries the
placeholder UUID returned to that agent, so reconciliation looks up that
placeholder without a session filter. Calls, results, retrieval evidence,
and candidate replies still require the transcript session. Evidence stores
`placeholder_session_id`; explicit future answer IDs use the same rule.
Late citations revisit persisted observations by returned placeholder ID.

Migration `0006_cross_session_answer_bindings` removes session equality from
the binding validation trigger without rewriting events or old bindings.
It preserves the placeholder, actual-reply, and linear-correction checks.
Audit resolves the same bindings and checks retrieval ownership against
the original context-link endpoint, rather than requiring a captured reply
to live in the MCP session. Raw context links and historical provenance stay
unchanged. Missing, ambiguous, or out-of-session transcript evidence does
not become a successful binding.

## Judge version boundary

New runs use `questions_v4` and `labels_v4`. Structured lessons and per-claim
checks remain. Admission adds the whole-draft `rule_supported` gate:

> Do the supported claims and cited evidence justify this exact rule,
> including its scope and conditions, without an unsupported causal leap
> or overgeneralization?

Grounded, faithful, reusable, not already enforced, and rule supported are
all required for a write. Missing or uncertain support cannot yield a write.
Stage A failure is descriptive, not a prerequisite.

Blinded human labels include rule support and "Would you store this exact
draft?" Turn-level worth remains a discovery/opportunity measure, not draft
admission ground truth. Missing/unknown draft labels are unscored.
Calibration alone fits thresholds; frozen held-out behavior does not change.
Both labeling interfaces, edit history, hashes, cost reports, and decision
events follow the version. Decision events never contain lesson text.

Historical v1–v3 drafts, labels, thresholds, and report definitions remain
unchanged and reportable. Historical runs are read-only for draft/decide/label.
Deploy the judge version separately, after ongoing old-version labeling is
finished; ledger migrations do not rewrite experiment files.

## Preflight, migration, and recovery

`mem audit check` is read-only, supports JSON, and reports IDs/reason codes,
not transcript text. It covers invalid/pending/ambiguous corrections,
conflicting histories, unverified coverage, gaps, unresolved answers, and
retrieval attribution gaps. It must work against the old schema.

Use separate numbered migrations for lifecycle, observations, and bindings.
Never edit an applied migration. Update append-only guards, export ordering,
client write lists, and authorization coverage together.

Legacy supersedes edges are not automatic acceptance. Reconstruct only when
recorded trust, validity, and review history support it; record method and
evidence. Ambiguous histories block migration pending explicit review.
Preserve raw rows and retrieval snapshots. Do not invent an approval or
silently reactivate a previously hidden fact.

Live rollout needs separate approval: run preflight, verify a consistent
backup including blobs, stop the server, migrate, restart, and verify
read-only. Rollback restores the backup with the matching application
version, never destructive reverse migrations.

Loopback-only serving, token privacy, SQL quoting, and redaction remain.
The shared token authenticates access, not an individual agent's claimed
identity.

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
