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

Tool-origin memories require approval, which only a human gives (see
"Approval is a human's, as built" below); existing agent/user trust
defaults remain. Eligibility also requires `valid_from <= now`, when set, and
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

### Approval is a human's, as built (2026-11-11)

Any agent could approve any quarantined memory, its own included, so the
requirement that tool-origin memories be approved held nothing back. Now
only the `human` agent may ask for an approval. The `review` tool refuses
one from any other agent before it writes or queues anything, and replay
rejects a queued one as a validation failure. A human approves with
`mem review approve`, which shows the memory and asks for a yes typed at a
terminal.

Two approvals are still written without a human. A graph merge carries
over the approval of the relation it replaces, which is not a new
decision. The approver approves a lesson in admission's auto mode, which
`mem admit enable --auto` allows only after the held-out gate passes. The
check is therefore made where an approval is asked for, not where one is
written.

`quarantine`, `retire`, `confirm`, and `flag_contradiction` stay open to
every agent. `trust_level` is still the writer's claim: a memory written
with `user` trust needs no approval.

`mem review` writes in a session of the `human` agent, like `mem graph`
run with the default `--agent`. Its listing of what waits is not a
recall: it reads memories that no agent may be shown, and records
nothing.

The commands, the replay rule, and the limits of a rule that the tools
keep and the server does not are in "Only a human approves a memory" in
[phase-1-spec.md](phase-1-spec.md).

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

### A session's start, as built (2026-11-11)

A session's row is written with the first record the ledger receives of
the session, and a row is never changed. A record received later can be
earlier. A prompt hook writes the row at the time of the prompt, and the
transcript imported when that turn ends holds the session from its
beginning. Until 2026-11-11 an import also dated a session's start at
the time of the read, so the rows of sessions imported before then carry
the time of their import.

Migration `0008_session_start` replaces `session_summary`. Its
`started_at` is the earlier of the row's start and the session's first
event. `recorded_started_at` is the row's own. Nothing else in the view
changed, and no row was rewritten.

`mem audit check` compares where coverage begins with that start.
Coverage that begins where the row does, after an earlier event, is
therefore a gap.

`mem judge live --since` and admission's search for sessions it has not
seen still read the row's start, and a judge item's `session_started_at`
is the row's.

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

### Recalls at the CLI, as built (2026-11-11)

A human at the `mem` CLI reads and never answers, so nothing can cite a
recall made in a session of the `human` agent. `mem audit check` reported
each one as `retrieval_missing_attribution` for good. It no longer expects
a citation for those recalls. This covers
`mem graph show`, `entities` and `resolve` run with the default `--agent`,
including the ones already in a ledger. A recall made with `--agent` set
to a harness's agent is still expected to be cited, and the other
retrieval checks apply to every session.

The session of such a command now says what it is. Its metadata carries
`via: mem graph`, and its capture observation has source mode `record` and
adapter version `mem_cli_v1`. It was recorded as `via: mem-mcp` in mode
`mcp`, because the command runs through the MCP tools' code. Sessions
already in a ledger keep what they say.

### Audit exit code, as built (2026-11-11)

`mem audit check` exits 1 only when a finding is an error: a correction
history that cannot be trusted, an invalid binding, or retrieval rows that
do not agree. Warnings describe capture coverage and attribution. A session
that only used the MCP tools is partial by design and an unfinished session
has a gap, so every ledger in use carries warnings, and exiting 1 for them
said nothing about the ledger. Warnings are still listed and counted.
`--strict` exits 1 on any finding, as the command did before. Exit code 2
still means the audit could not complete.

The JSON report is `version` 2. Its `status` is `clean`, `warnings`,
`findings` (at least one error) or `error` (incomplete), whatever flags
were passed. Its `exit_code` is the code the command exits with.

## Judge version boundary

New runs use `questions_v4` and `labels_v4`. Structured lessons and per-claim
checks remain. Admission adds the whole-draft `rule_supported` gate:

> Do the supported claims and cited evidence justify this exact rule,
> including its scope and conditions, without an unsupported causal leap
> or overgeneralization?

Grounded, faithful, reusable, not already enforced, and rule supported are
all required for a write. Missing or uncertain support cannot yield a write.
The turn question `failed` is descriptive, not a prerequisite.

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

## Delivery record

The delivery order, the test plan, and the local timing check of 2026-11-09
are kept in
[archive/logic-hardening-delivery.md](archive/logic-hardening-delivery.md).
They record how this change was validated and are not part of the contract.
