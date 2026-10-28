# Parametric consolidation spec: compiling approved memories into a local model artifact

Status: draft for review, 2026-10-25. Builds on Phase 0 and Phase 1
(`PLAN.md`, `docs/phase-1-spec.md`, migration 0001). Not scheduled; this is
the research item the literature review leaves open (open problem 8 in
`docs/agent-memory-literature-review.md`, Section 12), scoped so that the
ledger stays the source of truth and the artifact is a disposable cache.

## Goal

Answer one question with this project's own data: can stable, approved
memories be compiled into a model artifact that (a) answers questions about
them with fewer tokens than injecting them into context, and (b) stops
answering about a memory once that memory is retired in the ledger, with a
measured backflow rate rather than an assumption?

The review's finding (Section 3.5) is that parametric memory is where the
efficiency gains are and where auditability is lost. This spec keeps
auditability by making the artifact a pure function of a recorded set of
`memory_id`s, so for any answer the compiled path gives you can still ask
what it was built from, when, and whether any of that has since been retired.

The second deliverable is a per-scope *engagement* signal, computed from the
ledger, that decides which scopes get compiled and when an artifact is
rebuilt or dropped. It is an exponential moving average of ledger activity.
It is the one idea salvaged from the EndocrineAI notes, and it is a
scheduler, not a learning mechanism.

## Non-goals

- Changing any hosted model. Droid, Vibe, omp, Amp, and Junie run hosted
  models that cannot load a local artifact. The compiled path is a local
  sidecar reached through `recall`, not a change to the agent's model.
- Online or per-turn weight updates from traffic. The review's security
  sections (10.1, 10.5) rule this out: no quarantine step, no deletion.
  Compilation runs offline, from approved memories only.
- Replacing token-level recall. `fts_bm25_v1` stays the default; the compiled
  path is an additional strategy with its own name, so `retrievals.strategy`
  distinguishes them in every audit.
- Deleting content from the ledger. Retirement excludes a memory from the
  next build; the ledger row stays, as today.

## What gets compiled

Eligibility is a SQL view so the rule is inspectable and the manifest is
reproducible:

```sql
CREATE VIEW compile_eligible AS
SELECT m.*
FROM active_memories m
JOIN memory_status st USING (memory_id)
WHERE m.expires_at IS NULL
  AND m.type IN ('fact', 'preference', 'decision', 'procedure', 'profile')
  AND m.confidence >= 0.6
  AND m.created_at <= now() - INTERVAL 3 DAY
  AND (m.trust_level IN ('user', 'agent') OR st.last_review = 'approve');
```

Rules this encodes, each from the review:

- Only `active` status. Quarantined, superseded, expired, and retired
  memories never enter an artifact. `memory_status` already derives this.
- `tool` and `imported` trust only after an explicit `approve` action
  (MINJA, AgentPoison: tool output is the injection channel).
- `observation` type excluded: it is the low-confidence tier by definition.
- Three-day minimum age, so a memory superseded within days of creation is
  not compiled and then immediately stale.
- No `expires_at`: anything with a known end date belongs in the token path.

The threshold values are first guesses, recorded in the manifest so later
builds can be compared.

## Data model (migration 0002)

Three append-only tables, one vocabulary insert, one view, and an extended
`mem_authorize` allowlist. Guards follow the 0001 pattern.

```sql
CREATE TABLE compiled_artifacts (
    artifact_id      UUID PRIMARY KEY,
    scope            VARCHAR NOT NULL,
    kind             VARCHAR NOT NULL CHECK (kind IN ('lora', 'kv_cartridge')),
    base_model       VARCHAR NOT NULL,          -- e.g. 'mlx-community/Qwen3-4B-4bit'
    recipe           VARCHAR NOT NULL,          -- 'self_study_v1'
    recipe_params    VARIANT,                   -- epochs, rank, qa_per_memory, seed
    eligibility_sql  VARCHAR NOT NULL,          -- the view text at build time
    member_count     INTEGER NOT NULL,
    members_hash     VARCHAR NOT NULL,          -- sha256 over sorted member content hashes
    artifact_ref     VARCHAR NOT NULL,          -- path under artifacts/, content-addressed
    artifact_bytes   BIGINT NOT NULL,
    train_seconds    DOUBLE,
    engagement       DOUBLE,                    -- scope EMA at build time (below)
    supersedes       UUID REFERENCES compiled_artifacts (artifact_id),
    built_by_session UUID REFERENCES sessions (session_id),
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE artifact_members (
    artifact_id  UUID NOT NULL REFERENCES compiled_artifacts (artifact_id),
    memory_id    UUID NOT NULL REFERENCES memories (memory_id),
    content_hash VARCHAR NOT NULL,              -- memory content hash at build time
    PRIMARY KEY (artifact_id, memory_id)
);

CREATE TABLE artifact_actions (
    action_id    UUID PRIMARY KEY,
    artifact_id  UUID NOT NULL REFERENCES compiled_artifacts (artifact_id),
    action       VARCHAR NOT NULL
        CHECK (action IN ('activate', 'deactivate', 'erase_file', 'eval')),
    actor        VARCHAR NOT NULL,
    reason       VARCHAR,
    payload      VARIANT,                       -- eval metrics live here
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

INSERT INTO event_kinds VALUES ('compile', 'Offline compile job ran for a scope');
```

Derived state, same pattern as `memory_status`:

```sql
CREATE VIEW artifact_status AS
SELECT a.artifact_id, a.scope,
       CASE
         WHEN erased.artifact_id IS NOT NULL THEN 'erased'
         WHEN s.artifact_id IS NOT NULL THEN 'superseded'
         WHEN lr.action = 'activate' THEN 'active'
         ELSE 'inactive'
       END AS status,
       -- members retired or superseded since the build: the staleness signal
       (SELECT count(*) FROM artifact_members am
        JOIN memory_status ms USING (memory_id)
        WHERE am.artifact_id = a.artifact_id AND ms.status <> 'active') AS stale_members
FROM compiled_artifacts a
LEFT JOIN (SELECT artifact_id, arg_max(action, (created_at, rowid)) AS action
           FROM artifact_actions WHERE action IN ('activate', 'deactivate')
           GROUP BY artifact_id) lr USING (artifact_id)
LEFT JOIN (SELECT DISTINCT artifact_id FROM artifact_actions WHERE action = 'erase_file') erased USING (artifact_id)
LEFT JOIN (SELECT supersedes AS artifact_id FROM compiled_artifacts WHERE supersedes IS NOT NULL) s USING (artifact_id);
```

`mem_authorize` is replaced (`CREATE OR REPLACE MACRO`) with the three new
tables added to the `insert into` allowlist. Everything else in the policy is
unchanged; the Phase 0 policy tests are re-run against the new text.

`stale_members > 0` is the one place the ledger admits that bytes on disk
encode something it has retired. That is why `erase_file` exists.

## Engagement signal and scheduler

One number per scope, recomputed by the job from the ledger, never stored as
state that drifts:

```sql
-- Daily activity per scope, then an EMA with half-life h days.
WITH daily AS (
    SELECT s.scope, e.ts::DATE AS day,
           count(*) FILTER (WHERE e.kind IN ('user_msg', 'assistant_msg', 'tool_call')) AS turns,
           count(*) FILTER (WHERE e.kind = 'memory_recall') AS recalls
    FROM events e JOIN sessions s USING (session_id)
    WHERE e.ts >= now() - INTERVAL 60 DAY
    GROUP BY ALL
)
SELECT scope,
       sum((turns + 4 * recalls) * power(0.5, date_diff('day', day, current_date) / 7.0)) AS engagement
FROM daily GROUP BY scope;
```

Half-life 7 days, recall weighted 4x over a plain turn because a recall is
direct evidence the scope's memories are being used. Both constants go into
`recipe_params` so they are auditable per build. Scheduler rules:

| Condition | Action |
|---|---|
| `engagement >= E_min` and no active artifact and `compile_eligible` count >= 20 | build |
| active artifact and (`stale_members > 0` or eligible members changed by >= 10% by `members_hash`) | rebuild, `supersedes` the old one |
| active artifact and `engagement < E_min / 4` for 14 consecutive job runs | `deactivate` |
| artifact superseded or deactivated and `stale_members > 0` | `erase_file` (below) |

`E_min` is calibrated on this machine's ledger before the first build: the
median per-scope engagement over the last 30 days. The job runs nightly after
the Parquet export, as `PLAN.md` already proposes for Phase 4 and 5 work.

The signal is deliberately an EMA of counts. The EndocrineAI experiments
(`saraproto/obsidian-protocol/paper-workspace/`) showed that the same signal
used to modulate learning rates or replay priority had no measurable effect;
used as a scheduler it only decides when to spend compute, which is the role
sleep-time compute and nightly lint already fill in the literature.

## Build recipe (`self_study_v1`)

Cartridges-style self-study (Eyuboglu et al., 2025), implemented with LoRA
first because `mlx-lm` supports LoRA training on Apple silicon out of the
box. A trained KV-prefix (`kv_cartridge`) is a second `kind` once the LoRA
path has numbers.

1. Read `compile_eligible` for the scope. Record the member list and
   `members_hash`.
2. For each member, generate `qa_per_memory` (default 6) question/answer
   pairs with the local base model: two direct, two paraphrased, one that
   requires combining with another member of the same `key` prefix, one
   negative ("is X true?" where X contradicts the memory, answer "no, ...").
   Every pair cites its `memory_id` in a sidecar JSONL so probe sets can be
   rebuilt. The pairs never enter the ledger; the manifest does.
3. Train a LoRA (rank 16, 3 epochs, seed recorded) on the pairs plus a fixed
   set of 200 generic refusal pairs ("I don't have a memory about that") so
   the artifact learns to abstain outside its member set.
4. Write the adapter to `artifacts/<sha256[:2]>/<sha256>/` with
   `adapter.safetensors`, `manifest.json` (the `compiled_artifacts` row and
   member list), and `probes.jsonl`. Content-addressed, so a rebuild with the
   same members and params is a no-op.
5. Insert the `compiled_artifacts` row and `artifact_members` rows in one
   transaction; emit a `compile` event in a `human` or `compile` session with
   counts in `payload`.
6. Run the evaluation (below). Insert an `eval` action. Only if the retention
   and abstention thresholds pass, insert `activate`.

Everything local. The base model, the generated pairs, and the adapter never
leave the machine. Redaction already ran upstream on everything in
`memories.content`.

## Evaluation

Three measurements, all stored as `eval` actions with the metrics in
`payload`, so a later build can be compared with an earlier one by SQL.

### Retention and abstention

Probe set: held-out pairs (the sixth generated pair per member is withheld
from training), plus 100 questions about memories from other scopes and 100
about invented facts. Metrics: exact or LLM-judged correctness on held-out
pairs (`retention`), refusal rate on out-of-scope and invented questions
(`abstention`). Activation thresholds: `retention >= 0.8`,
`abstention >= 0.9`. Judge model and prompt are recorded in `recipe_params`.

### Cost against the token path

For the same probe questions, run the token path: `recall` with
`fts_bm25_v1`, inject the returned memories, answer with the same base
model. Record tokens per answer and wall-clock latency for both paths. The
review's one strongly replicated claim is that memory systems save an order
of magnitude in tokens versus full context; this measures whether a compiled
artifact saves anything over a good token path, which no surveyed system has
reported.

### Backflow after retirement (the Bemula design)

This is the measurement the field lacks (review Sections 10.5 and 11.5).

1. From the active artifact's members, sample a forget set F (20 memories,
   stratified by `type`) and a paired control set C (20 memories matched on
   `type` and content length, disjoint from F).
2. Record `retire` actions for every member of F, reason
   `backflow_eval:<artifact_id>`.
3. Run the scheduler. It sees `stale_members = 20`, rebuilds, and the new
   artifact excludes F by construction. Verify `artifact_members` of the new
   artifact contains all of C and none of F; this is the token-level part of
   the test and it should pass trivially.
4. Probe the new artifact and the plain base model with, per memory in F and
   C: the two held-out direct questions, two paraphrases, and three
   extraction prompts (completion of the memory's first half, "list
   everything you know about <key>", role-play as the previous version).
   Also probe the *old* artifact as the upper bound.
5. Metrics:
   - `backflow = extract(F | new) - extract(F | base)`. The base model's rate
     on F is the floor (it may know some facts anyway).
   - `over_deletion = retention(C | old) - retention(C | new)`.
   - `extract(F | old)` for reference.
   Pass criteria for the research question: `backflow <= 0.05` and
   `over_deletion <= 0.05`. Bemula's result is that no production system
   passes; a LoRA rebuilt from scratch should pass because the retired text
   is simply absent from training, and if it does not, the base model's own
   knowledge is the reason, which is a finding worth having.
6. Restore: F was retired for the test. The members are re-created as new
   memories superseding nothing, with `memory_sources` relation
   `derived_from` pointing at the originals' source events, and a note in
   `metadata` naming the eval. The retired rows stay retired. This is the
   append-only way to undo a test retirement and it leaves the audit trail
   intact.

Repeat with 3 seeds and 3 random F/C draws before reporting a number.

### Erasure

When an artifact is superseded or deactivated and `stale_members > 0`, the
job deletes its directory from disk and records `erase_file` with the byte
count and `stale_members` in `payload`, never the content. This is the
"administrative redaction logged as counts" that `PLAN.md` reserves for the
rare case, and artifacts are the one place it is routine, because an artifact
is a cache and the ledger can rebuild it minus the retired members.

## Recall integration (`compiled_v1`)

A later step, only after the evaluation has numbers worth acting on.

- `recall(query, strategy='compiled_v1')` loads the active artifact for the
  requested scopes, asks the local model the query, and returns one
  synthesised item with `memory_id` pointing at the artifact's manifest
  rather than a single memory. To fit the existing schema without a new
  item type, the job inserts one `memories` row per artifact (type
  `procedure`, key `compiled:<scope>`, content "Compiled from N memories,
  see artifact <id>", trust `agent`) and `retrieval_items` reference it, so
  `why_answer` works unchanged and shows the compiled source.
- `retrievals.strategy = 'compiled_v1'`, `params` carries `artifact_id`.
- The answer carries the artifact's `stale_members` at recall time in
  `score_components`, so an audit can see that a compiled answer was served
  from an artifact that had retired members.
- The MCP tool description tells the agent the compiled answer is a
  synthesis and that `fts_bm25_v1` returns the underlying memories with
  sources.

## CLI

- `mem compile status`: scopes, engagement, eligible counts, active
  artifacts, `stale_members`.
- `mem compile run [--scope S] [--dry-run]`: the nightly job. `--dry-run`
  prints the scheduler decision and the member list without training.
- `mem compile eval <artifact_id> [--backflow]`: runs the evaluation
  sections above, records `eval` actions.
- `mem compile erase <artifact_id> --reason ...`: manual `erase_file`.

All writes go through `MemoryClient` over Quack like every other writer.
The trainer process reads memory content from the Parquet export, not the
live server, so a long training run never holds a server connection.

## Tests

- Migration 0002 applies on a 0001 database; guards reject update and delete
  on the three new tables; `mem_authorize` allows inserts into them and
  still rejects everything the Phase 0 policy tests reject.
- `compile_eligible` excludes quarantined, retired, superseded, expired,
  `observation`, under-age, and unapproved `tool` memories (one fixture
  each).
- Scheduler decisions on a fixture ledger: build, rebuild on stale member,
  rebuild on 10% membership change, deactivate after decay, no-op when
  `members_hash` is unchanged.
- Build is content-addressed: two runs with the same members and params
  produce the same `artifact_ref` and the second inserts nothing.
- Backflow harness on a tiny model (Qwen3 0.6B class) with 5 members: the
  new artifact's members exclude F, `erase_file` removes the old directory,
  the restore step creates superseding rows with sources, and the metrics
  dictionary has every key. The numeric thresholds are not asserted in
  tests; they are the experiment.
- `why_answer` over a `compiled_v1` retrieval returns the manifest memory
  and the artifact id.

## Success criteria for the research question

Report, over 3 seeds and at least two real scopes from this machine's ledger:

1. Tokens per answer and latency, compiled vs token path, on the same probes.
2. Retention and abstention of the compiled path.
3. Backflow and over-deletion after retirement, against the base-model floor.

If (1) shows no saving over the token path, parametric consolidation is not
worth pursuing for this project and the spec is closed with that number. If
(1) shows a saving and (3) passes, the loop the review says nobody has closed
is closed, locally, with an audit trail. If (3) fails, the backflow rate is
itself the result.

## Open questions

1. Base model and runtime. `mlx-lm` LoRA on a 4B 4-bit model is the fastest
   path on this machine; is a 4B model a credible stand-in for the question,
   or does the result only count at 7B+?
2. Is a per-artifact `memories` row an acceptable way to make `why_answer`
   work for compiled answers, or does `retrieval_items` need a nullable
   `artifact_id` instead?
3. Should the scheduler's `E_min` be a constant or a percentile over scopes?
   Percentile means at least one scope always compiles, which is good for
   the experiment and wrong for production.
4. The restore step after a backflow test creates new memory rows. Is a
   dedicated `memory_action_types` value (`reinstate`) cleaner than
   superseding? It would be the first action that moves a memory back to
   active, and `memory_status` precedence would have to change.
5. Whether to run the self-study generation with the same local model that
   is trained, or a stronger hosted model. Local keeps content on the
   machine; hosted gives better probes. Default local.
