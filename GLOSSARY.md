# Agent Memory

A local memory ledger shared by the AI coding agents on one machine. It exists so that for anything an agent said, you can ask what it was shown, where that came from, and what it used.

## Language

### Ledger and homes

**Ledger**:
The append-only record of everything captured and remembered, shared by every harness on the machine. Nothing in it is edited or deleted; current state is always derived from history.
_Avoid_: Database, store, memory DB

**Home**:
A directory that holds one ledger together with its token, spool, blobs, and exports.
_Avoid_: Data directory, workspace

**Production home**:
The home your agents use in daily work.
_Avoid_: Live ledger, real home

**Experiment home**:
A separate home used only for judge experiments, so that experiment data never mixes with production.
_Avoid_: Judge home, test home

### Harnesses and agents

**Harness**:
A coding-agent product whose sessions are captured, such as Factory Droid or Mistral Vibe.
_Avoid_: Agent (for the product), client, tool, IDE

**Agent**:
An identity registered to write to the ledger: one for each harness, one for a human at the CLI, and one for the judge. Subagents that a harness spawns write as that harness's agent.
_Avoid_: Harness (for the identity), actor, writer, model

### Capture

**Capture**:
Recording what happened in a harness session into the ledger.
_Avoid_: Ingestion, logging, tracking

**Session**:
One conversation in one harness, as recorded in the ledger.
_Avoid_: Thread, conversation, chat

**Event**:
One thing that happened in a session: a message, a tool call or its result, a recall, a memory write, a compaction, or an outcome.
_Avoid_: Log entry, record, message (for anything that is not a message)

**Turn**:
A user message and everything the agent did in response, up to the point where work on it closed.
_Avoid_: Exchange, round, step

**Capture record**:
The normalised form of one event on its way into the ledger, the same for every harness.
_Avoid_: Record (bare), payload, row

**Source key**:
The harness's own identifier for an event. Capturing the same source key twice yields one event, not two.
_Avoid_: External id, dedupe key

**Spool**:
The local queue where redacted capture records wait when the ledger cannot be reached.
_Avoid_: Queue, buffer, outbox

**Replay**:
Delivering spooled capture records to the ledger.
_Avoid_: Drain, flush

**Blob**:
Event content too large to keep inline, stored once and referenced from the event.
_Avoid_: Attachment, file

**Redaction**:
Replacing secrets with a marker before anything is persisted.
_Avoid_: Scrubbing, masking, sanitising

**Capture tier**:
What a harness is able to capture, declared once per harness. It describes capability, not what was actually captured.
_Avoid_: Completeness, coverage

**Capture observation**:
Evidence of what one capture pass actually covered in a session.
_Avoid_: Completeness flag

**Coverage**:
How much of a session the ledger actually holds, as shown by its capture observations rather than assumed from the capture tier.
_Avoid_: Completeness, tier

### Memory

**Memory**:
A curated statement that agents can recall, with a scope, a type, and a trust level.
_Avoid_: Note, knowledge, entry

**Scope**:
Where a memory applies: everywhere, or in one project.
_Avoid_: Namespace, project (as a synonym)

**Trust level**:
Who vouches for a memory: the user, an agent, a tool, or an import.
_Avoid_: Confidence, source

**Status**:
A memory's current standing, always derived from history and never stored: active, pending, quarantined, superseded, deduplicated, expired, or retired.
_Avoid_: State, lifecycle stage

**Active**:
Eligible to be returned by recall.
_Avoid_: Live, approved

**Quarantined**:
Held out of recall until it is approved. Memories that come from tools start here.
_Avoid_: Unreviewed, hidden

**Pending**:
Not yet in force, either because it starts in the future or because it is a correction that has not been accepted.
_Avoid_: Queued, draft

**Superseded**:
Replaced by an accepted correction.
_Avoid_: Overwritten, updated

**Deduplicated**:
Recorded as a duplicate of another memory, which is its canonical.
_Avoid_: Merged

**Expired**:
Past the end of its validity.
_Avoid_: Stale

**Retired**:
Permanently withdrawn because it is wrong or no longer useful.
_Avoid_: Deleted, removed

**Correction**:
A memory proposed as the replacement for another. Proposing it changes nothing about the memory it would replace.
_Avoid_: Update, edit

**Acceptance**:
The durable record that a correction has replaced its predecessor. Only acceptance makes the predecessor superseded.
_Avoid_: Approval (approval concerns one memory; acceptance concerns a replacement)

### Recall and audit

**Recall**:
An agent's request for memories relevant to a query.
_Avoid_: Search, lookup

**Retrieval**:
The ledger's record of one recall: what was asked, how it was ranked, and what came back, including nothing.
_Avoid_: Search result, recall log

**Retrieval item**:
One memory returned in a retrieval, with its rank, its score, and its status at that moment.
_Avoid_: Hit, result

**Miss**:
A retrieval that returned nothing. It is recorded like any other retrieval.
_Avoid_: Empty result

**Citation**:
An agent's declaration of which retrieval, memories, or other context it used.
_Avoid_: Reference, link

### Relation graph

**Relation**:
A memory that states a subject, predicate, and object connecting two entities.
_Avoid_: Edge, triple, link

**Entity**:
A named thing at either end of a relation.
_Avoid_: Node, concept

**Alias**:
Another name for an entity, kept when it is merged into its canonical name.
_Avoid_: Synonym, surface form

**Entity resolution**:
A model's proposal of which names refer to the same entity. It only proposes.
_Avoid_: Deduplication, clustering

**Entity merge**:
The recorded decision that an alias and a canonical name are the same entity. Earlier relations are superseded or retired, never rewritten.
_Avoid_: Rename, resolution

### Judge experiment

**Judge experiment**:
An offline study of whether a model can pick out the turns worth learning from, measured against human labels. It never writes memories.
_Avoid_: Eval, benchmark

**Projection**:
The bounded, deterministic text of a turn that a judge or drafter is shown.
_Avoid_: State, prompt, context, rendering

**Lesson**:
What a drafter writes about one turn: a rule, its kind, and the claims that support it. A lesson exists only before admission; what admission writes to the ledger is a memory.
_Avoid_: Memory (before admission), insight, takeaway

**Rule**:
The one sentence of a lesson that says what to do or know. On admission it becomes the memory's content.
_Avoid_: Lesson text, summary, instruction

**Claim**:
One statement in a lesson that supports its rule, citing the lines of the projection it rests on.
_Avoid_: Evidence, reason, justification

**Drafter**:
A model that writes a lesson from one turn, citing the evidence it relied on, or declines.
_Avoid_: Writer, summarizer, extractor

**Trigger**:
A deterministic signal, read only from what the ledger records, that a turn may hold something worth learning from.
_Avoid_: Heuristic, detector

**Stratum**:
The group of turns that share a trigger, used to sample an experiment evenly.
_Avoid_: Bucket, category

**Clean control**:
A completed turn with no trigger, included so that judges are also tested on turns with nothing to learn.
_Avoid_: Negative, baseline

**Verdict**:
A judge's recorded answers about one turn and its drafts, with the action that follows from them.
_Avoid_: Judgment, result, score

**Label**:
A human's answers to the same questions, treated as ground truth.
_Avoid_: Annotation, rating

**Calibration split**:
The labelled turns used to choose thresholds.
_Avoid_: Training set, dev set

**Held-out split**:
The labelled turns set aside to score the chosen thresholds, never used to choose them.
_Avoid_: Test set, validation set

**Freeze**:
Locking the thresholds chosen on the calibration split, after which held-out results count.
_Avoid_: Finalize, lock in

### Admission

**Admission**:
Turning lessons drafted from captured production turns into memories, quarantined unless a human or the approver lets them through.
_Avoid_: Promotion, ingestion, consolidation

**Approver**:
A model fitted on human labels that scores whether a drafted lesson should be admitted.
_Avoid_: Classifier, gate, judge

**Shadow mode**:
Admission in which the approver scores every lesson but every memory stays quarantined for a human to review.
_Avoid_: Dry run

**Auto mode**:
Admission in which the approver's approval makes a memory active without a human.
_Avoid_: Autopilot, unattended mode

### Model calls

**Egress**:
Sending session text to a model that is not on this machine. It is off unless the user opts in explicitly.
_Avoid_: Remote call, external access
