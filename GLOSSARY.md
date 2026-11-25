# Agent Memory

A local memory ledger shared by the AI coding agents on one machine. It exists so that for anything an agent said, you can ask what it was shown, where that came from, and what it used.

## Language

### Ledger and homes

**Ledger**:
The append-only record of everything captured and remembered, shared by every harness on the machine. Nothing in it is deleted or moved out, and nothing is edited except by a blanking; current state is always derived from history.
_Avoid_: Database, store, memory DB

**Blanking**:
The removal by a human of text that must not be kept, from an event or a memory already written. The row stays, and the ledger records who blanked it, when, and why.
_Avoid_: Redaction (which happens before anything is persisted), deletion, erasure, purge

**Home**:
A directory that holds one ledger together with its token, spool, blobs, and exports.
_Avoid_: Data directory, workspace

**Production home**:
The home your agents use in daily work.
_Avoid_: Live ledger, real home

**Experiment home**:
A separate home used only for one experiment, so that experiment data never mixes with production. The judge experiment has one and the usefulness experiment has another.
_Avoid_: Judge home, test home

### Harnesses and agents

**Harness**:
A coding-agent product that agents work in, such as Factory Droid or Amp. Capture reaches only a harness it has been built for and checked against.
_Avoid_: Agent (for the product), client, tool, IDE

**Agent**:
An identity registered to write to the ledger: one for each harness, one for a human at the CLI, one for the judge, and one for admission. Subagents that a harness spawns write as that harness's agent.
_Avoid_: Harness (for the identity), actor, writer, model

### Capture

**Capture**:
Recording what happened in a harness session into the ledger.
_Avoid_: Ingestion, logging, tracking

**Session**:
One stretch of activity by one agent, recorded in the ledger as a unit. A conversation in a harness is the usual kind; a judge run, an admission day, and a graph or review command run at the CLI are sessions too.
_Avoid_: Thread, chat, conversation (which is only one kind of session)

**Event**:
One thing that happened in a session: a message, a notice, a tool call or its result, a recall, a memory write, a compaction, a judge's verdict, or an outcome.
_Avoid_: Log entry, record, message (for anything that is not a message)

**Turn**:
A user message and everything the agent did in response, up to the point where work on it closed.
_Avoid_: Exchange, round, step

**Outcome**:
The harness's record of how a turn ended. A turn closes at its outcome, or without one at the next user message or at the end of the session.
_Avoid_: Result, action (which is what is done with a lesson), verdict

**Reply**:
An agent's message to its user.
_Avoid_: Answer (kept only in stored names; an answer is what a judge or a labeler gives to a question), response, output, assistant message

**Caller**:
Who started a session: a human, or an agent working in another session.
_Avoid_: Parent (for a human), owner, user

**Spawned session**:
A session whose caller is an agent: it was started from another session to do one task and report back.
_Avoid_: Subagent session, child session, task (Droid's name for the tool that starts one)

**Author**:
Who wrote a message that a transcript gives the role of user: a human, a parent agent, or the harness. The role alone does not say which.
_Avoid_: Provenance (which is the chain behind a reply), speaker, user (for the author)

**Notice**:
Something a harness tells the user and does not give the agent, such as a request the model provider refused or a usage limit that was reached. It is part of the session and no message of the conversation, whatever role a transcript gives it: no turn starts at it.
_Avoid_: User message, harness message (both of which the agent is given), alert

**Capture record**:
The normalised form of one event on its way into the ledger, the same for every harness.
_Avoid_: Record (bare), payload, row

**Hook**:
A command a harness runs when something happens in a session, handing it a payload that describes what happened. Hooks bring the ledger what the transcript does not hold, or not soon enough: a prompt as it arrives, a notification, the start and end of a session.
_Avoid_: Callback, trigger, plugin

**Extension**:
A module a harness loads into its own process and calls when something happens in a session, handing it what happened as values. It brings the ledger what a hook brings, for a harness that runs no hook commands.
_Avoid_: Plugin (Amp's name for one), hook (which is a command), add-on

**Hook event**:
A point in a session at which a harness runs its hooks or calls its extensions, such as the arrival of a prompt or the end of a turn. It is the harness's occasion and not an event in the ledger: capture records something at some hook events and nothing at others.
_Avoid_: Event (bare, for this), hook (for the occasion), trigger, lifecycle event

**Transcript**:
The harness's own file of a session. Where a harness writes one, it is the record of replies, tool calls, and compactions, and importing it is how they reach the ledger.
_Avoid_: Log, history, session file

**Import**:
Capturing from a transcript: reading what it holds, or what it has gained since the last import, into the ledger.
_Avoid_: Ingest (kept only as the name of the command), sync, load

**Source key**:
The harness's own identifier for an event, or, for an event the harness does not identify, the moment its hook ran. Capturing the same source key twice yields one event, not two.
_Avoid_: External id, dedupe key

**Spool**:
The local queue where redacted capture records wait when the ledger cannot be reached.
_Avoid_: Queue, buffer, outbox

**Replay**:
Delivering spooled capture records to the ledger.
_Avoid_: Drain, flush

**Delivery**:
Handing capture records to the ledger, or to the spool when the ledger cannot be reached. Every record ends up written, spooled, or rejected; none is dropped.
_Avoid_: Ingest, send, write (for the whole act)

**Rejected record**:
A capture record the ledger refuses, or one that refers to a refused record. It is kept aside with its reason and is delivered again only after a retry.
_Avoid_: Failed record, bad line, poison record, dropped record

**Retry**:
Returning rejected records to the spool on request, so that the next replay delivers them again.
_Avoid_: Replay (which delivers what the spool already holds), requeue, resend

**Capture error**:
The event that records a rejection in the ledger, so that the gap it leaves in a session is visible.
_Avoid_: Failure, exception

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
What the agent that wrote a memory says it rests on: the user's word, the agent's own inference, a tool's output, or an import. It is the writer's claim, and nothing checks it.
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

**Review action**:
A recorded act on one memory after it was written: quarantining it, approving it, retiring it, confirming that it still holds, or flagging that it contradicts another. Review actions are how a memory's status changes while the memory itself stays as written.
_Avoid_: Review (bare, which is an action on a lesson), memory action, moderation

**Approval**:
The decision that lets a quarantined memory be recalled. A human gives it, never an agent; the only approvals that are not a human's are a carried approval and the approver's in auto mode.
_Avoid_: Acceptance (which concerns a replacement), confirmation (which says a memory still holds and releases nothing)

**Carried approval**:
The approval a relation keeps when an entity merge rewrites it: the new relation is approved because the one it replaces was.
_Avoid_: Inherited approval, re-approval

**Awaiting approval**:
Said of a memory whose standing an approval would change: a quarantined memory, or a correction from a tool's output that has not been accepted.
_Avoid_: Waiting (bare, which is said of records in the spool), review queue, unreviewed, pending (which also covers a memory that starts in the future)

### Recall and audit

**Recall**:
An agent's request to be shown memories. Asking with a query, walking relations out from an entity, and listing every relation in a scope are all recalls.
_Avoid_: Search, lookup, graph traversal (a walk is a recall with its own strategy)

**Strategy**:
The way one recall chooses and orders memories: by matching the query's text, by importance when there is no query, by distance from an entity, or by listing every relation.
_Avoid_: Ranking, algorithm, search mode

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
An agent's declaration of which retrieval, memories, or other context a reply used.
_Avoid_: Reference, link

**Placeholder**:
An event that stands for a reply the ledger does not hold yet. A citation made before its reply is captured points at one.
_Avoid_: Stub, dummy event, pending reply

**Attribution**:
Working out which captured reply a citation was made for. It rests on what capture recorded about the turn, and it stays unresolved when that is not enough. It shows what the agent declared it used, never what caused the reply.
_Avoid_: Causal attribution, matching, inference

**Binding**:
The record that a placeholder turned out to be a particular captured reply. A later binding can replace an earlier one.
_Avoid_: Answer binding (kept only in stored names), link, match

**Provenance**:
The chain from a reply back to what the agent was shown and where that came from: its citations, the retrievals behind them, the memories returned, and the events those memories were written from.
_Avoid_: Lineage, audit trail, authorship (who wrote a message is its author)

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

**Judge**:
A model, or a fixed set of rules, that answers the turn questions and the draft questions.
_Avoid_: Backend (kept only as the name of the option and in stored names), decision model, classifier, approver

**Run**:
One pass of an experiment over one sample, together with everything produced about it. In the judge experiment the sample is of turns, and a run holds their drafts, verdicts, labels, and a report. In the usefulness experiment it is of tasks, and a run holds their trials, grades, and a report.
_Avoid_: Experiment (the study as a whole), batch, job

**Item**:
One turn sampled into a run.
_Avoid_: Candidate (kept only as the name of the command that samples them), example (which is what the approver is fitted on), case

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

**Draft**:
What one drafter produced for one item: a lesson, or a decline when it found nothing worth keeping.
_Avoid_: Lesson (a draft that declines holds none), proposal, candidate

**Trigger**:
A deterministic signal, read only from what the ledger records, that a turn may hold something worth learning from.
_Avoid_: Heuristic, detector

**Stratum**:
A group within a sample that is drawn or reported by itself: in the judge experiment the turns that share a trigger, used to sample evenly; in the usefulness experiment the tasks that a registered rule puts together.
_Avoid_: Bucket, category

**Clean control**:
A completed turn with no trigger, included so that judges are also tested on turns with nothing to learn.
_Avoid_: Negative, baseline

**Verdict**:
A judge's recorded answers about one draft of one turn, including its answers about the turn itself.
_Avoid_: Judgment, result, score, decision (kept only as the kind of event that holds a verdict in the ledger)

**Action**:
What is done with a lesson: write it, send it to review, or discard it. A verdict gives an action when it is read at a set of thresholds, so the same verdict can give a different action at different thresholds. The approver's score gives one too.
_Avoid_: Outcome, result, decision (kept only in stored report headings)

**Review**:
The action between write and discard: the lesson is kept for a human to decide. In admission the lesson becomes a quarantined memory.
_Avoid_: Hold, escalate, human review

**Turn questions**:
The questions a judge answers about a turn itself. They are asked once for a turn, and every verdict of that turn carries the answers.
_Avoid_: Stage A (kept only in report headings and stored verdicts, which already say it)

**Draft questions**:
The questions a judge answers about one draft of a turn.
_Avoid_: Stage B (kept only in report headings, which already say it)

**Question set**:
One version of the turn questions and draft questions. A run is judged, labeled, and reported with the one it was built with.
_Avoid_: Rubric, schema, questions version

**Answer**:
What a judge or a labeler gives to one question.
_Avoid_: Reply (which is an agent's message to its user), response, score

**Write gate**:
A draft question whose answer must pass its threshold before the lesson is written.
_Avoid_: Gate (for the approver, for the held-out bar, or for a threshold on a judge's confidence), check, filter, criterion

**Threshold**:
The value a judge's answer, or the approver's score, is compared with to choose an action.
_Avoid_: Cut, cutoff (which is the moment a memory set is made), gate

**Review band**:
The range of values that give the action review: a judge's answer too close to its threshold, or an approver's score between its two thresholds.
_Avoid_: Uncertainty band, grey zone, margin

**Unanswered question**:
A question a judge was asked but gave no usable answer to. It is evidence of nothing, so no lesson is written on the strength of one.
_Avoid_: Missing answer, fallback, default answer, no answer (which means the question was never asked)

**Label**:
A human's answers to the same questions, treated as ground truth.
_Avoid_: Annotation, rating

**Labeler**:
The person whose answers become labels.
_Avoid_: Reviewer, rater, annotator

**Sitting**:
One stretch of a person's work at labeling, at admission review, or at grading, from opening it to quitting.
_Avoid_: Session (which is a unit recorded in the ledger), labeling session, pass

**Calibration split**:
The part of a run's sample used to choose what is then frozen: labeled turns for a judge's thresholds, tasks for the settings of a usefulness run.
_Avoid_: Training set, dev set

**Held-out split**:
The part of a run's sample set aside to score what was frozen, never used to choose it.
_Avoid_: Test set, validation set

**Freeze**:
Locking what was chosen on the calibration split, after which held-out results count: a judge's thresholds, or the registration of a usefulness run.
_Avoid_: Finalize, lock in

**Partial freeze**:
A freeze made while some turns of the calibration split had no label, or a label with questions left. It has to be asked for, and the run records how many turns it went without.
_Avoid_: Early freeze, forced freeze

**Live day**:
One day's new turns, judged with a run's frozen thresholds and kept with that run.
_Avoid_: Sub-run, child run

### Admission

**Admission**:
Turning lessons drafted from captured production turns into memories, quarantined unless a human or the approver lets them through.
_Avoid_: Promotion, ingestion, consolidation

**Admission day**:
One day's captured turns as admission works through them, together with the approver, drafter, and mode that were in force when the day began.
_Avoid_: Run (admission works in days), batch, live day (which belongs to the judge experiment)

**Approver**:
A model fitted on examples that scores whether a drafted lesson should be admitted.
_Avoid_: Classifier, gate, judge

**Example**:
A lesson together with a human's judgment of it, which the approver is fitted on. Examples come from a run's labels and from admission review.
_Avoid_: Label (which belongs to a run), training row, sample

**Admission review**:
A human's blind rating of the lessons of an admission day, a sample of the discarded ones among them. Each answer approves or retires the memory the lesson became, and is kept for the approver to be fitted on.
_Avoid_: Review (bare), review queue, labeling (which belongs to the judge experiment)

**Shadow mode**:
Admission in which the approver scores every lesson but every memory stays quarantined until a human approves it.
_Avoid_: Dry run

**Auto mode**:
Admission in which the approver's approval makes a memory active without a human.
_Avoid_: Autopilot, unattended mode

**Held-out bar**:
What the approver's results on the held-out split must reach before auto mode may be turned on.
_Avoid_: Auto gate, held-out gate, precision bar

### Usefulness experiment

**Usefulness experiment**:
An offline study of whether having the memory tools helps an agent with a later task, measured by the task's own checks and never by a judge. It writes to no home in use.
_Avoid_: Usefulness study, utility evaluation, eval, benchmark

**Task**:
A piece of coding work an agent is given in the usefulness experiment: a repository at a fixed state, a request, and the checks that decide success.
_Avoid_: Case, problem, prompt, item (which is a turn sampled into a judge run)

**Checks**:
What decides whether a trial succeeded, written before any trial of the task: its tests, and a list of the behavior required.
_Avoid_: Rubric, grader, oracle

**Task family**:
Tasks close enough that one gives another away: reports of the same bug, revisions of the same change, a request and its follow-up. A family is never split between the calibration split and the held-out split, and its tasks do not count as independent of one another.
_Avoid_: Cluster, group, session family

**Arm**:
One of the two ways a task is run: the memory arm, with the memory tools, and the no-memory arm, without them. Nothing else differs between the two.
_Avoid_: Condition, group, treatment, control (a clean control is a turn in the judge experiment), baseline

**Trial**:
One go of an agent at one task in one arm: started in a fresh environment, ended by the agent's last reply or by a limit, and kept with everything recorded about it.
_Avoid_: Attempt (which is one request within a model call), run (which is the whole pass over a sample), episode, rollout

**Pair**:
Two trials of one task, one in each arm, alike in everything but the arm. A task tried more than once gives more than one pair.
_Avoid_: Match, comparison

**Cutoff**:
The moment a memory set is made. Nothing recorded after it is in the set, and every task of a run comes from work done after it.
_Avoid_: Freeze (which locks what calibration chose), as-of date, snapshot time

**Memory set**:
The ledger that the memory arm of a run recalls from: a copy of a home made at the cutoff and never changed afterwards.
_Avoid_: Snapshot (which is what the server's export to Parquet is called), memory snapshot, corpus, fixture

**Trial home**:
The throwaway home that one trial of the memory arm recalls from and writes to: its own copy of the memory set, used by no other trial.
_Avoid_: Sandbox, scratch home, experiment home (which holds the memory set and the runs)

**Registration**:
The record, made at the freeze, of everything about a run that must not change once a held-out task is opened: its tasks and how they were chosen, the memory set, the model, the wording the memory tools come with, the limits of a trial, and how the results are analyzed.
_Avoid_: Pre-registration, protocol, plan, manifest (which lists and hashes one thing, such as the tasks)

**Evaluator**:
The person who grades trials, without being told the arm.
_Avoid_: Labeler (who answers the judge experiment's questions), judge, reviewer, rater

**Grade**:
What is recorded about how one trial ended: whether it met its task's checks, and if not, which kind of failure it was.
_Avoid_: Outcome (which is the harness's record of how a turn ended), result, score, label, verdict

### Model calls

**Model call**:
Asking a model for one piece of work: one set of answers from a judge, one lesson from a drafter, or one block of entity clusters. It may take several attempts.
_Avoid_: Request, inference, completion, prompt

**Attempt**:
One request to a model within a model call, and what came back. A later attempt changes what the earlier one lacked: more room, or no thinking.
_Avoid_: Retry (which returns rejected records to the spool), try

**Model server**:
The program a model call is sent to, which runs the model: Ollama, llama-server, or OpenRouter. A local one runs on this machine; a hosted one does not, and sending it session text is egress.
_Avoid_: Server (bare, which is the process that serves the ledger), provider, endpoint, host, remote (for hosted)

**Model reply**:
What a model sends back in one attempt: its text, any thinking that came before it, and whether it was cut off.
_Avoid_: Reply (bare, which is an agent's message to its user), response, completion, output, answer

**Thinking**:
Text a model writes for itself before its reply. It uses the same token budget as the reply, so a model can spend the whole budget thinking and say nothing.
_Avoid_: Reasoning (kept only in "reasoning judge"), chain of thought

**Cut off**:
Said of a model reply that ended because its token budget ran out, not because the model finished.
_Avoid_: Truncated, incomplete

**Egress**:
Sending session text, or memories written from it, to a model that is not on this machine. It is off unless the user opts in explicitly. What a harness sends to its own model in daily work is not egress; what a trial sends is, because the experiment starts the trial.
_Avoid_: Remote call, external access
