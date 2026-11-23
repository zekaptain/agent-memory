# Does retrieved memory help future tasks?

Status: design only. Building or running this experiment requires separate
approval. Judge agreement is not evidence of task benefit.

## Question and arms

Compare memory-enabled and memory-disabled agents on matched future coding
tasks. Both arms use the same model version, decoding settings, instructions,
tool permissions, dependency versions, hardware class, and task budget.
Only access to the frozen memory snapshot differs.

Access to the snapshot is the memory tools together with the
instructions that come with them: the MCP server's instructions and each
tool's description. Those instructions are part of what differs between
the arms. Their wording is chosen in calibration and frozen with it (see
"Sampling and separation").

What memory gains depends on the model. In the one comparison across
models found, a vendor's research post that is not peer reviewed, the
gain in mean reward ran from 0.03 to 0.32 over six models, and the one
that gained least read its memory before answering less than half the
time unless told to ([The components of agent
memory](https://www.anthropic.com/research/components-of-agent-memory)).
Register the model or models before the run. With more than one, the
model is a registered factor: pairs are formed within a model, the
primary endpoint is reported for each, and the registration says whether
and how the models are pooled. With one, the report says that the result
holds for that model and is no estimate for a weaker or a stronger one.

Start each attempt in a fresh task environment. Clone the same pre-task
repository state into both arms; never reuse build artifacts, conversation
state, or task output across arms. The memory-enabled arm receives only
evidence recorded before the task cutoff. Hash the snapshot, its blobs,
instructions, environment manifest, and task manifest before assigning arms.
One arm's output must never enter the other arm's memory.

Use paired tasks and paired random seeds where supported. Randomize execution
order within each pair. Keep the evaluator blind to arm assignment and
retrieval/judge scores until task outcomes are finalized.

## Sampling and separation

Before running, register the task population, inclusion/exclusion rules,
sampling seed, sample size, repeated attempts per task, and family grouping.
Separate calibration and held-out tasks by time and task/session family,
not by random turns from the same session. Closely related bug reports,
repository revisions, and follow-ups belong to the same family.

Calibration chooses retrieval settings, memory selection policy, budgets,
scoring rubrics, and the wording of the instructions that come with the
memory tools: whether they say what memory is for and leave its use to
the agent, or give rules for when to recall and what to remember. The
same post reports that an instruction to read its memory before
answering, and to test what it then supposed, took the model that gained
least from 0.364 to 0.641, and that a rule to record only failures cut a
stronger model's gain by 0.23. So the wording is a setting like the
others. Freeze them before opening held-out tasks. Document excluded
tasks with reason codes; do not replace hard cases after seeing results.
A task/environment defect can invalidate a pair only under a
pre-registered rule applied blind to the arm.

Include tasks where memory should abstain: unrelated work, expired or
withdrawn facts, conflicting evidence, facts valid only in another scope,
and plausible but unapproved tool-origin content. Include recurrence tasks
with independently verified earlier errors, and tasks with no useful prior
evidence.

Include tasks whose success hinges on an exact value (an identifier, a
number, a path, a command) or on the output of a tool. A memory is a
rewrite of what happened, and an exact value is what a rewrite loses
first. One outside probe of a memory that compresses a conversation, a
public repository that is not peer reviewed, ran 38 support threads with
tool output, each condition once. 17 replies were confident and wrong,
against 4 with the whole conversation in context. The model that wrote
the memory was a small local one, and in a follow-up with a stronger one
the same memory answered 35 of the 38. The report marks both
comparisons as exploratory
([om-toolload](https://github.com/kashaff-codes/om-toolload)). Register
the rule that puts a task in this stratum before sampling.

Report these strata separately as well as the registered aggregate.

## Outcomes independent of the judge

The primary outcome is paired task success, checked with task-specific tests
and a prewritten rubric. Use hidden tests where appropriate, an independent
review of required behavior, and explicit safety/correctness failures.
A successful build alone does not establish task success.

Secondary measures:

| Measure | Definition |
|---|---|
| Repeat-error rate | Attempts repeating a pre-specified, independently verified earlier error, divided by recurrence opportunities |
| Harmful/stale memory use | Attempts where an evaluator links a declared/observed use to a concrete wrong action, plus attempted use of withdrawn, expired, or out-of-scope facts |
| Confident wrong reply | Attempts whose final reply gives the task as done or a fact as known, with no stated doubt or failure, that the task's checks show to be wrong, and where an evaluator links the wrong part to a declared/observed use of a memory; divided by attempts with a declared/observed use. Reported apart from wrong replies with no such use and from stated failures, beside the rate of confident wrong replies over all attempts of each arm |
| Total tokens | All input, output, reasoning tokens where available, including recall/tool payloads, retries, and preparation charged to the arm |
| Latency | End-to-end wall time from fresh-process start to termination, with model, retrieval, build, and retry time itemized |
| Retrieval coverage | Attempts in which the agent made no recall, and attempts with a recorded successful retrieval, miss, degraded retrieval, or missing record (a recall the transcript shows and the ledger does not hold), each reported separately |
| Citation coverage | Returned memories declared used, unresolved placeholders, and uniquely bound replies; no inference of causal reasoning |
| Abstention | Unnecessary or harmful retrieval/use on no-memory-needed tasks and appropriate non-use on those tasks |

An attempt of the memory-enabled arm in which the agent made no recall
stays in that arm for the primary outcome: having the tools is the
treatment. The difference over the attempts that did make a recall is an
exploratory slice, because which attempts recall is not assigned at
random.

Record token fields unavailable from a backend as missing, not zero. Retain
raw timing spans and failure categories. Separate timeouts, tool outages,
model refusals, evaluator failures, and ordinary incorrect answers.

## Analysis and stopping

The registration must name one primary endpoint and its practical benefit
threshold. Choose sample size from calibration variance and that threshold,
not from held-out effect estimates. Register exact per-attempt token,
wall-time, tool-call, and retry limits before the run. Budget exhaustion is
an outcome under the rubric, not a reason to rerun until success.

Estimate the paired success-rate difference with a 95% confidence interval.
Bootstrap whole task/session families, preserving pairs and repeats, with a
fixed registered seed and resample count. For small samples, also report an
exact paired binary analysis. Report task-family counts and effective sample
size; repeated seeds are not independent new tasks.

For tokens and latency, report paired differences and ratios, medians,
tail quantiles, and family-clustered intervals. Report denominators and
intervals for repeat-error, harm, confident wrong reply, and abstention
measures. Pre-register how multiple secondary endpoints are handled; label
exploratory slices.

Stop at the registered sample size or a pre-specified total resource budget.
Do not inspect significance repeatedly and stop on a favorable result.
A safety stop may halt the run after a registered harm threshold, with all
completed and interrupted attempts retained. Any sequential design needs its
own registered error-control rule before the first held-out attempt.

An inconclusive or negative result is valid. Report it without changing the
primary metric, dropping abstention tasks, or substituting judge agreement.
Publish a local aggregate report with uncertainty and limitations; exporting
private task content or transcripts needs explicit permission.

## Later work, in order

1. Verify and install Droid capture end to end. Add other adapters only from
   real, consented fixtures.
2. Build the local harness and run this registered usefulness experiment.
3. Tune retrieval or expand graph use only for measured failure modes.
4. Reconsider parametric consolidation only after retrieval shows value and
   withdrawal/backflow can be measured.
