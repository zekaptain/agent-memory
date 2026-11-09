# Does retrieved memory help future tasks?

Status: design only. Building or running this experiment requires separate
approval. Judge agreement is not evidence of task benefit.

## Question and arms

Compare memory-enabled and memory-disabled agents on matched future coding
tasks. Both arms use the same model version, decoding settings, instructions,
tool permissions, dependency versions, hardware class, and task budget.
Only access to the frozen memory snapshot differs.

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
and scoring rubrics. Freeze them before opening held-out tasks. Document
excluded tasks with reason codes; do not replace hard cases after seeing
results. A task/environment defect can invalidate a pair only under a
pre-registered rule applied blind to the arm.

Include tasks where memory should abstain: unrelated work, expired or
withdrawn facts, conflicting evidence, facts valid only in another scope,
and plausible but unapproved tool-origin content. Include recurrence tasks
with independently verified earlier errors, and tasks with no useful prior
evidence. Report these strata separately as well as the registered aggregate.

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
| Total tokens | All input, output, reasoning tokens where available, including recall/tool payloads, retries, and preparation charged to the arm |
| Latency | End-to-end wall time from fresh-process start to termination, with model, retrieval, build, and retry time itemized |
| Retrieval coverage | Attempts with a recorded successful retrieval, miss, degraded retrieval, or missing record, reported separately |
| Citation coverage | Returned memories declared used, unresolved placeholders, and uniquely bound replies; no inference of causal reasoning |
| Abstention | Unnecessary or harmful retrieval/use on no-memory-needed tasks and appropriate non-use on those tasks |

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
intervals for repeat-error, harm, and abstention measures. Pre-register how
multiple secondary endpoints are handled; label exploratory slices.

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
