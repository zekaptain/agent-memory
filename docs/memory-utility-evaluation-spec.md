# Does retrieved memory help future tasks?

Status: designed, with its build planned under "Building it" and the
first modules recorded under "As built". A read of a home in use, a model
that is not on this machine, and every run then each need the user's
go-ahead (see "What needs the user's go-ahead"). Judge agreement is not
evidence of task benefit.

## Question and arms

Compare agents with and without the memory tools on matched later coding
tasks: the memory arm and the no-memory arm. Both arms use the same model
version, decoding settings, instructions, tool permissions, dependency
versions, hardware class, and task budget. Only access to the memory set
differs. The terms are in `GLOSSARY.md` under "Usefulness experiment".

Access to the memory set is the memory tools together with the
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

Start each trial in a fresh task environment. Clone the same pre-task
repository state into both arms; never reuse build artifacts, conversation
state, or task output across arms. The memory arm recalls only from the
memory set, which holds nothing recorded after the cutoff. Hash the memory
set, its blobs, instructions, environment manifest, and task manifest before
assigning arms. One arm's output must never enter the other arm's memory.

Pair the two trials of a task, with the same random seed where supported.
Randomize execution order within each pair. Keep the evaluator blind to the
arm, to what was recalled, and to any judge's verdict until every grade is
final.

## Sampling and separation

Before running, register the task population, inclusion/exclusion rules,
sampling seed, sample size, repeated trials per task, and family grouping.
Separate calibration and held-out tasks by time and task/session family,
not by random turns from the same session. Closely related bug reports,
repository revisions, and follow-ups belong to the same family.

Calibration chooses retrieval settings, memory selection policy, budgets,
how checks are written, and the wording of the instructions that come with the
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
retired memories, conflicting evidence, facts valid only in another scope,
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

## Measures independent of the judge

The primary endpoint is paired task success, decided by each task's checks:
its tests, and a list of required behavior written before any trial of it.
Use hidden tests where appropriate, an independent review of required
behavior, and explicit safety/correctness failures. A successful build alone
does not establish task success.

Secondary measures:

| Measure | Definition |
|---|---|
| Repeat-error rate | Trials repeating a pre-specified, independently verified earlier error, divided by recurrence opportunities |
| Harmful memory use | Trials where an evaluator links a declared/observed use to a concrete wrong action, plus attempted use of a retired, expired, or out-of-scope memory |
| Confident wrong reply | Trials whose final reply gives the task as done or a fact as known, with no stated doubt or failure, that the task's checks show to be wrong, and where an evaluator links the wrong part to a declared/observed use of a memory; divided by trials with a declared/observed use. Reported apart from wrong replies with no such use and from stated failures, beside the rate of confident wrong replies over all trials of each arm |
| Total tokens | All input, output, reasoning tokens where available, including recall/tool payloads, retries, and preparation charged to the arm |
| Latency | End-to-end wall time from fresh-process start to termination, with model, retrieval, build, and retry time itemized |
| Recall and retrieval | Trials in which the agent made no recall, and trials with a retrieval that returned memories, a miss, a recall answered without the ledger (degraded), or a missing record (a recall the transcript shows and the ledger does not hold), each reported separately |
| Citations | Returned memories that a citation declares used, placeholders with no binding, and replies bound to one citation only; nothing is inferred about what caused a reply |
| Abstention | Unnecessary or harmful recall or use on tasks that need no memory, and appropriate non-use on those tasks |

A trial of the memory arm in which the agent made no recall stays in
that arm for the primary endpoint: having the tools is what the arm
tests. The difference over the trials that did make a recall is an
exploratory slice, because which trials recall is not assigned at
random.

Record token fields unavailable from a backend as missing, not zero. Retain
raw timing spans and failure categories. Separate timeouts, tool outages,
model refusals, evaluator failures, and ordinary incorrect answers.

## Analysis and stopping

The registration must name one primary endpoint and the difference that
would count as a practical benefit. Choose sample size from calibration
variance and that difference, not from held-out effect estimates. Register
exact per-trial token, wall-time, tool-call, and retry limits before the
run. A trial that runs out of budget is graded as it stands, and is not run
again until it succeeds.

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
A safety stop may halt the run after a registered count of harmful uses,
with all completed and interrupted trials retained. Any sequential design
needs its own registered error-control rule before the first held-out trial.

An inconclusive or negative result is valid. Report it without changing the
primary metric, dropping abstention tasks, or substituting judge agreement.
Publish a local aggregate report with uncertainty and limitations; exporting
private task content or transcripts needs explicit permission.

## Building it

Planned on 2026-11-22 (MEM-15). Nothing here is built, and what a stage
finds may change the stages after it.

### The experiment home and a run's files

The usefulness experiment has an experiment home of its own. It is not
the production home and not the judge experiment's. Its ledger is the
memory set. Beside the ledger it holds the tasks and, for each run, a
directory of files: the task manifest, the registration, a record for
each trial, the grades, the deviations, and the report.

The experiment adds no table, no migration, and no agent to any ledger,
and it writes to no home in use. A trial home is written by the agent of
the harness that runs the trial, as in daily work.

### The memory set

- The memory set is made once, at the cutoff: a copy of the database
  file and the blobs of a home that no server is serving at that moment.
  It is hashed when it is made and is not written to afterwards.
- Nothing is cut out of the copy. What the set holds is fixed by when it
  is made, so the tasks of a run are taken from work done after the
  cutoff. Cutting a ledger back to an earlier moment was turned down: it
  needs a rule for every table, and a way to show that nothing later
  came through.
- A recall writes. It records a retrieval, and it can carry out work
  that has fallen due on a correction; an agent can also remember,
  review, and relate. So each trial of the memory arm has a trial home,
  served for that trial alone. No trial recalls from the memory set
  itself or from a home another trial used. This keeps one trial's
  output out of another's memory, and makes "this trial made no recall"
  a count over one ledger.
- What a trial added to its trial home can still be read when the report
  is built.
- A memory's status is derived with the server's clock, so a memory
  whose validity starts or ends during a run can be active in one trial
  and not in the next. The memory set's manifest counts the memories
  whose validity starts or ends between the cutoff and the last day the
  registration gives the run, and the report gives that count as a
  limit.

### A trial

- A trial runs the harness with no interface, in a fresh clone of the
  task's repository at the task's commit. The clone is made for that
  trial and used by no other, and nothing one trial built or downloaded
  is there for the next.
- The two arms run the same command with the same settings. In the
  memory arm the memory tools are attached to the trial home, with the
  registered wording of their instructions and descriptions, and with
  the project the task names, so that a memory scoped to that project
  applies. In the no-memory arm there are no memory tools.
- A trial reads none of the user's own configuration of the harness:
  not the user's hooks, MCP servers, instruction files, skills, or
  subagents. The production home's capture hook in particular must not
  run, or every trial would be captured into the production ledger.
  What a trial reads instead is part of the environment manifest.
- A task's repository may carry harness configuration of its own. That
  is part of the task and the same in both arms, except that memory
  tools it configures are taken out of both: the only memory tools in a
  trial are the ones the memory arm is given.
- No capture hook runs during a trial, in either arm. When a trial of
  the memory arm has ended, its transcript is imported into its trial
  home, so that attribution can bind its citations.
- Tokens, wall time, tool calls, the last reply, and the changes to the
  repository are read the same way in both arms: from what the harness
  reports and from the clone, never from a ledger in one arm and from
  the harness in the other. A token count the harness does not report
  is recorded as missing.
- The runner enforces the registered limits that it can, and records
  the others. A trial that reaches a limit, is interrupted, or fails
  because of its environment is kept, with its failure category.
  Whether its pair is run again is decided by the registered rule and
  by nothing else.
- A trial sends the task, the repository, and in the memory arm what
  recall returned, to the model the harness runs. When that model is not
  on this machine, this is egress. The command that starts trials takes
  a model to be hosted unless the registered settings point the harness
  at a model server on this machine, and it refuses a hosted one unless
  the user has opted in, with the same check and the same opt-in as a
  hosted judge.

### Tasks

- A task is a directory in the experiment home, never in this
  repository: the repository and commit it starts from, the request,
  its checks, the project its memories are scoped to, when the work it
  was taken from was done, its family, and its strata. Tests among the
  checks that the agent must not see are added to the clone only after
  the agent has finished.
- The task manifest lists every task with a hash of each part, the
  families, the tasks left out with a reason code, and the split. It is
  written once, before the first calibration trial.
- The split is by time and by family: every calibration task comes from
  earlier work than every held-out task, and no family is in both.
- A held-out task is opened when its first trial starts. The runner
  refuses to open one in a run that has no registration.

### The registration

Before a held-out task is opened, the registration names:

- the primary endpoint, and the difference that would count as a
  practical benefit;
- the task manifest, by its hash: the population, the rules for taking
  a task in or leaving it out, the sampling seed, the sample size, the
  trials for each task, the families, the split, and the rule for each
  stratum, the one for tasks that hinge on an exact value among them;
- the memory set, by its hash, with its cutoff and the last day of the
  run;
- the model or models, the settings the harness lets a run fix, and
  whether and how models are pooled;
- the wording of the memory tools' instructions and descriptions, and
  every other setting that calibration chose;
- the environment manifest, by its hash: the harness and its version,
  the commit of this repository, the tools a trial may use, the
  hardware class, and what a trial reads in place of the user's
  configuration;
- the limits of a trial in tokens, wall time, tool calls, and retries,
  and which of them the runner enforces;
- the rule under which a defect of a task or of its environment makes a
  pair invalid;
- the analysis: the seed and the number of resamples of the bootstrap,
  how the secondary measures are handled, and which slices are
  exploratory;
- where the run stops: at the sample size or at a total budget, at what
  count of harmful uses a safety stop halts it, and the error-control
  rule of any sequential design.

The registration is one file in the run's directory, written once at
the freeze. The code that keeps a run's files has no way to change it.
Its SHA-256 is in the record of every held-out trial and in the report,
and it is committed to this repository, under "Runs" below, before the
first held-out task is opened. The hash gives nothing of the tasks
away, and the commit dates the registration. Whatever then turns out
different from the registration is appended to the run's deviations
with its date and its reason, and the report lists every one.

### Grades

- A task's checks are run on the trial's clone after the agent has
  finished, in the same way in both arms.
- The evaluator grades in the labeling helper. The evaluator is shown
  the request, the checks and what they gave, the changes to the
  repository, and the last reply; not the arm, not a call to a memory
  tool, and nothing from a ledger. Where what is shown gives the arm
  away, the evaluator says so, and the report counts those trials.
- Linking a wrong action, or a confident wrong reply, to a use of a
  memory needs the recalls. It is done after grading: it cannot be
  opened until every grade of the run is final, and it changes no
  grade.
- No model grades a trial. If a later change brings in a model's help,
  it goes through the shared model call and its egress check, and it
  stays out of the primary endpoint.

### What is taken from the judge experiment

Taken as it is:

- the guard that refuses the default home (`ensure_experiment_home`),
  but for its message, which names `mem judge`;
- the code that keeps a run's files whole (`judge/store.py`): a file
  replaced in one step, the lock, one writer for a file that grows,
  JSON lines, and a `RunStore` rooted in a directory of the
  experiment's own, as admission roots its own;
- the loop that shows a controller's screens (`drive`), with its plain
  interface and the Go helper;
- the check of whether a model server is on this machine, and the
  opt-in for one that is not (`endpoints.py`).

Taken as a pattern, in new code, because the judge's is written for
turns: sampling by a seeded shuffle of a sorted list, and the refusal
to show held-out results before the freeze.

Not taken:

- `mem judge import`. It replays transcript files into the judge's
  experiment home, to sample turns from. A trial's transcript goes in
  through `mem ingest --transcript`, the capture path.
- The shared model call, for running a trial. It asks for one JSON
  object and cannot run an agent that uses tools; the trial is the
  harness itself. It also has no token counts and records a missing
  cost as 0.0, where this design needs it missing.
- The judge's split, which is by turn.
- `judge/metrics.py`, which gives point estimates only. The helpers
  that feed the judge's report are not changed for this experiment,
  because stored reports are reproduced byte for byte.
- The labeling controller and its questions.

### What is new

A package of its own beside `judge/` and `admission/`, with a guide of
its own. Nothing under `judge/` or `admission/` imports it. Its modules:

- **Run files.** Files that are written once and hashed (the task
  manifest, the memory set's manifest, the registration), and the
  deviations, which only grow.
- **Memory set.** Making it, hashing it, and making a trial home from
  it.
- **Tasks.** Reading a task, the task manifest, and the split.
- **Trial runner.** One interface: a task, an arm, and the registered
  settings go in, and a trial's record comes out. Behind it are the
  clone, the trial home and its server, the limits, the import of the
  transcript, and what recall did. The harness is reached through an
  adapter for running it with no interface. Droid's is written from a
  real run, never from documentation. The tests run a second one, a
  scripted stand-in; no test calls a model.
- **Schedule.** Pairs in an order drawn from the registered seed, the
  order of the two arms drawn within each pair, and a run that can be
  continued after an interruption.
- **Grading.** A controller for the shared loop, and a trial as a
  subject in the helper's protocol, on the Python and the Go side.
- **Statistics.** Written against the standard library, with no new
  dependency: the paired difference and its interval from a bootstrap
  over whole task families, an exact paired test, the effective sample
  size, quantiles, and ratios. Each is tested against a worked example
  whose answer comes from outside the code.
- **Report.** The local aggregate. It holds no task content, no
  transcript text, and no memory text.

### What needs the user's go-ahead

Each of these waits for the user to say so, each time. None is covered
by accepting this plan.

| What | Why it waits |
|---|---|
| A count over the production home: active memories by scope and type, sessions captured, recalls and citations made | It reads a home in use. It says whether the ledger holds enough to measure, and so when the cutoff can be |
| Making the memory set | It copies a home in use, whose server is stopped while its file is copied |
| Writing tasks from work done after the cutoff | It reads sessions and private repositories |
| The first run of the harness with no interface, on a made-up task with no memory from a home | A hosted model, on the user's account |
| Calibration trials | Egress, when the model is hosted: tasks, repositories, and recalled memories. Cost |
| The freeze | The registration is the user's to accept |
| Held-out trials | Egress and cost, up to the registered stop |
| Showing anything but the local aggregate report | Task content and transcripts are private |

A run with a model on this machine sends nothing off it, and is still a
run: it waits for the user like any other.

### Stages

Each stage leaves something that can be used without the next one. A
stage's issues are in the tracker. A step that waits for a go-ahead is
kept in its project's spec until it can start.

1. **One trial** (MEM-17 to MEM-19). What a run of Droid with no
   interface reports and reads is recorded from a real run. The runner
   then takes one made-up task through both arms, the memory arm against
   a throwaway home, and writes two trial records, with what recall did
   counted in the memory arm's. It can be used to see, for one task,
   what having the memory tools changes.
2. **Memory set, tasks, and registration** (MEM-20 to MEM-23). A memory
   set can be made from a home and hashed. Tasks can be described,
   hashed, and split. Pairs are scheduled from a seed. A registration
   can be written at a freeze, and the runner holds a run to it. It can
   be used to run a calibration.
3. **Grades and report** (MEM-24 to MEM-27). Checks are run on finished
   trials, the evaluator grades blind, and the statistics and the
   report exist, tried on made-up trials. It can be used to read a
   calibration.
4. **The run.** In order: the count over the production home, the
   cutoff and the memory set, tasks from the work that follows,
   calibration trials, the freeze, held-out trials, grading, and the
   report. Every step waits for a go-ahead, so none is an issue yet.

The count over the production home is worth making first, before stage
1 is finished. If the production ledger holds too few active memories
to measure anything, the cutoff has to wait, and that is cheaper to
learn before the harness is built than after.

The checks of capture that were open when this plan was written
(MEM-14) were made on 2026-11-23. They concern a session in a terminal,
and a trial has no interface.

### What is not known yet

Stage 1 starts by recording these from a real run. Factory's
documentation speaks to the first three, and none of it has been
checked here against a run. The first six items were checked on
2026-11-25 (MEM-17, recorded under "As built" below); the seventh waits
for the memory set (MEM-20).

- What a run with no interface reports when it ends: the last reply,
  the session's id, how long it took, and token counts.
- How one run is given settings of its own, and whether those keep out
  the user's hooks and MCP servers.
- How one run is given an MCP server with an environment of its own.
- Which limits a run accepts, and whether a seed or a temperature can
  be fixed. The design pairs seeds only "where supported".
- Whether the MCP server's instructions reach the model, or only the
  tools' descriptions. The design counts both as part of what differs
  between the arms.
- Whether the citations in an imported transcript bind in a trial home.
  Attribution needs the MCP server's name as capture recorded it.
- How long a copy of the memory set takes, and how much disk the trial
  homes of a run need.

## Later work, in order

1. Finish verifying Droid capture end to end (MEM-14). Done on
   2026-11-23. Add other adapters only from real, consented fixtures.
2. Build the harness in the stages under "Building it", and run the
   registered usefulness experiment.
3. Tune retrieval or expand graph use only for measured failure modes.
4. Reconsider parametric consolidation only after retrieval shows value and
   withdrawal/backflow can be measured.

## As built

### Headless Droid probe (2026-11-25, MEM-17)

Probed with `droid 1.21.0` on macOS using `droid exec` against
`claude-sonnet-4-6` (default `reasoning_effort=high`) on synthetic tasks
in throwaway git repositories and a throwaway ledger home. A read-only
check of `~/.agent-memory` and its spool after the probes confirmed zero
captured sessions and zero spooled records in the production home.

#### What a run reports and what can be missing

- `-o json` prints one JSON object on stdout:
  - On success (exit `0`): `type` (`"result"`), `subtype` (`"success"`),
    `is_error` (`false`), `duration_ms`, `num_turns`, `result` (final
    assistant reply), `session_id`, and `usage` (`input_tokens`,
    `output_tokens`, `cache_read_input_tokens`,
    `cache_creation_input_tokens`, `factory_credits`, `thinking_tokens`,
    `ttft_ms`).
  - On failure (exit `1`, observed on authentication failure and on an
    `--auto medium` permission stop): `subtype` (`"failure"`), `is_error`
    (`true`), `num_turns` (`0`), and `result` carries the error message
    instead of an assistant reply. When failure happens before any model
    call (authentication failure), `usage.ttft_ms` is omitted and token
    counts are `0`. When failure happens after a model turn (permission
    stop on a tool call), `usage` reports the tokens and `ttft_ms` spent
    before termination.
  - `-o json` omits the resolved model ID, reasoning effort, and per-call
    tool invocations; those appear in `-o stream-json` and in the session
    transcript and sidecar settings file.
- `-o stream-json` prints NDJSON events: `system` (`subtype: "init"`,
  with `cwd`, `session_id`, `model`, `reasoning_effort`, `tools`),
  `message`, `reasoning` (when thinking is enabled), `tool_call`,
  `tool_result`, and a terminal `completion` (`finalText`, `numTurns`,
  `durationMs`, `session_id`, `timestamp`, `usage`) or `error`.
  - `system/init.tools` lists all built-in tools plus allowed MCP tools
    even when `--only-tools` restricts execution. The effective allowed
    list comes from `droid exec --list-tools --output-format json`
    (entries with `currentlyAllowed: true`) and is also recorded as
    `restrictToolIds` in `<session_id>.settings.json`.
- `input_tokens` counts non-cached input tokens only; total input tokens
  are `input_tokens + cache_read_input_tokens +
  cache_creation_input_tokens`. Droid reports end-to-end `duration_ms` /
  `durationMs` (and `assistantActiveTimeMs` in the sidecar) without
  itemizing model, retrieval, build, and retry durations; per-event
  timestamps are in the transcript, and retrieval latency is recorded in
  `retrievals.latency_ms`.

#### Isolating one run from user and repository configuration

- **Home override and macOS auth:** Inside Factory Desktop or another
  Droid session, `FACTORY_HOME_OVERRIDE` is inherited and takes
  precedence over `HOME`, so changing `HOME` alone does not isolate
  `.factory/`. Changing `HOME` on macOS also breaks Keychain lookup for
  `auth.v2.loginkeychain` (producing `Authentication failed` even with
  symlinked `auth.v2.*` files).
- **Working isolation recipe (macOS):** Keep real `HOME` (or set
  `FACTORY_API_KEY`), set `FACTORY_HOME_OVERRIDE=<trial_factory_home>` to
  a fresh per-trial directory, unset `FACTORY_SESSION_ID`, set
  `FACTORY_DROID_AUTO_UPDATE_ENABLED=false`, symlink `auth.v2.file`,
  `auth.v2.key`, and `auth.v2.loginkeychain` into
  `<trial_factory_home>/.factory/` (not needed with `FACTORY_API_KEY`),
  pass `--settings <path>` with `{"hooks": {"hooksDisabled": true},
  "cloudSessionSync": false}`, pass `--disable-builtin-skills`, and pass
  `--only-tools ...`.
- **Hooks:** `FACTORY_HOME_OVERRIDE` isolates `~/.factory/hooks.json` and
  `~/.factory/settings.json`. `hooks.hooksDisabled: true` in `--settings`
  additionally blocks repository `.factory/hooks.json` hooks (verified
  with a synthetic repo `SessionStart` hook, which fired without
  `hooksDisabled` and did not fire with it).
- **MCP servers and org connectors:** `FACTORY_HOME_OVERRIDE` isolates
  `~/.factory/mcp.json` (`droid mcp list` reported no configured
  servers), but org/account connectors and repository `.factory/mcp.json`
  servers remain exposed unless restricted. Passing
  `--only-tools Read,LS,Glob,Grep,Edit,Create,Execute` removes all org
  connectors, all repository `.factory/mcp.json` servers, and all other
  built-in tools (verified via `--list-tools --output-format json`: only
  those 7 tools had `currentlyAllowed: true`).
- **Instruction files:** `FACTORY_HOME_OVERRIDE` redirects both
  `.factory/` and `.agents/` (a synthetic
  `<trial_factory_home>/.agents/AGENTS.md` was read as
  `~/.agents/AGENTS.md`). Task clones should live in a neutral temp
  directory so Droid does not inherit an unrelated parent `AGENTS.md`.
- **Skills:** `FACTORY_HOME_OVERRIDE` keeps out user skills under
  `~/.factory/skills/`, `~/.agents/skills/`, and `~/.factory/plugins/`,
  and `--disable-builtin-skills` keeps out Factory built-in skills
  (verified: model reported zero visible skills).
- **Subagents:** `FACTORY_HOME_OVERRIDE` keeps out personal droids under
  `~/.factory/droids/`, leaving only built-in `worker` and `explorer`.
  Omitting `Task` from `--only-tools` disables subagent invocation
  altogether.

#### Attaching or withholding an MCP server with its own environment

- **Memory arm:** Write `<trial_factory_home>/.factory/mcp.json` with a
  `stdio` server under `mcpServers` (for example `"agent-memory"` running
  `mem-mcp` with `env` setting `AGENT_MEMORY_HOME`, `AGENT_MEMORY_PORT`,
  `AGENT_MEMORY_AGENT=factory-droid`, and
  `AGENT_MEMORY_PROJECT=<task.project>`), and pass
  `--only-tools <base_tools>,ToolSearch,MCP:agent-memory`. Verified that
  per-server `env` reaches both a synthetic probe server and `mem-mcp`.
- **Deferred tool loading:** MCP tools start deferred in `droid 1.21.0`
  and must be loaded via `ToolSearch` (`select:<server>___<tool>`), so
  `ToolSearch` is required in `--only-tools` for the memory arm.
- **Autonomy level:** An MCP tool declaring
  `ToolAnnotations(readOnlyHint=True)` succeeded under `--auto medium`,
  whereas MCP tools without `readOnlyHint=True` (including `mem-mcp`'s
  `recall` as currently shipped) exited `1` under `--auto medium`
  (`insufficient permission to proceed. Re-run with --auto high`). Under
  `--auto high`, both `agent-memory___recall` and `agent-memory___cite`
  succeeded. Because `cite`, `remember`, `review`, and `relate` write to
  the trial ledger and both arms must share identical permissions, both
  arms must run at `--auto high` (or `--skip-permissions-unsafe` inside
  an isolated container or VM).
- **No-memory arm:** Omit `"agent-memory"` from
  `<trial_factory_home>/.factory/mcp.json` and omit `MCP:agent-memory`
  from `--only-tools` (which also suppresses any memory server declared
  in a task repository's `.factory/mcp.json`). Verified in a paired probe
  that the no-memory arm had no memory tools available.

#### MCP server instructions vs tool descriptions

- In a canary probe against `claude-sonnet-4-6`,
  `MCPServer(instructions=...)` (`InitializeResult.instructions`) did
  **not** reach the model; only the tool description string reached the
  model.
- On `droid 1.21.0`, `mem-mcp`'s `INSTRUCTIONS` text in
  `src/agent_memory/mcp_server/__init__.py` is therefore not shown to the
  agent. Any memory-arm guidance varied during calibration must be placed
  in tool descriptions (or passed explicitly via Droid's
  `--append-system-prompt` / `--append-system-prompt-file` if the design
  is updated to use that channel).

#### Limits, seed, and temperature

- `droid exec` (`1.21.0`) accepts `-m/--model` and
  `-r/--reasoning-effort`, and accepts no CLI flags for wall-time limit,
  token cap, tool-call cap, random seed, or temperature. `settings.json`
  exposes only a per-request HTTP timeout (`llmRequestTimeoutMs`).
- The trial runner (`MEM-18`) must enforce wall-time limits externally
  (subprocess timeout and process-group kill). Live tool-call counting is
  possible by reading `tool_call` events from `-o stream-json`; aggregate
  token counts appear on the final `completion` event. Decoding seed and
  temperature cannot be fixed for hosted models via `droid exec`, so
  paired trials cannot share a seed on this harness.

#### Transcript location, import, and citation binding

- Droid writes each run's transcript to
  `<FACTORY_HOME_OVERRIDE>/.factory/sessions/<sanitized-cwd>/<session_id>.jsonl`
  (where `<sanitized-cwd>` is the resolved realpath of `cwd` with `/`
  replaced by `-`), alongside `<session_id>.settings.json`.
- Importing the memory-arm transcript into the throwaway trial home with
  `mem --home <trial_home> --port <port> ingest --harness droid
  --transcript <path>` exited `0` and stored the session with
  `model="claude-sonnet-4-6"`, the final assistant reply (`assistant_msg`
  with `modelId`, `apiProvider`, `reasoningEffort`, and
  `turn_complete=true`), and all `tool_call` / `tool_result` pairs
  (`ToolSearch`, `agent-memory___recall`, `agent-memory___cite`).
- Because no capture hook runs and Droid does not export
  `AGENT_MEMORY_SESSION` to stdio MCP servers, the trial home holds two
  sessions per memory-arm trial: the `mem-mcp` session
  (`mcp-factory-droid-<uuid>`, `project=<task.project>`, `model=NULL`,
  holding `retrievals`, `context_links`, and the placeholder
  `assistant_msg`) and the imported Droid session (holding the transcript
  events and `model`, with `project` derived from the clone directory
  name).
- On transcript import, cross-session attribution
  (`0006_cross_session_answer_bindings`) bound the MCP citation
  placeholder to the imported assistant reply in `answer_bindings` and
  `effective_answer_bindings` (`resolution_method="captured_cite_turn_v1"`,
  `reason_code="unique_captured_turn"`, with `captured_retrievals`
  populated).
- Attribution matches `<server>___cite`, `<server>___recall`, and
  `<server>___graph` using `AGENT_MEMORY_MCP_SERVER_NAME` (default
  `"agent-memory"`), so the server key in
  `<trial_factory_home>/.factory/mcp.json` must match that name during
  `mem ingest`.

#### What could not be seen

- Behavior on non-Claude models, models with reasoning disabled, and
  custom local model endpoints (`custom:...`), including whether
  `thinking_tokens` or `ttft_ms` are omitted and whether custom local
  model entries in `settings.json` support decoding parameters such as
  temperature or seed.
- Whether `droid exec` emits partial JSON or a terminal `completion` /
  `error` event when terminated mid-run by a runner timeout (`SIGTERM` /
  `SIGKILL`) or an upstream network or rate-limit failure.
- Whether `~/AGENTS.md` or `~/.claude/CLAUDE.md` in real `HOME` would be
  read when real `HOME` is preserved alongside `FACTORY_HOME_OVERRIDE`
  (none of those files existed on the probed machine).
- Whether org-managed settings (`org-managed-settings.cache.json`) can
  override `--settings` keys beyond the hooks, tools, skills, and
  subagents verified above, and how `--append-system-prompt` interacts
  with `--settings`.
- Copy duration and per-trial disk usage for a real memory set (waits
  for user go-ahead and `MEM-20`).

### Statistics (`usefulness/stats.py`, 2026-11-25, MEM-26)

`src/agent_memory/usefulness/stats.py` implements the standard-library
statistical functions with no new dependency and no file I/O:

- `paired_success_summary`: paired success-rate difference with a
  percentile bootstrap interval resampling whole task families with
  replacement (preserving pairs and repeated trials together) for a
  registered seed and resample count, alongside `exact_mcnemar_test`
  (exact two-sided and one-sided binomial test on discordant pairs) and
  `exact_family_sign_test` (exact sign test over family net differences,
  which matches McNemar when every family has one pair and respects
  family independence when families have repeats).
- `cluster_summary`: task-family count, one-way random-effects ANOVA
  intra-class correlation (Donner & Klar / Fleiss), Kish unequal-cluster
  design effect, and effective sample size (`N / DEFF`, clamping
  negative sample ICC to 0 and defaulting conservatively to `rho = 1`
  when repeated trials show zero variance).
- `paired_numeric_summary` and `distribution_summary`: paired
  differences (mean, median, tail quantiles), paired ratios (ratio of
  means and per-pair median/tail quantiles over pairs with positive
  no-memory denominator), and per-arm distributions, all with
  family-clustered bootstrap intervals and Hyndman & Fan type-7 linear
  quantiles.
- `clustered_rate_summary` and `wilson_interval`: denominator, missing
  count, family-clustered bootstrap CI, and closed-form Wilson score
  interval for secondary rates.
- Missing values (`None`) are excluded from estimates and counted
  explicitly; non-finite floats and booleans passed as numbers are
  rejected. Every statistic is checked against an independent worked
  example in `tests/test_usefulness_stats.py`.

## Runs

None yet. A run is listed here with the SHA-256 of its registration when
it is frozen, and with the date of its report when it ends.
