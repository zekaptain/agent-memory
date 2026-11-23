# Writing issues and projects

How an issue and a project are worded in this repo's tracker, after the
Linear Method (<https://linear.app/method>, and its chapter "Write issues
not user stories"). Where a skill's template says otherwise, this file
wins. `issue-tracker.md` says how each operation is done in Linear.

Say "issue" and "project", as Linear does, not "ticket" or "epic".

## An issue is one task

- An issue describes a task with a defined outcome: code, a document, a
  measurement, a decision record. Someone can tell when it is done.
- What is not yet a task does not go into the tracker. An undecided
  question goes under "Open decisions" in `docs/roadmap.md`. An idea
  that needs working out goes into a project's spec or a document.
- Exploration is written as its deliverable: "Write the spec for
  recalling events", "Record what Droid does with a hook's stdout".
- An issue that cannot start until something outside the tracker
  happens is not a task yet. Keep it in the project's spec until it can
  start.

## Title

- State the task, starting with a verb: "Recall events by text", not
  "An agent can recall events by text".
- Short enough to read in a list, about 60 characters at most.
- It makes sense alone. Do not point at another issue ("that block",
  "the re-check") without naming the thing.
- No user stories, in a title or anywhere else. Do not write "As a
  ..., I want ..., so that ...".
- Use the terms in `GLOSSARY.md`.

## Description

Write only what the assignee needs to do the task and what a reader
needs to know that it is being done. Shorter is better. A description
has, in this order, and only where each adds something:

1. One or two sentences on what changes, if the title does not say it
   all.
2. Acceptance criteria as a checklist. These stay, because an agent
   working alone needs to know when it is done. Each one can be checked
   by itself. Write the ones a reader would not assume; leave out what
   the guides already require of every change (tests, as-built notes,
   the glossary).
3. What is out of scope, when a reader would otherwise assume it is in.
4. A question still to settle, if there is one.

Leave these out of an issue:

- The reasons for the work and the evidence behind it, when the issue
  belongs to a project. They are written once, in the project's spec.
  An issue with no project carries its reason in a sentence, with a
  link.
- A footer or a sentence saying whether the work is approved. The
  workflow state and the triage label say that.
- A "Blocked by" section. Blocking is a relation in Linear.
- File paths, line numbers, and steps to follow. Describe behaviour and
  name interfaces; the skills' own rule on this stands.

When an issue rests on what someone said or wrote, quote the words and
link to them, and do not paraphrase. This covers public sources and
what the user says in a session about the work. It does not lift the
rule in `issue-tracker.md` about what never goes into an issue.

## An issue written for someone else

A report or a proposal for someone else to judge describes the problem,
not a solution: what happens, what was expected, and where it was seen.
It may stay in that form while it is `needs-triage`. Triage rewrites it
as a task, with a title that states the task, before it becomes
`ready-for-agent` or `ready-for-human`. The agent brief that `/triage`
posts follows this file too.

## Projects

- Work that takes more than two or three issues is a Linear project.
  One issue that stands alone needs none.
- Scope a project so that it can be finished in one to three weeks. If
  it cannot be cut down, split it into stages, each a project that
  leaves something usable.
- A project has a short spec as its body: why, what, and how, in that
  order. Aim for one screen. It holds the reasons, the evidence with
  its links, what is out of scope, and the decisions made. It has no
  list of user stories; where `/to-spec` asks for one, leave that
  section out and describe the behaviour under "what".
- The experience of whoever uses the feature is discussed in the spec.
  An issue does not repeat it.
- A project's spec does not replace a document under `docs/`. A design
  that the code must follow is still written there, and the spec links
  to it by path.
- A project has one lead.

## Owner and priority

Set both when an issue becomes `ready-for-agent` or `ready-for-human`,
not before.

- **Owner.** One assignee: the person answerable for the issue, also
  when an agent does the work.
- **Priority.** Give one line of reason in the triage comment. Ask two
  questions. Does this remove something that stops the ledger from
  being used or trusted (a blocker), or add something new (an
  enabler)? Does it matter now, for the roadmap's next approved work,
  or can it wait? A blocker to approved work is High. Urgent is for
  data loss and for capture that is failing.

## Keeping the backlog short

Not every idea needs an issue. One that is turned down at triage is
canceled with a comment, and is not kept in case. If it matters it will
come up again.
