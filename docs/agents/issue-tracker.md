# Issue tracker: Linear

Issues and specs for this repo live in Linear, in the team `agent-memory`
(key `MEM`). Use the Linear tools the harness is connected to. If a
session has none, stop and say so; do not fall back to GitHub Issues or
to files.

## Conventions

- **Create an issue**: in the team above, with a title and a Markdown
  description.
- **Read an issue**: by its identifier (`MEM-12`), with its comments,
  labels, and relations.
- **List issues**: filter by the team, then by label or workflow state.
- **Make an issue a sub-issue**: set its parent.
- **Blocking**: Linear's native relation. "A blocks B" is created once,
  on A. A ticket is unblocked when every issue that blocks it is Done or
  Canceled.
- **Comment on an issue**: add a comment to it.
- **Apply / remove labels**: the triage roles are labels on this team
  (see `triage-labels.md`). Add and remove them one at a time; never
  replace an issue's whole label set.
- **Close**: comment with the explanation, then move the issue to Done,
  or to Canceled for `wontfix`.
- **Create a project**: in the team above, with a name, a one-line
  summary, a lead, and the spec as its body. Put an issue into it by
  setting the issue's project.

The team holds this repo's work only. File nothing for this repo into
another team, and nothing for another project into this one.

How an issue and a project are worded, when work becomes a project, and
when an issue gets an owner and a priority are in `issue-writing.md`.

## Where an issue waits

- **Triage**: a proposal or a report nobody has approved, with
  `needs-triage` or `needs-info`.
- **Backlog**: approved work not yet picked up, with `ready-for-agent`
  or `ready-for-human`. Nothing unapproved waits here.
- **Todo** and after: work someone has taken.

Triage is a setting of the team in Linear, and the connected tools
cannot turn it on. If the team has no Triage state, file a proposal in
Backlog with `needs-triage` and say so.

## Pull requests as a triage surface

**PRs as a request surface: no.** _(Set to `yes` if this repo treats external PRs as feature requests; `/triage` reads this flag.)_

Pull requests live on GitHub (`zekaptain/agent-memory`), not in Linear.

## How issues relate to the roadmap

`docs/roadmap.md` stays the record of what is built, what is approved,
what is known to be wrong, and what is undecided. An issue is a unit of
work and approves nothing by existing. Apply `ready-for-agent` only to
work the roadmap's "Next approved work" covers or the user has approved;
a proposal is filed as `needs-triage`, in the Triage state (see "Where
an issue waits"). A defect found and not fixed
still goes into the roadmap, as the root `AGENTS.md` says.

The repository is public and the Linear workspace is not. In a committed
file, name an issue by its identifier (`MEM-12`), never by its URL.

## What never goes into an issue

Linear is a hosted service. Do not paste session text, transcripts,
ledger rows, tokens, or model outputs from a home into an issue or a
comment; use a small synthetic example.

## When a skill says "publish to the issue tracker"

Create a Linear issue in the team above.

## When a skill says "fetch the relevant ticket"

Read it as in **Read an issue** above.

## Wayfinding operations

Used by `/wayfinder`. The **map** is a single issue with **child** issues as tickets.

- **Map**: a single issue labelled `wayfinder:map`, holding the Notes / Decisions-so-far / Fog body.
- **Child ticket**: a sub-issue of the map, labelled `wayfinder:<type>` (`research`/`prototype`/`grilling`/`task`). Once claimed, the ticket is assigned to the driving dev.
- **Blocking**: see **Blocking** under Conventions.
- **Frontier query**: list the map's open sub-issues, drop any with an open blocker or an assignee; first in map order wins.
- **Claim**: assign the issue to yourself, the session's first write.
- **Resolve**: comment on the issue with the answer, move it to Done, then append a context pointer (gist + link) to the map's Decisions-so-far.
