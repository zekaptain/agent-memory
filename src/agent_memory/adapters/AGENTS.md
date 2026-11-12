# Working on harness adapters and `mem hooks`

This guide applies to changes under `adapters/` and to `hooks.py`, on top
of the root `AGENTS.md` and `src/agent_memory/AGENTS.md`. How records
reach the ledger is in `src/agent_memory/capture/AGENTS.md`. The spec is
`docs/phase-1-spec.md`. Paths here are relative to `src/agent_memory/`
unless they start with `docs/` or `tests/`.

- Write an adapter from payloads and transcripts recorded from the
  harness itself, never from its documentation alone. Droid's real
  payloads contradicted the hook mapping written from its documentation,
  and the Vibe adapter, written the same way and never checked, was
  removed.
- For Droid the transcript is the record of replies, tool calls,
  compactions, and what a subagent reports when its task is finished. A
  Droid hook payload carries no id for a tool call or for a finished
  task (checked against real payloads), so `PostToolUse` and
  `SubagentStop` record nothing and `PreCompact` records a note. Do not
  make a hook write an event the transcript also holds unless both carry
  the same harness id, as a prompt's `message_id` and a new session's id
  are. A hook-only event is keyed on `message_id` only when that id
  places it in the session (a resume, a request to compact). One that a
  message can raise several times, each saying something else, such as a
  notification, is keyed on the time the hook ran.
- One hook command does all of Droid's capture. `mem event` records the
  hook's own event and, at the hook events the adapter names
  (`transcript_due`: `Stop` and `SessionEnd`), imports what the
  transcript has gained since its last import. Do not give a hook a
  second command to run: `mem ingest` prints on stdout and can exit 2.
  The hook events to install at come from the adapter
  (`DroidAdapter.hook_events`); do not list them a second time.
- Droid's hooks file is its user's. `mem hooks` changes only a hook
  whose whole command is `mem ... event ...`. Every other hook stays as
  it is and where it is, also one that runs `mem event` inside a longer
  command. The file is written in one step, and one that is not in
  Droid's format is refused. Test `mem hooks` with `--file` in a
  temporary directory, never against `~/.factory/hooks.json`.
