# Working on admission

This guide applies to changes under `admission/`, to the `mem admit`
commands in `cli.py`, and to running those commands, on top of the root
`AGENTS.md` and `src/agent_memory/AGENTS.md`. Read
`src/agent_memory/judge/AGENTS.md` as well: admission keeps its files
through the judge's `RunStore` and shows its review through the judge's
loop, and the rules for both are there. The spec is
`docs/admission-spec.md`.

Admission lives in `admission/`: it reads human v4 experiment labels and
admits lessons as memories through the lifecycle, quarantined unless a
human approves them or auto mode is on, which the held-out bar has to
allow. Admission writes as agent `admission`, never as `judge`.
