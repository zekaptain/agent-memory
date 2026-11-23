# Domain docs

How the engineering skills should read this repo's domain documentation
when exploring the codebase.

## Layout

Single-context: one glossary and one set of decision records.

- `GLOSSARY.md` at the repo root: the project's terms, and for each the
  words to avoid.
- `docs/adr/`: the decision records. Read the ones that touch the area
  you are about to work in.

## Use the glossary's vocabulary

When your output names a domain concept (in an issue title, a refactor
proposal, a hypothesis, a test name), use the term as `GLOSSARY.md`
defines it, and none of the words it lists under "Avoid".

If the concept you need is not in the glossary, that is a signal: either
you are inventing language the project does not use (reconsider), or
there is a real gap (note it for `/domain-modeling`).

## Flag ADR conflicts

If your output contradicts a decision record, say so explicitly rather
than overriding it silently:

> _Contradicts ADR 0008 (capture never drops a record and never blocks
> the agent), but worth reopening because…_

When to update the glossary or add a decision record is in the root
`AGENTS.md` under "Change discipline", and is not repeated here.
