# Working on the documents

This guide applies to changes under `docs/`, and to acting on what a
document there says, on top of the root `AGENTS.md`.

## What each document is

- `docs/phase-1-spec.md`: capture, spool, adapters, and MCP tools.
- `docs/judge-experiment-spec.md`: the judge design for new runs and its
  dated "As built" notes. The logs of exp1 to exp4 are in
  `docs/archive/judge-experiments.md`, and the definitions only stored
  runs still use are in `docs/archive/judge-earlier-versions.md`. Check
  the later amendments and current code before implementing.
- `docs/admission-spec.md`: production admission, fitted approver,
  admission review, and calibration/held-out safeguards.
- `docs/logic-hardening-spec.md`: lifecycle, coverage, attribution,
  `mem audit check` preflight, and rollout policy.
- `docs/judge-how-to.md`: step-by-step experiment commands.
- `docs/memory-utility-evaluation-spec.md`: design only for the memory
  usefulness study; nothing is implemented from it.
- `docs/parametric-consolidation-spec.md`: a draft research proposal, not
  implemented functionality or authorization to add its proposed migration.
- `docs/agent-memory-literature-review.md` and `docs/paper/`: research
  grounding, separate from the runtime's feature contract.
- `docs/adr/`: decisions that are hard to reverse and would surprise a
  reader.
- `docs/archive/`: superseded design text and experiment logs, kept for
  the record. Nothing there is a contract.

## Research edits

For research edits, verify claims and bibliography metadata against primary
sources. Separate peer-reviewed results, preprints, and vendor claims;
state what remains unverified rather than inheriting an earlier session's
confidence. If editing the paper, rebuild `docs/paper/paper.pdf` from the
LaTeX/BibTeX sources and check citations and layout warnings.
