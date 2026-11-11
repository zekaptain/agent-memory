# A stored judge run is reproduced under its own question set

A judge run's human labels were given against the question set,
projection, and drafter of its day, so its numbers only mean something
under those definitions. The code therefore keeps every earlier version
executable (`questions_v1` to `questions_v3`, earlier projections and
drafters). A stored row is read under the version it names and written
back unchanged, a report over an old run must come out byte for byte as
it did, and a report refuses a run whose verdicts were judged with
another question set. Old runs are read-only for draft, decide, and
label. The cost is code in `judge/report.py` and `judge/questions.py`
that no new run reaches and that must stay.

## Consequences

- Do not delete a version's code because no new run uses it. Compare the
  reports of the stored runs before and after a change to that code.
- A new question set gets a new version name. An existing one is never
  edited.
- The definitions of the earlier versions are kept in
  [archive/judge-earlier-versions.md](../archive/judge-earlier-versions.md).
