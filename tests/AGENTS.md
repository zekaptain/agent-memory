# Working on the tests

This guide applies to changes under `tests/`, on top of the root
`AGENTS.md`. A test also follows the guide of the code it tests; the code
map in the root `AGENTS.md` pairs each test file with that code.

- Tests use real DuckDB databases and Quack servers, not just mocks.
  `tests/conftest.py` supplies temporary homes, direct connections, live
  servers on free ports, and an unreachable server configuration; reuse
  these and `tests/support.py`.
- Mock model HTTP in regression tests rather than making paid or
  private-data-bearing calls.
