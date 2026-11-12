# Working on the MCP tools

This guide applies to changes under `mcp_server/`, on top of the root
`AGENTS.md` and `src/agent_memory/AGENTS.md`, which holds the rules for
recall and for who may approve a memory. The spec is
`docs/phase-1-spec.md`.

When adding an MCP tool or changing a tool's inputs, update the
registration assertions, the inputs listed in `tests/test_mcp.py`
(`TOOL_INPUTS`), and the usage docs too; graph work previously exposed
this gap. `build_server` hands each input to `MemoryTools` by name.
