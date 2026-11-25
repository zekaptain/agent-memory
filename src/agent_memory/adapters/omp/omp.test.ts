import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { chmodSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import {
  AGENT_ID,
  OmpMapper,
  RECORD_VERSION,
  extractContentText,
  renderToolCall,
} from "./mapper.ts";
import type {
  CaptureRecord,
  OmpAgentInfo,
  OmpContext,
  OmpHookPayload,
  OmpSessionEntry,
} from "./mapper.ts";
import { RecordSink, buildMemArgs } from "./sink.ts";
import { HOOK_EVENTS, createRuntime, loadSinkConfig } from "./index.ts";
import type { OmpExtensionAPI } from "./index.ts";

class FakeSessionManager {
  sessionId: string;
  sessionFile: string;
  entries: OmpSessionEntry[];
  headerTs?: string;

  constructor(
    sessionId: string,
    sessionFile: string,
    entries: OmpSessionEntry[] = [
      { type: "model_change", id: "11111111", timestamp: "2026-11-25T03:00:00.000Z" },
      { type: "thinking_level_change", id: "22222222", timestamp: "2026-11-25T03:00:00.001Z" },
    ],
  ) {
    this.sessionId = sessionId;
    this.sessionFile = sessionFile;
    this.entries = [...entries];
  }

  push(entry: OmpSessionEntry): void {
    this.entries.push(entry);
  }

  getSessionId(): string {
    return this.sessionId;
  }

  getSessionFile(): string {
    return this.sessionFile;
  }

  getEntries(): OmpSessionEntry[] {
    return this.entries;
  }

  getLeafId(): string | null {
    return this.entries.length > 0 ? (this.entries[this.entries.length - 1].id ?? null) : null;
  }

  getHeader() {
    return {
      id: this.sessionId,
      timestamp: this.headerTs ?? "2026-11-25T03:00:00.000Z",
      cwd: "/tmp/demo-project",
      version: 3,
    };
  }
}

function makeCtx(
  sm: FakeSessionManager,
  overrides: Partial<OmpContext> = {},
): OmpContext {
  return {
    cwd: "/tmp/demo-project",
    mode: "print",
    hasUI: false,
    agent: { kind: "main", id: "Main", name: "main", depth: 0 },
    model: { provider: "ollama", id: "gemma4:26b" },
    sessionManager: sm,
    ...overrides,
  };
}

class FakePi implements OmpExtensionAPI {
  readonly handlers = new Map<string, Array<(payload: OmpHookPayload, ctx: OmpContext) => unknown>>();
  readonly pi = { VERSION: "19.1.1" };

  on(event: string, handler: (payload: OmpHookPayload, ctx: OmpContext) => unknown): void {
    const list = this.handlers.get(event) ?? [];
    list.push(handler);
    this.handlers.set(event, list);
  }

  emit(event: string, payload: OmpHookPayload, ctx: OmpContext): unknown[] {
    const list = this.handlers.get(event) ?? [];
    return list.map((fn) => fn(payload, ctx));
  }
}

function fixedClock(startMs = Date.parse("2026-11-25T03:00:00.000Z")): () => string {
  let current = startMs;
  return () => {
    const iso = new Date(current).toISOString();
    current += 10;
    return iso;
  };
}

async function waitFor(predicate: () => boolean, timeoutMs = 2000): Promise<void> {
  const deadline = Date.now() + timeoutMs;
  while (!predicate()) {
    if (Date.now() >= deadline) {
      throw new Error(`condition not met within ${timeoutMs}ms`);
    }
    await new Promise((resolve) => setTimeout(resolve, 15));
  }
}

describe("OmpMapper — verified event mapping", () => {
  it("maps session start, prompt, tool call/result (normal, blocked, MCP), reply, compaction, outcome, and shutdown", () => {
    const mapper = new OmpMapper({ harnessVersion: "19.1.1", clock: fixedClock() });
    const sid = "01a210a3-b133-705e-b720-09908b69e115";
    const sm = new FakeSessionManager(sid, `/tmp/sessions/${sid}.jsonl`);
    const ctx = makeCtx(sm);

    const all: CaptureRecord[] = [];
    const emit = (ev: string, payload: OmpHookPayload) => {
      all.push(...mapper.handle(ev, payload, ctx, 1));
    };

    // 1. session_start
    emit("session_start", { type: "session_start" });

    // 2. User prompt (entry present at message_end)
    const userMsg = {
      role: "user",
      content: [{ type: "text", text: "Check note.txt and call echo" }],
      attribution: "user",
      timestamp: 1795575570874,
    };
    sm.push({ type: "message", id: "5f67c9db", message: userMsg });
    emit("message_end", { type: "message_end", message: userMsg });

    // 3. Tool call 1 (normal) + assistant message_end (entry NOT yet in getEntries!)
    emit("tool_call", {
      type: "tool_call",
      toolName: "read",
      toolCallId: "call_read_1",
      input: { path: "note.txt" },
    });
    const toolOnlyAssistant = {
      role: "assistant",
      content: [{ type: "toolCall", id: "call_read_1", name: "read", arguments: { path: "note.txt" } }],
      stopReason: "toolUse",
      model: "gemma4:26b",
      provider: "ollama",
      responseId: "resp_1",
      timestamp: 1795575570898,
    };
    emit("message_end", { type: "message_end", message: toolOnlyAssistant });

    // Assistant entry appears before tool result; toolResult arrives at message_end
    sm.push({ type: "message", id: "6fb5082f", message: toolOnlyAssistant });
    const readResult = {
      role: "toolResult",
      toolCallId: "call_read_1",
      toolName: "read",
      content: [{ type: "text", text: "hello" }],
      isError: false,
      timestamp: 1795575573108,
    };
    sm.push({ type: "message", id: "97886fde", message: readResult });
    emit("message_end", { type: "message_end", message: readResult });
    emit("turn_end", { type: "turn_end", turnIndex: 0, message: toolOnlyAssistant });

    // 4. Blocked tool call (no tool_result hook event, only tool_call + message_end toolResult)
    emit("tool_call", {
      type: "tool_call",
      toolName: "wait",
      toolCallId: "call_blocked_2",
      input: {},
    });
    const blockedResult = {
      role: "toolResult",
      toolCallId: "call_blocked_2",
      toolName: "wait",
      content: [{ type: "text", text: "guard: tool wait is not allowed" }],
      isError: true,
      timestamp: 1795575573400,
    };
    sm.push({ type: "message", id: "b5cc3364", message: blockedResult });
    emit("message_end", { type: "message_end", message: blockedResult });

    // 5. MCP tool call + result with serverName / mcpToolName
    emit("tool_call", {
      type: "tool_call",
      toolName: "mcp__rec_echo_back",
      toolCallId: "call_mcp_3",
      input: { text: "hello" },
    });
    const mcpResult = {
      role: "toolResult",
      toolCallId: "call_mcp_3",
      toolName: "mcp__rec_echo_back",
      content: [{ type: "text", text: "echo: hello" }],
      details: { serverName: "rec", mcpToolName: "echo_back" },
      isError: false,
      timestamp: 1795575573600,
    };
    sm.push({ type: "message", id: "bd326b50", message: mcpResult });
    emit("message_end", { type: "message_end", message: mcpResult });

    // 6. Final assistant reply (entry NOT in getEntries at message_end; present at turn_end)
    const finalReply = {
      role: "assistant",
      content: [
        { type: "thinking", thinking: "private chain of thought" },
        { type: "text", text: "done" },
      ],
      stopReason: "stop",
      model: "gemma4:26b",
      provider: "ollama",
      responseId: "resp_2",
      usage: { input: 29, output: 3, totalTokens: 32 },
      timestamp: 1795575573903,
      completedAt: 1795575574144,
    };
    emit("message_end", { type: "message_end", message: finalReply });
    sm.push({ type: "message", id: "21a15f54", message: finalReply });
    emit("turn_end", { type: "turn_end", turnIndex: 1, message: finalReply });

    // 7. session_stop closes the turn; agent_end after it must NOT emit a second outcome
    emit("session_stop", {
      type: "session_stop",
      session_id: sid,
      turn_id: 1,
      last_assistant_message: finalReply,
    });
    emit("agent_end", { type: "agent_end" });

    // 8. Compaction
    sm.push({ type: "compaction", id: "7512dd62", timestamp: "2026-11-25T03:03:10.654Z" });
    emit("session_compact", {
      type: "session_compact",
      fromExtension: false,
      compactionEntry: {
        id: "7512dd62",
        parentId: "21a15f54",
        timestamp: "2026-11-25T03:03:10.654Z",
        summary: "Earlier conversation summarized.",
        shortSummary: "Summary",
        firstKeptEntryId: "21a15f54",
        tokensBefore: 2289,
        tokensAfter: 2121,
        method: "soft",
      },
    });

    // 9. Shutdown
    sm.push({ type: "custom", id: "4a40a89a", customType: "session_exit" });
    emit("session_shutdown", { type: "session_shutdown" });

    assert.deepEqual(
      all.map((r) => [r.kind, r.source_key]),
      [
        ["session_start", sid],
        ["user_msg", "5f67c9db"],
        ["tool_call", "call_read_1"],
        ["tool_result", "call_read_1"],
        ["tool_call", "call_blocked_2"],
        ["tool_result", "call_blocked_2"],
        ["tool_call", "call_mcp_3"],
        ["tool_result", "call_mcp_3"],
        ["assistant_msg", "21a15f54"],
        ["outcome", "turn:5f67c9db"],
        ["compaction", "7512dd62"],
        ["session_end", "session_end:4a40a89a"],
      ],
    );

    // Every record targets oh-my-pi with v=1 and session metadata.
    for (const rec of all) {
      assert.equal(rec.v, RECORD_VERSION);
      assert.equal(rec.agent_id, AGENT_ID);
      assert.equal(rec.harness_session_id, sid);
      assert.equal(rec.session?.harness_version, "19.1.1");
      assert.equal(rec.session?.project, "demo-project");
      assert.equal(rec.session?.model, "ollama/gemma4:26b");
    }

    // Tool results link to tool calls and preserve error/MCP metadata.
    const blocked = all.find((r) => r.kind === "tool_result" && r.source_key === "call_blocked_2")!;
    assert.equal(blocked.parent_key, "call_blocked_2");
    assert.equal(blocked.parent_kind, "tool_call");
    assert.equal(blocked.payload?.is_error, true);
    assert.equal(blocked.content, "guard: tool wait is not allowed");

    const mcp = all.find((r) => r.kind === "tool_result" && r.source_key === "call_mcp_3")!;
    assert.equal(mcp.payload?.mcp_server, "rec");
    assert.equal(mcp.payload?.mcp_tool_name, "echo_back");

    // Final reply excludes thinking block and captures text + attribution.
    const reply = all.find((r) => r.kind === "assistant_msg")!;
    assert.equal(reply.content, "done");
    assert.deepEqual(reply.attribution, { message_id: "21a15f54", has_tool_calls: false });

    // Outcome closes the turn cleanly.
    const outcome = all.find((r) => r.kind === "outcome")!;
    assert.equal(outcome.payload?.closed_by, "session_stop");
    assert.equal(outcome.payload?.result_kind, "text");
    assert.equal(outcome.payload?.turn_id, 1);

    // Compaction keeps summary and token counts.
    const comp = all.find((r) => r.kind === "compaction")!;
    assert.equal(comp.content, "Earlier conversation summarized.");
    assert.equal(comp.payload?.tokens_before, 2289);
    assert.equal(comp.payload?.tokens_after, 2121);
  });

  it("closes turns on agent_end (--max-time) and session_shutdown (signal)", () => {
    // --max-time: aborted toolUse turn -> agent_end without session_stop
    const mapper1 = new OmpMapper({ clock: fixedClock() });
    const sm1 = new FakeSessionManager("sid-maxtime", "/tmp/maxtime.jsonl");
    const ctx1 = makeCtx(sm1);
    const out1: CaptureRecord[] = [];
    const u1 = { role: "user", content: "sleep 30", timestamp: 1795575687961 };
    sm1.push({ type: "message", id: "99ad21fd", message: u1 });
    out1.push(...mapper1.handle("message_end", { message: u1 }, ctx1));
    out1.push(...mapper1.handle("tool_call", { toolName: "bash", toolCallId: "c1", input: { command: "sleep 30" } }, ctx1));
    const a1 = { role: "assistant", content: [{ type: "toolCall", id: "c1", name: "bash" }], stopReason: "toolUse", timestamp: 1795575687982 };
    sm1.push({ type: "message", id: "ec94dec9", message: a1 });
    const r1 = { role: "toolResult", toolCallId: "c1", toolName: "bash", content: [{ type: "text", text: "Command aborted" }], isError: true, timestamp: 1795575691813 };
    sm1.push({ type: "message", id: "39862a85", message: r1 });
    out1.push(...mapper1.handle("message_end", { message: r1 }, ctx1));
    out1.push(...mapper1.handle("turn_end", { turnIndex: 0, message: a1 }, ctx1));
    out1.push(...mapper1.handle("agent_end", {}, ctx1));
    sm1.push({ type: "custom", id: "d3288b5c", customType: "session_exit" });
    out1.push(...mapper1.handle("session_shutdown", {}, ctx1));

    const outcomes1 = out1.filter((r) => r.kind === "outcome");
    assert.equal(outcomes1.length, 1);
    assert.equal(outcomes1[0].source_key, "turn:99ad21fd");
    assert.equal(outcomes1[0].payload?.closed_by, "agent_end");
    assert.equal(outcomes1[0].payload?.result_kind, "interrupted");

    // SIGINT: session_shutdown arrives BEFORE final aborted toolResult and turn_end
    const mapper2 = new OmpMapper({ clock: fixedClock() });
    const sm2 = new FakeSessionManager("sid-sigint", "/tmp/sigint.jsonl");
    const ctx2 = makeCtx(sm2);
    const out2: CaptureRecord[] = [];
    const u2 = { role: "user", content: "sleep 30", timestamp: 1795575678119 };
    sm2.push({ type: "message", id: "09bfb400", message: u2 });
    out2.push(...mapper2.handle("message_end", { message: u2 }, ctx2));
    out2.push(...mapper2.handle("tool_call", { toolName: "bash", toolCallId: "c2", input: { command: "sleep 30" } }, ctx2));
    // Signal arrives while tool runs: session_shutdown fires first
    out2.push(...mapper2.handle("session_shutdown", {}, ctx2));
    // 4ms later: aborted tool result and turn_end arrive
    const r2 = { role: "toolResult", toolCallId: "c2", toolName: "bash", content: [{ type: "text", text: "Command aborted" }], isError: true, timestamp: 1795575679054 };
    sm2.push({ type: "message", id: "ef988c56", message: r2 });
    out2.push(...mapper2.handle("message_end", { message: r2 }, ctx2));
    const abortedAssistant = { role: "assistant", content: [], stopReason: "aborted", errorMessage: "Request was aborted", timestamp: 1795575679057 };
    sm2.push({ type: "message", id: "658fb734", message: abortedAssistant });
    out2.push(...mapper2.handle("turn_end", { turnIndex: 1, message: abortedAssistant }, ctx2));

    assert.deepEqual(
      out2.map((r) => r.kind),
      ["user_msg", "tool_call", "outcome", "session_end", "tool_result"],
    );
    const sigOutcome = out2.find((r) => r.kind === "outcome")!;
    assert.equal(sigOutcome.source_key, "turn:09bfb400");
    assert.equal(sigOutcome.payload?.closed_by, "session_shutdown");
    assert.equal(sigOutcome.payload?.result_kind, "interrupted");
  });
});

describe("OmpMapper — session lifecycle transitions", () => {
  it("follows omp session ids across resume, new, fork, branch, and clear", () => {
    const mapper = new OmpMapper({ clock: fixedClock() });
    const sm = new FakeSessionManager("sid-1", "/tmp/s1.jsonl");
    const ctx = makeCtx(sm, { mode: "rpc" });

    // Initial fresh start
    const rStart = mapper.handle("session_start", {}, ctx, 1);
    assert.equal(rStart.length, 1);
    assert.equal(rStart[0].harness_session_id, "sid-1");
    assert.equal(rStart[0].source_key, "sid-1");
    assert.equal(rStart[0].payload?.source, "startup");

    // Prompt + reply in sid-1
    const u1 = { role: "user", content: "hi", timestamp: 1795575727141 };
    sm.push({ type: "message", id: "192d5005", message: u1 });
    mapper.handle("message_end", { message: u1 }, ctx, 1);

    // Fork at entry via session_branch
    mapper.handle("session_before_branch", { reason: "fork", entryId: "192d5005" }, ctx, 1);
    sm.sessionId = "sid-fork-branch";
    sm.sessionFile = "/tmp/s-fork-branch.jsonl";
    const rForkBranch = mapper.handle(
      "session_branch",
      { reason: "fork", previousSessionFile: "/tmp/s1.jsonl" },
      ctx,
      1,
    );
    assert.equal(rForkBranch[0].harness_session_id, "sid-fork-branch");
    assert.equal(rForkBranch[0].source_key, "sid-fork-branch");
    assert.equal(rForkBranch[0].payload?.source, "fork");
    assert.equal(rForkBranch[0].payload?.previous_session_id, "sid-1");
    assert.equal(rForkBranch[0].payload?.branch_entry_id, "192d5005");
    assert.equal(rForkBranch[0].session?.metadata?.parent_session_id, "sid-1");

    // Fork without entry via session_switch
    mapper.handle("session_before_switch", { reason: "fork" }, ctx, 1);
    sm.sessionId = "sid-fork-switch";
    sm.sessionFile = "/tmp/s-fork-switch.jsonl";
    const rForkSwitch = mapper.handle(
      "session_switch",
      { reason: "fork", previousSessionFile: "/tmp/s-fork-branch.jsonl" },
      ctx,
      1,
    );
    assert.equal(rForkSwitch[0].harness_session_id, "sid-fork-switch");
    assert.equal(rForkSwitch[0].payload?.source, "fork");
    assert.equal(rForkSwitch[0].payload?.previous_session_id, "sid-fork-branch");

    // New session via session_switch
    mapper.handle("session_before_switch", { reason: "new" }, ctx, 1);
    sm.sessionId = "sid-new";
    sm.sessionFile = "/tmp/s-new.jsonl";
    sm.entries = [{ type: "model_change", id: "0ba27469" }];
    const rNew = mapper.handle(
      "session_switch",
      { reason: "new", previousSessionFile: "/tmp/s-fork-switch.jsonl" },
      ctx,
      1,
    );
    assert.equal(rNew[0].harness_session_id, "sid-new");
    assert.equal(rNew[0].source_key, "sid-new");
    assert.equal(rNew[0].payload?.source, "new");

    // Resume sid-1 via session_switch
    mapper.handle("session_before_switch", { reason: "resume", targetSessionFile: "/tmp/s1.jsonl" }, ctx, 1);
    sm.sessionId = "sid-1";
    sm.sessionFile = "/tmp/s1.jsonl";
    sm.entries = [{ type: "message", id: "2e0f62ae", message: { role: "assistant", content: "ok" } }];
    const rResumeSwitch = mapper.handle(
      "session_switch",
      { reason: "resume", previousSessionFile: "/tmp/s-new.jsonl" },
      ctx,
      1,
    );
    assert.equal(rResumeSwitch[0].harness_session_id, "sid-1");
    assert.equal(rResumeSwitch[0].source_key, "session_switch:resume:2e0f62ae");
    assert.equal(rResumeSwitch[0].payload?.source, "resume");

    // Branch at prompt via session_branch
    mapper.handle("session_before_branch", { reason: "branch", entryId: "2e0f62ae" }, ctx, 1);
    sm.sessionId = "sid-branch";
    sm.sessionFile = "/tmp/s-branch.jsonl";
    const rBranch = mapper.handle(
      "session_branch",
      { reason: "branch", previousSessionFile: "/tmp/s1.jsonl" },
      ctx,
      1,
    );
    assert.equal(rBranch[0].harness_session_id, "sid-branch");
    assert.equal(rBranch[0].source_key, "sid-branch");
    assert.equal(rBranch[0].payload?.source, "branch");
    assert.equal(rBranch[0].payload?.branch_entry_id, "2e0f62ae");

    // Clear with new session id
    sm.sessionId = "sid-cleared";
    sm.sessionFile = "/tmp/s-cleared.jsonl";
    sm.entries = [{ type: "model_change", id: "aaaa1111" }];
    const rClearNew = mapper.handle("session_clear", {}, ctx, 1);
    assert.equal(rClearNew[0].harness_session_id, "sid-cleared");
    assert.equal(rClearNew[0].source_key, "sid-cleared");
    assert.equal(rClearNew[0].payload?.source, "clear");
    assert.equal(rClearNew[0].payload?.previous_session_id, "sid-branch");

    // Clear keeping the same session id
    sm.entries = [{ type: "model_change", id: "bbbb2222" }];
    const rClearSame = mapper.handle("session_clear", {}, ctx, 1);
    assert.equal(rClearSame[0].harness_session_id, "sid-cleared");
    assert.equal(rClearSame[0].source_key, "session_clear:bbbb2222");
  });

  it("detects resumed session_start (-c / -r) from existing transcript entries", () => {
    const mapper = new OmpMapper({ clock: fixedClock() });
    const sm = new FakeSessionManager("sid-existing", "/tmp/existing.jsonl", [
      { type: "model_change", id: "11111111" },
      { type: "message", id: "885b6630", message: { role: "user", content: "hello" } },
      { type: "custom", id: "4a40a89a", customType: "session_exit" },
    ]);
    const records = mapper.handle("session_start", {}, makeCtx(sm), 1);
    assert.equal(records.length, 1);
    assert.equal(records[0].harness_session_id, "sid-existing");
    assert.equal(records[0].source_key, "session_start:resume:4a40a89a");
    assert.equal(records[0].payload?.source, "resume");
  });
});

describe("OmpMapper — subagents and concurrent sessions", () => {
  it("distinguishes main, depth-1, and depth-2 subagent sessions and links callers", () => {
    const mapper = new OmpMapper({ clock: fixedClock() });
    const mainSm = new FakeSessionManager("sid-main", "/tmp/main.jsonl");
    const sub1Sm = new FakeSessionManager("sid-sub1", "/tmp/main/Worker.jsonl");
    const sub2Sm = new FakeSessionManager("sid-sub2", "/tmp/main/Worker/Worker.Leaf.jsonl");

    const mainCtx = makeCtx(mainSm, {
      agent: { kind: "main", id: "Main", name: "main", depth: 0 },
    });
    const sub1Agent: OmpAgentInfo = {
      kind: "sub",
      id: "Worker",
      name: "task",
      depth: 1,
      parentId: "Main",
    };
    const sub1Ctx = makeCtx(sub1Sm, { agent: sub1Agent });
    const sub2Agent: OmpAgentInfo = {
      kind: "sub",
      id: "Worker.Leaf",
      name: "task",
      depth: 2,
      parentId: "Worker",
    };
    const sub2Ctx = makeCtx(sub2Sm, { agent: sub2Agent });

    const out: CaptureRecord[] = [];
    out.push(...mapper.handle("session_start", {}, mainCtx, 1));

    const mainPrompt = { role: "user", content: "delegate", attribution: "user", timestamp: 1000 };
    mainSm.push({ type: "message", id: "91b05380", message: mainPrompt });
    out.push(...mapper.handle("message_end", { message: mainPrompt }, mainCtx, 1));

    // Main spawns Worker
    mapper.handle(
      "before_subagent_spawn",
      { agent: "task", invocationKind: "task", spawnKey: "Worker" },
      mainCtx,
      1,
    );
    out.push(...mapper.handle("session_start", {}, sub1Ctx, 2));

    // Main yields with willContinue + awaitingAsyncWork -> must NOT emit outcome yet
    const waitOutcome = mapper.handle(
      "agent_end",
      { willContinue: true, awaitingAsyncWork: true },
      mainCtx,
      1,
    );
    assert.equal(waitOutcome.length, 0);

    // Worker spawns Worker.Leaf
    const sub1Prompt = { role: "user", content: "do work", attribution: "agent", timestamp: 2000 };
    sub1Sm.push({ type: "message", id: "fa590a4b", message: sub1Prompt });
    out.push(...mapper.handle("message_end", { message: sub1Prompt }, sub1Ctx, 2));
    mapper.handle(
      "before_subagent_spawn",
      { agent: "task", invocationKind: "task", spawnKey: "Worker.Leaf" },
      sub1Ctx,
      2,
    );
    out.push(...mapper.handle("session_start", {}, sub2Ctx, 3));

    // Worker.Leaf answers and finishes via agent_end (no session_stop in subagents)
    const sub2Prompt = { role: "user", content: "say sub", attribution: "agent", timestamp: 3000 };
    sub2Sm.push({ type: "message", id: "a44aef06", message: sub2Prompt });
    out.push(...mapper.handle("message_end", { message: sub2Prompt }, sub2Ctx, 3));
    const sub2Reply = { role: "assistant", content: [{ type: "text", text: "sub" }], stopReason: "stop", timestamp: 3500 };
    sub2Sm.push({ type: "message", id: "b5c1fc25", message: sub2Reply });
    out.push(...mapper.handle("turn_end", { turnIndex: 0, message: sub2Reply }, sub2Ctx, 3));
    out.push(...mapper.handle("agent_end", {}, sub2Ctx, 3));

    // Worker finishes via agent_end
    const sub1Reply = { role: "assistant", content: [{ type: "text", text: "worker done" }], stopReason: "stop", timestamp: 4000 };
    sub1Sm.push({ type: "message", id: "d12479b8", message: sub1Reply });
    out.push(...mapper.handle("turn_end", { turnIndex: 1, message: sub1Reply }, sub1Ctx, 2));
    out.push(...mapper.handle("agent_end", {}, sub1Ctx, 2));

    // Main gets custom async-result (NOT a user_msg!) then gives final answer and session_stop
    const asyncNotice = { role: "custom", customType: "async-result", content: "Background job Worker completed", timestamp: 4500 };
    mainSm.push({ type: "custom_message", id: "bc69e2ff", message: asyncNotice });
    assert.deepEqual(mapper.handle("message_end", { message: asyncNotice }, mainCtx, 1), []);

    const mainReply = { role: "assistant", content: [{ type: "text", text: "all done" }], stopReason: "stop", timestamp: 5000 };
    mainSm.push({ type: "message", id: "78fb933b", message: mainReply });
    out.push(...mapper.handle("turn_end", { turnIndex: 0, message: mainReply }, mainCtx, 1));
    out.push(...mapper.handle("session_stop", { session_id: "sid-main", turn_id: 0, last_assistant_message: mainReply }, mainCtx, 1));
    out.push(...mapper.handle("agent_end", {}, mainCtx, 1));

    const mainStart = out.find((r) => r.kind === "session_start" && r.harness_session_id === "sid-main")!;
    const sub1Start = out.find((r) => r.kind === "session_start" && r.harness_session_id === "sid-sub1")!;
    const sub2Start = out.find((r) => r.kind === "session_start" && r.harness_session_id === "sid-sub2")!;

    assert.equal(mainStart.session?.metadata?.agent_kind, "main");
    assert.equal(mainStart.session?.metadata?.calling_session_id, undefined);
    assert.equal(mainStart.payload?.subagent, undefined);

    assert.equal(sub1Start.session?.metadata?.agent_kind, "sub");
    assert.equal(sub1Start.session?.metadata?.agent_depth, 1);
    assert.equal(sub1Start.session?.metadata?.calling_session_id, "sid-main");
    assert.equal(sub1Start.payload?.calling_session_id, "sid-main");
    assert.deepEqual(sub1Start.payload?.subagent, {
      id: "Worker",
      name: "task",
      depth: 1,
      parent_id: "Main",
      calling_session_id: "sid-main",
    });

    assert.equal(sub2Start.session?.metadata?.agent_kind, "sub");
    assert.equal(sub2Start.session?.metadata?.agent_depth, 2);
    assert.equal(sub2Start.session?.metadata?.calling_session_id, "sid-sub1");
    assert.equal(sub2Start.payload?.calling_session_id, "sid-sub1");

    // Each session got exactly one outcome.
    const outcomes = out.filter((r) => r.kind === "outcome");
    assert.deepEqual(
      outcomes.map((r) => [r.harness_session_id, r.source_key, r.payload?.closed_by]),
      [
        ["sid-sub2", "turn:a44aef06", "agent_end"],
        ["sid-sub1", "turn:fa590a4b", "agent_end"],
        ["sid-main", "turn:91b05380", "session_stop"],
      ],
    );
  });
});

describe("OmpMapper — deterministic replay identity", () => {
  it("produces identical identity keys when a session is replayed a second time", () => {
    const runSession = (mapper: OmpMapper): CaptureRecord[] => {
      const sm = new FakeSessionManager("sid-replay", "/tmp/replay.jsonl");
      const ctx = makeCtx(sm);
      const out: CaptureRecord[] = [];
      out.push(...mapper.handle("session_start", {}, ctx, 1));
      const u = { role: "user", content: "ping", timestamp: 1795575492990 };
      sm.push({ type: "message", id: "885b6630", message: u });
      out.push(...mapper.handle("message_end", { message: u }, ctx, 1));
      out.push(...mapper.handle("tool_call", { toolName: "read", toolCallId: "call_1", input: { path: "a" } }, ctx, 1));
      const tr = { role: "toolResult", toolCallId: "call_1", toolName: "read", content: "x", timestamp: 1795575493000 };
      sm.push({ type: "message", id: "97886fde", message: tr });
      out.push(...mapper.handle("message_end", { message: tr }, ctx, 1));
      const a = { role: "assistant", content: [{ type: "text", text: "pong" }], stopReason: "stop", responseId: "resp_1", timestamp: 1795575493018 };
      sm.push({ type: "message", id: "770ba432", message: a });
      out.push(...mapper.handle("turn_end", { turnIndex: 0, message: a }, ctx, 1));
      out.push(...mapper.handle("session_stop", { session_id: "sid-replay", turn_id: 0, last_assistant_message: a }, ctx, 1));
      sm.push({ type: "compaction", id: "7512dd62", timestamp: "2026-11-25T03:03:10.654Z" });
      out.push(...mapper.handle("session_compact", { compactionEntry: { id: "7512dd62", timestamp: "2026-11-25T03:03:10.654Z", summary: "sum" } }, ctx, 1));
      sm.push({ type: "custom", id: "dc90ca8b", customType: "session_exit" });
      out.push(...mapper.handle("session_shutdown", {}, ctx, 1));
      return out;
    };

    const first = runSession(new OmpMapper({ clock: fixedClock(1_000_000) }));
    const second = runSession(new OmpMapper({ clock: fixedClock(9_000_000) }));
    const keysOf = (list: CaptureRecord[]) =>
      list.map((r) => [r.agent_id, r.harness_session_id, r.kind, r.source_key, r.parent_key ?? null]);
    assert.deepEqual(keysOf(first), keysOf(second));
  });
});

describe("RecordSink & extension runtime — non-blocking and single-report failures", () => {
  it("builds mem records arguments cleanly", () => {
    assert.deepEqual(buildMemArgs({ mem: "/bin/mem" }), ["records"]);
    assert.deepEqual(
      buildMemArgs({ mem: "/bin/mem", home: "/tmp/h", port: 9494, spoolOnly: true }),
      ["--home", "/tmp/h", "--port", "9494", "records", "--spool-only"],
    );
  });

  it("reports missing mem binary strictly once and handlers return immediately", async () => {
    const errors: string[] = [];
    const runtime = createRuntime({
      config: { mem: "/nonexistent/path/to/mem-missing-binary" },
      report: (msg) => errors.push(msg),
      clock: fixedClock(),
    });
    const pi = new FakePi();
    runtime.attach(pi);
    const sm = new FakeSessionManager("sid-missing", "/tmp/m.jsonl");
    const ctx = makeCtx(sm);

    const start = Date.now();
    const res1 = pi.emit("session_start", {}, ctx);
    assert.deepEqual(res1, [undefined]);
    const u = { role: "user", content: "hello", timestamp: 1795575492990 };
    sm.push({ type: "message", id: "885b6630", message: u });
    pi.emit("message_end", { message: u }, ctx);
    pi.emit("tool_call", { toolName: "read", toolCallId: "c1", input: { path: "x" } }, ctx);
    assert.ok(Date.now() - start < 200, "handlers must return synchronously");

    await waitFor(() => errors.length === 1);
    // Additional events after failure never report a second time.
    pi.emit("tool_call", { toolName: "read", toolCallId: "c2", input: { path: "y" } }, ctx);
    pi.emit("session_shutdown", {}, ctx);
    await new Promise((resolve) => setTimeout(resolve, 60));
    assert.equal(errors.length, 1);
    assert.match(errors[0], /agent-memory:/);
    runtime.closeNow();
  });

  it("reports dead mem process strictly once and keeps omp working", async () => {
    const dir = mkdtempSync(join(tmpdir(), "omp-ext-dead-"));
    try {
      const deadScript = join(dir, "mem");
      writeFileSync(deadScript, "#!/bin/sh\necho 'boom on stderr' >&2\nexit 1\n");
      chmodSync(deadScript, 0o755);

      const errors: string[] = [];
      const runtime = createRuntime({
        config: { mem: deadScript },
        report: (msg) => errors.push(msg),
        clock: fixedClock(),
      });
      const pi = new FakePi();
      runtime.attach(pi);
      const sm = new FakeSessionManager("sid-dead", "/tmp/d.jsonl");
      const ctx = makeCtx(sm);

      pi.emit("session_start", {}, ctx);
      await waitFor(() => errors.length === 1);
      // Subsequent events must not throw or report again.
      pi.emit("tool_call", { toolName: "bash", toolCallId: "c1", input: { command: "ls" } }, ctx);
      pi.emit("session_shutdown", {}, ctx);
      await new Promise((resolve) => setTimeout(resolve, 60));
      assert.equal(errors.length, 1);
      assert.match(errors[0], /exit code 1|pipe failed/);
      runtime.closeNow();
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });

  it("detects a slow/stalled mem that never reads stdin and reports strictly once", async () => {
    const dir = mkdtempSync(join(tmpdir(), "omp-ext-slow-"));
    try {
      const slowScript = join(dir, "mem");
      // Sleeps without reading stdin.
      writeFileSync(slowScript, "#!/bin/sh\nsleep 2\n");
      chmodSync(slowScript, 0o755);

      const errors: string[] = [];
      const sink = new RecordSink(
        {
          mem: slowScript,
          stallTimeoutMs: 100,
          maxBufferedBytes: 512 * 1024,
        },
        (msg) => errors.push(msg),
      );

      // Write enough to exceed both Node's and Bun's pipe buffer (~448 KiB on macOS).
      const bigContent = "x".repeat(64 * 1024);
      const t0 = Date.now();
      for (let i = 0; i < 20; i += 1) {
        sink.write({
          v: 1,
          agent_id: AGENT_ID,
          harness_session_id: "sid-slow",
          kind: "note",
          ts: "2026-11-25T03:00:00.000Z",
          source_key: `k-${i}`,
          content: bigContent,
        });
      }
      assert.ok(Date.now() - t0 < 200, "writes must not block when mem is slow");

      await waitFor(() => errors.length === 1, 2000);
      // Further writes after stall are silent no-ops.
      sink.write({
        v: 1,
        agent_id: AGENT_ID,
        harness_session_id: "sid-slow",
        kind: "note",
        ts: "2026-11-25T03:00:01.000Z",
        source_key: "after-stall",
        content: "ignored",
      });
      await new Promise((resolve) => setTimeout(resolve, 120));
      assert.equal(errors.length, 1);
      assert.equal(sink.isFailed, true);
      sink.closeNow();
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });

  it("reports malformed config.json once and disables capture cleanly", () => {
    const dir = mkdtempSync(join(tmpdir(), "omp-ext-cfg-"));
    try {
      const badCfg = join(dir, "config.json");
      writeFileSync(badCfg, "{not json");
      const loaded = loadSinkConfig(badCfg);
      assert.ok(loaded.error);

      const errors: string[] = [];
      const runtime = createRuntime({
        configPath: badCfg,
        report: (msg) => errors.push(msg),
      });
      const pi = new FakePi();
      runtime.attach(pi);
      assert.equal(errors.length, 1);
      pi.emit("session_start", {}, makeCtx(new FakeSessionManager("sid-1", "/tmp/a.jsonl")));
      assert.equal(errors.length, 1);
      runtime.closeNow();
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });

  it("delivers all records including post-shutdown signal tail and waits for all sessions", async () => {
    const dir = mkdtempSync(join(tmpdir(), "omp-ext-pipe-"));
    try {
      const outPath = join(dir, "captured.jsonl");
      const donePath = join(dir, "done");
      const fakeMem = join(dir, "mem");
      writeFileSync(
        fakeMem,
        `#!/bin/sh\ncat > "${outPath}"\necho done > "${donePath}"\n`,
      );
      chmodSync(fakeMem, 0o755);

      const errors: string[] = [];
      const runtime = createRuntime(
        {
          config: { mem: fakeMem, closeDelayMs: 40 },
          report: (msg) => errors.push(msg),
          clock: fixedClock(),
        },
        "19.1.1",
      );

      const mainPi = new FakePi();
      const subPi = new FakePi();
      runtime.attach(mainPi);
      runtime.attach(subPi);

      const mainSm = new FakeSessionManager("sid-main", "/tmp/m.jsonl");
      const subSm = new FakeSessionManager("sid-sub", "/tmp/m/Sub.jsonl");
      const mainCtx = makeCtx(mainSm);
      const subCtx = makeCtx(subSm, {
        agent: { kind: "sub", id: "Sub", name: "task", depth: 1, parentId: "Main" },
      });

      mainPi.emit("session_start", {}, mainCtx);
      subPi.emit("session_start", {}, subCtx);

      // Main shuts down first (as seen in r16-subagent); stdin must stay open for subagent!
      mainPi.emit("session_shutdown", {}, mainCtx);
      subPi.emit("tool_call", { toolName: "bash", toolCallId: "c1", input: { command: "sleep 30" } }, subCtx);
      subPi.emit("session_shutdown", {}, subCtx);

      // Signal tail: aborted tool result arrives 10ms after final session_shutdown!
      await new Promise((resolve) => setTimeout(resolve, 10));
      const abortedResult = {
        role: "toolResult",
        toolCallId: "c1",
        toolName: "bash",
        content: [{ type: "text", text: "Command aborted" }],
        isError: true,
        timestamp: 1795575679054,
      };
      subSm.push({ type: "message", id: "ef988c56", message: abortedResult });
      subPi.emit("message_end", { message: abortedResult }, subCtx);

      await waitFor(() => {
        try {
          return readFileSync(donePath, "utf8").trim() === "done";
        } catch {
          return false;
        }
      }, 2000);

      assert.deepEqual(errors, []);
      const lines = readFileSync(outPath, "utf8")
        .trim()
        .split("\n")
        .map((line) => JSON.parse(line) as CaptureRecord);
      assert.deepEqual(
        lines.map((r) => [r.harness_session_id, r.kind]),
        [
          ["sid-main", "session_start"],
          ["sid-sub", "session_start"],
          ["sid-main", "session_end"],
          ["sid-sub", "tool_call"],
          ["sid-sub", "session_end"],
          ["sid-sub", "tool_result"],
        ],
      );
      runtime.closeNow();
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });

  it("registers exactly the expected hook events and formats helpers deterministically", () => {
    assert.ok(HOOK_EVENTS.includes("session_start"));
    assert.ok(HOOK_EVENTS.includes("tool_call"));
    assert.ok(HOOK_EVENTS.includes("turn_end"));
    assert.ok(HOOK_EVENTS.includes("session_compact"));
    assert.ok(HOOK_EVENTS.includes("session_stop"));
    assert.ok(HOOK_EVENTS.includes("session_shutdown"));
    assert.equal(renderToolCall("read", { b: 2, a: 1 }), 'read {"a":1,"b":2}');
    assert.deepEqual(
      extractContentText([{ type: "text", text: "a" }, { type: "image", mimeType: "image/png" }]),
      { text: "a", hasToolCalls: false, nonTextTypes: ["image"] },
    );
  });
});
