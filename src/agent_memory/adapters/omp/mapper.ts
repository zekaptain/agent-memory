/**
 * Pure mapping from omp 19.1.1 extension hook events to agent-memory capture records (v1).
 *
 * Kept free of I/O so it can run inside omp's embedded Bun runtime with zero external
 * dependencies and be tested deterministically against recorded payload shapes.
 */

export const AGENT_ID = "oh-my-pi";
export const RECORD_VERSION = 1;

export type EventKind =
  | "session_start"
  | "session_end"
  | "user_msg"
  | "assistant_msg"
  | "tool_call"
  | "tool_result"
  | "compaction"
  | "outcome"
  | "note";

export interface SessionInfo {
  cwd?: string;
  model?: string;
  harness_version?: string;
  project?: string;
  started_at?: string;
  scope?: string;
  metadata?: Record<string, unknown>;
}

export interface CaptureRecord {
  v: 1;
  agent_id: typeof AGENT_ID;
  harness_session_id: string;
  kind: EventKind;
  ts: string;
  source_key: string;
  role?: string;
  content?: string;
  payload?: Record<string, unknown>;
  parent_key?: string;
  parent_kind?: string;
  seq?: number;
  session?: SessionInfo;
  attribution?: Record<string, unknown>;
}

export interface OmpAgentInfo {
  kind?: string;
  id?: string;
  name?: string;
  depth?: number;
  parentId?: string;
}

export interface OmpModelInfo {
  provider?: string;
  id?: string;
}

export interface OmpSessionEntry {
  type?: string;
  id?: string;
  parentId?: string | null;
  timestamp?: string | number;
  customType?: string;
  message?: OmpMessage;
}

export interface OmpSessionHeader {
  id?: string;
  timestamp?: string | number;
  cwd?: string;
  parentSession?: string;
  version?: number;
}

export interface OmpSessionManager {
  getSessionId?: () => string | undefined;
  getSessionFile?: () => string | undefined;
  getEntries?: () => OmpSessionEntry[];
  getLeafId?: () => string | null | undefined;
  getHeader?: () => OmpSessionHeader | undefined;
}

export interface OmpContext {
  cwd?: string;
  mode?: string;
  hasUI?: boolean;
  ui?: {
    notify?: (message: string, level?: string) => void;
  };
  agent?: OmpAgentInfo;
  model?: OmpModelInfo;
  sessionManager?: OmpSessionManager;
}

export interface OmpContentPart {
  type?: string;
  text?: string;
  thinking?: string;
  id?: string;
  name?: string;
  arguments?: unknown;
  mimeType?: string;
}

export interface OmpMessage {
  role?: string;
  content?: string | OmpContentPart[];
  timestamp?: string | number;
  completedAt?: string | number;
  attribution?: string;
  stopReason?: string;
  errorMessage?: string;
  model?: string;
  provider?: string;
  responseId?: string;
  usage?: Record<string, unknown>;
  toolCallId?: string;
  toolName?: string;
  isError?: boolean;
  details?: Record<string, unknown>;
  customType?: string;
}

export interface OmpCompactionEntry {
  id?: string;
  parentId?: string | null;
  timestamp?: string | number;
  summary?: string;
  shortSummary?: string;
  firstKeptEntryId?: string;
  tokensBefore?: number;
  tokensAfter?: number;
  method?: string;
  fromExtension?: boolean;
}

export interface OmpHookPayload {
  type?: string;
  reason?: string;
  previousSessionFile?: string;
  targetSessionFile?: string;
  entryId?: string;
  message?: OmpMessage;
  toolName?: string;
  toolCallId?: string;
  input?: unknown;
  turnIndex?: number;
  turn_id?: number;
  session_id?: string;
  sessionId?: string;
  last_assistant_message?: OmpMessage;
  willContinue?: boolean;
  awaitingAsyncWork?: boolean;
  fromExtension?: boolean;
  compactionEntry?: OmpCompactionEntry;
  spawnKey?: string;
  agent?: string;
  invocationKind?: string;
}

interface PendingTransition {
  fromSessionId?: string;
  fromLeafId?: string;
  reason?: string;
  entryId?: string;
}

interface SpawnOrigin {
  callingSessionId: string;
  invocationKind?: string;
  agent?: string;
}

interface SessionState {
  turnActive: boolean;
  turnOutcomeEmitted: boolean;
  turnPromptEntryId?: string;
  turnLastIndex?: number;
  turnLastStopReason?: string;
  turnLastErrorMessage?: string;
  turnLastEntryId?: string;
  turnLastTs?: string;
  usedEntryIds: Set<string>;
  callingSessionId?: string;
  invocationKind?: string;
  parentSessionFile?: string;
  parentSessionId?: string;
  branchEntryId?: string;
}

export interface MapperOptions {
  harnessVersion?: string;
  clock?: () => string;
}

const CONVERSATION_ENTRY_TYPES = new Set(["message", "compaction", "custom_message"]);

export function toIsoTimestamp(value: unknown, fallback: () => string): string {
  if (typeof value === "number" && Number.isFinite(value)) {
    const ms = value > 1e11 ? value : value * 1000;
    const date = new Date(ms);
    if (!Number.isNaN(date.getTime())) {
      return date.toISOString();
    }
  }
  if (typeof value === "string" && value.trim()) {
    const date = new Date(value.trim());
    if (!Number.isNaN(date.getTime())) {
      return date.toISOString();
    }
  }
  return fallback();
}

export function projectOf(cwd: string | undefined): string | undefined {
  if (!cwd) {
    return undefined;
  }
  const trimmed = cwd.replace(/[\\/]+$/, "");
  if (!trimmed) {
    return undefined;
  }
  const parts = trimmed.split(/[\\/]/);
  return parts[parts.length - 1] || undefined;
}

function stableJsonValue(value: unknown): unknown {
  if (Array.isArray(value)) {
    return value.map(stableJsonValue);
  }
  if (value !== null && typeof value === "object") {
    const out: Record<string, unknown> = {};
    for (const key of Object.keys(value as Record<string, unknown>).sort()) {
      const child = (value as Record<string, unknown>)[key];
      if (child !== undefined) {
        out[key] = stableJsonValue(child);
      }
    }
    return out;
  }
  return value;
}

export function compactJson(value: unknown): string {
  try {
    return JSON.stringify(stableJsonValue(value));
  } catch {
    return String(value);
  }
}

export function renderToolCall(name: string | undefined, input: unknown): string {
  const label = name || "tool";
  return input !== undefined && input !== null ? `${label} ${compactJson(input)}` : label;
}

export function extractContentText(content: string | OmpContentPart[] | undefined): {
  text: string;
  hasToolCalls: boolean;
  nonTextTypes: string[];
} {
  if (typeof content === "string") {
    return { text: content, hasToolCalls: false, nonTextTypes: [] };
  }
  if (!Array.isArray(content)) {
    return { text: "", hasToolCalls: false, nonTextTypes: [] };
  }
  const texts: string[] = [];
  const nonTextTypes: string[] = [];
  let hasToolCalls = false;
  for (const part of content) {
    if (typeof part === "string") {
      if (part) {
        texts.push(part);
      }
      continue;
    }
    if (!part || typeof part !== "object") {
      continue;
    }
    if (part.type === "text" && typeof part.text === "string") {
      if (part.text) {
        texts.push(part.text);
      }
    } else if (part.type === "toolCall") {
      hasToolCalls = true;
    } else if (part.type === "thinking") {
      // Thinking blocks are omitted by default, matching Droid capture.
    } else if (typeof part.type === "string") {
      nonTextTypes.push(part.type);
    }
  }
  return { text: texts.join("\n"), hasToolCalls, nonTextTypes };
}

function formatModel(model: OmpModelInfo | undefined, msg?: OmpMessage): string | undefined {
  const id = msg?.model || model?.id;
  if (!id) {
    return undefined;
  }
  const provider = msg?.provider || model?.provider;
  if (provider && !id.includes("/")) {
    return `${provider}/${id}`;
  }
  return id;
}

function isSubagent(agent: OmpAgentInfo | undefined): boolean {
  if (!agent) {
    return false;
  }
  return agent.kind === "sub" || (typeof agent.depth === "number" && agent.depth > 0);
}

function stopReasonToResultKind(
  stopReason: string | undefined,
  closedBy: "session_stop" | "agent_end" | "session_shutdown",
): string {
  if (closedBy === "session_shutdown") {
    return stopReason === "aborted" ? "aborted" : "interrupted";
  }
  if (stopReason === "stop") {
    return "text";
  }
  if (stopReason === "aborted") {
    return "aborted";
  }
  if (stopReason === "error") {
    return "error";
  }
  if (stopReason === "length") {
    return "length";
  }
  if (stopReason === "toolUse" && closedBy === "agent_end") {
    // `--max-time` stops after a toolUse turn with no follow-up assistant reply.
    return "interrupted";
  }
  return stopReason || "completed";
}

function pickUsage(usage: Record<string, unknown> | undefined): Record<string, unknown> | undefined {
  if (!usage || typeof usage !== "object") {
    return undefined;
  }
  const picked: Record<string, unknown> = {};
  for (const key of ["input", "output", "cacheRead", "cacheWrite", "totalTokens"]) {
    if (typeof usage[key] === "number") {
      picked[key] = usage[key];
    }
  }
  return Object.keys(picked).length > 0 ? picked : undefined;
}

function compactObject(obj: Record<string, unknown>): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(obj)) {
    if (value !== undefined && value !== null) {
      out[key] = value;
    }
  }
  return out;
}

export class OmpMapper {
  private readonly harnessVersion?: string;
  private readonly clock: () => string;
  private readonly sessions = new Map<string, SessionState>();
  private readonly pendingSpawns = new Map<string, SpawnOrigin>();
  private readonly agentToSession = new Map<string, string>();
  private readonly instanceSession = new Map<number, string>();
  private readonly pendingTransition = new Map<number, PendingTransition>();

  constructor(options: MapperOptions = {}) {
    this.harnessVersion = options.harnessVersion;
    this.clock = options.clock ?? (() => new Date().toISOString());
  }

  /**
   * Handle one omp hook event for an extension instance and return zero or more
   * capture records ready to serialize as JSON lines for `mem records`.
   */
  handle(
    eventName: string,
    payload: OmpHookPayload | undefined,
    ctx: OmpContext | undefined,
    instanceId = 0,
  ): CaptureRecord[] {
    const p = payload ?? {};
    const c = ctx ?? {};
    const hsid = this.sessionIdOf(p, c);
    if (!hsid) {
      return [];
    }

    const entries = this.entriesOf(c);
    const leafId = this.leafIdOf(c);
    const state = this.stateOf(hsid);

    if (c.agent?.id) {
      this.agentToSession.set(c.agent.id, hsid);
    }
    this.resolveSubagentCaller(c.agent, state);

    switch (eventName) {
      case "session_start": {
        this.instanceSession.set(instanceId, hsid);
        const resumed = this.hasConversationHistory(entries);
        const source = resumed ? "resume" : "startup";
        const sourceKey = resumed
          ? `session_start:resume:${leafId || hsid}`
          : hsid;
        const ts = this.sessionStartTs(c, entries, resumed);
        const sessionInfo = this.sessionInfo(c, state, undefined, resumed ? undefined : ts);
        return [
          this.record(hsid, "session_start", sourceKey, ts, c, state, {
            role: "harness",
            session: sessionInfo,
            payload: this.withSubagent(
              c,
              state,
              compactObject({
                hook: "session_start",
                source,
                leaf_id: resumed ? leafId : undefined,
                calling_session_id: state.callingSessionId,
              }),
            ),
          }),
        ];
      }

      case "session_before_switch":
      case "session_before_branch": {
        this.pendingTransition.set(instanceId, {
          fromSessionId: hsid,
          fromLeafId: leafId,
          reason: p.reason,
          entryId: p.entryId,
        });
        return [];
      }

      case "session_switch":
      case "session_branch": {
        const prev = this.pendingTransition.get(instanceId);
        this.pendingTransition.delete(instanceId);
        const previousSessionId = prev?.fromSessionId ?? this.instanceSession.get(instanceId);
        this.instanceSession.set(instanceId, hsid);

        const reason =
          p.reason || prev?.reason || (eventName === "session_branch" ? "branch" : "new");
        if (reason === "fork" || reason === "branch") {
          if (p.previousSessionFile) {
            state.parentSessionFile = p.previousSessionFile;
          }
          if (previousSessionId && previousSessionId !== hsid) {
            state.parentSessionId = previousSessionId;
          }
          if (prev?.entryId) {
            state.branchEntryId = prev.entryId;
          }
        }

        const isResume = reason === "resume";
        const sourceKey = isResume
          ? `${eventName}:resume:${leafId || hsid}`
          : hsid;
        const ts = this.clock();
        const sessionInfo = this.sessionInfo(c, state, undefined, isResume ? undefined : ts);
        return [
          this.record(hsid, "session_start", sourceKey, ts, c, state, {
            role: "harness",
            session: sessionInfo,
            payload: this.withSubagent(
              c,
              state,
              compactObject({
                hook: eventName,
                source: reason,
                previous_session_file: p.previousSessionFile,
                previous_session_id:
                  previousSessionId && previousSessionId !== hsid ? previousSessionId : undefined,
                branch_entry_id: prev?.entryId,
                leaf_id: isResume ? leafId : undefined,
                calling_session_id: state.callingSessionId,
              }),
            ),
          }),
        ];
      }

      case "session_clear": {
        const previousSessionId = this.instanceSession.get(instanceId);
        const sameSession = previousSessionId === hsid;
        this.instanceSession.set(instanceId, hsid);
        const sourceKey = sameSession ? `session_clear:${leafId || hsid}` : hsid;
        const ts = this.clock();
        const sessionInfo = this.sessionInfo(c, state, undefined, sameSession ? undefined : ts);
        return [
          this.record(hsid, "session_start", sourceKey, ts, c, state, {
            role: "harness",
            session: sessionInfo,
            payload: this.withSubagent(
              c,
              state,
              compactObject({
                hook: "session_clear",
                source: "clear",
                previous_session_id: !sameSession ? previousSessionId : undefined,
                leaf_id: sameSession ? leafId : undefined,
                calling_session_id: state.callingSessionId,
              }),
            ),
          }),
        ];
      }

      case "before_subagent_spawn": {
        if (p.spawnKey) {
          this.pendingSpawns.set(p.spawnKey, {
            callingSessionId: hsid,
            invocationKind: p.invocationKind,
            agent: p.agent,
          });
        }
        return [];
      }

      case "message_end": {
        const msg = p.message;
        if (!msg || !msg.role) {
          return [];
        }
        if (msg.role === "user") {
          return this.mapUserMessage(hsid, msg, entries, leafId, c, state);
        }
        if (msg.role === "toolResult") {
          return this.mapToolResultMessage(hsid, msg, entries, leafId, c, state);
        }
        // Assistant messages are mapped at `turn_end`, when their transcript entry id exists.
        // `custom` messages (such as `async-result`) are not user prompts.
        return [];
      }

      case "tool_call": {
        if (!p.toolCallId) {
          return [];
        }
        const ts = this.clock();
        return [
          this.record(hsid, "tool_call", p.toolCallId, ts, c, state, {
            role: "assistant",
            content: renderToolCall(p.toolName, p.input),
            payload: this.withSubagent(
              c,
              state,
              compactObject({
                tool_name: p.toolName,
                tool_input: p.input,
              }),
            ),
          }),
        ];
      }

      case "turn_end": {
        return this.mapTurnEnd(hsid, p, entries, leafId, c, state);
      }

      case "session_compact": {
        const ce = p.compactionEntry;
        if (!ce) {
          return [];
        }
        const ts = toIsoTimestamp(ce.timestamp, this.clock);
        const sourceKey = ce.id || leafId || `compaction:${ts}`;
        const summary = ce.summary || ce.shortSummary;
        return [
          this.record(hsid, "compaction", sourceKey, ts, c, state, {
            role: "harness",
            content: summary || undefined,
            payload: this.withSubagent(
              c,
              state,
              compactObject({
                from_extension: Boolean(p.fromExtension ?? ce.fromExtension),
                method: ce.method,
                tokens_before: ce.tokensBefore,
                tokens_after: ce.tokensAfter,
                first_kept_entry_id: ce.firstKeptEntryId,
                parent_entry_id: ce.parentId ?? undefined,
                short_summary:
                  ce.shortSummary && ce.shortSummary !== ce.summary ? ce.shortSummary : undefined,
              }),
            ),
          }),
        ];
      }

      case "session_stop": {
        return this.closeTurn(hsid, "session_stop", p, leafId, c, state);
      }

      case "agent_end": {
        if (p.willContinue || p.awaitingAsyncWork) {
          return [];
        }
        return this.closeTurn(hsid, "agent_end", p, leafId, c, state);
      }

      case "session_shutdown": {
        const out: CaptureRecord[] = [];
        out.push(...this.closeTurn(hsid, "session_shutdown", p, leafId, c, state));
        const ts = this.clock();
        const sourceKey = `session_end:${leafId || hsid}`;
        out.push(
          this.record(hsid, "session_end", sourceKey, ts, c, state, {
            role: "harness",
            payload: this.withSubagent(
              c,
              state,
              compactObject({
                hook: "session_shutdown",
                leaf_id: leafId,
              }),
            ),
          }),
        );
        return out;
      }

      default:
        return [];
    }
  }

  private mapUserMessage(
    hsid: string,
    msg: OmpMessage,
    entries: OmpSessionEntry[],
    leafId: string | undefined,
    ctx: OmpContext,
    state: SessionState,
  ): CaptureRecord[] {
    const ts = toIsoTimestamp(msg.timestamp, this.clock);
    const entryId =
      this.findMessageEntryId(entries, leafId, "user", msg, state.usedEntryIds) ??
      `user:${msg.timestamp ?? ts}`;
    state.usedEntryIds.add(entryId);

    // Start tracking the turn initiated by this user prompt.
    state.turnActive = true;
    state.turnOutcomeEmitted = false;
    state.turnPromptEntryId = entryId;
    state.turnLastIndex = undefined;
    state.turnLastStopReason = undefined;
    state.turnLastErrorMessage = undefined;
    state.turnLastEntryId = undefined;
    state.turnLastTs = undefined;

    const extracted = extractContentText(msg.content);
    const content =
      extracted.text ||
      (extracted.nonTextTypes.length > 0 ? `[${extracted.nonTextTypes.join(", ")}]` : "");

    return [
      this.record(hsid, "user_msg", entryId, ts, ctx, state, {
        role: "user",
        content,
        payload: this.withSubagent(
          ctx,
          state,
          compactObject({
            attribution: msg.attribution,
            content_parts: extracted.nonTextTypes.length > 0 ? extracted.nonTextTypes : undefined,
          }),
        ),
        attribution: { message_id: entryId },
      }),
    ];
  }

  private mapToolResultMessage(
    hsid: string,
    msg: OmpMessage,
    entries: OmpSessionEntry[],
    leafId: string | undefined,
    ctx: OmpContext,
    state: SessionState,
  ): CaptureRecord[] {
    const toolCallId = msg.toolCallId;
    if (!toolCallId) {
      return [];
    }
    const ts = toIsoTimestamp(msg.timestamp, this.clock);
    const entryId = this.findMessageEntryId(entries, leafId, "toolResult", msg, state.usedEntryIds);
    if (entryId) {
      state.usedEntryIds.add(entryId);
    }
    const extracted = extractContentText(msg.content);
    const details = msg.details;
    const mcpServer =
      details && typeof details.serverName === "string" ? details.serverName : undefined;
    const mcpToolName =
      details && typeof details.mcpToolName === "string" ? details.mcpToolName : undefined;

    return [
      this.record(hsid, "tool_result", toolCallId, ts, ctx, state, {
        role: "tool",
        parent_key: toolCallId,
        parent_kind: "tool_call",
        content: extracted.text,
        payload: this.withSubagent(
          ctx,
          state,
          compactObject({
            tool_name: msg.toolName,
            is_error: Boolean(msg.isError),
            mcp_server: mcpServer,
            mcp_tool_name: mcpToolName,
            message_id: entryId,
          }),
        ),
      }),
    ];
  }

  private mapTurnEnd(
    hsid: string,
    payload: OmpHookPayload,
    entries: OmpSessionEntry[],
    leafId: string | undefined,
    ctx: OmpContext,
    state: SessionState,
  ): CaptureRecord[] {
    const msg = payload.message;
    if (!msg || msg.role !== "assistant") {
      return [];
    }
    const ts = toIsoTimestamp(msg.timestamp, this.clock);
    const completedTs = toIsoTimestamp(msg.completedAt ?? msg.timestamp, this.clock);
    const entryId =
      this.findMessageEntryId(entries, leafId, "assistant", msg, state.usedEntryIds) ??
      msg.responseId ??
      `assistant:${msg.timestamp ?? ts}:${payload.turnIndex ?? 0}`;
    state.usedEntryIds.add(entryId);

    state.turnLastIndex = payload.turnIndex;
    state.turnLastStopReason = msg.stopReason;
    state.turnLastErrorMessage = msg.errorMessage;
    state.turnLastEntryId = entryId;
    state.turnLastTs = completedTs;

    const extracted = extractContentText(msg.content);
    if (!extracted.text) {
      // Pure tool-call or aborted steps have no reply text; their tool calls or outcome
      // record what happened without inflating reply counts.
      return [];
    }

    return [
      this.record(hsid, "assistant_msg", entryId, ts, ctx, state, {
        role: "assistant",
        content: extracted.text,
        session: this.sessionInfo(ctx, state, msg),
        payload: this.withSubagent(
          ctx,
          state,
          compactObject({
            model: msg.model,
            provider: msg.provider,
            stop_reason: msg.stopReason,
            turn_index: payload.turnIndex,
            response_id: msg.responseId,
            usage: pickUsage(msg.usage),
          }),
        ),
        attribution: {
          message_id: entryId,
          has_tool_calls: extracted.hasToolCalls,
        },
      }),
    ];
  }

  private closeTurn(
    hsid: string,
    closedBy: "session_stop" | "agent_end" | "session_shutdown",
    payload: OmpHookPayload,
    leafId: string | undefined,
    ctx: OmpContext,
    state: SessionState,
  ): CaptureRecord[] {
    if (!state.turnActive || state.turnOutcomeEmitted) {
      return [];
    }
    const lastMsg = payload.last_assistant_message;
    const stopReason = lastMsg?.stopReason ?? state.turnLastStopReason;
    const resultKind = stopReasonToResultKind(stopReason, closedBy);
    const turnKey =
      state.turnPromptEntryId ??
      state.turnLastEntryId ??
      leafId ??
      String(payload.turn_id ?? state.turnLastIndex ?? 0);
    const ts =
      closedBy === "session_stop"
        ? toIsoTimestamp(lastMsg?.completedAt ?? lastMsg?.timestamp ?? state.turnLastTs, this.clock)
        : closedBy === "agent_end" && state.turnLastTs
          ? state.turnLastTs
          : this.clock();

    state.turnOutcomeEmitted = true;
    state.turnActive = false;

    const reasonText =
      closedBy === "session_shutdown"
        ? state.turnLastErrorMessage || "session_shutdown"
        : state.turnLastErrorMessage || stopReason || resultKind;

    return [
      this.record(hsid, "outcome", `turn:${turnKey}`, ts, ctx, state, {
        role: "harness",
        content: reasonText,
        payload: this.withSubagent(
          ctx,
          state,
          compactObject({
            closed_by: closedBy,
            result_kind: resultKind,
            stop_reason: stopReason,
            turn_id: payload.turn_id ?? state.turnLastIndex,
            prompt_entry_id: state.turnPromptEntryId,
            last_reply_entry_id: state.turnLastEntryId,
          }),
        ),
      }),
    ];
  }

  private record(
    hsid: string,
    kind: EventKind,
    sourceKey: string,
    ts: string,
    ctx: OmpContext,
    state: SessionState,
    extra: Partial<CaptureRecord>,
  ): CaptureRecord {
    const rec: CaptureRecord = {
      v: RECORD_VERSION,
      agent_id: AGENT_ID,
      harness_session_id: hsid,
      kind,
      ts,
      source_key: sourceKey,
      session: extra.session ?? this.sessionInfo(ctx, state),
    };
    if (extra.role !== undefined) rec.role = extra.role;
    if (extra.content !== undefined) rec.content = extra.content;
    if (extra.payload && Object.keys(extra.payload).length > 0) rec.payload = extra.payload;
    if (extra.parent_key !== undefined) rec.parent_key = extra.parent_key;
    if (extra.parent_kind !== undefined) rec.parent_kind = extra.parent_kind;
    if (extra.seq !== undefined) rec.seq = extra.seq;
    if (extra.attribution !== undefined) rec.attribution = extra.attribution;
    return rec;
  }

  private sessionInfo(
    ctx: OmpContext,
    state: SessionState,
    msg?: OmpMessage,
    startedAt?: string,
  ): SessionInfo {
    const sessionFile = this.sessionFileOf(ctx);
    const agent = ctx.agent;
    const sub = isSubagent(agent);
    const metadata = compactObject({
      session_file: sessionFile,
      mode: ctx.mode,
      agent_kind: agent?.kind,
      agent_id: agent?.id,
      agent_name: agent?.name,
      agent_depth: agent?.depth,
      parent_agent_id: sub ? agent?.parentId : undefined,
      calling_session_id: sub ? state.callingSessionId : undefined,
      invocation_kind: sub ? state.invocationKind : undefined,
      parent_session_file: state.parentSessionFile,
      parent_session_id: state.parentSessionId,
      branch_entry_id: state.branchEntryId,
    });
    const info: SessionInfo = {};
    if (ctx.cwd) info.cwd = ctx.cwd;
    const project = projectOf(ctx.cwd);
    if (project) info.project = project;
    const model = formatModel(ctx.model, msg);
    if (model) info.model = model;
    if (this.harnessVersion) info.harness_version = this.harnessVersion;
    if (startedAt) info.started_at = startedAt;
    if (Object.keys(metadata).length > 0) info.metadata = metadata;
    return info;
  }

  private withSubagent(
    ctx: OmpContext,
    state: SessionState,
    payload: Record<string, unknown>,
  ): Record<string, unknown> {
    if (!isSubagent(ctx.agent)) {
      return payload;
    }
    return {
      ...payload,
      subagent: compactObject({
        id: ctx.agent?.id,
        name: ctx.agent?.name,
        depth: ctx.agent?.depth,
        parent_id: ctx.agent?.parentId,
        calling_session_id: state.callingSessionId,
      }),
    };
  }

  private resolveSubagentCaller(agent: OmpAgentInfo | undefined, state: SessionState): void {
    if (!isSubagent(agent) || state.callingSessionId) {
      return;
    }
    if (agent?.id) {
      const origin = this.pendingSpawns.get(agent.id);
      if (origin) {
        state.callingSessionId = origin.callingSessionId;
        state.invocationKind = origin.invocationKind;
        return;
      }
    }
    if (agent?.parentId) {
      const caller = this.agentToSession.get(agent.parentId);
      if (caller) {
        state.callingSessionId = caller;
      }
    }
  }

  private sessionIdOf(payload: OmpHookPayload, ctx: OmpContext): string | undefined {
    try {
      const id = ctx.sessionManager?.getSessionId?.();
      if (id) return id;
    } catch {
      // Ignore broken sessionManager in defensive host environments.
    }
    return payload.session_id || payload.sessionId || undefined;
  }

  private sessionFileOf(ctx: OmpContext): string | undefined {
    try {
      return ctx.sessionManager?.getSessionFile?.() || undefined;
    } catch {
      return undefined;
    }
  }

  private leafIdOf(ctx: OmpContext): string | undefined {
    try {
      return ctx.sessionManager?.getLeafId?.() || undefined;
    } catch {
      return undefined;
    }
  }

  private entriesOf(ctx: OmpContext): OmpSessionEntry[] {
    try {
      const list = ctx.sessionManager?.getEntries?.();
      return Array.isArray(list) ? list : [];
    } catch {
      return [];
    }
  }

  private hasConversationHistory(entries: OmpSessionEntry[]): boolean {
    return entries.some(
      (entry) =>
        (entry.type !== undefined && CONVERSATION_ENTRY_TYPES.has(entry.type)) ||
        (entry.type === "custom" && entry.customType === "session_exit"),
    );
  }

  private sessionStartTs(
    ctx: OmpContext,
    entries: OmpSessionEntry[],
    resumed: boolean,
  ): string {
    if (!resumed) {
      try {
        const headerTs = ctx.sessionManager?.getHeader?.()?.timestamp;
        if (headerTs !== undefined) {
          return toIsoTimestamp(headerTs, this.clock);
        }
      } catch {
        // Optional method.
      }
      if (entries.length > 0 && entries[0]?.timestamp !== undefined) {
        return toIsoTimestamp(entries[0].timestamp, this.clock);
      }
    }
    return this.clock();
  }

  private findMessageEntryId(
    entries: OmpSessionEntry[],
    leafId: string | undefined,
    role: string,
    msg: OmpMessage,
    usedEntryIds: Set<string>,
  ): string | undefined {
    // Pass 1: exact match by responseId, toolCallId, or timestamp (deterministic across replays).
    for (let i = entries.length - 1; i >= 0; i -= 1) {
      const entry = entries[i];
      if (!entry?.id || entry.type !== "message" || entry.message?.role !== role) {
        continue;
      }
      if (msg.responseId && entry.message.responseId === msg.responseId) {
        return entry.id;
      }
      if (msg.toolCallId && entry.message.toolCallId === msg.toolCallId) {
        return entry.id;
      }
      if (msg.timestamp !== undefined && entry.message.timestamp === msg.timestamp) {
        return entry.id;
      }
    }
    // Pass 2: latest matching role not yet claimed in this session.
    for (let i = entries.length - 1; i >= 0; i -= 1) {
      const entry = entries[i];
      if (
        entry?.id &&
        entry.type === "message" &&
        entry.message?.role === role &&
        !usedEntryIds.has(entry.id)
      ) {
        return entry.id;
      }
    }
    // Pass 3: leafId fallback when entries did not expose message bodies.
    if (leafId && !usedEntryIds.has(leafId)) {
      const leafEntry = entries.find((entry) => entry?.id === leafId);
      if (!leafEntry || (leafEntry.type === "message" && leafEntry.message?.role === role)) {
        return leafId;
      }
    }
    return undefined;
  }

  private stateOf(hsid: string): SessionState {
    let state = this.sessions.get(hsid);
    if (!state) {
      state = {
        turnActive: false,
        turnOutcomeEmitted: false,
        usedEntryIds: new Set<string>(),
      };
      this.sessions.set(hsid, state);
    }
    return state;
  }
}
