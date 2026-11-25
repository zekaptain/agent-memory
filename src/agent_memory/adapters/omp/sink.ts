/**
 * Non-blocking child-process sink feeding JSONL capture records to `mem records`.
 *
 * Guarantees:
 * - `write()` returns synchronously without waiting for the ledger or child I/O.
 * - Missing, dead, or stalled `mem` processes disable capture and report strictly once.
 * - The child process is unref'd so a slow or hung `mem` never blocks omp from exiting.
 * - On shutdown, stdin closes after a brief debounce window so events arriving right after
 *   `session_shutdown` (as seen on SIGINT/SIGTERM/SIGHUP) are still written.
 */

import { spawn } from "node:child_process";
import type { ChildProcess } from "node:child_process";
import type { CaptureRecord } from "./mapper.ts";

export interface SinkConfig {
  mem: string;
  home?: string;
  port?: number;
  spoolOnly?: boolean;
  /** How long stdin may stay backpressured without draining before treating `mem` as stalled. */
  stallTimeoutMs?: number;
  /** Maximum bytes buffered in JS waiting for the pipe before disabling capture. */
  maxBufferedBytes?: number;
  /** Grace window before closing stdin once all sessions have shut down. */
  closeDelayMs?: number;
}

export type FailureReporter = (message: string) => void;

const DEFAULT_STALL_TIMEOUT_MS = 2_000;
const DEFAULT_MAX_BUFFERED_BYTES = 512 * 1024;
const DEFAULT_CLOSE_DELAY_MS = 50;

export function buildMemArgs(config: SinkConfig): string[] {
  const args: string[] = [];
  if (config.home) {
    args.push("--home", config.home);
  }
  if (config.port !== undefined) {
    args.push("--port", String(config.port));
  }
  args.push("records");
  if (config.spoolOnly) {
    args.push("--spool-only");
  }
  return args;
}

export class RecordSink {
  private readonly config: SinkConfig;
  private readonly report: FailureReporter;
  private readonly stallTimeoutMs: number;
  private readonly maxBufferedBytes: number;
  private readonly closeDelayMs: number;

  private child: ChildProcess | null = null;
  private started = false;
  private closing = false;
  private closeScheduled = false;
  private failed = false;
  private reported = false;
  private pendingBytes = 0;
  private lastStderr = "";
  private stallTimer: ReturnType<typeof setTimeout> | null = null;
  private closeTimer: ReturnType<typeof setTimeout> | null = null;

  constructor(config: SinkConfig, report: FailureReporter) {
    this.config = config;
    this.report = report;
    this.stallTimeoutMs = config.stallTimeoutMs ?? DEFAULT_STALL_TIMEOUT_MS;
    this.maxBufferedBytes = config.maxBufferedBytes ?? DEFAULT_MAX_BUFFERED_BYTES;
    this.closeDelayMs = config.closeDelayMs ?? DEFAULT_CLOSE_DELAY_MS;
  }

  get isFailed(): boolean {
    return this.failed;
  }

  /**
   * Queue one capture record for `mem records`. Always returns synchronously.
   */
  write(record: CaptureRecord): void {
    if (this.failed || this.closing) {
      return;
    }
    const wasClosingSoon = this.closeScheduled;
    if (wasClosingSoon) {
      this.cancelScheduledClose();
    }
    if (!this.started) {
      this.start();
    }
    const stdin = this.child?.stdin;
    if (this.failed || !stdin || stdin.destroyed || stdin.writableEnded) {
      return;
    }

    let line: string;
    try {
      line = JSON.stringify(record) + "\n";
    } catch (err) {
      this.failOnce(`could not serialize capture record (${errorMessage(err)})`);
      return;
    }

    const bytes = Buffer.byteLength(line);
    if (this.pendingBytes + bytes > this.maxBufferedBytes) {
      this.failOnce("mem records is not keeping up with input; capture paused for this process");
      this.teardownChild();
      return;
    }

    this.pendingBytes += bytes;
    try {
      const ok = stdin.write(line, (err) => {
        this.pendingBytes = Math.max(0, this.pendingBytes - bytes);
        if (this.pendingBytes === 0) {
          this.clearStallTimer();
        }
        if (err && !this.closing && !this.failed) {
          this.failOnce(`mem records pipe failed (${errorMessage(err)})`);
          this.teardownChild();
        }
      });
      if (!ok) {
        this.armStallTimer();
      }
    } catch (err) {
      this.pendingBytes = Math.max(0, this.pendingBytes - bytes);
      this.failOnce(`could not write to mem records (${errorMessage(err)})`);
      this.teardownChild();
      return;
    }

    if (wasClosingSoon) {
      // Re-arm the shutdown debounce window after a post-shutdown signal tail event.
      this.scheduleClose();
    }
  }

  /**
   * Schedule closing stdin after a short grace window so post-shutdown signal events
   * still make it into the pipe. Safe to call repeatedly.
   */
  scheduleClose(): void {
    if (!this.started || this.failed || this.closing) {
      return;
    }
    this.cancelScheduledClose();
    this.closeScheduled = true;
    if (this.closeDelayMs <= 0) {
      this.closeNow();
      return;
    }
    this.closeTimer = setTimeout(() => {
      this.closeTimer = null;
      this.closeScheduled = false;
      this.closeNow();
    }, this.closeDelayMs);
    this.closeTimer.unref?.();
  }

  /**
   * End stdin immediately so `mem records` flushes its last pass and exits cleanly.
   */
  closeNow(): void {
    this.cancelScheduledClose();
    this.closeScheduled = false;
    this.clearStallTimer();
    if (!this.started || this.closing) {
      return;
    }
    this.closing = true;
    const stdin = this.child?.stdin;
    if (stdin && !stdin.destroyed && !stdin.writableEnded) {
      try {
        stdin.end();
      } catch {
        // Ignore pipe errors during shutdown.
      }
    }
  }

  private start(): void {
    this.started = true;
    this.closing = false;
    const args = buildMemArgs(this.config);

    let child: ChildProcess;
    try {
      child = spawn(this.config.mem, args, {
        stdio: ["pipe", "ignore", "pipe"],
      });
    } catch (err) {
      this.failOnce(`could not start ${this.config.mem} (${errorMessage(err)})`);
      return;
    }

    this.child = child;
    child.unref();

    const stdin = child.stdin;
    if (stdin) {
      stdin.on("drain", () => {
        this.clearStallTimer();
      });
      stdin.on("error", (err) => {
        if (this.closing) {
          return;
        }
        this.failOnce(`mem records pipe failed (${errorMessage(err)})`);
        this.teardownChild();
      });
    }

    const stderr = child.stderr;
    if (stderr) {
      stderr.setEncoding("utf8");
      stderr.on("data", (chunk: string) => {
        const lines = chunk
          .split(/\r?\n/)
          .map((line) => line.trim())
          .filter(Boolean);
        if (lines.length > 0) {
          this.lastStderr = lines[lines.length - 1];
        }
      });
      stderr.on("error", () => {
        // Ignore stderr pipe errors.
      });
      (stderr as unknown as { unref?: () => void }).unref?.();
    }

    child.on("error", (err) => {
      this.failOnce(`could not start ${this.config.mem} (${errorMessage(err)})`);
      this.teardownChild();
    });

    child.on("exit", (code, signal) => {
      this.clearStallTimer();
      this.child = null;
      if (!this.closing || (code !== null && code !== 0) || signal !== null) {
        this.failed = true;
      }
    });

    child.on("close", (code, signal) => {
      this.clearStallTimer();
      this.child = null;
      if (this.closing && code === 0) {
        return;
      }
      if (!this.closing || (code !== null && code !== 0) || signal !== null) {
        const detail = signal ? `signal ${signal}` : `exit code ${code ?? "unknown"}`;
        const extra = this.lastStderr ? `: ${this.lastStderr}` : "";
        this.failOnce(`mem records stopped (${detail})${extra}`);
      }
    });
  }

  private armStallTimer(): void {
    if (this.stallTimer || this.failed) {
      return;
    }
    this.stallTimer = setTimeout(() => {
      this.stallTimer = null;
      if (!this.failed && this.pendingBytes > 0) {
        this.failOnce(
          `mem records stopped reading input for ${this.stallTimeoutMs}ms; capture paused`,
        );
        this.teardownChild();
      }
    }, this.stallTimeoutMs);
    this.stallTimer.unref?.();
  }

  private clearStallTimer(): void {
    if (this.stallTimer) {
      clearTimeout(this.stallTimer);
      this.stallTimer = null;
    }
  }

  private cancelScheduledClose(): void {
    if (this.closeTimer) {
      clearTimeout(this.closeTimer);
      this.closeTimer = null;
    }
  }

  private teardownChild(): void {
    this.clearStallTimer();
    this.cancelScheduledClose();
    const child = this.child;
    this.child = null;
    if (!child) {
      return;
    }
    try {
      child.stdin?.destroy();
    } catch {
      // Ignore.
    }
    try {
      child.stderr?.destroy();
    } catch {
      // Ignore.
    }
  }

  private failOnce(detail: string): void {
    this.failed = true;
    if (this.reported) {
      return;
    }
    this.reported = true;
    try {
      this.report(`agent-memory: ${detail}`);
    } catch {
      // Never let failure reporting crash omp.
    }
  }
}

function errorMessage(err: unknown): string {
  if (err instanceof Error) {
    return err.message;
  }
  return String(err);
}
