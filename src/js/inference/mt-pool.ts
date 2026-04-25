/**
 * Multi-threaded WASM worker pool.
 *
 * Each worker holds its own WebAssembly instance imported with the same
 * SharedArrayBuffer-backed `WebAssembly.Memory` as the main thread.
 * Workers and main read and write the same address space, so model
 * weights (770 MB) live once in shared memory rather than being
 * duplicated per worker.
 *
 * Sync: a per-task "epoch" counter sits in a shared Int32Array. Main
 * increments the epoch and broadcasts a task; each worker checks the
 * epoch, runs its slice, and atomic-increments a "done" counter. Main
 * waits via `Atomics.waitAsync` (browser-friendly) until done == N.
 *
 * The worker script is created from a Blob URL containing inlined JS
 * + the WASM bytes (via SharedArrayBuffer reference, not a copy), so
 * the package ships as one bundle with no separate worker file.
 */

import { PII_WASM_MT_BYTES } from "../backends/pii-wasm-inline.js";
import type { PiiWasmExports } from "../backends/wasm.js";

/** One WASM kernel call: function name + args (all numbers). */
export interface KernelCall {
  kernel: keyof PiiWasmExports;
  args: number[];
}

/** Per-worker script: a sequence of kernel calls to run in order. */
export type WorkerScript = readonly KernelCall[];

const SIGNAL_BYTES = 64; // one cache line, two atomic counters live here.
const EPOCH_OFFSET = 0;  // increments with each batch of tasks.
const DONE_OFFSET = 1;   // workers atomic-add when finished.

/**
 * Inline worker script. Branches on whether `parentPort` from
 * `node:worker_threads` is available so the same source runs in both
 * the browser (Web Worker) and Node (worker_threads.Worker).
 */
const WORKER_SCRIPT = `
const _isNode = typeof process !== "undefined"
  && !!(process.versions && process.versions.node);
let _postMessage;
let _onMessage;
if (_isNode) {
  const { parentPort } = require("node:worker_threads");
  _postMessage = (msg) => parentPort.postMessage(msg);
  _onMessage = (handler) => parentPort.on("message", (data) => handler({ data }));
} else {
  _postMessage = (msg) => self.postMessage(msg);
  _onMessage = (handler) => { self.onmessage = handler; };
}

let wasm;
let signal;

_onMessage(async (e) => {
  const msg = e.data;
  if (msg.type === "init") {
    const { instance } = await WebAssembly.instantiate(msg.wasmBytes, {
      env: { memory: msg.memory },
    });
    wasm = instance.exports;
    signal = new Int32Array(msg.signalBuffer);
    _postMessage({ type: "ready" });
    return;
  }
  if (msg.type === "script") {
    const calls = msg.calls;
    for (let i = 0; i < calls.length; i++) {
      const call = calls[i];
      const fn = wasm[call.kernel];
      if (!fn) {
        _postMessage({ type: "error", error: "unknown kernel " + call.kernel });
        return;
      }
      fn.apply(null, call.args);
    }
    Atomics.add(signal, ${DONE_OFFSET}, 1);
    Atomics.notify(signal, ${DONE_OFFSET});
  }
});
`;

interface WorkerLike {
  postMessage(msg: unknown): void;
  addEventListener(type: "message", handler: (e: MessageEvent) => void): void;
  removeEventListener(type: "message", handler: (e: MessageEvent) => void): void;
  terminate(): void;
}

function isNodeRuntime(): boolean {
  return (
    typeof process !== "undefined" &&
    !!(process as { versions?: { node?: string } }).versions?.node
  );
}

let cachedBlobUrl: string | null = null;

async function spawnWorker(): Promise<WorkerLike> {
  if (isNodeRuntime()) {
    // Use Node's worker_threads. Adapt its `on("message")` to the
    // EventTarget shape the browser exposes so the rest of the pool
    // is environment-agnostic.
    const wt = await import("node:worker_threads");
    const w = new wt.Worker(WORKER_SCRIPT, { eval: true });
    const listeners = new Map<
      (e: MessageEvent) => void,
      (data: unknown) => void
    >();
    return {
      postMessage(msg: unknown) {
        w.postMessage(msg);
      },
      addEventListener(_type, handler) {
        const wrap = (data: unknown): void => handler({ data } as MessageEvent);
        listeners.set(handler, wrap);
        w.on("message", wrap);
      },
      removeEventListener(_type, handler) {
        const wrap = listeners.get(handler);
        if (wrap) {
          w.off("message", wrap);
          listeners.delete(handler);
        }
      },
      terminate() {
        w.terminate();
      },
    };
  }
  if (!cachedBlobUrl) {
    const blob = new Blob([WORKER_SCRIPT], { type: "application/javascript" });
    cachedBlobUrl = URL.createObjectURL(blob);
  }
  return new Worker(cachedBlobUrl, { type: "classic" }) as unknown as WorkerLike;
}

/**
 * Spin-wait helper for environments where `Atomics.waitAsync` isn't
 * available (older Safari pre-16.4). Cheap because each iteration is
 * a single atomic load and `Atomics.wait` from a worker thread blocks
 * only nanoseconds when the value matches.
 */
async function waitForCount(
  signal: Int32Array,
  index: number,
  target: number,
): Promise<void> {
  const Atomics_ = Atomics as unknown as {
    waitAsync?: (
      buf: Int32Array,
      idx: number,
      val: number,
    ) => { async: boolean; value: Promise<"ok" | "not-equal" | "timed-out"> | "ok" | "not-equal" | "timed-out" };
  };
  while (true) {
    const cur = Atomics.load(signal, index);
    if (cur >= target) return;
    if (Atomics_.waitAsync) {
      const result = Atomics_.waitAsync(signal, index, cur);
      if (result.async) await result.value;
    } else {
      // Microtask yield so the event loop can run worker postMessage handlers.
      await new Promise<void>((r) => queueMicrotask(r));
    }
  }
}

export class MtPool {
  readonly numThreads: number;
  private workers: WorkerLike[] = [];
  private signal: Int32Array;

  constructor(public readonly memory: WebAssembly.Memory, numThreads: number) {
    this.numThreads = Math.max(1, numThreads);
    const sigBuf = new SharedArrayBuffer(SIGNAL_BYTES);
    this.signal = new Int32Array(sigBuf);
  }

  async warmup(): Promise<void> {
    const inits: Promise<void>[] = [];
    for (let i = 0; i < this.numThreads; i++) {
      const w = await spawnWorker();
      this.workers.push(w);
      inits.push(
        new Promise<void>((resolve, reject) => {
          const onMsg = (e: MessageEvent): void => {
            const data = e.data as { type: string; error?: string };
            if (data.type === "ready") {
              w.removeEventListener("message", onMsg);
              resolve();
            } else if (data.type === "error") {
              reject(new Error(`worker init failed: ${data.error}`));
            }
          };
          w.addEventListener("message", onMsg);
          w.postMessage({
            type: "init",
            memory: this.memory,
            signalBuffer: this.signal.buffer,
            wasmBytes: PII_WASM_MT_BYTES,
          });
        }),
      );
    }
    await Promise.all(inits);
  }

  /**
   * Run a per-worker script in parallel. `scripts.length` must equal
   * `this.numThreads`; each script is a sequence of kernel calls the
   * worker runs in order. Returns when every worker has signalled
   * completion via the shared atomic counter.
   *
   * Caller is responsible for ensuring scripts don't write to
   * overlapping shared-memory regions — the pool doesn't enforce
   * disjointness.
   */
  async run(scripts: readonly WorkerScript[]): Promise<void> {
    if (scripts.length !== this.numThreads) {
      throw new Error(
        `MtPool.run: expected ${this.numThreads} scripts, got ${scripts.length}`,
      );
    }
    Atomics.store(this.signal, DONE_OFFSET, 0);
    Atomics.add(this.signal, EPOCH_OFFSET, 1);
    for (let i = 0; i < this.numThreads; i++) {
      this.workers[i]!.postMessage({ type: "script", calls: scripts[i] });
    }
    await waitForCount(this.signal, DONE_OFFSET, this.numThreads);
  }

  dispose(): void {
    for (const w of this.workers) w.terminate();
    this.workers = [];
  }
}
