/**
 * Multi-threaded WASM worker pool.
 *
 * Each worker holds its own WebAssembly instance imported with the same
 * SharedArrayBuffer-backed `WebAssembly.Memory` as the main thread.
 * Workers and main read and write the same address space, so model
 * weights (770 MB) live once in shared memory rather than being
 * duplicated per worker.
 *
 * Hot-path dispatch: shared-memory + atomics, no postMessage. Each
 * worker spins on `Atomics.wait(signal, EPOCH)` and wakes on
 * `Atomics.notify` from main (futex-level, ~µs latency vs ~50–200µs
 * postMessage delivery). Per-worker kernel call lists are encoded as
 * a Float64Array slot inside the WASM memory SAB; the WASM-memory
 * memory-fence atomic provides release/acquire across the slot writes
 * AND the kernel inputs/outputs in one go.
 *
 * Init still uses postMessage (one-time cost) so workers can receive
 * the wasmBytes / memory handle / per-worker stack top.
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

/**
 * Kernels that worker scripts are allowed to dispatch. Order matters —
 * the index of each name is the kernel ID encoded into worker slots.
 * Add a new kernel here when you want to call it from inside a
 * worker script.
 */
const DISPATCHABLE_KERNELS: readonly (keyof PiiWasmExports)[] = [
  "echo",
  "matmul_fp16_x_int4block",
  "matmul_fp16_x_int4block_out_f32",
  "matmul_f32_x_int4block",
  "matmul_f32_x_int4block_out_f32",
  "rope_apply",
  "banded_attention",
  "banded_attention_partial",
  "scale_fp16_inplace",
  "add_fp16",
  "rms_norm",
  "rms_norm_fp16_to_f32",
  "add_rmsnorm_fp16_to_f32",
  "gather_fp16",
  "gather_f32",
  "scatter_add_weighted_f32",
  "scatter_add_weighted_f32_scalar",
  "zero_f32",
  "convert_fp16_to_f32",
  "cast_f32_to_fp16_scaled",
  "softmax_f32",
  "swiglu_clamp_f32",
  "topk_partial_f32",
  "embed_lookup_int4",
];

const KERNEL_ID = new Map<string, number>(
  DISPATCHABLE_KERNELS.map((name, idx) => [name as string, idx]),
);

// Slot layout (per worker, in WASM memory, viewed as Float64Array):
//   [n_calls, ...repeated MAX_CALLS times of [kernel_id, n_args, args[0..MAX_ARGS-1]]]
// `f64` per cell avoids reinterpret tricks: pointers + sizes are exact
// integer-representable, scalar f32 args (eps, scale) round-trip
// losslessly enough for our kernels' tolerance.
const MAX_ARGS = 20;
const MAX_CALLS = 256;
const HEADER_F64S = 1;
const PER_CALL_F64S = 2 + MAX_ARGS;
const SLOT_F64S = HEADER_F64S + MAX_CALLS * PER_CALL_F64S;
export const MT_POOL_SLOT_BYTES = SLOT_F64S * 8;

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

const HEADER_F64S = ${HEADER_F64S};
const PER_CALL_F64S = ${PER_CALL_F64S};
const SLOT_F64S = ${SLOT_F64S};
const EPOCH_OFFSET = ${EPOCH_OFFSET};
const DONE_OFFSET = ${DONE_OFFSET};

let wasm;
let signal;
let memFence;
let slotsF64;
let workerIdx;
let kernelTable;

_onMessage(async (e) => {
  const msg = e.data;
  if (msg.type === "init") {
    const { instance } = await WebAssembly.instantiate(msg.wasmBytes, {
      env: { memory: msg.memory },
    });
    wasm = instance.exports;
    signal = new Int32Array(msg.signalBuffer);
    memFence = new Int32Array(msg.memory.buffer, msg.memFenceOffset, 1);
    slotsF64 = new Float64Array(msg.memory.buffer, msg.slotsByteOffset, SLOT_F64S * msg.numThreads);
    workerIdx = msg.workerIdx;
    kernelTable = msg.kernelNames.map((name) => {
      const fn = wasm[name];
      if (!fn) throw new Error("worker missing kernel: " + name);
      return fn;
    });
    // Each WebAssembly.Instance has its own __stack_pointer global
    // initialized to the same default — but the *stack memory itself*
    // lives in shared linear memory, so without per-instance stacks
    // every worker pushes locals to the same address and corrupts
    // each other's stack frames. Point this worker's stack at the
    // dedicated region the main thread allocated for it.
    if (typeof msg.stackTop === "number" && wasm.__stack_pointer) {
      wasm.__stack_pointer.value = msg.stackTop;
    }
    _postMessage({ type: "ready" });
    runDispatchLoop();
    return;
  }
});

function runDispatchLoop() {
  let lastEpoch = 0;
  const slotBase = workerIdx * SLOT_F64S;
  while (true) {
    // Block until main increments the epoch. \`Atomics.wait\` returns
    // immediately if the value already differs from \`lastEpoch\` (e.g.
    // if main bumped the epoch while we were processing the previous
    // task and hadn't yet looped back here).
    Atomics.wait(signal, EPOCH_OFFSET, lastEpoch);
    const newEpoch = Atomics.load(signal, EPOCH_OFFSET);
    if (newEpoch === lastEpoch) continue;
    lastEpoch = newEpoch;

    // Acquire fence on the WASM-memory SAB — pairs with main's release
    // \`Atomics.add(memFence, 0, 0)\` before the notify. Publishes both
    // the slot writes AND any kernel-input data main staged in WASM
    // memory before this dispatch.
    Atomics.load(memFence, 0);

    const nCalls = slotsF64[slotBase] | 0;
    for (let i = 0; i < nCalls; i++) {
      const callBase = slotBase + HEADER_F64S + i * PER_CALL_F64S;
      const kernelId = slotsF64[callBase] | 0;
      const nArgs = slotsF64[callBase + 1] | 0;
      const fn = kernelTable[kernelId];
      // Hand-unrolled dispatch by arity — avoids array allocation +
      // \`apply\` overhead in the hot path. Covers 0..20 args; the
      // generic apply path is the safety net for kernels that grow.
      const a = callBase + 2;
      switch (nArgs) {
        case 0: fn(); break;
        case 1: fn(slotsF64[a]); break;
        case 2: fn(slotsF64[a], slotsF64[a+1]); break;
        case 3: fn(slotsF64[a], slotsF64[a+1], slotsF64[a+2]); break;
        case 4: fn(slotsF64[a], slotsF64[a+1], slotsF64[a+2], slotsF64[a+3]); break;
        case 5: fn(slotsF64[a], slotsF64[a+1], slotsF64[a+2], slotsF64[a+3], slotsF64[a+4]); break;
        case 6: fn(slotsF64[a], slotsF64[a+1], slotsF64[a+2], slotsF64[a+3], slotsF64[a+4], slotsF64[a+5]); break;
        case 7: fn(slotsF64[a], slotsF64[a+1], slotsF64[a+2], slotsF64[a+3], slotsF64[a+4], slotsF64[a+5], slotsF64[a+6]); break;
        case 8: fn(slotsF64[a], slotsF64[a+1], slotsF64[a+2], slotsF64[a+3], slotsF64[a+4], slotsF64[a+5], slotsF64[a+6], slotsF64[a+7]); break;
        case 9: fn(slotsF64[a], slotsF64[a+1], slotsF64[a+2], slotsF64[a+3], slotsF64[a+4], slotsF64[a+5], slotsF64[a+6], slotsF64[a+7], slotsF64[a+8]); break;
        case 10: fn(slotsF64[a], slotsF64[a+1], slotsF64[a+2], slotsF64[a+3], slotsF64[a+4], slotsF64[a+5], slotsF64[a+6], slotsF64[a+7], slotsF64[a+8], slotsF64[a+9]); break;
        case 11: fn(slotsF64[a], slotsF64[a+1], slotsF64[a+2], slotsF64[a+3], slotsF64[a+4], slotsF64[a+5], slotsF64[a+6], slotsF64[a+7], slotsF64[a+8], slotsF64[a+9], slotsF64[a+10]); break;
        case 12: fn(slotsF64[a], slotsF64[a+1], slotsF64[a+2], slotsF64[a+3], slotsF64[a+4], slotsF64[a+5], slotsF64[a+6], slotsF64[a+7], slotsF64[a+8], slotsF64[a+9], slotsF64[a+10], slotsF64[a+11]); break;
        case 13: fn(slotsF64[a], slotsF64[a+1], slotsF64[a+2], slotsF64[a+3], slotsF64[a+4], slotsF64[a+5], slotsF64[a+6], slotsF64[a+7], slotsF64[a+8], slotsF64[a+9], slotsF64[a+10], slotsF64[a+11], slotsF64[a+12]); break;
        default: {
          const args = new Array(nArgs);
          for (let j = 0; j < nArgs; j++) args[j] = slotsF64[a + j];
          fn.apply(null, args);
        }
      }
    }

    // Release fence then signal completion.
    Atomics.add(memFence, 0, 0);
    Atomics.add(signal, DONE_OFFSET, 1);
    Atomics.notify(signal, DONE_OFFSET);
  }
}
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
  /**
   * A slot inside the *WASM memory* SAB used as a fence target.
   * Atomics.store/load on this slot synchronizes plain WASM-memory
   * reads/writes across threads — Atomics on a *different* SAB
   * (e.g. our signal buffer) doesn't fence accesses on the memory
   * SAB by spec.
   *
   * The slot offset is set externally via `setMemoryFenceSlot()` so
   * the caller can pin a 4-byte location they aren't using for data.
   */
  private memFence: Int32Array | null = null;
  private slotsF64: Float64Array | null = null;
  private slotsByteOffset = 0;

  constructor(public readonly memory: WebAssembly.Memory, numThreads: number) {
    this.numThreads = Math.max(1, numThreads);
    const sigBuf = new SharedArrayBuffer(SIGNAL_BYTES);
    this.signal = new Int32Array(sigBuf);
  }

  /**
   * Pin a 4-byte slot inside the WASM memory as the cross-thread
   * memory fence target. Caller is responsible for not using these
   * 4 bytes for data.
   */
  setMemoryFenceSlot(byteOffset: number): void {
    this.memFence = new Int32Array(this.memory.buffer, byteOffset, 1);
  }

  /**
   * Pin the per-worker kernel-call slot region inside WASM memory.
   * Size required: `MT_POOL_SLOT_BYTES * numThreads`. Workers parse
   * their slot on each dispatch instead of receiving a postMessage.
   */
  setSlotsBuffer(byteOffset: number): void {
    this.slotsByteOffset = byteOffset;
    this.slotsF64 = new Float64Array(this.memory.buffer, byteOffset, SLOT_F64S * this.numThreads);
  }

  /**
   * Per-worker shadow-stack tops. Each worker's `__stack_pointer` is
   * initialized to the corresponding `stackTops[i]` so workers don't
   * collide on the default shared shadow-stack region. Stack grows
   * downward from the top, so each top should be at the END of a
   * dedicated per-worker region.
   */
  private stackTops: number[] = [];
  setWorkerStackTops(tops: readonly number[]): void {
    if (tops.length !== this.numThreads) {
      throw new Error(
        `MtPool.setWorkerStackTops: need ${this.numThreads} entries, got ${tops.length}`,
      );
    }
    this.stackTops = [...tops];
  }

  async warmup(): Promise<void> {
    if (!this.memFence) throw new Error("MtPool.warmup: memFence not set; call setMemoryFenceSlot first");
    if (!this.slotsF64) throw new Error("MtPool.warmup: slots buffer not set; call setSlotsBuffer first");
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
            memFenceOffset: this.memFence!.byteOffset,
            slotsByteOffset: this.slotsByteOffset,
            numThreads: this.numThreads,
            workerIdx: i,
            kernelNames: DISPATCHABLE_KERNELS,
            stackTop: this.stackTops[i] ?? null,
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
    const slots = this.slotsF64;
    if (!slots) throw new Error("MtPool.run: slots buffer not initialised");

    Atomics.store(this.signal, DONE_OFFSET, 0);

    // Encode each worker's script into its slot. `slots` is a
    // Float64Array on the SAB-backed WASM memory; plain assignments
    // are non-atomic but become visible after the fence below.
    for (let w = 0; w < this.numThreads; w++) {
      const script = scripts[w]!;
      if (script.length > MAX_CALLS) {
        throw new Error(`MtPool.run: worker ${w} script has ${script.length} calls, max ${MAX_CALLS}`);
      }
      const base = w * SLOT_F64S;
      slots[base] = script.length;
      for (let i = 0; i < script.length; i++) {
        const call = script[i]!;
        const id = KERNEL_ID.get(call.kernel);
        if (id === undefined) {
          throw new Error(`MtPool.run: kernel '${String(call.kernel)}' not in DISPATCHABLE_KERNELS`);
        }
        if (call.args.length > MAX_ARGS) {
          throw new Error(`MtPool.run: kernel '${String(call.kernel)}' has ${call.args.length} args, max ${MAX_ARGS}`);
        }
        const callBase = base + HEADER_F64S + i * PER_CALL_F64S;
        slots[callBase] = id;
        slots[callBase + 1] = call.args.length;
        for (let j = 0; j < call.args.length; j++) {
          slots[callBase + 2 + j] = call.args[j]!;
        }
      }
    }

    // Release fence on the WASM-memory SAB before incrementing epoch
    // — pairs with each worker's Atomics.load(memFence, 0) acquire
    // after its Atomics.wait returns.
    if (this.memFence) Atomics.add(this.memFence, 0, 0);
    Atomics.add(this.signal, EPOCH_OFFSET, 1);
    Atomics.notify(this.signal, EPOCH_OFFSET, this.numThreads);

    await waitForCount(this.signal, DONE_OFFSET, this.numThreads);

    if (this.memFence) Atomics.add(this.memFence, 0, 0);
  }

  dispose(): void {
    for (const w of this.workers) w.terminate();
    this.workers = [];
  }
}
