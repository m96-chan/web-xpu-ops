/**
 * The harness contract without the harness's runtime: every type a kernel
 * caller needs, `params`, `compilationFailure`, and the kernel-source registry
 * — and **no import of the `webgpu` package**.
 *
 * Issue #224. `wgsl.ts` and `resident.ts` import Dawn's Node binding at module
 * scope, which is right for the test runner and fatal for a browser bundle:
 * every demo under `examples/*-web` carried an esbuild plugin whose only job
 * was to swap the whole harness for a hand-maintained copy, and the LLM engine
 * could not be published at all because `llm/kernels.ts` read its WGSL through
 * `readFileSync`. Everything below is what those files needed and none of what
 * they could not have. `wgsl.ts` and `resident.ts` re-export it, so nothing
 * that imported from them has to change.
 *
 * The interfaces are moved, not copied — the docs on them are the reasoning
 * the harness accumulated (why bind groups take whole buffers, why a zero
 * timestamp is not a duration) and it belongs with the contract, wherever the
 * contract is imported from.
 */

/** One entry of a shader's `@group(0)` layout, in binding order. */
export type Binding =
  | { kind: "storage"; data: Float32Array | Int32Array | Uint32Array }
  | { kind: "out"; type: "f32" | "i32" | "u32"; length: number }
  /**
   * Storage the kernel uses but nobody supplies or inspects: no upload, no
   * readback, contents undefined (zero in practice).
   *
   * It exists for the roofline calibration, which streams hundreds of megabytes
   * to find out how fast this device moves memory. Through `storage` that would
   * upload the buffer, and through `out` it would copy it back — either one puts
   * a transfer of the same size next to the thing being timed, which is exactly
   * what must not happen when the measurement *is* the transfer rate.
   */
  | { kind: "scratch"; length: number }
  | { kind: "uniform"; data: ArrayBuffer };

export interface Dispatch {
  code: string;
  entry?: string;
  bindings: Binding[];
  workgroups: [number] | [number, number] | [number, number, number];
}

export interface Runner {
  run(dispatch: Dispatch): Promise<(Float32Array | Int32Array | Uint32Array)[]>;
  /**
   * Seconds of GPU time for the dispatch, or null when this device cannot say.
   *
   * Read from a timestamp query written around the compute pass, not from a
   * clock on the host. Wall-clock here measures buffer creation, submission and
   * the round trip waiting for a mapped readback — about a millisecond on this
   * machine, which is several times a real dispatch and swamps exactly the
   * quantity being measured. Measured while building the roofline: a wall-clock
   * slope reported 5.3 TB/s on a card whose ceiling is 1.8.
   *
   * `timestamp-query` is optional, and the devices most in need of an honest
   * ceiling advertise the fewest features. Null rather than a guess is the point
   * — rule 9 says an unmeasured figure must say so, and a fabricated one is
   * worse than none because it looks authoritative.
   */
  time(dispatch: Dispatch): Promise<number | null>;
  destroy(): void;
}

/**
 * The reason a shader could not be used, or null when it compiled.
 *
 * Split out as a pure function so it can be tested without a device. The half
 * that needs a GPU — whether Dawn reports a bad shader at all — is platform
 * behaviour and is measured in `harness/README` notes and issue #46. The half
 * that is ours is this: given messages, do we refuse. That distinction matters
 * because provoking a real compile failure crashes this binding in roughly four
 * runs in five, so an end-to-end test of it cannot be kept green, while this
 * can.
 */
export function compilationFailure(
  messages: readonly Pick<GPUCompilationMessage, "type" | "lineNum" | "linePos" | "message">[],
): string | null {
  const errors = messages.filter((message) => message.type === "error");
  if (errors.length === 0) return null;
  const where = (m: (typeof errors)[number]) => `${m.lineNum}:${m.linePos}: ${m.message}`;
  return `shader failed to compile\n${errors.map(where).join("\n")}`;
}

/** Packs a params struct of mixed u32 / i32 / f32 into a uniform buffer. */
export function params(fields: ["u32" | "i32" | "f32", number][]): ArrayBuffer {
  const buffer = new ArrayBuffer(Math.max(16, fields.length * 4));
  const view = new DataView(buffer);
  fields.forEach(([kind, value], index) => {
    if (kind === "f32") view.setFloat32(index * 4, value, true);
    else if (kind === "i32") view.setInt32(index * 4, value, true);
    else view.setUint32(index * 4, value, true);
  });
  return buffer;
}

export type ResidentOp =
  | { kind: "dispatch"; pipeline: GPUComputePipeline; bindGroup: GPUBindGroup; workgroups: [number] | [number, number] | [number, number, number] }
  /**
   * A GPU-to-GPU byte copy recorded into the same encoder as every dispatch
   * around it — `llm/engine-q8-resident.ts` uses this for the KV-cache write
   * (see that file's doc for why a copy rather than `queue.writeBuffer`: the
   * new token's K/V already lives in a GPU buffer, written by `rope`'s own
   * dispatch a few ops earlier in the same batch, so routing it through the
   * CPU to satisfy `writeBuffer`'s `ArrayBufferView` signature would undo the
   * whole point of this module).
   */
  | { kind: "copy"; src: GPUBuffer; srcOffset: number; dst: GPUBuffer; dstOffset: number; size: number };

export interface ResidentReadback {
  /** Must be `MAP_READ | COPY_DST`, sized for `length * 4` bytes, and owned by the caller — never created here (see `stats.buffersCreated`'s doc). */
  staging: GPUBuffer;
  source: GPUBuffer;
  sourceOffset: number;
  length: number;
  type: "f32" | "i32" | "u32";
}

/**
 * Issue #131: an opt-in argument to `batch()` that asks it to also report
 * *where* one `batch()` call's own wall time went, split the way #131's own
 * background comment does — CPU submit-to-completion wait, the readback
 * `mapAsync` phase, and (when the device negotiated `timestamp-query` and
 * `labels` names a dispatch) that dispatch's own GPU-side duration.
 *
 * `labels` is parallel to the `ops` array `batch()` already takes — one
 * entry per op, `null`/absent for an op nobody asked to time individually.
 * A non-null label on a `"dispatch"` op makes `batch()` end whatever compute
 * pass was open and start a fresh one *just for that dispatch*, with its own
 * `timestampWrites` pair — WebGPU's `GPUComputePassTimestampWrites` only
 * covers the whole pass it is attached to (there is no per-dispatch
 * timestamp inside one pass), so per-dispatch GPU attribution costs a pass
 * boundary per labeled dispatch. That is real, measurable overhead next to
 * `llm/engine-q8-resident.ts`'s normal one-pass-per-batch encoding, and
 * exists only when a caller opts in by passing `labels` — every existing
 * caller (every real decode/prefill step) passes no third argument at all,
 * so nothing about its own encoding changes.
 *
 * `sink` is written into once `batch()` resolves, not returned separately —
 * the caller constructs it (typically `{ encodeMs: null,
 * submitToDoneMs: null, readbackMs: null, gpuEntries: [] }`) and passes the same object in, so a driving
 * script can read it straight off the object it already holds.
 */
export interface BatchProfile {
  labels?: (string | null | undefined)[];
  sink: BatchProfileSink;
}

export interface BatchProfileSink {
  /**
   * `performance.now()` elapsed from the top of `batch()` to the instant
   * before `queue.submit()` — recording the command buffer, and nothing that
   * waits on the GPU.
   *
   * Issue #182. Without it a forward's wall clock has a hole in it that reads
   * as GPU work: a browser forward summed 2014 ms of pass timestamps against
   * 3841 ms of wall, and the two fields below could not say where the rest
   * went because neither of them covers the encode. `null` until `batch()`
   * writes it.
   */
  encodeMs: number | null;
  /** `performance.now()` elapsed between `queue.submit()` and `queue.onSubmittedWorkDone()` resolving — the GPU-side wait `batch()` would otherwise fold silently into the readback `mapAsync` call below. `null` until `batch()` writes it. */
  submitToDoneMs: number | null;
  /** `performance.now()` elapsed across every `readback` entry's `mapAsync`+copy, timed *after* `onSubmittedWorkDone` above has already resolved — so this is the readback round trip on its own, not padded with GPU completion wait. `null` until `batch()` writes it. */
  readbackMs: number | null;
  /**
   * One entry per non-null `labels` entry whose pass produced a nonzero
   * timestamp delta (a zero delta means the driver declined to serve that
   * query — `wgsl.ts#dispatch`'s own doc on why zero is not reported as a
   * duration). Empty — not absent — when `labels` was given but this device
   * did not negotiate `timestamp-query` (`ResidentDevice.timestampsSupported`
   * is `false`): the caller can tell "no GPU breakdown" from "GPU breakdown
   * requested but every entry was exactly zero" by checking that flag
   * itself, not by inspecting this array's length alone.
   */
  gpuEntries: { label: string; seconds: number }[];
}

export interface ResidentDevice {
  /**
   * Counters a test can snapshot before and after a decode loop to prove the
   * loop itself allocates nothing (issue #110's "トークンループ内での
   * create*呼び出しゼロ") — the property code review alone cannot keep honest
   * once this file has more than one caller, per rule 1 ("観測点が間違っている"
   * failures do not show up in coverage).
   */
  readonly stats: {
    buffersCreated: number;
    pipelinesCreated: number;
    submits: number;
  /**
   * Wall spent inside `bindGroup`/`bindGroupSliced`, cumulative.
   *
   * Issue #182. `batch()`'s own timers said a browser forward spent 6 ms
   * recording, 61 ms waiting on the queue outside the passes and 0 ms reading
   * back — leaving 1653 ms of a 3645 ms forward in none of them, and therefore
   * *between* batches. Bind groups are the one thing built per dispatch there,
   * 3,238 of them a forward, and on this backend each one awaits a
   * `popErrorScope`. Whether that is where the time is is a measurement, which
   * is what this field is for.
   */
  bindGroupMs: number;
  /** How many were built, so the cost per bind group can be read off. */
  bindGroups: number;
  };
  /** Whether this device negotiated the `timestamp-query` feature — issue #131's `BatchProfile.sink.gpuEntries` is only ever populated when this is `true`; a caller on a device where it is `false` still gets `submitToDoneMs`/`readbackMs` (those need no GPU feature), just no per-dispatch GPU breakdown, and should say so rather than reporting an empty breakdown as "GPU took 0ms" (rule 9). */
  readonly timestampsSupported: boolean;
  createStorageBuffer(bytes: number, usage?: number): GPUBuffer;
  createUniformBuffer(bytes: number): GPUBuffer;
  /** `queue.writeBuffer`, not a submit — safe to call before `batch()`, never inside a decode loop's steady state except where the op's own doc says so (embedding upload, position counters). */
  upload(buffer: GPUBuffer, offset: number, data: ArrayBufferView): void;
  /** Compiles (or returns the cached module/pipeline for) `code`+`entry`. Async because shader validation is (`getCompilationInfo`); call during construction, not per token. */
  pipelineFor(code: string, entry?: string): Promise<GPUComputePipeline>;
  /**
   * `layout: "auto"`'s bind group for one pipeline, from already-created
   * whole buffers — see `wgsl.ts`'s note on why an error scope wraps this.
   *
   * Deliberately whole buffers, no byte offset into a shared one: WebGPU
   * requires a storage-binding offset to be a multiple of
   * `minStorageBufferOffsetAlignment` (measured 256 bytes on this device,
   * `harness/resident.test.ts`'s own history — an earlier version of this
   * API took `{ buffer, offset, size }` so `llm/engine-q8-resident.ts` could
   * bind a slice of one fused QKV/gate-up buffer as e.g. `rope`'s input, the
   * same way `llm/reshape.ts`'s CPU code slices a `Float32Array`; a
   * `createBindGroup` validation error at that offset — `does not satisfy
   * the minimum BufferBindingType::ReadOnlyStorage alignment (256)` —
   * is what caught it: the tiny fixture's own `kvDim = 32` floats puts `v`'s
   * slice at byte 384, not a multiple of 256, so the trick would have worked
   * by coincidence on some shapes and failed validation on others, silently
   * depending on `hiddenSize`/`headDim`/`numHeads` lining up. `llm/engine-q8-resident.ts`
   * gives every distinct tensor (Q, K, V, gate, up, …) its own buffer
   * instead — a few more `matvecQ8`/`activation` dispatches per layer than a
   * fused version would need, all bandwidth-bound and cheap next to the
   * weight traffic they move, and always valid regardless of shape.
   */
  bindGroup(pipeline: GPUComputePipeline, buffers: GPUBuffer[]): Promise<GPUBindGroup>;
  /**
   * A bind group over **slices** of buffers, for the one case whole buffers
   * cannot serve: a dispatch whose workgroup count is the row count and whose
   * row count is over WebGPU's 65,535 limit (issue #112). Splitting it means
   * binding a range, and a range is what `bindGroup` above deliberately does
   * not take.
   *
   * The caller owns the alignment. `minStorageBufferOffsetAlignment` is 256
   * bytes on the device this was measured on, and an offset that is not a
   * multiple of it fails validation — which is why `bindGroup` avoids offsets
   * by default and why this is a separate method rather than a wider signature.
   */
  bindGroupSliced(
    pipeline: GPUComputePipeline,
    slices: { buffer: GPUBuffer; offset: number; size: number }[],
  ): Promise<GPUBindGroup>;
  /**
   * Records every op into one `GPUCommandEncoder`, submits it exactly once,
   * then maps and reads back only `readback` — everything else recorded
   * (intermediate activations, the KV-cache copies) stays device-side.
   */
  batch(ops: ResidentOp[], readback: ResidentReadback[], profile?: BatchProfile): Promise<(Float32Array | Int32Array | Uint32Array)[]>;
  /**
   * Wait until the card actually has back what `destroy()` was called on.
   *
   * **`destroy()` schedules the freeing; it does not do it.** Dawn releases a
   * destroyed buffer when it next ticks, and it ticks on GPU work, not on a
   * timer. A stage that destroys 25 GB and immediately allocates 20 GB gets an
   * *invalid* buffer back, which does not throw. `harness/reclaim.ts` has the
   * measurement and the round-trip count. Issue #213.
   */
  reclaim(): Promise<void>;
  destroy(): void;
}

// --- kernel sources ---------------------------------------------------------

/** `{ [op]: { [entry]: wgslSource } }`, entry defaulting to `"kernel"` — one op directory's `wgsl/` folder, as strings. */
export type KernelSources = Readonly<Record<string, Readonly<Record<string, string>>>>;
/** A lazy alternative to a table: called once per `(op, entry)` the first time it is asked for. */
export type KernelResolver = (op: string, entry: string) => string | undefined;

let resolver: KernelResolver | null = null;
const resolved = new Map<string, string>();

/**
 * Tells `opKernel` where WGSL comes from.
 *
 * Node registers a file reader in `harness/index.ts` the moment it is
 * imported, so every test and verify script keeps working with no call of its
 * own. A browser (or any host that bundled its own copies of
 * `ops/<op>/wgsl/<entry>.wgsl`) calls this once with the table it holds — the
 * `LLM_KERNEL_SOURCES` / `DIT_KERNEL_SOURCES` lists say which entries an
 * engine will ask for.
 *
 * Registration replaces, never merges: two tables that disagree on an entry
 * would be a silent choice of one kernel over another.
 */
export function registerKernelSources(sources: KernelSources | KernelResolver): void {
  resolved.clear();
  resolver = typeof sources === "function" ? sources : (op, entry) => sources[op]?.[entry];
}

/** Forgets the registration — for tests that need to observe the unregistered state. */
export function resetKernelSources(): void {
  resolver = null;
  resolved.clear();
}

/**
 * What is registered right now, so a test can put it back.
 *
 * The registry is process-wide and Node registers on importing the harness
 * barrel; a test that resets it and walks away leaves every later file in the
 * same process unable to find a shader. Save, replace, restore.
 */
export function registeredKernelSources(): KernelResolver | null {
  return resolver;
}

/**
 * The WGSL for one op's entry point.
 *
 * Throws, naming the op and the entry, rather than returning `undefined`: an
 * `undefined` shader source is `"undefined"` by the time `createShaderModule`
 * sees it, and that compiles to an error message about a stray identifier
 * instead of about the table entry that was never filled in.
 */
export function opKernel(op: string, entry = "kernel"): string {
  if (!resolver) {
    throw new Error(
      `opKernel(${JSON.stringify(op)}, ${JSON.stringify(entry)}): no kernel sources are registered — ` +
        "call registerKernelSources() with the WGSL this host bundled (Node's harness/index.ts does it on import)",
    );
  }
  const key = `${op}/${entry}`;
  const hit = resolved.get(key);
  if (hit !== undefined) return hit;
  const source = resolver(op, entry);
  if (source === undefined) {
    throw new Error(`opKernel: no WGSL registered for op ${JSON.stringify(op)}, entry ${JSON.stringify(entry)}`);
  }
  resolved.set(key, source);
  return source;
}

/**
 * `opKernel` addressed the way `harness/suite.ts#kernel` is: by the URL of an
 * op's `index.ts`. The op name is the directory — the URL's path always ends
 * `/ops/<op>/index.ts` regardless of where the caller lives.
 */
export function kernelFromUrl(url: string | URL, name = "kernel"): string {
  const pathname = typeof url === "string" ? url : url.pathname;
  const match = /\/ops\/([^/]+)\/index\.ts$/.exec(pathname);
  if (!match) {
    throw new Error(`kernelFromUrl: expected a path ending ops/<op>/index.ts, got ${JSON.stringify(pathname)}`);
  }
  return opKernel(match[1]!, name);
}
