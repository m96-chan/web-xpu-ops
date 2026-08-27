import { create, globals } from "webgpu";
import { compilationFailure, type Binding, type Dispatch, type Runner } from "./wgsl.js";
import { reclaimByRoundTrips } from "./reclaim.js";
import { bindingTypeMismatch, kernelName, storageElementTypes, type ElementType } from "./binding-types.js";

/**
 * A lower-level counterpart to `harness/wgsl.ts#createRunner`, built for issue
 * #110: a caller that wants to build every buffer, pipeline and bind group
 * once, then run *many* dispatches — spread across many calls, not just one
 * dispatch list — behind a single `queue.submit`, reading back only the
 * bytes it actually asked for.
 *
 * `createRunner` cannot do this by design: every `run()` call allocates its
 * own bindings, submits its own command buffer and reads every `"out"`
 * binding back, because that is the right contract for a correctness harness
 * comparing one kernel against one reference at a time (rule 8). `LlamaEngineQ8`
 * (`llm/engine-q8.ts`) pays for that per token — issue #110's own measurement,
 * ~155 GPU↔CPU round trips per decode step for a model whose bottleneck is
 * weight bandwidth, not dispatch overhead.
 *
 * This module does not replace `createRunner` — every op's own `wgsl.test.ts`
 * keeps using it, unchanged, per rule 8 ("reference との突合が無い状態で最適化
 * しない"). `ResidentDevice` exists only for `llm/engine-q8-resident.ts`'s decode
 * path, where the buffers, pipelines and bind groups are already known correct
 * (built from the same WGSL, the same binding order, the same uniform layout
 * `llm/kernels.ts` already uses and `llm/kernels.wgsl.test.ts` already checks)
 * and the only thing left to fix is *how many times the CPU and GPU talk*.
 */

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

/** Instances kept reachable — see `wgsl.ts#retainedInstances` (#107). */
const retainedResidentInstances: unknown[] = [];

/** Resolves to null when no adapter is available, mirroring `wgsl.ts#createRunner`. */
export async function createResidentDevice(): Promise<ResidentDevice | null> {
  Object.assign(globalThis, globals);
  const gpu = create([]) as GPU;
  const adapter = await gpu.requestAdapter();
  // Kept reachable for the device's whole life — see `wgsl.ts#retainedInstances`
  // for the measurement and why this is not optional (#107).
  retainedResidentInstances.push(gpu, adapter);
  if (!adapter) return null;

  // Same reasoning as `wgsl.ts#createRunner`: request the adapter's own
  // ceiling rather than the spec's low defaults. A resident engine's weight
  // buffers (`lmHead` alone is ~175 MiB packed for Sarashina2.2-1B) need it.
  // The adapter's own ceiling — see `wgsl.ts` for what a fixed constant here
  // cost. `Math.min` below is what caps it to what the device actually offers.
  const wanted = Number.MAX_SAFE_INTEGER;
  // Issue #131: same feature-detection `wgsl.ts#createRunner` already does —
  // requested only when the adapter offers it, since asking for a feature an
  // adapter lacks fails `requestDevice` outright rather than degrading.
  const timestampsSupported = adapter.features.has("timestamp-query");
  const device = await adapter.requestDevice({
    requiredFeatures: timestampsSupported ? ["timestamp-query"] : [],
    requiredLimits: {
      maxStorageBufferBindingSize: Math.min(wanted, adapter.limits.maxStorageBufferBindingSize),
      maxBufferSize: Math.min(wanted, adapter.limits.maxBufferSize),
      // Same reason as the two above, and found the same way: a tiled matmul
      // wanting 17,408 B of workgroup storage was refused while Dawn's own
      // message said "This adapter supports a higher
      // maxComputeWorkgroupStorageSize of 49152, which can be specified in
      // requiredLimits". The spec default is 16,384 — enough for a 16x16 tile
      // and not for the register-tiled shapes #177 is about, so a sweep run
      // against the default measures only the shapes that were already there.
      maxComputeWorkgroupStorageSize: adapter.limits.maxComputeWorkgroupStorageSize,
      maxComputeInvocationsPerWorkgroup: adapter.limits.maxComputeInvocationsPerWorkgroup,
      // X only. Every kernel here declares a one-dimensional workgroup, so Y
      // and Z stay at their defaults rather than being raised for nothing.
      maxComputeWorkgroupSizeX: adapter.limits.maxComputeWorkgroupSizeX,
    },
  });

  const stats = { buffersCreated: 0, pipelinesCreated: 0, submits: 0, bindGroupMs: 0, bindGroups: 0 };
  /**
   * Pipelines, keyed by the **source string itself** and then by entry point.
   *
   * Not by `\`${entry} ${code}\``: a template literal builds a fresh
   * multi-kilobyte string on every call, and a fresh string has no cached hash,
   * so each lookup hashed the whole kernel. A caller passes the same string
   * *object* for a given kernel, so keying on it reuses the hash V8 already put
   * on it.
   *
   * **Measured at 0.1 µs a call either way**, so this is not a fix for
   * anything — it was changed while chasing a 502 ms attribution that turned
   * out to be a measurement error (a `performance.now()` pair straddling an
   * `await` charges whatever else the event loop runs to whatever is being
   * awaited). It stays because it is the cheaper shape and no longer allocates
   * per call, not because it made a forward faster.
   */
  const pipelines = new Map<string, Map<string, GPUComputePipeline>>();
  const modules = new Map<string, GPUShaderModule>();
  /**
   * A short name per pipeline, so a dispatch that fails validation can say
   * *which kernel* rather than only which index in the batch. Taken from the
   * WGSL's first `fn` — the source is the only identity a pipeline has here,
   * since `GPUComputePipeline` carries no label through this binding.
   */
  const pipelineNames = new WeakMap<GPUComputePipeline, string>();

  /**
   * Issue #221: the three things that have to meet for a binding's type to be
   * checkable, and which this API deliberately keeps apart.
   *
   * `pipelineFor` holds the WGSL, `upload` holds the TypedArray, and only
   * `bindGroup` knows which buffer is which binding of which pipeline. None of
   * the three can see the mismatch alone -- which is exactly why #217 survived
   * eighteen commits. `WeakMap` throughout so none of this keeps a device
   * object alive past its owner.
   */
  const declarations = new WeakMap<GPUComputePipeline, { kernel: string; types: (ElementType | null)[] }>();
  const uploaded = new WeakMap<GPUBuffer, ArrayBufferView>();

  /** Bytes handed out, so an allocation failure can say what was already in flight. */
  let allocated = 0;
  /** Bytes still held: `allocated` minus everything `destroy`ed. */
  let live = 0;
  /**
   * The first allocation that failed, kept until something can raise it.
   *
   * `createBuffer` is synchronous and its out-of-memory arrives through an
   * async error scope, so the failure is always known *after* the caller has
   * taken the buffer. Everything downstream of an invalid buffer is invalid
   * too, and reports itself rather than the cause.
   */
  let allocationFailure: string | null = null;
  /**
   * The error scopes still in flight.
   *
   * `popErrorScope` resolves on a later turn, and `createStorageBuffer` is
   * synchronous because every caller's inner loop is. So the failure is known
   * *after* the buffer has been handed out and after the dispatches that read
   * it have been recorded -- checking a flag at the bind group finds it still
   * unset, which is what the first version of this did. Draining here is the
   * one place that is both async and downstream of every allocation.
   */
  const pendingScopes: Promise<unknown>[] = [];
  /**
   * How many buffers of each size were created, so a failure can say where the
   * bytes went rather than only how many there are.
   *
   * "69 GB across 3,080 buffers" on a 32 GB card says the pool is not being
   * reused; it does not say *which* pool. The histogram does, and it costs a
   * map increment per allocation.
   */
  const sizeHistogram = new Map<number, number>();

  function createStorageBuffer(bytes: number, usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC): GPUBuffer {
    stats.buffersCreated += 1;
    const size = Math.max(4, bytes);
    // Checked at the allocation rather than at the bind: an out-of-memory
    // `createBuffer` returns an invalid buffer instead of throwing, and the
    // first thing to notice is `createBindGroup`, which reports a binding index
    // and no size. See the browser runtime's copy of this note.
    // **Both scopes.** An allocation can fail as out-of-memory *or* as
    // validation -- a size past `maxBufferSize` or
    // `maxStorageBufferBindingSize` is the second, and an out-of-memory scope
    // alone catches none of it. Issue #216 lost a run to exactly that: the OOM
    // scope was silent and the buffer was invalid anyway.
    device.pushErrorScope("validation");
    device.pushErrorScope("out-of-memory");
    const buffer = device.createBuffer({ size, usage });
    // **Remembered, not thrown.** This used to `throw` inside the `then`, which
    // is a rejection in a detached promise: nobody awaits it, the caller carries
    // on holding an invalid buffer, and the first thing that notices is a bind
    // group several dispatches later saying "[Invalid Buffer] is invalid due to
    // a previous error" with no size and no total. Issue #216 spent two runs on
    // that message. The scope catches the real cause; this keeps it until
    // something is in a position to raise it.
    const note = (kind: string) => (error: GPUError | null) => {
      if (!error || allocationFailure) return;
      allocationFailure = describeFailure(kind, error, size);
    };
    pendingScopes.push(device.popErrorScope().then(note("out of GPU memory")));
    pendingScopes.push(device.popErrorScope().then(note("invalid")));
    const describeFailure = (kind: string, error: GPUError, bytes: number): string =>
      `${kind}: a ${(bytes / 1e6).toFixed(0)} MB buffer was refused ` +
      `(${(live / 1e9).toFixed(2)} GB still held, ${(allocated / 1e9).toFixed(2)} GB handed out over the ` +
      `run across ${stats.buffersCreated} buffers; ` +
      `this device allows ${(device.limits.maxBufferSize / 1e9).toFixed(2)} GB a buffer and ` +
      `${(device.limits.maxStorageBufferBindingSize / 1e9).toFixed(2)} GB a storage binding).\n` +
      "  where the bytes went, largest first:\n" +
      [...sizeHistogram.entries()]
        .map(([bytes, count]) => ({ bytes, count, total: bytes * count }))
        .sort((x, y) => y.total - x.total)
        .slice(0, 8)
        .map((e) => `    ${(e.total / 1e9).toFixed(2)} GB — ${e.count} x ${(e.bytes / 1e6).toFixed(0)} MB`)
        .join("\n") +
      `\n  ${error.message}`;
    allocated += size;
    live += size;
    sizeHistogram.set(size, (sizeHistogram.get(size) ?? 0) + 1);
    // **`destroy` wrapped, so the number means something.** Without this
    // `allocated` is cumulative -- it counts the conditioner's 25 GB and the
    // encoder's, long after both were dropped -- and a failure message built on
    // it reads as "69 GB on a 32 GB card", which sounds like a leak and is not
    // one. Issue #216 misread exactly that. `live` is what is still held.
    const release = buffer.destroy.bind(buffer);
    let destroyed = false;
    buffer.destroy = () => {
      if (!destroyed) {
        destroyed = true;
        live -= size;
        sizeHistogram.set(size, (sizeHistogram.get(size) ?? 1) - 1);
      }
      release();
    };
    return buffer;
  }

  function createUniformBuffer(bytes: number): GPUBuffer {
    stats.buffersCreated += 1;
    return device.createBuffer({ size: Math.max(16, bytes), usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  }

  function upload(buffer: GPUBuffer, offset: number, data: ArrayBufferView): void {
    // Issue #221: remembered, not judged. Nothing here knows what this buffer
    // is *for* — `bindGroup` is the only place a buffer, a pipeline and a
    // binding number are in the same room, so that is where the comparison
    // happens and this only records the evidence.
    uploaded.set(buffer, data);
    // Same `as any` cast `wgsl.ts#dispatch` documents: `data` is always a
    // plain-`ArrayBuffer`-backed view here, never `SharedArrayBuffer`, but
    // `@webgpu/types` cannot express that without `DOM` lib.
    device.queue.writeBuffer(buffer, offset, data.buffer as any, data.byteOffset, data.byteLength);
  }

  /**
   * Does what was uploaded into each of these buffers agree with what this
   * pipeline declares at that binding? The message, or null.
   *
   * **Judged here and nowhere else, and the reason is buffer pooling.** The
   * first version also checked inside `upload`, against whatever binding the
   * buffer had been given last — and on Anima's real forward it reported
   * "RMSNorm: binding 0" for a buffer that was `ropeAxes`' `positions`. The
   * pool had handed the same buffer out for both. It caught the bug and named
   * the wrong kernel, and the same staleness would eventually reject a pooled
   * buffer legitimately used as `array<i32>` in one op and `array<f32>` in
   * another. A check that fails on correct code is worse than no check.
   *
   * At bind time there is no ambiguity: this pipeline, this index, this
   * buffer's most recent contents.
   *
   * **What that gives up**, stated rather than left to be discovered: a buffer
   * bound once and rewritten many times afterwards — `llm/engine-q8-resident.ts`
   * binds at construction and uploads per token — is checked on the upload
   * before its bind group and not on the ones after. The type a call site
   * uploads is fixed by its code rather than by the run, so the first pass
   * through that code is the one that matters, and that pass is covered.
   */
  function bindingMismatch(pipeline: GPUComputePipeline, buffers: readonly GPUBuffer[]): string | null {
    const declaration = declarations.get(pipeline);
    if (!declaration) return null;
    return bindingTypeMismatch(
      declaration.kernel,
      declaration.types,
      buffers.map((b) => uploaded.get(b) ?? null),
    );
  }

  async function pipelineFor(code: string, entry = "main"): Promise<GPUComputePipeline> {
    const byEntry = pipelines.get(code);
    const cached = byEntry?.get(entry);
    if (cached) return cached;
    let module = modules.get(code);
    if (!module) {
      module = device.createShaderModule({ code });
      const failure = compilationFailure((await module.getCompilationInfo()).messages);
      if (failure) throw new Error(failure);
      modules.set(code, module);
    }
    // Same reason `wgsl.ts#dispatch` (lines ~223-232) pushes an error scope
    // around its own `createComputePipeline`/`createBindGroup`: `layout:
    // "auto"` silently drops a binding the entry point never references
    // rather than failing pipeline creation, and the mismatch only surfaces
    // later as a bind-group validation error — or, without an error scope at
    // all, as a same-class-of-error output that just reads "zeros" (issue
    // #46's own failure mode). `bindGroup` below already has this scope;
    // `pipelineFor` did not (PR #116 review, item 4), so a wrong `entry`
    // string had no validation path here at all.
    device.pushErrorScope("validation");
    const pipeline = device.createComputePipeline({ layout: "auto", compute: { module, entryPoint: entry } });
    const invalid = await device.popErrorScope();
    if (invalid) throw new Error(`resident pipeline is not valid: ${invalid.message}`);
    // Recorded here rather than parsed at every bind group: `bindGroup` is
    // called thousands of times a forward and `pipelineFor` once per kernel.
    declarations.set(pipeline, { kernel: kernelName(code), types: storageElementTypes(code) });
    const entries = pipelines.get(code) ?? new Map<string, GPUComputePipeline>();
    entries.set(entry, pipeline);
    pipelines.set(code, entries);
    // The WGSL's own first line — every kernel in this repository opens with a
    // comment naming what it is, and the entry point is `main` in all of them,
    // so the source header is the only thing that tells them apart.
    const header = code.split("\n").find((line) => line.trim().length > 0)?.replace(/^\/\/\s*/, "").trim() ?? "";
    pipelineNames.set(pipeline, `${header.slice(0, 60)}${header.length > 60 ? "…" : ""} [${entry}]`);
    stats.pipelinesCreated += 1;
    return pipeline;
  }

  /** Waits out every allocation scope and raises the first failure. */
  async function drainAllocationScopes(): Promise<void> {
    if (pendingScopes.length > 0) {
      const waiting = pendingScopes.splice(0, pendingScopes.length);
      await Promise.all(waiting);
    }
    if (allocationFailure) throw new Error(allocationFailure);
  }

  async function bindGroupSliced(
    pipeline: GPUComputePipeline,
    slices: { buffer: GPUBuffer; offset: number; size: number }[],
  ): Promise<GPUBindGroup> {
    // The sliced path binds a *range* of a buffer, so what was uploaded into
    // the whole of it is not necessarily what this binding reads -- but the
    // element type is the same either way, which is the only thing checked.
    await drainAllocationScopes();
    const mismatch = bindingMismatch(pipeline, slices.map((slice) => slice.buffer));
    if (mismatch) throw new Error(mismatch);
    const t0 = performance.now();
    device.pushErrorScope("validation");
    const group = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: slices.map((slice, binding) => ({ binding, resource: slice })),
    });
    const invalid = await device.popErrorScope();
    if (invalid) throw new Error(`bind group is not valid: ${invalid.message}`);
    stats.bindGroupMs += performance.now() - t0;
    stats.bindGroups += 1;
    return group;
  }

  async function bindGroup(pipeline: GPUComputePipeline, buffers: GPUBuffer[]): Promise<GPUBindGroup> {
    await drainAllocationScopes();
    const mismatch = bindingMismatch(pipeline, buffers);
    if (mismatch) throw new Error(mismatch);
    const t0 = performance.now();
    device.pushErrorScope("validation");
    const group = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })),
    });
    const invalid = await device.popErrorScope();
    if (invalid) throw new Error(`resident bind group is not valid: ${invalid.message}`);
    stats.bindGroupMs += performance.now() - t0;
    stats.bindGroups += 1;
    return group;
  }

  async function batch(
    ops: ResidentOp[],
    readback: ResidentReadback[],
    profile?: BatchProfile,
  ): Promise<(Float32Array | Int32Array | Uint32Array)[]> {
    // Issue #131: which ops get their own timed pass, decided once up front
    // so the query set can be sized exactly (`GPUQuerySetDescriptor.count`
    // is fixed at creation) rather than grown as the loop below discovers
    // labels. `labels[i]` only matters for `ops[i].kind === "dispatch"` —
    // a label on a `"copy"` op is meaningless (copies never run inside a
    // pass) and silently ignored, not an error, since `BatchProfile.labels`
    // is positional against `ops` as a whole for the caller's convenience.
    const encodeT0 = profile ? performance.now() : 0;
    const wantsGpuTiming = !!(profile?.labels && timestampsSupported);
    const labeledSlots: string[] = [];
    if (wantsGpuTiming) {
      profile!.labels!.forEach((label, index) => {
        if (label != null && ops[index]?.kind === "dispatch") labeledSlots.push(label);
      });
    }
    const queryCount = labeledSlots.length * 2;
    const querySet = queryCount > 0 ? device.createQuerySet({ type: "timestamp", count: queryCount }) : null;

    const encoder = device.createCommandEncoder();
    let pass: GPUComputePassEncoder | null = null;
    const endPass = () => {
      if (pass) {
        pass.end();
        pass = null;
      }
    };
    let queryCursor = 0;
    for (const [i, op] of ops.entries()) {
      if (op.kind === "dispatch") {
        // **A dispatch past the device's grid limit is always a bug**, and
        // WebGPU reports it as an invalid *command buffer* — so every dispatch
        // recorded beside it is dropped too and the caller gets plausible
        // numbers computed from whatever the pool held. Issue #211's failures
        // all arrived that way. Named here, with the op's index, because by
        // the time `Submit` complains there is nothing left to point at.
        const ceiling = device.limits.maxComputeWorkgroupsPerDimension;
        for (const [axis, count] of op.workgroups.entries()) {
          if (count > ceiling) {
            throw new Error(
              `batch: op ${i} (${pipelineNames.get(op.pipeline) ?? "unnamed"}) dispatches ` +
                `${op.workgroups.join("x")} workgroups and this device allows ${ceiling} per dimension ` +
                `(axis ${axis} is ${count}) — see issue #211`,
            );
          }
        }
        const label = profile?.labels ? (profile.labels[i] ?? null) : null;
        // PR #141 review, item 3: gated on `wantsGpuTiming` (device negotiated
        // `timestamp-query` *and* the caller asked for a breakdown), not on
        // `profile?.labels` alone. An earlier version split every dispatch
        // into its own pass whenever `labels` was merely present, even on a
        // device that cannot serve `timestamp-query` at all — pure pass-
        // boundary overhead for zero GPU numbers, and (worse, on a device
        // that *can* serve timestamps) real overhead a caller who only wants
        // `ForwardProfile`'s CPU-side fields (`packMs`/`layerSetupMs`/
        // `submitToDoneMs`) never asked to pay — see `ForwardProfile.wantGpuBreakdown`'s
        // own doc for the caller-facing half of this fix.
        if (wantsGpuTiming) {
          // Profiling mode: every dispatch this batch records gets its own
          // pass — even an unlabeled one — so a labeled dispatch's own
          // timestamps never span work outside it. See `BatchProfile`'s own
          // doc for why this is real overhead confined to opt-in calls.
          endPass();
          const timeThis = label != null;
          pass = encoder.beginComputePass(
            timeThis ? { timestampWrites: { querySet: querySet!, beginningOfPassWriteIndex: queryCursor, endOfPassWriteIndex: queryCursor + 1 } } : undefined,
          );
          if (timeThis) queryCursor += 2;
        } else if (!pass) {
          pass = encoder.beginComputePass();
        }
        pass.setPipeline(op.pipeline);
        pass.setBindGroup(0, op.bindGroup);
        pass.dispatchWorkgroups(...(op.workgroups as [number, number?, number?]));
      } else {
        // `copyBufferToBuffer` cannot be recorded inside a compute pass —
        // close the current one, copy, and a later `dispatch` op reopens a
        // fresh pass. Still one encoder, still one submit.
        endPass();
        encoder.copyBufferToBuffer(op.src, op.srcOffset, op.dst, op.dstOffset, op.size);
      }
    }
    endPass();
    for (const r of readback) {
      encoder.copyBufferToBuffer(r.source, r.sourceOffset, r.staging, 0, r.length * 4);
    }
    let queryResolved: GPUBuffer | null = null;
    let queryReadable: GPUBuffer | null = null;
    if (querySet) {
      queryResolved = device.createBuffer({ size: queryCount * 8, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
      queryReadable = device.createBuffer({ size: queryCount * 8, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
      // PR #141 review, item 5: these two are real `GPUBuffer`s this call
      // allocates, same as any `createStorageBuffer`/`createUniformBuffer`
      // call — `stats.buffersCreated` is supposed to be the complete count a
      // test can snapshot to prove a loop allocates nothing (this file's own
      // `ResidentDevice.stats` doc), and creating them via a raw
      // `device.createBuffer(...)` here bypassed that count entirely.
      stats.buffersCreated += 2;
      encoder.resolveQuerySet(querySet, 0, queryCount, queryResolved, 0);
      encoder.copyBufferToBuffer(queryResolved, 0, queryReadable, 0, queryCount * 8);
    }

    try {
      // Before `submit`, so this is recording alone — `submitToDoneMs` below
      // starts where this stops and the two do not overlap.
      if (profile) profile.sink.encodeMs = performance.now() - encodeT0;
      device.queue.submit([encoder.finish()]);
      stats.submits += 1;

      // Issue #131 item 4's CPU half: how long the GPU took to actually
      // finish this submission, timed separately from the readback
      // `mapAsync` calls below (`onSubmittedWorkDone` resolves once the GPU
      // is done, before any buffer is mapped) — without this, `batch()`'s
      // only prior signal that work had finished was the first `mapAsync`
      // resolving, which folds GPU completion wait and the readback round
      // trip into one number.
      if (profile) {
        const t0 = performance.now();
        await device.queue.onSubmittedWorkDone();
        profile.sink.submitToDoneMs = performance.now() - t0;
      }

      const readbackT0 = profile ? performance.now() : 0;
      const results: (Float32Array | Int32Array | Uint32Array)[] = [];
      for (const r of readback) {
        await r.staging.mapAsync(GPUMapMode.READ);
        const bytes = r.staging.getMappedRange().slice(0);
        r.staging.unmap();
        results.push(r.type === "i32" ? new Int32Array(bytes) : r.type === "u32" ? new Uint32Array(bytes) : new Float32Array(bytes));
      }
      if (profile) profile.sink.readbackMs = performance.now() - readbackT0;

      if (querySet && queryReadable) {
        await queryReadable.mapAsync(GPUMapMode.READ);
        const stamps = new BigUint64Array(queryReadable.getMappedRange().slice(0));
        queryReadable.unmap();
        const entries: { label: string; seconds: number }[] = [];
        for (const [k, label] of labeledSlots.entries()) {
          // Timestamps are nanoseconds; a zero delta means the driver
          // declined to serve that particular query (`wgsl.ts#dispatch`'s
          // own doc) — not a real zero-duration dispatch, so it is left out
          // rather than reported as one.
          const elapsed = stamps[k * 2 + 1]! - stamps[k * 2]!;
          if (elapsed > 0n) entries.push({ label, seconds: Number(elapsed) / 1e9 });
        }
        profile!.sink.gpuEntries = entries;
      } else if (profile?.labels) {
        // Labels were requested but this device never negotiated
        // `timestamp-query` — `sink.gpuEntries` stays `[]`, distinguishable
        // from "measured, all zero" only via `timestampsSupported` (see that
        // field's own doc); nothing here claims a GPU number it does not
        // have.
        profile.sink.gpuEntries = [];
      }

      return results;
    } finally {
      // PR #141 review, item 6: unconditional, not only on the success path
      // an earlier version took — a `mapAsync` rejection (or anything else
      // thrown between `submit` and here) used to leave the query set and
      // its two staging buffers alive with nothing left to ever destroy
      // them, the same class of leak `runnerFromResident`'s own `finally`
      // block exists to avoid.
      querySet?.destroy();
      queryResolved?.destroy();
      queryReadable?.destroy();
    }
  }

  return {
    stats,
    timestampsSupported,
    createStorageBuffer,
    createUniformBuffer,
    upload,
    pipelineFor,
    bindGroup,
    bindGroupSliced,
    batch,
    reclaim: () => reclaimByRoundTrips(batch, createStorageBuffer),
    destroy() {
      device.destroy();
    },
  };
}


/**
 * A `Runner["run"]` built on top of an already-created `ResidentDevice`,
 * for a caller that needs the `wgsl.ts#createRunner` calling convention
 * (per-dispatch buffers, every `"out"` binding read back) without a second
 * native `webgpu` device.
 *
 * `LlamaEngineQ8Resident` (issue #110) needs exactly this: its prefill path
 * delegates whole to `LlamaEngineQ8`, which is written against `Runner["run"]`
 * ("プリフィルは現行方式のまま" — #110's own stated scope, not something this
 * class re-derives). The first version of that delegation took a *second*,
 * separately-constructed `Runner` (`createRunner()`'s own device) alongside
 * the `ResidentDevice` — two native `GPUDevice`s in one process — and it
 * reproducibly crashed this repository's Node/Dawn binding before prefill
 * even finished (`terminate called after throwing an instance of
 * 'std::system_error'`, the same failure family issue #38/#49/#107 already
 * document), not on every run and not always at the same call, which is
 * itself the signature of this binding's known instability under load
 * rather than a logic bug — a single `ResidentDevice`, wrapped as a `Runner`
 * for prefill and used directly for decode, removed the second device and
 * the crashes with it (`llm/engine-q8-resident.wgsl.test.ts` is the
 * regression coverage: green, repeatedly, where the two-device version was
 * not).
 *
 * Deliberately not resident itself: every call allocates fresh buffers and
 * reads every `"out"` binding back, exactly like `createRunner()`'s own
 * `run()` — correct for `LlamaEngineQ8`'s per-dispatch-await forward pass,
 * and irrelevant to the "no `create*` in the decode loop" property that
 * matters only for `LlamaEngineQ8Resident`'s own decode path, which never
 * calls this function.
 */
export function runnerFromResident(device: ResidentDevice): Runner["run"] {
  return async function run(dispatch: Dispatch) {
    const { code, entry = "main", bindings, workgroups } = dispatch;
    const pipeline = await device.pipelineFor(code, entry);

    const buffers: GPUBuffer[] = [];
    const outputs: { spec: Extract<Binding, { kind: "out" }>; buffer: GPUBuffer }[] = [];

    for (const binding of bindings) {
      if (binding.kind === "out") {
        const buffer = device.createStorageBuffer(binding.length * 4);
        buffers.push(buffer);
        outputs.push({ spec: binding, buffer });
      } else if (binding.kind === "scratch") {
        buffers.push(device.createStorageBuffer(Math.max(4, binding.length * 4)));
      } else if (binding.kind === "uniform") {
        const buffer = device.createUniformBuffer(binding.data.byteLength);
        device.upload(buffer, 0, new Uint8Array(binding.data));
        buffers.push(buffer);
      } else {
        const buffer = device.createStorageBuffer(binding.data.byteLength);
        device.upload(buffer, 0, binding.data);
        buffers.push(buffer);
      }
    }

    const bindGroup = await device.bindGroup(pipeline, buffers);
    const stagingBuffers = outputs.map(({ spec }) => device.createStorageBuffer(spec.length * 4, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ));
    const readback = outputs.map(({ spec, buffer }, index) => ({
      staging: stagingBuffers[index]!,
      source: buffer,
      sourceOffset: 0,
      length: spec.length,
      type: spec.type,
    }));

    try {
      return await device.batch([{ kind: "dispatch", pipeline, bindGroup, workgroups }], readback);
    } finally {
      // Every buffer this call created — inputs, outputs and staging — is
      // ephemeral (see this function's doc: unlike `LlamaEngineQ8Resident`'s
      // own buffers, nothing here outlives one dispatch). `wgsl.ts#dispatch`
      // destroys its own per-call buffers the same way; this function was
      // missing that step in an earlier version, and the leak was the actual
      // cause of what first looked like a Node/Dawn crash under
      // `llm/engine-q8-resident.wgsl.test.ts`'s full prefill-plus-six-decode-steps
      // run (~200 dispatches through this function, each leaking a handful of
      // buffers that `device.destroy()` only frees at the very end of the
      // test file) — measured by adding this `finally` block and watching the
      // same run go from hanging/crashing to consistently finishing in well
      // under a second.
      for (const buffer of buffers) buffer.destroy();
      for (const { buffer } of outputs) buffer.destroy();
      for (const staging of stagingBuffers) staging.destroy();
    }
  };
}
