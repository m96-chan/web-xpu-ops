/**
 * This demo's `navigator.gpu` counterpart to `harness/resident.ts` — same
 * contract (`ResidentDevice`: build every buffer/pipeline/bind group once,
 * then run many dispatches behind one `queue.submit`, reading back only
 * what was asked for), ported the same way `browser-runtime.ts` already
 * ports `harness/wgsl.ts#createRunner`: a deliberate, small duplication
 * across the exact module boundary the `webgpu` package (Node-native) draws
 * against `navigator.gpu` (browser-native) — see `browser-runtime.ts`'s own
 * module doc for why two copies exist rather than one shared implementation.
 *
 * `llm/engine-q8-resident.ts` imports only the **type** `ResidentDevice`
 * from `harness/index.js` (`import { ..., type ResidentDevice, ... }`) —
 * erased before `build.mjs`'s bundling ever resolves the value import, the
 * same reasoning that file's own module doc gives for `type Runner` — so
 * this file's `createBrowserResidentDevice()` never needs to be routed
 * through `harnessBrowserShim`; `main.ts` calls it directly and hands the
 * result to `LlamaEngineQ8Resident.create()`, which only cares that the
 * object it receives satisfies the structural `ResidentDevice` shape.
 *
 * `ResidentOp`/`ResidentReadback`/`ResidentDevice` themselves are imported
 * type-only from the real `harness/resident.ts` (below), the same
 * type-only-import pattern `browser-runtime.ts` already established for
 * `Binding`/`Dispatch` from `harness/wgsl.ts` — not hand-redeclared, as an
 * earlier version of this file did. That earlier version had already drifted
 * once from the real thing (`ResidentOp` split into two interfaces,
 * `ResidentOp`/`ResidentCopyOp`, where `harness/resident.ts` has always kept
 * `ResidentOp` as one discriminated union — PR #116 review, item 6):
 * `examples/llm-demo/tsconfig.json` only checks `src/`, so a mismatch here
 * is invisible to `tsc` and would only ever surface as an esbuild bundling
 * failure or, worse, a silent structural near-match. Importing the types
 * instead of restating them makes that class of drift impossible rather
 * than merely checked.
 */
import type { BatchProfile, ResidentDevice, ResidentOp, ResidentReadback } from "../../../harness/resident.js";

export type { BatchProfile, ResidentDevice, ResidentOp, ResidentReadback };

/** `harness/resident.ts#createResidentDevice`'s `navigator.gpu` counterpart — see that file's doc for what every piece below is for; this is a direct port, not a redesign. */
export async function createBrowserResidentDevice(): Promise<ResidentDevice> {
  const gpu = (navigator as Navigator & { gpu?: GPU }).gpu;
  if (!gpu) throw new Error("createBrowserResidentDevice: navigator.gpu is unavailable — WebGPU is not enabled in this browser");
  const adapter = await gpu.requestAdapter();
  if (!adapter) throw new Error("createBrowserResidentDevice: requestAdapter() returned null — no WebGPU adapter on this machine");

  // Same reasoning as `browser-runtime.ts#createBrowserRunner`: request the
  // adapter's own ceiling. A resident engine's weight buffers (`lmHead`
  // alone is ~175 MiB packed for Sarashina2.2-1B) need it.
  // Issue #131: same feature-detection `harness/resident.ts` (this file's
  // Node counterpart) does — requested only when the adapter offers it.
  const timestampsSupported = adapter.features.has("timestamp-query");
  const device = await adapter.requestDevice({
    requiredFeatures: timestampsSupported ? ["timestamp-query"] : [],
    requiredLimits: {
      maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
      maxBufferSize: adapter.limits.maxBufferSize,
      // The kernels in `ops/` need more than the spec defaults: the tiled
      // matmul is 512 invocations wide and stages 12 KB, against defaults of
      // 256 and 16384. Asked for here because a device that was not asked
      // refuses the pipeline outright — which is how this file was found, from
      // a browser console rather than from a test.
      maxComputeWorkgroupStorageSize: adapter.limits.maxComputeWorkgroupStorageSize,
      maxComputeInvocationsPerWorkgroup: adapter.limits.maxComputeInvocationsPerWorkgroup,
      maxComputeWorkgroupSizeX: adapter.limits.maxComputeWorkgroupSizeX,
    },
  });

  const stats = { buffersCreated: 0, pipelinesCreated: 0, submits: 0 };
  const pipelines = new Map<string, GPUComputePipeline>();
  const modules = new Map<string, GPUShaderModule>();

  function createStorageBuffer(bytes: number, usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC): GPUBuffer {
    stats.buffersCreated += 1;
    return device.createBuffer({ size: Math.max(4, bytes), usage });
  }

  function createUniformBuffer(bytes: number): GPUBuffer {
    stats.buffersCreated += 1;
    return device.createBuffer({ size: Math.max(16, bytes), usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  }

  function upload(buffer: GPUBuffer, offset: number, data: ArrayBufferView): void {
    device.queue.writeBuffer(buffer, offset, data.buffer as ArrayBuffer, data.byteOffset, data.byteLength);
  }

  async function pipelineFor(code: string, entry = "main"): Promise<GPUComputePipeline> {
    const key = `${entry} ${code}`;
    const cached = pipelines.get(key);
    if (cached) return cached;
    let module = modules.get(code);
    if (!module) {
      module = device.createShaderModule({ code });
      const info = await module.getCompilationInfo();
      const errors = info.messages.filter((m) => m.type === "error");
      if (errors.length > 0) {
        const where = (m: GPUCompilationMessage) => `${m.lineNum}:${m.linePos}: ${m.message}`;
        throw new Error(`shader failed to compile\n${errors.map(where).join("\n")}`);
      }
      modules.set(code, module);
    }
    // Same reason `bindGroup` below pushes an error scope, and the same fix
    // `harness/resident.ts#pipelineFor` (this file's Node counterpart) needed
    // (PR #116 review, item 4): `layout: "auto"` silently drops a binding an
    // entry point never references rather than failing pipeline creation, so
    // without this, a wrong `entry` string had no validation path here at
    // all.
    device.pushErrorScope("validation");
    const pipeline = device.createComputePipeline({ layout: "auto", compute: { module, entryPoint: entry } });
    const invalid = await device.popErrorScope();
    if (invalid) throw new Error(`resident pipeline is not valid: ${invalid.message}`);
    pipelines.set(key, pipeline);
    stats.pipelinesCreated += 1;
    return pipeline;
  }

  async function bindGroup(pipeline: GPUComputePipeline, buffers: GPUBuffer[]): Promise<GPUBindGroup> {
    device.pushErrorScope("validation");
    const group = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })),
    });
    const invalid = await device.popErrorScope();
    if (invalid) throw new Error(`resident bind group is not valid: ${invalid.message}`);
    return group;
  }

  /**
   * `bindGroup` over ranges.
   *
   * Added with `harness/resident.ts`'s own `bindGroupSliced` so this file still
   * satisfies `ResidentDevice`. Nothing in this demo calls it — the caller that
   * needs it is `examples/zimage`'s DiT, splitting a QK-Norm whose row count
   * exceeds WebGPU's 65,535 workgroup limit — but leaving it off would make the
   * type a lie about what a `ResidentDevice` provides.
   */
  async function bindGroupSliced(
    pipeline: GPUComputePipeline,
    slices: { buffer: GPUBuffer; offset: number; size: number }[],
  ): Promise<GPUBindGroup> {
    device.pushErrorScope("validation");
    const group = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: slices.map((slice, binding) => ({ binding, resource: slice })),
    });
    const invalid = await device.popErrorScope();
    if (invalid) throw new Error(`bind group is not valid: ${invalid.message}`);
    return group;
  }

  async function batch(
    ops: ResidentOp[],
    readback: ResidentReadback[],
    profile?: BatchProfile,
  ): Promise<(Float32Array | Int32Array | Uint32Array)[]> {
    // Issue #131 — see `harness/resident.ts#batch`'s own doc for the full
    // reasoning; this is a direct port, unchanged in structure.
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
        const label = profile?.labels ? (profile.labels[i] ?? null) : null;
        // PR #141 review, item 3 — gated on `wantsGpuTiming`, not merely
        // `profile?.labels`; see `harness/resident.ts#batch`'s own comment.
        if (wantsGpuTiming) {
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
        endPass();
        encoder.copyBufferToBuffer(op.src, op.srcOffset, op.dst, op.dstOffset, op.size);
      }
    }
    endPass();
    for (const r of readback) encoder.copyBufferToBuffer(r.source, r.sourceOffset, r.staging, 0, r.length * 4);
    let queryResolved: GPUBuffer | null = null;
    let queryReadable: GPUBuffer | null = null;
    if (querySet) {
      queryResolved = device.createBuffer({ size: queryCount * 8, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
      queryReadable = device.createBuffer({ size: queryCount * 8, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
      // PR #141 review, item 5 — see `harness/resident.ts#batch`'s own doc.
      stats.buffersCreated += 2;
      encoder.resolveQuerySet(querySet, 0, queryCount, queryResolved, 0);
      encoder.copyBufferToBuffer(queryResolved, 0, queryReadable, 0, queryCount * 8);
    }

    try {
      device.queue.submit([encoder.finish()]);
      stats.submits += 1;

      if (profile) {
        const t0 = performance.now();
        await device.queue.onSubmittedWorkDone();
        profile.sink.submitToDoneMs = performance.now() - t0;
      }

      const readbackT0 = profile ? performance.now() : 0;
      const results: (Float32Array | Int32Array | Uint32Array)[] = [];
      for (const r of readback) {
        // eslint-disable-next-line no-await-in-loop
        await r.staging.mapAsync(GPUMapMode.READ);
        const bytes = r.staging.getMappedRange().slice(0);
        r.staging.unmap();
        results.push(r.type === "i32" ? new Int32Array(bytes) : r.type === "u32" ? new Uint32Array(bytes) : new Float32Array(bytes));
      }
      if (profile) profile.sink.readbackMs = performance.now() - readbackT0;

      if (querySet && queryReadable) {
        // eslint-disable-next-line no-await-in-loop
        await queryReadable.mapAsync(GPUMapMode.READ);
        const stamps = new BigUint64Array(queryReadable.getMappedRange().slice(0));
        queryReadable.unmap();
        const entries: { label: string; seconds: number }[] = [];
        for (const [k, label] of labeledSlots.entries()) {
          const elapsed = stamps[k * 2 + 1]! - stamps[k * 2]!;
          if (elapsed > 0n) entries.push({ label, seconds: Number(elapsed) / 1e9 });
        }
        profile!.sink.gpuEntries = entries;
      } else if (profile?.labels) {
        profile.sink.gpuEntries = [];
      }

      return results;
    } finally {
      // PR #141 review, item 6 — see `harness/resident.ts#batch`'s own doc.
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
    destroy() {
      device.destroy();
    },
  };
}
