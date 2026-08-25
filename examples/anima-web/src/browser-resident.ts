/**
 * `harness/resident.ts`'s `ResidentDevice`, over `navigator.gpu`.
 *
 * The Node version imports the `webgpu` package at module scope — a native
 * Dawn addon with no browser build — so it cannot be bundled. Everything below
 * it is plain WebGPU, and this is that part ported, the same way
 * `examples/llm-demo/src/browser-runtime.ts` is the browser half of
 * `harness/wgsl.ts`.
 *
 * Two copies exist because the two run on opposite sides of exactly the module
 * boundary `webgpu` draws, not because anything here is demo-specific. If they
 * drift, `dit-resident.ts` produces different numbers in the browser than the
 * verifier measures in Node — which is visible, since both are held to the same
 * golden.
 *
 * The profiling half is ported too. It was left out at first — "nothing here
 * asks for it" — and then the page grew a profile checkbox wired to a device
 * that hardcoded `timestampsSupported: false` and dropped the argument on the
 * floor. The checkbox did nothing and said nothing, which is the failure mode
 * the original comment was trying to avoid and reached anyway by being the
 * only half that was implemented.
 */
import type {
  BatchProfile, ResidentDevice, ResidentOp, ResidentReadback,
} from "../../../harness/resident.js";

export async function createBrowserResidentDevice(): Promise<ResidentDevice> {
  const gpu = (navigator as Navigator & { gpu?: GPU }).gpu;
  if (!gpu) throw new Error("createBrowserResidentDevice: navigator.gpu is unavailable — WebGPU is not enabled");
  const adapter = await gpu.requestAdapter();
  if (!adapter) throw new Error("createBrowserResidentDevice: requestAdapter() returned null");

  // Optional, and asked for only when the adapter has it: a `requiredFeatures`
  // entry the adapter cannot serve makes `requestDevice` reject outright, so a
  // device without timestamps has to come back working and unprofilable rather
  // than not come back at all.
  const timestampsSupported = adapter.features.has("timestamp-query");

  // The adapter's own ceiling, for the reason both harnesses give: the spec's
  // defaults are far below what a real checkpoint's projections need, and a
  // fixed number would be a guess about this particular device.
  const device = await adapter.requestDevice({
    requiredFeatures: timestampsSupported ? ["timestamp-query"] : [],
    requiredLimits: {
      maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
      maxBufferSize: adapter.limits.maxBufferSize,
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
  const pipelines = new Map<string, GPUComputePipeline>();

  /**
   * Bytes handed out, so an allocation failure can say how much was already in
   * flight rather than only that one more did not fit.
   */
  let allocated = 0;

  function createStorageBuffer(
    bytes: number,
    usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
  ): GPUBuffer {
    stats.buffersCreated += 1;
    const size = Math.max(4, bytes);
    // **Checked here, not at the bind.** A `createBuffer` that runs out of
    // device memory returns an *invalid* buffer rather than throwing, and the
    // first thing that notices is `createBindGroup` several frames later —
    // reporting `entries[1] is invalid due to a previous error`, which names
    // the binding and not the allocation. This turns that into a message with
    // a size in it.
    device.pushErrorScope("out-of-memory");
    const buffer = device.createBuffer({ size, usage });
    void device.popErrorScope().then((error) => {
      if (!error) return;
      throw new Error(
        `out of GPU memory allocating ${(size / 1e6).toFixed(0)} MB ` +
          `(${(allocated / 1e9).toFixed(2)} GB already allocated by this device, ` +
          `${stats.buffersCreated} buffers). ${error.message}`,
      );
    });
    allocated += size;
    return buffer;
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

    const module = device.createShaderModule({ code });
    const info = await module.getCompilationInfo();
    const errors = info.messages.filter((m) => m.type === "error");
    if (errors.length > 0) {
      throw new Error(`shader compilation failed: ${errors.map((m) => `${m.lineNum}:${m.linePos} ${m.message}`).join("; ")}`);
    }

    // The error scope is what turns a silently invalid pipeline into an
    // exception — `harness/wgsl.ts`'s note on why applies here unchanged.
    device.pushErrorScope("validation");
    const pipeline = device.createComputePipeline({ layout: "auto", compute: { module, entryPoint: entry } });
    const invalid = await device.popErrorScope();
    if (invalid) throw new Error(`pipeline is not valid: ${invalid.message}`);

    stats.pipelinesCreated += 1;
    pipelines.set(key, pipeline);
    return pipeline;
  }

  async function bindGroup(pipeline: GPUComputePipeline, buffers: GPUBuffer[]): Promise<GPUBindGroup> {
    const t0 = performance.now();
    device.pushErrorScope("validation");
    const group = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })),
    });
    const invalid = await device.popErrorScope();
    if (invalid) throw new Error(`bind group is not valid: ${invalid.message}`);
    stats.bindGroupMs += performance.now() - t0;
    stats.bindGroups += 1;
    return group;
  }

  /** `bindGroup` over ranges — see `harness/resident.ts`'s note on the alignment. */
  async function bindGroupSliced(
    pipeline: GPUComputePipeline,
    slices: { buffer: GPUBuffer; offset: number; size: number }[],
  ): Promise<GPUBindGroup> {
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

  async function batch(
    ops: ResidentOp[],
    readback: ResidentReadback[],
    profile?: BatchProfile,
  ): Promise<(Float32Array | Int32Array | Uint32Array)[]> {
    // `GPUComputePassTimestampWrites` allows one pair per pass, so a timed
    // dispatch needs a pass of its own — and so does an *untimed* one next to
    // it, or a timed pass would span work outside the dispatch it names. That
    // is why this is opt-in: it turns one pass per batch into one per dispatch.
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
    const endPass = (): void => {
      if (pass) {
        pass.end();
        pass = null;
      }
    };

    let queryCursor = 0;
    for (const [i, op] of ops.entries()) {
      if (op.kind === "dispatch") {
        const label = profile?.labels ? (profile.labels[i] ?? null) : null;
        if (wantsGpuTiming) {
          endPass();
          const timeThis = label != null;
          pass = encoder.beginComputePass(
            timeThis
              ? {
                  timestampWrites: {
                    querySet: querySet!,
                    beginningOfPassWriteIndex: queryCursor,
                    endOfPassWriteIndex: queryCursor + 1,
                  },
                }
              : undefined,
          );
          if (timeThis) queryCursor += 2;
        } else if (!pass) {
          pass = encoder.beginComputePass();
        }
        pass.setPipeline(op.pipeline);
        pass.setBindGroup(0, op.bindGroup);
        pass.dispatchWorkgroups(...(op.workgroups as [number, number?, number?]));
      } else {
        // A copy cannot be recorded inside a compute pass, so the pass ends
        // and the next dispatch opens a new one. That ordering is the reason
        // `dit-resident.ts` can use `copy` to concatenate tensors at all.
        endPass();
        encoder.copyBufferToBuffer(op.src, op.srcOffset, op.dst, op.dstOffset, op.size);
      }
    }
    endPass();

    for (const r of readback) {
      encoder.copyBufferToBuffer(r.source, r.sourceOffset, r.staging, 0, r.length * 4);
    }

    let queryReadable: GPUBuffer | null = null;
    if (querySet) {
      const resolved = device.createBuffer({
        size: queryCount * 8,
        usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
      });
      queryReadable = device.createBuffer({
        size: queryCount * 8,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
      // Counted, because `stats.buffersCreated` is meant to be the complete
      // number a test can snapshot to prove a loop allocates nothing.
      stats.buffersCreated += 2;
      encoder.resolveQuerySet(querySet, 0, queryCount, resolved, 0);
      encoder.copyBufferToBuffer(resolved, 0, queryReadable, 0, queryCount * 8);
    }

    // Before `submit`, so recording is timed alone and does not overlap the
    // wait below. Issue #182.
    if (profile) profile.sink.encodeMs = performance.now() - encodeT0;
    device.pushErrorScope("validation");
    device.queue.submit([encoder.finish()]);
    stats.submits += 1;
    const invalid = await device.popErrorScope();
    if (invalid) throw new Error(`batch is not valid: ${invalid.message}`);

    const submitT0 = profile ? performance.now() : 0;
    await device.queue.onSubmittedWorkDone();
    if (profile) profile.sink.submitToDoneMs = performance.now() - submitT0;

    const readbackT0 = profile ? performance.now() : 0;
    const results: (Float32Array | Int32Array | Uint32Array)[] = [];
    for (const r of readback) {
      await r.staging.mapAsync(GPUMapMode.READ);
      const bytes = r.staging.getMappedRange().slice(0);
      r.staging.unmap();
      results.push(
        r.type === "i32" ? new Int32Array(bytes) : r.type === "u32" ? new Uint32Array(bytes) : new Float32Array(bytes),
      );
    }
    if (profile) profile.sink.readbackMs = performance.now() - readbackT0;

    if (querySet && queryReadable) {
      await queryReadable.mapAsync(GPUMapMode.READ);
      const stamps = new BigUint64Array(queryReadable.getMappedRange().slice(0));
      queryReadable.unmap();
      const entries: { label: string; seconds: number }[] = [];
      for (const [k, label] of labeledSlots.entries()) {
        // Nanoseconds. A zero delta means the driver declined that particular
        // query rather than a dispatch that really took no time, so it is left
        // out instead of reported as a duration.
        const elapsed = stamps[k * 2 + 1]! - stamps[k * 2]!;
        if (elapsed > 0n) entries.push({ label, seconds: Number(elapsed) / 1e9 });
      }
      profile!.sink.gpuEntries = entries;
      querySet.destroy();
      queryReadable.destroy();
    }
    return results;
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
    destroy: () => device.destroy(),
  };
}
