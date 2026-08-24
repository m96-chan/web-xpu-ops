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
 * The profiling half of the Node interface (`BatchProfile`, timestamp queries)
 * is left out: nothing here asks for it, and a stub that silently reports
 * nothing would be worse than its absence.
 */
import type { ResidentDevice, ResidentOp, ResidentReadback } from "../../../harness/resident.js";

export async function createBrowserResidentDevice(): Promise<ResidentDevice> {
  const gpu = (navigator as Navigator & { gpu?: GPU }).gpu;
  if (!gpu) throw new Error("createBrowserResidentDevice: navigator.gpu is unavailable — WebGPU is not enabled");
  const adapter = await gpu.requestAdapter();
  if (!adapter) throw new Error("createBrowserResidentDevice: requestAdapter() returned null");

  // The adapter's own ceiling, for the reason both harnesses give: the spec's
  // defaults are far below what a real checkpoint's projections need, and a
  // fixed number would be a guess about this particular device.
  const device = await adapter.requestDevice({
    requiredLimits: {
      maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
      maxBufferSize: adapter.limits.maxBufferSize,
    },
  });

  const stats = { buffersCreated: 0, pipelinesCreated: 0, submits: 0 };
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
    device.pushErrorScope("validation");
    const group = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })),
    });
    const invalid = await device.popErrorScope();
    if (invalid) throw new Error(`bind group is not valid: ${invalid.message}`);
    return group;
  }

  /** `bindGroup` over ranges — see `harness/resident.ts`'s note on the alignment. */
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
  ): Promise<(Float32Array | Int32Array | Uint32Array)[]> {
    const encoder = device.createCommandEncoder();
    let pass: GPUComputePassEncoder | null = null;
    const endPass = (): void => {
      if (pass) {
        pass.end();
        pass = null;
      }
    };

    for (const op of ops) {
      if (op.kind === "dispatch") {
        if (!pass) pass = encoder.beginComputePass();
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

    device.pushErrorScope("validation");
    device.queue.submit([encoder.finish()]);
    stats.submits += 1;
    const invalid = await device.popErrorScope();
    if (invalid) throw new Error(`batch is not valid: ${invalid.message}`);

    await device.queue.onSubmittedWorkDone();

    const results: (Float32Array | Int32Array | Uint32Array)[] = [];
    for (const r of readback) {
      await r.staging.mapAsync(GPUMapMode.READ);
      const bytes = r.staging.getMappedRange().slice(0);
      r.staging.unmap();
      results.push(
        r.type === "i32" ? new Int32Array(bytes) : r.type === "u32" ? new Uint32Array(bytes) : new Float32Array(bytes),
      );
    }
    return results;
  }

  return {
    stats,
    timestampsSupported: false,
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
