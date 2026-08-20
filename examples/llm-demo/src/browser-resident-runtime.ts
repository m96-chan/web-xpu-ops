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
 */

export interface ResidentOp {
  kind: "dispatch";
  pipeline: GPUComputePipeline;
  bindGroup: GPUBindGroup;
  workgroups: [number] | [number, number] | [number, number, number];
}

export interface ResidentCopyOp {
  kind: "copy";
  src: GPUBuffer;
  srcOffset: number;
  dst: GPUBuffer;
  dstOffset: number;
  size: number;
}

export interface ResidentReadback {
  staging: GPUBuffer;
  source: GPUBuffer;
  sourceOffset: number;
  length: number;
  type: "f32" | "i32" | "u32";
}

export interface ResidentDevice {
  readonly stats: { buffersCreated: number; pipelinesCreated: number; submits: number };
  createStorageBuffer(bytes: number, usage?: number): GPUBuffer;
  createUniformBuffer(bytes: number): GPUBuffer;
  upload(buffer: GPUBuffer, offset: number, data: ArrayBufferView): void;
  pipelineFor(code: string, entry?: string): Promise<GPUComputePipeline>;
  bindGroup(pipeline: GPUComputePipeline, buffers: GPUBuffer[]): Promise<GPUBindGroup>;
  batch(ops: (ResidentOp | ResidentCopyOp)[], readback: ResidentReadback[]): Promise<(Float32Array | Int32Array | Uint32Array)[]>;
  destroy(): void;
}

/** `harness/resident.ts#createResidentDevice`'s `navigator.gpu` counterpart — see that file's doc for what every piece below is for; this is a direct port, not a redesign. */
export async function createBrowserResidentDevice(): Promise<ResidentDevice> {
  const gpu = (navigator as Navigator & { gpu?: GPU }).gpu;
  if (!gpu) throw new Error("createBrowserResidentDevice: navigator.gpu is unavailable — WebGPU is not enabled in this browser");
  const adapter = await gpu.requestAdapter();
  if (!adapter) throw new Error("createBrowserResidentDevice: requestAdapter() returned null — no WebGPU adapter on this machine");

  // Same reasoning as `browser-runtime.ts#createBrowserRunner`: request the
  // adapter's own ceiling. A resident engine's weight buffers (`lmHead`
  // alone is ~175 MiB packed for Sarashina2.2-1B) need it.
  const device = await adapter.requestDevice({
    requiredLimits: {
      maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
      maxBufferSize: adapter.limits.maxBufferSize,
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
    const pipeline = device.createComputePipeline({ layout: "auto", compute: { module, entryPoint: entry } });
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

  async function batch(
    ops: (ResidentOp | ResidentCopyOp)[],
    readback: ResidentReadback[],
  ): Promise<(Float32Array | Int32Array | Uint32Array)[]> {
    const encoder = device.createCommandEncoder();
    let pass: GPUComputePassEncoder | null = null;
    const endPass = () => {
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
        endPass();
        encoder.copyBufferToBuffer(op.src, op.srcOffset, op.dst, op.dstOffset, op.size);
      }
    }
    endPass();
    for (const r of readback) encoder.copyBufferToBuffer(r.source, r.sourceOffset, r.staging, 0, r.length * 4);
    device.queue.submit([encoder.finish()]);
    stats.submits += 1;

    const results: (Float32Array | Int32Array | Uint32Array)[] = [];
    for (const r of readback) {
      // eslint-disable-next-line no-await-in-loop
      await r.staging.mapAsync(GPUMapMode.READ);
      const bytes = r.staging.getMappedRange().slice(0);
      r.staging.unmap();
      results.push(r.type === "i32" ? new Int32Array(bytes) : r.type === "u32" ? new Uint32Array(bytes) : new Float32Array(bytes));
    }
    return results;
  }

  return {
    stats,
    createStorageBuffer,
    createUniformBuffer,
    upload,
    pipelineFor,
    bindGroup,
    batch,
    destroy() {
      device.destroy();
    },
  };
}

/**
 * `harness/resident.ts#runnerFromResident`'s browser counterpart — see that
 * function's doc for why `LlamaEngineQ8Resident`'s prefill delegate needs
 * one of these rather than a second device, and why every buffer it creates
 * must be destroyed after the one dispatch it served (the leak that first
 * looked like Node/Dawn instability before this file's Node counterpart
 * added the same cleanup).
 */
export function runnerFromBrowserResident(
  device: ResidentDevice,
): (dispatch: {
  code: string;
  entry?: string;
  bindings: (
    | { kind: "storage"; data: Float32Array | Int32Array | Uint32Array }
    | { kind: "out"; type: "f32" | "i32" | "u32"; length: number }
    | { kind: "scratch"; length: number }
    | { kind: "uniform"; data: ArrayBuffer }
  )[];
  workgroups: [number] | [number, number] | [number, number, number];
}) => Promise<(Float32Array | Int32Array | Uint32Array)[]> {
  return async function run(dispatch) {
    const { code, entry = "main", bindings, workgroups } = dispatch;
    const pipeline = await device.pipelineFor(code, entry);

    const buffers: GPUBuffer[] = [];
    const outputs: { spec: Extract<(typeof bindings)[number], { kind: "out" }>; buffer: GPUBuffer }[] = [];

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
      for (const buffer of buffers) buffer.destroy();
      for (const { buffer } of outputs) buffer.destroy();
      for (const staging of stagingBuffers) staging.destroy();
    }
  };
}
