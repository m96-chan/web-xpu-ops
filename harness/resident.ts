import { create, globals } from "webgpu";
import { compilationFailure, type Binding, type Dispatch, type Runner } from "./wgsl.js";

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

export interface ResidentDevice {
  /**
   * Counters a test can snapshot before and after a decode loop to prove the
   * loop itself allocates nothing (issue #110's "トークンループ内での
   * create*呼び出しゼロ") — the property code review alone cannot keep honest
   * once this file has more than one caller, per rule 1 ("観測点が間違っている"
   * failures do not show up in coverage).
   */
  readonly stats: { buffersCreated: number; pipelinesCreated: number; submits: number };
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
   * Records every op into one `GPUCommandEncoder`, submits it exactly once,
   * then maps and reads back only `readback` — everything else recorded
   * (intermediate activations, the KV-cache copies) stays device-side.
   */
  batch(ops: ResidentOp[], readback: ResidentReadback[]): Promise<(Float32Array | Int32Array | Uint32Array)[]>;
  destroy(): void;
}

/** Resolves to null when no adapter is available, mirroring `wgsl.ts#createRunner`. */
export async function createResidentDevice(): Promise<ResidentDevice | null> {
  Object.assign(globalThis, globals);
  const gpu = create([]) as GPU;
  const adapter = await gpu.requestAdapter();
  if (!adapter) return null;

  // Same reasoning as `wgsl.ts#createRunner`: request the adapter's own
  // ceiling rather than the spec's low defaults. A resident engine's weight
  // buffers (`lmHead` alone is ~175 MiB packed for Sarashina2.2-1B) need it.
  const wanted = 2 * 1024 * 1024 * 1024;
  const device = await adapter.requestDevice({
    requiredLimits: {
      maxStorageBufferBindingSize: Math.min(wanted, adapter.limits.maxStorageBufferBindingSize),
      maxBufferSize: Math.min(wanted, adapter.limits.maxBufferSize),
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
    // Same `as any` cast `wgsl.ts#dispatch` documents: `data` is always a
    // plain-`ArrayBuffer`-backed view here, never `SharedArrayBuffer`, but
    // `@webgpu/types` cannot express that without `DOM` lib.
    device.queue.writeBuffer(buffer, offset, data.buffer as any, data.byteOffset, data.byteLength);
  }

  async function pipelineFor(code: string, entry = "main"): Promise<GPUComputePipeline> {
    const key = `${entry} ${code}`;
    const cached = pipelines.get(key);
    if (cached) return cached;
    let module = modules.get(code);
    if (!module) {
      module = device.createShaderModule({ code });
      const failure = compilationFailure((await module.getCompilationInfo()).messages);
      if (failure) throw new Error(failure);
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

  async function batch(ops: ResidentOp[], readback: ResidentReadback[]): Promise<(Float32Array | Int32Array | Uint32Array)[]> {
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
    device.queue.submit([encoder.finish()]);
    stats.submits += 1;

    const results: (Float32Array | Int32Array | Uint32Array)[] = [];
    for (const r of readback) {
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
