/**
 * `harness/wgsl.ts#createRunner` over `navigator.gpu`: one dispatch per
 * call, every binding allocated for that call and destroyed after it.
 *
 * Lived in `examples/llm-demo/src/browser-runtime.ts` next to that demo's
 * WGSL table until issue #224 published it: the runner is what conditioning
 * and the VAE use in every `*-web` demo and is not tied to any of them, while
 * the table is one demo's static imports. `Dispatch`/`Binding` come from
 * `harness/api.ts`, which is the harness contract without its Node runtime.
 * Everything below is a deliberate, small duplication of `harness/wgsl.ts`'s
 * Dawn version ported to the browser API — two copies because the two run on
 * opposite sides of exactly the module boundary the `webgpu` package draws.
 */
import type { Binding, Dispatch } from "../../../harness/api.js";

/**
 * Cumulative wall clock inside `run`, by phase, plus what it allocated.
 *
 * Issue #188. This runner does everything per dispatch — pipeline, buffers,
 * upload, bind group, submit, readback, destroy — and it is what conditioning
 * and decode use while the DiT runs batched on a resident device. Those two
 * phases are 42.7 s of a 225 s generation and nothing has ever measured where
 * inside them the time goes. A caller reads these as differences across the
 * span it cares about; nothing here resets.
 */
export interface RunnerStats {
  /** `run` calls. */
  dispatches: number;
  /** `createComputePipeline`, which this path does per call — only the shader *module* is cached. */
  pipelineMs: number;
  /** `createBuffer` plus `queue.writeBuffer` for every binding. */
  uploadMs: number;
  /** Bytes handed to `writeBuffer`. */
  uploadedBytes: number;
  /** `createBindGroup` and the `popErrorScope` that follows it. */
  bindGroupMs: number;
  /** Encoding the pass and `queue.submit`. */
  submitMs: number;
  /** `mapAsync` and the copy out. */
  readbackMs: number;
  /** `destroy()` on everything the call made. */
  destroyMs: number;
  /** Buffers created, so the per-dispatch churn is a number rather than an adjective. */
  buffersCreated: number;
}

export interface BrowserRunner {
  run(dispatch: Dispatch): Promise<(Float32Array | Int32Array | Uint32Array)[]>;
  readonly stats: RunnerStats;
  destroy(): void;
}

/**
 * `navigator.gpu`'s counterpart to `harness/wgsl.ts#createRunner` — the
 * dispatch sequence (compile-or-cache the shader, allocate + upload every
 * binding, validate the bind group, encode one compute pass, read every
 * `"out"` binding back) is a direct port. Left out on purpose, because
 * nothing in `llm/kernels.ts` needs it: `harness`'s `scratch` binding kind
 * (roofline calibration only) and its GPU-timestamp `time()` path (this demo
 * measures tok/s at the `engine.forward()` call boundary in `main.ts`
 * instead, which is simpler and sufficient for this issue's ask).
 */
export async function createBrowserRunner(): Promise<BrowserRunner> {
  const gpu = (navigator as Navigator & { gpu?: GPU }).gpu;
  if (!gpu) {
    throw new Error("createBrowserRunner: navigator.gpu is unavailable — WebGPU is not enabled in this browser");
  }
  const adapter = await gpu.requestAdapter();
  if (!adapter) {
    throw new Error("createBrowserRunner: requestAdapter() returned null — no WebGPU adapter on this machine");
  }
  // Asked for explicitly, the same reason `harness/wgsl.ts#createRunner`
  // does: the spec's default limits (e.g. a 128 MiB storage binding) are far
  // below what a real checkpoint's projections need — Sarashina2.2-1B's
  // packed `lmHead` alone is ~175 MiB. Requesting the adapter's own reported
  // ceiling (rather than a fixed number, and rather than nothing) is the
  // only value that is never a guess about what this specific device offers.
  const device = await adapter.requestDevice({
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

  /** Compiled modules, keyed by source — the same one-await-per-distinct-shader cache `harness/wgsl.ts` keeps. */
  const compiled = new Map<string, GPUShaderModule>();
  const stats: RunnerStats = {
    dispatches: 0, pipelineMs: 0, uploadMs: 0, uploadedBytes: 0,
    bindGroupMs: 0, submitMs: 0, readbackMs: 0, destroyMs: 0, buffersCreated: 0,
  };

  async function run(spec: Dispatch): Promise<(Float32Array | Int32Array | Uint32Array)[]> {
    const { code, entry = "main", bindings, workgroups } = spec;

    let module = compiled.get(code);
    if (!module) {
      module = device.createShaderModule({ code });
      const info = await module.getCompilationInfo();
      const errors = info.messages.filter((m) => m.type === "error");
      if (errors.length > 0) {
        const where = (m: GPUCompilationMessage) => `${m.lineNum}:${m.linePos}: ${m.message}`;
        throw new Error(`shader failed to compile\n${errors.map(where).join("\n")}`);
      }
      compiled.set(code, module);
    }

    stats.dispatches += 1;
    const uploadT0 = performance.now();
    const created: GPUBuffer[] = [];
    const outputs: { spec: Extract<Binding, { kind: "out" }>; buffer: GPUBuffer }[] = [];

    const bound: GPUBuffer[] = bindings.map((binding: Binding) => {
      if (binding.kind === "out") {
        const buffer = device.createBuffer({
          size: binding.length * 4,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        });
        created.push(buffer);
        outputs.push({ spec: binding, buffer });
        return buffer;
      }
      if (binding.kind === "scratch") {
        // Not reachable from `llm/kernels.ts` today — see this file's module doc.
        throw new Error("createBrowserRunner: 'scratch' bindings are not implemented in the browser runner");
      }
      const bytes =
        binding.kind === "uniform"
          ? new Uint8Array(binding.data)
          : new Uint8Array(binding.data.buffer, binding.data.byteOffset, binding.data.byteLength);
      const buffer = device.createBuffer({
        size: Math.max(binding.kind === "uniform" ? 16 : 4, bytes.byteLength),
        usage:
          binding.kind === "uniform"
            ? GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
            : GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      // `bytes` is always backed by a plain `ArrayBuffer` here (constructed
      // above from either `binding.data.buffer` — a `Float32Array` /
      // `Int32Array` / `Uint32Array` — or a uniform's own `ArrayBuffer`),
      // never a `SharedArrayBuffer`; `@webgpu/types`' `GPUAllowSharedBufferSource`
      // and TypeScript 5.7's generic `Uint8Array<ArrayBufferLike>` can't
      // express that, so this is a type-level cast, not a runtime one — the
      // same pattern `harness/wgsl.ts#dispatch` needs for the identical call
      // (see that file's comment for why `as any` rather than naming
      // `BufferSource` explicitly).
      device.queue.writeBuffer(buffer, 0, bytes as any);
      stats.uploadedBytes += bytes.byteLength;
      created.push(buffer);
      return buffer;
    });
    stats.uploadMs += performance.now() - uploadT0;

    // Same reason `harness/wgsl.ts` pushes an error scope here: `layout:
    // "auto"` silently drops a binding an entry point never references, and
    // the bind group then fails validation rather than the pipeline.
    device.pushErrorScope("validation");
    const pipelineT0 = performance.now();
    const pipeline = device.createComputePipeline({ layout: "auto", compute: { module, entryPoint: entry } });
    const bindGroupT0 = performance.now();
    stats.pipelineMs += bindGroupT0 - pipelineT0;
    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: bound.map((buffer, binding) => ({ binding, resource: { buffer } })),
    });
    const invalid = await device.popErrorScope();
    stats.bindGroupMs += performance.now() - bindGroupT0;
    if (invalid) throw new Error(`dispatch is not valid: ${invalid.message}`);

    const submitT0 = performance.now();
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(...(workgroups as [number, number?, number?]));
    pass.end();

    const staging = outputs.map(({ spec: outSpec, buffer }) => {
      const read = device.createBuffer({
        size: outSpec.length * 4,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
      encoder.copyBufferToBuffer(buffer, 0, read, 0, outSpec.length * 4);
      created.push(read);
      return read;
    });
    device.queue.submit([encoder.finish()]);
    stats.submitMs += performance.now() - submitT0;

    const readbackT0 = performance.now();
    const results: (Float32Array | Int32Array | Uint32Array)[] = [];
    for (const [index, read] of staging.entries()) {
      await read.mapAsync(GPUMapMode.READ);
      const bytes = read.getMappedRange().slice(0);
      read.unmap();
      const type = outputs[index]!.spec.type;
      results.push(
        type === "i32" ? new Int32Array(bytes) : type === "u32" ? new Uint32Array(bytes) : new Float32Array(bytes),
      );
    }
    stats.readbackMs += performance.now() - readbackT0;

    const destroyT0 = performance.now();
    stats.buffersCreated += created.length;
    for (const buffer of created) buffer.destroy();
    stats.destroyMs += performance.now() - destroyT0;
    return results;
  }

  return {
    run,
    stats,
    destroy() {
      device.destroy();
    },
  };
}
