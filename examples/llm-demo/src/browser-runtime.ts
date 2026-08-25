/**
 * This demo's browser-native counterpart to `harness/wgsl.ts` /
 * `harness/suite.ts`'s `kernel`/`params`, wired in for `llm/kernels.ts` at
 * bundle time (see `build.mjs`'s `harnessBrowserShim` for why and how the
 * redirect happens, and why `kernels.ts` itself needed zero changes).
 *
 * `Dispatch`/`Binding` are imported **type-only** from the real
 * `harness/wgsl.ts` — erased entirely before bundling, so this file never
 * pulls in that module's top-level `import { create, globals } from
 * "webgpu"` (a native Node addon with no browser build). Everything else
 * below — `params()`'s uniform-packing and `run()`'s dispatch sequence — is
 * a deliberate, small duplication of `harness/wgsl.ts`'s Node/Dawn version,
 * ported to `navigator.gpu` instead of the `webgpu` package. Two copies
 * exist because the two run on opposite sides of exactly the module
 * boundary `webgpu` draws (Node-native vs. browser-native WebGPU), not
 * because anything here is demo-specific; if the two ever disagree, a
 * generation in this browser demo says so directly (wrong tokens, or a
 * dispatch validation error), so drift does not stay silent.
 */
import type { Binding, Dispatch } from "../../../harness/wgsl.js";

// PR #119 review, item 8: this file used to re-export
// `runnerFromBrowserResident` under `harness/resident.ts#runnerFromResident`'s
// own name, for `build.mjs`'s `harnessBrowserShim` to redirect
// `llm/engine-q8-resident.ts`'s (pre-#117) value import of it to. Issue #117
// removed that import — `engine-q8-resident.ts` now imports only the *type*
// `ResidentDevice` (that file's own module doc) — so nothing in this bundle
// has resolved `runnerFromResident` as a value since, and
// `runnerFromBrowserResident` itself (`browser-resident-runtime.ts`) was
// deleted alongside this re-export rather than left dead.

// The kernel entry points `llm/kernels.ts#CODE` and `llm/engine-q8-resident.ts#CODE`
// build via `harness/suite.ts#kernel(url, name)` — one static import per
// `.wgsl` file so esbuild's `text` loader (see `build.mjs`) can inline each
// one as a JS string at bundle time. Both files import `kernel`/`params`
// from `harness/index.js`, which `build.mjs`'s `harnessBrowserShim`
// redirects to this module for *every* importer in the bundle graph, not
// just `kernels.ts` — so this table has to cover both files' entries, not
// only the ten `kernels.ts#CODE` builds (issue #106) plus `permute`/
// `dequant_transpose` (issue #117). `llm/kernels.browser-parity.test.ts`
// (run under `npm test`, no browser needed — it reads all three files as
// text rather than importing them) asserts this table is the exact union of
// both `CODE` objects' entries, per op.
import gatherKernel from "../../../ops/gather/wgsl/kernel.wgsl";
import rmsnormKernel from "../../../ops/rmsnorm/wgsl/kernel.wgsl";
import matvecKernel from "../../../ops/matvec/wgsl/kernel.wgsl";
import matvecQ8Kernel from "../../../ops/matvec/wgsl/q8.wgsl";
// Issue #111: `llm/engine-q8-resident.ts`'s two decode-only fused entry
// points — not referenced by `llm/kernels.ts#CODE` (that file's `opKernel`
// calls are what `kernels.browser-parity.test.ts` checks against this
// table), but still resolved through this same `harness/index.js` ->
// `browser-runtime.ts` redirect (`build.mjs`'s `harnessBrowserShim` has no
// way to tell which importer is asking), so they need an entry here too —
// see that test file's own doc for how it now covers both `kernels.ts` and
// `engine-q8-resident.ts`.
import matvecQ8FfnKernel from "../../../ops/matvec/wgsl/q8_ffn.wgsl";
import matvecQ8ResidualKernel from "../../../ops/matvec/wgsl/q8_residual.wgsl";
import matmulKernel from "../../../ops/matmul/wgsl/kernel.wgsl";
// Issue #128: `LlamaEngineQ8Resident.runPrefillResident`'s prefill weight
// path — reads the packed int8 weight directly, in-kernel, replacing the
// dequant_transpose+matmul pair below for that one call site (`kernels.ts`
// still asks for both, for parity/integration-test purposes — see those
// imports' own doc a few lines down).
import matmulQ8Kernel from "../../../ops/matmul/wgsl/q8.wgsl";
import ropeKernel from "../../../ops/rope/wgsl/kernel.wgsl";
import gqaScoresKernel from "../../../ops/gqa/wgsl/scores.wgsl";
import gqaContextKernel from "../../../ops/gqa/wgsl/context.wgsl";
import activationKernel from "../../../ops/activation/wgsl/kernel.wgsl";
import elementwiseKernel from "../../../ops/elementwise/wgsl/kernel.wgsl";
// `ops/permute` (issue #117's resident prefill reshape, `ops/permute/reference.ts`'s
// doc has the full story) — `llm/engine-q8-resident.ts` is its only caller
// today, but it is a `kernels.ts#CODE` entry too (kept, per that file's own
// "one function per kernel entry point" scope, even though nothing in
// `LlamaEngineQ8`'s own forward pass calls `runPermute` yet), so it stays
// covered by this table's usual parity check.
import permuteKernel from "../../../ops/permute/wgsl/kernel.wgsl";
// `ops/dequant_transpose` (`ops/dequant_transpose/reference.ts`'s doc has
// the full story) — issue #128 removed its one production caller
// (`LlamaEngineQ8Resident.runPrefillResident`, replaced by `matmulQ8` above),
// but `kernels.ts#CODE` still asks for it (its own `runDequantTranspose`
// wrapper is this op's Node-side integration test path — that file's own
// doc), so it stays in this table too.
import dequantTransposeKernel from "../../../ops/dequant_transpose/wgsl/kernel.wgsl";

/**
 * Mirrors `llm/kernels.ts`'s `CODE` object one-for-one: `{ [op]: { [entry]:
 * source } }`, `entry` defaulting to `"kernel"` the same way
 * `harness/suite.ts#kernel(url, name = "kernel")` does.
 */
export const WGSL_TABLE: Readonly<Record<string, Readonly<Record<string, string>>>> = {
  gather: { kernel: gatherKernel },
  rmsnorm: { kernel: rmsnormKernel },
  matvec: { kernel: matvecKernel, q8: matvecQ8Kernel, q8_ffn: matvecQ8FfnKernel, q8_residual: matvecQ8ResidualKernel },
  matmul: { kernel: matmulKernel, q8: matmulQ8Kernel },
  rope: { kernel: ropeKernel },
  permute: { kernel: permuteKernel },
  dequant_transpose: { kernel: dequantTransposeKernel },
  gqa: { scores: gqaScoresKernel, context: gqaContextKernel },
  activation: { kernel: activationKernel },
  elementwise: { kernel: elementwiseKernel },
};

/**
 * `llm/kernels.ts#opKernel` calls this as
 * `kernel(new URL("../ops/<op>/index.ts", import.meta.url), entry)` — the
 * URL's path always ends `/ops/<op>/index.ts` regardless of where
 * `kernels.ts` itself lives, so the op name is pulled out of that shape
 * rather than assumed from the URL's structure otherwise.
 */
function opNameFromUrl(url: string | URL): string {
  const pathname = typeof url === "string" ? url : url.pathname;
  const match = /\/ops\/([^/]+)\/index\.ts$/.exec(pathname);
  if (!match) {
    throw new Error(`browser-runtime: cannot derive an op name from ${JSON.stringify(pathname)}`);
  }
  return match[1]!;
}

/** Same signature as `harness/suite.ts#kernel` — see that file's doc for the convention this reproduces. */
export function kernel(url: string | URL, name = "kernel"): string {
  const op = opNameFromUrl(url);
  const entries = WGSL_TABLE[op];
  if (!entries) throw new Error(`browser-runtime: no bundled WGSL for op "${op}" — is WGSL_TABLE missing an import?`);
  const source = entries[name];
  if (source === undefined) {
    throw new Error(`browser-runtime: no bundled WGSL entry "${name}" for op "${op}"`);
  }
  return source;
}

/**
 * Packs a params struct of mixed u32 / i32 / f32 into a uniform buffer.
 *
 * Byte-for-byte identical to `harness/wgsl.ts#params` — duplicated rather
 * than imported (see this file's module doc for why importing it is not an
 * option) and kept simple and stable on purpose: this function's contract is
 * "match the uniform-buffer layout every `.wgsl` kernel declares", the same
 * contract `harness/wgsl.ts`'s copy holds, so there is little for the two to
 * drift over.
 */
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
