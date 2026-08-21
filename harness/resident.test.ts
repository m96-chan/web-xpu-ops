import { describe, expect, it } from "vitest";
import { runnerFromResident, type BatchProfileSink } from "./resident.js";
import { params } from "./wgsl.js";
import { kernel, residentTest, skipUnlessPresent, useResidentGpu } from "./suite.js";

const elementwiseKernel = kernel(new URL("../ops/elementwise/index.ts", import.meta.url));

/**
 * `harness/resident.ts` exists for issue #110: build every buffer, pipeline
 * and bind group once, then run many dispatches behind one `queue.submit`,
 * reading back only what was asked for. These tests exercise that contract
 * directly, against a real op's real WGSL (`ops/elementwise`) rather than a
 * toy shader — `llm/engine-q8-resident.ts` chains exactly this kernel for
 * residual adds, so a resident-mode bug in binding order or offset handling
 * shows up here first, at a fraction of the setup.
 */
describe("resident device", () => {
  const getDevice = useResidentGpu();

  residentTest("chains two dispatches through a GPU-resident intermediate buffer, one submit, one readback", async (device) => {
    // a=[1,2,3,4], b=[10,10,10,10]: sum = a+b = [11,12,13,14], then
    // product = sum*a = [11*1, 12*2, 13*3, 14*4] = [11, 24, 39, 56] — the
    // second dispatch reads the first dispatch's *output buffer* straight
    // back in as an input, with no CPU round trip in between.
    const a = new Float32Array([1, 2, 3, 4]);
    const b = new Float32Array([10, 10, 10, 10]);
    const N = a.length;

    const aBuf = device.createStorageBuffer(N * 4);
    const bBuf = device.createStorageBuffer(N * 4);
    const sumBuf = device.createStorageBuffer(N * 4);
    const productBuf = device.createStorageBuffer(N * 4);
    const addParams = device.createUniformBuffer(16);
    const mulParams = device.createUniformBuffer(16);
    const staging = device.createStorageBuffer(N * 4, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);

    device.upload(aBuf, 0, a);
    device.upload(bBuf, 0, b);
    device.upload(addParams, 0, new Uint8Array(params([["u32", N], ["u32", 0]])));
    device.upload(mulParams, 0, new Uint8Array(params([["u32", N], ["u32", 1]])));

    const pipeline = await device.pipelineFor(elementwiseKernel);
    const addGroup = await device.bindGroup(pipeline, [aBuf, bBuf, sumBuf, addParams]);
    // Second dispatch reads the first dispatch's output straight back in as
    // an input — no CPU round trip between them, which is the entire point.
    const mulGroup = await device.bindGroup(pipeline, [sumBuf, aBuf, productBuf, mulParams]);

    const submitsBefore = device.stats.submits;
    const [result] = await device.batch(
      [
        { kind: "dispatch", pipeline, bindGroup: addGroup, workgroups: [1] },
        { kind: "dispatch", pipeline, bindGroup: mulGroup, workgroups: [1] },
      ],
      [{ staging, source: productBuf, sourceOffset: 0, length: N, type: "f32" }],
    );

    expect(device.stats.submits).toBe(submitsBefore + 1);
    expect(Array.from(result as Float32Array)).toEqual([11, 24, 39, 56]);
  });

  residentTest("a copy op moves bytes between persistent buffers within the same submit", async (device) => {
    const src = device.createStorageBuffer(16);
    const dst = device.createStorageBuffer(16);
    const staging = device.createStorageBuffer(16, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
    device.upload(src, 0, new Float32Array([1, 2, 3, 4]));

    const [result] = await device.batch(
      [{ kind: "copy", src, srcOffset: 0, dst, dstOffset: 0, size: 16 }],
      [{ staging, source: dst, sourceOffset: 0, length: 4, type: "f32" }],
    );

    expect(Array.from(result as Float32Array)).toEqual([1, 2, 3, 4]);
  });

  residentTest("buffers created before batch() are not recreated by repeated batches — the decode-loop invariant", async (device) => {
    const a = device.createStorageBuffer(16);
    const b = device.createStorageBuffer(16);
    const out = device.createStorageBuffer(16);
    const uniform = device.createUniformBuffer(16);
    const staging = device.createStorageBuffer(16, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
    device.upload(a, 0, new Float32Array([1, 1, 1, 1]));
    device.upload(b, 0, new Float32Array([2, 2, 2, 2]));
    device.upload(uniform, 0, new Uint8Array(params([["u32", 4], ["u32", 0]])));

    const pipeline = await device.pipelineFor(elementwiseKernel);
    const bindGroup = await device.bindGroup(pipeline, [a, b, out, uniform]);

    const buffersBefore = device.stats.buffersCreated;
    const pipelinesBefore = device.stats.pipelinesCreated;
    for (let step = 0; step < 5; step += 1) {
      // eslint-disable-next-line no-await-in-loop
      await device.batch(
        [{ kind: "dispatch", pipeline, bindGroup, workgroups: [1] }],
        [{ staging, source: out, sourceOffset: 0, length: 4, type: "f32" }],
      );
    }

    expect(device.stats.buffersCreated).toBe(buffersBefore);
    expect(device.stats.pipelinesCreated).toBe(pipelinesBefore);
  });

  residentTest("two readbacks in one batch — the lmHead chunking shape — return independent slices", async (device) => {
    // Mirrors how `llm/engine-q8-resident.ts` reads `lmHead`'s two chunked
    // dispatches back (`MAX_WORKGROUPS_PER_DISPATCH`, `llm/kernels.ts`) as
    // two separate buffers rather than one buffer sliced by an unaligned
    // byte offset (see `resident.ts#bindGroup`'s doc for why offset slicing
    // is not used here) — one `batch()` call, one submit, two outputs.
    const a1 = device.createStorageBuffer(16);
    const b1 = device.createStorageBuffer(16);
    const out1 = device.createStorageBuffer(16);
    const uniform1 = device.createUniformBuffer(16);
    const staging1 = device.createStorageBuffer(16, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
    const a2 = device.createStorageBuffer(16);
    const b2 = device.createStorageBuffer(16);
    const out2 = device.createStorageBuffer(16);
    const uniform2 = device.createUniformBuffer(16);
    const staging2 = device.createStorageBuffer(16, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
    device.upload(a1, 0, new Float32Array([1, 2, 3, 4]));
    device.upload(b1, 0, new Float32Array([10, 10, 10, 10]));
    device.upload(uniform1, 0, new Uint8Array(params([["u32", 4], ["u32", 0]])));
    device.upload(a2, 0, new Float32Array([5, 6, 7, 8]));
    device.upload(b2, 0, new Float32Array([20, 20, 20, 20]));
    device.upload(uniform2, 0, new Uint8Array(params([["u32", 4], ["u32", 0]])));

    const pipeline = await device.pipelineFor(elementwiseKernel);
    const group1 = await device.bindGroup(pipeline, [a1, b1, out1, uniform1]);
    const group2 = await device.bindGroup(pipeline, [a2, b2, out2, uniform2]);

    const [chunk0, chunk1] = await device.batch(
      [
        { kind: "dispatch", pipeline, bindGroup: group1, workgroups: [1] },
        { kind: "dispatch", pipeline, bindGroup: group2, workgroups: [1] },
      ],
      [
        { staging: staging1, source: out1, sourceOffset: 0, length: 4, type: "f32" },
        { staging: staging2, source: out2, sourceOffset: 0, length: 4, type: "f32" },
      ],
    );

    expect(Array.from(chunk0 as Float32Array)).toEqual([11, 12, 13, 14]);
    expect(Array.from(chunk1 as Float32Array)).toEqual([25, 26, 27, 28]);
  });

  /**
   * PR #116 review, item 4: `pipelineFor` built its `GPUComputePipeline` with
   * no error scope around `createComputePipeline` — unlike `bindGroup`
   * (this file's own tests already exercise that scope indirectly) and
   * `wgsl.ts#dispatch`'s identical push/pop around the same call. A bogus
   * `entry` name is exactly the case that scope exists for (issue #46's own
   * failure mode: `layout: "auto"` still builds *a* pipeline layout for a
   * shader with no such entry point, so nothing before this failed loudly)
   * — without it, this call would instead throw much later and more opaquely
   * out of `bindGroup`'s own validation (a different call, a confusing
   * error), or not until the compute pass never wrote its expected output.
   */
  residentTest("pipelineFor rejects a shader/entry pair the module does not export", async (device) => {
    await expect(device.pipelineFor(elementwiseKernel, "no_such_entry_point")).rejects.toThrow(/pipeline is not valid/);
  });

  /**
   * PR #119 review, item 8: `runnerFromResident` had zero direct test
   * coverage — every existing check of it was indirect, through
   * `llm/engine-q8-resident.wgsl.test.ts`'s pre-#117 history (see that
   * function's own doc), and issue #117 removed the one production call site
   * that history describes (`llm/engine-q8-resident.ts` no longer imports
   * it — that file's own module doc says so directly). A function reachable
   * only from its own doc comment, with no test pinning its behaviour, is
   * exactly the kind of drift this repository's own culture (rule 8) exists
   * to catch before it rots silently; this test exercises it directly
   * against the same `Runner["run"]` calling convention `createRunner()`
   * uses (`harness/wgsl.ts`), so a future caller of `ResidentDevice` that
   * needs a `Runner` (the reason this file's own doc says it is kept) has a
   * green regression test to lean on rather than an unverified utility.
   */
  residentTest("runnerFromResident wraps a ResidentDevice as a Runner['run'] — one dispatch, ephemeral buffers, per-call readback", async (device) => {
    const run = runnerFromResident(device);
    const a = new Float32Array([1, 2, 3, 4]);
    const b = new Float32Array([10, 10, 10, 10]);
    const N = a.length;

    const buffersBefore = device.stats.buffersCreated;
    const [sum] = await run({
      code: elementwiseKernel,
      bindings: [
        { kind: "storage", data: a },
        { kind: "storage", data: b },
        { kind: "out", type: "f32", length: N },
        { kind: "uniform", data: params([["u32", N], ["u32", 0]]) },
      ],
      workgroups: [1],
    });

    expect(Array.from(sum as Float32Array)).toEqual([11, 12, 13, 14]);
    // Every buffer this call touched is `createStorageBuffer`/`createUniformBuffer`
    // ephemeral scaffolding (inputs, the "out" buffer, and its staging
    // buffer) — unlike `ResidentDevice`'s own persistent buffers (the ones
    // the "decode-loop invariant" test above pins), a second call through
    // the same `run` must not find any of that scaffolding still around to
    // reuse; each call creates its own.
    const buffersAfterFirst = device.stats.buffersCreated;
    expect(buffersAfterFirst).toBeGreaterThan(buffersBefore);

    const [sum2] = await run({
      code: elementwiseKernel,
      bindings: [
        { kind: "storage", data: a },
        { kind: "storage", data: b },
        { kind: "out", type: "f32", length: N },
        { kind: "uniform", data: params([["u32", N], ["u32", 0]]) },
      ],
      workgroups: [1],
    });
    expect(Array.from(sum2 as Float32Array)).toEqual([11, 12, 13, 14]);
    expect(device.stats.buffersCreated).toBeGreaterThan(buffersAfterFirst);
  });
});

/**
 * Issue #131: `batch()`'s optional third argument. These exercise the real
 * contract `llm/engine-q8-resident.ts` will drive it through — `submitToDoneMs`/
 * `readbackMs` filled in on every profiled call, `gpuEntries` populated only
 * for dispatches the caller actually labeled, and only when this device
 * negotiated `timestamp-query` (checked via `device.timestampsSupported`
 * rather than assumed, since a CI machine without it must still pass this
 * file — the assertions below are structured so they hold either way,
 * per this repository's own "対応しない環境ではCPU側計時にフォールバック" scope).
 */
describe("resident device / BatchProfile (issue #131)", () => {
  const getDevice = useResidentGpu();

  function twoDispatchOps(device: NonNullable<ReturnType<typeof getDevice>>) {
    const a = device.createStorageBuffer(16);
    const b = device.createStorageBuffer(16);
    const sum = device.createStorageBuffer(16);
    const product = device.createStorageBuffer(16);
    const addParams = device.createUniformBuffer(16);
    const mulParams = device.createUniformBuffer(16);
    device.upload(a, 0, new Float32Array([1, 2, 3, 4]));
    device.upload(b, 0, new Float32Array([10, 10, 10, 10]));
    device.upload(addParams, 0, new Uint8Array(params([["u32", 4], ["u32", 0]])));
    device.upload(mulParams, 0, new Uint8Array(params([["u32", 4], ["u32", 1]])));
    return { a, b, sum, product, addParams, mulParams };
  }

  residentTest("submitToDoneMs and readbackMs are both real numbers once a profiled batch resolves", async (device) => {
    const pipeline = await device.pipelineFor(elementwiseKernel);
    const { a, b, sum, product, addParams, mulParams } = twoDispatchOps(device);
    const addGroup = await device.bindGroup(pipeline, [a, b, sum, addParams]);
    const mulGroup = await device.bindGroup(pipeline, [sum, a, product, mulParams]);
    const staging = device.createStorageBuffer(16, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);

    const sink: BatchProfileSink = { submitToDoneMs: null, readbackMs: null, gpuEntries: [] };
    const [result] = await device.batch(
      [
        { kind: "dispatch", pipeline, bindGroup: addGroup, workgroups: [1] },
        { kind: "dispatch", pipeline, bindGroup: mulGroup, workgroups: [1] },
      ],
      [{ staging, source: product, sourceOffset: 0, length: 4, type: "f32" }],
      { sink },
    );

    // Correctness is unaffected by asking for a profile — same result the
    // unprofiled version of this exact dispatch pair produces above.
    expect(Array.from(result as Float32Array)).toEqual([11, 24, 39, 56]);
    expect(typeof sink.submitToDoneMs).toBe("number");
    expect(sink.submitToDoneMs).toBeGreaterThanOrEqual(0);
    expect(typeof sink.readbackMs).toBe("number");
    expect(sink.readbackMs).toBeGreaterThanOrEqual(0);
    // No `labels` were passed — nothing here asked for a GPU breakdown, so
    // none should appear (an empty array reads as "not requested", not
    // "requested and measured zero" — the `timestampsSupported`-gated test
    // below is what exercises the populated case).
    expect(sink.gpuEntries).toEqual([]);
  });

  it("labeled dispatches produce one gpuEntries row each, by label, when timestamp-query is supported", async (ctx) => {
    const device = getDevice();
    if (!skipUnlessPresent(ctx, device)) return;
    if (!device.timestampsSupported) {
      // Documented fallback (issue #131's own scope: "timestamp-query非対応
      // 環境ではCPU側計時にフォールバック") — this repository's own dev/CI
      // machines negotiate it (confirmed via `adapter.features` directly),
      // so this branch is not expected to run here, but a future
      // environment without it must not fail this file.
      ctx.skip();
      return;
    }
    const pipeline = await device.pipelineFor(elementwiseKernel);
    const { a, b, sum, product, addParams, mulParams } = twoDispatchOps(device);
    const addGroup = await device.bindGroup(pipeline, [a, b, sum, addParams]);
    const mulGroup = await device.bindGroup(pipeline, [sum, a, product, mulParams]);
    const staging = device.createStorageBuffer(16, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);

    const sink: BatchProfileSink = { submitToDoneMs: null, readbackMs: null, gpuEntries: [] };
    await device.batch(
      [
        { kind: "dispatch", pipeline, bindGroup: addGroup, workgroups: [1] },
        { kind: "dispatch", pipeline, bindGroup: mulGroup, workgroups: [1] },
      ],
      [{ staging, source: product, sourceOffset: 0, length: 4, type: "f32" }],
      { labels: ["add", "mul"], sink },
    );

    const labels = sink.gpuEntries.map((e) => e.label).sort();
    expect(labels).toEqual(["add", "mul"]);
    for (const entry of sink.gpuEntries) expect(entry.seconds).toBeGreaterThan(0);
  });

  it("a null label leaves that dispatch out of gpuEntries", async (ctx) => {
    const device = getDevice();
    if (!skipUnlessPresent(ctx, device)) return;
    if (!device.timestampsSupported) {
      ctx.skip();
      return;
    }
    const pipeline = await device.pipelineFor(elementwiseKernel);
    const { a, b, sum, product, addParams, mulParams } = twoDispatchOps(device);
    const addGroup = await device.bindGroup(pipeline, [a, b, sum, addParams]);
    const mulGroup = await device.bindGroup(pipeline, [sum, a, product, mulParams]);
    const staging = device.createStorageBuffer(16, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);

    const sink: BatchProfileSink = { submitToDoneMs: null, readbackMs: null, gpuEntries: [] };
    await device.batch(
      [
        { kind: "dispatch", pipeline, bindGroup: addGroup, workgroups: [1] },
        { kind: "dispatch", pipeline, bindGroup: mulGroup, workgroups: [1] },
      ],
      [{ staging, source: product, sourceOffset: 0, length: 4, type: "f32" }],
      { labels: ["add", null], sink },
    );

    expect(sink.gpuEntries.map((e) => e.label)).toEqual(["add"]);
  });
});
