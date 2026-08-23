import { describe, expect } from "vitest";
import { agree, params } from "../../harness/index.js";
import { kernel, residentTest, useResidentGpu } from "../../harness/suite.js";
import { axpy } from "./reference.js";

/**
 * `wgsl/inplace.wgsl` — `y[i] += a * x[i]`, the entry point a rectified-flow
 * sampler's step loop actually calls.
 *
 * Driven through `harness/resident.ts` rather than `harness/wgsl.ts`, and that
 * is the point of the file rather than a convenience. `Runner` has no binding
 * kind for a buffer that is uploaded *and* read back — `storage` uploads and
 * never returns, `out` returns and never uploads — so in-place is not
 * expressible through it at all, and the question this op had to answer (issue
 * #152: "GPU バッファの読み書き同時アクセスになる。常駐バッファ運用と整合するか")
 * is a question about the resident path specifically: buffers, pipeline and
 * bind group built once, then many steps, each of which reads and writes the
 * same latent.
 *
 * What the answer turned out to be, measured on this device (Dawn / Vulkan /
 * RTX 5090) and written up in `wgsl/inplace.wgsl`:
 *
 *  - **One `read_write` binding read and written by the same invocation is
 *    fine.** Invocation `i` touches element `i` and no other, so there is no
 *    cross-invocation ordering to depend on. Four steps below, chained through
 *    the same buffer, agree with four chained applications of the reference.
 *  - **Aliasing one buffer into `kernel.wgsl`'s `y` and `output` is not fine,
 *    and does not say so.** `createBindGroup` succeeds; `finish()` then
 *    invalidates the command buffer ("usage (Storage(read-write)|Storage(read-
 *    only)) includes writable usage and another usage in the same
 *    synchronization scope"), the submit is dropped, and the readback is all
 *    zeros with nothing thrown. Hence a separate entry point rather than a
 *    documented calling trick.
 */

const inplaceCode = kernel(new URL("./index.ts", import.meta.url), "inplace");
const outOfPlaceCode = kernel(new URL("./index.ts", import.meta.url));

const WG = 256;
/** Not a multiple of 256: elements 300..511 exist in the buffer and must survive every step. */
const N = 300;
const PADDED = 2 * WG;
const SENTINEL = -12.5;

const x = new Float32Array(PADDED).fill(SENTINEL);
const y0 = new Float32Array(PADDED).fill(SENTINEL);
for (let i = 0; i < N; i += 1) {
  // `(i + 1)`, not `i`: `sin(0)` is exactly zero, and an element whose `x` is
  // zero cannot tell `y + a*x` from `y` — element 0 would be blind to the
  // scalar being dropped entirely.
  x[i] = Math.sin((i + 1) * 0.37) * 3;
  y0[i] = Math.cos(i * 0.11) * 2;
}

/**
 * Four steps, and the coefficient changes on every one of them — which is the
 * whole reason `a` sits in a uniform that gets rewritten rather than in a
 * buffer uploaded once. `0` is in the list because it is the one value that
 * would let a kernel ignoring `x` still look right at the first step.
 */
const SCHEDULE = [0.5, -0.25, 2, 0];

/** The reference chain: each step's output is the next step's `y`, in f32 all the way. */
const expectedSteps: Float32Array[] = [];
{
  let live: Float32Array = y0.slice(0, N);
  for (const a of SCHEDULE) {
    live = axpy({ x: x.subarray(0, N), y: live, a });
    const padded = new Float32Array(PADDED).fill(SENTINEL);
    padded.set(live);
    expectedSteps.push(padded);
  }
}

describe("axpy / inplace / resident", () => {
  useResidentGpu();

  residentTest("steps a resident buffer in place, one submit and one 16-byte upload per step", async (device) => {
    const xBuf = device.createStorageBuffer(PADDED * 4);
    const yBuf = device.createStorageBuffer(PADDED * 4);
    const uniform = device.createUniformBuffer(16);
    const staging = device.createStorageBuffer(PADDED * 4, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
    device.upload(xBuf, 0, x);
    device.upload(yBuf, 0, y0);

    const pipeline = await device.pipelineFor(inplaceCode);
    const group = await device.bindGroup(pipeline, [xBuf, yBuf, uniform]);

    // Snapshotted after every allocation and every bind group, so what the
    // loop below is allowed to do is exactly: write 16 bytes, submit, read
    // back. This is the property #116 exists for, checked for this op rather
    // than assumed from the harness — `x` and `y` never move, and `a` is the
    // only thing that crosses the bus per step.
    const buffersBefore = device.stats.buffersCreated;
    const pipelinesBefore = device.stats.pipelinesCreated;
    const submitsBefore = device.stats.submits;

    const seen: Float32Array[] = [];
    for (const a of SCHEDULE) {
      device.upload(uniform, 0, new Uint8Array(params([["u32", N], ["f32", a]])));
      // eslint-disable-next-line no-await-in-loop
      const [step] = await device.batch(
        [{ kind: "dispatch", pipeline, bindGroup: group, workgroups: [Math.ceil(N / WG)] }],
        [{ staging, source: yBuf, sourceOffset: 0, length: PADDED, type: "f32" }],
      );
      seen.push(step as Float32Array);
    }

    expect(device.stats.buffersCreated).toBe(buffersBefore);
    expect(device.stats.pipelinesCreated).toBe(pipelinesBefore);
    expect(device.stats.submits).toBe(submitsBefore + SCHEDULE.length);

    // Every step, not only the last: an error introduced at step 2 and undone
    // at step 3 would be invisible in the final buffer alone.
    seen.forEach((got, step) => {
      const worst = agree(got, expectedSteps[step]!);
      expect(worst, worst ? `step ${step} (a = ${SCHEDULE[step]}): ${JSON.stringify(worst)}` : undefined).toBeNull();
    });

    // The tail: 212 elements past `N` that four in-place steps must have left
    // exactly as they were. In-place makes this test sharper than the
    // out-of-place one — there the tail is a buffer nobody wrote, here it is
    // the caller's own live data sitting in the same allocation.
    const last = seen[seen.length - 1]!;
    for (let i = N; i < PADDED; i += 1) expect(last[i]).toBe(SENTINEL);
  });

  residentTest("computes bit-for-bit what the out-of-place entry point computes", async (device) => {
    // The two entry points differ only in where the answer lands, so anything
    // else that differed — a stale read of `y`, a lost update — would show up
    // as a numeric difference here. Bit-for-bit, not `agree`: both run the
    // same expression on the same device from the same bytes, so a single ulp
    // of difference would mean one of them is not doing what it says.
    const a = -0.75;
    const xBuf = device.createStorageBuffer(PADDED * 4);
    const inPlaceBuf = device.createStorageBuffer(PADDED * 4);
    const sourceBuf = device.createStorageBuffer(PADDED * 4);
    const outBuf = device.createStorageBuffer(PADDED * 4);
    const uniform = device.createUniformBuffer(16);
    const stagingA = device.createStorageBuffer(PADDED * 4, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
    const stagingB = device.createStorageBuffer(PADDED * 4, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
    device.upload(xBuf, 0, x);
    device.upload(inPlaceBuf, 0, y0);
    device.upload(sourceBuf, 0, y0);
    // One uniform, two entry points: `{N, a}` is the same block for both, and
    // both read it, so there is no hazard in sharing it inside one submit.
    device.upload(uniform, 0, new Uint8Array(params([["u32", N], ["f32", a]])));

    const inPlace = await device.pipelineFor(inplaceCode);
    const outOfPlace = await device.pipelineFor(outOfPlaceCode);
    const inPlaceGroup = await device.bindGroup(inPlace, [xBuf, inPlaceBuf, uniform]);
    // Deliberately a *different* buffer from `inPlaceBuf` holding the same
    // bytes: the two dispatches share one compute pass, so pointing both at
    // one `y` would be the aliasing hazard this op has a second entry point
    // to avoid, not a test of anything.
    const outOfPlaceGroup = await device.bindGroup(outOfPlace, [xBuf, sourceBuf, outBuf, uniform]);

    const [updated, written] = await device.batch(
      [
        { kind: "dispatch", pipeline: inPlace, bindGroup: inPlaceGroup, workgroups: [Math.ceil(N / WG)] },
        { kind: "dispatch", pipeline: outOfPlace, bindGroup: outOfPlaceGroup, workgroups: [Math.ceil(N / WG)] },
      ],
      [
        { staging: stagingA, source: inPlaceBuf, sourceOffset: 0, length: PADDED, type: "f32" },
        { staging: stagingB, source: outBuf, sourceOffset: 0, length: PADDED, type: "f32" },
      ],
    );

    const live = (buffer: Float32Array) => Array.from(buffer.subarray(0, N));
    expect(live(updated as Float32Array)).toEqual(live(written as Float32Array));
    const worst = agree(
      (updated as Float32Array).subarray(0, N),
      axpy({ x: x.subarray(0, N), y: y0.subarray(0, N), a }),
    );
    expect(worst, worst ? JSON.stringify(worst) : undefined).toBeNull();

    // And the two entry points differ in exactly the way they are supposed to:
    // past `N`, the in-place buffer still holds the caller's own bytes, while
    // the out-of-place output buffer — which nobody wrote — is still zero.
    expect((updated as Float32Array)[N]).toBe(SENTINEL);
    expect((written as Float32Array)[N]).toBe(0);
  });
});
