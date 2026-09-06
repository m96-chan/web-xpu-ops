import { describe, expect, it } from "vitest";
import { agree, expectAgrees, gpuTest, kernel, useGpu } from "../../harness/index.js";
import { rope } from "./reference.js";
import { TRANSCENDENTAL, axesScenario } from "./testing.js";

/**
 * The `axes` entry point — multi-axis RoPE — against `ops/rope/reference.ts`.
 *
 * Its own file rather than more cases in `wgsl.test.ts`, for the dispatch
 * budget `testing.ts` documents (#68): four dispatches here, and the files
 * beside it stay at the sizes they were measured to survive at.
 *
 * What the *reference* is checked against — Z-Image's own implementation — is
 * `axes.test.ts`, which needs no GPU and so still runs on a machine with no
 * adapter. This file only has to show that the kernel computes what the
 * reference does.
 */
const code = kernel(import.meta.url, "axes");

// Z-Image's own geometry: head dim 128 split 32/48/48 at theta 256, with a
// token whose positions are all zero (the identity) and tokens that are not.
// 5 × 3 × 64 = 960 pairs, so the 256-wide workgroup is crossed three times and
// the tail is ragged.
const ZIMAGE = { axisDims: [32, 48, 48], thetaBase: 256, N: 5, numHeads: 3 } as const;
const zimage = axesScenario(code, {
  ...ZIMAGE,
  positions: Float32Array.from([0, 0, 0, 1, 0, 1, 2, 3, 5, 7, 1, 0, 3, 11, 4]),
});

// Unequal axes, and negative positions on two of them. Both halves are
// load-bearing: unequal dims are what tell `f32(axis_dims[axis])` from
// `f32(params.head_dim)` in the exponent, and a negative position read as u32
// becomes ~4.3e9 and rotates by an angle with nothing to do with -3.
// 9 × 8 × 8 = 576 pairs, ragged over the workgroup again.
const uneven = axesScenario(code, {
  axisDims: [4, 6, 6],
  thetaBase: 256,
  N: 9,
  numHeads: 8,
  positions: Float32Array.from(
    Array.from({ length: 9 }, (_, token) => [token, -3 * token, 40 - token]).flat(),
  ),
});

// One axis, checked against `rope` rather than against `ropeAxes`: on the GPU
// as on the host, the generalisation has to reproduce the op it generalises.
// The two kernels are different files, so this is a real comparison rather than
// a restatement — `axes.wgsl` reaches the same numbers with none of
// `kernel.wgsl`'s cache, scaling or head-range machinery.
const ONE = { axisDims: [16], thetaBase: 10000, N: 9, numHeads: 8 } as const;
const oneAxisInput = Float32Array.from({ length: 9 * 8 * 16 }, (_, i) => Math.sin(i * 0.29) * 2);
const oneAxis = axesScenario(
  code,
  { ...ONE, positions: Float32Array.from({ length: 9 }, (_, token) => token + 200) },
  oneAxisInput,
  rope({ input: oneAxisInput, N: 9, numHeads: 8, headDim: 16, posOffset: 200, thetaBase: 10000 }),
);

// Every position zero is the identity on every axis at once, so the whole
// tensor is a copy — and the input is nowhere zero (`+ 3` puts it in [1, 5]),
// so a channel the kernel never wrote reads back as zero and cannot pass for
// one it copied. This is what pins the *block offsets*: a pair the axis search
// assigned to no axis at all takes the fallback copy and would pass here, but
// it would also have to pass `zimage` above, where copying is wrong.
const still = Float32Array.from({ length: 5 * 3 * 128 }, (_, i) => Math.sin(i * 0.29) * 2 + 3);
const identity = axesScenario(
  code,
  { ...ZIMAGE, positions: new Float32Array(5 * 3) },
  still,
);

/**
 * What makes the dispatches above observable, asserted rather than assumed.
 *
 * No GPU, so it holds on a machine with no adapter — where the tests below skip
 * themselves and would otherwise leave nothing behind.
 */
describe("rope / axes / what the fixtures pin", () => {
  it("has expectations that moved, and one that is exactly the input", () => {
    // "Agrees with the reference" is worth nothing if the reference barely
    // moved: a kernel that copied its input would satisfy it. Measured on these
    // fixtures, the rotated cases move by more than 2 against the 1e-3 allowed.
    const moved = (expected: Float32Array, input: ArrayLike<number>) =>
      Math.max(...Array.from(input, (v, i) => Math.abs(expected[i]! - v)));
    expect(moved(zimage.expected, Array.from({ length: 5 * 3 * 128 }, (_, i) => Math.sin(i * 0.29) * 2)))
      .toBeGreaterThan(2);
    expect(moved(uneven.expected, Array.from({ length: 9 * 8 * 16 }, (_, i) => Math.sin(i * 0.29) * 2)))
      .toBeGreaterThan(2);
    expect(moved(oneAxis.expected, oneAxisInput)).toBeGreaterThan(2);

    // And the identity case has to be the input, all of it, or "the kernel
    // copied it" would be satisfied by a kernel that did anything at all.
    expect(Array.from(identity.expected.slice(0, still.length))).toEqual(Array.from(still));
  });
});

describe("rope / axes / wgsl", () => {
  useGpu();

  gpuTest("agrees with the reference at Z-Image's [32, 48, 48]", async (run) => {
    await expectAgrees(run, zimage.dispatch, [zimage.expected], TRANSCENDENTAL);
  });

  gpuTest("agrees with the reference on unequal axes and negative positions", async (run) => {
    await expectAgrees(run, uneven.dispatch, [uneven.expected], TRANSCENDENTAL);
  });

  gpuTest("reproduces 1-D rope when there is one axis", async (run) => {
    await expectAgrees(run, oneAxis.dispatch, [oneAxis.expected], TRANSCENDENTAL);
  });

  /**
   * A dispatch too wide for one row of workgroups.
   *
   * 65,535 is the ceiling on every backend measured (#211), and the caller that
   * found it is `examples/h3-video`'s decoder at 42 latent frames. The fold
   * reads `num_workgroups` rather than taking a uniform, so every existing
   * one-dimensional caller keeps working — this one re-dispatches the same
   * scenario over a deliberately narrow grid.
   */
  gpuTest("folds a two-dimensional dispatch back into one pair index", async (run) => {
    const tiles = zimage.dispatch.workgroups[0];
    const x = 2;
    const y = Math.ceil(tiles / x);
    expect(y).toBeGreaterThan(1);
    await expectAgrees(
      run,
      { ...zimage.dispatch, workgroups: [x, y] },
      [zimage.expected],
      TRANSCENDENTAL,
    );
  });

  gpuTest("copies the tensor when every position is zero", async (run) => {
    const [actual] = await run(identity.dispatch);
    // Bit-exact, not `TRANSCENDENTAL`: `cos(0)` and `sin(0)` are exactly 1 and
    // 0 on any hardware, so a copy that rounds is not a copy. The padding past
    // the data stays zero, which is the guard still doing its job.
    expect(agree(actual!, identity.expected, { abs: 0, rel: 0 })).toBeNull();
  });
});
