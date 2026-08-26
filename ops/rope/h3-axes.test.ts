/**
 * `ropeAxes` reproduces MiniMax-H3's 3D RoPE.
 *
 * Issue #200. `positions` took `Int32Array` until this model. H3's visual VAE
 * normalises each axis to `(-1, 1)` — `2 * (i + 0.5) / n - 1` — and multiplies
 * the angle by `2π`, so its positions are **fractional**. A rotation by a
 * fractional angle is the same rotation, and nothing in the op ever needed a
 * whole number; the alternative was a second kernel differing by a binding
 * type, which is two conventions waiting to drift.
 *
 * This file pins the *mapping* as well as the capability, because the mapping
 * is where a port goes wrong quietly — a rope applied to the wrong channel
 * pairs still returns a tensor of the right shape whose norm is unchanged.
 *
 * | | H3 | `ropeAxes` | |
 * | --- | --- | --- | --- |
 * | frequencies | `θ^-arange(0, 1, 2·n_dim/rot)` | `θ^-(2·pair/axis_dim)` | **already the same eight** |
 * | pairing | tile `[t8,h8,w8]` twice, rotate halves | adjacent inside each axis block | `H3_ROPE_PERMUTATION` |
 * | extent | 48 of 64 channels | the whole head | a fourth axis at position 0 |
 *
 * The permutation is applied to the **weights**, once, at conversion — the same
 * thing `permuteForRope` does for Anima. Doing it here, to the activations,
 * would be a per-forward shuffle of every query and key.
 */
import { describe, expect, it } from "vitest";
import { ropeAxes } from "./index.js";
import { H3_ROPE_CASE, H3_ROPE_PERMUTATION } from "./h3-cases.js";

const { numHeads, dimHead, rotDim, thetaBase } = H3_ROPE_CASE;
const input = Float32Array.from(H3_ROPE_CASE.input);
const want = Float32Array.from(H3_ROPE_CASE.want);
const N = want.length / (numHeads * dimHead);

/** `[.., dimHead]` from H3's channel order into `ropeAxes`'s, or back. */
function reorder(data: Float32Array, forward: boolean): Float32Array {
  const out = new Float32Array(data.length);
  for (let row = 0; row < data.length / dimHead; row += 1) {
    for (let c = 0; c < dimHead; c += 1) {
      const from = H3_ROPE_PERMUTATION[c]!;
      if (forward) out[row * dimHead + c] = data[row * dimHead + from]!;
      else out[row * dimHead + from] = data[row * dimHead + c]!;
    }
  }
  return out;
}

/** `[N, 4]`: H3's three axes, and a fourth pinned at zero. */
function positionsWithIdleAxis(): Float32Array {
  const out = new Float32Array(N * 4);
  for (let token = 0; token < N; token += 1) {
    for (let axis = 0; axis < 3; axis += 1) {
      out[token * 4 + axis] = H3_ROPE_CASE.positions[token * 3 + axis]!;
    }
  }
  return out;
}

const AXIS_DIMS = [16, 16, 16, 16];

describe("rope / axes / MiniMax-H3", () => {
  it("the fixture is the geometry H3's config states", () => {
    // 32 heads of 64 at `rope_dim_ratio: 0.75` is 48 rotated channels over
    // three axes; the fixture uses two heads so it stays readable, but the head
    // geometry is the model's. A fixture regenerated against a different
    // configuration would silently weaken every comparison below.
    expect(dimHead).toBe(64);
    expect(rotDim).toBe(48);
    expect(thetaBase).toBe(100);
    expect(AXIS_DIMS.reduce((a, b) => a + b, 0)).toBe(dimHead);
    expect(N).toBe(H3_ROPE_CASE.patchDims.reduce((a, b) => a * b, 1));
  });

  it("the positions are fractional, which is the whole reason this exists", () => {
    // If they were whole numbers the `Int32Array` binding would have served and
    // none of this would be needed. Asserted so that a regenerated fixture
    // cannot quietly become the easy case.
    //
    // **Not all of them.** `2 * (i + 0.5) / n - 1` is exactly 0 at the middle of
    // an odd-length axis — the fixture's `H = 3` has one — and 0 is an integer.
    // The first version of this test asserted every position was fractional and
    // failed on that, which is the fixture being right and the assertion being
    // careless.
    // The count is exact rather than a ratio: with `[T, H, W] = [2, 3, 4]` the
    // only whole positions are the middle of the odd axis, once per `T * W`
    // token — eight of seventy-two. Stating the structure means a regenerated
    // fixture with different dims fails here and is looked at, instead of
    // sliding under a threshold.
    const whole = [...H3_ROPE_CASE.positions].filter((v) => Number.isInteger(v));
    const [T, H, W] = H3_ROPE_CASE.patchDims;
    const oddAxes = H3_ROPE_CASE.patchDims.filter((n) => n % 2 === 1).length;
    expect(oddAxes).toBe(1);
    expect(whole.length).toBe(T! * W!);
    expect(H! % 2).toBe(1);
    expect(whole.every((v) => v === 0)).toBe(true);
    expect(H3_ROPE_CASE.positions.length - whole.length).toBe(64);
  });

  it("reproduces H3's rope, given the channel permutation", () => {
    const got = reorder(
      ropeAxes({
        input: reorder(input, true),
        N,
        numHeads,
        axisDims: AXIS_DIMS,
        positions: positionsWithIdleAxis(),
        thetaBase,
      }),
      false,
    );

    let worst = 0;
    for (let i = 0; i < want.length; i += 1) worst = Math.max(worst, Math.abs(got[i]! - want[i]!));
    // Measured: **2.384e-7**, which is f32 rounding on values of order 1 — the
    // fixture is written to seven decimals, so this is as close as the
    // comparison can get. 1e-6 is four times that, and every wrong mapping
    // tried against it (the permutation dropped, the idle axis given a real
    // position, the halves paired the other way) moves values by 1e-1.
    expect(worst).toBeLessThan(1e-6);
  });

  it("the last sixteen channels pass through untouched", () => {
    // The fourth axis is the identity, so `rope_dim_ratio < 1` costs no branch.
    // Checked against the *input* rather than against `want`: comparing to the
    // golden would pass even if both were wrong in the same way.
    const got = reorder(
      ropeAxes({
        input: reorder(input, true),
        N,
        numHeads,
        axisDims: AXIS_DIMS,
        positions: positionsWithIdleAxis(),
        thetaBase,
      }),
      false,
    );
    for (let row = 0; row < N * numHeads; row += 1) {
      for (let c = rotDim; c < dimHead; c += 1) {
        expect(got[row * dimHead + c]).toBe(input[row * dimHead + c]);
      }
    }
  });

  it("the permutation is a permutation", () => {
    // A duplicate entry would drop a channel and copy another, which every
    // comparison above would still mostly pass: 63 of 64 channels would be
    // right.
    expect(new Set(H3_ROPE_PERMUTATION).size).toBe(dimHead);
    expect(Math.min(...H3_ROPE_PERMUTATION)).toBe(0);
    expect(Math.max(...H3_ROPE_PERMUTATION)).toBe(dimHead - 1);
    // The unrotated tail is fixed: only the first 48 are shuffled.
    for (let c = rotDim; c < dimHead; c += 1) expect(H3_ROPE_PERMUTATION[c]).toBe(c);
  });
});
