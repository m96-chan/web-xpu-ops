import { describe, expect, it } from "vitest";
import { matvecQ8, matvecQ8Ffn, matvecQ8Residual, packQ8 } from "./reference.js";

/**
 * `packQ8` / `matvecQ8` only: the questions a GPU dispatch cannot ask directly
 * — the exact bit layout of a packed word, and whether the reference round-
 * trips through it correctly. `ops/matvec/wgsl.test.ts` covers kernel-vs-
 * reference agreement; this covers the reference's own packing contract, which
 * the kernel is written against but which nothing forces it to get right by
 * accident.
 */

describe("packQ8", () => {
  it("packs four codes into one word, least-significant byte first", () => {
    // Codes chosen so each byte is distinguishable in the packed word: if lane
    // 0 and lane 3 were swapped, or any byte landed in the wrong shift, the
    // expected word below would not match.
    const codes = Int32Array.from([1, 2, 3, 4]);
    const packed = packQ8({ codes, N: 1, K: 4 });
    expect(packed).toHaveLength(1);
    // lane 0 -> bits 0..7, lane 1 -> bits 8..15, lane 2 -> 16..23, lane 3 -> 24..31
    const want = 1 | (2 << 8) | (3 << 16) | (4 << 24);
    expect(packed[0]).toBe(want >>> 0);
  });

  it("stores a negative code as its two's-complement byte, not a masked positive", () => {
    const codes = Int32Array.from([-1, -127, 127, 0]);
    const packed = packQ8({ codes, N: 1, K: 4 });
    const want = (0xff | (0x81 << 8) | (0x7f << 16) | (0 << 24)) >>> 0;
    expect(packed[0]).toBe(want);
  });

  it("zero-fills the unused lanes of a row's final word when K % 4 != 0", () => {
    // K=6: one full word (cols 0-3) and a half-full second word (cols 4-5).
    // Lanes 2 and 3 of the second word carry nothing from this row or the next.
    const codes = Int32Array.from([9, 9, 9, 9, 5, 6]);
    const packed = packQ8({ codes, N: 1, K: 6 });
    expect(packed).toHaveLength(2);
    const wantSecond = (5 | (6 << 8)) >>> 0; // lanes 2, 3 stay 0
    expect(packed[1]).toBe(wantSecond);
  });

  it("gives each row its own word run, independent of every other row", () => {
    // K=5 -> ceil(5/4) = 2 words per row. If row stride were computed from K
    // instead of from the word count, row 1 would start mid-word into row 0.
    const codes = Int32Array.from([1, 1, 1, 1, 1, 2, 2, 2, 2, 2]);
    const packed = packQ8({ codes, N: 2, K: 5 });
    expect(packed).toHaveLength(4);
    const wantRow1Word0 = (2 | (2 << 8) | (2 << 16) | (2 << 24)) >>> 0;
    expect(packed[2]).toBe(wantRow1Word0); // row 1's first word: codes 2,2,2,2
    expect(packed[3]).toBe(2); // row 1's second word: code 2, lane 0 (col 4)
  });
});

describe("matvecQ8", () => {
  it("agrees with a hand-computed dot product for a single row", () => {
    // codes = [1, -2, 3, -4], vector = [10, 20, 30, 40] — deliberately not
    // uniform, so a lane-order slip (either in packing or in unpacking)
    // changes which code multiplies which vector element and the sum comes out
    // wrong, rather than being invisible the way a uniform vector would make
    // any permutation of the same four products look.
    // dot = 1*10 - 2*20 + 3*30 - 4*40 = 10 - 40 + 90 - 160 = -100; scale 0.5: -50
    const codes = Int32Array.from([1, -2, 3, -4]);
    const weight = packQ8({ codes, N: 1, K: 4 });
    const scale = Float32Array.from([0.5]);
    const vector = Float32Array.from([10, 20, 30, 40]);
    const out = matvecQ8({ weight, scale, vector, N: 1, K: 4 });
    expect(out[0]).toBeCloseTo(-50, 5);
  });

  it("applies each row's own scale, not the first row's", () => {
    // Both rows have the same codes and see the same vector, so only the scale
    // distinguishes their outputs — a scale mix-up (wrong row, or the scale
    // applied once for the whole call) collapses the two to the same number.
    const codes = Int32Array.from([1, 1, 1, 1, 1, 1, 1, 1]);
    const weight = packQ8({ codes, N: 2, K: 4 });
    const scale = Float32Array.from([2, 5]);
    const vector = Float32Array.from([1, 1, 1, 1]);
    const out = matvecQ8({ weight, scale, vector, N: 2, K: 4 });
    expect(out[0]).toBeCloseTo(8, 6); // 4 * 2
    expect(out[1]).toBeCloseTo(20, 6); // 4 * 5
  });

  it("does not read the padding lanes of the last word when K % 4 != 0", () => {
    // K=5, so the second word has 3 used lanes and 1 padding lane. Pack a
    // sentinel into the padding lane directly (bypassing packQ8) to prove the
    // reference's `col < K` guard, not packQ8's zero-fill, is what protects it.
    const codes = Int32Array.from([1, 1, 1, 1, 1]);
    const weight = packQ8({ codes, N: 1, K: 5 });
    weight[1] = weight[1]! | (99 << 8); // lane 1 of word 1 = column 5, out of range
    const scale = Float32Array.from([1]);
    const vector = Float32Array.from([1, 1, 1, 1, 1, 1000]); // vector[5] would blow up the sum if read
    const out = matvecQ8({ weight, scale, vector, N: 1, K: 5 });
    expect(out[0]).toBeCloseTo(5, 6);
  });
});

/**
 * PR #127 review, item 1: `ops/matvec/q8_ffn.wgsl.test.ts`/`q8_residual.wgsl.test.ts`
 * only ever compare the *kernel* against *this file's own reference
 * functions* — so a bug that swaps which weight is "gate" and which is
 * "up" **inside `matvecQ8Ffn` itself** (silu applied to the wrong
 * projection) would still "agree" with the kernel doing the identical
 * swap, and neither test file would ever go red. These two `describe`
 * blocks give `matvecQ8Ffn`/`matvecQ8Residual` an independent ground truth
 * — hand-computed dot products, `silu`/the residual add written out as
 * plain arithmetic rather than called through any of this module's own
 * functions — the same shape `matvecQ8`'s own "agrees with a hand-computed
 * dot product" test above already provides for `matvecQ8`.
 */
describe("matvecQ8Ffn", () => {
  it("agrees with silu(gate)*up computed by hand from two distinct dot products", () => {
    // gate: codes [1,1,1,1], scale 1, vector [1,1,1,1] -> dot = 4, scaled 4
    // up:   codes [3,3,3,3], scale 1, same vector      -> dot = 12, scaled 12
    // Gate and up are deliberately different values (not a shared vector.length
    // coincidence): swapping them would compute silu(12)*4 ~= 47.9997 instead
    // of silu(4)*12 ~= 47.1367 below — a ~0.86 absolute difference, far outside
    // this test's own tolerance, so a gate/up mix-up inside matvecQ8Ffn itself
    // (not just in a caller) cannot pass this test by accident.
    const gateCodes = Int32Array.from([1, 1, 1, 1]);
    const upCodes = Int32Array.from([3, 3, 3, 3]);
    const vector = Float32Array.from([1, 1, 1, 1]);
    const weightGate = packQ8({ codes: gateCodes, N: 1, K: 4 });
    const weightUp = packQ8({ codes: upCodes, N: 1, K: 4 });
    const scaleGate = Float32Array.from([1]);
    const scaleUp = Float32Array.from([1]);

    const out = matvecQ8Ffn({ weightGate, scaleGate, weightUp, scaleUp, vector, N: 1, K: 4 });

    // silu(x) = x / (1 + e^-x) — the plain formula, not a call into
    // ops/activation or matvecQ8Ffn itself, so this expected value shares no
    // code path with what is under test.
    const gateDot = 4;
    const upDot = 12;
    const expected = (gateDot / (1 + Math.exp(-gateDot))) * upDot;
    expect(expected).toBeCloseTo(47.1367, 3); // sanity-check the hand computation itself
    expect(out[0]).toBeCloseTo(expected, 5);
  });

  it("applies each row's own gate/up scale and dot product, not row 0's", () => {
    // Row 0: gate dot 4 (codes all 1, scale 1) -> silu(4); up dot 4 (codes
    // all 1, scale 1) -> 4. Row 1: gate dot 8 (codes all 2, scale 1) ->
    // silu(8); up dot 2 (codes all 1, scale 0.5) -> 2. Distinct per-row
    // shapes on both projections, so a scale or row index applied once for
    // the whole call (instead of per row, per projection) changes both rows'
    // answers, not just one.
    const gateCodes = Int32Array.from([1, 1, 1, 1, 2, 2, 2, 2]);
    const upCodes = Int32Array.from([1, 1, 1, 1, 1, 1, 1, 1]);
    const vector = Float32Array.from([1, 1, 1, 1]);
    const weightGate = packQ8({ codes: gateCodes, N: 2, K: 4 });
    const weightUp = packQ8({ codes: upCodes, N: 2, K: 4 });
    const scaleGate = Float32Array.from([1, 1]);
    const scaleUp = Float32Array.from([1, 0.5]);

    const out = matvecQ8Ffn({ weightGate, scaleGate, weightUp, scaleUp, vector, N: 2, K: 4 });

    const silu = (x: number) => x / (1 + Math.exp(-x));
    expect(out[0]).toBeCloseTo(silu(4) * 4, 5);
    expect(out[1]).toBeCloseTo(silu(8) * 2, 5);
  });
});

describe("matvecQ8Residual", () => {
  it("agrees with residual + a hand-computed dot product", () => {
    // Same codes/scale/vector as matvecQ8's own "agrees with a hand-computed
    // dot product" test above: dot = 1*10 - 2*20 + 3*30 - 4*40 = -100,
    // scaled by 0.5 = -50. residual = 7, deliberately not 0 or a round
    // number that could coincide with the projection alone, so a residual
    // dropped on the floor (output === projection alone, -50) is
    // distinguishable from the correct -43.
    const codes = Int32Array.from([1, -2, 3, -4]);
    const weight = packQ8({ codes, N: 1, K: 4 });
    const scale = Float32Array.from([0.5]);
    const vector = Float32Array.from([10, 20, 30, 40]);
    const residual = Float32Array.from([7]);

    const out = matvecQ8Residual({ weight, scale, vector, residual, N: 1, K: 4 });
    expect(out[0]).toBeCloseTo(-43, 6); // -50 (the dot product) + 7 (residual)
  });

  it("adds each row's own residual and scale, not row 0's", () => {
    // Both rows share codes/vector (dot = 4 for both), so only scale and
    // residual distinguish the two rows' outputs — a mix-up (wrong row's
    // scale, or residual[0] broadcast to every row) collapses the two
    // outputs to the same number.
    const codes = Int32Array.from([1, 1, 1, 1, 1, 1, 1, 1]);
    const weight = packQ8({ codes, N: 2, K: 4 });
    const scale = Float32Array.from([2, 5]);
    const vector = Float32Array.from([1, 1, 1, 1]);
    const residual = Float32Array.from([100, 1000]);

    const out = matvecQ8Residual({ weight, scale, vector, residual, N: 2, K: 4 });
    expect(out[0]).toBeCloseTo(100 + 4 * 2, 6); // 108
    expect(out[1]).toBeCloseTo(1000 + 4 * 5, 6); // 1020
  });
});
