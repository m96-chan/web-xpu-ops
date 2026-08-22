import { describe, expect, it } from "vitest";
import { Q4_GROUP_SIZE, matvec, matvecQ4G128, packQ4, quantizeQ4G128 } from "./reference.js";

/**
 * The q4 format's own CPU tests — the questions a GPU dispatch cannot ask
 * (the exact nibble layout of a packed word, which scale a column belongs to,
 * what the quantizer rounds a tie to), kept in their own file rather than
 * added to `reference.test.ts` so that a red here names the q4 contract and
 * nothing else.
 *
 * `ops/matvec/q4_g128.wgsl.test.ts` covers kernel-vs-reference agreement. That
 * comparison cannot catch a reference that packs and unpacks its *own* wrong
 * layout consistently, which is what this file is for — the same reasoning
 * `reference.test.ts`'s own header gives for `packQ8`.
 */

describe("packQ4", () => {
  it("packs eight codes into one word, least-significant nibble first", () => {
    // Every code distinct, so any lane permutation changes the word.
    const codes = Int32Array.from([1, 2, 3, 4, 5, 6, 7, 0]);
    const packed = packQ4({ codes, N: 1, K: 8 });
    expect(packed).toHaveLength(1);
    // lane 0 -> bits 0..3, lane 1 -> bits 4..7, ... lane 7 -> bits 28..31
    let want = 0;
    for (let lane = 0; lane < 8; lane += 1) want |= (codes[lane]! & 0xf) << (lane * 4);
    expect(packed[0]).toBe(want >>> 0);
    expect(packed[0]).toBe(0x07654321);
  });

  it("stores a negative code as its two's-complement nibble, not a masked positive", () => {
    // -1 is 0xf, -7 is 0x9. A quantizer output of -7 that packed as 7 would be
    // a sign flip on the largest code in the format.
    const codes = Int32Array.from([-1, -7, 7, 0, -2, 2, -6, 6]);
    const packed = packQ4({ codes, N: 1, K: 8 });
    const nibbles = [0xf, 0x9, 0x7, 0x0, 0xe, 0x2, 0xa, 0x6];
    let want = 0;
    for (let lane = 0; lane < 8; lane += 1) want |= nibbles[lane]! << (lane * 4);
    expect(packed[0]).toBe(want >>> 0);
  });

  it("zero-fills the unused lanes of a row's final word when K % 8 != 0", () => {
    const codes = Int32Array.from([1, 1, 1, 1, 1, 1, 1, 1, 5, 6]);
    const packed = packQ4({ codes, N: 1, K: 10 });
    expect(packed).toHaveLength(2);
    expect(packed[1]).toBe((5 | (6 << 4)) >>> 0); // lanes 2..7 stay 0
  });

  it("gives each row its own word run, independent of every other row", () => {
    // K=9 -> ceil(9/8) = 2 words per row. A row stride computed from K instead
    // of from the word count would start row 1 mid-word into row 0.
    const codes = Int32Array.from([
      1, 1, 1, 1, 1, 1, 1, 1, 1,
      2, 2, 2, 2, 2, 2, 2, 2, 2,
    ]);
    const packed = packQ4({ codes, N: 2, K: 9 });
    expect(packed).toHaveLength(4);
    expect(packed[2]).toBe(0x22222222); // row 1's first word
    expect(packed[3]).toBe(2); // row 1's second word: one code, lane 0
  });
});

describe("quantizeQ4G128", () => {
  it("gives each group of 128 columns its own absmax scale", () => {
    // Two groups in one row with absmaxes an order of magnitude apart: a
    // per-*row* scale (the thing #137 measured as 3x worse on real logits)
    // would give the small group codes near zero instead of near ±7.
    const K = 2 * Q4_GROUP_SIZE;
    const input = new Float32Array(K);
    for (let col = 0; col < Q4_GROUP_SIZE; col += 1) input[col] = col === 0 ? 7 : 1;
    for (let col = Q4_GROUP_SIZE; col < K; col += 1) input[col] = col === Q4_GROUP_SIZE ? 70 : 10;
    const { codes, scales } = quantizeQ4G128({ input, N: 1, K });
    expect(scales).toHaveLength(2);
    expect(scales[0]).toBeCloseTo(1, 6); // 7 / 7
    expect(scales[1]).toBeCloseTo(10, 6); // 70 / 7
    expect(codes[0]).toBe(7);
    expect(codes[1]).toBe(1);
    expect(codes[Q4_GROUP_SIZE]).toBe(7);
    expect(codes[Q4_GROUP_SIZE + 1]).toBe(1);
  });

  it("scales a group by absmax/7, not absmax/8 — the [-7, 7] range #137 chose", () => {
    // The whole point of decision 2 in issue #137: `[-8, 7]` (Q4_0's range)
    // measured *better* on weight RMS error and flipped the argmax in 4/4
    // cases. A row whose absmax sits at exactly 8 tells the two apart: with
    // absmax/7 the extreme column is code 7 and dequantizes back to 8 exactly;
    // with Q4_0's `d = max / -8` it would be code -8 (or 8 clipped) instead.
    const input = Float32Array.from([8, -8, 4, 0]);
    const { codes, scales } = quantizeQ4G128({ input, N: 1, K: 4 });
    expect(scales[0]).toBeCloseTo(8 / 7, 6);
    expect(codes[0]).toBe(7);
    expect(codes[1]).toBe(-7);
    expect(Math.min(...codes)).toBeGreaterThanOrEqual(-7);
    expect(Math.max(...codes)).toBeLessThanOrEqual(7);
  });

  it("rounds ties toward +Infinity, as `quantize`/`quant_common.py` do", () => {
    // absmax 7 makes the scale and its inverse exactly 1, so the row's own
    // values sit on the rounding boundary directly. Math.round(2.5) is 3 and
    // Math.round(-2.5) is -2 — not round-half-away-from-zero, and not
    // banker's rounding, which numpy would give. Pinned here because
    // `llm/tools/quant_common.py` has to reproduce it byte-for-byte to make
    // a converted checkpoint match this reference (rule 7).
    const input = Float32Array.from([7, 2.5, -2.5, 0.5, -0.5]);
    const { codes, scales } = quantizeQ4G128({ input, N: 1, K: 5 });
    expect(scales[0]).toBeCloseTo(1, 6);
    // Math.round(-0.5) is -0; an Int32Array stores that as 0, which is why the
    // last expectation reads 0 and not -0.
    expect(Array.from(codes)).toEqual([7, 3, -2, 1, 0]);
  });

  it("gives an all-zero group scale 1 rather than dividing by zero", () => {
    const input = new Float32Array(4);
    const { codes, scales } = quantizeQ4G128({ input, N: 1, K: 4 });
    expect(scales[0]).toBe(1);
    expect(Array.from(codes)).toEqual([0, 0, 0, 0]);
  });

  it("sizes the scale array [N, ceil(K/128)] and never pools two rows into one group", () => {
    // K = 300 -> 3 groups per row (128, 128, 44). The last group is short and
    // must take its absmax from its own 44 columns, not from a full 128 that
    // would run into the next row.
    const N = 2;
    const K = 300;
    const input = new Float32Array(N * K);
    input[0] = 1; // row 0, group 0
    input[K - 1] = 100; // row 0, group 2 (the short one)
    input[K] = 5; // row 1, group 0
    const { scales } = quantizeQ4G128({ input, N, K });
    expect(scales).toHaveLength(N * 3);
    expect(scales[0]).toBeCloseTo(1 / 7, 6);
    expect(scales[1]).toBe(1); // all-zero group
    expect(scales[2]).toBeCloseTo(100 / 7, 6);
    expect(scales[3]).toBeCloseTo(5 / 7, 6);
  });
});

describe("matvecQ4G128", () => {
  it("agrees with a hand-computed dot product for a single row", () => {
    // codes [1, -2, 3, -4], vector [10, 20, 30, 40], one group, scale 0.5.
    // dot = 10 - 40 + 90 - 160 = -100; scaled: -50. The vector is deliberately
    // non-uniform so a nibble-order slip changes the answer instead of
    // permuting equal products.
    const codes = Int32Array.from([1, -2, 3, -4]);
    const weight = packQ4({ codes, N: 1, K: 4 });
    const scale = Float32Array.from([0.5]);
    const vector = Float32Array.from([10, 20, 30, 40]);
    expect(matvecQ4G128({ weight, scale, vector, N: 1, K: 4 })[0]).toBeCloseTo(-50, 5);
  });

  it("applies each group's own scale, not the row's first one", () => {
    // K = 256: two groups, identical codes and vector, scales 1 and 1000.
    // A kernel or reference that reads scale[row] instead of scale[row, group]
    // gets 256 instead of 128 * 1001.
    const K = 2 * Q4_GROUP_SIZE;
    const codes = Int32Array.from({ length: K }, () => 1);
    const weight = packQ4({ codes, N: 1, K });
    const scale = Float32Array.from([1, 1000]);
    const vector = Float32Array.from({ length: K }, () => 1);
    const out = matvecQ4G128({ weight, scale, vector, N: 1, K });
    expect(out[0]).toBeCloseTo(Q4_GROUP_SIZE * 1 + Q4_GROUP_SIZE * 1000, 3);
  });

  it("gives each row its own run of group scales", () => {
    // Two rows, same codes and vector; only the scales differ, and each row
    // owns ceil(K/128) = 2 of them. A row stride of 1 (per-row scale) or of K
    // would read row 1's answer out of row 0's scales.
    const K = 2 * Q4_GROUP_SIZE;
    const codes = Int32Array.from({ length: 2 * K }, () => 1);
    const weight = packQ4({ codes, N: 2, K });
    const scale = Float32Array.from([1, 2, 10, 20]);
    const vector = Float32Array.from({ length: K }, () => 1);
    const out = matvecQ4G128({ weight, scale, vector, N: 2, K });
    expect(out[0]).toBeCloseTo(Q4_GROUP_SIZE * (1 + 2), 3);
    expect(out[1]).toBeCloseTo(Q4_GROUP_SIZE * (10 + 20), 3);
  });

  it("does not read the padding nibbles of the last word when K % 8 != 0", () => {
    // K=5, so the last word has 3 unused lanes. A sentinel is written into one
    // of them directly, bypassing packQ4, so this tests the reference's own
    // `col < K` bound rather than packQ4's zero-fill.
    const codes = Int32Array.from([1, 1, 1, 1, 1]);
    const weight = packQ4({ codes, N: 1, K: 5 });
    weight[0] = (weight[0]! | (0x7 << 20)) >>> 0; // lane 5 = column 5, out of range
    const scale = Float32Array.from([1]);
    const vector = Float32Array.from([1, 1, 1, 1, 1, 1000]);
    expect(matvecQ4G128({ weight, scale, vector, N: 1, K: 5 })[0]).toBeCloseTo(5, 6);
  });

  it("round-trips a quantized matrix to within the format's own error bound", () => {
    // The pipeline this format exists for: quantizeQ4G128 -> packQ4 ->
    // matvecQ4G128, against plain `matvec` on the unquantized weight. The
    // bound is the format's, not a fitted number: every weight is off by at
    // most half a step (scale/2 = absmax/14), so one row's error is at most
    // sum_k |vector[k]| * scale(group of k) / 2. Asserting against that
    // computed bound rather than a constant keeps this test honest if the
    // shapes below change.
    const N = 3;
    const K = 300;
    const matrix = Float32Array.from({ length: N * K }, (_, i) => Math.sin(i * 0.37) * (1 + (i % 7)));
    const vector = Float32Array.from({ length: K }, (_, i) => Math.cos(i * 0.11) * 1.5);
    const { codes, scales } = quantizeQ4G128({ input: matrix, N, K });
    const weight = packQ4({ codes, N, K });
    const got = matvecQ4G128({ weight, scale: scales, vector, N, K });
    const want = matvec({ matrix, vector, M: N, K });
    const groupsPerRow = Math.ceil(K / Q4_GROUP_SIZE);
    for (let row = 0; row < N; row += 1) {
      let bound = 0;
      for (let col = 0; col < K; col += 1) {
        bound += Math.abs(vector[col]!) * scales[row * groupsPerRow + Math.floor(col / Q4_GROUP_SIZE)]! / 2;
      }
      expect(Math.abs(got[row]! - want[row]!)).toBeLessThanOrEqual(bound);
    }
  });
});
