import { describe, expect, it } from "vitest";
import { Q4_GROUP_SIZE, matvecQ4G128, packQ4, quantizeQ4G128 } from "../matvec/index.js";
import { matmulQ4G128 } from "./reference.js";

/**
 * `matmulQ4G128`'s CPU-only tests. `ops/matmul` has never had a
 * `reference.test.ts` — `matmul` and `matmulQ8` are checked by their kernels
 * agreeing with them — and for those two that is defensible, because the
 * reference is a triple loop anyone can read. This op has something those do
 * not: **a second reader of the same wire format**. `matvecQ4G128` already
 * defines what `[M, ceil(K/8)]` packed nibbles plus `[M, ceil(K/128)]` scales
 * mean, and if this file's GEMM disagrees with that GEMV, one of them is wrong
 * no matter how well its own kernel agrees with it (issue #149's own "同じ慣習
 * を二度選ばない" — the format is #137's, and this is the test that says so
 * rather than the doc claiming it).
 *
 * `ops/matmul/q4_g128.wgsl.test.ts` covers kernel-vs-reference agreement.
 */

const wave = (n: number, k: number, phase = 0) =>
  Float32Array.from({ length: n }, (_, i) => Math.sin(i * k + phase) * 1.5);

describe("matmulQ4G128", () => {
  it("computes the same numbers as matvecQ4G128, row by row", () => {
    // The cross-check that matters: the GEMM reads #137's format, not a second
    // format that happens to be spelled the same way. Three input rows against
    // the same weight, compared against three independent GEMV calls. K spans
    // three scale groups with the last one short, so a group index built from
    // the wrong dimension (or from `N` instead of `K`) cannot survive.
    const N = 3;
    const M = 5;
    const K = 300;
    const a = wave(N * K, 0.37);
    const matrix = Float32Array.from({ length: M * K }, (_, i) =>
      Math.sin(i * 2.1) * Math.pow(10, Math.floor((i % K) / Q4_GROUP_SIZE) % 3),
    );
    const { codes, scales } = quantizeQ4G128({ input: matrix, N: M, K });
    const weight = packQ4({ codes, N: M, K });

    const got = matmulQ4G128({ a, weight, scale: scales, N, M, K });
    for (let row = 0; row < N; row += 1) {
      const vector = a.slice(row * K, (row + 1) * K);
      const want = matvecQ4G128({ weight, scale: scales, vector, N: M, K });
      for (let col = 0; col < M; col += 1) {
        expect(got[row * M + col]!).toBeCloseTo(want[col]!, 3);
      }
    }
  });

  it("agrees with a hand-computed product", () => {
    // a = [[10, 20, 30, 40]], one weight row of codes [1, -2, 3, -4] with a
    // single group scale of 0.5:
    //   10 - 40 + 90 - 160 = -100, times 0.5 = -50.
    // The activation values are deliberately distinct so a nibble-order slip
    // changes the answer rather than permuting equal products.
    const a = Float32Array.from([10, 20, 30, 40]);
    const weight = packQ4({ codes: Int32Array.from([1, -2, 3, -4]), N: 1, K: 4 });
    const scale = Float32Array.from([0.5]);
    const out = matmulQ4G128({ a, weight, scale, N: 1, M: 1, K: 4 });
    expect(out).toHaveLength(1);
    expect(out[0]).toBeCloseTo(-50, 5);
  });

  it("gives each weight row its own run of group scales", () => {
    // Two output features with identical codes, two groups each; only the
    // scales differ. A scale indexed per row (`scale[m]`) instead of per row
    // and group, or with the row stride taken as 1, collapses the two columns
    // to the same number.
    const K = 2 * Q4_GROUP_SIZE;
    const M = 2;
    const codes = Int32Array.from({ length: M * K }, () => 1);
    const weight = packQ4({ codes, N: M, K });
    const scale = Float32Array.from([1, 2, 10, 20]);
    const a = Float32Array.from({ length: K }, () => 1);
    const out = matmulQ4G128({ a, weight, scale, N: 1, M, K });
    expect(out[0]).toBeCloseTo(Q4_GROUP_SIZE * (1 + 2), 3);
    expect(out[1]).toBeCloseTo(Q4_GROUP_SIZE * (10 + 20), 3);
  });

  it("returns zeros for an empty K rather than throwing", () => {
    // `matmul`'s own convention, inherited: `torch.mm` on [M, 0] @ [0, N]
    // returns zeros, and the empty sum is 0. Worth pinning here because this
    // op divides by the group size to size its scale array, and ceil(0/128)
    // is 0 — an off-by-one there would index a scale that does not exist.
    const out = matmulQ4G128({
      a: new Float32Array(0),
      weight: new Uint32Array(0),
      scale: new Float32Array(0),
      N: 3,
      M: 5,
      K: 0,
    });
    expect(out).toHaveLength(15);
    expect(Array.from(out).every((v) => v === 0)).toBe(true);
  });

  it("reads neither the padding nibbles of a ragged row nor the next row's words", () => {
    // K = 5: one packed word per row with three unused nibbles. A sentinel is
    // written into row 0's padding directly, bypassing packQ4, so this catches
    // a reference that unpacks all eight lanes and lets the tail reach the sum
    // — and row 1's codes differ from row 0's, so a word stride taken from K
    // rather than from ceil(K/8) also shows.
    const K = 5;
    const M = 2;
    const codes = Int32Array.from([1, 1, 1, 1, 1, 2, 2, 2, 2, 2]);
    const weight = packQ4({ codes, N: M, K });
    weight[0] = (weight[0]! | (0x7 << 20)) >>> 0; // lane 5 of row 0 = column 5
    const scale = Float32Array.from([1, 1]);
    const a = Float32Array.from([1, 1, 1, 1, 1]);
    const out = matmulQ4G128({ a, weight, scale, N: 1, M, K });
    expect(out[0]).toBeCloseTo(5, 6);
    expect(out[1]).toBeCloseTo(10, 6);
  });
});
