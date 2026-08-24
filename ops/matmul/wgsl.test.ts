import { describe } from "vitest";
import { expectAgrees, gpuTest, kernel, params, useGpu } from "../../harness/index.js";
import { matmul } from "./reference.js";
import { matmulGrid } from "./index.js";

const code = kernel(import.meta.url);

/** Must match `TILE` in `wgsl/kernel.wgsl`; the shapes below are chosen around it. */
// The workgroup tile, so the ragged shapes below stay chosen *around* it. It
// used to be 16 in both dimensions; the kernel now covers 64 rows by 128
// columns per workgroup, and a catalogue built for 16 would stop probing any
// boundary at all.
const TILE = 64;
const TILE_N = 128;

const wave = (n: number, k: number, phase = 0) =>
  Float32Array.from({ length: n }, (_, i) => Math.sin(i * k + phase) * 1.5);

/**
 * Looser on absolute than the harness default, and measured rather than widened
 * until green.
 *
 * The reference sums in f64; any f32 kernel sums in f32, and on these inputs the
 * two disagree before a GPU is involved at all. Simulating the same sequential
 * f32 accumulation in JS (`Math.fround` per multiply and per add) and comparing
 * it to the f64 reference over the shapes below gives, at worst:
 *
 *   rel 2.2e-3 at M=33 N=47 K=65   (65 terms of magnitude ~1 cancelling to ~5e-4,
 *                                   where relative error stops meaning anything)
 *   abs 1.24e-5 at M=4 N=5 K=300   (a 300-term sum)
 *
 * The kernel measures slightly better than that simulation — rel 7.9e-4 and
 * abs 1.14e-5 at those same two shapes — so it is not adding error of its own.
 *
 * `abs` is therefore the measure that has to carry the deep-K cases, and 2e-5 is
 * 1.75x the worst deviation actually observed and 1.6x the floor a correct f32
 * accumulation cannot go below. Anything that breaks the tiling is wrong by ~1
 * or more in absolute terms, five orders of magnitude clear of this — checked by
 * mutation, not assumed.
 */
const TOLERANCE = { rel: 1e-5, abs: 2e-5 };

// A tiled kernel is fast and wrong on the shapes that do not divide by the tile,
// and a benchmark never notices. So every tail is covered on its own — M, N and
// K each ragged while the other two are exact — and then all three at once, at
// two sizes, because the three failures are different bugs:
//
//   N tail — an unguarded store writes C[row][col] for col >= N, which is
//            C[row + 1][col - N]: a live element, silently clobbered.
//   M tail — an unguarded store goes past the end of C, which this driver
//            discards, so it is the tail these shapes cannot catch.
//   K tail — a dropped last partial tile, caught here; a tile filled from past
//            the end of the operands is not, and needs the padded-buffer test
//            below to be visible at all.
const SHAPES: [M: number, N: number, K: number, why: string][] = [
  [1, 1, 1, "one element"],
  [TILE, TILE_N, TILE, "exactly one tile in every dimension"],
  [TILE + 1, TILE, TILE, "M tail only"],
  [TILE, TILE_N + 1, TILE, "N tail only"],
  [TILE, TILE, TILE + 1, "K tail only"],
  [TILE - 3, TILE - 5, TILE - 7, "every dimension shorter than one tile"],
  [17, 19, 23, "all three ragged, two tiles each"],
  [33, 47, 65, "all three ragged, several tiles each"],
  [1, 64, 300, "a single row against a deep K"],
  [64, 1, 300, "a single column against a deep K"],
  [4, 5, 300, "many K tiles with a ragged last one"],
  [3, 5, 0, "empty K, where the sum is zero (torch.mm agrees)"],
];

describe("matmul / wgsl", () => {
  useGpu();

  for (const [M, N, K, why] of SHAPES) {
    gpuTest(`agrees with the reference at M=${M} N=${N} K=${K} — ${why}`, async (run) => {
      const a = wave(M * K, 0.37);
      const b = wave(K * N, 0.11, 0.5);
      await expectAgrees(
        run,
        {
          code,
          bindings: [
            { kind: "storage", data: a },
            { kind: "storage", data: b },
            { kind: "out", type: "f32", length: M * N },
            { kind: "uniform", data: params([["u32", M], ["u32", N], ["u32", K]]) },
          ],
          workgroups: matmulGrid(M, N),
        },
        [matmul({ a, b, M, N, K })],
        TOLERANCE,
      );
    });
  }

  gpuTest("does not read past the last row of B when K is ragged", async (run) => {
    // Without this case the K tail cannot be observed at all on this device, and
    // a kernel with no tail handling whatsoever passes every shape above.
    //
    // Measured, not assumed: a storage read past the end of a buffer returns 0
    // here (probed with a 8-element buffer read 16 wide — elements 8..15 came
    // back 0). Every read past B's last row is past the end of B, because B is
    // exactly K*N long, so the junk a broken kernel picks up is multiplied into
    // the sum as zero and the wrong answer is silently right.
    //
    // WGSL does not promise that — an out-of-bounds read may return any value
    // within the buffer — so the guard is real, and the way to see it is to give
    // it a buffer with something after the operand. Both are padded here with a
    // value big enough that a single leaked term dwarfs the true sum.
    const M = 5;
    const N = 7;
    const K = 19; // 19 % 16 = 3, so the last K tile is 13 lanes of padding
    const PAD = 1e3;
    const a = new Float32Array(M * K + 4 * TILE).fill(PAD);
    const b = new Float32Array(K * N + 4 * TILE).fill(PAD);
    a.set(wave(M * K, 0.41));
    b.set(wave(K * N, 0.23, 0.75));
    await expectAgrees(
      run,
      {
        code,
        bindings: [
          { kind: "storage", data: a },
          { kind: "storage", data: b },
          { kind: "out", type: "f32", length: M * N },
          { kind: "uniform", data: params([["u32", M], ["u32", N], ["u32", K]]) },
        ],
        workgroups: matmulGrid(M, N),
      },
      [matmul({ a: a.subarray(0, M * K), b: b.subarray(0, K * N), M, N, K })],
      TOLERANCE,
    );
  });

  gpuTest("does not write past the last column of a row", async (run) => {
    // The ragged-N shapes above already catch a missing column guard. This says
    // the same thing in the smallest form that shows what actually goes wrong:
    // C is one column short of two tiles, so every lane in the overhang stores
    // into the *next row* of C rather than off the end of the buffer, and a
    // failure here reads as one wrong element per row instead of noise.
    const M = 4;
    const N = TILE + 1;
    const K = 8;
    const a = wave(M * K, 0.9);
    const b = wave(K * N, 1.3, 0.25);
    await expectAgrees(
      run,
      {
        code,
        bindings: [
          { kind: "storage", data: a },
          { kind: "storage", data: b },
          { kind: "out", type: "f32", length: M * N },
          { kind: "uniform", data: params([["u32", M], ["u32", N], ["u32", K]]) },
        ],
        workgroups: matmulGrid(M, N),
      },
      [matmul({ a, b, M, N, K })],
      TOLERANCE,
    );
  });
});
