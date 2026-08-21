import { describe } from "vitest";
import { expectAgrees, gpuTest, kernel, params, useGpu } from "../../harness/index.js";
import { packQ8 } from "../matvec/index.js";
import { matmulQ8 } from "./reference.js";

/**
 * `matmulQ8`'s three targeted edge cases, split out of `q8.wgsl.test.ts` —
 * see that file's own doc for why: the twelve shape cases plus these three,
 * fifteen `gpuTest`s in one process, crashed this binding's GPU worker
 * (measured, issue #38's "GPUオブジェクト数の崖" family); either group alone
 * runs clean.
 */
const code = kernel(import.meta.url, "q8");

/** Must match `TILE` in `wgsl/q8.wgsl`. */
const TILE = 16;

const wave = (n: number, k: number, phase = 0) =>
  Float32Array.from({ length: n }, (_, i) => Math.sin(i * k + phase) * 1.5);
const codeWave = (n: number, k: number, phase = 0) =>
  Int32Array.from({ length: n }, (_, i) => Math.round(Math.sin(i * k + phase) * 100));
const scaleWave = (n: number) => Float32Array.from({ length: n }, (_, i) => 0.02 * (i + 1) + 0.007);

/** Same measured tolerance as `q8.wgsl.test.ts` — see that file's own doc. */
const TOLERANCE = { rel: 1e-4, abs: 6e-4 };

describe("matmulQ8 / wgsl edges", () => {
  useGpu();

  gpuTest("does not read past the last packed word of a weight row when K is ragged", async (run) => {
    // Without this case the K tail on the *weight* side cannot be observed at
    // all: every shape in `q8.wgsl.test.ts` pads `a`'s own K tail implicitly
    // (the loop bound is the same `K` for both operands), but a kernel that
    // computes `words_per_row` from the wrong field, or reads one packed word
    // too far per row, only shows up against a weight buffer with something
    // meaningful sitting right after each row's real words.
    //
    // K = 19 -> 5 words/row (19 % 4 = 3, last word half-empty). Weight is
    // packed tightly (`packQ8`'s own zero-fill for the unused lanes), then
    // padded with a sentinel word large enough that a leaked byte dwarfs the
    // true sum in any lane.
    //
    // `a` itself is *not* padded here — an earlier version of this test
    // built a padded, sentinel-filled `a` buffer, then bound and referenced
    // only `a.subarray(0, N * K)` of it, so the padding was never uploaded
    // and never observed by either side of the comparison (caught by rule
    // 1's "コードを消しても通るテストは、観測点が間違っている": deleting the
    // padding left this test green). `a`'s own K-tail read guard already has
    // real coverage elsewhere — `q8.wgsl.test.ts`'s ragged-K shapes (e.g.
    // `[TILE, TILE, TILE + 1]`) bind `a` at exactly its logical size, so an
    // over-read there lands on whatever the test runner's next allocation
    // holds, not a buffer this test controls; this test's own job is the
    // *weight* row's word count specifically, which is what the padding
    // below actually exercises.
    const N = 5;
    const M = 7;
    const K = 19;
    const SENTINEL_WORD = 0x7f7f7f7f;
    const a = wave(N * K, 0.41);
    const codes = codeWave(M * K, 0.23);
    const packedRows = packQ8({ codes, N: M, K });
    const wordsPerRow = Math.ceil(K / 4);
    const weight = Uint32Array.from({ length: M * wordsPerRow + 64 }, (_, i) =>
      i < M * wordsPerRow ? packedRows[i]! : SENTINEL_WORD,
    );
    const scale = scaleWave(M);
    await expectAgrees(
      run,
      {
        code,
        bindings: [
          { kind: "storage", data: a },
          { kind: "storage", data: weight },
          { kind: "storage", data: scale },
          { kind: "out", type: "f32", length: N * M },
          { kind: "uniform", data: params([["u32", N], ["u32", M], ["u32", K]]) },
        ],
        workgroups: [Math.ceil(M / TILE), Math.ceil(N / TILE)],
      },
      [matmulQ8({ a, weight: packedRows, scale, N, M, K })],
      TOLERANCE,
    );
  });

  gpuTest("does not write past the last output feature of a row", async (run) => {
    // The ragged-M shapes in `q8.wgsl.test.ts` already catch a missing column
    // guard, in the smallest form that shows what actually goes wrong: output
    // is one column past one tile (`M = TILE + 1`, not `2 * TILE - 1`), so
    // every lane in the second tile's overhang stores into the *next row* of
    // output rather than off the end of the buffer.
    const N = 4;
    const M = TILE + 1;
    const K = 8;
    const a = wave(N * K, 0.9);
    const codes = codeWave(M * K, 0.31);
    const weight = packQ8({ codes, N: M, K });
    const scale = scaleWave(M);
    await expectAgrees(
      run,
      {
        code,
        bindings: [
          { kind: "storage", data: a },
          { kind: "storage", data: weight },
          { kind: "storage", data: scale },
          { kind: "out", type: "f32", length: N * M },
          { kind: "uniform", data: params([["u32", N], ["u32", M], ["u32", K]]) },
        ],
        workgroups: [Math.ceil(M / TILE), Math.ceil(N / TILE)],
      },
      [matmulQ8({ a, weight, scale, N, M, K })],
      TOLERANCE,
    );
  });

  gpuTest("a row picking up a neighbouring output feature's scale changes the answer", async (run) => {
    // `scaleWave` (used throughout `q8.wgsl.test.ts`) is already strictly
    // increasing so this cannot pass by coincidence in the general shapes
    // there, but this case isolates it: two output features whose *codes*
    // are identical row-for-row, so the only way their outputs can differ is
    // if each reads its own scale.
    const N = 2;
    const M = 2;
    const K = 12;
    const a = wave(N * K, 0.5);
    const sharedCodes = codeWave(K, 0.27);
    const codes = Int32Array.from({ length: M * K }, (_, i) => sharedCodes[i % K]!);
    const weight = packQ8({ codes, N: M, K });
    const scale = Float32Array.from([0.03, 0.11]);
    await expectAgrees(
      run,
      {
        code,
        bindings: [
          { kind: "storage", data: a },
          { kind: "storage", data: weight },
          { kind: "storage", data: scale },
          { kind: "out", type: "f32", length: N * M },
          { kind: "uniform", data: params([["u32", N], ["u32", M], ["u32", K]]) },
        ],
        workgroups: [Math.ceil(M / TILE), Math.ceil(N / TILE)],
      },
      [matmulQ8({ a, weight, scale, N, M, K })],
      TOLERANCE,
    );
  });
});
