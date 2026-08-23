import { describe } from "vitest";
import { expectAgrees, gpuTest, kernel, params, useGpu } from "../../harness/index.js";
import { Q4_GROUP_SIZE, packQ4 } from "../matvec/index.js";
import { matmulQ4G128 } from "./reference.js";

/**
 * `matmulQ4G128`'s three targeted edge cases, split out of
 * `q4_g128.wgsl.test.ts` — same reason `matmulQ8`'s own edge file is split out
 * of its shape file (`ops/matmul/q8-edges.wgsl.test.ts`): fifteen `gpuTest`s
 * of a quantized GEMM in one process crashed this binding's GPU worker, and
 * 12/3 across two files did not.
 *
 * Two of the three are `q8-edges.wgsl.test.ts`'s cases carried over to the
 * narrower field width. The third has no q8 counterpart because q8 has no
 * group axis: it is the one that says a scale is read per *group of 128
 * contracted positions*, not per weight row.
 */
const code = kernel(import.meta.url, "q4_g128");

/** Must match `TILE` in `wgsl/q4_g128.wgsl`. */
const TILE = 16;

const wave = (n: number, k: number, phase = 0) =>
  Float32Array.from({ length: n }, (_, i) => Math.sin(i * k + phase) * 1.5);
/** Deterministic q4 codes, mixed sign, inside `[-7, 7]`. */
const codeWave = (n: number, k: number, phase = 0) =>
  Int32Array.from({ length: n }, (_, i) => Math.round(Math.sin(i * k + phase) * 7));
/** Strictly increasing and never 1 — a scale read from the wrong slot changes the answer, not just its magnitude. */
const scaleWave = (n: number) => Float32Array.from({ length: n }, (_, i) => 0.02 * (i + 1) + 0.007);

/** Same measured tolerance as `q4_g128.wgsl.test.ts` — see that file's own doc. */
const TOLERANCE = { rel: 1e-4, abs: 1e-3 };

describe("matmulQ4G128 / wgsl edges", () => {
  useGpu();

  gpuTest("does not read past the last packed word or last scale of a weight row", async (run) => {
    // Two buffers are padded past what the kernel is told to read, each with a
    // value nothing here could otherwise produce: the weight words (K = 19 is
    // 3 words per row, the last one five nibbles empty) and the scales
    // (K = 19 is one group per row, so a row's scale run is a single f32 and
    // an off-by-one lands on the next row's). Without the padding an
    // over-read of the last row falls outside the buffer, where WGSL's robust
    // access quietly returns zero and hides the mistake.
    //
    // `a` is deliberately *not* padded: an earlier version of `q8-edges`'s
    // equivalent padded `a` and then bound only a subarray of it, so the
    // padding was never uploaded and the test proved nothing (rule 1's
    // "コードを消しても通るテストは、観測点が間違っている"). `a`'s own K-tail
    // guard is covered by the ragged-K shapes in `q4_g128.wgsl.test.ts`.
    const N = 5;
    const M = 7;
    const K = 19;
    const SENTINEL_WORD = 0x77777777;
    const SENTINEL_SCALE = 1000;
    const a = wave(N * K, 0.41);
    const codes = codeWave(M * K, 0.23);
    const packedRows = packQ4({ codes, N: M, K });
    const wordsPerRow = Math.ceil(K / 8);
    const groupsPerRow = Math.ceil(K / Q4_GROUP_SIZE);
    const weight = Uint32Array.from({ length: M * wordsPerRow + 64 }, (_, i) =>
      i < M * wordsPerRow ? packedRows[i]! : SENTINEL_WORD,
    );
    const scaleRows = scaleWave(M * groupsPerRow);
    const scale = Float32Array.from({ length: M * groupsPerRow + 64 }, (_, i) =>
      i < M * groupsPerRow ? scaleRows[i]! : SENTINEL_SCALE,
    );
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
      [matmulQ4G128({ a, weight: packedRows, scale: scaleRows, N, M, K })],
      TOLERANCE,
    );
  });

  gpuTest("reads each group's own scale within a weight row, not the row's first", async (run) => {
    // The case q8 cannot have. K = 384 is exactly three groups. Every code in
    // the weight is 1 and every activation is 1, so each output element is
    // just the sum of that row's three group scales times 128 — and the three
    // scales are 1, 100 and 10000, so a kernel that reads only the first (or
    // that indexes the group with the wrong stride) is wrong by orders of
    // magnitude rather than by a rounding error. The two output features
    // carry the same codes and *different* scale runs, so a scale row stride
    // of 1 is caught by the same case.
    const N = 3;
    const M = 2;
    const K = 3 * Q4_GROUP_SIZE;
    const a = Float32Array.from({ length: N * K }, () => 1);
    const codes = Int32Array.from({ length: M * K }, () => 1);
    const weight = packQ4({ codes, N: M, K });
    const scale = Float32Array.from([1, 100, 10000, 2, 200, 20000]);
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
      [matmulQ4G128({ a, weight, scale, N, M, K })],
      TOLERANCE,
    );
  });

  gpuTest("does not write past the last output feature of a row", async (run) => {
    // M = TILE + 1: output is one column past one tile, so every lane in the
    // second tile's overhang would store into the *next row* of output rather
    // than off the end of the buffer — a live element, silently overwritten.
    // K = 130 puts a ragged second scale group under the same case.
    const N = 4;
    const M = TILE + 1;
    const K = Q4_GROUP_SIZE + 2;
    const a = wave(N * K, 0.9);
    const codes = codeWave(M * K, 0.31);
    const weight = packQ4({ codes, N: M, K });
    const scale = scaleWave(M * Math.ceil(K / Q4_GROUP_SIZE));
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
      [matmulQ4G128({ a, weight, scale, N, M, K })],
      TOLERANCE,
    );
  });
});
