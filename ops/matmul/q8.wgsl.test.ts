import { describe } from "vitest";
import { expectAgrees, gpuTest, kernel, params, useGpu } from "../../harness/index.js";
import { packQ8 } from "../matvec/index.js";
import { matmulQ8 } from "./reference.js";

/**
 * `matmulQ8`'s own file rather than a shared `wgsl.test.ts` — same reasoning
 * as `ops/matvec/q8.wgsl.test.ts` (issue #38: `scripts/test.mjs` gives each
 * test file its own process, but that only helps *between* files, not within
 * one). The edge-case tests (`q8-edges.wgsl.test.ts`) are a *second* split
 * beyond that, measured rather than assumed (rule 1/2, "GPUオブジェクト数の
 * 崖"): this file's twelve shape cases plus those three edge cases (fifteen
 * `gpuTest`s total, each with a 5-binding dispatch — one more binding than
 * plain `matmul`'s own four, `weight`+`scale` where that kernel has one `b`)
 * crashed this binding's GPU worker (`Fatal glibc error ... tpp.c`) when run
 * together in one process; the same fifteen split 12/3 across two files, or
 * either group alone, ran clean every time it was tried. Plain `matmul`'s
 * own `wgsl.test.ts` gets away with fourteen 4-binding dispatches in one
 * file — this op's extra binding per dispatch is enough to move the cliff.
 */
const code = kernel(import.meta.url, "q8");

/** Must match `TILE` in `wgsl/q8.wgsl`; the shapes below are chosen around it. */
const TILE = 16;

const wave = (n: number, k: number, phase = 0) =>
  Float32Array.from({ length: n }, (_, i) => Math.sin(i * k + phase) * 1.5);
/** Deterministic int8 codes, mixed sign, within [-100, 100] — `ops/matvec/q8.wgsl.test.ts#codeWave`, copied (rule 2). */
const codeWave = (n: number, k: number, phase = 0) =>
  Int32Array.from({ length: n }, (_, i) => Math.round(Math.sin(i * k + phase) * 100));
/** Strictly increasing and never 1 — a row picking up its neighbour's scale changes the answer, not just its magnitude. */
const scaleWave = (n: number) => Float32Array.from({ length: n }, (_, i) => 0.02 * (i + 1) + 0.007);

/**
 * Widened past `ops/matmul/wgsl.test.ts`'s own `TOLERANCE` and measured, not
 * assumed (rule 2/9): codes here run to ±100 against that file's operand
 * values of order 1 — a term in this dot product is roughly two orders of
 * magnitude larger, the same reasoning `ops/matvec/q8.wgsl.test.ts`'s own
 * widened tolerance gives. Instrumented with the tolerance forced to zero,
 * the worst observed relative difference across the shapes below is
 * 4.1e-5 (at N=33 M=47 K=65), and the worst absolute difference is 3.2e-4
 * (at N=1 M=64 K=300, a single deep-K row). `rel: 1e-4` and `abs: 6e-4` both
 * keep roughly 1.5-2x headroom over those measured worsts.
 */
const TOLERANCE = { rel: 1e-4, abs: 6e-4 };

// Same shape catalogue as `ops/matmul/wgsl.test.ts`, for the same reason: a
// tiled kernel is fast and wrong on shapes that do not divide the tile, and a
// benchmark never notices. K is additionally exercised against the packed
// weight's own unit (a word = 4 columns), inherited automatically since K
// itself is already ragged in several of these.
const SHAPES: [N: number, M: number, K: number, why: string][] = [
  [1, 1, 1, "one element"],
  [TILE, TILE, TILE, "exactly one tile in every dimension"],
  [TILE + 1, TILE, TILE, "N tail only"],
  [TILE, TILE + 1, TILE, "M tail only"],
  [TILE, TILE, TILE + 1, "K tail only, one packed word past a tile"],
  [TILE - 3, TILE - 5, TILE - 7, "every dimension shorter than one tile"],
  [17, 19, 23, "all three ragged, two tiles each"],
  [33, 47, 65, "all three ragged, several tiles each"],
  [1, 64, 300, "a single row against a deep K"],
  [64, 1, 300, "a single output feature against a deep K"],
  [4, 5, 300, "many K tiles with a ragged last one"],
  [3, 5, 0, "empty K, where the sum is zero"],
];

describe("matmulQ8 / wgsl", () => {
  useGpu();

  for (const [N, M, K, why] of SHAPES) {
    gpuTest(`agrees with the reference at N=${N} M=${M} K=${K} — ${why}`, async (run) => {
      const a = wave(N * K, 0.37);
      const codes = codeWave(M * K, 0.19);
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
  }
});
