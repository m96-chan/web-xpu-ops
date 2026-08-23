import { describe } from "vitest";
import { expectAgrees, gpuTest, kernel, params, useGpu } from "../../harness/index.js";
import { Q4_GROUP_SIZE, packQ4, quantizeQ4G128 } from "../matvec/index.js";
import { matmulQ4G128 } from "./reference.js";

/**
 * `matmulQ4G128`'s shape cases. Its three targeted edge cases live in
 * `q4_g128-edges.wgsl.test.ts`, split the same 12/3 way `matmulQ8`'s own two
 * files are — see `ops/matmul/q8.wgsl.test.ts`'s header: fifteen `gpuTest`s of
 * that op in one process crashed this binding's GPU worker, and either group
 * alone ran clean. This op has the same binding count and one more storage
 * read per lane, so the split is inherited rather than re-derived.
 */
const code = kernel(import.meta.url, "q4_g128");

/** Must match `TILE` in `wgsl/q4_g128.wgsl`; the shapes below are chosen around it. */
const TILE = 16;

const wave = (n: number, k: number, phase = 0) =>
  Float32Array.from({ length: n }, (_, i) => Math.sin(i * k + phase) * 1.5);

/**
 * A weight matrix with per-128-column magnitude bands, so the group scales in
 * one row differ by two orders of magnitude and a kernel reading the wrong
 * group is off by a factor of ten rather than by a rounding error. The
 * frequency is 2.1 rather than something slower so that even a K of 1 carries
 * a mix of signs across rows — `ops/matvec/q4_g128.wgsl.test.ts` found the
 * hard way that a slow sine leaves the smallest shapes with no negative codes
 * at all, and therefore blind to sign extension.
 */
const bandedWeight = (M: number, K: number) =>
  Float32Array.from({ length: M * K }, (_, i) => {
    const row = Math.floor(i / K);
    const band = Math.floor((i % K) / Q4_GROUP_SIZE) % 3;
    return Math.sin(i * 2.1 + row) * Math.pow(10, band) * (1 + row);
  });

/**
 * Widened past `ops/matmul/wgsl.test.ts`'s default and measured, not assumed
 * (rules 2/9), on this machine's RTX 5090 / Dawn (webgpu 0.4.x). `agree`
 * passes an element on *either* measure, so the two limits were probed
 * separately rather than read off one run:
 *
 * - `{ rel: 0, abs: 0 }` fails everything and reports the worst **relative**
 *   difference: **1.57e-3**, on an output element of 7.2e-5 — a sum of terms
 *   four orders of magnitude larger that cancelled, whose absolute difference
 *   is 1.13e-7. Relative error is the wrong measure for that element, which is
 *   exactly why `agree` has two.
 * - `{ rel: 0, abs: X }` fails only elements whose absolute difference exceeds
 *   `X`. At `X = 3e-4` one element in one shape still fails (abs 3.84e-4, rel
 *   3.97e-4, N=4 M=5 K=300); at `X = 4e-4` the survivor is abs 8.54e-4 on an
 *   output of -669.8, whose relative difference is 1.28e-6.
 *
 * The binding case is the middle one — the only element measured that is bad
 * on both measures at once (3.97e-4 rel / 3.84e-4 abs). `abs: 1e-3` clears it
 * with ~2.6x headroom, and `rel: 1e-4` clears everything large enough for
 * absolute error to matter (their relative errors are ~1e-6) with ~75x. Both
 * were chosen from those probes, not opened until green.
 */
const TOLERANCE = { rel: 1e-4, abs: 1e-3 };

// `ops/matmul/wgsl.test.ts`'s shape catalogue, plus the two this format adds:
// a K that is exactly one scale group, and a K that is one group plus a
// remainder. A tiled kernel is fast and wrong on shapes that do not divide the
// tile, and a quantized one is additionally wrong on shapes that do not divide
// the group; a benchmark notices neither.
const SHAPES: [N: number, M: number, K: number, why: string][] = [
  [1, 1, 1, "one element"],
  [TILE, TILE, TILE, "exactly one tile in every dimension"],
  [TILE + 1, TILE, TILE, "N tail only"],
  [TILE, TILE + 1, TILE, "M tail only"],
  [TILE, TILE, TILE + 1, "K tail only, one column past a tile"],
  [TILE - 3, TILE - 5, TILE - 7, "every dimension shorter than one tile"],
  [17, 19, 23, "all three ragged, two tiles each"],
  [5, 7, Q4_GROUP_SIZE, "K is exactly one scale group, eight whole K tiles"],
  [3, 5, Q4_GROUP_SIZE + 2, "two scale groups, the second two columns long"],
  [1, 64, 300, "a single row against a deep K, three scale groups"],
  [4, 5, 300, "many K tiles with a ragged last group"],
  [3, 5, 0, "empty K, where the sum is zero"],
];

describe("matmulQ4G128 / wgsl", () => {
  useGpu();

  for (const [N, M, K, why] of SHAPES) {
    gpuTest(`agrees with the reference at N=${N} M=${M} K=${K} — ${why}`, async (run) => {
      const a = wave(N * K, 0.37);
      const { codes, scales } = quantizeQ4G128({ input: bandedWeight(M, K), N: M, K });
      const weight = packQ4({ codes, N: M, K });
      await expectAgrees(
        run,
        {
          code,
          bindings: [
            { kind: "storage", data: a },
            { kind: "storage", data: weight },
            { kind: "storage", data: scales },
            { kind: "out", type: "f32", length: N * M },
            { kind: "uniform", data: params([["u32", N], ["u32", M], ["u32", K]]) },
          ],
          workgroups: [Math.ceil(M / TILE), Math.ceil(N / TILE)],
        },
        [matmulQ4G128({ a, weight, scale: scales, N, M, K })],
        TOLERANCE,
      );
    });
  }
});
