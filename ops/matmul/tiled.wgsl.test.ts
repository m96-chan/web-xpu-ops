/**
 * `tiled.wgsl` against `reference.ts`, on the same shapes `wgsl.test.ts` uses.
 *
 * Issue #177. It is a second implementation of a function this repository
 * already has, so it is held to the first one — not to its own output, and not
 * to `kernel.wgsl`'s, since two kernels agreeing says nothing about either.
 *
 * The shapes matter more here than for the one-output-per-thread kernel. This
 * one covers 64x128 outputs per workgroup and stages 16 of K at a time, so
 * every edge case is a shape that divides by none of those — which is most real
 * ones, Anima's M of 3,952 included.
 */
import { describe, expect } from "vitest";
import { gpuTest } from "../../harness/suite.js";
import { params } from "../../harness/wgsl.js";
import { matmul } from "./reference.js";
import { matmulTiledGrid } from "./index.js";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

const CODE = readFileSync(fileURLToPath(new URL("./wgsl/tiled.wgsl", import.meta.url)), "utf8");

/**
 * Ragged on purpose, and around the tile boundaries on purpose.
 *
 * 64 and 128 are the workgroup tile; 16 is the K chunk. One under, one over and
 * one exactly on each is what catches a guard that is off by one, and a kernel
 * that only works when everything divides is exactly what a sweep would
 * otherwise have reported as the fastest.
 */
const SHAPES: [number, number, number][] = [
  [1, 1, 1],
  [37, 43, 29],
  [63, 15, 127],
  [64, 16, 128],
  [65, 17, 129],
  [3952, 64, 128],
  [128, 2048, 64],
  [200, 300, 100],
];

describe("tiled matmul", () => {
  for (const [M, K, N] of SHAPES) {
    gpuTest(`matches the reference at ${M}x${K}x${N}`, async (run) => {
      const a = Float32Array.from({ length: M * K }, (_, i) => Math.sin(i * 0.7) * 0.5);
      const b = Float32Array.from({ length: K * N }, (_, i) => Math.cos(i * 0.3) * 0.5);
      const [got] = await run({
        code: CODE,
        bindings: [
          { kind: "storage", data: a },
          { kind: "storage", data: b },
          { kind: "out", type: "f32", length: M * N },
          { kind: "uniform", data: params([["u32", M], ["u32", N], ["u32", K]]) },
        ],
        workgroups: matmulTiledGrid(M, N),
      });
      const want = matmul({ a, b, M, N, K });
      const out = got as Float32Array;
      let worst = 0;
      for (let i = 0; i < want.length; i += 1) worst = Math.max(worst, Math.abs(out[i]! - want[i]!));
      // f32 accumulation over K in a different order than the reference walks
      // it; the bound is the summation, not the arithmetic.
      expect(worst).toBeLessThan(1e-3);
    });
  }
});
