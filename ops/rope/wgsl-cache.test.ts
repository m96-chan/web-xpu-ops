import { describe } from "vitest";
import { expectAgrees, gpuTest, kernel, useGpu } from "../../harness/index.js";
import { BEYOND_CONTEXT, TRANSCENDENTAL, YARN_64, scenario } from "./testing.js";

/**
 * The rotary cache: reading the table, and what happens at the end of it.
 *
 * The same kernel as `wgsl.test.ts`, at the same shape — 576 pairs, so the
 * 256-wide workgroup is crossed twice and the tail is ragged — over positions
 * 200..208. Split into its own file for the dispatch budget; see `testing.ts`.
 */
const code = kernel(import.meta.url);

// Fully covered: a table of 209 positions holds every position these read.
const covered = { positions: 209 };
const cachedPlain = scenario(code, BEYOND_CONTEXT, covered);
// The table is built from the frequencies, so scaling changes the table. A
// table built by a `ropeCache` that dropped `scaling` would still be a valid
// table of plain-RoPE angles, and this is where that shows: at position 200
// against a trained context of 64, plain and YaRN are 3.75 apart (measured in
// `reference.test.ts`) against the 1e-5 allowed below.
const cachedYarn = scenario(code, { ...BEYOND_CONTEXT, scaling: YARN_64 }, covered);

// The boundary. A table of 204 positions against positions 200..208: tokens
// 0..3 are in it and tokens 4..8 are past it, in one dispatch, so both arms run
// side by side and an off-by-one moves a whole token across the line.
//
// Every entry halved. Nothing subtler works: with a correct table the two arms
// agree to an f32 ulp by construction, so a kernel that ignored the table
// outright would pass. Halved, the cached tokens come back at half amplitude
// and the fallback tokens do not.
const boundary = scenario(code, BEYOND_CONTEXT, { positions: 204, poison: (v) => v * 0.5 });

describe("rope / cache / wgsl", () => {
  useGpu();

  // Two orders of magnitude tighter than `TRANSCENDENTAL`, and the tightness is
  // doing work rather than being tidy. The table arrives already rounded from
  // f64, so the GPU only multiplies; the fallback calls the GPU's own `sin` and
  // `cos`. Worst absolute error measured against the uncached f64 reference at
  // this exact geometry, positions 200..208:
  //   read from the table   2.384e-7 plain, 2.384e-7 YaRN — one f32 ulp at
  //                         magnitude 2, which is the table's own rounding
  //   recomputed            8.32e-5 plain, 1.01e-4 YaRN
  // 1e-5 sits between them: forty times the measured error of the path under
  // test, and eight times below the best the other path can do. A kernel that
  // ignored the table and recomputed fails this by being insufficiently
  // accurate, quite apart from failing the boundary test below.
  const tabulated = { abs: 1e-5 };

  gpuTest("reads the table instead of recomputing, and lands on the f64 angles", async (run) => {
    await expectAgrees(run, cachedPlain.dispatch, [cachedPlain.expected], tabulated);
  });

  gpuTest("tabulates YaRN's frequencies, not plain RoPE's", async (run) => {
    await expectAgrees(run, cachedYarn.dispatch, [cachedYarn.expected], tabulated);
  });

  // Mixed: the cached tokens are exact against a table the reference read too,
  // and the fallback tokens go through GPU `sin`/`cos`, so the loose bound
  // applies to the dispatch as a whole. Measured worst here: 8.32e-5, all of it
  // in the fallback half.
  gpuTest("reads the table below its length and computes past the end of it", async (run) => {
    await expectAgrees(run, boundary.dispatch, [boundary.expected], TRANSCENDENTAL);
  });
});
