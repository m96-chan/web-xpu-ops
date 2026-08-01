import { describe } from "vitest";
import { expectAgrees, gpuTest, kernel, useGpu } from "../../harness/index.js";
import { BEYOND_CONTEXT, TRANSCENDENTAL, YARN_64, scenario } from "./testing.js";

/**
 * RoPE without a table: `cache_positions = 0`, which selects the kernel's
 * fallback arm and nothing else. That arm is the whole op as it stood before
 * the cache existed, so these are also the regression test for "adding a cache
 * changed nothing for callers who did not ask for one".
 *
 * The table's own cases are in `wgsl-cache.test.ts` — split for the dispatch
 * budget rather than by subject; see `testing.ts`.
 */
const code = kernel(import.meta.url);

// Plain RoPE, at the two sizes this op has always been checked at.
const plain = [0, 7].map((posOffset) => ({
  posOffset,
  ...scenario(code, { N: 3, numHeads: 4, headDim: 16, posOffset, thetaBase: 10000 }),
}));

const ntk = scenario(code, { ...BEYOND_CONTEXT, scaling: { kind: "ntk", factor: 8 } });
const yarn = scenario(code, { ...BEYOND_CONTEXT, scaling: YARN_64 });

// A second YaRN case, at Llama-2's actual geometry — D = 128, L = 2048, s = 8,
// so 2048 tokens of training read out to 16384 — and here for one specific
// reason: at D = 16, L = 64 the ramp starts at pair 0, and the *lower* clamp in
// the kernel never binds. Delete it and nothing goes red. At this geometry the
// ramp runs from pair 16 to pair 41 of 64, so pairs 0..15 arrive with a
// negative raw ramp and the clamp is load-bearing.
const llama2 = scenario(code, {
  N: 3, numHeads: 2, headDim: 128, posOffset: 6000, thetaBase: 10000,
  scaling: { kind: "yarn", factor: 8, originalContextLength: 2048 },
});

describe("rope / wgsl", () => {
  useGpu();

  for (const { posOffset, dispatch, expected } of plain) {
    gpuTest(`agrees with the reference at posOffset=${posOffset}`, async (run) => {
      await expectAgrees(run, dispatch, [expected], TRANSCENDENTAL);
    });
  }

  gpuTest("agrees with the NTK reference past the trained context", async (run) => {
    await expectAgrees(run, ntk.dispatch, [ntk.expected], TRANSCENDENTAL);
  });

  gpuTest("agrees with the YaRN reference past the trained context", async (run) => {
    await expectAgrees(run, yarn.dispatch, [yarn.expected], TRANSCENDENTAL);
  });

  // Position 6000 puts theta near 6000 radians for the fastest pair, some 955
  // turns. The GPU reduces that argument in f32; the reference does not reduce
  // at all. So this case gets its own bound, and the reason it needs one is
  // measurable rather than asserted — this same geometry, moving only the
  // position:
  //   posOffset  200   9.87e-5
  //   posOffset 2100   9.86e-4
  //   posOffset 6000   3.09e-3
  // The error tracks the position and nothing else, which is argument
  // reduction and not the ramp, the blend or the gain. 1e-2 is three times the
  // worst of those.
  const longRange = { abs: 1e-2 };

  gpuTest("agrees with YaRN at Llama-2's geometry, where the ramp starts past pair 0", async (run) => {
    await expectAgrees(run, llama2.dispatch, [llama2.expected], longRange);
  });
});
