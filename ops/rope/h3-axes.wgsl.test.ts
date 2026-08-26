/**
 * The `axes` kernel with **fractional** positions.
 *
 * Issue #200. `h3-axes.test.ts` checks the reference against MiniMax-H3's own
 * rope; this checks the kernel against the reference on the same case, and it
 * exists because of a mutation that survived without it.
 *
 * Every case in `wgsl-axes.test.ts` uses whole-number positions — Z-Image's
 * geometry, which is what the op was written for. So putting `f32(i32(pos))`
 * back into the kernel, which is exactly the change that would undo the
 * widening, left all of them green. The capability was in the binding and in no
 * test. One dispatch, in its own file for the budget `testing.ts` documents.
 */
import { describe } from "vitest";
import { expectAgrees, gpuTest, kernel, useGpu } from "../../harness/index.js";
import { axesScenario } from "./testing.js";
import { H3_ROPE_CASE, H3_ROPE_PERMUTATION } from "./h3-cases.js";

const code = kernel(import.meta.url, "axes");

const { numHeads, dimHead } = H3_ROPE_CASE;
const N = H3_ROPE_CASE.want.length / (numHeads * dimHead);

/** H3's channel order into `ropeAxes`'s — the same reorder the reference test does. */
const permuted = Float32Array.from({ length: N * numHeads * dimHead }, (_, i) => {
  const row = Math.floor(i / dimHead);
  return H3_ROPE_CASE.input[row * dimHead + H3_ROPE_PERMUTATION[i % dimHead]!]!;
});

/** `[N, 4]`: H3's three angles and a fourth axis pinned at zero. */
const positions = Float32Array.from({ length: N * 4 }, (_, i) =>
  i % 4 === 3 ? 0 : H3_ROPE_CASE.positions[Math.floor(i / 4) * 3 + (i % 4)]!,
);

const h3 = axesScenario(
  code,
  { axisDims: [16, 16, 16, 16], thetaBase: H3_ROPE_CASE.thetaBase, N, numHeads, positions },
  permuted,
);

describe("rope / axes / wgsl / fractional positions", () => {
  useGpu();

  gpuTest("agrees with the reference on MiniMax-H3's normalised coordinates", async (run) => {
    // The reference is the comparison, as everywhere in this directory; what
    // makes this case worth a dispatch of its own is that its positions are not
    // integers, which no other case here can say.
    await expectAgrees(run, h3.dispatch, [h3.expected], { abs: 4e-6 });
  });
});
