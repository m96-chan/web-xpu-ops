import { describe } from "vitest";
import { expectAgrees, gpuTest, kernel, params, useGpu } from "../../harness/index.js";
import { alibiSlopes } from "./reference.js";

/**
 * The slopes, on their own — separate kernel, separate file, separate device.
 *
 * Separate kernel because the two halves of ALiBi fail differently and must not
 * be able to hide inside each other: a wrong slope is a model that trains
 * badly, a wrong index decomposition is a model that does not train at all, and
 * a single fused dispatch could not say which one broke. The bias kernel takes
 * the slopes as a buffer for the same reason.
 *
 * Separate *file* is not organisation, it is the repository's own workaround
 * for #38: one device per file, one file per vitest process. Measured here —
 * the slopes and the bias in one file came to 27 GPU tests on one device, and
 * one run in five died inside Dawn before reporting anything. Split, each half
 * reports.
 *
 * What the numbers are checked against is in `reference.test.ts`, which pins
 * them to the paper. This file only says the shader and the reference agree.
 */

const code = kernel(import.meta.url, "slopes");
const WG = 256;

describe("alibi slopes / wgsl", () => {
  useGpu();

  // Head counts either side of the power-of-two branch, and either side of the
  // 256-wide workgroup. 9, 12, 15 and 300 exercise the appended tail — the part
  // the paper does not interpolate and implementations get wrong; 255/256/257
  // are below, exactly at, and one past the dispatch width.
  const HEAD_COUNTS = [1, 2, 3, 4, 8, 9, 12, 15, 16, 17, 32, 255, 256, 257, 300];

  for (const numHeads of HEAD_COUNTS) {
    gpuTest(`computes the slopes for ${numHeads} heads`, async (run) => {
      // The kernel derives the run in closed form — exp2 of a rational exponent
      // — while the reference is the paper's recursion. Two different
      // computations, which is the point: the derivation is checked here rather
      // than assumed.
      const length = Math.ceil(numHeads / WG) * WG;

      // The buffer is a whole dispatch long and the tail must stay zero. Sized
      // to the head count instead, a missing bounds check would write off the
      // end and this driver would discard it, leaving the guard unobservable.
      // At full length the stray write lands, and it is not a rounding-sized
      // number: the first lane past the end evaluates the appended branch and
      // produces ~0.04 at 12 heads, ~0.7 at 8.
      const expected = new Float32Array(length);
      expected.set(alibiSlopes(numHeads));

      await expectAgrees(
        run,
        {
          code,
          bindings: [
            { kind: "out", type: "f32", length },
            { kind: "uniform", data: params([["u32", numHeads]]) },
          ],
          workgroups: [Math.ceil(numHeads / WG)],
        },
        [expected],
      );
    });
  }
});
