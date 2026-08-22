import { describe } from "vitest";
import { expectAgrees, gpuTest, kernel, params, useGpu } from "../../harness/index.js";
import { nearestUpsample2d, nearestUpsampleScale } from "./reference.js";

const code = kernel(import.meta.url);

/** Must match `@workgroup_size` in the kernel: it sets the dispatch grid. */
const WORKGROUP = 256;

/**
 * Every element distinct, so a value fetched from the wrong row, column, plane
 * or batch is a failure rather than a coincidence. Nearest upsampling does no
 * arithmetic — it only decides which element to copy — and a smooth input hides
 * an off-by-one source index behind a small difference. These are exact in f32
 * and no two are equal.
 */
const distinct = (n: number) => Float32Array.from({ length: n }, (_, i) => (i + 1) * 0.25);

describe("nearestUpsample2d / wgsl", () => {
  useGpu();

  // Every output is a copy of an input element, so any tolerance at all would
  // let a kernel that picked the neighbouring source pass on smooth data.
  const exact = { rel: 0, abs: 0 };

  const shapes = [
    // The 2x every VAE / codec decoder asks for.
    { N: 1, C: 1, H: 3, W: 4, outH: 6, outW: 8, why: "2x, one plane" },
    // Two different non-integer ratios in one call, so a kernel that computes
    // one scale and uses it on both axes cannot pass.
    { N: 1, C: 1, H: 3, W: 4, outH: 5, outW: 7, why: "3->5 by 4->7" },
    // Batch and channels both above 1: with either at 1 the two plane strides
    // that could be swapped are equal.
    { N: 2, C: 3, H: 5, W: 7, outH: 9, outW: 13, why: "N=2, C=3" },
    // Wider than one workgroup and not a multiple of it: the last workgroup of
    // every row runs surplus threads, whose unguarded write lands on a real
    // value elsewhere in the buffer rather than faulting.
    { N: 1, C: 2, H: 2, W: 3, outH: 3, outW: 300, why: "outW spans 2 ragged workgroups" },
    { N: 1, C: 1, H: 4, W: 256, outH: 4, outW: 256, why: "outW exactly one workgroup, identity" },
    // The shapes where the f32 index arithmetic decides the answer: 14 -> 46
    // has a destination whose exact source is an integer and whose f32 source
    // is one less, 2 -> 50 has one where rounding the product to f32 moves the
    // boundary the other way. Both axes, because the kernel computes them with
    // separate code.
    { N: 1, C: 1, H: 14, W: 2, outH: 46, outW: 50, why: "f32 boundary on both axes" },
    { N: 1, C: 1, H: 2, W: 14, outH: 50, outW: 46, why: "the same two, swapped" },
    // Degenerate extents, which is where a stride bug stops cancelling out.
    { N: 1, C: 1, H: 1, W: 1, outH: 7, outW: 9, why: "one pixel to a block" },
    { N: 3, C: 1, H: 6, W: 1, outH: 6, outW: 5, why: "single column, W only" },
    { N: 1, C: 4, H: 1, W: 6, outH: 5, outW: 6, why: "single row, H only" },
  ] as const;

  for (const { N, C, H, W, outH, outW, why } of shapes) {
    gpuTest(`agrees with the reference at ${N}x${C}x${H}x${W} -> ${outH}x${outW} (${why})`, async (run) => {
      const input = distinct(N * C * H * W);
      await expectAgrees(
        run,
        {
          code,
          bindings: [
            { kind: "storage", data: input },
            { kind: "out", type: "f32", length: N * C * outH * outW },
            {
              kind: "uniform",
              data: params([
                ["u32", H],
                ["u32", W],
                ["u32", outH],
                ["u32", outW],
                // The one division, done here rather than in the shader: WGSL
                // allows f32 `/` 2.5 ULP of error and this ratio's last bit
                // moves whole rows. Same call the reference makes, so there is
                // one definition of the scale rather than two.
                ["f32", nearestUpsampleScale(H, outH)],
                ["f32", nearestUpsampleScale(W, outW)],
                ["u32", 0],
                ["u32", 0],
              ]),
            },
          ],
          // x walks the output's columns, y its rows, z the flat (n, c) plane.
          workgroups: [Math.ceil(outW / WORKGROUP), outH, N * C],
        },
        [nearestUpsample2d({ input, N, C, H, W, outH, outW })],
        exact,
      );
    });
  }
});
