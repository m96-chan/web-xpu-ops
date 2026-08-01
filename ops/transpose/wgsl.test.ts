import { describe } from "vitest";
import { expectAgrees, gpuTest, kernel, params, useGpu } from "../../harness/index.js";
import { transpose } from "./reference.js";

const code = kernel(import.meta.url);

/** Must match `TILE` in the kernel: it sets the dispatch grid. */
const TILE = 16;

/**
 * Every element distinct, so a misplaced one is a failure rather than a
 * coincidence. Transpose does no arithmetic — it only decides where a value
 * lands — and a smooth input like a sine wave hides a small displacement behind
 * a small difference. These are exact in f32 and no two are equal.
 */
const distinct = (n: number) => Float32Array.from({ length: n }, (_, i) => (i + 1) * 0.25);

describe("transpose / wgsl", () => {
  useGpu();

  // Nothing is computed, so nothing may drift: the output has to be the input
  // bit for bit, only somewhere else. Any tolerance at all would let a kernel
  // that reads the neighbouring element pass on smooth data.
  const exact = { rel: 0, abs: 0 };

  // Shapes span the 256-wide workgroup — which is a 16x16 tile here — from both
  // sides: under one tile, exactly one tile, several whole tiles, and shapes
  // where neither side divides by 16 so the last row and column of tiles hang
  // over the edge.
  //
  // The overhang is the whole reason the ragged shapes are here. Threads out
  // there have no element to move, and the index they would write still lands
  // inside the output buffer — so an unguarded write does not fault, it quietly
  // replaces a real value somewhere else in the matrix. On 16x16 every thread is
  // in range and the guard is dead code, which is why the square power-of-two
  // shape everyone tries first proves nothing about it.
  const shapes = [
    [3, 5], //     under one tile in both directions
    [16, 16], //   exactly one tile, exactly 256 elements
    [17, 33], //   neither side a multiple of the tile; 561 elements
    [64, 256], //  a whole 4x16 grid of tiles, 256 exactly on one axis
    [37, 129], //  ragged in both directions, several tiles
    [1, 300], //   a single row becomes a single column
    [300, 1], //   and back
  ] as const;

  for (const [rows, cols] of shapes) {
    gpuTest(`agrees with the reference at ${rows}x${cols}`, async (run) => {
      const input = distinct(rows * cols);
      await expectAgrees(
        run,
        {
          code,
          bindings: [
            { kind: "storage", data: input },
            { kind: "out", type: "f32", length: rows * cols },
            { kind: "uniform", data: params([["u32", rows], ["u32", cols]]) },
          ],
          // x walks the input's columns, y its rows: the grid is the shape of
          // the input in tiles, not of the output.
          workgroups: [Math.ceil(cols / TILE), Math.ceil(rows / TILE)],
        },
        [transpose({ input, rows, cols })],
        exact,
      );
    });
  }
});
