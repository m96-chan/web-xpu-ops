import { describe, expect } from "vitest";
import { expectAgrees, gpuTest, kernel, useGpu } from "../../harness/index.js";
import { bindings, EXACT, expected, pseudoPath, scoresFor } from "./cases.js";
import { ctcDecode } from "./reference.js";

const code = kernel(import.meta.url);

/**
 * The shapes the collapse has to survive.
 *
 * One of three GPU files for this op rather than one, because a vitest process
 * on this machine aborts after enough dispatches — see the note in `cases.ts`.
 * What the op *means* is pinned in `wgsl.test.ts`; the edges around it are in
 * `wgsl-edges.test.ts`; this file is about where the lanes fall.
 */
describe("ctc_decode / wgsl / shapes", () => {
  useGpu();

  // C is what the lanes divide, so the cases span the 256-wide workgroup: under
  // it, exactly on it, one past it, and ragged past it. T and B vary alongside,
  // because the scan over frames is serial and the batch is the dispatch.
  for (const [B, T, C] of [
    [1, 12, 255],
    [1, 12, 256],
    [1, 12, 257],
    [2, 5, 300],
    [3, 3, 300],
    [7, 3, 37],
  ] as const) {
    gpuTest(`agrees with the reference at B=${B} T=${T} C=${C}`, async (run) => {
      const paths = Array.from({ length: B }, (_, b) => pseudoPath(T, C, 17 + b * 91));
      const scores = scoresFor(paths, C, 0.29);
      const want = ctcDecode({ scores, B, T, C });
      // A corpus that decoded to nothing at all would make this vacuous, and a
      // generator quietly degenerating to blanks is exactly what happened once.
      // Individual rows are allowed to come out empty; that is coverage.
      expect(want.lengths.reduce((sum, n) => sum + n, 0)).toBeGreaterThan(0);
      await expectAgrees(
        run,
        { code, bindings: bindings(scores, { B, T, C }), workgroups: [B] },
        expected(want.tokens, want.lengths),
        EXACT,
      );
    });
  }
});
