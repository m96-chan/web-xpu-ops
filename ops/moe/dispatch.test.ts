import { describe } from "vitest";
import { expectAgrees, gpuTest, kernel, useGpu } from "../../harness/index.js";
import { router } from "./reference.js";
import { dispatched, moeParams, wave } from "./testing.js";

const code = kernel(import.meta.url, "dispatch");

/**
 * Fixtures at module scope, and the edge cases in `dispatch-edges.test.ts`
 * rather than here. Both are measured constraints of this harness, not taste —
 * see the note in `testing.ts`.
 */
const routedFor = (T: number, k: number, E: number) =>
  router({ logits: wave(T * E, 0.23), T, E, k, normalize: true }).expert;

// D crosses the 256-wide copy loop: below, exactly on, an exact multiple, and
// not a multiple. C is below the average load in most of these so that the
// capacity rule is exercised rather than merely present.
const CASES = [
  [3, 2, 4, 3, 8], // fewer tokens than seats
  [8, 2, 4, 4, 256], // D exactly one copy loop
  [20, 1, 4, 8, 300], // D not a multiple, wider than the workgroup
  [100, 4, 8, 30, 7], // many pairs, narrow rows, capacity bites
  [2, 1, 3, 1, 1], // the smallest shape that still routes
  [257, 2, 5, 120, 3], // pair count past 256
].map(([T, k, E, C, D]) => dispatched(T!, k!, E!, C!, D!, routedFor(T!, k!, E!), 1));

describe("moe dispatch / wgsl", () => {
  useGpu();

  for (const { shape, x, expert, pos, expected, length } of CASES) {
    const { T, k, E, C, D } = shape;
    gpuTest(`agrees with the reference at T=${T} k=${k} E=${E} C=${C} D=${D}`, async (run) => {
      await expectAgrees(
        run,
        {
          code,
          bindings: [
            { kind: "storage", data: x },
            { kind: "storage", data: expert },
            { kind: "out", type: "f32", length },
            { kind: "out", type: "i32", length: T * k },
            { kind: "uniform", data: moeParams(T, k, E, C, D) },
          ],
          workgroups: [T * k],
        },
        [expected, pos],
      );
    });
  }
});
