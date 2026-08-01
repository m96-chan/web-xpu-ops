import { describe } from "vitest";
import { expectAgrees, gpuTest, kernel, useGpu } from "../../harness/index.js";
import { router } from "./reference.js";
import { routerParams, tokenThreads, wave } from "./testing.js";

const code = kernel(import.meta.url, "router");

/**
 * Fixtures at module scope, and the edge cases in `router-edges.test.ts` rather
 * than here. Both are measured constraints of this harness, not taste — see the
 * note in `testing.ts`.
 */
const CASES = [
  // One thread per token, so the workgroup boundary falls on T: below it,
  // exactly on it, an exact multiple, and one past. E and k vary independently
  // because the top-k scan is a loop over E per rank, and k = E is the case
  // where the last rank has exactly one candidate left.
  [3, 4, 1, false], // below one workgroup; Switch-style single expert
  [256, 8, 2, true], // exactly one workgroup
  [512, 8, 2, false], // an exact multiple
  [257, 16, 4, true], // one thread past the boundary
  [60, 64, 3, true], // E wider than any k
  [5, 2, 2, false], // k == E: every expert selected
].map(([T, E, k, normalize]) => {
  const shape = { T: T as number, E: E as number, k: k as number, normalize: normalize as boolean };
  const logits = wave(shape.T * shape.E, 0.23);
  return { ...shape, logits, ...router({ logits, ...shape }) };
});

describe("moe router / wgsl", () => {
  useGpu();

  for (const { T, E, k, normalize, logits, expert, gate } of CASES) {
    gpuTest(`agrees with the reference at T=${T} E=${E} k=${k} normalize=${normalize}`, async (run) => {
      await expectAgrees(
        run,
        {
          code,
          bindings: [
            { kind: "storage", data: logits },
            { kind: "out", type: "i32", length: T * k },
            { kind: "out", type: "f32", length: T * k },
            { kind: "uniform", data: routerParams(T, E, k, normalize) },
          ],
          workgroups: tokenThreads(T),
        },
        [expert, gate],
      );
    });
  }
});
