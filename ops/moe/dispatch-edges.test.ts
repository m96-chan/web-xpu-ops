import { describe, expect } from "vitest";
import { expectAgrees, gpuTest, kernel, useGpu } from "../../harness/index.js";
import { dispatched, moeParams } from "./testing.js";

const code = kernel(import.meta.url, "dispatch");

// Rank-major, then token order — GShard's rule. Flattening (token, rank)
// row-major instead, which is what a stable sort by expert id gives, fills both
// seats from token 0 and drops token 1 entirely. The shape is chosen so the two
// orders disagree on every position.
const PRIORITY = dispatched(3, 2, 2, 1, 4, Int32Array.from([0, 1, 1, 0, 0, 1]), 0);

// Dispatch does not get to assume the router wrote its input: `-1` is what the
// router itself emits when k exceeds E. Clamping would route a token to an
// expert it never chose, which nothing downstream can detect.
const OUT_OF_RANGE = dispatched(4, 2, 3, 4, 5, Int32Array.from([-1, 2, 5, -1, 0, 1, 300, 2]), 0);

// Every token to expert 1, which has room for three of them. The tokens that fit
// are the first three, the rest are dropped, and experts 0 and 2 stay untouched
// — a capacity guard that clamped instead of dropping would pile the surplus
// onto the last seat, which `pos` alone would not show.
const OVERFLOW = dispatched(10, 1, 3, 3, 4, new Int32Array(10).fill(1), 0);

describe("moe dispatch edges / wgsl", () => {
  useGpu();

  for (const [name, fixture, positions] of [
    ["gives a token's higher-ranked choice the seat", PRIORITY, [0, -1, 0, -1, -1, -1]],
    ["drops expert ids outside [0, E)", OUT_OF_RANGE, [-1, 0, -1, -1, 0, 0, -1, 1]],
    [
      "overflows a single expert without touching its neighbours",
      OVERFLOW,
      [0, 1, 2, -1, -1, -1, -1, -1, -1, -1],
    ],
  ] as const) {
    gpuTest(name, async (run) => {
      const { shape, x, expert, pos, expected, length } = fixture;
      const { T, k, E, C, D } = shape;
      // Spelled out rather than only compared: these three are the policy, and a
      // reference that quietly changed its mind would otherwise take the kernel
      // with it.
      expect(Array.from(pos)).toEqual(positions);
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
