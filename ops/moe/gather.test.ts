import { describe, expect } from "vitest";
import { expectAgrees, gpuTest, kernel, useGpu } from "../../harness/index.js";
import { moeDispatch, moeGather, router } from "./reference.js";
import { moeParams, wave, withJunk } from "./testing.js";

const code = kernel(import.meta.url, "gather");

/** Fixtures at module scope, and shapes kept small — see the note in `testing.ts`. */
const CASES = [
  [3, 2, 4, 3, 8], // narrower than one lane loop
  [8, 2, 4, 4, 256], // D exactly one lane loop
  [20, 1, 4, 8, 300], // D wider than the workgroup, not a multiple
  [100, 4, 8, 30, 7], // many tokens, narrow rows, k = 4
  [257, 2, 5, 120, 3], // token count past 256
].map(([T, k, E, C, D]) => {
  const shape = { T: T!, k: k!, E: E!, C: C!, D: D! };
  const { expert, gate } = router({ logits: wave(shape.T * shape.E, 0.23), ...shape, normalize: true });
  const { pos } = moeDispatch({ x: wave(shape.T * shape.D, 0.11), expert, ...shape });
  // Expert outputs, not the dispatched tokens: gather must not be able to pass
  // by ignoring `pos` and handing the token back unchanged.
  const y = wave(shape.E * shape.C * shape.D, 0.05);
  return { shape, expert, gate, pos, y, expected: moeGather({ y, expert, pos, gate, ...shape }) };
});

/**
 * `pos = -1` is what dispatch writes for a dropped token, and it has to
 * contribute nothing rather than address y from the end. The other two are
 * inconsistent inputs the kernel is not entitled to trust: an expert past E and
 * a position past C both address inside the flat buffer, so an unchecked read
 * returns a real number belonging to somebody else.
 */
const RANGES = (() => {
  const shape = { T: 4, k: 2, E: 3, C: 2, D: 3 };
  const expert = Int32Array.from([0, -1, 1, 2, 7, 0, -1, -1]);
  const pos = Int32Array.from([0, 0, 1, 0, 0, 5, -1, -1]);
  const gate = Float32Array.from([0.5, 0.25, 0.5, 0.5, 0.25, 0.25, 0, 0]);
  // Padded well past E*C*D: expert 7 addresses y[42..44], and on this device a
  // read that far past the *buffer* returns zero, which is what skipping the
  // slot returns too. The check would be untestable without something real out
  // there.
  const y = withJunk(wave(shape.E * shape.C * shape.D, 0.05), 40);
  return { shape, expert, pos, gate, y, expected: moeGather({ y, expert, pos, gate, ...shape }) };
})();

describe("moe gather / wgsl", () => {
  useGpu();

  // One workgroup per token, lanes walking the row, so D crosses the 256-wide
  // loop: below it, exactly on it, wider but not a multiple, and narrow with
  // many tokens.
  for (const { shape, expert, gate, pos, y, expected } of CASES) {
    const { T, k, E, C, D } = shape;
    gpuTest(`agrees with the reference at T=${T} k=${k} E=${E} C=${C} D=${D}`, async (run) => {
      await expectAgrees(
        run,
        {
          code,
          bindings: [
            { kind: "storage", data: y },
            { kind: "storage", data: expert },
            { kind: "storage", data: pos },
            { kind: "storage", data: gate },
            { kind: "out", type: "f32", length: T * D },
            { kind: "uniform", data: moeParams(T, k, E, C, D) },
          ],
          workgroups: [T],
        },
        [expected],
      );
    });
  }

  gpuTest("ignores a slot whose expert or position is out of range", async (run) => {
    const { shape, expert, pos, gate, y, expected } = RANGES;
    const { T, k, E, C, D } = shape;
    // Token 3 lost both slots: a zero row, not a scaled copy of anything.
    expect(Array.from(expected.subarray(9))).toEqual([0, 0, 0]);
    await expectAgrees(
      run,
      {
        code,
        bindings: [
          { kind: "storage", data: y },
          { kind: "storage", data: expert },
          { kind: "storage", data: pos },
          { kind: "storage", data: gate },
          { kind: "out", type: "f32", length: T * D },
          { kind: "uniform", data: moeParams(T, k, E, C, D) },
        ],
        workgroups: [T],
      },
      [expected],
    );
  });
});
