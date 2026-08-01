import { describe, expect } from "vitest";
import {
  agree,
  expectAgrees,
  gpuTest,
  kernel,
  params,
  useGpu,
  type Runner,
} from "../../harness/index.js";
import { REDUCE, reduce, type ReduceKind } from "./reference.js";

const code = kernel(import.meta.url);

/**
 * Positive on average but not everywhere: `[-0.5, 3.5]`.
 *
 * The bias is not cosmetic. A zero-mean wave summed over a thousand terms
 * cancels down to O(1), and the f32 tree reduction then disagrees with the f64
 * reference by more than the default relative tolerance — which would say
 * something about float addition, not about this kernel. The sign changes are
 * kept so that max and min land on different elements.
 */
const wave = (n: number, k = 0.37) => Float32Array.from({ length: n }, (_, i) => 1.5 + Math.sin(i * k) * 2);

interface Shape {
  outer: number;
  axis: number;
  inner: number;
}

const dispatch = (
  input: Float32Array,
  { outer, axis, inner }: Shape,
  kind: ReduceKind,
  outputLength = outer * inner,
) => ({
  code,
  bindings: [
    { kind: "storage" as const, data: input },
    { kind: "out" as const, type: "f32" as const, length: outputLength },
    {
      kind: "uniform" as const,
      data: params([["u32", outer], ["u32", axis], ["u32", inner], ["u32", kind]]),
    },
  ],
  workgroups: [outputLength] as [number],
});

/**
 * One case, asserted with the shape and the reduction named in the failure.
 *
 * The shapes are looped inside a test rather than given a test each, which is
 * the shape the other ops use. That is a concession to the harness, not a
 * preference. Four reductions over six shapes is 24 cases; written as 24
 * `it()` blocks plus the edges, this file hung every time it was run, while
 * the same 24 cases as nine blocks finish in 230ms. Dispatches are not what
 * costs — 225 of them inside one block are fine, measured. It is the blocks.
 * See #38, which is the same instability at the file boundary.
 *
 * The label is what keeps a failure legible, so it names the case the way a
 * test name would.
 */
async function check(
  run: Runner["run"],
  label: string,
  shape: Shape,
  kind: ReduceKind,
  input: Float32Array,
): Promise<void> {
  const [got] = await run(dispatch(input, shape, kind));
  const worst = agree(got!, reduce({ input, ...shape, kind }));
  expect(worst, worst ? `${label}: ${JSON.stringify(worst)}` : undefined).toBeNull();
}

const KINDS: [string, ReduceKind][] = [
  ["sum", REDUCE.sum],
  ["max", REDUCE.max],
  ["min", REDUCE.min],
  ["mean", REDUCE.mean],
];

// The axis spans the 256-wide workgroup deliberately: below it, exactly on it,
// and not a multiple of it. The strided load loop and the tree reduction behave
// differently in each, and only one of the three is the common case.
//
// `inner` spans 1 and >1 just as deliberately: at `inner = 1` the axis is
// contiguous, which is the case rmsnorm and softmax already handle, and it is
// the only case where an implementation that ignores `inner` while walking the
// axis still gets the right answer.
const SHAPES: Shape[] = [
  { outer: 2, axis: 8, inner: 1 },
  { outer: 1, axis: 256, inner: 1 },
  { outer: 3, axis: 300, inner: 1 },
  { outer: 1, axis: 1000, inner: 1 },
  { outer: 2, axis: 300, inner: 3 },
  { outer: 3, axis: 5, inner: 7 },
];

describe("reduce / wgsl", () => {
  useGpu();

  for (const [name, kind] of KINDS) {
    gpuTest(`${name} agrees with the reference across the workgroup boundary`, async (run) => {
      for (const shape of SHAPES) {
        const input = wave(shape.outer * shape.axis * shape.inner);
        const label = `${name} at [${shape.outer}, ${shape.axis}, ${shape.inner}]`;
        await check(run, label, shape, kind, input);
      }
    });
  }

  gpuTest("max and min do not take the identity for an answer", async (run) => {
    // Zero is a tempting identity for a reduction and it is wrong for both. On
    // any axis that straddles zero the mistake is invisible, because a real
    // element beats the identity anyway. Here every element is on one side of
    // it, and the 248 lanes that never load anything must not win.
    const shape = { outer: 1, axis: 8, inner: 1 };
    const negative = Float32Array.from({ length: 8 }, (_, i) => -1 - Math.abs(Math.sin(i * 0.7)));
    const positive = Float32Array.from({ length: 8 }, (_, i) => 1 + Math.abs(Math.sin(i * 0.7)));
    await check(run, "max of an all-negative axis", shape, REDUCE.max, negative);
    await check(run, "min of an all-positive axis", shape, REDUCE.min, positive);
  });

  gpuTest("mean divides by the axis length, not the workgroup width", async (run) => {
    // An axis of 4 ones has mean 1. Dividing by the 256-wide workgroup, or by
    // the number of lanes that loaded something, gives a different answer, and
    // both mistakes survive every shape above where the axis is a round number.
    const axis = 4;
    const input = new Float32Array(axis).fill(1);
    await expectAgrees(
      run,
      dispatch(input, { outer: 1, axis, inner: 1 }, REDUCE.mean),
      [reduce({ input, outer: 1, axis, inner: 1, kind: REDUCE.mean })],
    );
  });

  gpuTest("sum over an empty axis is zero", async (run) => {
    // PyTorch's `torch.sum(torch.empty(0))` is `tensor(0.)` — the identity, not
    // an error. Measured, not assumed; see reference.ts.
    const [out] = await run(dispatch(new Float32Array(0), { outer: 2, axis: 0, inner: 1 }, REDUCE.sum));
    expect(Array.from(out!)).toEqual([0, 0]);
  });

  gpuTest("mean over an empty axis is NaN", async (run) => {
    // `torch.mean(torch.empty(0))` is `tensor(nan)`, because PyTorch divides by
    // the count without guarding it. This kernel gets there the same way, and
    // this GPU was measured to return NaN rather than Infinity for 0.0 / 0.0.
    //
    // Asserted directly rather than through the comparator: NaN never agrees
    // with NaN under a numeric tolerance.
    const [out] = await run(dispatch(new Float32Array(0), { outer: 2, axis: 0, inner: 1 }, REDUCE.mean));
    expect(Array.from(out!).every(Number.isNaN)).toBe(true);
  });

  gpuTest("writes nothing past outer * inner", async (run) => {
    // Over-dispatched on purpose. Without the bounds check the extra
    // workgroups reduce slices that are not theirs into slots that are not
    // theirs; the output buffer is longer than the result, so those writes land
    // somewhere observable instead of being swallowed by WGSL's bounds
    // checking — which is the only way this branch can be seen at all.
    const shape = { outer: 2, axis: 8, inner: 1 };
    const input = wave(shape.outer * shape.axis * shape.inner);
    const [out] = await run(dispatch(input, shape, REDUCE.sum, shape.outer * shape.inner + 2));
    const expected = reduce({ input, ...shape, kind: REDUCE.sum });
    // Exactly zero, because nothing wrote there at all.
    expect(Array.from(out!).slice(shape.outer * shape.inner)).toEqual([0, 0]);
    // Merely in agreement, because f32 addition got there and the reference
    // used f64. Asserting equality here fails on the last ulp.
    expect(agree(out!.slice(0, shape.outer * shape.inner), expected)).toBeNull();
  });
});
