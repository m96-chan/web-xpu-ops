import { describe, expect } from "vitest";
import { agree, expectAgrees, gpuTest, kernel, params, useGpu } from "../../harness/index.js";
import { ELEMENTWISE } from "../elementwise/reference.js";
import { axpy } from "./reference.js";

const code = kernel(import.meta.url);
/** The op this one exists to replace, read from its own directory rather than restated here. */
const elementwiseCode = kernel(new URL("../elementwise/index.ts", import.meta.url));

const WG = 256;

/**
 * Every case is built here, at module scope, before a device exists — the same
 * rule `ops/snake/wgsl.test.ts` documents: CPU work *between* dispatches kills
 * this binding (#49), so the references are computed up front and the
 * comparisons happen after the last dispatch has returned.
 */
interface Case {
  label: string;
  /** The live prefix — what the kernel is told `N` is. The buffers are longer. */
  n: number;
  a: number;
  x: Float32Array;
  y: Float32Array;
  expected: Float32Array;
  workgroups: number;
}

/**
 * Values past `N` that the kernel must not touch, chosen so that touching them
 * is visible.
 *
 * This device returns 0 for an out-of-range read, so a kernel that dropped its
 * bounds check would compute `y + a*x` from zeros and write 0 — which is
 * exactly what the untouched tail already holds, and the mutation would
 * survive. A sentinel makes the tail say what happened: `7.5 + a * 4.5` is
 * non-zero for every `a` used below, `a = 0` included.
 */
const SENTINEL_X = 4.5;
const SENTINEL_Y = 7.5;

function build(label: string, n: number, a: number, seed = 0.37): Case {
  const workgroups = Math.ceil(n / WG);
  const padded = workgroups * WG;
  const x = new Float32Array(padded).fill(SENTINEL_X);
  const y = new Float32Array(padded).fill(SENTINEL_Y);
  for (let i = 0; i < n; i += 1) {
    // `(i + 1)`, not `i`: at `i = 0` a `sin(0)` x is exactly zero, and then
    // `y + a*x` is `y` whatever `a` does — the N=1 case was passing a kernel
    // with the scalar deleted until this was fixed (rule 1: a test the code
    // can be removed from is measuring the wrong thing).
    x[i] = Math.sin((i + 1) * seed) * 3;
    y[i] = Math.cos(i * 0.11) * 2;
  }
  // Only the first `n` elements are the op's; the rest must stay 0.
  const expected = new Float32Array(padded);
  expected.set(axpy({ x: x.subarray(0, n), y: y.subarray(0, n), a }));
  return { label, n, a, x, y, expected, workgroups };
}

const cases = [
  // Not a multiple of 256, so the bounds check is load-bearing.
  build("N=300, a=0.375", 300, 0.375),
  // a = 0: the output is y, and nothing in the kernel special-cases it.
  build("N=300, a=0", 300, 0),
  // a = 1: elementwise add, reached through this op instead.
  build("N=300, a=1", 300, 1),
  // Negative, and larger than 1, so a sign or magnitude slip cannot hide.
  build("N=1024, a=-2.5", 1024, -2.5, 0.61),
  // A single element — one invocation live out of a full workgroup.
  build("N=1, a=3.25", 1, 3.25),
  // Exactly one workgroup, no tail at all.
  build("N=256, a=0.5", 256, 0.5),
  // Several workgroups plus a tail: 4100 = 16 full groups + 4.
  build("N=4100, a=0.125", 4100, 0.125, 0.017),
];

const bindings = (c: Case) =>
  [
    { kind: "storage", data: c.x },
    { kind: "storage", data: c.y },
    { kind: "out", type: "f32", length: c.expected.length },
    {
      kind: "uniform",
      // `N` is the live prefix, not the padded buffer length — that is what
      // makes the tail a tail.
      data: params([
        ["u32", c.n],
        ["f32", c.a],
      ]),
    },
  ] as const;

describe("axpy / wgsl", () => {
  useGpu();

  for (const c of cases) {
    gpuTest(`agrees with the reference at ${c.label}`, async (run) => {
      // Default tolerance, and on this device it is not close: the kernel
      // reproduces the reference **bit for bit** here, because `a` is passed
      // as `Math.fround`-exact and this compiler contracts `y + a*x` into an
      // FMA, which is the same single rounding the f64 reference performs.
      // Measured over the sweep in the two-dispatch test below: 4096 of 4096
      // elements exactly equal. The tolerance is left at the default anyway —
      // a compiler that does not contract is a conformant one, and would land
      // within half an ulp of the product rather than on it.
      await expectAgrees(
        run,
        { code, bindings: [...bindings(c)], workgroups: [c.workgroups] },
        [c.expected],
      );
    });
  }
});

/* ------------------------------------------------------------------------ *
 * The fusion itself: one dispatch against the two it replaces (#152, #130)
 * ------------------------------------------------------------------------ */

const FUSED_N = 4096;
const FUSED_A = Math.fround(0.37);
const fusedX = Float32Array.from({ length: FUSED_N }, (_, i) => Math.sin(i * 0.37) * 3.1);
const fusedY = Float32Array.from({ length: FUSED_N }, (_, i) => Math.cos(i * 0.11) * 2.7);
/** What the two-dispatch spelling needs and this op does not: `a`, N times over. */
const fusedFill = new Float32Array(FUSED_N).fill(FUSED_A);
const fusedReference = axpy({ x: fusedX, y: fusedY, a: FUSED_A });
const fusedGroups = FUSED_N / WG;

/**
 * The cancellation case, where the two spellings do not differ by an ulp —
 * one of them is zero.
 *
 * `a = f32(0.1)`, `x = 3`: the exact product is 0.3000000044703484, and
 * rounding it to f32 before the add lands on f32(0.3) = 0.30000001192092896.
 * With `y = -f32(0.3)` that rounding error is the entire answer: multiply-
 * then-add gives exactly 0, one rounding gives -2^-27.
 */
const CANCEL_N = WG;
const CANCEL_A = Math.fround(0.1);
const cancelX = new Float32Array(CANCEL_N).fill(3);
const cancelY = new Float32Array(CANCEL_N).fill(-Math.fround(0.3));
const cancelFill = new Float32Array(CANCEL_N).fill(CANCEL_A);

describe("axpy / wgsl / against the two dispatches it replaces", () => {
  useGpu();

  gpuTest("computes what elementwise multiply + elementwise add compute", async (run) => {
    // Both paths run on the GPU, through the real `ops/elementwise` kernel for
    // the two-dispatch side — not a CPU restatement of it. The comparisons are
    // all done after the last dispatch returns.
    const [fused] = await run({
      code,
      bindings: [
        { kind: "storage", data: fusedX },
        { kind: "storage", data: fusedY },
        { kind: "out", type: "f32", length: FUSED_N },
        { kind: "uniform", data: params([["u32", FUSED_N], ["f32", FUSED_A]]) },
      ],
      workgroups: [fusedGroups],
    });
    const [scaled] = await run({
      code: elementwiseCode,
      bindings: [
        { kind: "storage", data: fusedX },
        { kind: "storage", data: fusedFill },
        { kind: "out", type: "f32", length: FUSED_N },
        { kind: "uniform", data: params([["u32", FUSED_N], ["u32", ELEMENTWISE.multiply]]) },
      ],
      workgroups: [fusedGroups],
    });
    const [twoStep] = await run({
      code: elementwiseCode,
      bindings: [
        { kind: "storage", data: fusedY },
        { kind: "storage", data: scaled as Float32Array },
        { kind: "out", type: "f32", length: FUSED_N },
        { kind: "uniform", data: params([["u32", FUSED_N], ["u32", ELEMENTWISE.add]]) },
      ],
      workgroups: [fusedGroups],
    });

    const one = fused as Float32Array;
    const two = twoStep as Float32Array;

    // 1. The two paths agree to within the intermediate rounding the fused one
    //    does not perform. Measured on this device: 882 of 4096 elements
    //    differ, worst absolute 2.38e-7 — under the default absolute
    //    tolerance, which is why this is a fusion and not a change of answer.
    expect(agree(one, two)).toBeNull();

    // 2. Where they differ, the fused one is the better answer — never further
    //    from the correctly-rounded result than the two-dispatch path is.
    //    This is the portable half of the claim: on a compiler that does not
    //    contract, every element is an equality; on this one, the left side is
    //    0 wherever the right side is not.
    let fusedExact = 0;
    let twoStepExact = 0;
    let worse: { index: number; fused: number; twoStep: number; want: number } | null = null;
    for (let i = 0; i < FUSED_N; i += 1) {
      const want = fusedReference[i]!;
      if (one[i]! === want) fusedExact += 1;
      if (two[i]! === want) twoStepExact += 1;
      if (Math.abs(one[i]! - want) > Math.abs(two[i]! - want)) {
        worse ??= { index: i, fused: one[i]!, twoStep: two[i]!, want };
      }
    }
    expect(worse, worse ? `fused is further from the reference: ${JSON.stringify(worse)}` : undefined).toBeNull();
    expect(fusedExact).toBeGreaterThanOrEqual(twoStepExact);

    // 3. And on this device it is not a tie: the fused path is exact on all
    //    4096 elements, the two-dispatch path on 3214 of them. This one is a
    //    **measured, device-specific** assertion, unlike (1) and (2): a WGSL
    //    compiler that declines to contract is conformant, and only this line
    //    would fail on it. The right response to that failure is to record
    //    which device did it, not to widen the line — it is also what catches
    //    the fusion being written back out into two rounded statements, which
    //    (1) and (2) would both survive.
    expect(fusedExact).toBe(FUSED_N);
    expect(twoStepExact).toBeLessThan(FUSED_N);
  });

  gpuTest("rounds once where multiply-then-add cancels to zero", async (run) => {
    const [fused] = await run({
      code,
      bindings: [
        { kind: "storage", data: cancelX },
        { kind: "storage", data: cancelY },
        { kind: "out", type: "f32", length: CANCEL_N },
        { kind: "uniform", data: params([["u32", CANCEL_N], ["f32", CANCEL_A]]) },
      ],
      workgroups: [1],
    });
    const [scaled] = await run({
      code: elementwiseCode,
      bindings: [
        { kind: "storage", data: cancelX },
        { kind: "storage", data: cancelFill },
        { kind: "out", type: "f32", length: CANCEL_N },
        { kind: "uniform", data: params([["u32", CANCEL_N], ["u32", ELEMENTWISE.multiply]]) },
      ],
      workgroups: [1],
    });
    const [twoStep] = await run({
      code: elementwiseCode,
      bindings: [
        { kind: "storage", data: cancelY },
        { kind: "storage", data: scaled as Float32Array },
        { kind: "out", type: "f32", length: CANCEL_N },
        { kind: "uniform", data: params([["u32", CANCEL_N], ["u32", ELEMENTWISE.add]]) },
      ],
      workgroups: [1],
    });

    // The two-dispatch path cancels to exactly zero — this is what fusing is
    // being compared against, asserted rather than assumed.
    expect((scaled as Float32Array)[0]).toBe(Math.fround(0.3));
    expect((twoStep as Float32Array)[0]).toBe(0);

    // The fused path returns one of exactly two legal answers: -2^-27 if the
    // compiler contracted, 0 if it did not. This device returns -2^-27, which
    // is what `torch.add(y, x, alpha=0.1)` returns on CPU and on CUDA
    // (measured, torch 2.10.0+cu128) and what `axpy`'s reference returns.
    expect([-(2 ** -27), 0]).toContain((fused as Float32Array)[0]);
    expect((fused as Float32Array)[0]).toBe(axpy({ x: cancelX, y: cancelY, a: CANCEL_A })[0]);
  });
});
