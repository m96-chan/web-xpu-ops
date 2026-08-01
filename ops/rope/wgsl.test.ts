import { describe } from "vitest";
import { expectAgrees, gpuTest, kernel, params, useGpu, type Dispatch } from "../../harness/index.js";
import { rope, ropeFrequencyParams, type RoPEArgs } from "./reference.js";

const code = kernel(import.meta.url);
const wave = (n: number, k = 0.37) => Float32Array.from({ length: n }, (_, i) => Math.sin(i * k) * 2);

/**
 * Anything but zero.
 *
 * This GPU returns 0 for reads past the end of a storage buffer and drops
 * writes past the end, so a buffer sized exactly to the data hides a missing
 * bounds check twice over: the stray thread reads zeros, rotates them into
 * zeros, and writes them where the expectation is already zero. Padding the
 * input with a live value and sizing the output for the whole dispatch breaks
 * both halves of that.
 */
const SENTINEL = 3.5;

/**
 * Everything but the `run` call, hoisted out of the test bodies.
 *
 * The device dies if more than ~10ms passes between creating it and the first
 * dispatch (#49), so no expectation is built inside a test.
 */
function scenario({
  N, numHeads, headDim, posOffset, thetaBase, scaling,
}: Omit<RoPEArgs, "input">): { dispatch: Dispatch; expected: Float32Array } {
  const length = N * numHeads * headDim;
  const totalPairs = length / 2;
  const workgroups = Math.ceil(totalPairs / 256);
  // Thread p writes slots 2p and 2p+1, so a dispatch rounded up to the
  // workgroup width can reach twice that width. Both buffers are sized for the
  // dispatch rather than for the data, which is what makes the kernel's
  // `pair_idx >= total_pairs` guard observable at all.
  const slots = workgroups * 256 * 2;

  const input = wave(length, 0.29);
  const padded = new Float32Array(slots).fill(SENTINEL);
  padded.set(input);
  const expected = new Float32Array(slots);
  expected.set(rope({ input, N, numHeads, headDim, posOffset, thetaBase, scaling }));

  // The five scalars NTK and YaRN reduce to. Computed in f64 here and narrowed
  // to f32 on the way into the uniform, which is exactly how a caller would
  // use them: once per model, not once per element.
  const freq = ropeFrequencyParams(headDim, thetaBase, scaling);

  return {
    dispatch: {
      code,
      bindings: [
        { kind: "storage", data: padded },
        { kind: "out", type: "f32", length: slots },
        {
          kind: "uniform",
          data: params([
            ["u32", N], ["u32", numHeads], ["u32", headDim], ["u32", posOffset],
            ["f32", freq.effectiveBase], ["f32", freq.interpolationFactor],
            ["f32", freq.rampLow], ["f32", freq.rampHigh], ["f32", freq.attentionFactor],
          ]),
        },
      ],
      workgroups: [workgroups],
    },
    expected,
  };
}

// Plain RoPE, at the two sizes this op has always been checked at. Scaling is
// absent, so these are also the regression test for "nothing changed when
// nothing was asked for".
const plain = [0, 7].map((posOffset) => ({
  posOffset,
  ...scenario({ N: 3, numHeads: 4, headDim: 16, posOffset, thetaBase: 10000 }),
}));

// Positions 200..202 against a trained context of 64 — three times past the
// end of it, which is the only place any of this is observable. Inside the
// trained window the schemes stay near plain RoPE and near each other, so a
// case that stayed there could be satisfied by a kernel that ignored `scaling`
// outright; past it they are 3–5 apart in absolute terms, measured in
// `reference.test.ts`, against the 1e-3 allowed below.
//
// 576 pairs also crosses the 256-wide workgroup twice without landing on a
// multiple of it, so the tail is ragged in both of these.
const beyondContext = { N: 9, numHeads: 8, headDim: 16, posOffset: 200, thetaBase: 10000 } as const;
const ntk = scenario({ ...beyondContext, scaling: { kind: "ntk", factor: 8 } });
const yarn = scenario({
  ...beyondContext,
  scaling: { kind: "yarn", factor: 8, originalContextLength: 64 },
});

// A second YaRN case, at Llama-2's actual geometry — D = 128, L = 2048, s = 8,
// so 2048 tokens of training read out to 16384 — and here for one specific
// reason: at D = 16, L = 64 the ramp starts at pair 0, and the *lower* clamp in
// the kernel never binds. Delete it and nothing goes red. At this geometry the
// ramp runs from pair 16 to pair 41 of 64, so pairs 0..15 arrive with a
// negative raw ramp and the clamp is load-bearing.
const llama2 = scenario({
  N: 3, numHeads: 2, headDim: 128, posOffset: 6000, thetaBase: 10000,
  scaling: { kind: "yarn", factor: 8, originalContextLength: 2048 },
});

describe("rope / wgsl", () => {
  useGpu();

  // Loosened on measurement, not on principle. GPU sin and cos carry up to
  // 1.86e-4 of absolute error here — three orders of magnitude worse than f32
  // epsilon — and this op calls both per element. `pow` was measured separately
  // and agrees to 2.8e-7, so the transcendentals are the whole difference; the
  // kernel and the reference compute the same expression.
  //
  // The scaled cases do not need more than the plain ones did, and were not
  // given more. They carry two extra error sources — an f32 `effectiveBase`
  // where the reference has f64, and YaRN's 1.208 gain on top of an input of
  // magnitude 2 — but the worst absolute error measured on this GPU is
  //   plain posOffset=0  4.77e-7      NTK   8.32e-5
  //   plain posOffset=7  1.57e-6      YaRN  1.01e-4
  // an order of magnitude inside the bound in the worst case. Positions were
  // kept at 200 rather than pushed to thousands for the same reason: theta
  // rounds in f32 here and in f64 in the reference, so the gap grows with
  // position and would eventually report argument reduction rather than RoPE.
  //
  // Nothing tighter is reachable, and a test demanding it would be reporting
  // the hardware rather than the shader.
  const transcendental = { abs: 1e-3 };

  for (const { posOffset, dispatch, expected } of plain) {
    gpuTest(`agrees with the reference at posOffset=${posOffset}`, async (run) => {
      await expectAgrees(run, dispatch, [expected], transcendental);
    });
  }

  gpuTest("agrees with the NTK reference past the trained context", async (run) => {
    await expectAgrees(run, ntk.dispatch, [ntk.expected], transcendental);
  });

  gpuTest("agrees with the YaRN reference past the trained context", async (run) => {
    await expectAgrees(run, yarn.dispatch, [yarn.expected], transcendental);
  });

  // Position 6000 puts theta near 6000 radians for the fastest pair, some 955
  // turns. The GPU reduces that argument in f32; the reference does not reduce
  // at all. So this case gets its own bound, and the reason it needs one is
  // measurable rather than asserted — this same geometry, moving only the
  // position:
  //   posOffset  200   9.87e-5
  //   posOffset 2100   9.86e-4
  //   posOffset 6000   3.09e-3
  // The error tracks the position and nothing else, which is argument
  // reduction and not the ramp, the blend or the gain. 1e-2 is three times the
  // worst of those.
  const longRange = { abs: 1e-2 };

  gpuTest("agrees with YaRN at Llama-2's geometry, where the ramp starts past pair 0", async (run) => {
    await expectAgrees(run, llama2.dispatch, [llama2.expected], longRange);
  });
});
