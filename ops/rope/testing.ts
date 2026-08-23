import { params, type Dispatch } from "../../harness/index.js";
import {
  rope,
  ropeAxes,
  ropeCache,
  ropeFrequencyParams,
  type RoPEArgs,
  type RoPEAxesArgs,
} from "./reference.js";

/**
 * Shared by `wgsl.test.ts` and `wgsl-cache.test.ts`.
 *
 * The two files exist because of the dispatch budget, not because the cases
 * belong apart: eight dispatches in one file kills the vitest worker on this
 * setup often enough to be useless — measured at 5 deaths in 8 runs against 0
 * in 8 for the five-dispatch file it grew from, interleaved so that machine
 * load hit both equally. Five and three each survive. Same failure family as
 * #38, and the same remedy `ops/moe` and `ops/ctc_decode` already use. Filed
 * as #68, with the sweep and the things it is *not*; nothing here works around
 * it beyond staying under the budget.
 */

/**
 * Anything but zero.
 *
 * This GPU returns 0 for reads past the end of a storage buffer and drops
 * writes past the end, so a buffer sized exactly to the data hides a missing
 * bounds check twice over: the stray thread reads zeros, rotates them into
 * zeros, and writes them where the expectation is already zero. Padding the
 * input with a live value and sizing the output for the whole dispatch breaks
 * both halves of that.
 *
 * The cache buffer is padded with it for the same reason: a guard that let one
 * position too many through would otherwise read zeros, and a zero cos with a
 * zero sin is a zero output, which is hard to tell from a thread that never
 * ran.
 */
export const SENTINEL = 3.5;

export const wave = (n: number, k = 0.37) =>
  Float32Array.from({ length: n }, (_, i) => Math.sin(i * k) * 2);

export interface Cached {
  /** Positions `[0, positions)` are tabulated. 0 leaves the kernel uncached. */
  positions: number;
  /**
   * A wrong table, on purpose. When the table is right the two paths agree by
   * construction, so nothing else can show which one ran.
   */
  poison?: (v: number) => number;
}

/**
 * Everything but the `run` call, hoisted out of the test bodies.
 *
 * The device dies if more than ~10ms passes between creating it and the first
 * dispatch (#49), so no expectation is built inside a test.
 */
export function scenario(
  code: string,
  {
    N, numHeads, headDim, posOffset, thetaBase, scaling,
    headOffset = 0, headCount = numHeads,
  }: Omit<RoPEArgs, "input" | "cache">,
  cached: Cached = { positions: 0 },
  /**
   * Defaults to the wave every other case uses.
   *
   * Head-range cases pass one that is nowhere zero: a passthrough head the
   * kernel never wrote reads back as zero, and an input that is itself zero
   * somewhere cannot tell that apart from a copy. See `wgsl-heads.test.ts`.
   */
  input: Float32Array = wave(N * numHeads * headDim, 0.29),
): { dispatch: Dispatch; expected: Float32Array } {
  const length = N * numHeads * headDim;
  const totalPairs = length / 2;
  const workgroups = Math.ceil(totalPairs / 256);
  // Thread p writes slots 2p and 2p+1, so a dispatch rounded up to the
  // workgroup width can reach twice that width. Both buffers are sized for the
  // dispatch rather than for the data, which is what makes the kernel's
  // `pair_idx >= total_pairs` guard observable at all.
  const slots = workgroups * 256 * 2;

  const padded = new Float32Array(slots).fill(SENTINEL);
  padded.set(input);

  // The table the kernel reads, and the same table the reference reads, so that
  // a poisoned one is not a disagreement — it is the agreed-upon wrong answer,
  // which is what makes "did it read the table" observable.
  const built = ropeCache({ headDim, thetaBase, positions: cached.positions, scaling });
  const cache = cached.poison ? { ...built, table: built.table.map(cached.poison) } : built;
  // Four positions of slack past the end, filled with a live value: a guard
  // that read one position too far lands here rather than in returned-as-zero
  // territory.
  const table = new Float32Array(cache.table.length + headDim * 4).fill(SENTINEL);
  table.set(cache.table);

  // A correct table is compared against the *uncached* reference on purpose:
  // that is the issue's "cached and uncached agree", and it is the stronger
  // statement, since the reference then keeps its angles in f64 throughout. It
  // is also self-checking — the fallback would miss an f64 angle by up to
  // 1.86e-4 on this GPU and the table by an f32 ulp, so the tolerance the test
  // passes at says which path ran.
  //
  // A poisoned table has no uncached counterpart, so there the reference reads
  // the same wrong table and the expectation is the agreed-upon wrong answer.
  const expected = new Float32Array(slots);
  expected.set(
    rope({
      input, N, numHeads, headDim, posOffset, thetaBase, scaling, headOffset, headCount,
      ...(cached.poison ? { cache } : {}),
    }),
  );

  // The five scalars NTK and YaRN reduce to. Computed in f64 here and narrowed
  // to f32 on the way into the uniform, which is exactly how a caller would
  // use them: once per model, not once per element.
  const freq = ropeFrequencyParams(headDim, thetaBase, scaling);

  return {
    dispatch: {
      code,
      bindings: [
        { kind: "storage", data: padded },
        { kind: "storage", data: table },
        { kind: "out", type: "f32", length: slots },
        {
          kind: "uniform",
          data: params([
            ["u32", N], ["u32", numHeads], ["u32", headDim], ["u32", posOffset],
            ["u32", cached.positions],
            ["f32", freq.effectiveBase], ["f32", freq.interpolationFactor],
            ["f32", freq.rampLow], ["f32", freq.rampHigh], ["f32", freq.attentionFactor],
            // Appended rather than grouped with the geometry above so that the
            // offsets the kernel already reads did not move when the range
            // arrived — the default case stays the same bytes it was.
            ["u32", headOffset], ["u32", headCount],
          ]),
        },
      ],
      workgroups: [workgroups],
    },
    expected,
  };
}

/**
 * Positions 200..208 against a trained context of 64 — well past the end of it,
 * which is the only place any of this is observable. Inside the trained window
 * the schemes stay near plain RoPE and near each other, so a case that stayed
 * there could be satisfied by a kernel that ignored `scaling` outright; past it
 * they are 3–5 apart in absolute terms, measured in `reference.test.ts`.
 *
 * 576 pairs also crosses the 256-wide workgroup twice without landing on a
 * multiple of it, so the tail is ragged.
 */
export const BEYOND_CONTEXT = {
  N: 9, numHeads: 8, headDim: 16, posOffset: 200, thetaBase: 10000,
} as const;

export const YARN_64 = { kind: "yarn", factor: 8, originalContextLength: 64 } as const;

/**
 * Loosened on measurement, not on principle. GPU sin and cos carry up to
 * 1.86e-4 of absolute error here — three orders of magnitude worse than f32
 * epsilon — and the uncached path calls both per element. `pow` was measured
 * separately and agrees to 2.8e-7, so the transcendentals are the whole
 * difference; the kernel and the reference compute the same expression.
 *
 * The scaled cases do not need more than the plain ones did, and were not given
 * more. They carry two extra error sources — an f32 `effectiveBase` where the
 * reference has f64, and YaRN's 1.208 gain on top of an input of magnitude 2 —
 * but the worst absolute error measured on this GPU is
 *   plain posOffset=0  4.77e-7      NTK   8.32e-5
 *   plain posOffset=7  1.57e-6      YaRN  1.01e-4
 * an order of magnitude inside the bound in the worst case. Positions were kept
 * at 200 rather than pushed to thousands for the same reason: theta rounds in
 * f32 here and in f64 in the reference, so the gap grows with position and
 * would eventually report argument reduction rather than RoPE.
 *
 * Nothing tighter is reachable, and a test demanding it would be reporting the
 * hardware rather than the shader.
 */
export const TRANSCENDENTAL = { abs: 1e-3 };

/**
 * The same job `scenario` does, for the `axes` entry point.
 *
 * Separate rather than a flag on `scenario`: the two kernels take different
 * bindings — an axis table and a position per axis here, a cache and five
 * scaling scalars there — so one builder would be two builders sharing a name.
 *
 * The padding argument is `scenario`'s and is not repeated: input padded with a
 * live value, output sized for the whole dispatch, so the kernel's
 * `pair_idx >= total_pairs` guard is observable rather than hidden by a buffer
 * that ends exactly where the data does.
 */
export function axesScenario(
  code: string,
  { N, numHeads, axisDims, positions, thetaBase }: Omit<RoPEAxesArgs, "input">,
  /**
   * Defaults to the wave the other cases use. Cases that assert a *copy* pass
   * one that is nowhere zero, since an output nobody wrote reads back as zero
   * and the wave is zero at index 0.
   */
  input: Float32Array = wave(N * numHeads * axisDims.reduce((sum, dim) => sum + dim, 0), 0.29),
  /**
   * Overrides the reference. One case checks the axes kernel against `rope`
   * itself rather than against `ropeAxes`, which is the GPU half of "one axis
   * is the op this generalises".
   */
  expectation: Float32Array = ropeAxes({ input, N, numHeads, axisDims, positions, thetaBase }),
): { dispatch: Dispatch; expected: Float32Array } {
  const headDim = axisDims.reduce((sum, dim) => sum + dim, 0);
  const length = N * numHeads * headDim;
  const workgroups = Math.ceil(length / 2 / 256);
  const slots = workgroups * 256 * 2;

  const padded = new Float32Array(slots).fill(SENTINEL);
  padded.set(input);

  // Four tokens of slack past the positions, filled with something visible: a
  // thread whose `token` ran past `N` reads here, and if it read zeros instead
  // it would apply the identity rotation — which against a zeroed output is
  // indistinguishable from the guard having worked.
  const positionSlack = new Int32Array(positions.length + axisDims.length * 4).fill(9999);
  positionSlack.set(positions);

  const expected = new Float32Array(slots);
  expected.set(expectation);

  return {
    dispatch: {
      code,
      bindings: [
        { kind: "storage", data: padded },
        { kind: "storage", data: Uint32Array.from(axisDims) },
        { kind: "storage", data: positionSlack },
        { kind: "out", type: "f32", length: slots },
        {
          kind: "uniform",
          data: params([
            ["u32", N], ["u32", numHeads], ["u32", headDim], ["u32", axisDims.length],
            ["f32", thetaBase],
          ]),
        },
      ],
      workgroups: [workgroups],
    },
    expected,
  };
}
