import { expect } from "vitest";
import { agree, expectAgrees, kernel, params, type Runner } from "../../harness/index.js";
import { defaultScale, groupedAttention, type GroupedAttentionArgs, type GroupedAttentionResult } from "./reference.js";

/**
 * Shared by `wgsl.test.ts` and `wgsl-seff.test.ts`.
 *
 * The two files exist because of the dispatch budget, not because the cases
 * belong apart — the same reason `ops/rope` already split (`ops/rope/testing.ts`'s
 * own doc) and the same issue, #68: too many GPU dispatches in one vitest
 * worker kills it often enough to be useless, regardless of how much CPU work
 * happens between them. (An earlier read of this file's own flakiness blamed
 * `SEFF_EQUIVALENCE`'s module-collect-time CPU cost instead — PR #119's first
 * review round deferred that computation into a lazy function, which did not
 * fix the instability, because CPU cost was never the cause; #68 was. That fix
 * stays, since a per-test lazy computation is the right shape regardless, but
 * the actual remedy is this split.) `sEff`'s own two GPU-dispatch cases
 * (`SEFF_CASES` in `wgsl-seff.test.ts`) plus its five CPU-only checks are moved
 * out entire, not because they are less important, but because isolating
 * "issue #117's whole point" in its own file is also the more legible split —
 * the same shape `ops/rope/wgsl-heads.test.ts` and `wgsl-cache.test.ts` chose
 * over an arbitrary case-count bisection.
 */

/** Two files rather than two entry points in one, for the reason `attention`
 * measured and wrote down (#46): `layout: "auto"` drops the bindings an entry
 * point does not statically name, the harness binds group 0 as a contiguous
 * list, and the mismatch reads back zeros without raising. */
export const scoresCode = kernel(import.meta.url, "scores");
export const contextCode = kernel(import.meta.url, "context");

/** Must match `WORKGROUP_SIZE` in both kernels. */
export const WG = 256;

export const wave = (n: number, k: number, phase = 0) =>
  Float32Array.from({ length: n }, (_, i) => Math.sin(i * k + phase) * 1.5);

/**
 * Measured, not widened until green.
 *
 * The arithmetic is `attention`'s — the same three f32 sums (the Q.K dots, the
 * softmax row sum, the probs@V sum) against an f64 reference — and sharing KV
 * heads changes which K rows are read, not how they are summed. Worst actually
 * observed across every case in `wgsl.test.ts`, from an instrumented run with
 * the tolerance disabled:
 *
 *   probs   rel 1.71e-6   abs 2.38e-7
 *   output  rel 1.88e-3   abs 9.54e-7
 *
 * Identical across repeated runs, to every digit — this device's f32 reduction
 * order is fixed, so these are the numbers and not a sample of them.
 *
 * `abs` is the measure that carries this op, for the reason `attention` records:
 * `output` sums probabilities against a sine wave, so elements cancel to near
 * zero and a relative bound against a near-zero true value stops meaning
 * anything. That 1.88e-3 relative is 9.54e-7 absolute — one ulp at exponent
 * -20 — on an element whose true value is near zero. `agree` passes an element
 * on either measure, so it passes on absolute, and a relative-only bound of 1e-5
 * would fail a correct kernel.
 *
 * abs 2e-6 is 2.1x the worst absolute deviation observed. Deliberately not more:
 * everything this file exists to catch moves elements by order 0.1 to 10, five
 * orders of magnitude clear of the bound, so there is nothing to buy by widening
 * it. Confirmed by mutation, not assumed.
 */
export const TOLERANCE = { rel: 1e-5, abs: 2e-6 };

export type Prepared = {
  name: string;
  args: GroupedAttentionArgs;
  want: GroupedAttentionResult;
  scoresParams: ArrayBuffer;
  contextParams: ArrayBuffer;
  /**
   * What gets bound at binding 2. Not optional: a bias of zeros *is* "no mask",
   * which is what torch's own reference does (`attn_bias = torch.zeros(L, S)`,
   * added unconditionally). See `ops/attention/wgsl.test.ts`.
   */
  maskData: Float32Array;
};

/**
 * Every input, expected value and uniform is built at module scope by this
 * function's callers; the test bodies only dispatch and compare.
 *
 * Required, not stylistic — #49. After `useGpu`'s `beforeAll` has made the
 * device, a test body that spends ~10ms before its first dispatch kills the
 * vitest worker and the file reports nothing at all. The same work before the
 * device exists is fine at a full second. This op walks B x H x L x S x D twice
 * over (once for the reference, once for the expanded cross-check), so it is
 * squarely in that trap.
 */
export function prepare(name: string, args: GroupedAttentionArgs): Prepared {
  const [mb, mh, mr] = args.maskShape ?? [args.B, 1, 1];
  return {
    name,
    args,
    want: groupedAttention(args),
    maskData: args.mask ?? new Float32Array(args.B * args.S),
    scoresParams: params([
      ["u32", args.H],
      ["u32", args.kvHeads],
      ["u32", args.L],
      ["u32", args.S],
      ["u32", args.D],
      ["f32", args.scale ?? defaultScale(args.D)],
      ["u32", args.causal ? 1 : 0],
      ["i32", args.queryOffset ?? 0],
      ["u32", mb],
      ["u32", mh],
      ["u32", mr],
      ["u32", args.sEff ?? args.S],
    ]),
    contextParams: params([
      ["u32", args.H],
      ["u32", args.kvHeads],
      ["u32", args.L],
      ["u32", args.S],
      ["u32", args.Dv],
      ["u32", args.sEff ?? args.S],
    ]),
  };
}

/**
 * Runs the real two-dispatch pipeline and checks both halves.
 *
 * The attention matrix fed to the second dispatch is the one the first dispatch
 * produced, not the reference's, so the seam between them is exercised.
 */
export async function check(run: Runner["run"], p: Prepared): Promise<void> {
  const { B, H, L, S, Dv } = p.args;

  const [raw] = await run({
    code: scoresCode,
    bindings: [
      { kind: "storage", data: p.args.q },
      { kind: "storage", data: p.args.k },
      { kind: "storage", data: p.maskData },
      { kind: "out", type: "f32", length: B * H * L * S },
      { kind: "uniform", data: p.scoresParams },
    ],
    workgroups: [L, H, B],
  });
  const probs = raw as Float32Array;
  const worst = agree(probs, p.want.probs, TOLERANCE);
  expect(worst, worst ? `probs: ${JSON.stringify(worst)}` : undefined).toBeNull();

  await expectAgrees(
    run,
    {
      code: contextCode,
      bindings: [
        { kind: "storage", data: probs },
        { kind: "storage", data: p.args.v },
        { kind: "out", type: "f32", length: B * H * L * Dv },
        { kind: "uniform", data: p.contextParams },
      ],
      workgroups: [L, H, B],
    },
    [p.want.output],
    TOLERANCE,
  );
}
