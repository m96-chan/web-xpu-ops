import { describe } from "vitest";
import { expectAgrees, gpuTest, kernel, params, useGpu } from "../../harness/index.js";
import { snake, snakeBeta } from "./reference.js";

const code = kernel(import.meta.url);
const betaCode = kernel(import.meta.url, "beta");

/**
 * Every case is built here, at module scope, before a device exists.
 *
 * Not style: CPU work *between* dispatches kills this binding. Measured while
 * writing this file — forty back-to-back dispatches are fine, seven with a
 * 4096-element f64 reference computed between them are not (#49).
 */
interface Case {
  label: string;
  N: number;
  C: number;
  L: number;
  alpha: Float32Array;
  padded: Float32Array;
  expected: Float32Array;
  workgroups: number;
}

const WG = 256;

/**
 * A non-zero value in every element past the ones the kernel is meant to touch.
 *
 * This device returns 0 for an out-of-range read, so a kernel that dropped its
 * bounds check would read zeros, and *every* activation here maps 0 to 0 — the
 * mutation would write the right answer by accident and survive. Feeding it a
 * sentinel instead makes the tail say what happened: `snake(4.5)` with the
 * out-of-range `alpha` reading 0 is 4.5, against an expected 0.
 */
const SENTINEL = 4.5;

/** Deliberately non-uniform, and spanning the magnitudes a trained α covers. */
const ALPHAS = [0.01, 0.37, 1, 2.5, 10, 50, 0.8];

function build(label: string, N: number, C: number, L: number): Case {
  const total = N * C * L;
  const workgroups = Math.ceil(total / WG);
  const alpha = Float32Array.from({ length: C }, (_, c) => ALPHAS[c % ALPHAS.length]!);
  const input = Float32Array.from({ length: total }, (_, i) => Math.sin(i * 0.37) * 10);
  const padded = new Float32Array(workgroups * WG).fill(SENTINEL);
  padded.set(input);
  const expected = new Float32Array(workgroups * WG);
  expected.set(snake({ input, alpha, N, C, L }));
  return { label, N, C, L, alpha, padded, expected, workgroups };
}

const cases = [
  // Below one workgroup, and C > 1 with L > 1 so the channel index is a real
  // division rather than an identity.
  build("N=1 C=4 L=7", 1, 4, 7),
  // N > 1: the batch axis has to be skipped over, not folded into the channel.
  build("N=2 C=3 L=5", 2, 3, 5),
  // Exactly one workgroup.
  build("N=1 C=2 L=128", 1, 2, 128),
  // Not a multiple of 256, so the bounds check is load-bearing.
  build("N=2 C=5 L=30", 2, 5, 30),
  build("N=3 C=7 L=13", 3, 7, 13),
];

const bindings = (c: Case) =>
  [
    { kind: "storage", data: c.padded },
    { kind: "storage", data: c.alpha },
    { kind: "out", type: "f32", length: c.expected.length },
    {
      kind: "uniform",
      data: params([
        ["u32", c.N],
        ["u32", c.C],
        ["u32", c.L],
      ]),
    },
  ] as const;

/**
 * α = 0: the whole reason the epsilon exists.
 *
 * Upstream's scale is `(α + 1e-9).reciprocal()`, so at α = 0 it is 1e9 and the
 * sine it multiplies is exactly 0 — the answer is x. Written as `1/α` it is
 * `inf * 0`, which is NaN, and no tolerance rescues a NaN. Channel 1 carries an
 * ordinary α in the same dispatch so this cannot pass by the kernel ignoring
 * the buffer.
 */
const zeroAlpha = (() => {
  const [N, C, L] = [1, 2, 6];
  const alpha = Float32Array.from([0, 1.5]);
  const input = Float32Array.from({ length: N * C * L }, (_, i) => (i - 5.5) * 0.8);
  const padded = new Float32Array(WG).fill(SENTINEL);
  padded.set(input);
  const expected = new Float32Array(WG);
  expected.set(snake({ input, alpha, N, C, L }));
  return { label: "α = 0", N, C, L, alpha, padded, expected, workgroups: 1 };
})();

/**
 * The one regime where f32 can see the epsilon's *value* rather than only its
 * presence.
 *
 * At α = 1e-9 the scale is 1/(2e-9) rather than 1/1e-9, a factor of two, but
 * `sin²(αx)` is around α²x² — so the term it scales is only visible once |x| is
 * large enough. At x = 1e6 the term is ≈ 500 against an output of ≈ 1,000,500;
 * dropping the epsilon doubles it to 1000 and moves the answer by 5e-4
 * relative. Everywhere else — α = 1e-9 at x = 10, or any α of ordinary size —
 * the three candidate spellings (`1/(α+ε)`, `1/α`, `α/(α²+ε)`) agree to well
 * inside an f32 ulp, which is precisely why the expression has to be copied
 * from upstream rather than reasoned about.
 */
const tinyAlpha = (() => {
  const [N, C, L] = [1, 2, 4];
  const alpha = Float32Array.from([1e-9, 3]);
  const input = Float32Array.from([1e6, -1e6, 5e5, -5e5, 1.5, -2.5, 0.25, 7]);
  const padded = new Float32Array(WG).fill(SENTINEL);
  padded.set(input);
  const expected = new Float32Array(WG);
  expected.set(snake({ input, alpha, N, C, L }));
  return { label: "α = 1e-9", N, C, L, alpha, padded, expected, workgroups: 1 };
})();

describe("snake / wgsl", () => {
  useGpu();

  for (const c of [...cases, zeroAlpha, tinyAlpha]) {
    gpuTest(`agrees with the reference at ${c.label}`, async (run) => {
      // Default tolerance, and it is not a coincidence that it holds. Measured
      // on this device: `sin` is accurate to about 1.2e-7 × |argument|, so
      // sin(αx) is wrong by 1.2e-7·α|x| — an error that grows with α. The op
      // then divides by α, which divides it straight back out: the whole
      // expression is off by ≈ 2.4e-7·|x|, independent of α. Measured over
      // α ∈ [0.01, 50] with |x| ≤ 10 (so |αx| up to 500): worst absolute
      // 5.64e-6, worst relative 1.37e-6. The absolute figure is above the
      // default 1e-6 and the relative one is well under the default 1e-5,
      // which is what carries these, because the large absolute errors land
      // where |output| ≈ |x| is large.
      await expectAgrees(
        run,
        { code, bindings: [...bindings(c)], workgroups: [c.workgroups] },
        [c.expected],
      );
    });
  }
});

/* ------------------------------------------------------------------------ *
 * snake_beta — the second entry point (#88)
 * ------------------------------------------------------------------------ */

interface BetaCase extends Case {
  beta: Float32Array;
}

/**
 * Deliberately **different from `ALPHAS`, element for element**.
 *
 * The whole point of SnakeBeta is that the sine's frequency and the amplitude
 * it is added at are separate numbers. A test where they happen to match is a
 * test of `snake` wearing a second buffer: swapping the two bindings, or
 * reading α where β belongs, would sail through it.
 */
const BETAS = [3, 0.05, 0.5, 12, 1, 0.2, 7];

function buildBeta(label: string, N: number, C: number, L: number): BetaCase {
  const total = N * C * L;
  const workgroups = Math.ceil(total / WG);
  const alpha = Float32Array.from({ length: C }, (_, c) => ALPHAS[c % ALPHAS.length]!);
  const beta = Float32Array.from({ length: C }, (_, c) => BETAS[c % BETAS.length]!);
  const input = Float32Array.from({ length: total }, (_, i) => Math.sin(i * 0.37) * 10);
  const padded = new Float32Array(workgroups * WG).fill(SENTINEL);
  padded.set(input);
  const expected = new Float32Array(workgroups * WG);
  expected.set(snakeBeta({ input, alpha, beta, N, C, L }));
  return { label, N, C, L, alpha, beta, padded, expected, workgroups };
}

const betaCases = [
  // C > 1 with L > 1, so both α and β are indexed by a real division.
  buildBeta("N=1 C=4 L=7", 1, 4, 7),
  // The batch axis has to be skipped over by *both* parameter lookups.
  buildBeta("N=2 C=3 L=5", 2, 3, 5),
  // Not a multiple of 256: the bounds check is load-bearing, and the sentinel
  // tail is `snake_beta(4.5)` with α and β read out of range rather than 0.
  buildBeta("N=2 C=5 L=30", 2, 5, 30),
];

const betaBindings = (c: BetaCase) =>
  [
    { kind: "storage", data: c.padded },
    { kind: "storage", data: c.alpha },
    { kind: "storage", data: c.beta },
    { kind: "out", type: "f32", length: c.expected.length },
    {
      kind: "uniform",
      data: params([
        ["u32", c.N],
        ["u32", c.C],
        ["u32", c.L],
      ]),
    },
  ] as const;

/**
 * β = α, which must reproduce `snake` exactly — that is the claim that this
 * generalises rather than replaces.
 *
 * Checked against `snake`'s **own** reference rather than against
 * `snakeBeta`'s, so it is an equivalence between the two functions and not a
 * restatement of one of them.
 */
const betaEqualsAlpha = (() => {
  const [N, C, L] = [1, 3, 9];
  const alpha = Float32Array.from([0.37, 2.5, 10]);
  const input = Float32Array.from({ length: N * C * L }, (_, i) => Math.sin(i * 0.61) * 8);
  const padded = new Float32Array(WG).fill(SENTINEL);
  padded.set(input);
  const expected = new Float32Array(WG);
  expected.set(snake({ input, alpha, N, C, L }));
  return {
    label: "β = α reproduces snake",
    N,
    C,
    L,
    alpha,
    beta: alpha,
    padded,
    expected,
    workgroups: 1,
  };
})();

/**
 * α = 0 with an ordinary β, and β = 0 with an ordinary α. They are **not**
 * symmetric, and that asymmetry is the reason the epsilon moved.
 *
 * At α = 0 the sine is 0, so the answer is x whatever β does — benign, exactly
 * as in `snake`. At β = 0 the scale is 1e9 and it multiplies a sine that is
 * *not* zero, so the output is enormous and finite. Written as `1/β` it would
 * be `inf · sin²` — still enormous, but infinite, and no tolerance survives
 * that. Channel 2 carries ordinary values in the same dispatch so neither case
 * can pass by the kernel ignoring a buffer.
 *
 * Reachable through this API, not through a checkpoint: MioCodec stores log β
 * and the loader exponentiates, and `exp` is never 0. The guard is defensive.
 */
const zeroParams = (() => {
  const [N, C, L] = [1, 3, 6];
  const alpha = Float32Array.from([0, 1.5, 2]);
  const beta = Float32Array.from([2, 0, 0.75]);
  const input = Float32Array.from({ length: N * C * L }, (_, i) => (i % 6) * 0.4 - 1);
  const padded = new Float32Array(WG).fill(SENTINEL);
  padded.set(input);
  const expected = new Float32Array(WG);
  expected.set(snakeBeta({ input, alpha, beta, N, C, L }));
  return { label: "α = 0 and β = 0", N, C, L, alpha, beta, padded, expected, workgroups: 1 };
})();

describe("snake_beta / wgsl", () => {
  useGpu();

  for (const c of [...betaCases, betaEqualsAlpha, zeroParams]) {
    gpuTest(`agrees with the reference at ${c.label}`, async (run) => {
      // Default tolerance, and `snake`'s argument for why does **not** carry
      // over. There, `sin(αx)`'s error of ≈1.2e-7·α|x| was divided straight
      // back out by the `1/α` in front, leaving ≈2.4e-7·|x| whatever α was.
      // Here the divisor is β, so nothing cancels and the absolute error
      // scales as ≈2.4e-7·(α/β)·|x| — these cases run α/β up to 200 (α = 10
      // against β = 0.05).
      //
      // What holds it together is that the same 1/β amplifies the output, so
      // the *relative* error stays where `snake`'s was. Measured on this
      // device with a zero tolerance, worst element per case:
      //
      //   N=1 C=4 L=7      abs 6.68e-6   rel 1.44e-6
      //   N=2 C=3 L=5      abs 2.09e-7   rel 1.04e-6
      //   N=2 C=5 L=30     abs 6.20e-6   rel 4.60e-6
      //   β = α            abs 1.19e-7   rel 1.29e-7
      //   α = 0 and β = 0  abs 1.34e-7   rel 1.31e-6
      //
      // So the absolute figures sit above the default 1e-6 and the relative
      // ones under the default 1e-5, which is what carries these — the same
      // shape as `snake`, reached for a different reason. Measured over the
      // ratios above rather than proven, so a case with a far smaller β would
      // need measuring again rather than assuming.
      await expectAgrees(
        run,
        { code: betaCode, bindings: [...betaBindings(c)], workgroups: [c.workgroups] },
        [c.expected],
      );
    });
  }
});
