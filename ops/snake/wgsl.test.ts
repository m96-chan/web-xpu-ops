import { describe } from "vitest";
import { expectAgrees, gpuTest, kernel, params, useGpu } from "../../harness/index.js";
import { snake } from "./reference.js";

const code = kernel(import.meta.url);

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
