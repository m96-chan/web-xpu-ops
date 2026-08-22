import { describe, expect, it } from "vitest";
import { ELEMENTWISE, elementwise } from "../elementwise/reference.js";
import { axpy } from "./reference.js";

/**
 * The reference's own contract — the parts a GPU dispatch cannot ask about:
 * the operand order, the out-of-place promise, the length check, and the
 * PyTorch conformance this op claims (rule 7). `ops/axpy/wgsl.test.ts` covers
 * kernel-vs-reference agreement.
 *
 * Every expected value in the `torch.add` block below was **produced by
 * PyTorch**, not derived here: `torch 2.10.0+cu128`, CPU, `float32`, with
 * `x = [1.5, -2.0, inf, nan, 3.0]` and `y = [0.5, 0.25, 1.0, 2.0, -3.0]`.
 */

const X = Float32Array.from([1.5, -2, Infinity, NaN, 3]);
const Y = Float32Array.from([0.5, 0.25, 1, 2, -3]);

describe("axpy", () => {
  it("computes y + a*x elementwise", () => {
    // Hand-computable, and deliberately asymmetric: swapping x and y (a very
    // easy slip in an op whose two operands have the same shape) gives
    // [0.5*2 + 1.5, ...] = [2.5, ...] rather than [1.75, ...].
    const x = Float32Array.from([1.5, -2, 0.25, 8]);
    const y = Float32Array.from([1, 1, 1, 1]);
    expect(Array.from(axpy({ x, y, a: 0.5 }))).toEqual([1.75, 0, 1.125, 5]);
  });

  it("is out-of-place: neither input is written", () => {
    // The kernel has an in-place entry point (`wgsl/inplace.wgsl`) and this
    // function does not — see the reference's own doc for why. If that ever
    // changed by accident, a caller reusing `y` for the next step would read
    // an already-updated latent and the error would compound silently over a
    // diffusion loop rather than announce itself.
    const x = Float32Array.from([1, 2, 3]);
    const y = Float32Array.from([10, 20, 30]);
    const out = axpy({ x, y, a: 2 });
    expect(Array.from(x)).toEqual([1, 2, 3]);
    expect(Array.from(y)).toEqual([10, 20, 30]);
    expect(Array.from(out)).toEqual([12, 24, 36]);
  });

  it("at a = 1 is exactly elementwise add", () => {
    // The claim that this generalises rather than replaces, checked against
    // `ops/elementwise`'s *own* reference rather than a restatement of this
    // one — the same shape as `snake_beta`'s "β = α reproduces snake" test.
    const x = Float32Array.from({ length: 64 }, (_, i) => Math.sin(i * 0.37) * 5);
    const y = Float32Array.from({ length: 64 }, (_, i) => Math.cos(i * 0.11) * 3);
    expect(Array.from(axpy({ x, y, a: 1 }))).toEqual(
      Array.from(elementwise({ a: y, b: x, kind: ELEMENTWISE.add })),
    );
  });

  it("at a = 0 returns y for finite x — and NaN where x is not finite", () => {
    // Not a special case in the code, and deliberately not: `0 * inf` is NaN
    // and PyTorch propagates it rather than short-circuiting `alpha == 0`.
    // Measured, not assumed — `torch.add(y, x, alpha=0.0)` on the tensors at
    // the top of this file returns [0.5, 0.25, nan, nan, -3.0].
    const out = axpy({ x: X, y: Y, a: 0 });
    expect(Array.from(out.slice(0, 2))).toEqual([0.5, 0.25]);
    expect(out[2]).toBeNaN();
    expect(out[3]).toBeNaN();
    expect(out[4]).toBe(-3);
  });

  it("matches torch.add(y, x, alpha=a) for a negative a", () => {
    // torch: [-3.25, 5.25, -inf, nan, -10.5]
    const out = axpy({ x: X, y: Y, a: -2.5 });
    expect(Array.from(out.slice(0, 2))).toEqual([-3.25, 5.25]);
    expect(out[2]).toBe(-Infinity);
    expect(out[3]).toBeNaN();
    expect(out[4]).toBe(-10.5);
  });

  it("matches torch.add(y, x, alpha=a) for a fractional a", () => {
    // torch: [0.6875, 0.0, inf, nan, -2.625]
    const out = axpy({ x: X, y: Y, a: 0.125 });
    expect(Array.from(out.slice(0, 2))).toEqual([0.6875, 0]);
    expect(out[2]).toBe(Infinity);
    expect(out[3]).toBeNaN();
    expect(out[4]).toBe(-2.625);
  });

  it("rounds once, the way PyTorch does, not twice the way multiply-then-add does", () => {
    // The one numerical claim this op makes, and the reason the reference
    // computes in f64 and rounds on store.
    //
    // `a = f32(0.1)`, `x = 3`: the exact product is 0.3000000044703484, which
    // f32 cannot hold — rounding it to f32 first lands on f32(0.3) =
    // 0.30000001192092896, an error of +2^-27. With `y = -f32(0.3)` that error
    // is the entire answer: multiply-then-add cancels to exactly **0**, while
    // one rounding of `y + a*x` gives **-2^-27**. Not an ulp apart — one of
    // them is zero.
    //
    // Which one PyTorch returns is measured, not assumed: `torch 2.10.0+cu128`
    // returns `-7.450580596923828e-09` (= -2^-27) on CPU *and* on CUDA. Over
    // 100,000 random normal f32 pairs at a = 0.37 this f64-then-store
    // reference reproduced `torch.add`'s output bit for bit on every element,
    // while the multiply-then-add spelling differed on 20,732 of them.
    const x = Float32Array.from([3]);
    const y = Float32Array.from([-Math.fround(0.3)]);
    expect(axpy({ x, y, a: Math.fround(0.1) })[0]).toBe(-(2 ** -27));
  });

  it("handles a single element", () => {
    expect(Array.from(axpy({ x: Float32Array.from([3]), y: Float32Array.from([4]), a: 2 }))).toEqual([10]);
  });

  it("handles an array far larger than any workgroup", () => {
    // 262144 = the element count of one 128x128 latent at 16 channels, the
    // shape this op exists for. Checked at both ends and in the middle rather
    // than by a second loop, which would only restate the implementation.
    const n = 262_144;
    const x = Float32Array.from({ length: n }, (_, i) => (i % 17) - 8);
    const y = Float32Array.from({ length: n }, (_, i) => (i % 5) * 0.25);
    const out = axpy({ x, y, a: -0.5 });
    expect(out).toHaveLength(n);
    for (const i of [0, 1, 12345, n - 1]) {
      expect(out[i]).toBe(Math.fround(y[i]! + -0.5 * x[i]!));
    }
  });

  it("refuses mismatched lengths rather than broadcasting", () => {
    // Scalar `a` is this op's only broadcast. Row broadcast is #150's, and a
    // silent short read here would be indistinguishable from it working.
    expect(() => axpy({ x: Float32Array.from([1, 2]), y: Float32Array.from([1, 2, 3]), a: 1 })).toThrow(
      /length mismatch/,
    );
  });
});
