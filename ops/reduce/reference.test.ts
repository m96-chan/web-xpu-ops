import { describe, expect, it } from "vitest";
import { REDUCE, reduce } from "./reference.js";

/**
 * The parts of the contract a kernel cannot be asked about.
 *
 * A shader has no way to raise, so the empty-axis refusal for max and min
 * lives only here — and it is the edge most likely to be quietly "fixed" into
 * returning an identity by someone who has not read why PyTorch refuses.
 * Touches no GPU, so it costs the suite nothing.
 *
 * Every expectation below was measured against torch 2.10.0+cu128.
 */
describe("reduce / reference", () => {
  it("matches PyTorch on an empty axis", () => {
    const empty = new Float32Array(0);
    const shape = { outer: 2, axis: 0, inner: 1 };

    // torch.sum(torch.empty(0)) === tensor(0.)
    expect(Array.from(reduce({ input: empty, ...shape, kind: REDUCE.sum }))).toEqual([0, 0]);

    // torch.mean(torch.empty(0)) === tensor(nan): PyTorch divides by the count
    // without guarding it, and so does this.
    expect(Array.from(reduce({ input: empty, ...shape, kind: REDUCE.mean })).every(Number.isNaN)).toBe(true);

    // torch.amax(torch.empty(0), dim=0) raises IndexError. There is no finite
    // identity that is also a valid answer, so returning -Infinity here would
    // hand back a number PyTorch considers undefined.
    expect(() => reduce({ input: empty, ...shape, kind: REDUCE.max })).toThrow(/non-zero size/);
    expect(() => reduce({ input: empty, ...shape, kind: REDUCE.min })).toThrow(/non-zero size/);
  });

  it("propagates NaN through all four, as PyTorch does", () => {
    // torch.sum / mean / amax / amin of [1, nan, 3] are all nan. The WGSL
    // kernel does not match on max and min; see wgsl/kernel.wgsl for the
    // measurement and why the gap is left open.
    const input = Float32Array.from([1, NaN, 3]);
    const shape = { outer: 1, axis: 3, inner: 1 };
    for (const kind of [REDUCE.sum, REDUCE.max, REDUCE.min, REDUCE.mean]) {
      expect(reduce({ input, ...shape, kind })[0]).toBeNaN();
    }
  });

  it("means by the axis length, not by the count of finite elements", () => {
    // Dividing by the finite count is nanmean, a different function. With one
    // NaN in four, the two differ: 3 vs 4 in the denominator.
    const input = Float32Array.from([1, 2, 3, 4]);
    expect(reduce({ input, outer: 1, axis: 4, inner: 1, kind: REDUCE.mean })[0]).toBe(2.5);
  });

  it("refuses an input whose length does not match the shape", () => {
    // Silently reducing the wrong window is worse than not running.
    expect(() => reduce({ input: new Float32Array(5), outer: 1, axis: 3, inner: 1, kind: REDUCE.sum }))
      .toThrow(/expected 3 elements, got 5/);
  });
});
