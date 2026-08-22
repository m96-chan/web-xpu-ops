import { describe, expect, it } from "vitest";
import { nearestUpsample2d, nearestUpsampleScale } from "./reference.js";

/**
 * The reference pinned against PyTorch, and the parts of the contract a kernel
 * cannot be asked about.
 *
 * Every expectation below was measured against torch 2.10.0+cu128 by running
 * `torch.nn.functional.interpolate(x, size=..., mode='nearest')`, not read off
 * the docs. Nearest resampling does no arithmetic, so every golden is an exact
 * copy of an input element and a mismatch is a mismatch rather than a rounding
 * argument. Touches no GPU.
 */
describe("nearestUpsample2d / reference", () => {
  /**
   * The source row each destination row copies, read out of the op itself:
   * a `[1, 1, H, 1]` input whose row `r` holds the value `r` makes the output
   * *be* the index map. Compact enough to write the torch answer down, and it
   * pins the index formula rather than a picture that happens to look right.
   */
  const rowMap = (H: number, outH: number): number[] =>
    Array.from(
      nearestUpsample2d({
        input: Float32Array.from({ length: H }, (_, r) => r),
        N: 1,
        C: 1,
        H,
        W: 1,
        outH,
        outW: 1,
      }),
    );

  /** The same, along W: `[1, 1, 1, W]`. */
  const colMap = (W: number, outW: number): number[] =>
    Array.from(
      nearestUpsample2d({
        input: Float32Array.from({ length: W }, (_, c) => c),
        N: 1,
        C: 1,
        H: 1,
        W,
        outH: 1,
        outW,
      }),
    );

  it("doubles by repeating each row and column once", () => {
    // F.interpolate(arange(1,13).reshape(1,1,3,4), size=(6,8), mode='nearest')
    // The 2x case every VAE / codec decoder actually asks for, and the one
    // shape where the size path and the scale_factor path cannot disagree.
    const got = nearestUpsample2d({
      input: Float32Array.from({ length: 12 }, (_, i) => i + 1),
      N: 1,
      C: 1,
      H: 3,
      W: 4,
      outH: 6,
      outW: 8,
    });
    expect(Array.from(got)).toEqual([
      1, 1, 2, 2, 3, 3, 4, 4,
      1, 1, 2, 2, 3, 3, 4, 4,
      5, 5, 6, 6, 7, 7, 8, 8,
      5, 5, 6, 6, 7, 7, 8, 8,
      9, 9, 10, 10, 11, 11, 12, 12,
      9, 9, 10, 10, 11, 11, 12, 12,
    ]);
  });

  it("maps each axis independently at a non-integer ratio", () => {
    // F.interpolate(arange(1,13).reshape(1,1,3,4), size=(5,7), mode='nearest')
    // 3 -> 5 and 4 -> 7 in one call: neither ratio is an integer and they are
    // not the same ratio, so an implementation that derives one scale for both
    // axes cannot pass this.
    const got = nearestUpsample2d({
      input: Float32Array.from({ length: 12 }, (_, i) => i + 1),
      N: 1,
      C: 1,
      H: 3,
      W: 4,
      outH: 5,
      outW: 7,
    });
    expect(Array.from(got)).toEqual([
      1, 1, 2, 2, 3, 3, 4,
      1, 1, 2, 2, 3, 3, 4,
      5, 5, 6, 6, 7, 7, 8,
      5, 5, 6, 6, 7, 7, 8,
      9, 9, 10, 10, 11, 11, 12,
    ]);
  });

  it("keeps N, C, H and W apart in a [N, C, H, W] buffer", () => {
    // F.interpolate(arange(1,25).reshape(2,2,2,3), size=(3,5), mode='nearest')
    // Every element of the input is distinct, so a plane read from the wrong
    // batch or channel lands as a wrong number rather than a coincidence. The
    // channel count is not 1 and neither is the batch, because with either at 1
    // the two strides that could be swapped are equal.
    const got = nearestUpsample2d({
      input: Float32Array.from({ length: 24 }, (_, i) => i + 1),
      N: 2,
      C: 2,
      H: 2,
      W: 3,
      outH: 3,
      outW: 5,
    });
    expect(Array.from(got)).toEqual([
      1, 1, 2, 2, 3, 1, 1, 2, 2, 3, 4, 4, 5, 5, 6,
      7, 7, 8, 8, 9, 7, 7, 8, 8, 9, 10, 10, 11, 11, 12,
      13, 13, 14, 14, 15, 13, 13, 14, 14, 15, 16, 16, 17, 17, 18,
      19, 19, 20, 20, 21, 19, 19, 20, 20, 21, 22, 22, 23, 23, 24,
    ]);
  });

  describe("the index map torch actually computes", () => {
    it("matches torch on the plain ratios", () => {
      expect(rowMap(3, 6)).toEqual([0, 0, 1, 1, 2, 2]);
      expect(rowMap(4, 5)).toEqual([0, 0, 1, 2, 3]);
      expect(rowMap(3, 5)).toEqual([0, 0, 1, 1, 2]);
      expect(rowMap(7, 21)).toEqual([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6]);
      expect(rowMap(1, 4)).toEqual([0, 0, 0, 0]);
      expect(rowMap(5, 5)).toEqual([0, 1, 2, 3, 4]);
      // Along W as well: the same formula, and the same goldens, applied to the
      // other axis. A kernel that computes W's source index from H's scale
      // passes every square shape.
      expect(colMap(3, 6)).toEqual([0, 0, 1, 1, 2, 2]);
      expect(colMap(4, 5)).toEqual([0, 0, 1, 2, 3]);
    });

    /**
     * The f32 multiply, pinned. Destination row 23 of 46 has exact source
     * `23 * 14 / 46 = 7`, dead on a pixel boundary — and torch answers **6**,
     * because `float32(14/46)` is a hair below `14/46`. Exact integer
     * arithmetic answers 7 and is wrong by a whole row against torch.
     *
     * This is the test that decides which of the two the reference is, so it is
     * written with the disagreement spelled out rather than as one more golden.
     */
    it("floors an f32 product, not an exact ratio (H=14 -> 46)", () => {
      const map = rowMap(14, 46);
      expect(map).toEqual([
        0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6,
        6, 7, 7, 7, 8, 8, 8, 9, 9, 9, 10, 10, 10, 10, 11, 11, 11, 12, 12, 12,
        13, 13, 13,
      ]);
      expect(map[23]).toBe(6);
      // What `(23 * 14 / 46) | 0` — exact integer arithmetic — would have said.
      expect(Math.floor((23 * 14) / 46)).toBe(7);
      // And along W, measured separately: the axes share a formula, not code.
      expect(colMap(14, 46)[23]).toBe(6);
    });

    /** The same divergence with the boundary as late as it gets: 2 -> 82. */
    it("puts the 2 -> 82 boundary at 42, where torch puts it (not 41)", () => {
      const map = rowMap(2, 82);
      expect(map.indexOf(1)).toBe(42);
      expect(Math.floor((41 * 2) / 82)).toBe(1); // exact arithmetic would start row 1 here
      expect(map[41]).toBe(0);
    });

    /**
     * The other half of "the product is f32 too": here the exact product is
     * `25 * 2 / 50 = 1` and the f32 scale is a hair *below* `1/25`, so the f64
     * product `25 * float32(0.04)` lands just under 1.0 and floors to 0 —
     * while rounding that product to f32 first snaps it to exactly 1.0 and
     * floors to 1. torch answers 1: its boundary is at destination 25.
     *
     * So the scale being f32 is not enough; the multiply has to be f32 as
     * well, in both directions. 14 -> 46 above pins the down side.
     */
    it("rounds the product to f32, not only the scale (H=2 -> 50)", () => {
      const map = rowMap(2, 50);
      expect(map.indexOf(1)).toBe(25);
      // What multiplying in f64 and flooring that would have said.
      expect(Math.floor(25 * Math.fround(2 / 50))).toBe(0);
      expect(colMap(2, 50).indexOf(1)).toBe(25);
    });

    /**
     * `mode='nearest'` and `mode='nearest-exact'` are two functions, and this
     * op is the first. Both measured; they differ at destination row 2.
     */
    it("is 'nearest', not 'nearest-exact' (H=3 -> 7)", () => {
      expect(rowMap(3, 7)).toEqual([0, 0, 0, 1, 1, 2, 2]);
      expect(rowMap(3, 7)).not.toEqual([0, 0, 1, 1, 1, 2, 2]); // nearest-exact
    });
  });

  it("copies values rather than blending them", () => {
    // Neighbouring values chosen so that any average of two of them is a value
    // this op must never produce: every output has to be one of the inputs.
    const input = Float32Array.from([1, 1000, -7, 0.5]);
    const got = Array.from(
      nearestUpsample2d({ input, N: 1, C: 1, H: 1, W: 4, outH: 1, outW: 9 }),
    );
    expect(got).toEqual([1, 1, 1, 1000, 1000, -7, -7, 0.5, 0.5]);
    for (const v of got) expect([1, 1000, -7, 0.5]).toContain(v);
  });

  describe("what it refuses", () => {
    const base = { input: Float32Array.from({ length: 6 }, (_, i) => i), N: 1, C: 1, H: 2, W: 3 };

    it("throws when an axis shrinks, rather than silently downsampling", () => {
      // The formula would happily answer; ISSUE #146 scopes downsampling out
      // and nothing here is measured against torch for it.
      expect(() => nearestUpsample2d({ ...base, outH: 1, outW: 3 })).toThrow(/shrinks an axis/);
      expect(() => nearestUpsample2d({ ...base, outH: 2, outW: 2 })).toThrow(/shrinks an axis/);
      // Equal is not shrinking: resampling one axis only has to work.
      expect(() => nearestUpsample2d({ ...base, outH: 2, outW: 6 })).not.toThrow();
    });

    it("throws on a fractional output size", () => {
      // `outH: H * 1.5` without the floor — the mistake a caller who wanted a
      // scale factor actually makes. Allocating `Math.floor` of it silently
      // would return an array whose shape nothing downstream can explain.
      expect(() => nearestUpsample2d({ ...base, outH: 3, outW: 4.5 })).toThrow(/positive integers/);
      expect(() => nearestUpsample2d({ ...base, outH: 0, outW: 3 })).toThrow(/positive integers/);
    });

    it("throws when the input length does not match [N, C, H, W]", () => {
      expect(() => nearestUpsample2d({ ...base, C: 2, outH: 4, outW: 6 })).toThrow(/expected 12 input elements/);
    });
  });

  describe("nearestUpsampleScale", () => {
    it("rounds the ratio to f32 once, as torch's compute_scales_value does", () => {
      expect(nearestUpsampleScale(14, 46)).toBe(Math.fround(14 / 46));
      // Below the exact ratio — this is the bit that moves row 23 to 6.
      expect(nearestUpsampleScale(14, 46)).toBeLessThan(14 / 46);
      // Powers of two are exact, which is why the 2x case has no such trouble.
      expect(nearestUpsampleScale(3, 6)).toBe(0.5);
      expect(nearestUpsampleScale(5, 5)).toBe(1);
    });
  });
});
