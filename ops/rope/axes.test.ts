import { describe, expect, it } from "vitest";
import { AXES_CASES, axesCaseInput } from "./axes-cases.js";
import { rope, ropeAxes } from "./reference.js";

/**
 * `ropeAxes` against the implementation it exists to run, and against the
 * 1-axis op it generalises.
 *
 * Touches no GPU, so it costs the suite nothing and it runs on a machine with
 * no adapter — which matters here, because the two claims this file makes are
 * the ones the issue names: that the op agrees with Z-Image, and that
 * generalising `rope` did not change `rope`.
 *
 * `wgsl-axes.test.ts` holds the kernel's own cases.
 */

/** Worst absolute difference, and where. */
const worst = (a: ArrayLike<number>, b: ArrayLike<number>): { at: number; by: number } => {
  let found = { at: -1, by: 0 };
  for (let i = 0; i < a.length; i += 1) {
    const by = Math.abs(a[i]! - b[i]!);
    if (by > found.by) found = { at: i, by };
  }
  return found;
};

describe("ropeAxes / against Z-Image", () => {
  /**
   * Measured, not chosen. The worst absolute difference per case:
   *   zimage (theta 256, pos ≤ 3)     2.384e-7
   *   uneven (theta 256, pos ≤ 9)     1.192e-7
   *   two    (theta 10000, pos ≤ 5)   2.384e-7
   *   single (theta 10000, pos ≤ 2)   2.384e-7
   *
   * 2.384e-7 is `2^-22` — one f32 ulp at the input's magnitude of 2 — so what
   * this measures is the last bit and nothing else. That is the expected gap:
   * upstream rounds the *angle* to f32 before calling `cos`/`sin`
   * (`torch.outer(...).float()`, then `torch.polar`), while this reference
   * stays in f64 until the product is stored. The rotation is the same one.
   *
   * 1e-6 is four ulp — enough that a machine whose `sin` rounds the other way
   * does not turn this red, and far too tight to admit a wrong angle: getting
   * the axis's denominator wrong moves the `uneven` case by 4.80, and using
   * axis 0's position everywhere moves it by 4.51, both measured by mutation
   * on the `uneven` case (and by 2.6 and 3.3 on `zimage`).
   */
  const AGAINST_UPSTREAM = 1e-6;

  for (const kase of AXES_CASES) {
    const { name, why, axisDims, thetaBase, positions, N, numHeads, output } = kase;

    it(`matches Z-Image's own RoPE — ${name}`, () => {
      const headDim = axisDims.reduce((a, b) => a + b, 0);
      const input = axesCaseInput(N * numHeads * headDim);
      const actual = ropeAxes({
        input,
        N,
        numHeads,
        axisDims,
        positions: Int32Array.from(positions),
        thetaBase,
      });

      // `why` is carried from the generator so a failure says what this case
      // was for rather than only which numbers moved.
      const gap = worst(actual, output);
      expect(gap.by, `${why} — worst ${JSON.stringify(gap)}`).toBeLessThan(AGAINST_UPSTREAM);

      // And the case has to be a rotation rather than a copy, or agreeing with
      // it would say nothing: every one of these carries at least one token
      // whose positions are all zero, which is the identity, so "close to
      // upstream" is satisfiable by returning the input unless this holds.
      expect(worst(actual, input).by).toBeGreaterThan(1);
    });
  }
});

describe("ropeAxes / one axis is rope", () => {
  const [N, numHeads, headDim, thetaBase] = [3, 4, 16, 10000];
  const input = Float32Array.from({ length: N * numHeads * headDim }, (_, i) => Math.sin(i * 0.29) * 2);

  /**
   * The direct evidence that generalising the op did not change it.
   *
   * Exact equality, not a tolerance. With one axis the two compute the same
   * expression: `rope`'s unscaled path has `interpolationFactor === 1`, which
   * makes `interpolation - extrapolation` a true IEEE zero and the ramp a
   * multiply by it, and `attentionFactor === 1`, which is an exact multiply. So
   * anything but bit-for-bit agreement is a difference in the arithmetic, not
   * in the rounding — and a tolerance here would hide precisely the kind of
   * change this test exists to catch.
   */
  for (const posOffset of [0, 7, 200]) {
    it(`equals rope bit for bit at posOffset=${posOffset}`, () => {
      const positions = Int32Array.from({ length: N }, (_, token) => token + posOffset);
      expect(
        Array.from(ropeAxes({ input, N, numHeads, axisDims: [headDim], positions, thetaBase })),
      ).toEqual(Array.from(rope({ input, N, numHeads, headDim, posOffset, thetaBase })));
    });
  }
});

describe("ropeAxes / the axes are independent", () => {
  const [N, numHeads, thetaBase] = [3, 2, 256];
  const axisDims = [4, 12];
  const headDim = 16;
  const input = Float32Array.from({ length: N * numHeads * headDim }, (_, i) => Math.sin(i * 0.29) * 2 + 3);

  /** Channels `[from, from + width)` of every head, in order. */
  const block = (t: ArrayLike<number>, from: number, width: number): number[] => {
    const out: number[] = [];
    for (let row = 0; row < N * numHeads; row += 1) {
      for (let i = 0; i < width; i += 1) out.push(t[row * headDim + from + i]!);
    }
    return out;
  };

  /**
   * Each axis's block is 1-D RoPE over that block alone.
   *
   * This is the whole definition, stated where it can be checked without
   * upstream: the frequency denominator is the **axis's** channel count (4 and
   * 12 here, neither of them the head's 16), the position is the axis's own,
   * and the block starts where the earlier axes end. An implementation that
   * divided by `headDim`, that read axis 0's position for both, or that placed
   * axis 1 at the wrong offset fails this and each fails it differently.
   *
   * Bit-exact, for the reason the file above gives: with one axis `ropeAxes`
   * and `rope` evaluate the same expression, so a block of it must too.
   */
  it("rotates each block exactly as rope would rotate it alone", () => {
    const offsets = [5, 100];
    const positions = new Int32Array(N * axisDims.length);
    for (let token = 0; token < N; token += 1) {
      for (let axis = 0; axis < axisDims.length; axis += 1) {
        positions[token * axisDims.length + axis] = token + offsets[axis]!;
      }
    }

    const actual = ropeAxes({ input, N, numHeads, axisDims, positions, thetaBase });

    let from = 0;
    for (const [axis, width] of axisDims.entries()) {
      const alone = rope({
        input: Float32Array.from(block(input, from, width)),
        N,
        numHeads,
        headDim: width,
        posOffset: offsets[axis]!,
        thetaBase,
      });
      expect(block(actual, from, width), `axis ${axis}`).toEqual(Array.from(alone));
      from += width;
    }
  });

  it("copies the tensor when every position is zero", () => {
    // Position 0 is theta 0 is cos 1, sin 0 — the identity, on every axis at
    // once. The input is nowhere zero (`+ 3` puts it in [1, 5]), so an output
    // nobody wrote reads back as zeros and cannot pass for this.
    const positions = new Int32Array(N * axisDims.length);
    expect(Array.from(ropeAxes({ input, N, numHeads, axisDims, positions, thetaBase }))).toEqual(
      Array.from(input),
    );
  });

  it("moves only the block belonging to the axis whose position is not zero", () => {
    // The channel split, pinned from the other side: with only axis 1 turning,
    // axis 0's channels must come back bit-identical. An off-by-one in the
    // block offset shows up here as one moved channel at each end.
    for (const [moving, still] of [
      [0, 1],
      [1, 0],
    ]) {
      const positions = new Int32Array(N * axisDims.length);
      for (let token = 0; token < N; token += 1) positions[token * axisDims.length + moving!] = 7;
      const actual = ropeAxes({ input, N, numHeads, axisDims, positions, thetaBase });

      const at = (axis: number) => (axis === 0 ? 0 : axisDims[0]!);
      expect(block(actual, at(still!), axisDims[still!]!), `axis ${still} should not move`).toEqual(
        block(input, at(still!), axisDims[still!]!),
      );
      expect(worst(block(actual, at(moving!), axisDims[moving!]!), block(input, at(moving!), axisDims[moving!]!)).by)
        .toBeGreaterThan(1);
    }
  });

  it("turns a negative position backwards rather than wrapping to the end of a table", () => {
    // Upstream indexes a precomputed table, where `-1` is Python's last row.
    // This op computes the angle, so `-p` is the inverse rotation of `+p` —
    // rotating by one and then the other returns the input. Checked as a round
    // trip rather than against a hand-written angle so that the property being
    // asserted is the one that matters: the two are inverses.
    const forward = new Int32Array(N * axisDims.length);
    const backward = new Int32Array(N * axisDims.length);
    for (let i = 0; i < forward.length; i += 1) {
      forward[i] = 3 + i;
      backward[i] = -(3 + i);
    }
    const there = ropeAxes({ input, N, numHeads, axisDims, positions: forward, thetaBase });
    const back = ropeAxes({ input: there, N, numHeads, axisDims, positions: backward, thetaBase });

    // Not exact: the round trip is two f32 roundings of a rotation, so it
    // returns to within 2.384e-7 — one ulp at this magnitude, measured. It
    // must also have gone
    // somewhere, or the identity would satisfy this too.
    expect(worst(back, input).by).toBeLessThan(1e-6);
    expect(worst(there, input).by).toBeGreaterThan(1);
  });
});

describe("ropeAxes / refuses what it cannot mean", () => {
  const base = { N: 2, numHeads: 1, thetaBase: 256 };

  it("rejects an odd axis dim rather than inventing a half pair", () => {
    // Upstream cannot express one either — `axes_dims = [3, 3]` fails in torch
    // with "The size of tensor a (3) must match the size of tensor b (4)",
    // measured on torch 2.10.0. So there is no convention to inherit.
    expect(() =>
      ropeAxes({
        ...base,
        input: new Float32Array(2 * 6),
        axisDims: [3, 3],
        positions: new Int32Array(4),
      }),
    ).toThrow(/axis 0 has 3 channels/);
  });

  it("rejects an empty axis list", () => {
    expect(() =>
      ropeAxes({ ...base, input: new Float32Array(0), axisDims: [], positions: new Int32Array(0) }),
    ).toThrow(/at least one axis/);
  });

  it("rejects a positions array that is not one position per token per axis", () => {
    // The shape upstream asserts (`ids.shape[-1] == len(axes_dims)`). Silently
    // reading past the end would rotate later tokens by whatever followed, and
    // come back as a tensor.
    expect(() =>
      ropeAxes({
        ...base,
        input: new Float32Array(2 * 8),
        axisDims: [4, 4],
        positions: new Int32Array(3),
      }),
    ).toThrow(/positions holds 3/);
  });

  it("rejects an input whose length is not N × numHeads × sum(axisDims)", () => {
    // The head dim is derived from `axisDims`, so this is the one check that
    // can catch an `axisDims` that does not describe the tensor it was handed.
    expect(() =>
      ropeAxes({
        ...base,
        input: new Float32Array(2 * 9),
        axisDims: [4, 4],
        positions: new Int32Array(4),
      }),
    ).toThrow(/input holds 18/);
  });
});
