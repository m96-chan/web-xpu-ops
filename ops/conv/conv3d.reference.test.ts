/**
 * `conv3d`, against torch 2.10.0+cu130.
 *
 * Issue #200. MiniMax-H3's visual VAE compresses time as well as space —
 * `temporal_downsample_factors [1,2,2,1,1,1]` — so every convolution in it is
 * 3D. `reference.ts`'s own doc listed 3D as *deliberately absent*; this is the
 * model that makes it necessary, and the conventions it settles for 1D and 2D
 * carry over unchanged.
 *
 * **Every expectation was measured, not read off the docs.** The inputs are
 * integers so the goldens are exact in f32: a mismatch is a wrong answer, never
 * a rounding difference.
 *
 * The trick that makes 3D legible is a **corner-tap kernel** over an input
 * whose elements name their own position — `x[d][h][w] = 100d + 10h + w`. One
 * output element then reads back the coordinate of the window it started on, so
 * a swapped axis, a flipped kernel or an off-by-one in the padding is visible
 * in the digits rather than hidden in a sum. 1D could not test this: a flip
 * reverses one axis and looks like an offset. 3D has three axes to get wrong
 * and a `[D,H,W]` triple to get out of order.
 */
import { describe, expect, it } from "vitest";
import { conv3d, conv3dOutputSize } from "./reference.js";

/** `x[n=0][c=0][d][h][w] = 100d + 10h + w`, so every element names its position. */
function positional(D: number, H: number, W: number): Float32Array {
  const out = new Float32Array(D * H * W);
  for (let d = 0; d < D; d += 1)
    for (let h = 0; h < H; h += 1)
      for (let w = 0; w < W; w += 1) out[(d * H + h) * W + w] = 100 * d + 10 * h + w;
  return out;
}

/** A `[1, 1, KD, KH, KW]` kernel with a single 1 at `(td, th, tw)`. */
function tap(KD: number, KH: number, KW: number, td: number, th: number, tw: number): Float32Array {
  const out = new Float32Array(KD * KH * KW);
  out[(td * KH + th) * KW + tw] = 1;
  return out;
}

const X = positional(3, 4, 5);
const SHAPE = { N: 1, Cin: 1, Cout: 1, D: 3, H: 4, W: 5 } as const;

describe("conv3d / reference", () => {
  it("reads each window from its origin — no kernel flip, axes in [D,H,W] order", () => {
    // torch: F.conv3d(x, corner_tap_2x2x2) -> shape (1,1,2,3,4)
    // Every value is the 100d+10h+w of the element the window starts on.
    const got = conv3d({ ...SHAPE, input: X, weight: tap(2, 2, 2, 0, 0, 0), KD: 2, KH: 2, KW: 2 });
    expect([...got]).toEqual([
      0, 1, 2, 3, 10, 11, 12, 13, 20, 21, 22, 23,
      100, 101, 102, 103, 110, 111, 112, 113, 120, 121, 122, 123,
    ]);
  });

  it("a flip on any of the three axes would show here", () => {
    // The opposite corner of the same kernel. Under a flip this would return
    // the previous test's values; instead every coordinate is one greater on
    // all three axes at once, which no single-axis mistake produces.
    const got = conv3d({ ...SHAPE, input: X, weight: tap(2, 2, 2, 1, 1, 1), KD: 2, KH: 2, KW: 2 });
    expect([...got]).toEqual([
      111, 112, 113, 114, 121, 122, 123, 124, 131, 132, 133, 134,
      211, 212, 213, 214, 221, 222, 223, 224, 231, 232, 233, 234,
    ]);
  });

  it("takes a different kernel extent per axis", () => {
    // KD=1, KH=2, KW=3 -> (1,1,3,3,3). A tool that read the triple as [W,H,D]
    // would produce (1,1,1,3,3) and fail on length alone.
    const got = conv3d({ ...SHAPE, input: X, weight: tap(1, 2, 3, 0, 0, 0), KD: 1, KH: 2, KW: 3 });
    expect(got.length).toBe(27);
    expect([...got]).toEqual([
      0, 1, 2, 10, 11, 12, 20, 21, 22,
      100, 101, 102, 110, 111, 112, 120, 121, 122,
      200, 201, 202, 210, 211, 212, 220, 221, 222,
    ]);
  });

  it("strides per axis", () => {
    // stride [1,2,2] -> (1,1,2,2,2). H and W advance by two, D by one.
    const got = conv3d({
      ...SHAPE, input: X, weight: tap(2, 2, 2, 0, 0, 0), KD: 2, KH: 2, KW: 2, stride: [1, 2, 2],
    });
    expect([...got]).toEqual([0, 2, 20, 22, 100, 102, 120, 122]);
  });

  it("pads the temporal axis alone", () => {
    // padding [1,0,0] -> (1,1,4,3,4). The first frame of output is entirely
    // zero: its window starts one frame before the data. H3's causal
    // convolutions pad exactly this axis and nothing else.
    const got = conv3d({
      ...SHAPE, input: X, weight: tap(2, 2, 2, 0, 0, 0), KD: 2, KH: 2, KW: 2, padding: [1, 0, 0],
    });
    expect(got.length).toBe(48);
    expect([...got.slice(0, 12)]).toEqual([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
    expect([...got.slice(12, 24)]).toEqual([0, 1, 2, 3, 10, 11, 12, 13, 20, 21, 22, 23]);
  });

  it("pads both ends of each spatial axis, and by a different amount per axis", () => {
    // padding [0,1,2] -> (1,1,2,5,8). Zero padding on both ends, so the row
    // begins with two zeros (W padded by 2) and the plane begins with a fully
    // zero row (H padded by 1).
    const got = conv3d({
      ...SHAPE, input: X, weight: tap(2, 2, 2, 0, 0, 0), KD: 2, KH: 2, KW: 2, padding: [0, 1, 2],
    });
    expect(got.length).toBe(80);
    expect([...got.slice(0, 8)]).toEqual([0, 0, 0, 0, 0, 0, 0, 0]);
    expect([...got.slice(8, 16)]).toEqual([0, 0, 0, 1, 2, 3, 4, 0]);
  });

  it("dilates per axis", () => {
    // dilation [2,1,2] -> (1,1,1,3,3). The dilated kernel spans 3 frames, so
    // only one output frame fits.
    const got = conv3d({
      ...SHAPE, input: X, weight: tap(2, 2, 2, 0, 0, 0), KD: 2, KH: 2, KW: 2, dilation: [2, 1, 2],
    });
    expect([...got]).toEqual([0, 1, 2, 10, 11, 12, 20, 21, 22]);
  });

  it("sums over every input channel and adds the bias once", () => {
    // x = [[1..8],[9..16]] as [1,2,2,2,2]; w as [2,2,2,2,2].
    const x2 = Float32Array.from({ length: 16 }, (_, i) => i + 1);
    const w2 = Float32Array.from([
      1, 1, 1, 1, 1, 1, 1, 1, /* oc0 ic0 */ 1, 0, 0, 0, 0, 0, 0, 0, /* oc0 ic1 */
      0, 0, 0, 0, 0, 0, 0, 1, /* oc1 ic0 */ 2, 0, 0, 0, 0, 0, 0, 0, /* oc1 ic1 */
    ]);
    const shape = { N: 1, Cin: 2, Cout: 2, D: 2, H: 2, W: 2, KD: 2, KH: 2, KW: 2 };
    expect([...conv3d({ ...shape, input: x2, weight: w2 })]).toEqual([45, 26]);
    expect([...conv3d({ ...shape, input: x2, weight: w2, bias: Float32Array.from([100, -100]) })]).toEqual([145, -74]);
  });

  it("splits both channel axes by groups", () => {
    // groups=2: weight is [Cout, Cin/groups, ...], so oc0 sees ic0 only.
    const x2 = Float32Array.from({ length: 16 }, (_, i) => i + 1);
    const w3 = Float32Array.from([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3]);
    const got = conv3d({
      N: 1, Cin: 2, Cout: 2, D: 2, H: 2, W: 2, KD: 2, KH: 2, KW: 2, input: x2, weight: w3, groups: 2,
    });
    expect([...got]).toEqual([1, 48]);
  });

  describe("what it refuses", () => {
    const base = { N: 1, Cin: 1, Cout: 1, D: 3, H: 4, W: 5, KD: 2, KH: 2, KW: 2, input: X };

    it("rejects groups that do not divide both channel counts", () => {
      // torch: "expected weight to be divisible by groups at dimension 0".
      expect(() => conv3d({ ...base, Cin: 2, Cout: 3, weight: new Float32Array(8), groups: 2 })).toThrow(/groups/);
    });

    it("rejects a kernel larger than the padded input on any axis", () => {
      // torch raises "Kernel size can't be greater than actual input size" —
      // and it must be checked per axis, since D is the short one here.
      expect(() => conv3d({ ...base, KD: 4, weight: new Float32Array(16) })).toThrow(/exceeds|kernel/i);
    });

    it("rejects an input whose length does not match its declared shape", () => {
      expect(() => conv3d({ ...base, input: new Float32Array(10), weight: tap(2, 2, 2, 0, 0, 0) })).toThrow(/input/);
    });

    it("rejects a weight whose length does not match its declared shape", () => {
      expect(() => conv3d({ ...base, weight: new Float32Array(7) })).toThrow(/weight/);
    });
  });

  describe("conv3dOutputSize", () => {
    it("agrees with the shapes torch produced above", () => {
      expect(conv3dOutputSize({ D: 3, H: 4, W: 5, KD: 2, KH: 2, KW: 2 })).toEqual({ Dout: 2, Hout: 3, Wout: 4 });
      expect(conv3dOutputSize({ D: 3, H: 4, W: 5, KD: 2, KH: 2, KW: 2, stride: [1, 2, 2] })).toEqual({ Dout: 2, Hout: 2, Wout: 2 });
      expect(conv3dOutputSize({ D: 3, H: 4, W: 5, KD: 2, KH: 2, KW: 2, padding: [0, 1, 2] })).toEqual({ Dout: 2, Hout: 5, Wout: 8 });
      expect(conv3dOutputSize({ D: 3, H: 4, W: 5, KD: 2, KH: 2, KW: 2, dilation: [2, 1, 2] })).toEqual({ Dout: 1, Hout: 3, Wout: 3 });
    });
  });
});
