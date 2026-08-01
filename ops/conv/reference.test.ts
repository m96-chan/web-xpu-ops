import { describe, expect, it } from "vitest";
import { conv1d, conv1dOutputLength } from "./reference.js";

/**
 * The reference pinned against PyTorch, and the parts of the contract a kernel
 * cannot be asked about.
 *
 * Every expectation below was measured against torch 2.10.0+cu128, not read off
 * the docs. Values are integers so the goldens are exact in f32 and a mismatch
 * is a mismatch rather than a rounding argument. Touches no GPU.
 */
describe("conv1d / reference", () => {
  it("cross-correlates: it does not flip the kernel", () => {
    // F.conv1d(tensor([[[1,2,3,4]]]), tensor([[[1,10,100]]])) === [[[321, 432]]]
    //   321 = 1*1 + 2*10 + 3*100   (no flip — what PyTorch calls conv1d)
    //   123 = 1*100 + 2*10 + 3*1   (flipped — a true convolution)
    // The two agree on every symmetric kernel, so only an asymmetric one can
    // tell them apart. This is that kernel.
    const got = conv1d({
      input: Float32Array.from([1, 2, 3, 4]),
      weight: Float32Array.from([1, 10, 100]),
      N: 1,
      Cin: 1,
      Cout: 1,
      L: 4,
      K: 3,
    });
    expect(Array.from(got)).toEqual([321, 432]);
    expect(Array.from(got)).not.toEqual([123, 234]);
  });

  it("pads with zeros on both ends", () => {
    // F.conv1d(tensor([[[1,2,3,4]]]), tensor([[[1,0,0]]]), padding=2)
    //   === [[[0, 0, 1, 2, 3, 4]]]
    // The identity tap walks the padded window, so the output is the input with
    // the pad made visible: two leading zeros, none trailing.
    const got = conv1d({
      input: Float32Array.from([1, 2, 3, 4]),
      weight: Float32Array.from([1, 0, 0]),
      N: 1,
      Cin: 1,
      Cout: 1,
      L: 4,
      K: 3,
      padding: 2,
    });
    expect(Array.from(got)).toEqual([0, 0, 1, 2, 3, 4]);
  });

  it("adds bias once per output element, not once per tap", () => {
    // F.conv1d(..., bias=tensor([1000.])) === [[[1321, 1432]]]
    const got = conv1d({
      input: Float32Array.from([1, 2, 3, 4]),
      weight: Float32Array.from([1, 10, 100]),
      bias: Float32Array.from([1000]),
      N: 1,
      Cin: 1,
      Cout: 1,
      L: 4,
      K: 3,
    });
    expect(Array.from(got)).toEqual([1321, 1432]);
  });

  it("matches PyTorch with stride, padding, dilation, groups and bias at once", () => {
    // torch:
    //   x = (arange(2*4*7).reshape(2,4,7) - 10).float()
    //   w = ((arange(6*2*3) % 5 - 2).reshape(6,2,3)).float()
    //   b = (arange(6) * 3 - 4).float()
    //   F.conv1d(x, w, b, stride=2, padding=2, dilation=2, groups=2)
    // Every knob is off its default, and each one changes the answer, so a
    // single golden covers all four interactions.
    const input = Float32Array.from({ length: 2 * 4 * 7 }, (_, i) => i - 10);
    const weight = Float32Array.from({ length: 6 * 2 * 3 }, (_, i) => (i % 5) - 2);
    const bias = Float32Array.from({ length: 6 }, (_, i) => i * 3 - 4);
    const got = conv1d({
      input,
      weight,
      bias,
      N: 2,
      Cin: 4,
      Cout: 6,
      L: 7,
      K: 3,
      stride: 2,
      padding: 2,
      dilation: 2,
      groups: 2,
    });
    expect(Array.from(got)).toEqual([
      2, 17, 13, 19, -2, -2, -4, 1, -21, -11, -11, -7, 14, 9, 11, 18, 31, 39, 43, 21, 3, 4, 0, 34, -26, -39, -43, 19,
      -58, -30, -32, -27, 35, -11, -11, -63, 42, 37, 39, 74, 31, 95, 99, 49, -25, -52, -56, 34,
    ]);
  });

  it("matches PyTorch on a depthwise conv (groups = Cin = Cout)", () => {
    // The shape speech front-ends actually run: one kernel per channel, no
    // mixing. w is [C, 1, K].
    //   x = ((arange(3*6) % 7 - 3).reshape(1,3,6)).float()
    //   w = ((arange(3*4) % 4 - 1).reshape(3,1,4)).float()
    //   F.conv1d(x, w, None, padding=1, groups=3)
    const input = Float32Array.from({ length: 3 * 6 }, (_, i) => (i % 7) - 3);
    const weight = Float32Array.from({ length: 3 * 4 }, (_, i) => (i % 4) - 1);
    const got = conv1d({ input, weight, N: 1, Cin: 3, Cout: 3, L: 6, K: 4, padding: 1, groups: 3 });
    expect(Array.from(got)).toEqual([-4, 2, 4, 6, 2, -7, -7, 2, 4, 2, -3, -9, -7, 2, 2]);
  });

  it("computes the output length PyTorch does", () => {
    // Measured: L=10, K=3, stride=2, padding=1, dilation=2 → 4.
    expect(conv1dOutputLength({ L: 10, K: 3, stride: 2, padding: 1, dilation: 2 })).toBe(4);
    expect(conv1dOutputLength({ L: 4, K: 3 })).toBe(2);
    expect(conv1dOutputLength({ L: 4, K: 3, padding: 2 })).toBe(6);
  });

  it("refuses a window wider than the padded input, as PyTorch does", () => {
    // torch raises: "Kernel size can't be greater than actual input size".
    // Returning an empty array instead would read downstream as "ran, found
    // nothing" — a shader cannot raise, so this edge lives only here.
    expect(() =>
      conv1d({ input: new Float32Array(2), weight: new Float32Array(5), N: 1, Cin: 1, Cout: 1, L: 2, K: 5 }),
    ).toThrow(/exceeds padded input size/);
  });

  it("refuses channel counts that groups does not divide, as PyTorch does", () => {
    // torch raises on both axes separately; one message covers both here.
    expect(() =>
      conv1d({ input: new Float32Array(24), weight: new Float32Array(6), N: 1, Cin: 3, Cout: 2, L: 8, K: 3, groups: 2 }),
    ).toThrow(/divisible by groups=2/);
    expect(() =>
      conv1d({ input: new Float32Array(32), weight: new Float32Array(18), N: 1, Cin: 4, Cout: 3, L: 8, K: 3, groups: 2 }),
    ).toThrow(/divisible by groups=2/);
  });

  it("refuses operands whose length does not match the shape", () => {
    // Convolving the wrong window silently is worse than not running.
    expect(() =>
      conv1d({ input: new Float32Array(5), weight: new Float32Array(3), N: 1, Cin: 1, Cout: 1, L: 8, K: 3 }),
    ).toThrow(/expected 8 input elements, got 5/);
    expect(() =>
      conv1d({ input: new Float32Array(8), weight: new Float32Array(4), N: 1, Cin: 1, Cout: 1, L: 8, K: 3 }),
    ).toThrow(/expected 3 weight elements, got 4/);
  });
});
