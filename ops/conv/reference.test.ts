import { describe, expect, it } from "vitest";
import { conv1d, conv1dOutputLength, conv2d, conv2dOutputSize } from "./reference.js";

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

/**
 * `conv2d` pinned against PyTorch, and the two decisions its signature makes
 * that `conv1d`'s could not: what a spatial argument may be, and what it means.
 *
 * Same rules as above — every expectation measured against torch 2.10.0+cu128,
 * integers so f32 is exact, no GPU.
 */
describe("conv2d / reference", () => {
  // [[1,2,3],[4,5,6],[7,8,9]] and a 2x2 kernel whose taps are powers of ten, so
  // each output digit position names the input element that reached it.
  const RAMP = Float32Array.from([1, 2, 3, 4, 5, 6, 7, 8, 9]);
  const DECADES = Float32Array.from([1, 10, 100, 1000]);
  // (arange(20) % 9 - 4).reshape(1,1,4,5) — non-square, and no two rows alike,
  // which is what lets a swapped H/W show up as a different number rather than
  // a different shape.
  const GRID = Float32Array.from([-4, -3, -2, -1, 0, 1, 2, 3, 4, -4, -3, -2, -1, 0, 1, 2, 3, 4, -4, -3]);
  const grid = (rest: Partial<Parameters<typeof conv2d>[0]>) =>
    Array.from(
      conv2d({ input: GRID, weight: DECADES, N: 1, Cin: 1, Cout: 1, H: 4, W: 5, KH: 2, KW: 2, ...rest }),
    );

  it("cross-correlates: it does not flip the kernel", () => {
    // F.conv2d(arange-ramp 3x3, [[[[1,10],[100,1000]]]]) === [[5421,6532],[8754,9865]]
    //   5421 = 1*1 + 2*10 + 4*100 + 5*1000   (no flip — what PyTorch calls conv2d)
    //   1245 = 1*1000 + 2*100 + 4*10 + 5*1   (flipped — a true convolution)
    // Flipping in 2D reverses *both* axes, so a kernel that is symmetric in
    // neither is needed to tell them apart. This is that kernel, and the
    // flipped answer is pinned as a non-match so the test cannot pass by
    // accidentally agreeing with the other convention.
    const got = conv2d({ input: RAMP, weight: DECADES, N: 1, Cin: 1, Cout: 1, H: 3, W: 3, KH: 2, KW: 2 });
    expect(Array.from(got)).toEqual([5421, 6532, 8754, 9865]);
    expect(Array.from(got)).not.toEqual([1245, 2356, 4578, 5689]);
  });

  it("reads a spatial pair as [H, W], the order PyTorch's tuples use", () => {
    // Each pair below is measured twice, swapped, on the same operands. A
    // reference that read [W, H] would return the other member of each pair —
    // same element count in two of the three, so shape alone would not catch it.
    //   F.conv2d(grid, decades, stride=(2,1))   === [[2066,3177,4288,-3601],[3177,4288,-3601,-3390]]
    //   F.conv2d(grid, decades, stride=(1,2))   === [[2066,4288],[-2279,-57],[3177,-3601]]
    expect(grid({ stride: [2, 1] })).toEqual([2066, 3177, 4288, -3601, 3177, 4288, -3601, -3390]);
    expect(grid({ stride: [1, 2] })).toEqual([2066, 4288, -2279, -57, 3177, -3601]);
    //   F.conv2d(grid, decades, padding=(1,0)) / (0,1)
    expect(grid({ padding: [1, 0] })).toEqual([
      -3400, -2300, -1200, -100, 2066, 3177, 4288, -3601, -2279, -1168, -57, 964, 3177, 4288, -3601, -3390, 32, 43,
      -36, -34,
    ]);
    expect(grid({ padding: [0, 1] })).toEqual([
      960, 2066, 3177, 4288, -3601, -400, -2990, -2279, -1168, -57, 964, 96, 1970, 3177, 4288, -3601, -3390, -299,
    ]);
    //   F.conv2d(grid, decades, dilation=(2,1)) / (1,2)
    expect(grid({ dilation: [2, 1] })).toEqual([-2334, -1223, -112, 999, 3221, 4332, -3557, -3436]);
    expect(grid({ dilation: [1, 2] })).toEqual([3076, 4187, -3702, -1269, -158, 863, 4187, -3702, -2591]);
  });

  it("reads a lone number as both axes, as PyTorch's `int` form does", () => {
    // F.conv2d(grid, decades, stride=2) === F.conv2d(grid, decades, stride=(2,2))
    //   === [[2066,4288],[3177,-3601]]
    expect(grid({ stride: 2 })).toEqual([2066, 4288, 3177, -3601]);
    expect(grid({ stride: 2 })).toEqual(grid({ stride: [2, 2] }));
  });

  it("pads with zeros on all four edges", () => {
    // F.conv2d(tensor([[[[1,2],[3,4]]]]), a 3x3 kernel with only tap (0,0) set,
    //          padding=2) === [[0,0,0,0],[0,0,0,0],[0,0,1,2],[0,0,3,4]]
    // The corner tap walks the padded window, so the output is the input pushed
    // down and right by the pad: two blank rows and two blank columns before it,
    // none after. Only a corner tap can show that both leading edges pad; a
    // centred one is symmetric and would agree with a reference that padded the
    // trailing edges twice.
    const corner = new Float32Array(9);
    corner[0] = 1;
    const got = conv2d({
      input: Float32Array.from([1, 2, 3, 4]),
      weight: corner,
      N: 1,
      Cin: 1,
      Cout: 1,
      H: 2,
      W: 2,
      KH: 3,
      KW: 3,
      padding: 2,
    });
    expect(Array.from(got)).toEqual([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 3, 4]);
  });

  it("adds bias once per output element, not once per tap", () => {
    // F.conv2d(ramp, decades, bias=tensor([1000.])) === [[6421,7532],[9754,10865]]
    // Four taps per output, so a bias added per tap would be out by 3000.
    const got = conv2d({
      input: RAMP,
      weight: DECADES,
      bias: Float32Array.from([1000]),
      N: 1,
      Cin: 1,
      Cout: 1,
      H: 3,
      W: 3,
      KH: 2,
      KW: 2,
    });
    expect(Array.from(got)).toEqual([6421, 7532, 9754, 10865]);
  });

  it("matches PyTorch with per-axis stride, padding, dilation, groups and bias at once", () => {
    // torch:
    //   x = (arange(2*4*5*7) % 11 - 5).reshape(2,4,5,7).float()
    //   w = (arange(6*2*3*2) % 5 - 2).reshape(6,2,3,2).float()
    //   b = (arange(6) * 3 - 4).float()
    //   F.conv2d(x, w, b, stride=(2,1), padding=(1,2), dilation=(2,3), groups=2)
    // Every knob is off its default *and* different on the two axes, the kernel
    // is non-square and so is the input, so one golden covers every interaction
    // this signature has. Shape is (2,6,2,8).
    const input = Float32Array.from({ length: 2 * 4 * 5 * 7 }, (_, i) => (i % 11) - 5);
    const weight = Float32Array.from({ length: 6 * 2 * 3 * 2 }, (_, i) => (i % 5) - 2);
    const bias = Float32Array.from({ length: 6 }, (_, i) => i * 3 - 4);
    const got = conv2d({
      input,
      weight,
      bias,
      N: 2,
      Cin: 4,
      Cout: 6,
      H: 5,
      W: 7,
      KH: 3,
      KW: 2,
      stride: [2, 1],
      padding: [1, 2],
      dilation: [2, 3],
      groups: 2,
    });
    expect(got).toHaveLength(2 * 6 * 2 * 8);
    expect(Array.from(got)).toEqual([
      22, 0, 22, -10, -20, -19, -11, -10, -18, -16, -26, -15, -4, -4, 9, 7, -15, -6, -17, 13, 32, 29, -3, -4, 25, 3,
      25, -7, -17, -16, -8, -7, 3, -7, -11, -19, -16, -13, 5, 7, -12, -3, -14, 16, 35, 32, 0, -1, 2, 1, -7, -8, -9,
      -10, -2, -2, -1, 0, 4, 7, 10, 35, 16, 18, 13, 15, 30, 30, 30, 8, 13, 11, 5, 4, -4, -5, -6, -7, 1, 1, 4, 4, -3,
      -2, -1, 11, 8, 9, 16, 18, 33, 33, 33, 11, 16, 14, -11, 0, -3, 20, 21, 22, -3, -13, 9, -11, -4, -15, -26, -26,
      -18, -9, 2, -11, -19, -22, -14, -17, 0, 21, -8, 3, 0, 23, 24, 25, 0, -10, 0, 23, 35, 16, -14, -11, -12, -10, 5,
      -8, -16, -19, -11, -14, 3, 24, 27, 26, 29, 6, 5, -7, -2, -2, -4, -3, -16, -13, -10, 4, 10, 12, -4, -2, -3, 8, 8,
      30, 19, 17, 30, 29, 32, 9, 8, -4, 1, 1, 15, 15, 5, -5, -4, -3, 5, 6, -1, 1, 0, 11, 11, 33, 22, 20,
    ]);
  });

  it("matches PyTorch on a depthwise conv (groups = Cin = Cout)", () => {
    // One kernel per channel, no mixing. w is [C, 1, KH, KW].
    //   x = ((arange(3*4*5) % 7 - 3).reshape(1,3,4,5)).float()
    //   w = ((arange(3*2*2) % 4 - 1).reshape(3,1,2,2)).float()
    //   F.conv2d(x, w, None, padding=(1,0), groups=3)
    const input = Float32Array.from({ length: 3 * 4 * 5 }, (_, i) => (i % 7) - 3);
    const weight = Float32Array.from({ length: 3 * 2 * 2 }, (_, i) => (i % 4) - 1);
    const got = conv2d({
      input,
      weight,
      N: 1,
      Cin: 3,
      Cout: 3,
      H: 4,
      W: 5,
      KH: 2,
      KW: 2,
      padding: [1, 0],
      groups: 3,
    });
    expect(Array.from(got)).toEqual([
      -7, -4, -1, 2, 11, -1, -6, -4, 0, 2, 11, -1, -4, -2, 0, 2, 2, 1, 0, -1, -3, -7, -4, -1, 2, 11, -1, -6, -2, 0, 2,
      11, -6, -4, -2, 0, 3, 2, 1, 0, 8, -3, -7, -4, 0, 2, 11, -1, -4, -2, 0, 2, -1, -6, -4, -2, -3, 3, 2, 1,
    ]);
  });

  it("computes the output size PyTorch does, on each axis independently", () => {
    // Measured: H=10,W=13,KH=3,KW=2,stride=(2,3),padding=(1,0),dilation=(2,4)
    //   → torch shape (1,1,4,3).
    expect(
      conv2dOutputSize({ H: 10, W: 13, KH: 3, KW: 2, stride: [2, 3], padding: [1, 0], dilation: [2, 4] }),
    ).toEqual({ Hout: 4, Wout: 3 });
    // Measured: 4x4 input, 3x3 kernel, padding=2 → (1,1,6,6).
    expect(conv2dOutputSize({ H: 4, W: 4, KH: 3, KW: 3, padding: 2 })).toEqual({ Hout: 6, Wout: 6 });
  });

  it("refuses a window wider than the padded input on either axis, as PyTorch does", () => {
    // torch raises on a 4x5 input with a 5x2 kernel — the W axis fits and the H
    // axis does not, and one axis is enough:
    //   "Calculated padded input size per channel: (4 x 5). Kernel size: (5 x 2)."
    expect(() =>
      conv2d({
        input: new Float32Array(20),
        weight: new Float32Array(10),
        N: 1,
        Cin: 1,
        Cout: 1,
        H: 4,
        W: 5,
        KH: 5,
        KW: 2,
      }),
    ).toThrow(/exceeds padded input size/);
    // ... and the same the other way round, so an implementation that only
    // checks one axis cannot pass.
    expect(() =>
      conv2d({
        input: new Float32Array(20),
        weight: new Float32Array(10),
        N: 1,
        Cin: 1,
        Cout: 1,
        H: 5,
        W: 4,
        KH: 2,
        KW: 5,
      }),
    ).toThrow(/exceeds padded input size/);
  });

  it("refuses channel counts that groups does not divide, as PyTorch does", () => {
    expect(() =>
      conv2d({
        input: new Float32Array(3 * 16),
        weight: new Float32Array(2 * 1 * 9),
        N: 1,
        Cin: 3,
        Cout: 2,
        H: 4,
        W: 4,
        KH: 3,
        KW: 3,
        groups: 2,
      }),
    ).toThrow(/divisible by groups=2/);
    expect(() =>
      conv2d({
        input: new Float32Array(4 * 16),
        weight: new Float32Array(3 * 2 * 9),
        N: 1,
        Cin: 4,
        Cout: 3,
        H: 4,
        W: 4,
        KH: 3,
        KW: 3,
        groups: 2,
      }),
    ).toThrow(/divisible by groups=2/);
  });

  it("refuses operands whose length does not match the shape", () => {
    // Convolving the wrong window silently is worse than not running.
    expect(() =>
      conv2d({
        input: new Float32Array(15),
        weight: new Float32Array(4),
        N: 1,
        Cin: 1,
        Cout: 1,
        H: 4,
        W: 5,
        KH: 2,
        KW: 2,
      }),
    ).toThrow(/expected 20 input elements, got 15/);
    expect(() =>
      conv2d({
        input: new Float32Array(20),
        weight: new Float32Array(6),
        N: 1,
        Cin: 1,
        Cout: 1,
        H: 4,
        W: 5,
        KH: 2,
        KW: 2,
      }),
    ).toThrow(/expected 4 weight elements, got 6/);
  });

  it("refuses `'same'` rather than guessing what it would mean", () => {
    // Not an oversight: `padding: 'same'` is PyTorch's, and this signature does
    // not take it (see reference.ts for why). A caller who writes it must find
    // out here, not receive a silent `padding = 0`.
    expect(() =>
      conv2d({
        input: new Float32Array(20),
        weight: new Float32Array(4),
        N: 1,
        Cin: 1,
        Cout: 1,
        H: 4,
        W: 5,
        KH: 2,
        KW: 2,
        // @ts-expect-error — the string form is rejected at the type level too,
        // and this line asserts that as well: the directive itself fails to
        // compile if `Conv2dSpatial` ever grows a string member. The test below
        // is for JavaScript callers, who get no type check at all.
        padding: "same",
      }),
    ).toThrow(/padding must be a number or \[H, W\]/);
  });
});
