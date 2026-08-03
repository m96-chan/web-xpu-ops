import { describe, expect, it } from "vitest";
import { convTranspose1d, convTranspose1dOutputLength } from "./reference.js";

/**
 * The reference pinned against PyTorch, and the parts of the contract a kernel
 * cannot be asked about.
 *
 * Every expectation below was measured against torch 2.10.0+cu128 by running
 * `torch.nn.functional.conv_transpose1d`, not read off the docs. Values are
 * integers so the goldens are exact in f32 and a mismatch is a mismatch rather
 * than a rounding argument. Touches no GPU.
 */
describe("convTranspose1d / reference", () => {
  it("does not flip the kernel", () => {
    // F.conv_transpose1d(tensor([[[1,2]]]), tensor([[[1,10,100]]]))
    //   === [[[1, 12, 120, 200]]]
    // Tap k of input sample l lands at l + k, so w[0] is what sample 0 puts at
    // output 0. Flipping the kernel would give [100, 210, 21, 2] instead. The
    // two agree on every symmetric kernel, so only an asymmetric one can tell
    // them apart. This is that kernel.
    const got = convTranspose1d({
      input: Float32Array.from([1, 2]),
      weight: Float32Array.from([1, 10, 100]),
      N: 1,
      Cin: 1,
      Cout: 1,
      L: 2,
      K: 3,
    });
    expect(Array.from(got)).toEqual([1, 12, 120, 200]);
    expect(Array.from(got)).not.toEqual([100, 210, 21, 2]);
  });

  it("reads weight as [Cin, Cout/groups, K], not conv1d's [Cout, Cin/groups, K]", () => {
    // The layout only becomes visible when Cin != Cout, which is exactly when a
    // codec is upsampling. Here Cin=2, Cout=3, K=2: weight is [2, 3, 2].
    //   x = arange(2*4).reshape(1,2,4).float()
    //   w = (arange(2*3*2) % 5 - 2).reshape(2,3,2).float()
    //   F.conv_transpose1d(x, w)
    // Reading the same buffer as [Cout, Cin/groups, K] = [3, 2, 2] would be a
    // legal reshape of twelve elements and would silently produce a different
    // answer of the same shape.
    const input = Float32Array.from({ length: 2 * 4 }, (_, i) => i);
    const weight = Float32Array.from({ length: 2 * 3 * 2 }, (_, i) => (i % 5) - 2);
    const got = convTranspose1d({ input, weight, N: 1, Cin: 2, Cout: 3, L: 4, K: 2 });
    expect(Array.from(got)).toEqual([-4, -7, -11, -15, -3, 4, 13, 17, 21, 17, -8, -12, -15, -18, -13]);
  });

  it("subtracts padding: the argument crops the output rather than padding it", () => {
    // The identity tap makes the crop legible. Uncropped, sample l lands at l.
    //   F.conv_transpose1d([[[1,2,3]]], [[[1,0,0]]])            === [[[1,2,3,0,0]]]
    //   F.conv_transpose1d([[[1,2,3]]], [[[1,0,0]]], padding=1) === [[[2,3,0]]]
    //   F.conv_transpose1d([[[1,2,3]]], [[[1,0,0]]], padding=2) === [[[3]]]
    // Each unit of padding removes one element from *each* end. A reader who
    // assumes it adds gets a longer waveform that still sounds like audio.
    const input = Float32Array.from([1, 2, 3]);
    const weight = Float32Array.from([1, 0, 0]);
    const shape = { N: 1, Cin: 1, Cout: 1, L: 3, K: 3 };
    expect(Array.from(convTranspose1d({ input, weight, ...shape }))).toEqual([1, 2, 3, 0, 0]);
    expect(Array.from(convTranspose1d({ input, weight, ...shape, padding: 1 }))).toEqual([2, 3, 0]);
    expect(Array.from(convTranspose1d({ input, weight, ...shape, padding: 2 }))).toEqual([3]);
  });

  it("adds output_padding to the trailing end only, and it takes no part in the sum", () => {
    // F.conv_transpose1d([[[1,2]]], [[[1,10,100]]], stride=2)
    //   === [[[1, 10, 102, 20, 200]]]
    // With output_padding=1 the same five values gain a sixth slot, at the end:
    //   === [[[1, 10, 102, 20, 200, 0]]]
    // Nothing shifts, so the extra slot is not a pad the convolution sees — it
    // exists only because several input lengths map to the same output length
    // under a strided forward conv, and this picks which one to undo.
    const input = Float32Array.from([1, 2]);
    const weight = Float32Array.from([1, 10, 100]);
    const shape = { N: 1, Cin: 1, Cout: 1, L: 2, K: 3, stride: 2 };
    expect(Array.from(convTranspose1d({ input, weight, ...shape }))).toEqual([1, 10, 102, 20, 200]);
    expect(Array.from(convTranspose1d({ input, weight, ...shape, outputPadding: 1 }))).toEqual([1, 10, 102, 20, 200, 0]);
    // bias still lands on the appended slot: it is an output element like any
    // other, just one no input sample reaches. Measured with bias=[1000.].
    expect(
      Array.from(
        convTranspose1d({ input, weight, bias: Float32Array.from([1000]), ...shape, outputPadding: 1 }),
      ),
    ).toEqual([1001, 1010, 1102, 1020, 1200, 1000]);
  });

  it("matches PyTorch with stride, padding, output_padding, dilation, groups and bias at once", () => {
    // torch:
    //   x = (arange(2*4*5) - 10).reshape(2,4,5).float()
    //   w = ((arange(4*3*3) % 5) - 2).reshape(4,3,3).float()
    //   b = (arange(6) * 3 - 4).float()
    //   F.conv_transpose1d(x, w, b, stride=2, padding=2, output_padding=1,
    //                      groups=2, dilation=2)
    // Every knob is off its default and each one changes the answer, so one
    // golden covers all six interactions. Note weight is [Cin, Cout/groups, K]
    // = [4, 3, 3]: Cin leads, Cout/groups follows.
    const input = Float32Array.from({ length: 2 * 4 * 5 }, (_, i) => i - 10);
    const weight = Float32Array.from({ length: 4 * 3 * 3 }, (_, i) => (i % 5) - 2);
    const bias = Float32Array.from({ length: 6 }, (_, i) => i * 3 - 4);
    const got = convTranspose1d({
      input,
      weight,
      bias,
      N: 2,
      Cin: 4,
      Cout: 6,
      L: 5,
      K: 3,
      stride: 2,
      padding: 2,
      outputPadding: 1,
      dilation: 2,
      groups: 2,
    });
    expect(Array.from(got)).toEqual([
      26, -4, 28, -4, 24, -4, 20, -4, 6, -4, -35, -1, -21, -1, -17, -1, -13, -1, -4, -1, 24, 2, 10, 2, 7, 2, 4, 2, -4,
      2, 11, 5, 25, 5, 29, 5, 33, 5, 32, 5, -10, 8, -14, 8, -17, 8, -20, 8, 2, 8, 29, 11, 22, 11, 22, 11, 22, 11, 2, 11,
      -34, -4, -52, -4, -56, -4, -60, -4, -74, -4, 45, -1, 59, -1, 63, -1, 67, -1, 56, -1, -56, 2, -50, 2, -53, 2, -56,
      2, -4, 2, 91, 5, 105, 5, 109, 5, 113, 5, 92, 5, -90, 8, -74, 8, -77, 8, -80, 8, 2, 8, 89, 11, 22, 11, 22, 11, 22,
      11, -58, 11,
    ]);
  });

  it("matches PyTorch on a DACVAE upsampling stage (odd stride, output_padding=1)", () => {
    // dacvae/nn/layers.py, NormConvTranspose1d.__init__ builds each decoder
    // stage as kernel_size = 2*stride, padding = (stride+1)//2,
    // output_padding = 1 if stride % 2 else 0. Odd stride is the only case
    // where output_padding is non-zero, so it is the only case that can catch
    // an implementation that ignores the argument. stride=3 → K=6, padding=2,
    // output_padding=1.
    //   x = ((arange(1*2*4) % 7) - 3).reshape(1,2,4).float()
    //   w = ((arange(2*2*6) % 5) - 2).reshape(2,2,6).float()
    //   F.conv_transpose1d(x, w, None, stride=3, padding=2, output_padding=1)
    const input = Float32Array.from({ length: 1 * 2 * 4 }, (_, i) => (i % 7) - 3);
    const weight = Float32Array.from({ length: 2 * 2 * 6 }, (_, i) => (i % 5) - 2);
    const shape = { N: 1, Cin: 2, Cout: 2, L: 4, K: 6, stride: 3, padding: 2 };
    const got = convTranspose1d({ input, weight, ...shape, outputPadding: 1 });
    expect(Array.from(got)).toEqual([
      2, -1, -3, 10, -4, -2, 10, -7, -8, -4, 6, 3, -5, -3, 10, -2, -2, 10, -3, -8, -4, 10, 3, 0,
    ]);

    // The same stage with output_padding=0, measured separately. The appended
    // slot is *not* zero here: padding already cropped the tail of the full
    // convolution, so extending the window by one uncovers a real term (6 → 6,3
    // on channel 0). An implementation that appended zeros instead of widening
    // the crop would pass the stride=2 case above and fail this one.
    expect(Array.from(convTranspose1d({ input, weight, ...shape }))).toEqual([
      2, -1, -3, 10, -4, -2, 10, -7, -8, -4, 6, -5, -3, 10, -2, -2, 10, -3, -8, -4, 10, 3,
    ]);
  });

  it("computes the output length PyTorch does", () => {
    // (L-1)*stride - 2*padding + dilation*(K-1) + output_padding + 1, each
    // measured as the last dimension torch returns.
    expect(convTranspose1dOutputLength({ L: 4, K: 3 })).toBe(6);
    expect(convTranspose1dOutputLength({ L: 2, K: 3, stride: 2, outputPadding: 1 })).toBe(6);
    expect(convTranspose1dOutputLength({ L: 10, K: 3, stride: 2, padding: 1, dilation: 2 })).toBe(21);
    expect(convTranspose1dOutputLength({ L: 3, K: 3, padding: 2 })).toBe(1);
    expect(convTranspose1dOutputLength({ L: 4, K: 6, stride: 3, padding: 2, outputPadding: 1 })).toBe(12);
  });

  it("refuses an output_padding that is not smaller than stride or dilation, as PyTorch does", () => {
    // torch raises: "output padding must be smaller than either stride or
    // dilation". Either is enough — output_padding=1 with stride=1 raises, but
    // the same call with dilation=2 is accepted. Measured both ways.
    const args = { input: new Float32Array(3), weight: new Float32Array(3), N: 1, Cin: 1, Cout: 1, L: 3, K: 3 };
    expect(() => convTranspose1d({ ...args, outputPadding: 1 })).toThrow(/must be smaller than either stride/);
    expect(() => convTranspose1d({ ...args, outputPadding: 1, dilation: 2 })).not.toThrow();
    expect(() => convTranspose1d({ ...args, outputPadding: 2, stride: 2 })).toThrow(/must be smaller than either stride/);
    expect(() => convTranspose1d({ ...args, outputPadding: 2, stride: 3 })).not.toThrow();
  });

  it("refuses a crop that leaves nothing, as PyTorch does", () => {
    // torch raises "Output size is too small" at zero as well as below it, so
    // an empty result is never a legal answer. Measured: L=1, K=2, padding=1
    // gives 0 and raises; L=3, K=3, padding=3 gives -1 and raises.
    expect(() =>
      convTranspose1d({ input: new Float32Array(1), weight: new Float32Array(2), N: 1, Cin: 1, Cout: 1, L: 1, K: 2, padding: 1 }),
    ).toThrow(/output size is too small/i);
    expect(() =>
      convTranspose1d({ input: new Float32Array(3), weight: new Float32Array(3), N: 1, Cin: 1, Cout: 1, L: 3, K: 3, padding: 3 }),
    ).toThrow(/output size is too small/i);
  });

  it("refuses negative padding, as PyTorch does", () => {
    // torch raises "negative padding is not supported". Cropping from the
    // outside is the caller's job (NormConvTranspose1d does its causal unpad
    // that way); letting it in here would make the output length formula lie.
    expect(() =>
      convTranspose1d({ input: new Float32Array(3), weight: new Float32Array(3), N: 1, Cin: 1, Cout: 1, L: 3, K: 3, padding: -1 }),
    ).toThrow(/negative padding/);
  });

  it("refuses channel counts that groups does not divide", () => {
    // torch raises on Cin: "Given groups=2, expected weight to be divisible by
    // 2 at dimension 0". It cannot raise on Cout, because torch infers
    // Cout = weight.shape[1] * groups and so it is divisible by construction.
    // This signature takes Cout explicitly, so it has to check what torch gets
    // for free.
    expect(() =>
      convTranspose1d({ input: new Float32Array(24), weight: new Float32Array(9), N: 1, Cin: 3, Cout: 2, L: 8, K: 3, groups: 2 }),
    ).toThrow(/divisible by groups=2/);
    expect(() =>
      convTranspose1d({ input: new Float32Array(32), weight: new Float32Array(18), N: 1, Cin: 4, Cout: 3, L: 8, K: 3, groups: 2 }),
    ).toThrow(/divisible by groups=2/);
  });

  it("refuses operands whose length does not match the shape", () => {
    // Scattering the wrong window silently is worse than not running. The
    // weight check is the one that catches a caller who passed conv1d's
    // [Cout, Cin/groups, K] buffer whenever Cin != Cout.
    expect(() =>
      convTranspose1d({ input: new Float32Array(5), weight: new Float32Array(3), N: 1, Cin: 1, Cout: 1, L: 8, K: 3 }),
    ).toThrow(/expected 8 input elements, got 5/);
    expect(() =>
      convTranspose1d({ input: new Float32Array(8), weight: new Float32Array(4), N: 1, Cin: 1, Cout: 1, L: 8, K: 3 }),
    ).toThrow(/expected 3 weight elements, got 4/);
  });
});
