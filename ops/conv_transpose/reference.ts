export interface ConvTranspose1dArgs {
  /** `[N, Cin, L]`, row-major. */
  input: Float32Array;
  /**
   * `[Cin, Cout / groups, K]`, row-major — the *input* channel axis leads.
   * This is the transpose of `conv1d`'s `[Cout, Cin / groups, K]`.
   */
  weight: Float32Array;
  /** `[Cout]`. Omitted is PyTorch's `bias=None`: nothing is added. */
  bias?: Float32Array;
  /** Batch. */
  N: number;
  /** Input channels. */
  Cin: number;
  /** Output channels. */
  Cout: number;
  /** Input length. */
  L: number;
  /** Kernel width. */
  K: number;
  /** Default 1. */
  stride?: number;
  /** Elements removed from *both* ends of the output. Default 0. */
  padding?: number;
  /** Extra elements appended to the trailing end only. Default 0. */
  outputPadding?: number;
  /** Spacing between kernel taps. Default 1. */
  dilation?: number;
  /** Default 1. */
  groups?: number;
}

/**
 * `(L - 1) * stride - 2 * padding + dilation * (K - 1) + output_padding + 1`,
 * the length `torch.nn.functional.conv_transpose1d` returns.
 *
 * Mirrors `conv1dOutputLength` in `ops/conv`, and is the exact inverse of it on
 * the lengths a forward conv can produce — which is why `output_padding` has to
 * exist at all. `floor` in the forward formula makes it many-to-one: with
 * `stride = 2, K = 3, padding = 1`, both `L = 8` and `L = 9` convolve down to
 * 4, so undoing that has to be told which of the two was meant.
 *
 * Measured against torch 2.10.0+cu128 at `L=10, K=3, stride=2, padding=1,
 * dilation=2` → 21, and at the DACVAE stage `L=4, K=6, stride=3, padding=2,
 * output_padding=1` → 12, both of which the formula gives.
 */
export function convTranspose1dOutputLength({
  L,
  K,
  stride = 1,
  padding = 0,
  outputPadding = 0,
  dilation = 1,
}: {
  L: number;
  K: number;
  stride?: number;
  padding?: number;
  outputPadding?: number;
  dilation?: number;
}): number {
  return (L - 1) * stride - 2 * padding + dilation * (K - 1) + outputPadding + 1;
}

/**
 * Transposed 1D convolution over `[N, Cin, L]`, matching
 * `torch.nn.functional.conv_transpose1d(input, weight, bias, stride, padding,
 * output_padding, groups, dilation)`.
 *
 * The definition of correct for this op. Every backend is measured against it,
 * and it is deliberately the slowest, plainest expression of the maths — its
 * job is to be obviously right, not fast. It is written as a **scatter**: every
 * input sample is multiplied by every kernel tap and deposited where that tap
 * reaches. That is the operation itself, stated once. The WGSL kernel inverts it
 * into a gather (one thread per output element) because a shader cannot have
 * many threads accumulating into one address, and the two are compared rather
 * than sharing code.
 *
 * DAC-style neural audio codecs decode with nothing else: `dacvae`'s decoder is
 * a stack of `NormConvTranspose1d` upsampling stages, and there is no iSTFT head
 * to fall back on. Without this op the library stops at latents (ISSUE #75/#80).
 *
 * ## Conventions (rule 7: follow PyTorch)
 *
 * Measured against torch 2.10.0+cu128 rather than read off the docs. The four
 * things below each have more than one reasonable answer, so they are named
 * here instead of chosen quietly.
 *
 * - **`weight` is `[Cin, Cout / groups, K]`.** The input-channel axis comes
 *   *first*, which is the transpose of `conv1d`'s `[Cout, Cin / groups, K]` and
 *   the reason the op is called transposed. Getting this backwards yields a
 *   legal reshape of the same buffer, and therefore a wrong answer of the right
 *   shape, precisely when `Cin != Cout` — which is precisely when a codec is
 *   upsampling. `Cin=2, Cout=3, K=2` is pinned in the tests for that reason.
 * - **`padding` is subtracted, not added.** `torch.nn.ConvTranspose1d`
 *   documents it as `dilation * (kernel_size - 1) - padding` of implicit zero
 *   padding, i.e. the argument *crops* `padding` elements off each end of the
 *   full convolution. Measured: `conv_transpose1d([[[1,2,3]]], [[[1,0,0]]])` is
 *   `[[[1,2,3,0,0]]]` and the same call with `padding=1` is `[[[2,3,0]]]`. A
 *   reader who assumes it pads gets a waveform a few samples too long at both
 *   ends, which still sounds like audio.
 * - **`output_padding` is added to the trailing end only, and takes no part in
 *   the convolution.** It widens the crop on the right by that many elements;
 *   it does not insert zeros the kernel then reads. The slot it uncovers is
 *   often not zero — with `padding > 0` the full convolution still has terms
 *   out there, and the DACVAE case in the tests shows one. It exists because
 *   the forward conv's `floor` makes several input lengths collapse to the same
 *   output length, so the inverse has to be told which one. torch requires
 *   `output_padding < stride || output_padding < dilation` and raises
 *   otherwise; measured both ways, since either bound alone suffices.
 * - **The kernel is NOT flipped.** `conv1d` here is a cross-correlation because
 *   PyTorch's is; this op is the gradient of that with respect to its input, so
 *   the indexing falls out of the transpose rather than out of flipping
 *   anything. Tap `k` of input sample `l` lands at `l*stride + k*dilation`.
 *   Measured: `conv_transpose1d([[[1,2]]], [[[1,10,100]]])` is
 *   `[[[1, 12, 120, 200]]]`; flipping would give `[[[100, 210, 21, 2]]]`. The
 *   two agree on every symmetric kernel and disagree silently on every other
 *   one, which is the bug that survives a hand-checked test.
 * - **`groups`** splits both channel axes, as in `conv`. Input channel `ic`
 *   belongs to group `floor(ic / (Cin/groups))` and feeds only that group's
 *   output channels. torch raises when `Cin` is not divisible by `groups`; it
 *   never has to check `Cout`, because it infers `Cout = weight.shape[1] *
 *   groups`. This signature takes `Cout` explicitly and so checks both.
 * - **`bias` defaults to none**, as in the functional form (`bias=None`), and
 *   is added once per output element — including the elements `output_padding`
 *   appended, which no input sample reaches. The WGSL kernel takes it as a
 *   required binding instead: a shader cannot have an absent buffer, and
 *   callers with no bias pass zeros.
 *
 * ## Not this op's problem
 *
 * - **`weight_norm`.** DACVAE wraps these layers in
 *   `torch.nn.utils.weight_norm`, which stores `weight_g` / `weight_v` and
 *   rebuilds `weight = g * v / ||v||` on every forward. Folding that into a
 *   plain `[Cin, Cout/groups, K]` buffer is an offline conversion done once at
 *   load time, not work for a kernel that runs per frame. There is deliberately
 *   no `normalize` flag here to go looking for.
 * - **The causal unpad.** `NormConvTranspose1d.forward` slices the result again
 *   afterwards by an amount derived from `pad_mode`. That is a caller-side crop
 *   on the output, not part of the convolution.
 * - **2D**, for the same reason as `conv` (ISSUE #14): speech front-ends and
 *   codec decoders need 1D.
 */
export function convTranspose1d({
  input,
  weight,
  bias,
  N,
  Cin,
  Cout,
  L,
  K,
  stride = 1,
  padding = 0,
  outputPadding = 0,
  dilation = 1,
  groups = 1,
}: ConvTranspose1dArgs): Float32Array {
  if (padding < 0) {
    // torch: "negative padding is not supported".
    throw new Error(`convTranspose1d(): negative padding is not supported, got padding=${padding}`);
  }
  if (outputPadding < 0 || (outputPadding >= stride && outputPadding >= dilation)) {
    // torch: "output padding must be smaller than either stride or dilation".
    // Either bound on its own is enough — output_padding=1 with stride=1 is
    // rejected, but the same call with dilation=2 is accepted.
    throw new Error(
      `convTranspose1d(): output_padding=${outputPadding} must be smaller than either stride=${stride} or dilation=${dilation}`,
    );
  }
  if (Cin % groups !== 0 || Cout % groups !== 0) {
    // torch: "expected weight to be divisible by groups at dimension 0".
    throw new Error(`convTranspose1d(): Cin=${Cin} and Cout=${Cout} must both be divisible by groups=${groups}`);
  }
  const Lout = convTranspose1dOutputLength({ L, K, stride, padding, outputPadding, dilation });
  if (Lout <= 0) {
    // torch: "Output size is too small". It raises at zero as well as below it,
    // so an empty result is never a legal answer — and an empty array reads
    // downstream as "ran, found nothing".
    throw new Error(`convTranspose1d(): output size is too small (${Lout}); padding=${padding} crops more than the convolution produces`);
  }
  if (input.length !== N * Cin * L) {
    throw new Error(`convTranspose1d(): expected ${N * Cin * L} input elements, got ${input.length}`);
  }
  if (weight.length !== Cin * (Cout / groups) * K) {
    // Also the check that catches a caller who handed over conv1d's
    // [Cout, Cin/groups, K] buffer — whenever Cin != Cout the two differ.
    throw new Error(`convTranspose1d(): expected ${Cin * (Cout / groups) * K} weight elements, got ${weight.length}`);
  }

  const inPerGroup = Cin / groups;
  const outPerGroup = Cout / groups;
  const output = new Float32Array(N * Cout * Lout);

  // Bias first, so the scatter below is pure accumulation. Every output element
  // gets it, including the ones output_padding appended that no sample reaches.
  if (bias) {
    for (let n = 0; n < N; n += 1) {
      for (let oc = 0; oc < Cout; oc += 1) {
        for (let ol = 0; ol < Lout; ol += 1) {
          output[(n * Cout + oc) * Lout + ol] = bias[oc]!;
        }
      }
    }
  }

  for (let n = 0; n < N; n += 1) {
    for (let ic = 0; ic < Cin; ic += 1) {
      const group = Math.floor(ic / inPerGroup);
      for (let ocLocal = 0; ocLocal < outPerGroup; ocLocal += 1) {
        const oc = group * outPerGroup + ocLocal;
        for (let l = 0; l < L; l += 1) {
          const x = input[(n * Cin + ic) * L + l]!;
          for (let k = 0; k < K; k += 1) {
            // No flip: tap k of sample l lands forward, at l*stride + k*dilation
            // in the uncropped convolution. Subtracting padding is what turns
            // that into an output index — the crop, not a pad.
            const ol = l * stride + k * dilation - padding;
            if (ol < 0 || ol >= Lout) continue; // cropped away at one end or the other
            output[(n * Cout + oc) * Lout + ol]! += x * weight[(ic * outPerGroup + ocLocal) * K + k]!;
          }
        }
      }
    }
  }
  return output;
}
