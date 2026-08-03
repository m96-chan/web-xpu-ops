export interface Conv1dArgs {
  /** `[N, Cin, L]`, row-major. */
  input: Float32Array;
  /** `[Cout, Cin / groups, K]`, row-major. */
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
  /** Zeros added to *both* ends. Default 0. */
  padding?: number;
  /** Spacing between kernel taps. Default 1. */
  dilation?: number;
  /** Default 1. */
  groups?: number;
}

/**
 * `floor((L + 2*padding - dilation*(K-1) - 1) / stride) + 1`, the length
 * `torch.nn.functional.conv1d` returns. Measured against torch 2.10.0+cu128 at
 * `L=10, K=3, stride=2, padding=1, dilation=2` → 4, which the formula gives.
 */
export function conv1dOutputLength({
  L,
  K,
  stride = 1,
  padding = 0,
  dilation = 1,
}: {
  L: number;
  K: number;
  stride?: number;
  padding?: number;
  dilation?: number;
}): number {
  return Math.floor((L + 2 * padding - dilation * (K - 1) - 1) / stride) + 1;
}

/**
 * 1D convolution over `[N, Cin, L]`, matching
 * `torch.nn.functional.conv1d(input, weight, bias, stride, padding, dilation, groups)`.
 *
 * The definition of correct for this op. Every backend is measured against it,
 * and it is deliberately the slowest, plainest expression of the maths — its
 * job is to be obviously right, not fast.
 *
 * ## Conventions (rule 7: follow PyTorch)
 *
 * Measured against torch 2.10.0+cu128 rather than read off the docs.
 *
 * - **The kernel is NOT flipped.** PyTorch's `conv1d` is a cross-correlation,
 *   despite the name. `F.conv1d([[[1,2,3,4]]], [[[1,10,100]]])` is
 *   `[[[321, 432]]]`, i.e. `1*1 + 2*10 + 3*100`. A true convolution would flip
 *   the kernel and give `[[[123, 234]]]`. This reference does not flip, and the
 *   test suite pins the same numbers, because the two agree on every symmetric
 *   kernel and disagree silently on every other one — which is exactly the bug
 *   that survives a hand-checked test.
 * - **Padding is zeros, on both ends.** `F.conv1d([[[1,2,3,4]]], [[[1,0,0]]],
 *   padding=2)` is `[[[0, 0, 1, 2, 3, 4]]]`: the two leading zeros are the pad
 *   reaching the identity tap. `padding_mode` is not a parameter of the
 *   functional form — `reflect` / `replicate` / `circular` live on the `Conv1d`
 *   module, which pre-pads and then calls this with `padding=0`. They are left
 *   out here for the same reason: they are a pad op, not a conv op, and giving
 *   this kernel a mode flag would bake one caller's choice into every caller's
 *   inner loop.
 * - **Output length** is `conv1dOutputLength` above, and PyTorch *raises* when
 *   that is not positive: `Kernel size can't be greater than actual input
 *   size`. This throws to match, rather than returning an empty array, because
 *   an empty result reads downstream as "ran, found nothing".
 * - **`groups`** splits both channel axes. Output channel `oc` belongs to group
 *   `floor(oc / (Cout/groups))` and sees only that group's input channels;
 *   `weight` is `[Cout, Cin/groups, K]`. `groups = Cin = Cout` is a depthwise
 *   conv, which is what the speech front-ends this op exists for actually use.
 *   PyTorch raises when either channel count is not divisible by `groups`, and
 *   so does this.
 * - **`bias` defaults to none**, as in the functional form (`bias=None`). The
 *   WGSL kernel takes it as a required binding instead — a shader cannot have an
 *   absent buffer, and callers with no bias pass zeros. Adding a `hasBias` flag
 *   would buy a branch whose only observable effect is adding zero.
 *
 * Deliberately absent: transposed convolution, which lives in
 * `ops/conv_transpose` as `convTranspose1d` (ISSUE #75). It is a different op,
 * not a flag: its weight is `[Cin, Cout/groups, K]` rather than this one's
 * `[Cout, Cin/groups, K]`, its `padding` crops the output instead of extending
 * the input, and it has an `output_padding` this signature has nowhere to put.
 * Also absent: `'same'` / `'valid'` string
 * padding (sugar over the integer form, and `'same'` with an even effective
 * kernel needs asymmetric padding this signature cannot express), and 2D. 2D is
 * out of scope by ISSUE #14: speech front-ends need 1D, and 2D waits until
 * something asks for it.
 */
export function conv1d({
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
  dilation = 1,
  groups = 1,
}: Conv1dArgs): Float32Array {
  if (Cin % groups !== 0 || Cout % groups !== 0) {
    // PyTorch: "expected weight to be divisible by groups at dimension 0".
    throw new Error(`conv1d(): Cin=${Cin} and Cout=${Cout} must both be divisible by groups=${groups}`);
  }
  const Lout = conv1dOutputLength({ L, K, stride, padding, dilation });
  if (Lout <= 0) {
    // PyTorch: "Kernel size can't be greater than actual input size".
    throw new Error(`conv1d(): kernel size ${K} (dilated ${dilation * (K - 1) + 1}) exceeds padded input size ${L + 2 * padding}`);
  }
  if (input.length !== N * Cin * L) {
    throw new Error(`conv1d(): expected ${N * Cin * L} input elements, got ${input.length}`);
  }
  if (weight.length !== Cout * (Cin / groups) * K) {
    throw new Error(`conv1d(): expected ${Cout * (Cin / groups) * K} weight elements, got ${weight.length}`);
  }

  const inPerGroup = Cin / groups;
  const outPerGroup = Cout / groups;
  const output = new Float32Array(N * Cout * Lout);

  for (let n = 0; n < N; n += 1) {
    for (let oc = 0; oc < Cout; oc += 1) {
      const group = Math.floor(oc / outPerGroup);
      for (let ol = 0; ol < Lout; ol += 1) {
        let acc = bias ? bias[oc]! : 0;
        for (let icLocal = 0; icLocal < inPerGroup; icLocal += 1) {
          const ic = group * inPerGroup + icLocal;
          for (let k = 0; k < K; k += 1) {
            // No flip: tap k reads forward from the window start. This is the
            // cross-correlation PyTorch calls conv1d.
            const il = ol * stride + k * dilation - padding;
            if (il < 0 || il >= L) continue; // the zero pad
            acc += input[(n * Cin + ic) * L + il]! * weight[(oc * inPerGroup + icLocal) * K + k]!;
          }
        }
        output[(n * Cout + oc) * Lout + ol] = acc;
      }
    }
  }
  return output;
}
