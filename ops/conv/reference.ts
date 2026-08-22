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
 * kernel needs asymmetric padding this signature cannot express).
 *
 * 2D was out of scope by ISSUE #14 — speech front-ends need 1D, and 2D waited
 * until something asked for it. Something did (#145), so `conv2d` is below, in
 * this file rather than in an op of its own: every convention this doc settles
 * is the same one 2D uses, and the two disagreeing is the failure worth
 * designing against.
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

/**
 * A spatial argument: one number meaning both axes, or `[H, W]`.
 *
 * This is PyTorch's `int | tuple[int, int]`, and the order is the one torch
 * 2.10.0+cu128 actually uses rather than the one the docs imply. Measured on a
 * `[1, 1, 4, 6]` input with a `1x1` kernel: `stride=(2, 3)` returns
 * `[1, 1, 2, 2]` and `stride=(3, 2)` returns `[1, 1, 2, 3]`, so the first
 * member is the **H** (row) axis.
 *
 * Why a union rather than `strideH` / `strideW` fields, which would need no
 * normalising and no runtime check: rule 7 breaks the tie. Everything a caller
 * of this library reads about convolution — PyTorch, ONNX, every checkpoint's
 * config — writes the pair as one argument, so `stride: [2, 1]` ports across by
 * eye while `strideH: 2, strideW: 1` is a second spelling of the same idea that
 * only this library uses. The lone number is not a convenience either: the
 * shapes this op exists for (3x3, stride 1, padding 1) are square on every
 * axis, and forcing `[1, 1]` on all three arguments would put the interesting
 * case and the boring one at the same visual weight.
 */
export type Conv2dSpatial = number | readonly [number, number];

/**
 * Reads a spatial argument into `[H, W]`.
 *
 * Throws rather than coercing, and names the argument. The values this refuses
 * are what a JavaScript caller reaches for when porting: `'same'` (see
 * `conv2d`'s doc for why it is not accepted), `[1, 1, 1]` from a 3D config, a
 * lone `[2]`. Every one of them has a silent reading — `padding = 0` for the
 * string, "use the first" for the wrong-length array — and a wrong pad is a
 * wrong answer of the right shape.
 */
function spatialPair(value: Conv2dSpatial, name: string): [number, number] {
  if (typeof value === "number") return [value, value];
  if (!Array.isArray(value) || value.length !== 2 || typeof value[0] !== "number" || typeof value[1] !== "number") {
    throw new Error(`conv2d(): ${name} must be a number or [H, W], got ${JSON.stringify(value)}`);
  }
  return [value[0]!, value[1]!];
}

export interface Conv2dArgs {
  /** `[N, Cin, H, W]`, row-major — NCHW, the layout `conv1d`'s `[N, Cin, L]` extends to. */
  input: Float32Array;
  /** `[Cout, Cin / groups, KH, KW]`, row-major. */
  weight: Float32Array;
  /** `[Cout]`. Omitted is PyTorch's `bias=None`: nothing is added. */
  bias?: Float32Array;
  /** Batch. */
  N: number;
  /** Input channels. */
  Cin: number;
  /** Output channels. */
  Cout: number;
  /** Input height (rows). */
  H: number;
  /** Input width (columns) — the contiguous axis. */
  W: number;
  /** Kernel height. */
  KH: number;
  /** Kernel width. */
  KW: number;
  /** Default 1. */
  stride?: Conv2dSpatial;
  /** Zeros added to *both* ends of each axis. Default 0. */
  padding?: Conv2dSpatial;
  /** Spacing between kernel taps. Default 1. */
  dilation?: Conv2dSpatial;
  /** Default 1. */
  groups?: number;
}

/**
 * `conv1dOutputLength` on each axis independently — the `[H, W]`
 * `torch.nn.functional.conv2d` returns.
 *
 * Literally the 1D function called twice, not the same arithmetic written out
 * again. The formula is a convention, and rule 7 says a convention gets chosen
 * once: two copies can drift, and the drift would be a shape that is right on
 * one axis and wrong on the other, which reads as a transposed image rather
 * than as a crash.
 *
 * Measured against torch 2.10.0+cu128: `H=10, W=13, KH=3, KW=2,
 * stride=(2,3), padding=(1,0), dilation=(2,4)` returns shape `(1, 1, 4, 3)`,
 * which is what this gives.
 */
export function conv2dOutputSize({
  H,
  W,
  KH,
  KW,
  stride = 1,
  padding = 0,
  dilation = 1,
}: {
  H: number;
  W: number;
  KH: number;
  KW: number;
  stride?: Conv2dSpatial;
  padding?: Conv2dSpatial;
  dilation?: Conv2dSpatial;
}): { Hout: number; Wout: number } {
  const [strideH, strideW] = spatialPair(stride, "stride");
  const [padH, padW] = spatialPair(padding, "padding");
  const [dilationH, dilationW] = spatialPair(dilation, "dilation");
  return {
    Hout: conv1dOutputLength({ L: H, K: KH, stride: strideH, padding: padH, dilation: dilationH }),
    Wout: conv1dOutputLength({ L: W, K: KW, stride: strideW, padding: padW, dilation: dilationW }),
  };
}

/**
 * 2D convolution over `[N, Cin, H, W]`, matching
 * `torch.nn.functional.conv2d(input, weight, bias, stride, padding, dilation, groups)`.
 *
 * The definition of correct for this op, on the same terms as `conv1d` above:
 * the slowest, plainest expression of the maths, six nested loops, no attempt
 * at anything else.
 *
 * ## What it inherits, and what it had to decide
 *
 * Every convention `conv1d`'s doc settles is settled the same way here, and
 * that is why the two live in one file: **no kernel flip** (this is a
 * cross-correlation), **zero padding on both ends of each axis**, **bias added
 * once per output element**, **`groups` splitting both channel axes**, and
 * **throwing** where PyTorch raises. They are re-measured in 2D rather than
 * assumed, because two of them can only be checked in 2D — a flip reverses
 * *both* axes, and a pad has four edges. `reference.test.ts` pins each.
 *
 * Two things 1D never had to answer:
 *
 * - **`stride` / `padding` / `dilation` take `number | [H, W]`** — see
 *   `Conv2dSpatial` for the measurement that fixes the order and the reason a
 *   union beats two fields.
 * - **`padding` is an integer count, never `'same'` or `'valid'`.** `conv1d`
 *   refuses the strings for reasons that get stronger in 2D, and they were
 *   re-measured rather than inherited. `'valid'` is `0`. `'same'` is `0` too
 *   whenever the effective kernel is even: torch 2.10.0+cu128 splits the odd
 *   total pad **asymmetrically** — `F.conv2d([[[[1,2,3,4]]]], [[[[1,0]]]],
 *   padding='same')` returns `[1,2,3,4]`, i.e. nothing added before the data
 *   and one zero after — and a single integer per axis pads both ends equally,
 *   so this signature cannot express it. torch also raises
 *   `padding='same' is not supported for strided convolutions`, which means the
 *   string is not even a total function of the arguments beside it. Where a
 *   string *is* worth taking, this library takes it: `istft`'s
 *   `padding: "same"` (#92) exists because those samples are reachable from no
 *   torch mode at all. That is the test — sugar over an integer stays in the
 *   caller, a genuinely different result becomes an argument.
 *
 * Deliberately absent, all for reasons `conv1d` already gives: `padding_mode`
 * (a pad op, not a conv op), `conv_transpose2d` (a different op — and the 1D
 * pair is already split across two directories), and 3D.
 */
export function conv2d({
  input,
  weight,
  bias,
  N,
  Cin,
  Cout,
  H,
  W,
  KH,
  KW,
  stride = 1,
  padding = 0,
  dilation = 1,
  groups = 1,
}: Conv2dArgs): Float32Array {
  const [strideH, strideW] = spatialPair(stride, "stride");
  const [padH, padW] = spatialPair(padding, "padding");
  const [dilationH, dilationW] = spatialPair(dilation, "dilation");

  if (Cin % groups !== 0 || Cout % groups !== 0) {
    // PyTorch: "expected weight to be divisible by groups at dimension 0".
    throw new Error(`conv2d(): Cin=${Cin} and Cout=${Cout} must both be divisible by groups=${groups}`);
  }
  const { Hout, Wout } = conv2dOutputSize({ H, W, KH, KW, stride, padding, dilation });
  if (Hout <= 0 || Wout <= 0) {
    // PyTorch: "Calculated padded input size per channel: (4 x 5). Kernel size:
    // (5 x 2). Kernel size can't be greater than actual input size" — one axis
    // failing is enough, which is why this is an `||`.
    throw new Error(
      `conv2d(): kernel size ${KH}x${KW} (dilated ${dilationH * (KH - 1) + 1}x${dilationW * (KW - 1) + 1}) ` +
        `exceeds padded input size ${H + 2 * padH}x${W + 2 * padW}`,
    );
  }
  if (input.length !== N * Cin * H * W) {
    throw new Error(`conv2d(): expected ${N * Cin * H * W} input elements, got ${input.length}`);
  }
  if (weight.length !== Cout * (Cin / groups) * KH * KW) {
    throw new Error(`conv2d(): expected ${Cout * (Cin / groups) * KH * KW} weight elements, got ${weight.length}`);
  }

  const inPerGroup = Cin / groups;
  const outPerGroup = Cout / groups;
  const output = new Float32Array(N * Cout * Hout * Wout);

  for (let n = 0; n < N; n += 1) {
    for (let oc = 0; oc < Cout; oc += 1) {
      const group = Math.floor(oc / outPerGroup);
      for (let oh = 0; oh < Hout; oh += 1) {
        for (let ow = 0; ow < Wout; ow += 1) {
          let acc = bias ? bias[oc]! : 0;
          for (let icLocal = 0; icLocal < inPerGroup; icLocal += 1) {
            const ic = group * inPerGroup + icLocal;
            for (let kh = 0; kh < KH; kh += 1) {
              // No flip, on either axis: tap (kh, kw) reads down and right from
              // the window's top-left corner.
              const ih = oh * strideH + kh * dilationH - padH;
              if (ih < 0 || ih >= H) continue; // the zero pad, top and bottom
              for (let kw = 0; kw < KW; kw += 1) {
                const iw = ow * strideW + kw * dilationW - padW;
                if (iw < 0 || iw >= W) continue; // the zero pad, left and right
                acc +=
                  input[((n * Cin + ic) * H + ih) * W + iw]! *
                  weight[((oc * inPerGroup + icLocal) * KH + kh) * KW + kw]!;
              }
            }
          }
          output[((n * Cout + oc) * Hout + oh) * Wout + ow] = acc;
        }
      }
    }
  }
  return output;
}
