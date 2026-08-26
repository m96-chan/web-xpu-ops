/**
 * One `ResnetBlock3D` of MiniMax-H3's visual VAE **encoder**.
 *
 * Issue #200. The decoder is a ViT and needs no convolutions; the encoder is
 * where `ops/conv`'s 3D entry and `ops/pad` are used. An op with no caller is a
 * liability, and this is theirs.
 *
 * Two per level over six levels, so a port is right or wrong here in the same
 * way the decoder's transformer block was.
 *
 *     h = silu(groupNorm(x))          32 groups, eps 1e-6, **per frame**
 *     h = conv3d(pad(h))              k=3, reflect in space, causal in time
 *     h = silu(groupNorm(h))
 *     h = conv3d(pad(h))
 *     return (in == out ? x : conv3d(x, k=1)) + h
 *
 * ## The two things the padding has to get right
 *
 * **Space reflects; time does not.** `BaseConv3d._apply_padding` pads H and W
 * with `reflect` and then hands the temporal axis to
 * `_apply_temporal_padding`, which for a causal convolution prepends
 * `2 * padding` **zero** frames and appends none. A symmetric pad of any mode
 * would let frame `t` see `t + 1`, which is the property the encoder is built
 * to deny.
 *
 * **`2 * padding`, not `padding`.** A k=3 convolution with `padding=1` is given
 * *two* frames in front, so the output has the same length as the input and
 * every output frame is a function of `t`, `t-1` and `t-2`. Prepending one
 * frame instead gives an output one frame shorter and is the plausible mistake.
 *
 * ## Group norm here is per frame
 *
 * `use_t_isolated_gn: true` selects `TemporalIsolatedSpatialParallelGroupNorm`,
 * which merges time into the batch, normalises, and splits it back — so the
 * statistics are pooled over `C/32 x H x W` **within one frame** rather than
 * over the whole clip. Reshaping `[C, D, H, W]` into `[D, C, H*W]` for
 * `ops/group_norm` is what that merge is.
 */
import { ACTIVATION, activation } from "../../../ops/activation/index.js";
import { conv3d } from "../../../ops/conv/index.js";
import { groupNorm } from "../../../ops/group_norm/index.js";
import { pad } from "../../../ops/pad/index.js";

export interface ResnetBlockConfig {
  inChannels: number;
  outChannels: number;
  /** Frames, height, width of the input. */
  D: number;
  H: number;
  W: number;
  groups: number;
  normEps: number;
}

export interface ResnetBlockWeights {
  norm1Weight: Float32Array;
  norm1Bias: Float32Array;
  conv1Weight: Float32Array;
  conv1Bias: Float32Array;
  norm2Weight: Float32Array;
  norm2Bias: Float32Array;
  conv2Weight: Float32Array;
  conv2Bias: Float32Array;
  /** Present only when the channel counts differ. */
  shortcutWeight?: Float32Array;
  shortcutBias?: Float32Array;
}

/**
 * Group norm over each frame independently.
 *
 * `[C, D, H, W]` is reshaped to `[D, C, H*W]` — `_merge_time_to_batch` — and
 * back. The reshape is real work rather than a view: `D` is not the outermost
 * axis to begin with.
 */
export function perFrameGroupNorm(
  x: Float32Array,
  C: number,
  D: number,
  H: number,
  W: number,
  weight: Float32Array,
  bias: Float32Array,
  groups: number,
  eps: number,
): Float32Array {
  const plane = H * W;
  const merged = new Float32Array(x.length);
  for (let c = 0; c < C; c += 1) {
    for (let d = 0; d < D; d += 1) {
      const from = (c * D + d) * plane;
      merged.set(x.subarray(from, from + plane), (d * C + c) * plane);
    }
  }
  const normed = groupNorm({ input: merged, weight, bias, N: D, C, L: plane, G: groups, eps });
  const out = new Float32Array(x.length);
  for (let d = 0; d < D; d += 1) {
    for (let c = 0; c < C; c += 1) {
      const from = (d * C + c) * plane;
      out.set(normed.subarray(from, from + plane), (c * D + d) * plane);
    }
  }
  return out;
}

/**
 * `BaseConv3d`'s padding for a `k=3, padding=1` convolution.
 *
 * Reflect on W, reflect on H, then **two zero frames in front and none
 * behind**. `ops/pad` takes one axis at a time, and doing W then H then D is
 * measured to give what a single multi-axis `F.pad` gives — see
 * `ops/pad/reference.test.ts`.
 */
export function padForCausalConv(
  x: Float32Array,
  C: number,
  D: number,
  H: number,
  W: number,
  spatial: number,
  temporalBefore: number,
): { data: Float32Array; D: number; H: number; W: number } {
  let data = x;
  let width = W;
  let height = H;
  let depth = D;

  if (spatial > 0) {
    data = pad({
      input: data, outer: C * depth * height, L: width, inner: 1,
      before: spatial, after: spatial, mode: "reflect",
    });
    width += 2 * spatial;
    data = pad({
      input: data, outer: C * depth, L: height, inner: width,
      before: spatial, after: spatial, mode: "reflect",
    });
    height += 2 * spatial;
  }
  if (temporalBefore > 0) {
    // Zeros, and only in front: this is what makes the convolution causal.
    data = pad({
      input: data, outer: C, L: depth, inner: height * width,
      before: temporalBefore, after: 0, mode: "constant",
    });
    depth += temporalBefore;
  }
  return { data, D: depth, H: height, W: width };
}

/** One encoder resnet block, `[C_in, D, H, W]` to `[C_out, D, H, W]`. */
export function h3EncoderResnetBlock(
  cfg: ResnetBlockConfig,
  w: ResnetBlockWeights,
  x: Float32Array,
): Float32Array {
  const { inChannels, outChannels, D, H, W, groups, normEps } = cfg;

  const conv3 = (
    input: Float32Array,
    Cin: number,
    Cout: number,
    weight: Float32Array,
    bias: Float32Array,
  ): Float32Array => {
    const padded = padForCausalConv(input, Cin, D, H, W, 1, 2);
    return conv3d({
      input: padded.data, weight, bias,
      N: 1, Cin, Cout, D: padded.D, H: padded.H, W: padded.W,
      KD: 3, KH: 3, KW: 3,
    });
  };

  let h = activation({
    input: perFrameGroupNorm(x, inChannels, D, H, W, w.norm1Weight, w.norm1Bias, groups, normEps),
    kind: ACTIVATION.silu,
  });
  h = conv3(h, inChannels, outChannels, w.conv1Weight, w.conv1Bias);

  h = activation({
    input: perFrameGroupNorm(h, outChannels, D, H, W, w.norm2Weight, w.norm2Bias, groups, normEps),
    kind: ACTIVATION.silu,
  });
  h = conv3(h, outChannels, outChannels, w.conv2Weight, w.conv2Bias);

  // The 1x1 shortcut takes `padding=0`, and `BaseConv3d.forward` short-circuits
  // to `nn.Conv3d` when every padding is zero — so no `ops/pad` call at all,
  // and in particular **no causal frames**. Padding it would be a length change.
  let residual = x;
  if (inChannels !== outChannels) {
    if (!w.shortcutWeight || !w.shortcutBias) {
      throw new Error(`h3EncoderResnetBlock: ${inChannels} -> ${outChannels} needs a shortcut, and none was given`);
    }
    residual = conv3d({
      input: x, weight: w.shortcutWeight, bias: w.shortcutBias,
      N: 1, Cin: inChannels, Cout: outChannels, D, H, W, KD: 1, KH: 1, KW: 1,
    });
  }

  const out = new Float32Array(h.length);
  for (let i = 0; i < out.length; i += 1) out[i] = residual[i]! + h[i]!;
  return out;
}
