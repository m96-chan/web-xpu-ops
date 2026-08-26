/**
 * MiniMax-H3's visual VAE encoder: a video in, the latent's moments out.
 *
 * Issue #200. Six levels of two `ResnetBlock3D`s with a strided convolution
 * between them, a final norm and a projection — 16x compression in space and 4x
 * in time, out to `2 * z_channels` (mean and log-variance interleaved as
 * channels).
 *
 * `encoder-block.ts` is the unit this repeats and is what holds the arithmetic
 * to the model; this is the assembly around it. Together they are the only
 * caller `ops/conv`'s 3D entry and `ops/pad` have.
 *
 * ## The one thing the assembly has to get right on its own
 *
 * **`Downsample3D` pads asymmetrically before it strides.** With
 * `space_stride == 2` it prepends nothing and appends one column and one row —
 * `F.pad(x, (0, 1, 0, 1, 0, 0))` in the model's own source — and *then* runs a
 * `k=3, stride=2, padding=(1,0,0)` convolution whose spatial padding is zero.
 * A symmetric pad of one on each side would give the same output *size* and a
 * different alignment, which is the failure this whole file is written against:
 * a video that comes back shifted half a pixel per level.
 *
 * The temporal half of that convolution is still causal — two zero frames in
 * front, none behind — so a strided level halves the frame count without ever
 * letting one frame see a later one.
 */
import { ACTIVATION, activation } from "../../../ops/activation/index.js";
import { conv3d } from "../../../ops/conv/index.js";
import { pad } from "../../../ops/pad/index.js";
import {
  h3EncoderResnetBlock,
  padForCausalConv,
  perFrameGroupNorm,
  type ResnetBlockWeights,
} from "./encoder-block.js";

export interface EncoderConfig {
  ch: number;
  chMult: number[];
  numResBlocks: number;
  spaceDown: number[];
  timeDown: number[];
  inChannels: number;
  /** `z_channels`; the encoder emits `2 *` this, mean and log-variance. */
  zChannels: number;
  groups: number;
  normEps: number;
}

/** Every tensor, by the name the checkpoint gives it under `encoder.`. */
export type EncoderWeights = (name: string) => Float32Array;

/** A tensor and the shape that is live in it. */
interface Volume {
  data: Float32Array;
  C: number;
  D: number;
  H: number;
  W: number;
}

/** `[block_in, block_mid]` per level — the channel counts the model derives. */
export function channelPlan(cfg: EncoderConfig): { blockIn: number[]; blockMid: number[] } {
  const blockMid = cfg.chMult.map((m) => cfg.ch * m);
  const blockIn = [blockMid[0]!, ...blockMid.slice(0, -1)];
  return { blockIn, blockMid };
}

/** A `k=3, padding=1` convolution with H3's padding: reflect in space, causal in time. */
function conv3x3x3(v: Volume, Cout: number, weight: Float32Array, bias: Float32Array): Volume {
  const padded = padForCausalConv(v.data, v.C, v.D, v.H, v.W, 1, 2);
  return {
    data: conv3d({
      input: padded.data, weight, bias,
      N: 1, Cin: v.C, Cout, D: padded.D, H: padded.H, W: padded.W,
      KD: 3, KH: 3, KW: 3,
    }),
    C: Cout, D: v.D, H: v.H, W: v.W,
  };
}

/**
 * `Downsample3D`: pad one column and one row **after**, then stride.
 *
 * The asymmetry is the model's. A symmetric pad gives the same output size and
 * a different alignment.
 */
export function downsample(
  v: Volume,
  Cout: number,
  timeStride: number,
  spaceStride: number,
  weight: Float32Array,
  bias: Float32Array,
): Volume {
  let data = v.data;
  let H = v.H;
  let W = v.W;
  if (spaceStride === 2) {
    data = pad({ input: data, outer: v.C * v.D * H, L: W, inner: 1, before: 0, after: 1, mode: "reflect" });
    W += 1;
    data = pad({ input: data, outer: v.C * v.D, L: H, inner: W, before: 0, after: 1, mode: "reflect" });
    H += 1;
  }
  // `padding=(1, 0, 0)`: nothing in space, two causal frames in front.
  data = pad({ input: data, outer: v.C, L: v.D, inner: H * W, before: 2, after: 0, mode: "constant" });
  const D = v.D + 2;

  const Dout = Math.floor((D - 3) / timeStride) + 1;
  const Hout = Math.floor((H - 3) / spaceStride) + 1;
  const Wout = Math.floor((W - 3) / spaceStride) + 1;
  return {
    data: conv3d({
      input: data, weight, bias,
      N: 1, Cin: v.C, Cout, D, H, W, KD: 3, KH: 3, KW: 3,
      stride: [timeStride, spaceStride, spaceStride],
    }),
    C: Cout, D: Dout, H: Hout, W: Wout,
  };
}

/**
 * A video `[inChannels, T, H, W]` in, `[2 * zChannels, T/4, H/16, W/16]` out.
 *
 * `quantConv` is `AutoencoderKLLegacy`'s own `quant_conv`, a 1x1x1 convolution
 * the encoder's output passes through before anything samples from it. It is
 * here rather than in the caller because `encode` in the model is
 * `quant_conv(encoder(x))` and splitting them would invite one to be forgotten.
 */
export function h3Encode(cfg: EncoderConfig, w: EncoderWeights, video: Float32Array, T: number, H: number, W: number): Volume {
  const { blockIn, blockMid } = channelPlan(cfg);
  const levels = cfg.chMult.length;

  let v: Volume = { data: video, C: cfg.inChannels, D: T, H, W };
  v = conv3x3x3(v, blockIn[0]!, w("conv_in.weight"), w("conv_in.bias"));

  for (let level = 0; level < levels; level += 1) {
    for (let block = 0; block < cfg.numResBlocks; block += 1) {
      const inChannels = block === 0 ? blockIn[level]! : blockMid[level]!;
      const outChannels = blockMid[level]!;
      const p = `down.${level}.block.${block}.`;
      const weights: ResnetBlockWeights = {
        norm1Weight: w(`${p}norm1.weight`), norm1Bias: w(`${p}norm1.bias`),
        conv1Weight: w(`${p}conv1.weight`), conv1Bias: w(`${p}conv1.bias`),
        norm2Weight: w(`${p}norm2.weight`), norm2Bias: w(`${p}norm2.bias`),
        conv2Weight: w(`${p}conv2.weight`), conv2Bias: w(`${p}conv2.bias`),
        ...(inChannels === outChannels
          ? {}
          : { shortcutWeight: w(`${p}nin_shortcut.weight`), shortcutBias: w(`${p}nin_shortcut.bias`) }),
      };
      v = {
        data: h3EncoderResnetBlock(
          { inChannels, outChannels, D: v.D, H: v.H, W: v.W, groups: cfg.groups, normEps: cfg.normEps },
          weights,
          v.data,
        ),
        C: outChannels, D: v.D, H: v.H, W: v.W,
      };
    }
    // The model adds a downsample only when it actually changes the shape, and
    // a 1x1 convolution only when the level changes channel count — which, with
    // `block_out == block_mid`, it never does. So there is one branch here and
    // not two.
    if (cfg.spaceDown[level]! * cfg.timeDown[level]! > 1) {
      v = downsample(
        v, blockMid[level]!, cfg.timeDown[level]!, cfg.spaceDown[level]!,
        w(`down.${level}.downsample.conv.weight`), w(`down.${level}.downsample.conv.bias`),
      );
    }
  }

  v = {
    data: activation({
      input: perFrameGroupNorm(v.data, v.C, v.D, v.H, v.W, w("norm_out.weight"), w("norm_out.bias"), cfg.groups, cfg.normEps),
      kind: ACTIVATION.silu,
    }),
    C: v.C, D: v.D, H: v.H, W: v.W,
  };
  v = conv3x3x3(v, 2 * cfg.zChannels, w("conv_out.weight"), w("conv_out.bias"));

  // `quant_conv`, a 1x1x1 Conv3d: no padding at all, so `BaseConv3d.forward`
  // short-circuits and there is nothing for `ops/pad` to do.
  return {
    data: conv3d({
      input: v.data, weight: w("quant_conv.weight"), bias: w("quant_conv.bias"),
      N: 1, Cin: v.C, Cout: 2 * cfg.zChannels, D: v.D, H: v.H, W: v.W, KD: 1, KH: 1, KW: 1,
    }),
    C: 2 * cfg.zChannels, D: v.D, H: v.H, W: v.W,
  };
}

/**
 * The mean half of the moments — the latent a decoder is given.
 *
 * `DiagonalGaussianDistribution(moments).sample()` draws from
 * `mean + exp(0.5 * logvar) * noise`; the mean is what a deterministic
 * round-trip uses, and it is the first `zChannels` of the channel axis.
 */
export function latentMean(moments: Volume, zChannels: number): Float32Array {
  const per = moments.D * moments.H * moments.W;
  return moments.data.slice(0, zChannels * per);
}
