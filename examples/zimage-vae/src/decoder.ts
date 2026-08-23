/**
 * Z-Image's VAE decoder, composed from this repository's ops.
 *
 * A latent goes in `[1, 16, H/8, W/8]`, an image comes out `[1, 3, H, W]`. That
 * is the whole contract, and it is the first thing in this repository that
 * produces something a person can look at.
 *
 * Composition only — every line is an existing op. The structure follows
 * diffusers' `Decoder` as Z-Image ships it (musubi-tuner's copy,
 * `zimage_autoencoder.py:280`):
 *
 *     conv_in -> mid_block -> up_blocks[0..3] -> norm -> silu -> conv_out
 *
 * with `mid_block = resnet, attention, resnet` and each up block being
 * `layers_per_block + 1` resnets followed (except the last) by a 2x nearest
 * upsample and a 3x3 conv.
 *
 * Two things read from the model rather than assumed (rule 2):
 *
 *  - **No `post_quant_conv`.** diffusers inserts a 1x1 conv before the decoder
 *    when the config asks for one; Z-Image's does not, and the generator
 *    asserts that rather than leaving it to chance. A port that invented one
 *    would still produce a plausible image.
 *  - **The mid-block attention is over pixels, not channels.** `[B, C, H, W]`
 *    becomes `[B, H*W, C]` and each spatial position attends to every other,
 *    single-headed. Getting the transpose backwards gives an image that is
 *    wrong in a way that still looks like an image.
 */
import { ACTIVATION, activation } from "../../../ops/activation/index.js";
import { attention } from "../../../ops/attention/index.js";
import { conv2d, conv2dOutputSize } from "../../../ops/conv/index.js";
import { ELEMENTWISE, elementwise } from "../../../ops/elementwise/index.js";
import { groupNorm } from "../../../ops/group_norm/index.js";
import { matmul } from "../../../ops/matmul/index.js";
import { nearestUpsample2d } from "../../../ops/upsample/index.js";

/** A tensor plus the shape the ops need spelled out. */
export interface Map4 {
  data: Float32Array;
  C: number;
  H: number;
  W: number;
}

export interface DecoderConfig {
  blockOutChannels: number[];
  layersPerBlock: number;
  normNumGroups: number;
  latentChannels: number;
  outChannels: number;
  scalingFactor: number;
  shiftFactor: number;
}

/** Named exactly as the checkpoint does, so a missing weight names itself. */
export type Weights = (name: string) => Float32Array;

function conv(x: Map4, w: Weights, prefix: string, outC: number, kernel: number, padding: number): Map4 {
  const weight = w(`${prefix}.weight`);
  const bias = w(`${prefix}.bias`);
  const { Hout, Wout } = conv2dOutputSize({ H: x.H, W: x.W, KH: kernel, KW: kernel, padding });
  const data = conv2d({
    input: x.data, weight, bias,
    N: 1, Cin: x.C, Cout: outC, H: x.H, W: x.W, KH: kernel, KW: kernel,
    padding,
  });
  return { data, C: outC, H: Hout, W: Wout };
}

/** GroupNorm then SiLU, the pair every resnet uses twice. */
function normAct(x: Map4, w: Weights, prefix: string, groups: number): Map4 {
  const normed = groupNorm({
    input: x.data, weight: w(`${prefix}.weight`), bias: w(`${prefix}.bias`),
    N: 1, C: x.C, L: x.H * x.W, G: groups, eps: 1e-6,
  });
  return { data: activation({ input: normed, kind: ACTIVATION.silu }), C: x.C, H: x.H, W: x.W };
}

function resnet(x: Map4, w: Weights, prefix: string, outC: number, groups: number): Map4 {
  let h = normAct(x, w, `${prefix}.norm1`, groups);
  h = conv(h, w, `${prefix}.conv1`, outC, 3, 1);
  h = normAct(h, w, `${prefix}.norm2`, groups);
  h = conv(h, w, `${prefix}.conv2`, outC, 3, 1);

  // The shortcut is a 1x1 conv only when the channel count changes; otherwise
  // the input is added as-is. Deciding that on the weight's presence rather
  // than on `x.C !== outC` keeps the two in step — a checkpoint that disagrees
  // with the shape is a loading bug, not something to paper over.
  let skip = x;
  try {
    skip = conv(x, w, `${prefix}.conv_shortcut`, outC, 1, 0);
  } catch {
    /* no shortcut in this block */
  }
  return { data: elementwise({ a: skip.data, b: h.data, kind: ELEMENTWISE.add }), C: outC, H: h.H, W: h.W };
}

/** `[C, H*W]` to `[H*W, C]` — the layout the attention wants. */
function toTokens(x: Map4): Float32Array {
  const hw = x.H * x.W;
  const out = new Float32Array(hw * x.C);
  for (let c = 0; c < x.C; c += 1) {
    for (let p = 0; p < hw; p += 1) out[p * x.C + c] = x.data[c * hw + p]!;
  }
  return out;
}

/** The inverse of `toTokens`. */
function toMap(tokens: Float32Array, C: number, H: number, W: number): Float32Array {
  const hw = H * W;
  const out = new Float32Array(C * hw);
  for (let p = 0; p < hw; p += 1) {
    for (let c = 0; c < C; c += 1) out[c * hw + p] = tokens[p * C + c]!;
  }
  return out;
}

/** `torch.nn.Linear` over `[rows, in]` with the weight stored `[out, in]`. */
function linear(x: Float32Array, w: Weights, prefix: string, rows: number, inC: number, outC: number): Float32Array {
  const weight = w(`${prefix}.weight`);
  const bias = w(`${prefix}.bias`);
  const wT = new Float32Array(inC * outC);
  for (let o = 0; o < outC; o += 1) {
    for (let i = 0; i < inC; i += 1) wT[i * outC + o] = weight[o * inC + i]!;
  }
  const y = matmul({ a: x, b: wT, M: rows, N: outC, K: inC });
  for (let r = 0; r < rows; r += 1) {
    for (let o = 0; o < outC; o += 1) y[r * outC + o] = y[r * outC + o]! + bias[o]!;
  }
  return y;
}

function midAttention(x: Map4, w: Weights, prefix: string, groups: number): Map4 {
  const normed = groupNorm({
    input: x.data, weight: w(`${prefix}.group_norm.weight`), bias: w(`${prefix}.group_norm.bias`),
    N: 1, C: x.C, L: x.H * x.W, G: groups, eps: 1e-6,
  });
  const hw = x.H * x.W;
  const tokens = toTokens({ data: normed, C: x.C, H: x.H, W: x.W });

  const q = linear(tokens, w, `${prefix}.to_q`, hw, x.C, x.C);
  const k = linear(tokens, w, `${prefix}.to_k`, hw, x.C, x.C);
  const v = linear(tokens, w, `${prefix}.to_v`, hw, x.C, x.C);

  // Single head, so the `[B, H, L, D]` layout `ops/attention` wants is the
  // token layout unchanged.
  const attended = attention({
    q, k, v, B: 1, H: 1, L: hw, S: hw, D: x.C, Dv: x.C, causal: false,
  }).output;

  const projected = linear(attended, w, `${prefix}.to_out.0`, hw, x.C, x.C);
  return {
    data: elementwise({ a: x.data, b: toMap(projected, x.C, x.H, x.W), kind: ELEMENTWISE.add }),
    C: x.C, H: x.H, W: x.W,
  };
}

export function decode(cfg: DecoderConfig, w: Weights, latent: Float32Array, latentH: number, latentW: number): Map4 {
  const groups = cfg.normNumGroups;

  // The sampler hands over a scaled latent; the model decodes the unscaled one.
  const unscaled = new Float32Array(latent.length);
  for (let i = 0; i < latent.length; i += 1) unscaled[i] = latent[i]! / cfg.scalingFactor + cfg.shiftFactor;

  let x: Map4 = { data: unscaled, C: cfg.latentChannels, H: latentH, W: latentW };
  const top = cfg.blockOutChannels[cfg.blockOutChannels.length - 1]!;
  x = conv(x, w, "conv_in", top, 3, 1);

  x = resnet(x, w, "mid_block.resnets.0", top, groups);
  x = midAttention(x, w, "mid_block.attentions.0", groups);
  x = resnet(x, w, "mid_block.resnets.1", top, groups);

  const reversed = [...cfg.blockOutChannels].reverse();
  for (const [i, outC] of reversed.entries()) {
    for (let r = 0; r < cfg.layersPerBlock + 1; r += 1) {
      x = resnet(x, w, `up_blocks.${i}.resnets.${r}`, outC, groups);
    }
    const isFinal = i === cfg.blockOutChannels.length - 1;
    if (!isFinal) {
      x = {
        data: nearestUpsample2d({ input: x.data, N: 1, C: x.C, H: x.H, W: x.W, outH: x.H * 2, outW: x.W * 2 }),
        C: x.C, H: x.H * 2, W: x.W * 2,
      };
      x = conv(x, w, `up_blocks.${i}.upsamplers.0.conv`, outC, 3, 1);
    }
  }

  x = normAct(x, w, "conv_norm_out", groups);
  return conv(x, w, "conv_out", cfg.outChannels, 3, 1);
}
