/**
 * Wan 2.1's VAE decoder, for one frame.
 *
 * Issue #174. Anima decodes with `comfy/ldm/wan/vae.py`, a 3D causal VAE that
 * shares nothing with `examples/zimage-vae`'s: `CausalConv3d`, an RMS norm over
 * channels, temporal upsampling, a frame cache. For video all of that is real.
 *
 * **For one frame none of it runs, and that is ComfyUI's own control flow
 * rather than a simplification taken here.** With `T = 1`:
 *
 *   - `WanVAE.decode` computes `iter_ = 1 + z.shape[2] // 2` = 1, so `feat_map`
 *     stays `None` and there is no cache to thread.
 *   - `CausalConv3d.forward` takes its fast path when `x.shape[2] == 1`, and
 *     that path is `weight = weight[:, :, -input.shape[2]:]` (`comfy/ops.py:613`)
 *     — the **last temporal tap alone**, with the temporal padding already
 *     removed in `__init__`. Each one is exactly a `Conv2d` over `w[:, :, -1]`.
 *   - `Resample`'s `upsample3d` branch is entirely inside `if feat_cache is not
 *     None`, so `time_conv` never runs. Its 768x384x3x1x1 weight is dead weight
 *     for a still image, and this port does not read it.
 *
 * So this is Conv2d, an RMS norm, SiLU, nearest 2x upsampling, and one
 * single-head attention — every one of them an op this repository already has.
 * No `conv3d` was needed, which the issue asked to establish by measuring
 * rather than by assuming either way.
 *
 * `tools/gen_vae_golden.py` bakes the golden and asserts the one place the
 * reduction could still bite: `nearest-exact`, which `Resample` asks for, and
 * `nearest`, which `ops/upsample` provides, are **not** the same resampler in
 * general — but they agree at every integer scale factor, and every stage here
 * is an exact doubling.
 */
import { ACTIVATION, activation } from "../../../ops/activation/index.js";
import { attention } from "../../../ops/attention/index.js";
import { conv2d } from "../../../ops/conv/index.js";
import { ELEMENTWISE, elementwise } from "../../../ops/elementwise/index.js";
import { nearestUpsample2d } from "../../../ops/upsample/index.js";

export interface WanVaeConfig {
  /** The head's output width — 96 for `qwen_image_vae`. */
  dim: number;
  /** Latent channels — 16. */
  zDim: number;
  dimMult: number[];
  numResBlocks: number;
}

/** Anything that hands back a tensor by name, and says whether it has one. */
export interface VaeWeights {
  get(name: string): Float32Array;
  has(name: string): boolean;
}

/** A tensor and the shape the decoder is currently carrying it in. */
interface Plane {
  data: Float32Array;
  C: number;
  H: number;
  W: number;
}

/**
 * `RMS_norm` (`vae.py:50`), which is **not** `ops/rmsnorm`'s eps convention.
 *
 * Upstream is `F.normalize(x, dim=1) * sqrt(dim) * gamma`, and `F.normalize`
 * divides by `max(||x||_2, eps)` with eps 1e-12 — a *clamp on the norm*, where
 * an RMS norm adds eps *inside* the square root. The two agree except when the
 * norm is around 1e-12, and writing `x * rsqrt(mean + eps)` here would be a
 * different function at exactly the inputs nobody tests. Ported as upstream
 * writes it.
 *
 * The normalization is over **channels**, at each pixel — not over a row.
 */
function rmsNorm(x: Plane, gamma: Float32Array): Plane {
  const { C, H, W } = x;
  const hw = H * W;
  const out = new Float32Array(x.data.length);
  const scale = Math.sqrt(C);
  for (let i = 0; i < hw; i += 1) {
    let sq = 0;
    for (let c = 0; c < C; c += 1) {
      const v = x.data[c * hw + i]!;
      sq += v * v;
    }
    const norm = Math.max(Math.sqrt(sq), 1e-12);
    for (let c = 0; c < C; c += 1) {
      out[c * hw + i] = (x.data[c * hw + i]! / norm) * scale * gamma[c]!;
    }
  }
  return { data: out, C, H, W };
}

/**
 * A `CausalConv3d` at `T = 1`: a `Conv2d` over the weight's last temporal tap.
 *
 * The weight arrives as `[Cout, Cin, KT, KH, KW]` and only `[..., KT-1, :, :]`
 * is read. Slicing here rather than at load time keeps the caller's weight
 * source a plain map from the checkpoint's own names.
 */
function causalConv(x: Plane, weight: Float32Array, bias: Float32Array | undefined, Cout: number, kernel: number): Plane {
  const { C: Cin, H, W } = x;
  const perTap = Cout * Cin * kernel * kernel;
  const taps = weight.length / perTap;
  if (!Number.isInteger(taps) || taps < 1) {
    throw new Error(`causalConv: weight of ${weight.length} is not [${Cout}, ${Cin}, KT, ${kernel}, ${kernel}]`);
  }
  const last = new Float32Array(perTap);
  for (let o = 0; o < Cout; o += 1) {
    for (let i = 0; i < Cin; i += 1) {
      const from = ((o * Cin + i) * taps + (taps - 1)) * kernel * kernel;
      last.set(weight.subarray(from, from + kernel * kernel), (o * Cin + i) * kernel * kernel);
    }
  }
  const padding = (kernel - 1) / 2;
  return {
    data: conv2d({
      input: x.data, weight: last, bias, N: 1, Cin, Cout, H, W,
      KH: kernel, KW: kernel, stride: 1, padding,
    }),
    C: Cout, H, W,
  };
}

/** A plain `nn.Conv2d`, for the pieces that were never 3D. */
function conv(x: Plane, weight: Float32Array, bias: Float32Array | undefined, Cout: number, kernel: number): Plane {
  const { C: Cin, H, W } = x;
  const padding = (kernel - 1) / 2;
  return {
    data: conv2d({
      input: x.data, weight, bias, N: 1, Cin, Cout, H, W,
      KH: kernel, KW: kernel, stride: 1, padding,
    }),
    C: Cout, H, W,
  };
}

function silu(x: Plane): Plane {
  return { ...x, data: activation({ input: x.data, kind: ACTIVATION.silu }) };
}

function add(a: Plane, b: Plane): Plane {
  return { ...a, data: elementwise({ a: a.data, b: b.data, kind: ELEMENTWISE.add }) };
}

/**
 * `ResidualBlock` (`vae.py:154`): norm, SiLU, conv, norm, SiLU, conv, plus a
 * shortcut that is a 1x1 conv only when the width changes.
 *
 * The `nn.Sequential` indices are the checkpoint's own names — `residual.0` is
 * the first norm, `residual.2` the first conv, `residual.3` the second norm,
 * `residual.6` the second conv. `residual.5` is the dropout, which is why the
 * numbering skips.
 */
function residualBlock(x: Plane, weights: VaeWeights, prefix: string, outDim: number): Plane {
  let h = rmsNorm(x, weights.get(`${prefix}residual.0.gamma`));
  h = silu(h);
  h = causalConv(h, weights.get(`${prefix}residual.2.weight`), weights.get(`${prefix}residual.2.bias`), outDim, 3);
  h = rmsNorm(h, weights.get(`${prefix}residual.3.gamma`));
  h = silu(h);
  h = causalConv(h, weights.get(`${prefix}residual.6.weight`), weights.get(`${prefix}residual.6.bias`), outDim, 3);

  const shortcut = weights.has(`${prefix}shortcut.weight`)
    ? causalConv(x, weights.get(`${prefix}shortcut.weight`), weights.get(`${prefix}shortcut.bias`), outDim, 1)
    : x;
  return add(h, shortcut);
}

/**
 * `AttentionBlock` (`vae.py:183`) — one head over the whole image.
 *
 * The sequence is every pixel and the head dimension is every channel, which is
 * what `pytorch_attention` builds: `[B, C, H, W]` viewed as `[B, 1, HW, C]`.
 * At the middle block the spatial size is still the latent's, so this is
 * `(H*W)^2` scores — 15,808 tokens at 832x1216, which is why the GPU path
 * tiles it.
 *
 * Its norm is `RMS_norm(dim)` with `images=True`, giving a `[dim, 1, 1]` gamma
 * against the residual blocks' `[dim, 1, 1, 1]`. Both are one number per
 * channel; the shapes differ only because one was written for 4D and the other
 * for 5D.
 */
function attentionBlock(x: Plane, weights: VaeWeights, prefix: string): Plane {
  const { C, H, W } = x;
  const hw = H * W;
  const normed = rmsNorm(x, weights.get(`${prefix}norm.gamma`));
  const qkv = conv(normed, weights.get(`${prefix}to_qkv.weight`), weights.get(`${prefix}to_qkv.bias`), C * 3, 1);

  // `[3C, H, W]` to three `[HW, C]`: the sequence is the pixel and the head
  // dimension is the channel, which is the transpose of how it is stored.
  const take = (part: number): Float32Array => {
    const out = new Float32Array(hw * C);
    for (let c = 0; c < C; c += 1) {
      for (let i = 0; i < hw; i += 1) out[i * C + c] = qkv.data[(part * C + c) * hw + i]!;
    }
    return out;
  };
  const attended = attention({
    q: take(0), k: take(1), v: take(2),
    B: 1, H: 1, L: hw, S: hw, D: C, Dv: C, causal: false,
  });

  // Back to `[C, H, W]`.
  const planes = new Float32Array(C * hw);
  for (let c = 0; c < C; c += 1) {
    for (let i = 0; i < hw; i += 1) planes[c * hw + i] = attended.output[i * C + c]!;
  }
  const projected = conv({ data: planes, C, H, W }, weights.get(`${prefix}proj.weight`), weights.get(`${prefix}proj.bias`), C, 1);
  return add(projected, x);
}

/**
 * `Resample` in either upsample mode: nearest 2x, then a 3x3 conv that halves
 * the width.
 *
 * `time_conv` is not read. For `T = 1` the branch that would call it never
 * runs — see the module comment.
 */
function resample(x: Plane, weights: VaeWeights, prefix: string, outDim: number): Plane {
  const { C, H, W } = x;
  const up = nearestUpsample2d({ input: x.data, N: 1, C, H, W, outH: H * 2, outW: W * 2 });
  return conv({ data: up, C, H: H * 2, W: W * 2 }, weights.get(`${prefix}resample.1.weight`), weights.get(`${prefix}resample.1.bias`), outDim, 3);
}

export interface VaeTrace {
  [name: string]: Float32Array;
}

/**
 * `WanVAE.decode` for a single frame: `[zDim, H, W]` in, `[3, H*8, W*8]` out.
 *
 * The output is in the VAE's own range, roughly `[-1, 1]`, which is what
 * `render.ts`'s `encodePng` takes. The caller is responsible for `process_out`
 * having already been applied to the latent — that is `latentToVae` in
 * `sampler.ts`, and it belongs to the latent format rather than to the decoder.
 */
export function wanVaeDecode(
  cfg: WanVaeConfig,
  weights: VaeWeights,
  latent: Float32Array,
  H: number,
  W: number,
  trace?: VaeTrace,
  onProgress?: (label: string, done: number, total: number) => void,
): Float32Array {
  if (latent.length !== cfg.zDim * H * W) {
    throw new Error(`wanVaeDecode: latent is ${latent.length} values, expected ${cfg.zDim} x ${H} x ${W}`);
  }
  const dims = [cfg.dim * cfg.dimMult[cfg.dimMult.length - 1]!, ...[...cfg.dimMult].reverse().map((m) => cfg.dim * m)];

  // Every `ResidualBlock` and `Resample`, plus the head, plus `conv1`/`conv2`.
  const stages = 2 + 3 + (cfg.dimMult.length * (cfg.numResBlocks + 1) + cfg.dimMult.length - 1) + 1;
  let done = 0;
  const step = (label: string): void => {
    done += 1;
    onProgress?.(label, done, stages);
  };

  step("conv2");
  let x: Plane = causalConv(
    { data: latent, C: cfg.zDim, H, W },
    weights.get("conv2.weight"), weights.get("conv2.bias"), cfg.zDim, 1,
  );

  step("conv1");
  x = causalConv(x, weights.get("decoder.conv1.weight"), weights.get("decoder.conv1.bias"), dims[0]!, 3);
  if (trace) trace.afterConv1 = x.data.slice();

  step("middle");
  x = residualBlock(x, weights, "decoder.middle.0.", dims[0]!);
  x = attentionBlock(x, weights, "decoder.middle.1.");
  if (trace) trace.afterAttention = x.data.slice();
  x = residualBlock(x, weights, "decoder.middle.2.", dims[0]!);
  if (trace) trace.afterMiddle = x.data.slice();

  // The upsample stack, walked by the same rule `Decoder3d.__init__` builds it
  // with: `numResBlocks + 1` residual blocks per stage, then a `Resample`
  // except after the last. `in_dim` halves at stages 1, 2 and 3 because the
  // preceding `Resample` halved the width.
  let at = 0;
  for (let stage = 0; stage < cfg.dimMult.length; stage += 1) {
    const outDim = dims[stage + 1]!;
    for (let block = 0; block < cfg.numResBlocks + 1; block += 1) {
      step(`upsample block ${at}`);
      x = residualBlock(x, weights, `decoder.upsamples.${at}.`, outDim);
      at += 1;
    }
    if (stage !== cfg.dimMult.length - 1) {
      step(`resample ${at}`);
      x = resample(x, weights, `decoder.upsamples.${at}.`, outDim / 2);
      if (trace) trace[`afterUpsample${at}`] = x.data.slice();
      at += 1;
    }
  }

  step("head");
  x = rmsNorm(x, weights.get("decoder.head.0.gamma"));
  x = silu(x);
  x = causalConv(x, weights.get("decoder.head.2.weight"), weights.get("decoder.head.2.bias"), 3, 3);
  return x.data;
}
