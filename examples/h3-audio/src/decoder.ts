/**
 * MiniMax-H3's audio VAE decoder, on the CPU, from this repository's references.
 *
 * Issue #200. H3 generates video with **native stereo audio**, and the audio
 * half of it is a BigVGAN vocoder over a 32-channel latent at 40 Hz. That is a
 * DAC-lineage codec, which is to say it is the same shape as the one
 * `ops/conv_transpose`, `ops/group_norm`, `ops/snake` and `ops/istft` were added
 * for — those five commits landed for VoxShot, and every one of them turns out
 * to be what this model's decoder needs. Nothing here is a new kernel.
 *
 * **This file is the slow, obvious version.** It calls the reference
 * implementations, one allocation per layer, and it exists so the GPU version
 * has something to be wrong against (rule 8). `decoder.test.ts` checks it
 * against a waveform produced by the model's own Python, not by a reading of it.
 *
 * ## The structure, from `dac_audio_vae.py` and `dac_bigvgan.py`
 *
 *     decode(z):                       z is [1, 32, T]
 *       x = conv1d(z, dec_in_proj)     k=1        -> [1, 2048, T]
 *       x = conv1d(x, conv_pre)        k=7 pad=3  -> [1, 1024, T]
 *       for i in 0..6:
 *         x = convTranspose1d(x, ups[i], stride=rate[i], k=kernel[i],
 *                             padding=(k - stride) / 2)
 *         x = mean over j of ampBlock(x, resblocks[3i + j])
 *       x = activation1d(x, post)
 *       x = conv1d(x, conv_post)       k=7 pad=3, no bias -> [1, 1, L]
 *       x = clamp(x, -1, 1)
 *
 * The upsample rates multiply to **800**, so a latent frame is 800 samples and
 * 32 kHz / 800 is the 40 Hz the model card states. Channels halve at every
 * stage, 1024 down to 8, which keeps `channels * length` roughly constant.
 *
 * ## Two conventions that are not guesses
 *
 * **`snake_logscale` is true for this configuration**, so the stored `alpha`
 * and `beta` are *logarithms* and are exponentiated before use. `ops/snake`
 * deliberately does not exponentiate — its own doc says so, because the DACVAE
 * family stores values and the BigVGAN family stores logs, and a kernel that
 * chose one would be wrong for the other. The exponential is here, where the
 * checkpoint's convention is known.
 *
 * **`use_tanh_at_final` is false**, so the output is *clamped* to [-1, 1]
 * rather than squashed. The two agree in the middle and differ at the edges,
 * which is exactly where a loud sample is.
 */
import { conv1d } from "../../../ops/conv/index.js";
import { convTranspose1d } from "../../../ops/conv_transpose/index.js";
import { pad } from "../../../ops/pad/index.js";
import { snakeBeta } from "../../../ops/snake/index.js";

/** What `convert_audio_vae.py` writes beside `decoder.bin`. */
export interface AudioVaeManifest {
  sampleRate: number;
  latentChannels: number;
  latentDim: number;
  decoderDim: number;
  hopLength: number;
  upsampleRates: number[];
  upsampleKernelSizes: number[];
  resblockKernelSizes: number[];
  resblockDilations: number[][];
  snakeLogscale: boolean;
  useTanhAtFinal: boolean;
  antialiasRatio: number;
  antialiasKernelSize: number;
  tensors: { name: string; shape: number[]; offset: number; count: number }[];
  elements: number;
}

/** Named views into the one flat buffer the converter wrote. */
export class AudioVaeWeights {
  private readonly views = new Map<string, Float32Array>();
  readonly shapes = new Map<string, number[]>();

  constructor(manifest: AudioVaeManifest, data: Float32Array) {
    if (data.length !== manifest.elements) {
      throw new Error(`decoder.bin holds ${data.length} f32, the manifest says ${manifest.elements}`);
    }
    for (const entry of manifest.tensors) {
      this.views.set(entry.name, data.subarray(entry.offset, entry.offset + entry.count));
      this.shapes.set(entry.name, entry.shape);
    }
  }

  get(name: string): Float32Array {
    const view = this.views.get(name);
    // Named rather than returning undefined: a missing weight otherwise
    // becomes a silent zero-length multiply somewhere three layers down.
    if (!view) throw new Error(`decoder.bin has no tensor named "${name}"`);
    return view;
  }

  shape(name: string): number[] {
    const shape = this.shapes.get(name);
    if (!shape) throw new Error(`decoder.bin has no tensor named "${name}"`);
    return shape;
  }
}

/** `[N=1, C, L]`, the only layout this decoder moves between layers. */
export interface Signal {
  data: Float32Array;
  C: number;
  L: number;
}

/**
 * BigVGAN's padding for a dilated kernel: `int((k * d - d) / 2)`.
 *
 * `dac_utils.py`'s `get_padding`. It keeps the length unchanged for odd `k`,
 * which every resblock kernel here is (3, 7, 11).
 */
export function amplePadding(kernel: number, dilation: number): number {
  return Math.floor((kernel * dilation - dilation) / 2);
}

/**
 * Anti-aliased SnakeBeta: upsample 2x, apply, downsample 2x.
 *
 * BigVGAN's `Activation1d`. The point is that `sin²` doubles the bandwidth of
 * whatever it is given, so applying it at the signal's own rate folds energy
 * back down as aliasing; running it at twice the rate and filtering on the way
 * out does not.
 *
 * Both filters are the same 12-tap Kaiser-windowed sinc — checked to be the
 * same tensor by the converter, and read from the checkpoint rather than
 * recomputed. The paddings are `alias_free_resample.py`'s, and they are
 * asymmetric on the way down (5 before, 6 after) because the kernel is even.
 */
export function antialiasedSnake(
  x: Signal,
  alpha: Float32Array,
  beta: Float32Array,
  filter: Float32Array,
  ratio: number,
  kernelSize: number,
): Signal {
  const { C, L } = x;
  // UpSample1d: replicate-pad, transposed convolution at `ratio`, scale by
  // `ratio`, then trim the filter's own delay from both ends.
  const padWidth = Math.floor(kernelSize / ratio) - 1;
  const padded = pad({ input: x.data, outer: C, L, inner: 1, before: padWidth, after: padWidth, mode: "replicate" });
  const upLength = L + 2 * padWidth;
  const trimLeft = padWidth * ratio + Math.floor((kernelSize - ratio) / 2);
  const trimRight = padWidth * ratio + Math.floor((kernelSize - ratio + 1) / 2);
  // `alias_free_resample.py` slices the transposed convolution's output. The
  // two trims are **equal** whenever `kernelSize - ratio` is even, which the
  // shipped 12-tap/2x pair is (15 and 15) — and a symmetric trim of the output
  // is exactly what `padding` means to a transposed convolution. So there is no
  // slice here and no slice kernel on the GPU: the crop is an argument.
  //
  // Asserted rather than assumed. An asymmetric trim would need a real slice,
  // and silently taking one extra sample off the front is a phase error, which
  // is inaudible in a unit test and not in a waveform.
  if (trimLeft !== trimRight) {
    throw new Error(
      `antialiasedSnake: kernel ${kernelSize} at ratio ${ratio} trims ${trimLeft}/${trimRight}, ` +
        "which a transposed convolution's symmetric padding cannot express",
    );
  }
  const upperLength = (upLength - 1) * ratio + kernelSize - trimLeft - trimRight;
  // The `ratio *` scale rides on the filter rather than costing a pass: a
  // convolution is linear in its weight, so scaling the taps scales the result.
  const upper = convTranspose1d({
    input: padded,
    weight: repeatPerChannel(filter, C, ratio),
    N: 1,
    Cin: C,
    Cout: C,
    L: upLength,
    K: kernelSize,
    stride: ratio,
    padding: trimLeft,
    groups: C,
  });

  const activated = snakeBeta({ input: upper, alpha, beta, N: 1, C, L: upperLength });

  // DownSample1d: the same filter as a stride-`ratio` convolution, with the
  // even kernel's asymmetric padding.
  // Asymmetric for an even kernel, as upstream. Not observable at this ratio:
  // the upsample above makes `upperLength` exactly `2 * L`, and a stride-2
  // window over an even length never reaches the last padded sample. Kept
  // because it is the model's, and it would matter at another ratio.
  const downLeft = Math.floor(kernelSize / 2) - (kernelSize % 2 === 0 ? 1 : 0);
  const downRight = Math.floor(kernelSize / 2);
  const downPadded = pad({
    input: activated,
    outer: C,
    L: upperLength,
    inner: 1,
    before: downLeft,
    after: downRight,
    mode: "replicate",
  });
  const downLength = upperLength + downLeft + downRight;
  const out = conv1d({
    input: downPadded,
    weight: repeatPerChannel(filter, C),
    N: 1,
    Cin: C,
    Cout: C,
    L: downLength,
    K: kernelSize,
    stride: ratio,
    groups: C,
  });
  return { data: out, C, L: Math.floor((downLength - kernelSize) / ratio) + 1 };
}

/** `filter.expand(C, -1, -1)` — the same taps for every channel, as a depthwise weight. */
function repeatPerChannel(filter: Float32Array, C: number, scale = 1): Float32Array {
  const out = new Float32Array(C * filter.length);
  for (let c = 0; c < C; c += 1) {
    for (let k = 0; k < filter.length; k += 1) out[c * filter.length + k] = scale * filter[k]!;
  }
  return out;
}

/** `exp` when the checkpoint stores logarithms, identity when it stores values. */
function snakeParam(raw: Float32Array, logscale: boolean): Float32Array {
  if (!logscale) return raw;
  return Float32Array.from(raw, Math.exp);
}

/**
 * One AMPBlock: three (activate, dilated conv, activate, conv) pairs, each added
 * back to its input.
 *
 * `dac_bigvgan.py`'s `AMPBlock1.forward` pairs `activations[::2]` with `convs1`
 * and `activations[1::2]` with `convs2`, which is why the activation index is
 * `2 * c` and `2 * c + 1` rather than running.
 */
function ampBlock(
  x: Signal,
  weights: AudioVaeWeights,
  index: number,
  kernel: number,
  dilations: number[],
  manifest: AudioVaeManifest,
  filter: Float32Array,
): Signal {
  let current = x;
  for (let c = 0; c < dilations.length; c += 1) {
    const dilation = dilations[c]!;
    let t = antialiasedSnake(
      current,
      snakeParam(weights.get(`resblocks.${index}.act.${2 * c}.alpha`), manifest.snakeLogscale),
      snakeParam(weights.get(`resblocks.${index}.act.${2 * c}.beta`), manifest.snakeLogscale),
      filter,
      manifest.antialiasRatio,
      manifest.antialiasKernelSize,
    );
    t = {
      data: conv1d({
        input: t.data,
        weight: weights.get(`resblocks.${index}.convs1.${c}.weight`),
        bias: weights.get(`resblocks.${index}.convs1.${c}.bias`),
        N: 1,
        Cin: t.C,
        Cout: t.C,
        L: t.L,
        K: kernel,
        dilation,
        padding: amplePadding(kernel, dilation),
      }),
      C: t.C,
      L: t.L,
    };
    t = antialiasedSnake(
      t,
      snakeParam(weights.get(`resblocks.${index}.act.${2 * c + 1}.alpha`), manifest.snakeLogscale),
      snakeParam(weights.get(`resblocks.${index}.act.${2 * c + 1}.beta`), manifest.snakeLogscale),
      filter,
      manifest.antialiasRatio,
      manifest.antialiasKernelSize,
    );
    t = {
      data: conv1d({
        input: t.data,
        weight: weights.get(`resblocks.${index}.convs2.${c}.weight`),
        bias: weights.get(`resblocks.${index}.convs2.${c}.bias`),
        N: 1,
        Cin: t.C,
        Cout: t.C,
        L: t.L,
        K: kernel,
        dilation: 1,
        padding: amplePadding(kernel, 1),
      }),
      C: t.C,
      L: t.L,
    };
    const summed = new Float32Array(current.data.length);
    for (let i = 0; i < summed.length; i += 1) summed[i] = current.data[i]! + t.data[i]!;
    current = { data: summed, C: current.C, L: current.L };
  }
  return current;
}

/** A latent `[1, latentChannels, T]` in, a waveform `[samples]` out. */
export function decodeAudio(latent: Float32Array, T: number, manifest: AudioVaeManifest, weights: AudioVaeWeights): Float32Array {
  const filter = weights.get("antialias.filter");

  let x: Signal = {
    data: conv1d({
      input: latent,
      weight: weights.get("dec_in_proj.weight"),
      bias: weights.get("dec_in_proj.bias"),
      N: 1,
      Cin: manifest.latentChannels,
      Cout: manifest.latentDim,
      L: T,
      K: 1,
    }),
    C: manifest.latentDim,
    L: T,
  };

  x = {
    data: conv1d({
      input: x.data,
      weight: weights.get("conv_pre.weight"),
      bias: weights.get("conv_pre.bias"),
      N: 1,
      Cin: manifest.latentDim,
      Cout: manifest.decoderDim,
      L: x.L,
      K: 7,
      padding: 3,
    }),
    C: manifest.decoderDim,
    L: x.L,
  };

  const kernels = manifest.resblockKernelSizes;
  for (let i = 0; i < manifest.upsampleRates.length; i += 1) {
    const stride = manifest.upsampleRates[i]!;
    const K = manifest.upsampleKernelSizes[i]!;
    const outChannels = manifest.decoderDim / 2 ** (i + 1);
    const padding = (K - stride) / 2;
    const L = (x.L - 1) * stride - 2 * padding + K;
    x = {
      data: convTranspose1d({
        input: x.data,
        weight: weights.get(`ups.${i}.weight`),
        bias: weights.get(`ups.${i}.bias`),
        N: 1,
        Cin: x.C,
        Cout: outChannels,
        L: x.L,
        K,
        stride,
        padding,
      }),
      C: outChannels,
      L,
    };

    // The multi-receptive-field sum: three blocks over the same input, averaged.
    let summed: Float32Array | null = null;
    for (let j = 0; j < kernels.length; j += 1) {
      const block = ampBlock(x, weights, i * kernels.length + j, kernels[j]!, manifest.resblockDilations[j]!, manifest, filter);
      if (summed === null) summed = block.data.slice();
      else for (let k = 0; k < summed.length; k += 1) summed[k] += block.data[k]!;
    }
    x = { data: Float32Array.from(summed!, (v) => v / kernels.length), C: x.C, L: x.L };
  }

  x = antialiasedSnake(
    x,
    snakeParam(weights.get("activation_post.alpha"), manifest.snakeLogscale),
    snakeParam(weights.get("activation_post.beta"), manifest.snakeLogscale),
    filter,
    manifest.antialiasRatio,
    manifest.antialiasKernelSize,
  );

  const raw = conv1d({
    input: x.data,
    weight: weights.get("conv_post.weight"),
    N: 1,
    Cin: x.C,
    Cout: 1,
    L: x.L,
    K: 7,
    padding: 3,
  });

  // `use_tanh_at_final` is false for this configuration: the output is clamped,
  // not squashed. They agree in the middle and differ exactly where a loud
  // sample is.
  if (manifest.useTanhAtFinal) return Float32Array.from(raw, Math.tanh);
  return Float32Array.from(raw, (v) => Math.min(1, Math.max(-1, v)));
}
