/**
 * The whole Z-Image DiT forward, composed from this repository's ops.
 *
 * `block.ts` is the repeated part and `#166` stage 1 showed it matches the
 * model on real weights. This is everything *around* it, which is where the
 * conventions live and therefore where a port goes wrong. A stack of correct
 * blocks says nothing about any of these:
 *
 *  - **The timestep embedding puts `cos` first.** `torch.cat([cos, sin])`, not
 *    the `sin`-then-`cos` that diffusers and most DiT code use. Swapping them
 *    produces a valid-looking embedding of the wrong timestep, and every image
 *    would be denoised for the wrong amount of time.
 *  - **`t` is scaled by 1000** before it is embedded. The sampler works in
 *    `[0, 1]`; the embedder expects `[0, 1000]`.
 *  - **Image positions start after the caption**: `(cap_seq_len + 1 + f, h, w)`,
 *    while caption positions are `(i + 1, 0, 0)`. Both axes 2 and 3 are zero
 *    for the caption, so the three-axis RoPE degenerates to one axis there.
 *  - **The caption's padding is handled twice, and both are needed.** The row
 *    is zeroed and `cap_pad_token` added, so it carries a learned vector; and
 *    it is *also* dropped from attention — not masked, **trimmed**, with the
 *    output padded back with zeros. Doing only the first measured 1.21
 *    relative RMS at `afterContextRefiner`, and doing the second as a key mask
 *    measured 6.6e-1. See `zimageBlock`'s `validSeq` for why they differ.
 *  - **The final layer is a LayerNorm**, not the RMSNorm used everywhere else,
 *    with `eps=1e-6` and no affine of its own, scaled by `1 + adaLN(SiLU(c))`.
 *  - **`context_refiner`'s blocks have no modulation** and genuinely lack the
 *    adaLN tensors — see `zimageBlock`.
 *
 * Every one of those was read out of `zimage_model.py` (rule 2). Correctness is
 * `fixtures/forward.*`, which is the model's own forward with hooks on the
 * intermediates, not this file's reasoning.
 *
 * Deliberately CPU: this is `ops/*`'s reference path throughout, so it is a
 * statement about correctness only. The GPU version is #166 stage 5's problem
 * and has a verified answer to check itself against once this lands.
 */
import { ACTIVATION, activation } from "../../../ops/activation/index.js";
import { layernorm } from "../../../ops/layernorm/index.js";
import { matmul } from "../../../ops/matmul/index.js";
import { rmsnorm } from "../../../ops/rmsnorm/index.js";
import { type BlockConfig, type BlockWeights, zimageBlock } from "./block.js";
/** Anything that can hand back a tensor by name — `DitWeights` or the lazy
 *  reader in `weights-node.ts`, which is what makes a 12 GB dense copy avoidable. */
export interface WeightSource {
  get(name: string): Float32Array;
}

export interface DitConfig {
  dim: number;
  nHeads: number;
  nLayers: number;
  nRefinerLayers: number;
  inChannels: number;
  patchSize: number;
  capFeatDim: number;
  normEps: number;
  ropeAxesDims: number[];
  ropeTheta: number;
  tScale: number;
  adalnEmbedDim: number;
  frequencyEmbeddingSize: number;
  maxPeriod: number;
}

/** `nn.Linear` with an optional bias: `y = x @ W^T + b`, `W` stored `[out, in]`. */
function linear(
  x: Float32Array,
  weight: Float32Array,
  rows: number,
  inDim: number,
  outDim: number,
  bias?: Float32Array,
): Float32Array {
  const wT = new Float32Array(inDim * outDim);
  for (let o = 0; o < outDim; o += 1) {
    for (let i = 0; i < inDim; i += 1) wT[i * outDim + o] = weight[o * inDim + i]!;
  }
  const out = matmul({ a: x, b: wT, M: rows, N: outDim, K: inDim });
  if (bias) {
    for (let r = 0; r < rows; r += 1) {
      for (let o = 0; o < outDim; o += 1) out[r * outDim + o] = out[r * outDim + o]! + bias[o]!;
    }
  }
  return out;
}

/**
 * The sinusoidal timestep embedding, `cos` first.
 *
 * `freqs = exp(-log(maxPeriod) * arange(half) / half)`, then
 * `cat([cos(t * freqs), sin(t * freqs)])`. The order is the model's
 * (`zimage_model.py:70`) and is the opposite of the convention most DiT
 * implementations use, so it is spelled out rather than reached for.
 */
export function timestepEmbedding(t: number, dim: number, maxPeriod: number): Float32Array {
  const half = Math.floor(dim / 2);
  const out = new Float32Array(dim);
  for (let i = 0; i < half; i += 1) {
    const freq = Math.exp((-Math.log(maxPeriod) * i) / half);
    const arg = t * freq;
    out[i] = Math.cos(arg);
    out[half + i] = Math.sin(arg);
  }
  return out;
}

/**
 * `[C, F, H, W]` to `[seq, pF*pH*pW*C]`, the model's `patchify`.
 *
 * The channel axis ends up **last** within a patch (`permute(..., 3, 5, 7, 1)`),
 * which is the detail worth stating: putting it first would give a tensor of
 * the same shape whose every row is a different linear combination.
 */
export function patchify(
  x: Float32Array,
  channels: number,
  F: number,
  H: number,
  W: number,
  patch: number,
  fPatch: number,
): Float32Array {
  const fTokens = F / fPatch;
  const hTokens = H / patch;
  const wTokens = W / patch;
  const patchDim = fPatch * patch * patch * channels;
  const out = new Float32Array(fTokens * hTokens * wTokens * patchDim);
  let at = 0;
  for (let ft = 0; ft < fTokens; ft += 1) {
    for (let ht = 0; ht < hTokens; ht += 1) {
      for (let wt = 0; wt < wTokens; wt += 1) {
        for (let pf = 0; pf < fPatch; pf += 1) {
          for (let ph = 0; ph < patch; ph += 1) {
            for (let pw = 0; pw < patch; pw += 1) {
              for (let c = 0; c < channels; c += 1) {
                const f = ft * fPatch + pf;
                const h = ht * patch + ph;
                const w = wt * patch + pw;
                out[at] = x[((c * F + f) * H + h) * W + w]!;
                at += 1;
              }
            }
          }
        }
      }
    }
  }
  return out;
}

/** The inverse, taking only the first `fTokens*hTokens*wTokens` rows. */
export function unpatchify(
  x: Float32Array,
  channels: number,
  F: number,
  H: number,
  W: number,
  patch: number,
  fPatch: number,
): Float32Array {
  const fTokens = F / fPatch;
  const hTokens = H / patch;
  const wTokens = W / patch;
  const patchDim = fPatch * patch * patch * channels;
  const out = new Float32Array(channels * F * H * W);
  let at = 0;
  for (let ft = 0; ft < fTokens; ft += 1) {
    for (let ht = 0; ht < hTokens; ht += 1) {
      for (let wt = 0; wt < wTokens; wt += 1) {
        for (let pf = 0; pf < fPatch; pf += 1) {
          for (let ph = 0; ph < patch; ph += 1) {
            for (let pw = 0; pw < patch; pw += 1) {
              for (let c = 0; c < channels; c += 1) {
                const f = ft * fPatch + pf;
                const h = ht * patch + ph;
                const w = wt * patch + pw;
                out[((c * F + f) * H + h) * W + w] = x[at]!;
                at += 1;
              }
            }
          }
        }
      }
    }
  }
  return out;
}

/** `(capSeqLen + 1 + f, h, w)` per image token, row-major over `(f, h, w)`. */
export function imagePositionIds(fTokens: number, hTokens: number, wTokens: number, capSeqLen: number): Int32Array {
  const out = new Int32Array(fTokens * hTokens * wTokens * 3);
  let at = 0;
  for (let f = 0; f < fTokens; f += 1) {
    for (let h = 0; h < hTokens; h += 1) {
      for (let w = 0; w < wTokens; w += 1) {
        out[at] = capSeqLen + 1 + f;
        out[at + 1] = h;
        out[at + 2] = w;
        at += 3;
      }
    }
  }
  return out;
}

/** `(i + 1, 0, 0)` per caption token. */
export function captionPositionIds(capSeqLen: number): Int32Array {
  const out = new Int32Array(capSeqLen * 3);
  for (let i = 0; i < capSeqLen; i += 1) out[i * 3] = i + 1;
  return out;
}

/** The tensor names one block needs, in the checkpoint's spelling. */
const BLOCK_TENSORS = [
  "attention.to_q.weight",
  "attention.to_k.weight",
  "attention.to_v.weight",
  "attention.to_out.0.weight",
  "attention.norm_q.weight",
  "attention.norm_k.weight",
  "feed_forward.w1.weight",
  "feed_forward.w2.weight",
  "feed_forward.w3.weight",
  "attention_norm1.weight",
  "ffn_norm1.weight",
  "attention_norm2.weight",
  "ffn_norm2.weight",
] as const;
const MODULATION_TENSORS = ["adaLN_modulation.0.weight", "adaLN_modulation.0.bias"] as const;

function collect(weights: WeightSource, prefix: string, modulated: boolean): BlockWeights {
  const names = modulated ? [...BLOCK_TENSORS, ...MODULATION_TENSORS] : BLOCK_TENSORS;
  const out: Record<string, Float32Array> = {};
  for (const name of names) out[name.replace(/\./g, "_")] = weights.get(`${prefix}${name}`);
  return out as unknown as BlockWeights;
}

/** What the caller can ask to be kept, to see where a mismatch starts. */
export interface DitTrace {
  adalnInput?: Float32Array;
  afterNoiseRefiner?: Float32Array;
  afterContextRefiner?: Float32Array;
  afterLayer0?: Float32Array;
  afterLayers?: Float32Array;
}

export interface DitInput {
  /** `[inChannels, F, H, W]`, the noised latent. */
  latent: Float32Array;
  F: number;
  H: number;
  W: number;
  /** The rectified-flow time in `[0, 1]`; scaled by `tScale` inside. */
  t: number;
  /** `[capSeqLen, capFeatDim]` from the text encoder. */
  capFeats: Float32Array;
  /** `[capSeqLen]`, non-zero for a real token. */
  capMask: Float32Array;
}

/**
 * One denoising step's velocity prediction: latent in, latent-shaped out.
 *
 * Batch is fixed at one. The model supports more, but every caller here has a
 * single image and a batch axis threaded through the CPU reference path would
 * be shape bookkeeping with nothing to check it.
 */
export function ditForward(
  cfg: DitConfig,
  weights: WeightSource,
  input: DitInput,
  trace?: DitTrace,
): Float32Array {
  const { dim, nHeads, patchSize, inChannels, normEps } = cfg;
  const headDim = dim / nHeads;
  const { F, H, W } = input;
  const fTokens = F;
  const hTokens = H / patchSize;
  const wTokens = W / patchSize;
  const xSeq = fTokens * hTokens * wTokens;
  const capSeq = input.capMask.length;
  const patchDim = patchSize * patchSize * inChannels;

  // --- timestep ---
  const tFreq = timestepEmbedding(input.t * cfg.tScale, cfg.frequencyEmbeddingSize, cfg.maxPeriod);
  const tMid = linear(
    tFreq,
    weights.get("t_embedder.mlp.0.weight"),
    1,
    cfg.frequencyEmbeddingSize,
    weights.get("t_embedder.mlp.0.bias").length,
    weights.get("t_embedder.mlp.0.bias"),
  );
  const adalnInput = linear(
    activation({ input: tMid, kind: ACTIVATION.silu }),
    weights.get("t_embedder.mlp.2.weight"),
    1,
    tMid.length,
    cfg.adalnEmbedDim,
    weights.get("t_embedder.mlp.2.bias"),
  );
  if (trace) trace.adalnInput = adalnInput;

  // --- image tokens ---
  const key = `${patchSize}-1`;
  let x = linear(
    patchify(input.latent, inChannels, F, H, W, patchSize, 1),
    weights.get(`all_x_embedder.${key}.weight`),
    xSeq,
    patchDim,
    dim,
    weights.get(`all_x_embedder.${key}.bias`),
  );

  const xPositions = imagePositionIds(fTokens, hTokens, wTokens, capSeq);
  const ffnHidden = weights.get("layers.0.feed_forward.w1.weight").length / dim;
  const blockCfg = (seq: number): BlockConfig => ({
    dim,
    nHeads,
    headDim,
    seq,
    ffnHidden,
    normEps,
    ropeAxesDims: cfg.ropeAxesDims,
    ropeTheta: cfg.ropeTheta,
  });

  for (let i = 0; i < cfg.nRefinerLayers; i += 1) {
    x = zimageBlock(blockCfg(xSeq), collect(weights, `noise_refiner.${i}.`, true), x, adalnInput, xPositions);
  }
  if (trace) trace.afterNoiseRefiner = x.slice();

  // --- caption tokens ---
  const capNormed = rmsnorm({
    input: input.capFeats,
    weight: weights.get("cap_embedder.0.weight"),
    N: capSeq,
    D: cfg.capFeatDim,
    eps: normEps,
  });
  let cap = linear(
    capNormed,
    weights.get("cap_embedder.1.weight"),
    capSeq,
    cfg.capFeatDim,
    dim,
    weights.get("cap_embedder.1.bias"),
  );

  // Padding is replaced, not masked: the row is zeroed and `cap_pad_token` put
  // in its place, so it attends as a learned vector. A port that only skipped
  // the rows would give every short prompt a different context.
  const capPad = weights.get("cap_pad_token");
  for (let i = 0; i < capSeq; i += 1) {
    if (input.capMask[i]) continue;
    for (let d = 0; d < dim; d += 1) cap[i * dim + d] = capPad[d]!;
  }

  // How many caption tokens are real. The model's `seqlens`, and the length it
  // trims `q`/`k`/`v` to.
  let capValid = 0;
  while (capValid < capSeq && input.capMask[capValid]) capValid += 1;
  for (let i = capValid; i < capSeq; i += 1) {
    if (input.capMask[i]) {
      // The trim only means what the model means by it if the real tokens are a
      // prefix. Upstream assumes this without checking; here it is an error
      // rather than a silently different set of attended tokens.
      throw new Error(
        `ditForward: capMask must be a prefix of real tokens — token ${i} is real but ${capValid} is not.`,
      );
    }
  }

  const capPositions = captionPositionIds(capSeq);
  for (let i = 0; i < cfg.nRefinerLayers; i += 1) {
    // `modulation=False` here — no adaLN tensors exist for these layers.
    cap = zimageBlock(
      blockCfg(capSeq),
      collect(weights, `context_refiner.${i}.`, false),
      cap,
      null,
      capPositions,
      capValid,
    );
  }
  if (trace) trace.afterContextRefiner = cap.slice();

  // --- unified stack: image tokens first, then caption ---
  const unifiedSeq = xSeq + capSeq;
  let unified: Float32Array = new Float32Array(unifiedSeq * dim);
  unified.set(x, 0);
  unified.set(cap, xSeq * dim);
  const positions = new Int32Array(unifiedSeq * 3);
  positions.set(xPositions, 0);
  positions.set(capPositions, xSeq * 3);

  // Every image token is real, so the unified valid length is the image half
  // plus the caption's — the model's `seqlens = cap_mask.sum(1) + img_len`.
  const unifiedValid = xSeq + capValid;

  for (let i = 0; i < cfg.nLayers; i += 1) {
    unified = zimageBlock(
      blockCfg(unifiedSeq),
      collect(weights, `layers.${i}.`, true),
      unified,
      adalnInput,
      positions,
      unifiedValid,
    );
    if (trace && i === 0) trace.afterLayer0 = unified.slice();
  }
  if (trace) trace.afterLayers = unified.slice();

  // --- final layer: LayerNorm without affine, scaled by 1 + adaLN(SiLU(c)) ---
  const scale = linear(
    activation({ input: adalnInput, kind: ACTIVATION.silu }),
    weights.get(`all_final_layer.${key}.adaLN_modulation.1.weight`),
    1,
    cfg.adalnEmbedDim,
    dim,
    weights.get(`all_final_layer.${key}.adaLN_modulation.1.bias`),
  );
  for (let d = 0; d < dim; d += 1) scale[d] = 1 + scale[d]!;

  // `elementwise_affine=False` in the model, so ones and zeros here rather than
  // a learned pair — `ops/layernorm` always applies both, and 1/0 is the
  // identity for them.
  const normed = layernorm({
    input: unified,
    weight: new Float32Array(dim).fill(1),
    bias: new Float32Array(dim),
    N: unifiedSeq,
    D: dim,
    eps: 1e-6,
  });
  for (let s = 0; s < unifiedSeq; s += 1) {
    for (let d = 0; d < dim; d += 1) normed[s * dim + d] = normed[s * dim + d]! * scale[d]!;
  }

  const projected = linear(
    normed,
    weights.get(`all_final_layer.${key}.linear.weight`),
    unifiedSeq,
    dim,
    patchDim,
    weights.get(`all_final_layer.${key}.linear.bias`),
  );

  // `slice`, not `subarray`: the caption rows are dropped here and a view onto
  // them would keep the whole projection alive for the caller's lifetime.
  return unpatchify(projected.slice(0, xSeq * patchDim), inChannels, F, H, W, patchSize, 1);
}
