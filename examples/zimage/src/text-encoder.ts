/**
 * Z-Image's text encoder — Qwen3-4B, stopped two hidden states from the end.
 *
 * The DiT takes `cap_feats` of width 2560, which is Qwen3-4B's `hidden_size`,
 * and Z-Image reads them from `hidden_states[-2]` (`zimage_utils.py:214`).
 * Which layer that names was **measured**, not read: the loop in this version
 * of `transformers` does not collect the hidden states itself, so the golden's
 * generator hooks every decoder layer and matches. The answer is in
 * `fixtures/text-encoder.manifest.json` under `hiddenStatesIndex`, and it is
 * what `stopAfterLayer` is set from.
 *
 * Being off by one produces embeddings of the right shape and the wrong
 * content, which the DiT consumes without complaint.
 *
 * ## Why this is not `llm/engine.ts`
 *
 * It nearly is. Qwen3's decoder layer is Llama's with **QK-Norm** added —
 * RMSNorm over each head's own 128 channels, on Q and K, between the
 * projection and RoPE. `llm/`'s three engines have no such step (checked:
 * `q_norm`/`k_norm` appear nowhere in them), so none of them can run this
 * checkpoint. Adding it there is the right home for it and its own piece of
 * work, because it has to land in the CPU engine and both GPU ones together.
 * Stage 3 needs a *correct* encoder before it needs a fast one, so this is
 * composed from `ops/*` the way `dit.ts` is, and the golden it is verified
 * against is what the engine can be checked with later.
 *
 * ## Two conventions that are traps
 *
 *  - **HF pairs RoPE channels half a head apart**, where `ops/rope` pairs them
 *    adjacently. `llm/weights.ts`'s `permuteRopeChannels` already fixes this at
 *    weight-load time, and is reused here rather than re-derived.
 *  - **QK-Norm's weight must be permuted with the projection.** RMSNorm's
 *    normalisation is permutation-invariant, but its per-channel multiply is
 *    not: the weight is `[headDim]` and applies channel by channel *inside* a
 *    head, exactly where the permutation reorders things. Permuting `q_proj`
 *    and leaving `q_norm.weight` alone multiplies each channel by another
 *    channel's scale — a plausible tensor, entirely wrong.
 *
 * ## Padding
 *
 * Upstream pads the prompt to 512 and passes the mask. It does not need to be
 * reproduced: attention here is **causal** and the padding is on the right, so
 * a real token never attends to one. Running the first `validTokens` rows
 * gives bit-identical results for those rows, and the DiT is handed a mask for
 * the rest anyway. That is an argument from the mask's shape, so the golden is
 * generated *padded* and compared against this *unpadded* run — if the
 * reasoning were wrong, the numbers would say so.
 */
import { ACTIVATION, activation } from "../../../ops/activation/index.js";
import { ELEMENTWISE, elementwise } from "../../../ops/elementwise/index.js";
import { groupedAttention } from "../../../ops/gqa/index.js";
import { matmul } from "../../../ops/matmul/index.js";
import { rmsnorm } from "../../../ops/rmsnorm/index.js";
import { rope } from "../../../ops/rope/index.js";
import { permuteRopeChannels } from "../../../llm/weights.js";

export interface Qwen3Config {
  hiddenSize: number;
  numLayers: number;
  numHeads: number;
  numKvHeads: number;
  headDim: number;
  ffnHidden: number;
  rmsNormEps: number;
  ropeTheta: number;
  vocabSize: number;
  /**
   * The last layer to run, inclusive. `hidden_states[-2]` is this layer's
   * output, taken **before** `model.norm`, so the final norm and the LM head
   * are never reached.
   */
  stopAfterLayer: number;
}

/** One decoder layer, in the checkpoint's own naming. */
export interface Qwen3LayerWeights {
  input_layernorm: Float32Array;
  q_proj: Float32Array;
  k_proj: Float32Array;
  v_proj: Float32Array;
  o_proj: Float32Array;
  q_norm: Float32Array;
  k_norm: Float32Array;
  post_attention_layernorm: Float32Array;
  gate_proj: Float32Array;
  up_proj: Float32Array;
  down_proj: Float32Array;
}

/**
 * Weights as *providers*, not as arrays.
 *
 * 35 layers of Qwen3-4B are 14 GB as dense f32 and the embedding table alone is
 * 1.55 GB, of which a 15-token prompt reads 15 rows. Holding either in full to
 * run one prompt is the mistake `weights-node.ts` already had to undo for the
 * DiT, so the interface asks for a layer at a time and for the rows of the
 * table that are wanted — which is also how the browser will have them.
 */
export interface Qwen3Weights {
  /** The embedding rows for these ids, `[ids.length, hiddenSize]`. */
  embed(ids: Int32Array): Float32Array;
  numLayers: number;
  /** One layer, already permuted for `ops/rope` — see `permuteLayerForRope`. */
  layer(index: number): Qwen3LayerWeights;
}

/** `nn.Linear` without bias — Qwen3 sets `attention_bias: false` throughout. */
function linear(x: Float32Array, weight: Float32Array, rows: number, inDim: number, outDim: number): Float32Array {
  const wT = new Float32Array(inDim * outDim);
  for (let o = 0; o < outDim; o += 1) {
    for (let i = 0; i < inDim; i += 1) wT[i * outDim + o] = weight[o * inDim + i]!;
  }
  return matmul({ a: x, b: wT, M: rows, N: outDim, K: inDim });
}

/** `[S, H*D]` token-major to `[H, S, D]` head-major. */
function splitHeads(x: Float32Array, seq: number, heads: number, dim: number): Float32Array {
  const out = new Float32Array(seq * heads * dim);
  for (let s = 0; s < seq; s += 1) {
    for (let h = 0; h < heads; h += 1) {
      for (let d = 0; d < dim; d += 1) out[(h * seq + s) * dim + d] = x[(s * heads + h) * dim + d]!;
    }
  }
  return out;
}

/** The inverse. */
function mergeHeads(x: Float32Array, seq: number, heads: number, dim: number): Float32Array {
  const out = new Float32Array(seq * heads * dim);
  for (let h = 0; h < heads; h += 1) {
    for (let s = 0; s < seq; s += 1) {
      for (let d = 0; d < dim; d += 1) out[(s * heads + h) * dim + d] = x[(h * seq + s) * dim + d]!;
    }
  }
  return out;
}

/**
 * Puts one layer's Q/K weights into `ops/rope`'s channel order.
 *
 * The norm weights go through the same permutation with `heads = 1` and
 * `inFeatures = 1`, which is the `[headDim]` case of the same relabelling —
 * see the note at the top on why leaving them alone is wrong.
 */
export function permuteLayerForRope(layer: Qwen3LayerWeights, cfg: Qwen3Config): Qwen3LayerWeights {
  const { numHeads, numKvHeads, headDim, hiddenSize } = cfg;
  return {
    ...layer,
    q_proj: permuteRopeChannels(layer.q_proj, numHeads, headDim, hiddenSize),
    k_proj: permuteRopeChannels(layer.k_proj, numKvHeads, headDim, hiddenSize),
    q_norm: permuteRopeChannels(layer.q_norm, 1, headDim, 1),
    k_norm: permuteRopeChannels(layer.k_norm, 1, headDim, 1),
  };
}

/**
 * Token ids to `cap_feats`, `[tokens, hiddenSize]`.
 *
 * `weights.layers` must already be permuted — `permuteLayerForRope` — because
 * doing it here would repeat the work on every call and hide where it happens.
 */
export function qwen3Encode(cfg: Qwen3Config, weights: Qwen3Weights, tokenIds: Int32Array): Float32Array {
  const { hiddenSize, numHeads, numKvHeads, headDim, ffnHidden, rmsNormEps, ropeTheta } = cfg;
  const seq = tokenIds.length;
  if (seq === 0) throw new Error("qwen3Encode: tokenIds must be non-empty");
  const qDim = numHeads * headDim;
  const kvDim = numKvHeads * headDim;

  let hidden = weights.embed(tokenIds);
  if (hidden.length !== seq * hiddenSize) {
    throw new Error(`qwen3Encode: embed returned ${hidden.length} values, expected ${seq * hiddenSize}.`);
  }

  const last = cfg.stopAfterLayer;
  if (last < 0 || last >= weights.numLayers) {
    throw new Error(`qwen3Encode: stopAfterLayer ${last} is outside [0, ${weights.numLayers - 1}].`);
  }

  for (let l = 0; l <= last; l += 1) {
    const w = weights.layer(l);

    // --- attention ---
    const normed = rmsnorm({ input: hidden, weight: w.input_layernorm, N: seq, D: hiddenSize, eps: rmsNormEps });

    let q = linear(normed, w.q_proj, seq, hiddenSize, qDim);
    let k = linear(normed, w.k_proj, seq, hiddenSize, kvDim);
    const v = linear(normed, w.v_proj, seq, hiddenSize, kvDim);

    // QK-Norm: over each head's own channels, so the rows are counted per head
    // rather than per token — the same shape `block.ts` uses for Z-Image's.
    q = rmsnorm({ input: q, weight: w.q_norm, N: seq * numHeads, D: headDim, eps: rmsNormEps });
    k = rmsnorm({ input: k, weight: w.k_norm, N: seq * numKvHeads, D: headDim, eps: rmsNormEps });

    q = rope({ input: q, N: seq, numHeads, headDim, posOffset: 0, thetaBase: ropeTheta });
    k = rope({ input: k, N: seq, numHeads: numKvHeads, headDim, posOffset: 0, thetaBase: ropeTheta });

    const attn = groupedAttention({
      q: splitHeads(q, seq, numHeads, headDim),
      k: splitHeads(k, seq, numKvHeads, headDim),
      v: splitHeads(v, seq, numKvHeads, headDim),
      B: 1, H: numHeads, kvHeads: numKvHeads, L: seq, S: seq, D: headDim, Dv: headDim,
      causal: true,
    });
    const projected = linear(mergeHeads(attn.output, seq, numHeads, headDim), w.o_proj, seq, qDim, hiddenSize);
    hidden = elementwise({ a: hidden, b: projected, kind: ELEMENTWISE.add });

    // --- SwiGLU MLP ---
    const normed2 = rmsnorm({
      input: hidden, weight: w.post_attention_layernorm, N: seq, D: hiddenSize, eps: rmsNormEps,
    });
    const gate = activation({
      input: linear(normed2, w.gate_proj, seq, hiddenSize, ffnHidden),
      kind: ACTIVATION.silu,
    });
    const up = linear(normed2, w.up_proj, seq, hiddenSize, ffnHidden);
    const down = linear(
      elementwise({ a: gate, b: up, kind: ELEMENTWISE.multiply }),
      w.down_proj,
      seq,
      ffnHidden,
      hiddenSize,
    );
    hidden = elementwise({ a: hidden, b: down, kind: ELEMENTWISE.add });
  }

  // No `model.norm` and no LM head: `hidden_states[-2]` is taken before both.
  return hidden;
}
