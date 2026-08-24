/**
 * Anima's conditioning path: a prompt to the `context` the DiT cross-attends to.
 *
 *     text --Qwen2 BPE--> ids --Qwen3-0.6B--> source [Lq, 1024]
 *     text --T5 unigram--> ids --.
 *                                '--LLMAdapter--> context [Lt, 1024]
 *                                   * t5 weights, zero-padded to 512
 *
 * Two tokenizers over the same words, which is the surprising part. The Qwen
 * ids condition a 0.6B whose hidden states become the *keys* the adapter reads;
 * the T5 ids index the adapter's own `Embedding(32128, 1024)` and become its
 * *queries*. The T5 model itself is never loaded — only its tokenizer's ids,
 * which act as a second, coarser view of the same prompt.
 *
 * **The adapter ships inside the DiT checkpoint** (`net.llm_adapter.*`), so it
 * arrives already q8-quantized by `convert_dit.py`, while the encoder is read
 * dense from its own file. That split is measured, not stylistic: q8 on the
 * 0.6B's 196 layer matrices moves its output by rel-RMS 0.223, against 0.028
 * for the adapter and 0.040 for the whole 52-block DiT. A 0.6B has one absmax
 * scale per 1024 numbers to spend and its outlier channels do not fit in it.
 *
 * **This is the `native` path, and it is complete.** The released workflow also
 * offers `expanded`, which the custom node builds as (`prompt.py:219`):
 *
 *     expanded = native + strength * (qwen35_adapter(...) - native)
 *
 * — a residual on top of this one, whose own tooltip calls strength 0.0 "native
 * Anima". Issue #173's selective state-space scan is what the Qwen3.5-4B in
 * that branch needs; nothing on this path does.
 */
import { ACTIVATION, activation } from "../../../ops/activation/index.js";
import { attention } from "../../../ops/attention/index.js";
import { ELEMENTWISE, elementwise, elementwiseRows } from "../../../ops/elementwise/index.js";
import { matmul } from "../../../ops/matmul/index.js";
import { rmsnorm } from "../../../ops/rmsnorm/index.js";
import { rope } from "../../../ops/rope/index.js";
import { permuteRopeChannels } from "../../../llm/weights.js";
import type { Qwen3Config } from "../../zimage/src/text-encoder.js";

/**
 * Qwen3-0.6B, read off `Qwen3_06BConfig` (`comfy/text_encoders/llama.py:129`).
 *
 * `stopAfterLayer` is the last layer rather than the second-to-last: Z-Image
 * wants `hidden_states[-2]`, Anima wants `layer="last"`, which is every layer
 * *and* `model.norm`. The norm arrives as `Qwen3Weights.finalNorm`.
 */
export const QWEN3_06B: Qwen3Config = {
  hiddenSize: 1024,
  numLayers: 28,
  numHeads: 16,
  numKvHeads: 8,
  headDim: 128,
  ffnHidden: 3072,
  rmsNormEps: 1e-6,
  ropeTheta: 1000000,
  vocabSize: 151936,
  stopAfterLayer: 27,
};

/**
 * The adapter's own shape (`comfy/ldm/anima/model.py:143`), all defaults —
 * `LLMAdapter()` is constructed with none of them overridden.
 *
 * `in_proj` is absent from the checkpoint because `model_dim === targetDim`
 * makes it `nn.Identity`; there is nothing to load and nothing to apply.
 */
export const ADAPTER = {
  sourceDim: 1024,
  targetDim: 1024,
  modelDim: 1024,
  numLayers: 6,
  numHeads: 16,
  /** `model_dim // num_heads` — 64, not the encoder's 128. */
  headDim: 64,
  mlpRatio: 4,
  vocabSize: 32128,
  /** `RotaryEmbedding.rope_theta`, hardcoded to 10000 in the class body. */
  ropeTheta: 10000,
  /** `operations.RMSNorm(..., eps=1e-6)` at every site. */
  eps: 1e-6,
  /** `torch.nn.functional.pad` to this many tokens, after the weighting. */
  contextLength: 512,
} as const;

/** One adapter block, in the checkpoint's own naming. */
export interface AdapterBlockWeights {
  norm_self_attn: Float32Array;
  self_attn_q_proj: Float32Array;
  self_attn_k_proj: Float32Array;
  self_attn_v_proj: Float32Array;
  self_attn_o_proj: Float32Array;
  self_attn_q_norm: Float32Array;
  self_attn_k_norm: Float32Array;
  norm_cross_attn: Float32Array;
  cross_attn_q_proj: Float32Array;
  cross_attn_k_proj: Float32Array;
  cross_attn_v_proj: Float32Array;
  cross_attn_o_proj: Float32Array;
  cross_attn_q_norm: Float32Array;
  cross_attn_k_norm: Float32Array;
  norm_mlp: Float32Array;
  mlp_0_weight: Float32Array;
  mlp_0_bias: Float32Array;
  mlp_2_weight: Float32Array;
  mlp_2_bias: Float32Array;
}

export interface AdapterWeights {
  /** The `[ids.length, targetDim]` rows of `llm_adapter.embed.weight`. */
  embed(ids: Int32Array): Float32Array;
  /** One block, already permuted for `ops/rope` — see `permuteAdapterBlock`. */
  block(index: number): AdapterBlockWeights;
  out_proj: Float32Array;
  out_proj_bias: Float32Array;
  norm: Float32Array;
}

/**
 * `apply_rotary_pos_emb` splits the head in half (`rotate_half`, `model.py:7`)
 * where `ops/rope` rotates adjacent pairs. Permuting the projections that
 * *produce* a head turns one convention into the other once, offline, rather
 * than on every token — the same relabelling `permuteLayerForRope` does for
 * Z-Image's encoder, and for the same reason.
 *
 * The QK-Norm weights are permuted with the projection they follow. Forgetting
 * that pair is not hypothetical: it cost 1.002e-1 on the Z-Image port before it
 * was found, because a `[headDim]` weight looks like a norm rather than like a
 * per-channel thing that moved.
 */
export function permuteAdapterBlock(w: AdapterBlockWeights): AdapterBlockWeights {
  const { numHeads, headDim, modelDim, sourceDim } = ADAPTER;
  return {
    ...w,
    self_attn_q_proj: permuteRopeChannels(w.self_attn_q_proj, numHeads, headDim, modelDim),
    self_attn_k_proj: permuteRopeChannels(w.self_attn_k_proj, numHeads, headDim, modelDim),
    self_attn_q_norm: permuteRopeChannels(w.self_attn_q_norm, 1, headDim, 1),
    self_attn_k_norm: permuteRopeChannels(w.self_attn_k_norm, 1, headDim, 1),
    cross_attn_q_proj: permuteRopeChannels(w.cross_attn_q_proj, numHeads, headDim, modelDim),
    cross_attn_k_proj: permuteRopeChannels(w.cross_attn_k_proj, numHeads, headDim, sourceDim),
    cross_attn_q_norm: permuteRopeChannels(w.cross_attn_q_norm, 1, headDim, 1),
    cross_attn_k_norm: permuteRopeChannels(w.cross_attn_k_norm, 1, headDim, 1),
  };
}

/** `nn.Linear`, with the bias only where the checkpoint carries one. */
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
  return bias ? elementwiseRows({ a: out, b: bias, S: rows, D: outDim, kind: ELEMENTWISE.add }) : out;
}

/** `[N, H*D]` to `[H, N, D]`, which is what `ops/attention` reads. */
function splitHeads(x: Float32Array, n: number, heads: number, dim: number): Float32Array {
  const out = new Float32Array(x.length);
  for (let t = 0; t < n; t += 1) {
    for (let h = 0; h < heads; h += 1) {
      for (let d = 0; d < dim; d += 1) out[(h * n + t) * dim + d] = x[(t * heads + h) * dim + d]!;
    }
  }
  return out;
}

/** The inverse of `splitHeads`. */
function mergeHeads(x: Float32Array, n: number, heads: number, dim: number): Float32Array {
  const out = new Float32Array(x.length);
  for (let h = 0; h < heads; h += 1) {
    for (let t = 0; t < n; t += 1) {
      for (let d = 0; d < dim; d += 1) out[(t * heads + h) * dim + d] = x[(h * n + t) * dim + d]!;
    }
  }
  return out;
}

/**
 * One `Attention` (`model.py:41`), used for both of the block's two calls.
 *
 * Self-attention passes the same tensor as query and key/value and the same
 * positions to both; cross-attention passes the encoder's hidden states and
 * their own positions. Rope is applied in **both** cases — the cross-attention
 * call is given `position_embeddings_context`, not `None` — which is the part
 * a reader expects to be missing and is not.
 *
 * No mask. `F.scaled_dot_product_attention(..., attn_mask=mask)` is reached
 * with `mask=None` for a single unpadded prompt, so the attention is fully
 * bidirectional in both directions. The encoder before it is causal; this is
 * not, and the two sit two lines apart in the port for that reason.
 */
function attend(
  query: Float32Array,
  keyValue: Float32Array,
  w: AdapterBlockWeights,
  which: "self_attn" | "cross_attn",
  L: number,
  S: number,
  kvDim: number,
): Float32Array {
  const { numHeads, headDim, modelDim, eps, ropeTheta } = ADAPTER;
  const inner = numHeads * headDim;
  const g = (part: string): Float32Array => w[`${which}_${part}` as keyof AdapterBlockWeights] as Float32Array;

  let q = linear(query, g("q_proj"), L, modelDim, inner);
  let k = linear(keyValue, g("k_proj"), S, kvDim, inner);
  const v = linear(keyValue, g("v_proj"), S, kvDim, inner);

  q = rmsnorm({ input: q, weight: g("q_norm"), N: L * numHeads, D: headDim, eps });
  k = rmsnorm({ input: k, weight: g("k_norm"), N: S * numHeads, D: headDim, eps });

  // Both sides start at position 0: `LLMAdapter.forward` builds each from its
  // own `torch.arange`, so a cross-attention key at index 3 is rotated as
  // position 3 of the *context*, not as a continuation of the query.
  q = rope({ input: q, N: L, numHeads, headDim, posOffset: 0, thetaBase: ropeTheta });
  k = rope({ input: k, N: S, numHeads, headDim, posOffset: 0, thetaBase: ropeTheta });

  const attended = attention({
    q: splitHeads(q, L, numHeads, headDim),
    k: splitHeads(k, S, numHeads, headDim),
    v: splitHeads(v, S, numHeads, headDim),
    B: 1, H: numHeads, L, S, D: headDim, Dv: headDim,
    causal: false,
  });
  return linear(mergeHeads(attended.output, L, numHeads, headDim), g("o_proj"), L, inner, modelDim);
}

/**
 * The adapter: T5 ids in, `context` out, unpadded and unweighted.
 *
 * `source` is the encoder's `[Lq, sourceDim]` output. The weighting by
 * `t5xxl_weights` and the pad to 512 belong to `preprocess_text_embeds`
 * (`model.py:196`) and are `padContext`'s job, not this one's — they are the
 * two steps a caller might reasonably want to skip when checking the adapter.
 */
export function llmAdapterForward(
  weights: AdapterWeights,
  source: Float32Array,
  sourceSeq: number,
  targetIds: Int32Array,
  onBlock?: (index: number, hidden: Float32Array) => void,
): Float32Array {
  const { modelDim, sourceDim, numLayers, mlpRatio, eps } = ADAPTER;
  const L = targetIds.length;
  if (L === 0) throw new Error("llmAdapterForward: targetIds must be non-empty");
  if (source.length !== sourceSeq * sourceDim) {
    throw new Error(`llmAdapterForward: source is ${source.length} values, expected ${sourceSeq * sourceDim}.`);
  }
  const mlpHidden = modelDim * mlpRatio;

  // `self.in_proj` is `nn.Identity` for this checkpoint — see `ADAPTER`.
  let x = weights.embed(targetIds);
  if (x.length !== L * modelDim) {
    throw new Error(`llmAdapterForward: embed returned ${x.length} values, expected ${L * modelDim}.`);
  }

  for (let index = 0; index < numLayers; index += 1) {
    const w = weights.block(index);

    const selfNormed = rmsnorm({ input: x, weight: w.norm_self_attn, N: L, D: modelDim, eps });
    x = elementwise({
      a: x,
      b: attend(selfNormed, selfNormed, w, "self_attn", L, L, modelDim),
      kind: ELEMENTWISE.add,
    });

    const crossNormed = rmsnorm({ input: x, weight: w.norm_cross_attn, N: L, D: modelDim, eps });
    x = elementwise({
      a: x,
      b: attend(crossNormed, source, w, "cross_attn", L, sourceSeq, sourceDim),
      kind: ELEMENTWISE.add,
    });

    // `nn.Sequential(Linear, nn.GELU(), Linear)` — `nn.GELU()` with no
    // `approximate` argument is the exact erf form, which is `ACTIVATION.gelu`
    // here and not `gelu_tanh`. Both linears carry a bias.
    const mlpNormed = rmsnorm({ input: x, weight: w.norm_mlp, N: L, D: modelDim, eps });
    const hidden = activation({
      input: linear(mlpNormed, w.mlp_0_weight, L, modelDim, mlpHidden, w.mlp_0_bias),
      kind: ACTIVATION.gelu,
    });
    x = elementwise({
      a: x,
      b: linear(hidden, w.mlp_2_weight, L, mlpHidden, modelDim, w.mlp_2_bias),
      kind: ELEMENTWISE.add,
    });

    onBlock?.(index, x);
  }

  const projected = linear(x, weights.out_proj, L, modelDim, ADAPTER.targetDim, weights.out_proj_bias);
  return rmsnorm({ input: projected, weight: weights.norm, N: L, D: ADAPTER.targetDim, eps });
}

/**
 * `preprocess_text_embeds`' tail: weight each token, then zero-pad to 512.
 *
 * The weights are the tokenizer's per-token emphasis — all 1.0 for a plain
 * prompt (`anima.py:31` sets them so on the Qwen side, and the T5 side carries
 * whatever `(text:1.2)` syntax produced). The pad is **zeros, and unmasked**:
 * the DiT cross-attends to all 512 rows, so the padding is not hidden from it
 * but contributes nothing because its keys and values are zero.
 */
export function padContext(
  context: Float32Array,
  tokens: number,
  weights?: Float32Array | null,
  length = ADAPTER.contextLength,
): Float32Array {
  const dim = ADAPTER.targetDim;
  if (context.length !== tokens * dim) {
    throw new Error(`padContext: context is ${context.length} values, expected ${tokens * dim}.`);
  }
  if (tokens > length) {
    throw new Error(`padContext: ${tokens} tokens exceeds the ${length} the DiT expects.`);
  }
  const out = new Float32Array(length * dim);
  for (let t = 0; t < tokens; t += 1) {
    const scale = weights ? (weights[t] ?? 1) : 1;
    for (let d = 0; d < dim; d += 1) out[t * dim + d] = context[t * dim + d]! * scale;
  }
  return out;
}
