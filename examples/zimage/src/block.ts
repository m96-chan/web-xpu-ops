/**
 * One `ZImageTransformerBlock`, composed from this repository's ops.
 *
 * The point is not the block. It is the question issue #163 asks: after
 * #145/#146/#149-#152 landed, does the op set actually reach a real model, or
 * does it only reach each op's own reference? Those are different claims, and
 * the second one is the easy one.
 *
 * So this is deliberately a *composition* and not a kernel. Every line below
 * is an existing op; nothing new is introduced, and where something had to be
 * done on the CPU that is called out rather than hidden, because a CPU step
 * here is a missing op there.
 *
 * The structure follows Z-Image's own `_forward` (Tongyi-MAI, via
 * musubi-tuner's copy, `zimage_model.py:259`):
 *
 *     scale_msa, gate_msa, scale_mlp, gate_mlp = adaLN(adaln_input).chunk(4)
 *     gate_msa, gate_mlp = tanh(gate_msa), tanh(gate_mlp)
 *     scale_msa, scale_mlp = 1 + scale_msa, 1 + scale_mlp
 *     attn = attention(norm1(x) * scale_msa, freqs)
 *     x = x + gate_msa * norm2(attn)
 *     x = x + gate_mlp * ffn_norm2(feed_forward(ffn_norm1(x) * scale_mlp))
 *
 * Two facts about the model that were read from its source rather than assumed
 * (rule 2), because getting either wrong produces a plausible wrong answer:
 *
 * `ropeAxes` takes the position ids themselves rather than precomputed
 * cos/sin tables, so upstream's `RopeEmbedder` (which gathers from per-axis
 * tables) has no counterpart here — the gather is the indexing.
 *
 *  - **RoPE pairs adjacent channels.** `apply_rotary_emb` does
 *    `view_as_complex(x.reshape(..., -1, 2))`, i.e. `(2i, 2i+1)` — the same
 *    convention `ops/rope` already uses. HF Llama's `rotate_half` pairs `i`
 *    with `i + D/2` and would have needed the permutation `llm/weights.ts`
 *    carries for the LLM path. Z-Image needs none.
 *  - **Attention is MHA, not GQA.** The shipped config is `n_heads=30,
 *    n_kv_heads=30`; the code keeps a separate `n_kv_heads` but never uses a
 *    smaller one. `ops/gqa` would work, but `ops/attention` is what the model
 *    is.
 */
import { ACTIVATION, activation } from "../../../ops/activation/index.js";
import { attention } from "../../../ops/attention/index.js";
import { ELEMENTWISE, elementwise, elementwiseRows } from "../../../ops/elementwise/index.js";
import { matmul } from "../../../ops/matmul/index.js";
import { rmsnorm } from "../../../ops/rmsnorm/index.js";
import { ropeAxes } from "../../../ops/rope/index.js";

export interface BlockConfig {
  dim: number;
  nHeads: number;
  headDim: number;
  seq: number;
  ffnHidden: number;
  normEps: number;
  ropeAxesDims: number[];
  ropeTheta: number;
}

/** Every parameter of one block, in the model's own naming. */
export interface BlockWeights {
  attention_to_q_weight: Float32Array;
  attention_to_k_weight: Float32Array;
  attention_to_v_weight: Float32Array;
  attention_to_out_0_weight: Float32Array;
  attention_norm_q_weight: Float32Array;
  attention_norm_k_weight: Float32Array;
  feed_forward_w1_weight: Float32Array;
  feed_forward_w2_weight: Float32Array;
  feed_forward_w3_weight: Float32Array;
  attention_norm1_weight: Float32Array;
  ffn_norm1_weight: Float32Array;
  attention_norm2_weight: Float32Array;
  ffn_norm2_weight: Float32Array;
  adaLN_modulation_0_weight: Float32Array;
  adaLN_modulation_0_bias: Float32Array;
}

/**
 * `torch.nn.Linear`: `y = x @ W^T` with `W` stored `[out, in]`.
 *
 * `matmul` takes `b` as `[K, N]`, so the weight is transposed on the way in.
 * That transpose is CPU work here on purpose — for a real run the weights are
 * uploaded once and `ops/transpose` (or a converter) does it ahead of time, so
 * paying it per call would measure the wrong thing. It is written out rather
 * than reached for silently so the cost is visible.
 */
function linear(x: Float32Array, weight: Float32Array, rows: number, inDim: number, outDim: number): Float32Array {
  const wT = new Float32Array(inDim * outDim);
  for (let o = 0; o < outDim; o += 1) {
    for (let i = 0; i < inDim; i += 1) wT[i * outDim + o] = weight[o * inDim + i]!;
  }
  return matmul({ a: x, b: wT, M: rows, N: outDim, K: inDim });
}

/** `[S, H*D]` token-major to `[H, S, D]` head-major, `ops/attention`'s layout. */
function splitHeads(x: Float32Array, seq: number, heads: number, dim: number): Float32Array {
  const out = new Float32Array(seq * heads * dim);
  for (let s = 0; s < seq; s += 1) {
    for (let h = 0; h < heads; h += 1) {
      for (let d = 0; d < dim; d += 1) out[(h * seq + s) * dim + d] = x[(s * heads + h) * dim + d]!;
    }
  }
  return out;
}

/** The inverse of `splitHeads`. */
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
 * QK-Norm: RMSNorm over each head's own `headDim`, not over the whole row.
 *
 * `ops/rmsnorm` normalises the last axis of an `[N, D]` view, so feeding it
 * `[S*H, headDim]` gives exactly the model's `norm_q(query)` on a
 * `[B, S, H, headDim]` tensor. No grouped variant is needed — the grouping is
 * in how the rows are counted.
 */
function qkNorm(x: Float32Array, rows: number, headDim: number, weight: Float32Array, eps: number): Float32Array {
  return rmsnorm({ input: x, weight, N: rows, D: headDim, eps });
}

export function zimageBlock(cfg: BlockConfig, w: BlockWeights, x: Float32Array, adalnInput: Float32Array, positions: Int32Array): Float32Array {
  const { dim, nHeads, headDim, seq, ffnHidden, normEps, ropeAxesDims, ropeTheta } = cfg;

  // adaLN: one Linear with bias, then four chunks along the last axis.
  const mod = linear(adalnInput, w.adaLN_modulation_0_weight, 1, adalnInput.length, 4 * dim);
  for (let i = 0; i < mod.length; i += 1) mod[i] = mod[i]! + w.adaLN_modulation_0_bias[i % (4 * dim)]!;

  const chunk = (k: number): Float32Array => mod.subarray(k * dim, (k + 1) * dim);
  const scaleMsa = chunk(0).slice();
  const gateMsa = activation({ input: chunk(1).slice(), kind: ACTIVATION.tanh });
  const scaleMlp = chunk(2).slice();
  const gateMlp = activation({ input: chunk(3).slice(), kind: ACTIVATION.tanh });
  // `1.0 + scale`, which is what makes an all-zero modulation the identity.
  for (let i = 0; i < dim; i += 1) {
    scaleMsa[i] = 1 + scaleMsa[i]!;
    scaleMlp[i] = 1 + scaleMlp[i]!;
  }

  // --- attention half ---
  const normed1 = rmsnorm({ input: x, weight: w.attention_norm1_weight, N: seq, D: dim, eps: normEps });
  const scaled1 = elementwiseRows({ a: normed1, b: scaleMsa, S: seq, D: dim, kind: ELEMENTWISE.multiply });

  let q = linear(scaled1, w.attention_to_q_weight, seq, dim, nHeads * headDim);
  let k = linear(scaled1, w.attention_to_k_weight, seq, dim, nHeads * headDim);
  const v = linear(scaled1, w.attention_to_v_weight, seq, dim, nHeads * headDim);

  q = qkNorm(q, seq * nHeads, headDim, w.attention_norm_q_weight, normEps);
  k = qkNorm(k, seq * nHeads, headDim, w.attention_norm_k_weight, normEps);

  // RoPE over three axes, applied to Q and K in token-major layout — one head's
  // 128 channels are 32+48+48, and each axis brings its own cos/sin table.
  q = ropeAxes({ input: q, N: seq, numHeads: nHeads, axisDims: ropeAxesDims, positions, thetaBase: ropeTheta });
  k = ropeAxes({ input: k, N: seq, numHeads: nHeads, axisDims: ropeAxesDims, positions, thetaBase: ropeTheta });

  const attnOut = attention({
    q: splitHeads(q, seq, nHeads, headDim),
    k: splitHeads(k, seq, nHeads, headDim),
    v: splitHeads(v, seq, nHeads, headDim),
    B: 1, H: nHeads, L: seq, S: seq, D: headDim, Dv: headDim,
    causal: false,
  });
  const projected = linear(mergeHeads(attnOut.output, seq, nHeads, headDim), w.attention_to_out_0_weight, seq, nHeads * headDim, dim);

  const normed2 = rmsnorm({ input: projected, weight: w.attention_norm2_weight, N: seq, D: dim, eps: normEps });
  const gated1 = elementwiseRows({ a: normed2, b: gateMsa, S: seq, D: dim, kind: ELEMENTWISE.multiply });
  let h = elementwise({ a: x, b: gated1, kind: ELEMENTWISE.add });

  // --- feed-forward half: w2(silu(w1(x)) * w3(x)) ---
  const normed3 = rmsnorm({ input: h, weight: w.ffn_norm1_weight, N: seq, D: dim, eps: normEps });
  const scaled2 = elementwiseRows({ a: normed3, b: scaleMlp, S: seq, D: dim, kind: ELEMENTWISE.multiply });

  const gate = activation({ input: linear(scaled2, w.feed_forward_w1_weight, seq, dim, ffnHidden), kind: ACTIVATION.silu });
  const up = linear(scaled2, w.feed_forward_w3_weight, seq, dim, ffnHidden);
  const ffn = linear(elementwise({ a: gate, b: up, kind: ELEMENTWISE.multiply }), w.feed_forward_w2_weight, seq, ffnHidden, dim);

  const normed4 = rmsnorm({ input: ffn, weight: w.ffn_norm2_weight, N: seq, D: dim, eps: normEps });
  const gated2 = elementwiseRows({ a: normed4, b: gateMlp, S: seq, D: dim, kind: ELEMENTWISE.multiply });
  h = elementwise({ a: h, b: gated2, kind: ELEMENTWISE.add });

  return h;
}
