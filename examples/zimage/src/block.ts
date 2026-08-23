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

/**
 * Every parameter of one block, in the model's own naming.
 *
 * The adaLN pair is optional because `context_refiner`'s blocks are built with
 * `modulation=False` and genuinely do not have those tensors — 13 per layer in
 * the checkpoint against the main stack's 15. Declaring them optional is how
 * the type says so; a caller that forgets `adalnInput` for a modulated block
 * gets an error naming it rather than a silent identity.
 */
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
  adaLN_modulation_0_weight?: Float32Array;
  adaLN_modulation_0_bias?: Float32Array;
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

/**
 * One block.
 *
 * `adalnInput` is `null` for a `modulation=False` block — `context_refiner`'s
 * two layers. The model writes that as a separate branch without the scales or
 * the gates; here it is the same arithmetic with all four set to exactly 1.0,
 * which multiplies without rounding in f32 and keeps one path instead of two.
 * A second branch would be a second place for the residuals to be wrong.
 */
export function zimageBlock(
  cfg: BlockConfig,
  w: BlockWeights,
  x: Float32Array,
  adalnInput: Float32Array | null,
  positions: Int32Array,
  /**
   * How many leading tokens are real. Padding is the rest, and there is no
   * `validSeq` for a block whose tokens are all real.
   *
   * **This trims, it does not mask**, and the difference is the whole reason
   * the parameter is a count rather than a mask. `modules/attention.py:130`
   * takes the single-batch path: when a mask is present and every sequence in
   * the batch is the same length, it slices `q`, `k` and `v` to that length,
   * **drops the mask**, runs attention, and pads the result back with **zeros**
   * (`:295`). So a padded row's attention output is not "the attention it would
   * have had, minus the padded keys" — it is exactly zero, and the residual
   * `x + norm2(attn_out)` leaves that row untouched by the attention half.
   *
   * An additive `-Infinity` key mask was tried first and measured 6.6e-1
   * relative RMS at `afterContextRefiner`: right about the keys, wrong about
   * the query rows. Both are needed to notice, which is why it is written down.
   *
   * The trim assumes the real tokens are a **prefix**. They are — padding sits
   * at the end of the caption, and the unified stack puts all image tokens
   * first — and it is checked rather than assumed, because the model would
   * silently attend to the wrong tokens too.
   */
  validSeq?: number,
): Float32Array {
  const { dim, nHeads, headDim, seq, ffnHidden, normEps, ropeAxesDims, ropeTheta } = cfg;

  const ones = (): Float32Array => new Float32Array(dim).fill(1);
  let scaleMsa: Float32Array;
  let gateMsa: Float32Array;
  let scaleMlp: Float32Array;
  let gateMlp: Float32Array;

  if (adalnInput === null) {
    scaleMsa = ones();
    gateMsa = ones();
    scaleMlp = ones();
    gateMlp = ones();
  } else {
    if (!w.adaLN_modulation_0_weight || !w.adaLN_modulation_0_bias) {
      throw new Error("zimageBlock: adalnInput was given but the block carries no adaLN_modulation weights.");
    }
    // adaLN: one Linear with bias, then four chunks along the last axis.
    const mod = linear(adalnInput, w.adaLN_modulation_0_weight, 1, adalnInput.length, 4 * dim);
    for (let i = 0; i < mod.length; i += 1) mod[i] = mod[i]! + w.adaLN_modulation_0_bias[i % (4 * dim)]!;

    const chunk = (k: number): Float32Array => mod.subarray(k * dim, (k + 1) * dim);
    scaleMsa = chunk(0).slice();
    gateMsa = activation({ input: chunk(1).slice(), kind: ACTIVATION.tanh });
    scaleMlp = chunk(2).slice();
    gateMlp = activation({ input: chunk(3).slice(), kind: ACTIVATION.tanh });
    // `1.0 + scale`, which is what makes an all-zero modulation the identity.
    for (let i = 0; i < dim; i += 1) {
      scaleMsa[i] = 1 + scaleMsa[i]!;
      scaleMlp[i] = 1 + scaleMlp[i]!;
    }
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

  // The trim happens *after* QK-Norm and RoPE, matching the model: those run on
  // the full sequence and only `attention()` sees the shortened one.
  const live = validSeq ?? seq;
  if (live < 0 || live > seq) throw new Error(`zimageBlock: validSeq ${live} is outside [0, ${seq}].`);
  const width = nHeads * headDim;

  const attnOut = attention({
    q: splitHeads(q.subarray(0, live * width), live, nHeads, headDim),
    k: splitHeads(k.subarray(0, live * width), live, nHeads, headDim),
    v: splitHeads(v.subarray(0, live * width), live, nHeads, headDim),
    B: 1, H: nHeads, L: live, S: live, D: headDim, Dv: headDim,
    causal: false,
  });
  // Padded back with zeros, which is what makes those rows skip the attention
  // residual entirely rather than receive a masked version of it.
  const merged = new Float32Array(seq * width);
  merged.set(mergeHeads(attnOut.output, live, nHeads, headDim), 0);
  const projected = linear(merged, w.attention_to_out_0_weight, seq, width, dim);

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
