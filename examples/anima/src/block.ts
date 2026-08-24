/**
 * One Anima transformer block, composed from this repository's ops.
 *
 * Issue #170's first stage, and the same question `examples/zimage`'s
 * `block.ts` asked: does the op set reach this model, or only its own
 * references? The answer for Z-Image was yes with no new kernels. Anima's
 * *DiT* is the same answer; its **text encoder** is not, which is #173.
 *
 * The structure is ComfyUI's `Block` (`comfy/ldm/cosmos/predict2.py:480-590`),
 * which Anima subclasses without changing:
 *
 *     shift_s, scale_s, gate_s = (adaLN_self(emb) + lora).chunk(3)
 *     shift_c, scale_c, gate_c = (adaLN_cross(emb) + lora).chunk(3)
 *     shift_m, scale_m, gate_m = (adaLN_mlp(emb) + lora).chunk(3)
 *     x += gate_s * self_attn( norm(x) * (1 + scale_s) + shift_s )
 *     x += gate_c * cross_attn( norm(x) * (1 + scale_c) + shift_c, context )
 *     x += gate_m * mlp( norm(x) * (1 + scale_m) + shift_m )
 *
 * **Five things differ from Z-Image**, each read from the source rather than
 * assumed, because each produces a plausible wrong answer:
 *
 *  - **LayerNorm, not RMSNorm** — `elementwise_affine=False`, `eps=1e-6`
 *    (`:438`). Three of them, one per sub-block. Z-Image is RMSNorm with a
 *    learned weight throughout.
 *  - **Three adaLN modulations**, not one, and each chunks into **three**
 *    parts rather than Z-Image's four.
 *  - **`shift, scale, gate`** in that order, and the arithmetic is
 *    `norm(x) * (1 + scale) + shift`. Z-Image has no shift at all.
 *  - **No `tanh` on the gate.** Z-Image applies one to both of its gates.
 *  - **adaLN is a LoRA**: `SiLU -> Linear(dim, 256) -> Linear(256, 3*dim)`
 *    (`:451-465`), plus an `adaln_lora` term added before the chunk. That is
 *    why the checkpoint stores `[6144, 256]` and not `[6144, 2048]`.
 *
 * Correctness is `fixtures/block.*`, produced by running ComfyUI's own block on
 * the shipped weights — not by reading this file and agreeing with it.
 */
import { permuteRopeChannels } from "../../../llm/weights.js";
import { ACTIVATION, activation } from "../../../ops/activation/index.js";
import { attention } from "../../../ops/attention/index.js";
import { ELEMENTWISE, elementwise, elementwiseRows } from "../../../ops/elementwise/index.js";
import { layernorm } from "../../../ops/layernorm/index.js";
import { matmul } from "../../../ops/matmul/index.js";
import { rmsnorm } from "../../../ops/rmsnorm/index.js";
import { ropeAxes } from "../../../ops/rope/index.js";

export interface AnimaBlockConfig {
  dim: number;
  numHeads: number;
  headDim: number;
  contextDim: number;
  mlpHidden: number;
  adalnLoraDim: number;
  /** Image tokens. */
  seq: number;
  /** Context tokens the cross-attention reads. */
  contextSeq: number;
  normEps: number;
  /**
   * The three RoPE axes, `t, h, w`.
   *
   * `VideoRopePosition3DEmb` splits `head_dim` as `dim_h = dim // 6 * 2`,
   * `dim_w = dim_h`, `dim_t = dim - 2 * dim_h` — for head_dim 128 that is
   * **44 / 42 / 42**, measured by building the embedder rather than by doing
   * the arithmetic by hand, which got it wrong first.
   */
  ropeAxisDims: number[];
  /**
   * `10000 * extrapolation_ratio ** (dim / (dim - 2))`.
   *
   * The ratio is **4.0** for h and w and **1.0** for t in this checkpoint
   * (`model_detection.py:869`), which is not a default anyone would guess and
   * changes every spatial frequency. Computed by the caller from the config
   * rather than hardcoded.
   */
  ropeTheta: number;
}

/** `head_dim` split into `t, h, w` the way `VideoRopePosition3DEmb` does. */
export function ropeAxisDims(headDim: number): number[] {
  const spatial = Math.floor(headDim / 6) * 2;
  return [headDim - 2 * spatial, spatial, spatial];
}

/**
 * Puts a Q or K projection into `ops/rope`'s channel order.
 *
 * The model pairs channels **half a head apart** — HF's convention, which the
 * fused kernel's name says outright (`rms_rope_split_half`) and which measuring
 * confirms: applying the model's 2x2 matrices under the two pairings differs on
 * 106 of 128 channels. `ops/rope` pairs adjacent channels, so the projection's
 * rows are relabelled once at load time, exactly as `llm/weights.ts` does for
 * Qwen3. The QK-Norm weight goes through the same permutation — it is
 * `[headDim]` and multiplies channel by channel inside a head, so leaving it
 * alone would scale each channel by another's factor.
 */
export function permuteForRope(weight: Float32Array, heads: number, headDim: number, inFeatures: number): Float32Array {
  return permuteRopeChannels(weight, heads, headDim, inFeatures);
}

/** One block's parameters, in the checkpoint's own naming (dots to underscores). */
export interface AnimaBlockWeights {
  self_attn_q_proj_weight: Float32Array;
  self_attn_k_proj_weight: Float32Array;
  self_attn_v_proj_weight: Float32Array;
  self_attn_output_proj_weight: Float32Array;
  self_attn_q_norm_weight: Float32Array;
  self_attn_k_norm_weight: Float32Array;
  cross_attn_q_proj_weight: Float32Array;
  cross_attn_k_proj_weight: Float32Array;
  cross_attn_v_proj_weight: Float32Array;
  cross_attn_output_proj_weight: Float32Array;
  cross_attn_q_norm_weight: Float32Array;
  cross_attn_k_norm_weight: Float32Array;
  mlp_layer1_weight: Float32Array;
  mlp_layer2_weight: Float32Array;
  adaln_modulation_self_attn_1_weight: Float32Array;
  adaln_modulation_self_attn_2_weight: Float32Array;
  adaln_modulation_cross_attn_1_weight: Float32Array;
  adaln_modulation_cross_attn_2_weight: Float32Array;
  adaln_modulation_mlp_1_weight: Float32Array;
  adaln_modulation_mlp_2_weight: Float32Array;
}

/**
 * `torch.nn.Linear` without bias: `y = x @ W^T`, `W` stored `[out, in]`.
 *
 * The transpose is CPU work here on purpose, as in `examples/zimage`: a real
 * run uploads weights once and transposes ahead of time, so paying it per call
 * would measure the wrong thing. Written out rather than hidden so the cost is
 * visible.
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
 * LayerNorm with `elementwise_affine=False`.
 *
 * `ops/layernorm` always applies a weight and a bias, so ones and zeros are the
 * identity for them. Cheaper than a second kernel, and the arithmetic is the
 * same one the model does.
 */
function normNoAffine(x: Float32Array, rows: number, dim: number, eps: number): Float32Array {
  return layernorm({
    input: x,
    weight: new Float32Array(dim).fill(1),
    bias: new Float32Array(dim),
    N: rows,
    D: dim,
    eps,
  });
}

/**
 * One adaLN LoRA: `SiLU(emb) -> Linear(dim, rank) -> Linear(rank, 3*dim)`,
 * plus the shared `lora` term, chunked `shift, scale, gate`.
 *
 * The `lora` argument is added to the **whole** `3*dim` output before the chunk
 * (`predict2.py:487-495`), so it is one vector spanning all three parts rather
 * than three separate ones.
 */
function modulation(
  emb: Float32Array,
  lora: Float32Array,
  first: Float32Array,
  second: Float32Array,
  cfg: AnimaBlockConfig,
): { shift: Float32Array; scale: Float32Array; gate: Float32Array } {
  const { dim, adalnLoraDim } = cfg;
  const activated = activation({ input: emb.slice(), kind: ACTIVATION.silu });
  const hidden = linear(activated, first, 1, dim, adalnLoraDim);
  const projected = linear(hidden, second, 1, adalnLoraDim, 3 * dim);
  for (let i = 0; i < projected.length; i += 1) projected[i] = projected[i]! + lora[i]!;
  return {
    shift: projected.slice(0, dim),
    scale: projected.slice(dim, 2 * dim),
    gate: projected.slice(2 * dim, 3 * dim),
  };
}

/** `norm(x) * (1 + scale) + shift`, the model's `_fn` (`predict2.py:520`). */
function modulate(
  normed: Float32Array,
  scale: Float32Array,
  shift: Float32Array,
  rows: number,
  dim: number,
): Float32Array {
  const scaled = new Float32Array(normed.length);
  for (let r = 0; r < rows; r += 1) {
    for (let c = 0; c < dim; c += 1) {
      const at = r * dim + c;
      scaled[at] = normed[at]! * (1 + scale[c]!) + shift[c]!;
    }
  }
  return scaled;
}

/**
 * Attention with QK-Norm, over `q` from `x` and `k`/`v` from `context`.
 *
 * Self-attention is the case where `context === x`, which is how the model
 * writes it too (`model.py:63`: `context = x if context is None else context`).
 * QK-Norm is RMSNorm over each head's own channels, so the rows are counted per
 * head rather than per token — the same shape `examples/zimage` uses.
 */
function attend(
  x: Float32Array,
  context: Float32Array,
  qWeight: Float32Array,
  kWeight: Float32Array,
  vWeight: Float32Array,
  outWeight: Float32Array,
  qNorm: Float32Array,
  kNorm: Float32Array,
  seq: number,
  contextSeq: number,
  contextDim: number,
  cfg: AnimaBlockConfig,
  /**
   * Positions for the three RoPE axes, `[seq, 3]` as `(t, h, w)`.
   *
   * `undefined` for cross-attention: `predict2.py:184` applies `rope_emb` only
   * when `is_selfattn`, so the context's keys carry no position at all.
   */
  positions?: Int32Array,
): Float32Array {
  const { dim, numHeads, headDim, normEps } = cfg;
  const inner = numHeads * headDim;

  let q = linear(x, qWeight, seq, dim, inner);
  let k = linear(context, kWeight, contextSeq, contextDim, inner);
  const v = linear(context, vWeight, contextSeq, contextDim, inner);

  q = rmsnorm({ input: q, weight: qNorm, N: seq * numHeads, D: headDim, eps: normEps });
  k = rmsnorm({ input: k, weight: kNorm, N: contextSeq * numHeads, D: headDim, eps: normEps });

  if (positions) {
    // Self-attention only. The weights were permuted at load time into
    // `ops/rope`'s adjacent-pair order — see `ROPE_AXES` on why the model's
    // 2x2 matrices are the same rotation in HF's pairing.
    q = ropeAxes({ input: q, N: seq, numHeads, axisDims: cfg.ropeAxisDims, positions, thetaBase: cfg.ropeTheta });
    k = ropeAxes({ input: k, N: seq, numHeads, axisDims: cfg.ropeAxisDims, positions, thetaBase: cfg.ropeTheta });
  }

  const attended = attention({
    q: splitHeads(q, seq, numHeads, headDim),
    k: splitHeads(k, contextSeq, numHeads, headDim),
    v: splitHeads(v, contextSeq, numHeads, headDim),
    B: 1,
    H: numHeads,
    L: seq,
    S: contextSeq,
    D: headDim,
    Dv: headDim,
    causal: false,
  });
  return linear(mergeHeads(attended.output, seq, numHeads, headDim), outWeight, seq, inner, dim);
}

/**
 * One block: `[seq, dim]` in, `[seq, dim]` out.
 *
 * `emb` is `[dim]` — the timestep conditioning — and `lora` is `[3 * dim]`,
 * the shared adaLN LoRA term. `context` is `[contextSeq, contextDim]` from the
 * adapter.
 */
export function animaBlock(
  cfg: AnimaBlockConfig,
  w: AnimaBlockWeights,
  x: Float32Array,
  emb: Float32Array,
  lora: Float32Array,
  context: Float32Array,
  /** `[seq, 3]` as `(t, h, w)`; self-attention only. */
  positions?: Int32Array,
): Float32Array {
  const { dim, mlpHidden, contextDim, seq, contextSeq, normEps } = cfg;

  const selfMod = modulation(emb, lora, w.adaln_modulation_self_attn_1_weight, w.adaln_modulation_self_attn_2_weight, cfg);
  const crossMod = modulation(emb, lora, w.adaln_modulation_cross_attn_1_weight, w.adaln_modulation_cross_attn_2_weight, cfg);
  const mlpMod = modulation(emb, lora, w.adaln_modulation_mlp_1_weight, w.adaln_modulation_mlp_2_weight, cfg);

  // --- self-attention ---
  let h = x;
  {
    const normed = normNoAffine(h, seq, dim, normEps);
    const modulated = modulate(normed, selfMod.scale, selfMod.shift, seq, dim);
    const result = attend(
      modulated, modulated,
      w.self_attn_q_proj_weight, w.self_attn_k_proj_weight, w.self_attn_v_proj_weight, w.self_attn_output_proj_weight,
      w.self_attn_q_norm_weight, w.self_attn_k_norm_weight,
      seq, seq, dim, cfg, positions,
    );
    const gated = elementwiseRows({ a: result, b: selfMod.gate, S: seq, D: dim, kind: ELEMENTWISE.multiply });
    h = elementwise({ a: h, b: gated, kind: ELEMENTWISE.add });
  }

  // --- cross-attention ---
  {
    const normed = normNoAffine(h, seq, dim, normEps);
    const modulated = modulate(normed, crossMod.scale, crossMod.shift, seq, dim);
    const result = attend(
      modulated, context,
      w.cross_attn_q_proj_weight, w.cross_attn_k_proj_weight, w.cross_attn_v_proj_weight, w.cross_attn_output_proj_weight,
      w.cross_attn_q_norm_weight, w.cross_attn_k_norm_weight,
      seq, contextSeq, contextDim, cfg,
    );
    const gated = elementwiseRows({ a: result, b: crossMod.gate, S: seq, D: dim, kind: ELEMENTWISE.multiply });
    h = elementwise({ a: h, b: gated, kind: ELEMENTWISE.add });
  }

  // --- MLP: Linear -> GELU -> Linear, no gate ---
  {
    const normed = normNoAffine(h, seq, dim, normEps);
    const modulated = modulate(normed, mlpMod.scale, mlpMod.shift, seq, dim);
    const hidden = linear(modulated, w.mlp_layer1_weight, seq, dim, mlpHidden);
    // `nn.GELU()` with no argument is the **exact** one (`erf`), not the tanh
    // approximation — `ops/activation` has both and the default matters.
    const activated = activation({ input: hidden, kind: ACTIVATION.gelu });
    const result = linear(activated, w.mlp_layer2_weight, seq, mlpHidden, dim);
    const gated = elementwiseRows({ a: result, b: mlpMod.gate, S: seq, D: dim, kind: ELEMENTWISE.multiply });
    h = elementwise({ a: h, b: gated, kind: ELEMENTWISE.add });
  }

  return h;
}
