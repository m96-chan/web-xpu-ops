/**
 * One transformer block of MiniMax-H3's visual VAE decoder.
 *
 * Issue #200. The decoder is **36 identical blocks over 2048 channels** — 9.69
 * GB of the checkpoint's 10.42 — so a port is right or wrong here and the rest
 * is a loop. Built in the order `examples/zimage` and `examples/anima` were:
 * the block against the model's own output first, the forward after.
 *
 * ## The shape, from `base_module.py`'s `TransformerBlock` and `attention.py`
 *
 *     h = x + attn(rmsnorm(x, norm1)) * scale1
 *     h = h + ffn(rmsnorm(h, norm2)) * scale2
 *
 * A pre-norm block with **LayerScale**: `scale1` and `scale2` are per-channel
 * parameters that multiply each branch before it is added back. They are
 * initialised to zero, so an untrained block is the identity — and a port that
 * dropped them would be a block that trains from scratch every forward.
 *
 * Three things it does that Z-Image's block does not:
 *
 * - **Q, K and V come from one `to_qkv` and are interleaved per head.** The
 *   model does `qkv.view(B, L, -1, 3 * dim_head).chunk(3, dim=-1)`, so head `h`
 *   owns the contiguous 192 channels `[q(64), k(64), v(64)]` — not three
 *   separate `[heads, 64]` blocks. Reading it the other way gives a well-formed
 *   tensor of the same shape whose heads are mixed.
 * - **QK-norm has no weights.** `qk_norm_affine: false`, so it is a plain
 *   `x * rsqrt(mean(x²) + eps)` per head. Passed as a weight of ones, which
 *   multiplies without rounding in f32 and keeps one code path.
 * - **RoPE turns 48 of 64 channels** (`rope_dim_ratio: 0.75`) about
 *   *fractional* positions. `ops/rope`'s `axes` entry covers both — the fourth
 *   axis is pinned at 0, which is the identity, and positions have been `f32`
 *   since #200 — given the channel permutation `H3_ROPE_PERMUTATION` describes.
 *   **That permutation belongs in the weights**, applied once at conversion, and
 *   `blockWeightsFromFlat` is where it happens.
 */
import { ACTIVATION, activation } from "../../../ops/activation/index.js";
import { attention } from "../../../ops/attention/index.js";
import { ELEMENTWISE, elementwise, elementwiseRows } from "../../../ops/elementwise/index.js";
import { matmul } from "../../../ops/matmul/index.js";
import { rmsnorm } from "../../../ops/rmsnorm/index.js";
import { ropeAxes } from "../../../ops/rope/index.js";
import { H3_ROPE_PERMUTATION } from "../../../ops/rope/h3-cases.js";

export interface H3BlockConfig {
  /** Residual width — 2048. */
  dim: number;
  nHeads: number;
  headDim: number;
  /** Tokens, patches **and** the five suffix tokens. */
  seq: number;
  /** `dim * 4`; the gated `w1` is twice this wide. */
  ffnHidden: number;
  normEps: number;
  /** Channels each rope axis owns. `[16, 16, 16, 16]`, the fourth idle. */
  ropeAxesDims: number[];
  ropeTheta: number;
}

export interface H3BlockWeights {
  norm1Weight: Float32Array;
  /** `[3 * dim, dim]`, already permuted for RoPE — see `blockWeightsFromFlat`. */
  toQkvWeight: Float32Array;
  toQkvBias: Float32Array;
  toOutWeight: Float32Array;
  toOutBias: Float32Array;
  scale1: Float32Array;
  norm2Weight: Float32Array;
  w1Weight: Float32Array;
  w1Bias: Float32Array;
  w2Weight: Float32Array;
  w2Bias: Float32Array;
  scale2: Float32Array;
}

/** `x @ Wᵀ`, the layout `nn.Linear` stores. */
function linear(x: Float32Array, weight: Float32Array, rows: number, inDim: number, outDim: number): Float32Array {
  const wT = new Float32Array(inDim * outDim);
  for (let o = 0; o < outDim; o += 1) {
    for (let i = 0; i < inDim; i += 1) wT[i * outDim + o] = weight[o * inDim + i]!;
  }
  return matmul({ a: x, b: wT, M: rows, N: outDim, K: inDim });
}

function addBias(x: Float32Array, bias: Float32Array, rows: number, width: number): Float32Array {
  for (let r = 0; r < rows; r += 1) {
    for (let c = 0; c < width; c += 1) x[r * width + c] = x[r * width + c]! + bias[c]!;
  }
  return x;
}

/** `[seq, heads, dim]` to `ops/attention`'s `[heads, seq, dim]`. */
function splitHeads(x: Float32Array, seq: number, heads: number, dim: number): Float32Array {
  const out = new Float32Array(seq * heads * dim);
  for (let s = 0; s < seq; s += 1) {
    for (let h = 0; h < heads; h += 1) {
      for (let d = 0; d < dim; d += 1) out[(h * seq + s) * dim + d] = x[(s * heads + h) * dim + d]!;
    }
  }
  return out;
}

/**
 * Permutes each head's 64 rows of a `[3 * dim, dim]` `to_qkv` weight.
 *
 * H3 rotates channel `c` with `c + 24` over the first 48; `ropeAxes` rotates
 * adjacent pairs inside per-axis blocks. Permuting the weight rows makes the
 * two agree, and it costs nothing at run time — `permuteForRope` does the same
 * for Anima.
 *
 * **Only Q and K.** V is never rotated, and permuting it would reorder channels
 * the output projection then reads in the original order. The permutation is
 * invisible downstream precisely because `q · k` is unchanged when both sides
 * are permuted the same way.
 */
export function permuteQkvForRope(
  weight: Float32Array,
  bias: Float32Array,
  nHeads: number,
  headDim: number,
  dim: number,
): { weight: Float32Array; bias: Float32Array } {
  const outWeight = weight.slice();
  const outBias = bias.slice();
  for (let head = 0; head < nHeads; head += 1) {
    for (const part of [0, 1]) {
      // Head `h` owns `[q(64), k(64), v(64)]` contiguously; part 0 is q, 1 is k.
      const base = head * 3 * headDim + part * headDim;
      for (let c = 0; c < headDim; c += 1) {
        const from = base + H3_ROPE_PERMUTATION[c]!;
        const to = base + c;
        outWeight.set(weight.subarray(from * dim, (from + 1) * dim), to * dim);
        outBias[to] = bias[from]!;
      }
    }
  }
  return { weight: outWeight, bias: outBias };
}

/** One block, from `[seq, dim]` to `[seq, dim]`. */
export function h3VideoBlock(
  cfg: H3BlockConfig,
  w: H3BlockWeights,
  x: Float32Array,
  positions: Float32Array,
): Float32Array {
  const { dim, nHeads, headDim, seq, ffnHidden, normEps, ropeAxesDims, ropeTheta } = cfg;
  const width = nHeads * headDim;

  // --- attention half ---
  const normed1 = rmsnorm({ input: x, weight: w.norm1Weight, N: seq, D: dim, eps: normEps });
  const qkv = addBias(linear(normed1, w.toQkvWeight, seq, dim, 3 * width), w.toQkvBias, seq, 3 * width);

  // `[seq, heads, 3 * headDim]` -> three `[seq, heads, headDim]`. Head-major
  // inside a token, which is what `view(B, L, -1, 3 * dim_head)` means.
  const take = (part: number): Float32Array => {
    const out = new Float32Array(seq * width);
    for (let s = 0; s < seq; s += 1) {
      for (let h = 0; h < nHeads; h += 1) {
        const from = (s * nHeads + h) * 3 * headDim + part * headDim;
        out.set(qkv.subarray(from, from + headDim), (s * nHeads + h) * headDim);
      }
    }
    return out;
  };
  let q = take(0);
  let k = take(1);
  const v = take(2);

  // `qk_norm_affine: false` — a weight of ones rather than a second code path.
  const ones = new Float32Array(headDim).fill(1);
  q = rmsnorm({ input: q, weight: ones, N: seq * nHeads, D: headDim, eps: normEps });
  k = rmsnorm({ input: k, weight: ones, N: seq * nHeads, D: headDim, eps: normEps });

  q = ropeAxes({ input: q, N: seq, numHeads: nHeads, axisDims: ropeAxesDims, positions, thetaBase: ropeTheta });
  k = ropeAxes({ input: k, N: seq, numHeads: nHeads, axisDims: ropeAxesDims, positions, thetaBase: ropeTheta });

  // `causal_decoder: false` in the checkpoint's own config, so there is no mask
  // — every token sees every token, suffix tokens included.
  const attended = attention({
    q: splitHeads(q, seq, nHeads, headDim),
    k: splitHeads(k, seq, nHeads, headDim),
    v: splitHeads(v, seq, nHeads, headDim),
    B: 1, H: nHeads, L: seq, S: seq, D: headDim, Dv: headDim,
    causal: false,
  });
  const merged = new Float32Array(seq * width);
  for (let h = 0; h < nHeads; h += 1) {
    for (let s = 0; s < seq; s += 1) {
      for (let d = 0; d < headDim; d += 1) {
        merged[(s * nHeads + h) * headDim + d] = attended.output[(h * seq + s) * headDim + d]!;
      }
    }
  }

  const projected = addBias(linear(merged, w.toOutWeight, seq, width, dim), w.toOutBias, seq, dim);
  const gated1 = elementwiseRows({ a: projected, b: w.scale1, S: seq, D: dim, kind: ELEMENTWISE.multiply });
  let hidden = elementwise({ a: x, b: gated1, kind: ELEMENTWISE.add });

  // --- feed-forward half: SwiGLU ---
  const normed2 = rmsnorm({ input: hidden, weight: w.norm2Weight, N: seq, D: dim, eps: normEps });
  const wide = addBias(linear(normed2, w.w1Weight, seq, dim, 2 * ffnHidden), w.w1Bias, seq, 2 * ffnHidden);

  // `chunk(2, dim=-1)`: the **gate first**, then the value. Swapping them is a
  // different function whose output has the same shape and a plausible range.
  const gate = new Float32Array(seq * ffnHidden);
  const up = new Float32Array(seq * ffnHidden);
  for (let s = 0; s < seq; s += 1) {
    gate.set(wide.subarray(s * 2 * ffnHidden, s * 2 * ffnHidden + ffnHidden), s * ffnHidden);
    up.set(wide.subarray(s * 2 * ffnHidden + ffnHidden, (s + 1) * 2 * ffnHidden), s * ffnHidden);
  }
  const activated = elementwise({
    a: activation({ input: gate, kind: ACTIVATION.silu }),
    b: up,
    kind: ELEMENTWISE.multiply,
  });

  const ff = addBias(linear(activated, w.w2Weight, seq, ffnHidden, dim), w.w2Bias, seq, dim);
  const gated2 = elementwiseRows({ a: ff, b: w.scale2, S: seq, D: dim, kind: ELEMENTWISE.multiply });
  hidden = elementwise({ a: hidden, b: gated2, kind: ELEMENTWISE.add });
  return hidden;
}
