/**
 * Qwen3-VL's text decoder — the conditioner MiniMax-H3's R2V reads.
 *
 * Issue #212. `hidden_states[50]` of this stack, of 64 layers, is what
 * `ref2va` conditions on. Unlike `examples/h3-dit-web`'s prompt embeddings it
 * **cannot be precomputed**: the reference is the input.
 *
 * **No new kernel.** `matmul`, `rmsnorm`, `gqa`, `activation`, `elementwise`
 * and `rope`'s axes entry cover it — which was checked against the checkpoint's
 * own parameter shapes before any of this was written, not asserted afterwards.
 *
 * ## The rotation, which `ropeAxes` can express and only just
 *
 * `mrope_section` says how many of the `headDim / 2` frequencies each of the
 * three axes owns — but they are **not contiguous**. `apply_interleaved_mrope`
 * starts from the time axis everywhere and then overwrites channel `c` from
 * axis `c % 3` while `c < 3 * section[1]`. So the axis of a channel is
 * `c % 3` in the interleaved range and time after it, and upstream calls the
 * reason "preserving frequency continuity": **every channel keeps the frequency
 * its global index gives it**, whichever axis it reads its position from.
 *
 * `ropeAxes` gives each axis its own frequency sweep, normalised by that axis's
 * channel count — which is not this. Two things make it fit anyway:
 *
 * - **Sixty-four axes of two channels each.** At `dim = 2` the only pair index
 *   is 0, so `theta ** (-2 * 0 / 2)` is exactly 1 and the angle is the position
 *   verbatim. The frequency is then whatever the caller folds into the
 *   position — which is #206's fractional positions doing the work they were
 *   added for.
 * - **A channel permutation**, because Qwen rotates `c` against `c + headDim/2`
 *   (`rotate_half`) and `ropeAxes` rotates adjacent pairs. The same mismatch
 *   `h3RopePermutation` fixes for H3, at a different geometry.
 *
 * Both belong in the weights and the positions rather than in a kernel, which
 * is why `mropePermutation` and `mropePositions` are exported: a converter
 * applies the first once, and a caller builds the second per request.
 */
import { ACTIVATION, activation } from "../../../ops/activation/index.js";
import { ELEMENTWISE, elementwise } from "../../../ops/elementwise/index.js";
import { groupedAttention } from "../../../ops/gqa/index.js";
import { matmul } from "../../../ops/matmul/index.js";
import { rmsnorm } from "../../../ops/rmsnorm/index.js";
import { ropeAxes } from "../../../ops/rope/index.js";

export interface TextEncoderConfig {
  hiddenSize: number;
  intermediateSize: number;
  numHiddenLayers: number;
  numAttentionHeads: number;
  numKeyValueHeads: number;
  headDim: number;
  rmsNormEps: number;
  ropeTheta: number;
  /** How many of the `headDim / 2` frequencies each axis owns. Sums to `headDim / 2`. */
  mropeSection: [number, number, number];
}

export interface TextLayerWeights {
  inputLayernorm: Float32Array;
  qProj: Float32Array;
  kProj: Float32Array;
  vProj: Float32Array;
  oProj: Float32Array;
  qNorm: Float32Array;
  kNorm: Float32Array;
  postAttentionLayernorm: Float32Array;
  gateProj: Float32Array;
  upProj: Float32Array;
  downProj: Float32Array;
}

export interface TextEncoderWeights {
  layers: TextLayerWeights[];
  /** `norm.weight`, the stack's final RMSNorm. See `textEncoderHiddenStates`. */
  finalNorm: Float32Array;
}

/** `[seq, 3]`, token-major: the `(t, h, w)` coordinate of every row. */
export type PositionGrid = { t: number[]; h: number[]; w: number[] };

/**
 * Which axis each of the `headDim / 2` channels reads its position from.
 *
 * `0` time, `1` height, `2` width. Channel `c` is `c % 3` while
 * `c < 3 * section[1]`, and time after that — which is what
 * `apply_interleaved_mrope` leaves behind when it overwrites two strided slices
 * of an all-time array.
 *
 * **`section[1]` and `section[2]` are both used, and they are assumed equal**
 * upstream: the loop writes `slice(1, 3 * section[1], 3)` for height and
 * `slice(2, 3 * section[2], 3)` for width, so unequal sections would leave a
 * ragged boundary. Rejected here rather than silently interpreted.
 */
export function mropeAxisOfChannel(headDim: number, section: [number, number, number]): number[] {
  const half = headDim / 2;
  if (section[0] + section[1] + section[2] !== half) {
    throw new Error(`mropeAxisOfChannel: sections sum to ${section[0] + section[1] + section[2]}, not ${half}`);
  }
  if (section[1] !== section[2]) {
    throw new Error(`mropeAxisOfChannel: the interleave assumes equal height and width sections, got ${section[1]} and ${section[2]}`);
  }
  const axes = new Array<number>(half).fill(0);
  for (let axis = 1; axis <= 2; axis += 1) {
    for (let c = axis; c < 3 * section[axis]!; c += 3) axes[c] = axis;
  }
  return axes;
}

/**
 * The channel order `ropeAxes` reads, from the order `rotate_half` writes.
 *
 * Qwen rotates channel `c` against `c + headDim / 2`; `ropeAxes` rotates
 * adjacent pairs. So pair `c` lands at destination `2c`, `2c + 1`. Applied to
 * the **rows** of the Q and K projections once at conversion — and to
 * `q_norm` / `k_norm`'s per-channel weights, which index the channels the
 * projection produced. Permuting the projection and forgetting the norm cost 8%
 * in #208.
 */
export function mropePermutation(headDim: number): number[] {
  const half = headDim / 2;
  const order = new Array<number>(headDim);
  for (let c = 0; c < half; c += 1) {
    order[2 * c] = c;
    order[2 * c + 1] = c + half;
  }
  return order;
}

/**
 * `[seq, headDim / 2]` positions, one per two-channel axis, frequency folded in.
 *
 * `ropeAxes` at `dim = 2` computes `theta * pos` with `theta = 1`, so the
 * frequency has to arrive in the position. Qwen's is
 * `ropeTheta ** (-2c / headDim)` — indexed by the **global** channel, not by a
 * per-axis index, which is the whole of "preserving frequency continuity".
 */
export function mropePositions(
  grid: PositionGrid,
  headDim: number,
  section: [number, number, number],
  ropeTheta: number,
): Float32Array {
  const half = headDim / 2;
  const axisOf = mropeAxisOfChannel(headDim, section);
  const seq = grid.t.length;
  if (grid.h.length !== seq || grid.w.length !== seq) {
    throw new Error("mropePositions: the three rows of the grid are not the same length");
  }
  const rows = [grid.t, grid.h, grid.w];
  const out = new Float32Array(seq * half);
  for (let token = 0; token < seq; token += 1) {
    for (let c = 0; c < half; c += 1) {
      const invFreq = Math.pow(ropeTheta, (-2 * c) / headDim);
      out[token * half + c] = rows[axisOf[c]!]![token]! * invFreq;
    }
  }
  return out;
}

/** `x @ Wᵀ`, the layout `nn.Linear` stores. No bias anywhere in this stack. */
function linear(x: Float32Array, weight: Float32Array, rows: number, inDim: number, outDim: number): Float32Array {
  const wT = new Float32Array(inDim * outDim);
  for (let o = 0; o < outDim; o += 1) {
    for (let i = 0; i < inDim; i += 1) wT[i * outDim + o] = weight[o * inDim + i]!;
  }
  return matmul({ a: x, b: wT, M: rows, N: outDim, K: inDim });
}

/** Reorders each head's channels of a `[heads * headDim, hidden]` projection. */
export function permuteProjection(
  weight: Float32Array,
  heads: number,
  headDim: number,
  hidden: number,
  order: number[],
): Float32Array {
  const out = weight.slice();
  for (let head = 0; head < heads; head += 1) {
    for (let c = 0; c < headDim; c += 1) {
      const from = head * headDim + order[c]!;
      out.set(weight.subarray(from * hidden, (from + 1) * hidden), (head * headDim + c) * hidden);
    }
  }
  return out;
}

/** Reorders a `[headDim]` per-channel vector to match a permuted projection. */
export function permuteChannels(weight: Float32Array, order: number[]): Float32Array {
  return Float32Array.from(order, (from) => weight[from]!);
}

/** `[seq, heads, dim]` to `ops/gqa`'s `[heads, seq, dim]`. */
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
 * One decoder layer.
 *
 * The weights arrive **already permuted** for the rotation, as a converter
 * would leave them, so nothing here reorders channels per token.
 */
export function textLayer(
  cfg: TextEncoderConfig,
  w: TextLayerWeights,
  x: Float32Array,
  seq: number,
  positions: Float32Array,
): Float32Array {
  const { hiddenSize, numAttentionHeads, numKeyValueHeads, headDim, rmsNormEps } = cfg;
  const qWidth = numAttentionHeads * headDim;
  const kvWidth = numKeyValueHeads * headDim;
  // Sixty-four two-channel axes — see this file's own note on why.
  const axisDims = new Array<number>(headDim / 2).fill(2);

  const normed = rmsnorm({ input: x, weight: w.inputLayernorm, N: seq, D: hiddenSize, eps: rmsNormEps });
  let q = linear(normed, w.qProj, seq, hiddenSize, qWidth);
  let k = linear(normed, w.kProj, seq, hiddenSize, kvWidth);
  const v = linear(normed, w.vProj, seq, hiddenSize, kvWidth);

  // **Before the rotation**, and with weights — Qwen3's addition to the llama
  // block, and the same arrangement `examples/h3-dit`'s block has.
  q = rmsnorm({ input: q, weight: w.qNorm, N: seq * numAttentionHeads, D: headDim, eps: rmsNormEps });
  k = rmsnorm({ input: k, weight: w.kNorm, N: seq * numKeyValueHeads, D: headDim, eps: rmsNormEps });

  q = ropeAxes({ input: q, N: seq, numHeads: numAttentionHeads, axisDims, positions, thetaBase: cfg.ropeTheta });
  k = ropeAxes({ input: k, N: seq, numHeads: numKeyValueHeads, axisDims, positions, thetaBase: cfg.ropeTheta });

  const attended = groupedAttention({
    q: splitHeads(q, seq, numAttentionHeads, headDim),
    k: splitHeads(k, seq, numKeyValueHeads, headDim),
    v: splitHeads(v, seq, numKeyValueHeads, headDim),
    B: 1, H: numAttentionHeads, kvHeads: numKeyValueHeads,
    L: seq, S: seq, D: headDim, Dv: headDim,
    causal: true,
  });
  const merged = new Float32Array(seq * qWidth);
  for (let h = 0; h < numAttentionHeads; h += 1) {
    for (let s = 0; s < seq; s += 1) {
      for (let d = 0; d < headDim; d += 1) {
        merged[(s * numAttentionHeads + h) * headDim + d] = attended.output[(h * seq + s) * headDim + d]!;
      }
    }
  }
  let hidden = elementwise({
    a: x, b: linear(merged, w.oProj, seq, qWidth, hiddenSize), kind: ELEMENTWISE.add,
  });

  const normed2 = rmsnorm({
    input: hidden, weight: w.postAttentionLayernorm, N: seq, D: hiddenSize, eps: rmsNormEps,
  });
  // `down(silu(gate(x)) * up(x))` — **`gate` is the one that is activated**,
  // which is the opposite of `examples/h3-dit`'s SwiGLU, where the first half
  // is the value. Two conventions, two models, one repository.
  const gate = activation({
    input: linear(normed2, w.gateProj, seq, hiddenSize, cfg.intermediateSize), kind: ACTIVATION.silu,
  });
  const up = linear(normed2, w.upProj, seq, hiddenSize, cfg.intermediateSize);
  const ff = linear(
    elementwise({ a: gate, b: up, kind: ELEMENTWISE.multiply }),
    w.downProj, seq, cfg.intermediateSize, hiddenSize,
  );
  hidden = elementwise({ a: hidden, b: ff, kind: ELEMENTWISE.add });
  return hidden;
}

/**
 * Every hidden state, `numHiddenLayers + 1` of them, as `transformers` reports them.
 *
 * **The last one is not like the others.** Entries `0 … n-1` are the *inputs*
 * of each layer — entry `i` is the output of `i` layers — but the last is
 * `last_hidden_state`, which has **the final norm applied**. That is what
 * `output_hidden_states=True` returns, and a port that appends its raw last
 * layer output instead disagrees with the golden by 0.53 while matching every
 * earlier entry, which is exactly what happened here.
 *
 * It does not change what MiniMax-H3 reads. `hidden_states[50]` of a 64-layer
 * stack is an ordinary layer input, nowhere near the end — which is *why* the
 * discrepancy could sit unnoticed in a port that only ever indexed 50.
 */
export function textEncoderHiddenStates(
  cfg: TextEncoderConfig,
  w: TextEncoderWeights,
  embeds: Float32Array,
  seq: number,
  positions: Float32Array,
): Float32Array[] {
  if (w.layers.length !== cfg.numHiddenLayers) {
    throw new Error(`textEncoderHiddenStates: ${w.layers.length} layers of weights for ${cfg.numHiddenLayers}`);
  }
  const states: Float32Array[] = [embeds.slice()];
  let x = embeds;
  for (const [index, layer] of w.layers.entries()) {
    x = textLayer(cfg, layer, x, seq, positions);
    states.push(
      index === w.layers.length - 1
        ? rmsnorm({ input: x, weight: w.finalNorm, N: seq, D: cfg.hiddenSize, eps: cfg.rmsNormEps })
        : x.slice(),
    );
  }
  return states;
}
