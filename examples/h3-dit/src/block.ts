/**
 * One transformer block of MiniMax-H3's DiT — the generator half of the model.
 *
 * Issue #200. Fifty identical blocks over 5,376 channels, and this library
 * already decodes what they produce (`examples/h3-video`). A port is right or
 * wrong at the block and the rest is a loop, which is the order
 * `examples/zimage`, `examples/anima` and `examples/h3-video` were built in.
 *
 * ## The shape, from diffusers' `MiniMaxH3TransformerBlock`
 *
 *     shift, scale, gate (x2) = adaln_proj(silu(temb))
 *     h = x + gate_msa * attn(norm1(x) * (1 + scale_msa) + shift_msa)
 *     h = h + gate_mlp * ff  (norm2(h) * (1 + scale_mlp) + shift_mlp)
 *
 * Anima's block is the same arrangement, which is why nothing new is needed.
 * Three things differ, and each returns a well-formed tensor when got wrong:
 *
 * - **The modulation table is indexed per row.** `adaln_proj` projects one
 *   timestep embedding into `6 * hidden * 3` values — three *modalities*, text,
 *   video and audio — and every token picks its row with
 *   `timestep * 3 + tag`. A forward that is all one modality still has to pick
 *   the right one of the three.
 * - **SwiGLU here is `hidden * silu(gate)` with `hidden` the FIRST half.**
 *   `examples/h3-video`'s block is the other way round. Two files in the same
 *   model with opposite conventions is exactly the thing to check rather than
 *   remember.
 * - **RoPE turns 96 of 128 channels** — three axes of 16 frequencies, tiled —
 *   against the visual VAE's 48 of 64. `h3RopePermutation` covers both.
 *
 * Nothing here has a bias except `adaln_proj`.
 */
import { ACTIVATION, activation } from "../../../ops/activation/index.js";
import { attention } from "../../../ops/attention/index.js";
import { ELEMENTWISE, elementwise } from "../../../ops/elementwise/index.js";
import { matmul } from "../../../ops/matmul/index.js";
import { rmsnorm } from "../../../ops/rmsnorm/index.js";
import { h3RopePermutation, ropeAxes } from "../../../ops/rope/index.js";

export interface DitBlockConfig {
  hiddenSize: number;
  numHeads: number;
  headDim: number;
  ffnDim: number;
  timeEmbedDim: number;
  seq: number;
  normEps: number;
  qkNormEps: number;
  /** Three axes of `ropeFreqDim` frequencies, tiled — so `2 * 3 * freq` channels turn. */
  ropeFreqDim: number;
  ropeTheta: number;
  /** Rows the modulation table holds per timestep: text, video, audio. */
  modalityNum: number;
}

export interface DitBlockWeights {
  norm1Weight: Float32Array;
  /** `[heads * headDim, hidden]`, already permuted for RoPE. */
  toQWeight: Float32Array;
  toKWeight: Float32Array;
  toVWeight: Float32Array;
  normQWeight: Float32Array;
  normKWeight: Float32Array;
  toOutWeight: Float32Array;
  norm2Weight: Float32Array;
  /** `[2 * ffnDim, hidden]`. */
  ffProjWeight: Float32Array;
  /** `[hidden, ffnDim]`. */
  ffOutWeight: Float32Array;
  adalnWeight: Float32Array;
  adalnBias: Float32Array;
}

/** `x @ Wᵀ + b`, the layout `nn.Linear` stores. */
export function linear(
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

/** `[seq, heads, dim]` to `ops/attention`'s `[heads, seq, dim]`. */
export function splitHeads(x: Float32Array, seq: number, heads: number, dim: number): Float32Array {
  const out = new Float32Array(seq * heads * dim);
  for (let s = 0; s < seq; s += 1) {
    for (let h = 0; h < heads; h += 1) {
      for (let d = 0; d < dim; d += 1) out[(h * seq + s) * dim + d] = x[(s * heads + h) * dim + d]!;
    }
  }
  return out;
}

/** `ops/attention`'s `[heads, seq, dim]` back to `[seq, heads * dim]`. */
export function mergeHeads(x: Float32Array, seq: number, heads: number, dim: number): Float32Array {
  const out = new Float32Array(seq * heads * dim);
  for (let h = 0; h < heads; h += 1) {
    for (let s = 0; s < seq; s += 1) {
      for (let d = 0; d < dim; d += 1) out[(s * heads + h) * dim + d] = x[(h * seq + s) * dim + d]!;
    }
  }
  return out;
}

/**
 * Permutes each head's rows of a `[heads * headDim, hidden]` projection.
 *
 * Q and K only, and never V — see `h3RopePermutation`'s own doc. Applied at
 * conversion in a shipping port; here so the test can hold the block to the
 * model without a converter in between.
 *
 * **Whatever else reads those channels has to be permuted too.** This block's
 * QK-norm has per-channel weights (`norm_q.weight`, `[headDim]`), and they are
 * indexed by the channel the projection produced — so permuting the projection
 * and not the norm scales the wrong channels. It cost an afternoon here: the
 * visual VAE has `qk_norm_affine: false` and no weights at all, so the same
 * permutation is complete there and incomplete here, and the symptom was an
 * 8% error in attention with the rope itself exact to 2.4e-7.
 * `permuteChannelWeight` is that half.
 */
export function permuteProjectionForRope(
  weight: Float32Array,
  heads: number,
  headDim: number,
  rotDim: number,
  hidden: number,
): Float32Array {
  const permutation = h3RopePermutation(headDim, rotDim);
  const out = weight.slice();
  for (let head = 0; head < heads; head += 1) {
    for (let c = 0; c < headDim; c += 1) {
      const from = head * headDim + permutation[c]!;
      const to = head * headDim + c;
      out.set(weight.subarray(from * hidden, (from + 1) * hidden), to * hidden);
    }
  }
  return out;
}

/**
 * The whole `(timestep, modality)` modulation table, `[rows, 6 * hidden]`.
 *
 * `linear(silu(temb))` produces `6 * hidden * modalityNum` values per timestep,
 * which the model views as `[-1, 6 * hidden]`. Row `timestep * modalityNum +
 * tag` is the one a token uses, and **one packed sequence uses many of them at
 * once** — text, video and audio rows sit in the same tensor at possibly
 * different noise levels, which is the whole reason the table exists.
 */
export function modulationTable(cfg: DitBlockConfig, w: DitBlockWeights, temb: Float32Array): Float32Array {
  const { hiddenSize, timeEmbedDim, modalityNum } = cfg;
  const width = 6 * hiddenSize * modalityNum;
  const timesteps = temb.length / timeEmbedDim;
  const projected = linear(
    activation({ input: temb.slice(), kind: ACTIVATION.silu }),
    w.adalnWeight,
    timesteps,
    timeEmbedDim,
    width,
  );
  for (let t = 0; t < timesteps; t += 1) {
    for (let i = 0; i < width; i += 1) projected[t * width + i] = projected[t * width + i]! + w.adalnBias[i]!;
  }
  return projected;
}

/** The six modulation vectors for one row of the `(timestep, modality)` table. */
export function modulation(
  cfg: DitBlockConfig,
  w: DitBlockWeights,
  temb: Float32Array,
  row: number,
): { shiftMsa: Float32Array; scaleMsa: Float32Array; gateMsa: Float32Array; shiftMlp: Float32Array; scaleMlp: Float32Array; gateMlp: Float32Array } {
  const { hiddenSize, timeEmbedDim, modalityNum } = cfg;
  const table = modulationTable(cfg, w, temb);
  const rows = (temb.length / timeEmbedDim) * modalityNum;
  if (row < 0 || row >= rows) {
    throw new Error(`modulation: row ${row} is outside the ${rows} the table holds`);
  }
  const at = row * 6 * hiddenSize;
  const chunk = (k: number): Float32Array => table.slice(at + k * hiddenSize, at + (k + 1) * hiddenSize);
  return {
    shiftMsa: chunk(0), scaleMsa: chunk(1), gateMsa: chunk(2),
    shiftMlp: chunk(3), scaleMlp: chunk(4), gateMlp: chunk(5),
  };
}

/**
 * Gathers one of the six chunks for every row of the sequence, `[seq, hidden]`.
 *
 * `rows[s]` is the table row token `s` uses. A whole-sequence `number` is the
 * degenerate case and still goes through here, so the two paths cannot drift.
 */
function gatherChunk(
  table: Float32Array,
  rows: Int32Array,
  chunk: number,
  seq: number,
  hidden: number,
): Float32Array {
  const out = new Float32Array(seq * hidden);
  const stride = 6 * hidden;
  for (let s = 0; s < seq; s += 1) {
    const at = rows[s]! * stride + chunk * hidden;
    out.set(table.subarray(at, at + hidden), s * hidden);
  }
  return out;
}

/** `x * (1 + scale) + shift`, both `[seq, dim]`. */
function modulate(x: Float32Array, scale: Float32Array, shift: Float32Array, seq: number, dim: number): Float32Array {
  const out = new Float32Array(x.length);
  for (let s = 0; s < seq; s += 1) {
    for (let d = 0; d < dim; d += 1) {
      out[s * dim + d] = x[s * dim + d]! * (1 + scale[s * dim + d]!) + shift[s * dim + d]!;
    }
  }
  return out;
}

/** Permutes a `[headDim]` per-channel vector to match a permuted projection. */
export function permuteChannelWeight(weight: Float32Array, headDim: number, rotDim: number): Float32Array {
  const permutation = h3RopePermutation(headDim, rotDim);
  return Float32Array.from(permutation, (from) => weight[from]!);
}

/**
 * One block, `[seq, hidden]` to `[seq, hidden]`.
 *
 * `adaln` is either one table row for the whole sequence or **one row per
 * token**. A packed sequence needs the second: its text, video and audio rows
 * carry different modality tags and can sit at different noise levels, and a
 * block that modulated them all alike would still return the right shape.
 */
export function h3DitBlock(
  cfg: DitBlockConfig,
  w: DitBlockWeights,
  x: Float32Array,
  temb: Float32Array,
  adaln: number | Int32Array,
  positions: Float32Array,
): Float32Array {
  const { hiddenSize, numHeads, headDim, ffnDim, seq, normEps, qkNormEps, ropeFreqDim, ropeTheta } = cfg;
  const width = numHeads * headDim;
  const rotDim = 2 * 3 * ropeFreqDim;
  // Three real axes of `2 * ropeFreqDim` channels each, and a fourth pinned at
  // position zero covering the channels H3 leaves unrotated — a rotation by
  // zero is the identity, so `rope` needs no second path.
  const axisDims = [2 * ropeFreqDim, 2 * ropeFreqDim, 2 * ropeFreqDim, headDim - rotDim];

  const rows = typeof adaln === "number" ? new Int32Array(seq).fill(adaln) : adaln;
  if (rows.length !== seq) {
    throw new Error(`h3DitBlock: ${rows.length} AdaLN rows for ${seq} tokens`);
  }
  const tableRows = (temb.length / cfg.timeEmbedDim) * cfg.modalityNum;
  for (const row of rows) {
    if (row < 0 || row >= tableRows) {
      throw new Error(`h3DitBlock: AdaLN row ${row} is outside the ${tableRows} the table holds`);
    }
  }
  const table = modulationTable(cfg, w, temb);
  const m = (chunk: number): Float32Array => gatherChunk(table, rows, chunk, seq, hiddenSize);
  const [shiftMsa, scaleMsa, gateMsa, shiftMlp, scaleMlp, gateMlp] = [m(0), m(1), m(2), m(3), m(4), m(5)];

  const normed1 = modulate(
    rmsnorm({ input: x, weight: w.norm1Weight, N: seq, D: hiddenSize, eps: normEps }),
    scaleMsa, shiftMsa, seq, hiddenSize,
  );

  let q = linear(normed1, w.toQWeight, seq, hiddenSize, width);
  let k = linear(normed1, w.toKWeight, seq, hiddenSize, width);
  const v = linear(normed1, w.toVWeight, seq, hiddenSize, width);

  q = rmsnorm({ input: q, weight: w.normQWeight, N: seq * numHeads, D: headDim, eps: qkNormEps });
  k = rmsnorm({ input: k, weight: w.normKWeight, N: seq * numHeads, D: headDim, eps: qkNormEps });

  q = ropeAxes({ input: q, N: seq, numHeads, axisDims, positions, thetaBase: ropeTheta });
  k = ropeAxes({ input: k, N: seq, numHeads, axisDims, positions, thetaBase: ropeTheta });

  // One packed sequence, no mask: "MiniMax-H3 packs one request into a single
  // attention document, so the model passes no mask" — the processor's own note.
  const attended = attention({
    q: splitHeads(q, seq, numHeads, headDim),
    k: splitHeads(k, seq, numHeads, headDim),
    v: splitHeads(v, seq, numHeads, headDim),
    B: 1, H: numHeads, L: seq, S: seq, D: headDim, Dv: headDim,
    causal: false,
  });
  const merged = mergeHeads(attended.output, seq, numHeads, headDim);

  const projected = linear(merged, w.toOutWeight, seq, width, hiddenSize);
  // A gate per row, not one broadcast down the sequence: `elementwiseRows` is
  // the wrong shape here for the same reason `modulate` changed.
  const gatedAttn = elementwise({ a: projected, b: gateMsa, kind: ELEMENTWISE.multiply });
  let hidden = elementwise({ a: x, b: gatedAttn, kind: ELEMENTWISE.add });

  const normed2 = modulate(
    rmsnorm({ input: hidden, weight: w.norm2Weight, N: seq, D: hiddenSize, eps: normEps }),
    scaleMlp, shiftMlp, seq, hiddenSize,
  );

  // diffusers' `SwiGLU`: `hidden, gate = proj(x).chunk(2, -1); hidden * silu(gate)`.
  // **`hidden` is the first half here** — `examples/h3-video`'s decoder is the
  // other way round, and both are in the same model.
  const wide = linear(normed2, w.ffProjWeight, seq, hiddenSize, 2 * ffnDim);
  const value = new Float32Array(seq * ffnDim);
  const gate = new Float32Array(seq * ffnDim);
  for (let s = 0; s < seq; s += 1) {
    value.set(wide.subarray(s * 2 * ffnDim, s * 2 * ffnDim + ffnDim), s * ffnDim);
    gate.set(wide.subarray(s * 2 * ffnDim + ffnDim, (s + 1) * 2 * ffnDim), s * ffnDim);
  }
  const activated = elementwise({
    a: value,
    b: activation({ input: gate, kind: ACTIVATION.silu }),
    kind: ELEMENTWISE.multiply,
  });

  const ff = linear(activated, w.ffOutWeight, seq, ffnDim, hiddenSize);
  const gatedFf = elementwise({ a: ff, b: gateMlp, kind: ELEMENTWISE.multiply });
  hidden = elementwise({ a: hidden, b: gatedFf, kind: ELEMENTWISE.add });
  return hidden;
}
