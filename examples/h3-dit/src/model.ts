/**
 * The whole MiniMax-H3 DiT forward — the generator half of the model.
 *
 * Issue #210. `block.ts` holds one block to diffusers' own implementation
 * (#208), and the assumption there was that "the rest is a loop". It is not.
 * Around the fifty blocks sit five things that each return a well-formed tensor
 * when they are wrong:
 *
 * - **One packed sequence, not cross-attention.** Text, audio and video rows
 *   are projected separately and *scattered into one buffer* at caller-chosen
 *   positions. There is no cross-attention anywhere in H3.
 * - **The text stream is refined first**, by two *plain* pre-norm blocks — no
 *   AdaLN, no rotary — and a final norm. Skipping them leaves the shape intact.
 * - **The AdaLN table is addressed per row.** `timestep_indices * 3 +
 *   token_tags` picks a row per token, so one forward serves rows at different
 *   noise levels. A sequence modulated uniformly still comes out the right size.
 * - **`norm_out` modulates per row too**, with `shift` before `scale` — the
 *   opposite order to the block's six-way chunk.
 * - **Both heads run over every row**, and the modality's rows are selected
 *   *afterwards*. Selecting first would change which rows attention saw.
 *
 * The arithmetic is diffusers' `MiniMaxH3Transformer3DModel`; what is held
 * against it is `tools/gen_forward_golden.py`, which instantiates that class.
 */
import { ACTIVATION, activation } from "../../../ops/activation/index.js";
import { attention } from "../../../ops/attention/index.js";
import { ELEMENTWISE, elementwise } from "../../../ops/elementwise/index.js";
import { rmsnorm } from "../../../ops/rmsnorm/index.js";
import {
  h3DitBlock,
  linear,
  mergeHeads,
  splitHeads,
  type DitBlockConfig,
  type DitBlockWeights,
} from "./block.js";

/** Rows the modulation table holds per timestep: video, text, audio. */
export const MODALITY_NUM = 3;

export interface DitConfig {
  numHeads: number;
  headDim: number;
  hiddenSize: number;
  numLayers: number;
  numRefinerLayers: number;
  ffnDim: number;
  inChannels: number;
  audioInChannels: number;
  /** `(t, h, w)`; the video patch dim is `inChannels * t * h * w`. */
  patchSize: [number, number, number];
  textDim: number;
  freqDim: number;
  timeEmbedHiddenDim: number;
  timeEmbedDim: number;
  ropeFreqDim: number;
  ropeTheta: number;
  normEps: number;
  qkNormEps: number;
  finalNormEps: number;
}

/** One plain pre-norm block of the text refiner. No AdaLN and no rotary. */
export interface RefinerBlockWeights {
  norm1Weight: Float32Array;
  toQWeight: Float32Array;
  toKWeight: Float32Array;
  toVWeight: Float32Array;
  normQWeight: Float32Array;
  normKWeight: Float32Array;
  toOutWeight: Float32Array;
  norm2Weight: Float32Array;
  ffProjWeight: Float32Array;
  ffOutWeight: Float32Array;
}

export interface DitWeights {
  projInWeight: Float32Array;
  projInBias: Float32Array;
  audioProjInWeight: Float32Array;
  audioProjInBias: Float32Array;
  contextEmbedderWeight: Float32Array;
  contextEmbedderBias: Float32Array;
  timeLinear1Weight: Float32Array;
  timeLinear1Bias: Float32Array;
  timeLinear2Weight: Float32Array;
  timeLinear2Bias: Float32Array;
  refinerBlocks: RefinerBlockWeights[];
  refinerFinalNormWeight: Float32Array;
  blocks: DitBlockWeights[];
  normOutWeight: Float32Array;
  normOutLinearWeight: Float32Array;
  normOutLinearBias: Float32Array;
  projOutWeight: Float32Array;
  projOutBias: Float32Array;
  audioProjOutWeight: Float32Array;
  audioProjOutBias: Float32Array;
}

/**
 * Where every row of the packed sequence sits, and what it is.
 *
 * The transformer builds none of this: the caller orders the rows, tags each
 * with its modality and its noise level, and hands over the `(t, h, w)` grid.
 */
export interface PackedLayout {
  seq: number;
  /** `0` video, `1` text, `2` audio — one per row. */
  tokenTags: Int32Array;
  /** Index into `timestep` — one per row. */
  timestepIndices: Int32Array;
  /** `[seq, 3]`, token-major. */
  positionIds: Float32Array;
  videoIndices: Int32Array;
  audioIndices: Int32Array;
  textIndices: Int32Array;
}

export interface DitInputs {
  /** `[numVideoTokens, inChannels * prod(patchSize)]`, ordered as `videoIndices`. */
  video: Float32Array;
  /** `[numAudioTokens, audioInChannels]`. */
  audio: Float32Array;
  /** `[numTextTokens, textDim]`. */
  text: Float32Array;
  /** The *distinct* noise levels in the sequence, in `[0, 1]` and unscaled. */
  timestep: Float32Array;
}

/**
 * diffusers' `get_timestep_embedding` with H3's settings.
 *
 * `flip_sin_to_cos=True` and `downscale_freq_shift=0`: cosines first, and the
 * exponent divides by `half_dim` rather than `half_dim - 1`. Both are choices
 * a plausible reimplementation gets the other way round, and neither changes
 * the shape. Timesteps are consumed **unscaled**, in `[0, 1]`.
 */
export function timestepEmbedding(timesteps: Float32Array, dim: number, maxPeriod = 10000): Float32Array {
  if (dim % 2 !== 0) throw new Error(`timestepEmbedding: odd dim ${dim} would need a zero pad`);
  const half = dim / 2;
  const out = new Float32Array(timesteps.length * dim);
  for (let t = 0; t < timesteps.length; t += 1) {
    for (let i = 0; i < half; i += 1) {
      const angle = timesteps[t]! * Math.exp((-Math.log(maxPeriod) * i) / half);
      out[t * dim + i] = Math.cos(angle);
      out[t * dim + half + i] = Math.sin(angle);
    }
  }
  return out;
}

/** `linear_1 -> silu -> linear_2`, diffusers' `TimestepEmbedding`. */
function timeEmbedder(cfg: DitConfig, w: DitWeights, projected: Float32Array): Float32Array {
  const rows = projected.length / cfg.freqDim;
  const first = linear(projected, w.timeLinear1Weight, rows, cfg.freqDim, cfg.timeEmbedHiddenDim, w.timeLinear1Bias);
  return linear(
    activation({ input: first, kind: ACTIVATION.silu }),
    w.timeLinear2Weight,
    rows,
    cfg.timeEmbedHiddenDim,
    cfg.timeEmbedDim,
    w.timeLinear2Bias,
  );
}

/**
 * One refiner block: `x + attn(norm1(x))`, then `x + ff(norm2(x))`.
 *
 * The same SwiGLU as the DiT block — `hidden * silu(gate)`, hidden first — and
 * the same QK norm, but **no rotary**: the text stream carries no position
 * grid. Rotating it anyway would be a plausible-looking mistake that leaves
 * every shape correct.
 */
function refinerBlock(cfg: DitConfig, w: RefinerBlockWeights, x: Float32Array, seq: number): Float32Array {
  const { hiddenSize, numHeads, headDim, ffnDim, normEps, qkNormEps } = cfg;
  const width = numHeads * headDim;

  const normed1 = rmsnorm({ input: x, weight: w.norm1Weight, N: seq, D: hiddenSize, eps: normEps });
  let q = linear(normed1, w.toQWeight, seq, hiddenSize, width);
  let k = linear(normed1, w.toKWeight, seq, hiddenSize, width);
  const v = linear(normed1, w.toVWeight, seq, hiddenSize, width);
  q = rmsnorm({ input: q, weight: w.normQWeight, N: seq * numHeads, D: headDim, eps: qkNormEps });
  k = rmsnorm({ input: k, weight: w.normKWeight, N: seq * numHeads, D: headDim, eps: qkNormEps });

  const attended = attention({
    q: splitHeads(q, seq, numHeads, headDim),
    k: splitHeads(k, seq, numHeads, headDim),
    v: splitHeads(v, seq, numHeads, headDim),
    B: 1, H: numHeads, L: seq, S: seq, D: headDim, Dv: headDim,
    causal: false,
  });
  const projected = linear(mergeHeads(attended.output, seq, numHeads, headDim), w.toOutWeight, seq, width, hiddenSize);
  let hidden = elementwise({ a: x, b: projected, kind: ELEMENTWISE.add });

  const normed2 = rmsnorm({ input: hidden, weight: w.norm2Weight, N: seq, D: hiddenSize, eps: normEps });
  const wide = linear(normed2, w.ffProjWeight, seq, hiddenSize, 2 * ffnDim);
  const value = new Float32Array(seq * ffnDim);
  const gate = new Float32Array(seq * ffnDim);
  for (let s = 0; s < seq; s += 1) {
    value.set(wide.subarray(s * 2 * ffnDim, s * 2 * ffnDim + ffnDim), s * ffnDim);
    gate.set(wide.subarray(s * 2 * ffnDim + ffnDim, (s + 1) * 2 * ffnDim), s * ffnDim);
  }
  const ff = linear(
    elementwise({ a: value, b: activation({ input: gate, kind: ACTIVATION.silu }), kind: ELEMENTWISE.multiply }),
    w.ffOutWeight,
    seq,
    ffnDim,
    hiddenSize,
  );
  hidden = elementwise({ a: hidden, b: ff, kind: ELEMENTWISE.add });
  return hidden;
}

/** Writes `rows` of `[n, dim]` into `target` at the row positions `indices` names. */
function scatterRows(target: Float32Array, rows: Float32Array, indices: Int32Array, dim: number): void {
  for (let i = 0; i < indices.length; i += 1) {
    target.set(rows.subarray(i * dim, (i + 1) * dim), indices[i]! * dim);
  }
}

/** Reads the rows `indices` names out of `[seq, dim]`. */
function gatherRows(source: Float32Array, indices: Int32Array, dim: number): Float32Array {
  const out = new Float32Array(indices.length * dim);
  for (let i = 0; i < indices.length; i += 1) {
    out.set(source.subarray(indices[i]! * dim, (indices[i]! + 1) * dim), i * dim);
  }
  return out;
}

/** The block config the fifty blocks share, derived from the model config. */
export function blockConfig(cfg: DitConfig, seq: number): DitBlockConfig {
  return {
    hiddenSize: cfg.hiddenSize,
    numHeads: cfg.numHeads,
    headDim: cfg.headDim,
    ffnDim: cfg.ffnDim,
    timeEmbedDim: cfg.timeEmbedDim,
    seq,
    normEps: cfg.normEps,
    qkNormEps: cfg.qkNormEps,
    ropeFreqDim: cfg.ropeFreqDim,
    ropeTheta: cfg.ropeTheta,
    modalityNum: MODALITY_NUM,
  };
}

/**
 * `(t, h, w)` per row widened to the four axes `ropeAxes` takes.
 *
 * The fourth axis is pinned at zero and covers the `headDim - 2 * 3 *
 * ropeFreqDim` channels H3 leaves unrotated: a rotation by zero is the
 * identity, so one op serves both halves.
 */
export function ropePositions(positionIds: Float32Array, seq: number): Float32Array {
  if (positionIds.length !== seq * 3) {
    throw new Error(`ropePositions: ${positionIds.length} values for ${seq} rows of (t, h, w)`);
  }
  const out = new Float32Array(seq * 4);
  for (let token = 0; token < seq; token += 1) {
    for (let axis = 0; axis < 3; axis += 1) out[token * 4 + axis] = positionIds[token * 3 + axis]!;
  }
  return out;
}

export interface DitOutputs {
  /** `[numVideoTokens, inChannels * prod(patchSize)]` — the video velocity. */
  video: Float32Array;
  /** `[numAudioTokens, audioInChannels]` — the audio velocity. */
  audio: Float32Array;
}

/** One forward. Batch is a pure replication axis upstream, so this runs one item. */
export function h3DitForward(
  cfg: DitConfig,
  w: DitWeights,
  inputs: DitInputs,
  layout: PackedLayout,
  onBlock?: (index: number) => void,
): DitOutputs {
  const { hiddenSize } = cfg;
  const { seq } = layout;
  const videoPatchDim = cfg.inChannels * cfg.patchSize[0] * cfg.patchSize[1] * cfg.patchSize[2];

  if (layout.tokenTags.length !== seq || layout.timestepIndices.length !== seq) {
    throw new Error(`h3DitForward: tags and timestep indices must both be ${seq} long`);
  }
  if (w.blocks.length !== cfg.numLayers) {
    throw new Error(`h3DitForward: ${w.blocks.length} blocks for ${cfg.numLayers} layers`);
  }
  if (w.refinerBlocks.length !== cfg.numRefinerLayers) {
    throw new Error(`h3DitForward: ${w.refinerBlocks.length} refiner blocks for ${cfg.numRefinerLayers} layers`);
  }

  // 1. Project each modality, refine the text stream, scatter into one buffer.
  const numText = layout.textIndices.length;
  const videoEmbeds = linear(
    inputs.video, w.projInWeight, layout.videoIndices.length, videoPatchDim, hiddenSize, w.projInBias);
  const audioEmbeds = linear(
    inputs.audio, w.audioProjInWeight, layout.audioIndices.length, cfg.audioInChannels, hiddenSize, w.audioProjInBias);
  let textEmbeds = linear(
    inputs.text, w.contextEmbedderWeight, numText, cfg.textDim, hiddenSize, w.contextEmbedderBias);
  for (const block of w.refinerBlocks) textEmbeds = refinerBlock(cfg, block, textEmbeds, numText);
  textEmbeds = rmsnorm({
    input: textEmbeds, weight: w.refinerFinalNormWeight, N: numText, D: hiddenSize, eps: cfg.finalNormEps,
  });

  // The order matters only where the index sets overlap, and they must not; the
  // model would silently let a later `index_copy` win.
  const hidden = new Float32Array(seq * hiddenSize);
  scatterRows(hidden, textEmbeds, layout.textIndices, hiddenSize);
  scatterRows(hidden, videoEmbeds, layout.videoIndices, hiddenSize);
  scatterRows(hidden, audioEmbeds, layout.audioIndices, hiddenSize);

  // 2. One timestep embedding per distinct noise level, shared by every AdaLN.
  const temb = timeEmbedder(cfg, w, timestepEmbedding(inputs.timestep, cfg.freqDim));

  // 3. Row -> AdaLN table row.
  const adalnIndices = new Int32Array(seq);
  for (let s = 0; s < seq; s += 1) {
    adalnIndices[s] = layout.timestepIndices[s]! * MODALITY_NUM + layout.tokenTags[s]!;
  }

  const bcfg = blockConfig(cfg, seq);
  const positions = ropePositions(layout.positionIds, seq);
  // Annotated rather than inferred: `new Float32Array(...)` infers
  // `Float32Array<ArrayBuffer>`, the ops return `Float32Array<ArrayBufferLike>`,
  // and under the DOM lib those are not the same type.
  let x: Float32Array = hidden;
  for (let i = 0; i < cfg.numLayers; i += 1) {
    x = h3DitBlock(bcfg, w.blocks[i]!, x, temb, adalnIndices, positions);
    onBlock?.(i);
  }

  // 4. `norm_out`: RMSNorm, then a per-row shift and scale. **`shift` is the
  // first half** of the projection — the order `LTX2` and `Wan` use in their
  // output layers, and the reverse of the block's six-way chunk.
  const timesteps = inputs.timestep.length;
  const shiftScale = linear(
    activation({ input: temb.slice(), kind: ACTIVATION.silu }),
    w.normOutLinearWeight,
    timesteps,
    cfg.timeEmbedDim,
    2 * hiddenSize,
    w.normOutLinearBias,
  );
  const normed = rmsnorm({ input: x, weight: w.normOutWeight, N: seq, D: hiddenSize, eps: cfg.finalNormEps });
  const modulated = new Float32Array(normed.length);
  for (let s = 0; s < seq; s += 1) {
    const at = layout.timestepIndices[s]! * 2 * hiddenSize;
    for (let d = 0; d < hiddenSize; d += 1) {
      modulated[s * hiddenSize + d] =
        normed[s * hiddenSize + d]! * (1 + shiftScale[at + hiddenSize + d]!) + shiftScale[at + d]!;
    }
  }

  // 5. Both heads run over **every** row; the modality's rows are selected after.
  const videoAll = linear(modulated, w.projOutWeight, seq, hiddenSize, videoPatchDim, w.projOutBias);
  const audioAll = linear(modulated, w.audioProjOutWeight, seq, hiddenSize, cfg.audioInChannels, w.audioProjOutBias);
  return {
    video: gatherRows(videoAll, layout.videoIndices, videoPatchDim),
    audio: gatherRows(audioAll, layout.audioIndices, cfg.audioInChannels),
  };
}
