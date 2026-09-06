/**
 * The whole DiT forward against diffusers' own `MiniMaxH3Transformer3DModel`.
 *
 * Issue #210. Unlike `block.test.ts`, **this fixture is in the repository** and
 * runs everywhere. `tools/gen_forward_golden.py` instantiates upstream's class
 * at the geometry upstream's own tester uses — hidden 24, two layers, heads
 * 2x16 — with **random weights**, so the file is 155 KB and carries none of
 * MiniMax's licence. The arithmetic is still upstream's.
 *
 * Random weights lose nothing here. What the forward can get wrong is
 * structural — row order, where each modality is scattered, how the AdaLN table
 * is addressed, whether the refiner runs, which half of `norm_out`'s projection
 * is the shift — and none of that depends on the weights being trained. The
 * real checkpoint is what `block.test.ts` covers.
 *
 * Two of the tester's choices are load-bearing and deliberate: `numHeads *
 * headDim` (32) differs from `hiddenSize` (24), as in the released checkpoint,
 * and `2 * 3 * ropeFreqDim` (12) is smaller than `headDim` (16), so the
 * partial-rotary path is exercised rather than aliased away.
 */
import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { permuteChannelWeight, permuteProjectionForRope, type DitBlockWeights } from "./block.js";
import {
  h3DitForward,
  timestepEmbedding,
  type DitConfig,
  type DitWeights,
  type PackedLayout,
  type RefinerBlockWeights,
} from "./model.js";

const here = new URL("../fixtures/", import.meta.url);
const manifest = JSON.parse(readFileSync(new URL("forward.json", here), "utf8")) as {
  config: {
    num_attention_heads: number; attention_head_dim: number; hidden_size: number;
    num_layers: number; num_refiner_layers: number; ffn_dim: number;
    in_channels: number; audio_in_channels: number; patch_size: [number, number, number];
    text_dim: number; freq_dim: number; time_embed_hidden_dim: number; time_embed_dim: number;
    rope_freq_dim: number;
  };
  layout: {
    numTextTokens: number; numAudioTokens: number; numVideoTokens: number;
    tokenTags: number[]; timestepIndices: number[];
    videoIndices: number[]; audioIndices: number[]; textIndices: number[];
  };
  tensors: { name: string; shape: number[]; offset: number; count: number }[];
};
const blob = readFileSync(new URL("forward.bin", here));

const view = (name: string): Float32Array => {
  const entry = manifest.tensors.find((t) => t.name === name);
  if (!entry) throw new Error(`forward.bin has no tensor named "${name}"`);
  // Copied rather than mapped: the offsets are not guaranteed 4-byte aligned
  // against the Buffer's own byteOffset, and a misaligned view throws.
  return new Float32Array(
    blob.buffer.slice(blob.byteOffset + entry.offset, blob.byteOffset + entry.offset + entry.count * 4),
  );
};

const c = manifest.config;
const cfg: DitConfig = {
  numHeads: c.num_attention_heads,
  headDim: c.attention_head_dim,
  hiddenSize: c.hidden_size,
  numLayers: c.num_layers,
  numRefinerLayers: c.num_refiner_layers,
  ffnDim: c.ffn_dim,
  inChannels: c.in_channels,
  audioInChannels: c.audio_in_channels,
  patchSize: c.patch_size,
  textDim: c.text_dim,
  freqDim: c.freq_dim,
  timeEmbedHiddenDim: c.time_embed_hidden_dim,
  timeEmbedDim: c.time_embed_dim,
  ropeFreqDim: c.rope_freq_dim,
  // The three the tester leaves at their defaults.
  ropeTheta: 10000,
  normEps: 1e-5,
  qkNormEps: 1e-5,
  finalNormEps: 1e-5,
};

const rotDim = 2 * 3 * cfg.ropeFreqDim;

/**
 * The DiT blocks' Q and K are permuted for `ropeAxes`; **the refiner's are
 * not**, because the refiner has no rotary at all. Two attention modules with
 * identical parameter names and different weight preparation is exactly the
 * kind of thing that reads fine and computes the wrong answer.
 */
function ditBlockWeights(prefix: string): DitBlockWeights {
  const permute = (name: string): Float32Array =>
    permuteProjectionForRope(view(`${prefix}.${name}`), cfg.numHeads, cfg.headDim, rotDim, cfg.hiddenSize);
  return {
    norm1Weight: view(`${prefix}.norm1.weight`),
    toQWeight: permute("attn.to_q.weight"),
    toKWeight: permute("attn.to_k.weight"),
    toVWeight: view(`${prefix}.attn.to_v.weight`),
    normQWeight: permuteChannelWeight(view(`${prefix}.attn.norm_q.weight`), cfg.headDim, rotDim),
    normKWeight: permuteChannelWeight(view(`${prefix}.attn.norm_k.weight`), cfg.headDim, rotDim),
    toOutWeight: view(`${prefix}.attn.to_out.0.weight`),
    norm2Weight: view(`${prefix}.norm2.weight`),
    ffProjWeight: view(`${prefix}.ff.net.0.proj.weight`),
    ffOutWeight: view(`${prefix}.ff.net.2.weight`),
    adalnWeight: view(`${prefix}.adaln_proj.linear.weight`),
    adalnBias: view(`${prefix}.adaln_proj.linear.bias`),
  };
}

function refinerWeights(prefix: string): RefinerBlockWeights {
  return {
    norm1Weight: view(`${prefix}.norm1.weight`),
    toQWeight: view(`${prefix}.attn.to_q.weight`),
    toKWeight: view(`${prefix}.attn.to_k.weight`),
    toVWeight: view(`${prefix}.attn.to_v.weight`),
    normQWeight: view(`${prefix}.attn.norm_q.weight`),
    normKWeight: view(`${prefix}.attn.norm_k.weight`),
    toOutWeight: view(`${prefix}.attn.to_out.0.weight`),
    norm2Weight: view(`${prefix}.norm2.weight`),
    ffProjWeight: view(`${prefix}.ff.net.0.proj.weight`),
    ffOutWeight: view(`${prefix}.ff.net.2.weight`),
  };
}

const weights: DitWeights = {
  projInWeight: view("proj_in.weight"),
  projInBias: view("proj_in.bias"),
  audioProjInWeight: view("audio_proj_in.weight"),
  audioProjInBias: view("audio_proj_in.bias"),
  contextEmbedderWeight: view("context_embedder.weight"),
  contextEmbedderBias: view("context_embedder.bias"),
  timeLinear1Weight: view("time_embedder.linear_1.weight"),
  timeLinear1Bias: view("time_embedder.linear_1.bias"),
  timeLinear2Weight: view("time_embedder.linear_2.weight"),
  timeLinear2Bias: view("time_embedder.linear_2.bias"),
  refinerBlocks: Array.from({ length: cfg.numRefinerLayers }, (_, i) =>
    refinerWeights(`token_refiner.refiner_blocks.${i}`)),
  refinerFinalNormWeight: view("token_refiner.final_norm.weight"),
  blocks: Array.from({ length: cfg.numLayers }, (_, i) => ditBlockWeights(`transformer_blocks.${i}`)),
  normOutWeight: view("norm_out.norm.weight"),
  normOutLinearWeight: view("norm_out.linear.weight"),
  normOutLinearBias: view("norm_out.linear.bias"),
  projOutWeight: view("proj_out.weight"),
  projOutBias: view("proj_out.bias"),
  audioProjOutWeight: view("audio_proj_out.weight"),
  audioProjOutBias: view("audio_proj_out.bias"),
};

const L = manifest.layout;
const layout: PackedLayout = {
  seq: L.tokenTags.length,
  tokenTags: Int32Array.from(L.tokenTags),
  timestepIndices: Int32Array.from(L.timestepIndices),
  positionIds: view("input.position_ids"),
  videoIndices: Int32Array.from(L.videoIndices),
  audioIndices: Int32Array.from(L.audioIndices),
  textIndices: Int32Array.from(L.textIndices),
};

function compare(got: Float32Array, want: Float32Array): { worst: number; rms: number; peak: number } {
  expect(got.length).toBe(want.length);
  let worst = 0;
  let sum = 0;
  let peak = 0;
  for (let i = 0; i < want.length; i += 1) {
    const d = Math.abs(got[i]! - want[i]!);
    sum += d * d;
    if (d > worst) worst = d;
    peak = Math.max(peak, Math.abs(want[i]!));
  }
  return { worst, rms: Math.sqrt(sum / want.length), peak };
}

describe("h3 dit / forward", () => {
  it("reproduces the outputs diffusers' own model produced", () => {
    const got = h3DitForward(cfg, weights, {
      video: view("input.hidden_states"),
      audio: view("input.audio_hidden_states"),
      text: view("input.encoder_hidden_states"),
      timestep: view("input.timestep"),
    }, layout);

    const video = compare(got.video, view("output.video"));
    const audio = compare(got.audio, view("output.audio"));
    console.log(
      `h3 dit forward: video worst ${video.worst.toExponential(3)} rms ${video.rms.toExponential(3)} ` +
        `peak ${video.peak.toFixed(4)} | audio worst ${audio.worst.toExponential(3)} peak ${audio.peak.toFixed(4)}`,
    );
    // The golden is f64 and this port is f32 throughout, on an output of order
    // one, so what is left is f32 rounding over a two-block stack.
    expect(video.worst).toBeLessThan(1e-5);
    expect(audio.worst).toBeLessThan(1e-5);
  });

  it("refuses a layout that does not describe the sequence", () => {
    // Every one of these returns a well-formed tensor if it is not checked.
    const short = { ...layout, tokenTags: layout.tokenTags.slice(0, layout.seq - 1) };
    expect(() => h3DitForward(cfg, weights, {
      video: view("input.hidden_states"), audio: view("input.audio_hidden_states"),
      text: view("input.encoder_hidden_states"), timestep: view("input.timestep"),
    }, short)).toThrow(/timestep indices/);

    const fewer = { ...weights, blocks: weights.blocks.slice(0, 1) };
    expect(() => h3DitForward(cfg, fewer, {
      video: view("input.hidden_states"), audio: view("input.audio_hidden_states"),
      text: view("input.encoder_hidden_states"), timestep: view("input.timestep"),
    }, layout)).toThrow(/1 blocks for 2 layers/);
  });

  it("puts the cosines first in the timestep embedding", () => {
    // `flip_sin_to_cos=True`. At `t = 0` every angle is zero, so the first half
    // is all ones and the second all zeros — the one input where the flip is
    // visible without recomputing the schedule by hand. Reversed, this reads
    // zeros then ones.
    const e = timestepEmbedding(Float32Array.from([0]), 8);
    expect([...e]).toEqual([1, 1, 1, 1, 0, 0, 0, 0]);

    // And `downscale_freq_shift = 0`: the exponent divides by `half`, not
    // `half - 1`, so the last frequency is `maxPeriod ** (-(half-1)/half)`
    // rather than `1 / maxPeriod`.
    const one = timestepEmbedding(Float32Array.from([1]), 8);
    expect(one[7]).toBeCloseTo(Math.sin(Math.exp((-Math.log(10000) * 3) / 4)), 6);
  });
});
