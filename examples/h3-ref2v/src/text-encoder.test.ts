/**
 * Qwen3-VL's text decoder against `transformers`' own model.
 *
 * Issue #212. The fixture is **in this repository** and runs everywhere: the
 * golden is `Qwen3VLTextModel` at a tiny geometry with **random weights**, so
 * it is 122 KB and carries no model licence. What it establishes is structure,
 * and none of that depends on the weights being trained.
 *
 * The tiny config keeps the two things that make this stack different from the
 * DiT's: **64 query heads over 8 key/value heads** becomes 2 over 1, and
 * `mrope_section` keeps a genuine three-way interleave — `[4, 2, 2]` over
 * `headDim / 2 = 8` puts height at channels 1 and 4, width at 2 and 5, and
 * leaves 0, 3, 6 and 7 to time.
 */
import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import {
  mropeAxisOfChannel,
  mropePermutation,
  mropePositions,
  permuteChannels,
  permuteProjection,
  textEncoderHiddenStates,
  type TextEncoderConfig,
  type TextEncoderWeights,
  type TextLayerWeights,
} from "./text-encoder.js";

const here = new URL("../fixtures/", import.meta.url);
const manifest = JSON.parse(readFileSync(new URL("text-encoder.json", here), "utf8")) as {
  config: {
    hidden_size: number; intermediate_size: number; num_hidden_layers: number;
    num_attention_heads: number; num_key_value_heads: number; head_dim: number;
    rms_norm_eps: number; rope_theta: number;
    rope_scaling: { mrope_section: [number, number, number]; mrope_interleaved: boolean };
  };
  seq: number;
  positionIds: { t: number[]; h: number[]; w: number[] };
  numHiddenStates: number;
  tensors: { name: string; shape: number[]; offset: number; count: number }[];
};
const blob = readFileSync(new URL("text-encoder.bin", here));

const view = (name: string): Float32Array => {
  const entry = manifest.tensors.find((t) => t.name === name);
  if (!entry) throw new Error(`text-encoder.bin has no tensor named "${name}"`);
  return new Float32Array(
    blob.buffer.slice(blob.byteOffset + entry.offset, blob.byteOffset + entry.offset + entry.count * 4),
  );
};

const c = manifest.config;
const cfg: TextEncoderConfig = {
  hiddenSize: c.hidden_size,
  intermediateSize: c.intermediate_size,
  numHiddenLayers: c.num_hidden_layers,
  numAttentionHeads: c.num_attention_heads,
  numKeyValueHeads: c.num_key_value_heads,
  headDim: c.head_dim,
  rmsNormEps: c.rms_norm_eps,
  ropeTheta: c.rope_theta,
  mropeSection: c.rope_scaling.mrope_section,
};

const order = mropePermutation(cfg.headDim);

function layerWeights(index: number): TextLayerWeights {
  const p = `layers.${index}`;
  return {
    inputLayernorm: view(`${p}.input_layernorm.weight`),
    // Permuted for `ropeAxes`, as a converter would leave them…
    qProj: permuteProjection(
      view(`${p}.self_attn.q_proj.weight`), cfg.numAttentionHeads, cfg.headDim, cfg.hiddenSize, order),
    kProj: permuteProjection(
      view(`${p}.self_attn.k_proj.weight`), cfg.numKeyValueHeads, cfg.headDim, cfg.hiddenSize, order),
    vProj: view(`${p}.self_attn.v_proj.weight`),
    oProj: view(`${p}.self_attn.o_proj.weight`),
    // …and so are the per-head norms, which index the channels the projection
    // produced. #208 is what happens when only one of the two moves.
    qNorm: permuteChannels(view(`${p}.self_attn.q_norm.weight`), order),
    kNorm: permuteChannels(view(`${p}.self_attn.k_norm.weight`), order),
    postAttentionLayernorm: view(`${p}.post_attention_layernorm.weight`),
    gateProj: view(`${p}.mlp.gate_proj.weight`),
    upProj: view(`${p}.mlp.up_proj.weight`),
    downProj: view(`${p}.mlp.down_proj.weight`),
  };
}

const weights: TextEncoderWeights = {
  layers: Array.from({ length: cfg.numHiddenLayers }, (_, i) => layerWeights(i)),
  finalNorm: view("norm.weight"),
};

const positions = mropePositions(manifest.positionIds, cfg.headDim, cfg.mropeSection, cfg.ropeTheta);

function compare(got: Float32Array, want: Float32Array): { worst: number; rms: number } {
  expect(got.length).toBe(want.length);
  let worst = 0;
  let sum = 0;
  for (let i = 0; i < want.length; i += 1) {
    const d = Math.abs(got[i]! - want[i]!);
    sum += d * d;
    if (d > worst) worst = d;
  }
  return { worst, rms: Math.sqrt(sum / want.length) };
}

describe("h3 ref2v / text encoder", () => {
  it("has a config that exercises what makes this stack different", () => {
    // Grouped query attention, not multi-head: a port that ignored the
    // grouping would read the wrong keys, and at 1:1 it could not.
    expect(cfg.numAttentionHeads).toBeGreaterThan(cfg.numKeyValueHeads);
    // A genuine three-way interleave, and interleaved rather than chunked.
    expect(c.rope_scaling.mrope_interleaved).toBe(true);
    expect(new Set(mropeAxisOfChannel(cfg.headDim, cfg.mropeSection))).toEqual(new Set([0, 1, 2]));
  });

  it("reproduces every hidden state", () => {
    const states = textEncoderHiddenStates(cfg, weights, view("input.embeds"), manifest.seq, positions);
    expect(states.length).toBe(manifest.numHiddenStates);
    // The last entry carries the final norm; the rest do not. Asserted here
    // rather than left to the comparison, because that is the one difference
    // between the two that reads as an arithmetic error.
    expect(manifest.numHiddenStates).toBe(cfg.numHiddenLayers + 1);
    let worst = 0;
    for (let i = 0; i < states.length; i += 1) {
      const d = compare(states[i]!, view(`output.hidden.${i}`));
      // Reported per layer, so a divergence names where it started rather than
      // only that the end is wrong.
      if (d.worst > worst) worst = d.worst;
      expect(d.worst, `hidden state ${i}`).toBeLessThan(2e-5);
    }
    console.log(`h3 ref2v text encoder: worst over ${states.length} hidden states ${worst.toExponential(3)}`);
  });

  it("puts each channel on the axis the interleave gives it", () => {
    // `[4, 2, 2]` over eight: height at 1 and 4, width at 2 and 5, everything
    // else time. The *chunked* reading — `[0..3] = t, [4,5] = h, [6,7] = w` —
    // is the plausible wrong answer and produces a working model.
    expect(mropeAxisOfChannel(16, [4, 2, 2])).toEqual([0, 1, 2, 0, 1, 2, 0, 0]);
    // The released geometry: `[24, 20, 20]` over 64.
    const axes = mropeAxisOfChannel(128, [24, 20, 20]);
    expect(axes.length).toBe(64);
    expect(axes.slice(0, 6)).toEqual([0, 1, 2, 0, 1, 2]);
    // Past `3 * 20`, everything is time again.
    expect(axes.slice(60)).toEqual([0, 0, 0, 0]);
    expect(axes.filter((a) => a === 0).length).toBe(24);
    expect(axes.filter((a) => a === 1).length).toBe(20);
  });

  it("keeps the frequency the global channel index gives it", () => {
    // "Preserving frequency continuity": channel `c` uses
    // `theta ** (-2c / headDim)` **whichever axis it reads**. A per-axis sweep
    // — which is what `ropeAxes` would compute on its own — gives a different
    // number for every channel past the first.
    const grid = { t: [1], h: [1], w: [1] };
    const got = mropePositions(grid, 16, [4, 2, 2], 10000);
    for (let c = 0; c < 8; c += 1) {
      // f32, because that is what `ropeAxes` reads — 6 digits, not 12.
      expect(got[c], `channel ${c}`).toBeCloseTo(Math.pow(10000, (-2 * c) / 16), 6);
    }
  });

  it("reads each channel's position from its own axis", () => {
    // Distinct values per axis, so a channel taking the wrong one is visible.
    const got = mropePositions({ t: [100], h: [10], w: [1] }, 16, [4, 2, 2], 1);
    // theta 1 makes every inverse frequency 1, so the position comes through
    // unscaled and the axis is all that is left.
    expect([...got]).toEqual([100, 10, 1, 100, 10, 1, 100, 100]);
  });

  it("pairs c with c + headDim/2, which is what rotate_half does", () => {
    expect(mropePermutation(8)).toEqual([0, 4, 1, 5, 2, 6, 3, 7]);
    const permuted = mropePermutation(16);
    // A permutation, not a map with a duplicate: a repeat would drop a channel
    // and copy another, and 15 of 16 would still be right.
    expect(new Set(permuted).size).toBe(16);
  });

  it("refuses a section list that does not describe the head", () => {
    expect(() => mropeAxisOfChannel(16, [4, 2, 3])).toThrow(/sum to 9, not 8/);
    // Sums to 8 and still ragged: the height loop would write 3 strided slots
    // and the width loop 1, leaving a boundary neither section describes.
    expect(() => mropeAxisOfChannel(16, [4, 3, 1])).toThrow(/equal height and width/);
    expect(() => mropePositions({ t: [1, 2], h: [1], w: [1] }, 16, [4, 2, 2], 10000))
      .toThrow(/not the same length/);
  });

  it("refuses a weight list that is not the configured depth", () => {
    expect(() => textEncoderHiddenStates(
      cfg, { ...weights, layers: weights.layers.slice(0, 1) }, view("input.embeds"), manifest.seq, positions,
    )).toThrow(/1 layers of weights for 3/);
  });
});
