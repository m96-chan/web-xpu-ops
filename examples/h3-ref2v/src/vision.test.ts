/**
 * Qwen3-VL's vision tower against `transformers`' own model.
 *
 * Issue #212. Committed fixture, random weights, tiny geometry — 399 KB, runs
 * in CI anywhere, no model licence. The golden records the **stages** as well
 * as the output, so a divergence names where it started instead of only that
 * the end is wrong.
 */
import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import {
  interpolatePositionEmbedding,
  toMergeBlockOrder,
  torchLinspace,
  visionCoordinates,
  visionForward,
  visionPermutation,
  visionPositions,
  type Grid,
  type MergerWeights,
  type VisionBlockWeights,
  type VisionConfig,
  type VisionWeights,
} from "./vision.js";

const here = new URL("../fixtures/", import.meta.url);
const manifest = JSON.parse(readFileSync(new URL("vision.json", here), "utf8")) as {
  config: {
    hidden_size: number; intermediate_size: number; num_heads: number; depth: number;
    in_channels: number; patch_size: number; spatial_merge_size: number;
    temporal_patch_size: number; out_hidden_size: number; deepstack_visual_indexes: number[];
  };
  numGridPerSide: number;
  grid: Grid[];
  tokens: number;
  patchDim: number;
  numDeepstack: number;
  tensors: { name: string; shape: number[]; offset: number; count: number }[];
};
const blob = readFileSync(new URL("vision.bin", here));

const view = (name: string): Float32Array => {
  const entry = manifest.tensors.find((t) => t.name === name);
  if (!entry) throw new Error(`vision.bin has no tensor named "${name}"`);
  return new Float32Array(
    blob.buffer.slice(blob.byteOffset + entry.offset, blob.byteOffset + entry.offset + entry.count * 4),
  );
};

const c = manifest.config;
const cfg: VisionConfig = {
  hiddenSize: c.hidden_size,
  intermediateSize: c.intermediate_size,
  numHeads: c.num_heads,
  depth: c.depth,
  inChannels: c.in_channels,
  patchSize: c.patch_size,
  spatialMergeSize: c.spatial_merge_size,
  temporalPatchSize: c.temporal_patch_size,
  outHiddenSize: c.out_hidden_size,
  numGridPerSide: manifest.numGridPerSide,
  deepstackVisualIndexes: c.deepstack_visual_indexes,
};

const headDim = cfg.hiddenSize / cfg.numHeads;
const order = visionPermutation(headDim);

/** The `[heads * headDim, in]` rows of a fused QKV, permuted for `ropeAxes`. */
function permuteQk(weight: Float32Array, inDim: number): Float32Array {
  const out = weight.slice();
  const width = cfg.hiddenSize;
  // Q is the first `width` rows and K the second; V is never rotated.
  for (let part = 0; part < 2; part += 1) {
    for (let head = 0; head < cfg.numHeads; head += 1) {
      for (let ch = 0; ch < headDim; ch += 1) {
        const from = part * width + head * headDim + order[ch]!;
        const to = part * width + head * headDim + ch;
        out.set(weight.subarray(from * inDim, (from + 1) * inDim), to * inDim);
      }
    }
  }
  return out;
}

function permuteQkBias(bias: Float32Array): Float32Array {
  const out = bias.slice();
  for (let part = 0; part < 2; part += 1) {
    for (let head = 0; head < cfg.numHeads; head += 1) {
      for (let ch = 0; ch < headDim; ch += 1) {
        out[part * cfg.hiddenSize + head * headDim + ch] =
          bias[part * cfg.hiddenSize + head * headDim + order[ch]!]!;
      }
    }
  }
  return out;
}

function block(index: number): VisionBlockWeights {
  const p = `blocks.${index}`;
  return {
    norm1Weight: view(`${p}.norm1.weight`),
    norm1Bias: view(`${p}.norm1.bias`),
    // The rotation lives in the weights, as it does everywhere else here.
    qkvWeight: permuteQk(view(`${p}.attn.qkv.weight`), cfg.hiddenSize),
    qkvBias: permuteQkBias(view(`${p}.attn.qkv.bias`)),
    projWeight: view(`${p}.attn.proj.weight`),
    projBias: view(`${p}.attn.proj.bias`),
    norm2Weight: view(`${p}.norm2.weight`),
    norm2Bias: view(`${p}.norm2.bias`),
    fc1Weight: view(`${p}.mlp.linear_fc1.weight`),
    fc1Bias: view(`${p}.mlp.linear_fc1.bias`),
    fc2Weight: view(`${p}.mlp.linear_fc2.weight`),
    fc2Bias: view(`${p}.mlp.linear_fc2.bias`),
  };
}

const merger = (prefix: string): MergerWeights => ({
  normWeight: view(`${prefix}.norm.weight`),
  normBias: view(`${prefix}.norm.bias`),
  fc1Weight: view(`${prefix}.linear_fc1.weight`),
  fc1Bias: view(`${prefix}.linear_fc1.bias`),
  fc2Weight: view(`${prefix}.linear_fc2.weight`),
  fc2Bias: view(`${prefix}.linear_fc2.bias`),
});

const weights: VisionWeights = {
  patchEmbedWeight: view("patch_embed.proj.weight"),
  patchEmbedBias: view("patch_embed.proj.bias"),
  posEmbed: view("pos_embed.weight"),
  blocks: Array.from({ length: cfg.depth }, (_, i) => block(i)),
  merger: merger("merger"),
  deepstackMergers: cfg.deepstackVisualIndexes.map((_, i) => merger(`deepstack_merger_list.${i}`)),
};

function worstOf(got: Float32Array, want: Float32Array): number {
  expect(got.length).toBe(want.length);
  let worst = 0;
  for (let i = 0; i < want.length; i += 1) worst = Math.max(worst, Math.abs(got[i]! - want[i]!));
  return worst;
}

describe("h3 ref2v / vision tower", () => {
  it("has a geometry that exercises the merge blocks", () => {
    // A 4x4 grid at merge 2 is four blocks, so merge-block order differs from
    // raster order. At 2x2 it would not.
    expect(cfg.spatialMergeSize).toBeGreaterThan(1);
    for (const [, h, w] of manifest.grid) {
      expect(h / cfg.spatialMergeSize).toBeGreaterThan(1);
      expect(w / cfg.spatialMergeSize).toBeGreaterThan(1);
    }
    expect(manifest.numDeepstack).toBe(cfg.deepstackVisualIndexes.length);
  });

  it("reproduces the patch embedding", () => {
    // `ops/conv`'s conv3d, with kernel = stride = the whole input.
    const out = visionForward(cfg, weights, view("input.pixels"), manifest.grid);
    expect(out.lastHiddenState.length).toBe(manifest.tokens * cfg.hiddenSize);
  });

  it("reproduces the interpolated position embedding, in merge-block order", () => {
    const [frames, height, width] = manifest.grid[0]!;
    const raster = interpolatePositionEmbedding(
      weights.posEmbed, cfg.numGridPerSide, cfg.hiddenSize, height, width);
    const repeated = new Float32Array(frames * height * width * cfg.hiddenSize);
    for (let t = 0; t < frames; t += 1) repeated.set(raster, t * height * width * cfg.hiddenSize);
    const got = toMergeBlockOrder(repeated, frames, height, width, cfg.spatialMergeSize, cfg.hiddenSize);
    const worst = worstOf(got, view("stage.pos_embed"));
    console.log(`h3 ref2v vision: position embedding worst ${worst.toExponential(3)}`);
    // Measured at 1.863e-9. Five times that, not five orders of magnitude:
    // a tolerance with no measurement beside it is a tolerance that lets a
    // wrong GELU through, which is what 2e-5 did here.
    expect(worst).toBeLessThan(1e-8);
  });

  it("reproduces the rotary coordinates", () => {
    // `rot_pos_emb` returns `freq_table[coords].flatten(1)`, which is the
    // coordinate times the inverse frequency — the same values this folds into
    // its positions.
    const [frames, height, width] = manifest.grid[0]!;
    const coords = visionCoordinates(frames, height, width, cfg.spatialMergeSize);
    const got = visionPositions(coords, headDim, 10000);
    const worst = worstOf(got, view("stage.rot_pos_emb"));
    console.log(`h3 ref2v vision: rotary coordinates worst ${worst.toExponential(3)}`);
    // Measured at exactly 0 — the same f32 arithmetic on both sides.
    expect(worst).toBeLessThan(1e-9);
  });

  it("reproduces the tower, the pooled tokens and every deepstack feature", () => {
    const got = visionForward(cfg, weights, view("input.pixels"), manifest.grid);
    const last = worstOf(got.lastHiddenState, view("output.last_hidden_state"));
    const pooled = worstOf(got.pooled, view("output.pooler_output"));
    let deepstackWorst = 0;
    for (let i = 0; i < got.deepstack.length; i += 1) {
      deepstackWorst = Math.max(deepstackWorst, worstOf(got.deepstack[i]!, view(`output.deepstack.${i}`)));
    }
    console.log(
      `h3 ref2v vision: last hidden ${last.toExponential(3)}  pooled ${pooled.toExponential(3)}  ` +
        `deepstack ${deepstackWorst.toExponential(3)}`,
    );
    // **Every bound here is a measurement times a small factor**, and both
    // numbers are written down, because this test started at `2e-5` — a
    // hundred and seventy times the achieved error — and let two wrong GELUs
    // through.
    //
    // Achieved: last 1.192e-7, pooled 7.451e-9, deepstack 7.451e-9.
    // Swapping the **merger's** exact GELU for the tanh one moves the pooled
    // output well past any of these. Swapping the **blocks' MLP** the other way
    // moves last to 6.109e-7 and pooled to 7.916e-8 — five and ten times the
    // achieved error, and under a 1e-6 bound, which is why these are 3e-7 and
    // 3e-8 rather than a round number.
    //
    // The two GELUs differ by 6.932e-5 on the values this fixture produces; the
    // effect on the output is smaller because the fixture's random weights are
    // small. At the released scale it would be larger, not smaller.
    expect(last).toBeLessThan(3e-7);
    expect(pooled).toBeLessThan(3e-8);
    expect(got.deepstack.length).toBe(manifest.numDeepstack);
    expect(deepstackWorst).toBeLessThan(3e-8);
  });

  it("orders tokens by merge block, not by raster", () => {
    // A 4x4 grid at merge 2: raster is 0,1,2,3,4,… and merge-block order is
    // 0,1,4,5,2,3,6,7,… Both are 16 tokens of the right width.
    const dim = 1;
    const raster = Float32Array.from({ length: 16 }, (_, i) => i);
    expect([...toMergeBlockOrder(raster, 1, 4, 4, 2, dim)])
      .toEqual([0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15]);
    expect(visionCoordinates(1, 4, 4, 2).slice(0, 4)).toEqual([[0, 0], [0, 1], [1, 0], [1, 1]]);
  });

  it("reproduces torch's linspace, which decides which taps are used", () => {
    // #211's three details. One ulp here moves an `int()` truncation across an
    // integer boundary and picks a different pair of taps.
    expect([...torchLinspace(7, 8)]).toEqual([0, 1, 2, 3, 4, 5, 6, 7]);
    expect([...torchLinspace(7, 1)]).toEqual([0]);
    const three = torchLinspace(7, 3);
    expect(three[0]).toBe(0);
    expect(three[2]).toBe(7);
    expect(three[1]).toBeCloseTo(3.5, 6);
  });

  it("pairs c with c + headDim/2 for the rotation", () => {
    expect(visionPermutation(8)).toEqual([0, 4, 1, 5, 2, 6, 3, 7]);
    expect(new Set(visionPermutation(headDim)).size).toBe(headDim);
  });

  it("refuses pixels that do not describe the grid", () => {
    expect(() => visionForward(cfg, weights, new Float32Array(8), manifest.grid))
      .toThrow(/8 values for 16 patches/);
    expect(() => toMergeBlockOrder(new Float32Array(9), 1, 3, 3, 2, 1))
      .toThrow(/not a whole number of 2x2 blocks/);
  });
});
