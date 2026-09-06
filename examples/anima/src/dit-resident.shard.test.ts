/**
 * `animaForwardResident` over a range of blocks — the forward split across
 * devices (issue #225).
 *
 * The reference for a split forward is the unsplit one on the same device:
 * `verify-forward-gpu.ts` holds *that* to ComfyUI's golden with the real
 * weights, so what is left to show here is that cutting the block loop and
 * carrying `x` through a readback and an upload changes nothing. Nothing is
 * the bar — the carried tensor is f32 both ways and every kernel sees the same
 * bytes — so the assertion is equality, not a tolerance. A tolerance here
 * would be the place a real drift went to hide.
 *
 * The model is synthetic and small (4 blocks, 256 channels, 16 tokens) so the
 * test runs in the suite; its shapes follow the real checkpoint's rules
 * (`headDim = 128`, the padding-mask channel, the LoRA'd adaLN) so the same
 * code paths run.
 */
import { afterAll, beforeAll, describe, expect } from "vitest";
import { residentTest, useResidentGpu } from "../../../harness/suite.js";
import { ditKernels } from "../../zimage/src/kernels-node.js";
import type { AnimaConfig, AnimaInput } from "./dit.js";
import { type AnimaShard, type AnimaWeightSource, animaForwardResident, releaseAnimaWeights } from "./dit-resident.js";
import { gaussianNoise } from "./sampler.js";

const cfg: AnimaConfig = {
  numBlocks: 4, modelChannels: 256, numHeads: 2, adalnLoraDim: 32,
  inChannels: 16, outChannels: 16, patchSpatial: 2, patchTemporal: 1,
  crossattnEmbChannels: 64, concatPaddingMask: true, maxPeriod: 10000, normEps: 1e-6,
  ropeExtrapolation: { t: 1, h: 4, w: 4 },
};
const dim = cfg.modelChannels;
const headDim = dim / cfg.numHeads;
const mlpHidden = 2 * dim;
const patchDim = cfg.patchTemporal * cfg.patchSpatial * cfg.patchSpatial * (cfg.inChannels + 1);
const outPatchDim = cfg.patchTemporal * cfg.patchSpatial * cfg.patchSpatial * cfg.outChannels;
const H = 8, W = 8, T = 1;
const seq = (H / cfg.patchSpatial) * (W / cfg.patchSpatial);
const contextSeq = 8;

/** `[out, in]` shapes for every tensor the forward reads, by name. */
function shapes(): Map<string, number[]> {
  const s = new Map<string, number[]>();
  s.set("net.t_embedder.1.linear_1.weight", [dim, dim]);
  s.set("net.t_embedder.1.linear_2.weight", [3 * dim, dim]);
  s.set("net.t_embedding_norm.weight", [dim]);
  s.set("net.x_embedder.proj.1.weight", [dim, patchDim]);
  for (let b = 0; b < cfg.numBlocks; b += 1) {
    const p = `net.blocks.${b}.`;
    for (const attn of ["self_attn", "cross_attn"]) {
      const kv = attn === "self_attn" ? dim : cfg.crossattnEmbChannels;
      s.set(`${p}${attn}.q_proj.weight`, [dim, dim]);
      s.set(`${p}${attn}.k_proj.weight`, [dim, kv]);
      s.set(`${p}${attn}.v_proj.weight`, [dim, kv]);
      s.set(`${p}${attn}.output_proj.weight`, [dim, dim]);
      s.set(`${p}${attn}.q_norm.weight`, [headDim]);
      s.set(`${p}${attn}.k_norm.weight`, [headDim]);
    }
    s.set(`${p}mlp.layer1.weight`, [mlpHidden, dim]);
    s.set(`${p}mlp.layer2.weight`, [dim, mlpHidden]);
    for (const which of ["self_attn", "cross_attn", "mlp"]) {
      s.set(`${p}adaln_modulation_${which}.1.weight`, [cfg.adalnLoraDim, dim]);
      s.set(`${p}adaln_modulation_${which}.2.weight`, [3 * dim, cfg.adalnLoraDim]);
    }
  }
  s.set("net.final_layer.adaln_modulation.1.weight", [cfg.adalnLoraDim, dim]);
  s.set("net.final_layer.adaln_modulation.2.weight", [2 * dim, cfg.adalnLoraDim]);
  s.set("net.final_layer.linear.weight", [outPatchDim, dim]);
  return s;
}

/** Deterministic weights: norms near one, projections small, so activations stay O(1) across four blocks. */
function syntheticWeights(): AnimaWeightSource {
  const table = shapes();
  const cache = new Map<string, Float32Array>();
  let seed = 1;
  const get = (name: string): Float32Array => {
    const hit = cache.get(name);
    if (hit) return hit;
    const shape = table.get(name);
    if (!shape) throw new Error(`synthetic model has no ${name}`);
    const length = shape.reduce((a, b) => a * b, 1);
    const isNorm = /norm\.weight$/.test(name);
    const scale = isNorm ? 0.05 : 1 / Math.sqrt(shape[shape.length - 1]!);
    const values = gaussianNoise(length, (seed += 1)).map((v) => (isNorm ? 1 + v * scale : v * scale));
    cache.set(name, values);
    return values;
  };
  return { get, has: (name) => table.has(name), shapeOf: (name) => table.get(name), packedQ8: () => null };
}

const weights = syntheticWeights();
const input: AnimaInput = {
  latent: gaussianNoise(cfg.inChannels * T * H * W, 100), T, H, W, t: 0.7,
  context: gaussianNoise(contextSeq * cfg.crossattnEmbChannels, 101),
};

describe("animaForwardResident over a range of blocks", () => {
  useResidentGpu();
  const held = new Map<string, GPUBuffer>();
  const K = ditKernels();
  let full: Float32Array;

  beforeAll(() => { /* the device comes from useResidentGpu */ });
  afterAll(() => releaseAnimaWeights(held));

  const run = (device: Parameters<typeof animaForwardResident>[0], shard?: AnimaShard) =>
    animaForwardResident(device, K, cfg, weights, input, undefined, held, undefined, undefined, undefined, undefined, shard);

  residentTest("the full range is the unsharded forward, bit for bit", async (device) => {
    full = await run(device);
    const ranged = await run(device, { from: 0, to: cfg.numBlocks });
    expect(full.length).toBe(cfg.outChannels * T * H * W);
    expect(Array.from(ranged)).toEqual(Array.from(full));
  });

  residentTest("two shards carrying x through f32 reproduce the full forward exactly", async (device) => {
    const x = await run(device, { from: 0, to: 2 });
    expect(x.length).toBe(seq * dim);
    const out = await run(device, { from: 2, to: cfg.numBlocks, activation: x });
    expect(Array.from(out)).toEqual(Array.from(full));
  });

  residentTest("one block per shard reproduces it too", async (device) => {
    let x: Float32Array | undefined;
    for (let b = 0; b < cfg.numBlocks; b += 1) {
      x = await run(device, { from: b, to: b + 1, activation: x });
    }
    expect(Array.from(x!)).toEqual(Array.from(full));
  });

  residentTest("a middle shard's output is what the next shard reads, not the last block's", async (device) => {
    // Observability: a shard that ran too many blocks, or too few, still returns
    // `[seq, dim]` floats. Only the numbers say which blocks ran.
    const after2 = await run(device, { from: 0, to: 2 });
    const after3 = await run(device, { from: 2, to: 3, activation: after2 });
    const after3direct = await run(device, { from: 0, to: 3 });
    expect(Array.from(after3)).toEqual(Array.from(after3direct));
    expect(Array.from(after3)).not.toEqual(Array.from(after2));
  });

  residentTest("refuses a shard it cannot run, naming what is wrong", async (device) => {
    await expect(run(device, { from: 1, to: 2 })).rejects.toThrow(/activation/);
    await expect(run(device, { from: 1, to: 2, activation: new Float32Array(seq * dim - 1) })).rejects.toThrow(/length/);
    await expect(run(device, { from: 0, to: 1, activation: new Float32Array(seq * dim) })).rejects.toThrow(/from: 0/);
    await expect(run(device, { from: 2, to: 2 })).rejects.toThrow(/from < to/);
    await expect(run(device, { from: 0, to: cfg.numBlocks + 1 })).rejects.toThrow(/numBlocks/);
  });
});
