/**
 * Where one Anima forward's GPU time goes, by kernel, at a chosen size.
 *
 * The browser's own profile said `rows` — a row-broadcast add or multiply —
 * costs 40.6% of a forward against `matmulQ8`'s 44.5%, and that `permute`
 * moves comparable data 200x faster per dispatch. Either the kernel is that
 * slow or the measurement is, and neither can be settled from the browser: the
 * page does not say what shape each dispatch had.
 *
 * This runs the same forward in Node with the shape known and printed.
 *
 * **Profiling is not free and the numbers are not the shipping cost.**
 * `GPUComputePassTimestampWrites` allows one timestamp pair per pass, so every
 * dispatch gets its own pass — 3,290 of them against 55 submits' worth
 * unprofiled. Read the shares. The unprofiled wall clock is printed beside
 * them so the inflation is visible rather than assumed.
 *
 *     npx tsx examples/anima/src/profile-forward.ts --dit ~/anima-q8 --width 832 --height 1216
 */
import { readFileSync } from "node:fs";
import { createResidentDevice } from "../../../harness/resident.js";
import { ditKernels } from "../../zimage/src/kernels-node.js";
import type { AnimaConfig } from "./dit.js";
import { type AnimaProfile, animaForwardResident, releaseAnimaWeights } from "./dit-resident.js";
import { LATENT } from "./sampler.js";
import { loadAnimaSubset, withRopePermutation } from "./weights-node.js";

const arg = (name: string): string | undefined => {
  const at = process.argv.indexOf(name);
  return at >= 0 ? process.argv[at + 1] : undefined;
};
const ditDir = arg("--dit") ?? process.env.ANIMA_DIT_DIR;
if (!ditDir) {
  console.error("profile-forward: pass --dit <dir> [--width N] [--height N]");
  process.exit(2);
}
const width = Number(arg("--width") ?? 832);
const height = Number(arg("--height") ?? 1216);
const latentH = height / 8;
const latentW = width / 8;

const manifest = JSON.parse(readFileSync(`${ditDir}/dit.manifest.json`, "utf8")) as {
  config: Record<string, number | boolean>;
};
const c = manifest.config as unknown as {
  num_blocks: number; model_channels: number; num_heads: number; adaln_lora_dim: number;
  in_channels: number; out_channels: number; patch_spatial: number; patch_temporal: number;
  crossattn_emb_channels: number; concat_padding_mask: boolean;
  rope_t_extrapolation_ratio: number; rope_h_extrapolation_ratio: number; rope_w_extrapolation_ratio: number;
};
const cfg: AnimaConfig = {
  numBlocks: c.num_blocks, modelChannels: c.model_channels, numHeads: c.num_heads,
  adalnLoraDim: c.adaln_lora_dim, inChannels: c.in_channels, outChannels: c.out_channels,
  patchSpatial: c.patch_spatial, patchTemporal: c.patch_temporal,
  crossattnEmbChannels: c.crossattn_emb_channels, concatPaddingMask: c.concat_padding_mask,
  maxPeriod: 10000, normEps: 1e-6,
  ropeExtrapolation: {
    t: c.rope_t_extrapolation_ratio, h: c.rope_h_extrapolation_ratio, w: c.rope_w_extrapolation_ratio,
  },
};

const seq = (latentH / cfg.patchSpatial) * (latentW / cfg.patchSpatial);
console.log(`${width}x${height} — a ${latentW}x${latentH} latent, ${seq} tokens, ${cfg.numBlocks} blocks at ${cfg.modelChannels}`);

const permuting = withRopePermutation(
  loadAnimaSubset(ditDir), cfg.numHeads, cfg.modelChannels / cfg.numHeads, cfg.modelChannels,
);

/**
 * Which weights take the dense path, and why.
 *
 * `packedQ8` returning null is what sends a projection through `matmul` instead
 * of `matmulQ8`, and there are two unrelated reasons it can: the tensor was
 * never quantized (`should_quantize` wants both axes >= 128), or this wrapper
 * refuses the packed form because the weight has been permuted for rope. The
 * second is self-inflicted and worth separating from the first.
 */
const dense = new Map<string, number>();
const dit = {
  has: (name: string) => permuting.has(name),
  shapeOf: (name: string) => permuting.shapeOf(name),
  get: (name: string) => permuting.get(name),
  packedQ8: (name: string) => {
    const packed = permuting.packedQ8(name);
    if (packed === null) dense.set(name, (dense.get(name) ?? 0) + 1);
    return packed;
  },
};
const device = await createResidentDevice();
if (!device) {
  console.error("profile-forward: no WebGPU adapter available.");
  process.exit(2);
}
const held = new Map<string, GPUBuffer>();
const input = {
  latent: new Float32Array(LATENT.channels * latentH * latentW),
  T: 1, H: latentH, W: latentW, t: 0.7,
  context: new Float32Array(512 * cfg.crossattnEmbChannels),
};

// The first forward uploads the weights; the second is the steady state, and
// the third is the one profiled — so the profile measures neither the upload
// nor a cold cache.
console.log("\nuploading …");
await animaForwardResident(device, ditKernels(), cfg, dit, input, undefined, held);

const warmStart = Date.now();
await animaForwardResident(device, ditKernels(), cfg, dit, input, undefined, held);
const warmSeconds = (Date.now() - warmStart) / 1000;
console.log(`unprofiled forward: ${warmSeconds.toFixed(2)}s\n`);

const profile: AnimaProfile = { byKernel: new Map(), supported: device.timestampsSupported, submitToDoneMs: 0 };
const stats = { dispatches: 0, submits: 0, poolSlots: 0, poolBytes: 0, weightBuffers: 0, uploadedBytes: 0 };
const profStart = Date.now();
await animaForwardResident(device, ditKernels(), cfg, dit, input, stats, held, undefined, undefined, undefined, profile);
const dispatchCount = stats.dispatches;
const profSeconds = (Date.now() - profStart) / 1000;

releaseAnimaWeights(held);
await device.destroy();

if (!profile.supported) {
  console.error("This device has no timestamp-query, so there is no GPU breakdown — not a breakdown of zeros.");
  process.exit(1);
}

const rows = [...profile.byKernel.entries()].sort((a, b) => b[1].seconds - a[1].seconds);
const total = rows.reduce((sum, [, v]) => sum + v.seconds, 0);
const counted = rows.reduce((n, [, v]) => n + v.dispatches, 0);
console.log(`profiled forward: ${profSeconds.toFixed(2)}s wall, ${(total * 1000).toFixed(0)}ms of timed GPU`);
console.log(`  profiling inflates it ${(profSeconds / warmSeconds).toFixed(1)}x — one compute pass per dispatch\n`);
// The first profile timed 6.05 s of an 8.27 s forward and said nothing about
// the missing 2.2 s: `chunkedFlat` and the attention loop push their dispatches
// straight into `ops`, bypassing the labelling in `record`. Attention was
// invisible entirely. A share of a partial total is a wrong number, so the
// coverage is stated rather than assumed.
console.log(`  ${counted} of ${dispatchCount} dispatches timed, ` +
  `${(total / warmSeconds * 100).toFixed(0)}% of the unprofiled forward's wall clock accounted for\n`);
const permuted = [...dense.keys()].filter((n) => /\.self_attn\.(q|k)_proj\.weight$/.test(n));
const unquantized = [...dense.keys()].filter((n) => !/\.self_attn\.(q|k)_proj\.weight$/.test(n));
console.log(`  ${dense.size} weights take the dense path: ${permuted.length} refused by the rope ` +
  `permutation, ${unquantized.length} never quantized (${unquantized.join(", ") || "none"})\n`);
console.log("  kernel        total ms    share   dispatches   ms each");
for (const [name, v] of rows) {
  console.log(
    `  ${name.padEnd(12)} ${(v.seconds * 1000).toFixed(1).padStart(9)} ${((v.seconds / total) * 100).toFixed(1).padStart(7)}% ` +
      `${String(v.dispatches).padStart(11)} ${((v.seconds / v.dispatches) * 1000).toFixed(3).padStart(9)}`,
  );
}
