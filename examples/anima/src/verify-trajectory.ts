/**
 * Runs Anima's sampling loop against the same loop in torch, step by step.
 *
 * Issue #170 stage 5. Every piece was already checked on its own — ids exactly,
 * conditioning at 9.775e-7, the DiT at 8.3e-6 with a 512-token padded context,
 * the stepper at 2e-6 on a toy denoiser — and the pipeline still produced a flat
 * latent. That is what this exists for: parts that are each right can be joined
 * wrongly, and no test of a part will say so.
 *
 * The golden supplies **its own** noise and **its own** conditioning, so this
 * compares the loop and nothing else. A run that starts from different noise is
 * a different image and tells you nothing about whether the loop agrees.
 *
 *     npx tsx examples/anima/src/verify-trajectory.ts --dit ~/anima-q8
 */
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { createResidentDevice } from "../../../harness/resident.js";
import { ditKernels } from "../../zimage/src/kernels-node.js";
import type { AnimaConfig } from "./dit.js";
import { animaForwardResident, releaseAnimaWeights } from "./dit-resident.js";
import { LATENT, applyCfg, calculateDenoised, resMultistep, timestepOf } from "./sampler.js";
import { loadAnimaSubset, withRopePermutation } from "./weights-node.js";

interface Entry { name: string; shape: number[]; offset: number }
interface Golden {
  prompt: string; negative: string; steps: number; cfg: number;
  latent: number; seed: number; channels: number; finalSpread: number;
  contextLength: number; positiveRows: number; negativeRows: number;
  tensors: Entry[];
}

const at = process.argv.indexOf("--dit");
const ditDir = at >= 0 ? process.argv[at + 1] : process.env.ANIMA_DIT_DIR;
if (!ditDir) {
  console.error("verify-trajectory: pass --dit <convert_dit.py output dir>");
  process.exit(2);
}

const raw = readFileSync(fileURLToPath(new URL("../fixtures/trajectory-golden.bin", import.meta.url)));
const headerLength = Number(raw.readBigUInt64LE(0));
const golden = JSON.parse(raw.subarray(8, 8 + headerLength).toString("utf8")) as Golden;
const base = 8 + headerLength;
const f32 = (name: string): Float32Array => {
  const e = golden.tensors.find((t) => t.name === name);
  if (!e) throw new Error(`golden has no tensor ${name}`);
  const count = e.shape.reduce((a, b) => a * b, 1);
  return new Float32Array(raw.buffer.slice(raw.byteOffset + base + e.offset, raw.byteOffset + base + e.offset + count * 4));
};

/** Median per-channel spatial standard deviation — the number that showed the bug. */
function spread(latent: Float32Array, channels = LATENT.channels): number {
  const per = latent.length / channels;
  const stds: number[] = [];
  for (let ch = 0; ch < channels; ch += 1) {
    let sum = 0, sq = 0;
    for (let i = 0; i < per; i += 1) { const v = latent[ch * per + i]!; sum += v; sq += v * v; }
    stds.push(Math.sqrt(Math.max(0, sq / per - (sum / per) ** 2)));
  }
  stds.sort((a, b) => a - b);
  return stds[stds.length >> 1]!;
}

function relRms(got: Float32Array, want: Float32Array): number {
  let sqErr = 0, sqWant = 0;
  for (let i = 0; i < want.length; i += 1) {
    const d = got[i]! - want[i]!;
    sqErr += d * d;
    sqWant += want[i]! * want[i]!;
  }
  return Math.sqrt(sqErr) / Math.sqrt(sqWant);
}

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

// Wrapped, not raw. Passing the source straight through is what made this
// script's first run read 1.068e-1 against torch while the DiT itself was at
// 6.5e-6 — the self-attention rope permutation lives in `withRopePermutation`
// now precisely so no caller has to remember it.
const dit = withRopePermutation(
  loadAnimaSubset(ditDir),
  cfg.numHeads,
  cfg.modelChannels / cfg.numHeads,
  cfg.modelChannels,
);
const device = await createResidentDevice();
if (!device) {
  console.error("no adapter");
  process.exit(2);
}
const held = new Map<string, GPUBuffer>();

const sigmas = Array.from(f32("sigmas"));
/**
 * The golden stores only the adapter's real rows; the pad to 512 is zeros by
 * construction and would be four megabytes of them in a committed fixture.
 * Reconstructed rather than stored — and the reconstruction is checked, because
 * a wrong `contextLength` would change what the model attends over.
 */
const padded = (name: string, rows: number): Float32Array => {
  const trimmed = f32(name);
  const dim = cfg.crossattnEmbChannels;
  if (trimmed.length !== rows * dim) {
    throw new Error(`golden's ${name} is ${trimmed.length} values, expected ${rows} rows of ${dim}`);
  }
  const out = new Float32Array(golden.contextLength * dim);
  out.set(trimmed, 0);
  return out;
};
const positive = padded("positive", golden.positiveRows);
const negative = golden.cfg > 1.0 ? padded("negative", golden.negativeRows) : null;
const x0 = f32("x0");
const shape = { T: 1, H: golden.latent, W: golden.latent };

console.log(`prompt: ${JSON.stringify(golden.prompt)}`);
console.log(`  ${sigmas.length - 1} steps, ${golden.latent}x${golden.latent} latent, CFG ${golden.cfg}`);
console.log(`  context ${positive.length / cfg.crossattnEmbChannels} rows, torch's own`);
console.log(`  x0 spread ${spread(x0).toFixed(3)} (golden ${spread(f32("x0")).toFixed(3)})\n`);

const predictions: Float32Array[] = [];
let out = x0;
let failed = false;

for (let step = 0; step < sigmas.length - 1 && !failed; step += 1) {
  const sigma = sigmas[step]!;
  const t = timestepOf(sigma);
  const forward = (context: Float32Array): Promise<Float32Array> =>
    animaForwardResident(device, ditKernels(), cfg, dit, { latent: out, ...shape, t, context }, undefined, held);

  const cond = await forward(positive);
  const uncond = negative ? await forward(negative) : null;
  if (step === 0) {
    // The two model calls before they are combined. A CFG'd prediction that
    // disagrees could have a wrong model call or a wrong combination, and one
    // number cannot say which — so both are reported before either is used.
    console.log(`  raw cond   ${relRms(cond, f32("rawCond0")).toExponential(3)}`);
    if (uncond) {
      console.log(`  raw uncond ${relRms(uncond, f32("rawUncond0")).toExponential(3)}`);
      const gotDiff = cond.map((v, i) => v - uncond[i]!);
      const wantDiff = f32("rawCond0").map((v, i) => v - f32("rawUncond0")[i]!);
      // What CFG multiplies by 8. Two outputs can each agree to 1e-5 while
      // their difference — a thousandth of their size — agrees to nothing.
      console.log(
        `  cond - uncond ${relRms(gotDiff, wantDiff).toExponential(3)}` +
          `  (magnitude ${Math.sqrt(gotDiff.reduce((a, v) => a + v * v, 0) / gotDiff.length).toExponential(2)}` +
          ` vs ${Math.sqrt(wantDiff.reduce((a, v) => a + v * v, 0) / wantDiff.length).toExponential(2)})`,
      );
    }
  }
  const prediction = uncond ? applyCfg(cond, uncond, golden.cfg) : cond;
  const denoised = calculateDenoised(sigma, prediction, out);
  predictions.push(denoised);

  const wantDenoised = f32(`denoised${step}`);
  let cursor = 0;
  out = resMultistep(() => predictions[cursor++]!, x0, sigmas.slice(0, step + 2));
  const wantX = f32(`x${step + 1}`);

  const dRel = relRms(denoised, wantDenoised);
  const xRel = relRms(out, wantX);
  // The denoised prediction and the latent are reported separately on purpose:
  // the first is the model plus CFG, the second adds the stepper. A run where
  // only the second disagrees is a stepper bug; a run where both do is not.
  console.log(
    `  step ${String(step + 1).padStart(2)}/${sigmas.length - 1}  sigma ${sigma.toFixed(4)}  ` +
      `denoised ${dRel.toExponential(3)} (spread ${spread(denoised).toFixed(3)} vs ${spread(wantDenoised).toFixed(3)})  ` +
      `x ${xRel.toExponential(3)} (spread ${spread(out).toFixed(3)} vs ${spread(wantX).toFixed(3)})`,
  );
  if (dRel > 5e-2 || xRel > 5e-2) {
    console.log(`\nFirst disagreement at step ${step + 1}. Everything after it is downstream of the same cause.`);
    failed = true;
  }
}

releaseAnimaWeights(held);
await device.destroy();

if (!failed) {
  const rel = relRms(out, f32("final"));
  console.log(`\nfinal rel-RMS ${rel.toExponential(3)}, spread ${spread(out).toFixed(4)} vs ${golden.finalSpread.toFixed(4)}`);
  console.log("\nOK: the sampling loop matches torch's, step by step.");
}
process.exit(failed ? 1 : 0);
