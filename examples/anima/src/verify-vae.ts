/**
 * Runs Wan 2.1's VAE decoder on the shipped weights and compares it to the model.
 *
 * Issue #174. Checkpoints are compared **in order and it stops at the first
 * disagreement**: 108 tensors reported only at the image says that something is
 * wrong and nothing about where.
 *
 *     npx tsx examples/anima/src/verify-vae.ts --vae ~/anima-src/qwen_image_vae.safetensors
 */
import { readFileSync, writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { encodePng } from "../../zimage-vae/src/render.js";
import { SafetensorsFile } from "../../zimage/src/safetensors.js";
import { type VaeTrace, type VaeWeights, type WanVaeConfig, wanVaeDecode } from "./vae.js";
import { wanVaeDecodeGpu } from "./vae-gpu.js";
import { vaeKernels } from "./vae-kernels-node.js";
import { createRunner } from "../../../harness/wgsl.js";

interface Entry { name: string; shape: number[]; offset: number }
interface Golden {
  vae: string; latent: number; seed: number;
  dim: number; zDim: number; dimMult: number[]; numResBlocks: number;
  tensors: Entry[];
}

const at = process.argv.indexOf("--vae");
const vaePath = at >= 0 ? process.argv[at + 1] : process.env.ANIMA_VAE;
if (!vaePath) {
  console.error("verify-vae: pass --vae <qwen_image_vae.safetensors>");
  process.exit(2);
}

const raw = readFileSync(fileURLToPath(new URL("../fixtures/vae-golden.bin", import.meta.url)));
const headerLength = Number(raw.readBigUInt64LE(0));
const golden = JSON.parse(raw.subarray(8, 8 + headerLength).toString("utf8")) as Golden;
const base = 8 + headerLength;
const f32 = (name: string): Float32Array => {
  const e = golden.tensors.find((t) => t.name === name);
  if (!e) throw new Error(`golden has no tensor ${name}`);
  const count = e.shape.reduce((a, b) => a * b, 1);
  return new Float32Array(raw.buffer.slice(raw.byteOffset + base + e.offset, raw.byteOffset + base + e.offset + count * 4));
};

function stats(got: Float32Array, want: Float32Array): { absMax: number; relRms: number } {
  if (got.length !== want.length) throw new Error(`length ${got.length} against ${want.length}`);
  let absMax = 0, sqErr = 0, sqWant = 0;
  for (let i = 0; i < want.length; i += 1) {
    const d = got[i]! - want[i]!;
    absMax = Math.max(absMax, Math.abs(d));
    sqErr += d * d;
    sqWant += want[i]! * want[i]!;
  }
  return { absMax, relRms: Math.sqrt(sqErr) / Math.sqrt(sqWant) };
}

const file = new SafetensorsFile(vaePath);
const cache = new Map<string, Float32Array>();
const weights: VaeWeights = {
  // The decoder reads each weight once per call and the caller may decode more
  // than once; 108 tensors is small enough to keep and large enough that
  // re-reading them from disk on every call would dominate a 64x64 fixture.
  get(name) {
    const cached = cache.get(name);
    if (cached) return cached;
    const value = file.read(name);
    cache.set(name, value);
    return value;
  },
  has(name) {
    try {
      file.read(name);
      return true;
    } catch {
      return false;
    }
  },
};

const cfg: WanVaeConfig = {
  dim: golden.dim, zDim: golden.zDim, dimMult: golden.dimMult, numResBlocks: golden.numResBlocks,
};
console.log(`VAE: dim ${cfg.dim}, zDim ${cfg.zDim}, dimMult [${cfg.dimMult}]`);
console.log(`  ${golden.latent}x${golden.latent} latent -> ${golden.latent * 8}x${golden.latent * 8} image`);

// `--gpu` runs the same decomposition through `harness/wgsl.ts`. Both are
// compared against the same checkpoints: the reference is the definition of
// correctness and the GPU path is checked against the model, never against the
// reference — two kernels agreeing with each other says nothing.
const useGpu = process.argv.includes("--gpu");
const started = Date.now();
const trace: VaeTrace = {};
const progress = (label: string, done: number, total: number): void => {
  process.stdout.write(`\r  ${label.padEnd(22)} ${done}/${total}   `);
};
let image: Float32Array;
if (useGpu) {
  const runner = await createRunner();
  if (!runner) {
    console.error("verify-vae: --gpu given but no WebGPU adapter is available.");
    process.exit(2);
  }
  image = await wanVaeDecodeGpu(runner.run, vaeKernels(), cfg, weights, f32("z"), golden.latent, golden.latent, trace, progress);
  runner.destroy?.();
} else {
  image = wanVaeDecode(cfg, weights, f32("z"), golden.latent, golden.latent, trace, progress);
}
process.stdout.write("\n");
console.log(`  decoded in ${((Date.now() - started) / 1000).toFixed(1)}s (${useGpu ? "GPU" : "CPU"})\n`);

// In order. The upsample checkpoints are named by their index in the stack, so
// a disagreement points at a stage rather than at "the decoder".
const CHECKS = ["afterConv1", "afterAttention", "afterMiddle", "afterUpsample3", "afterUpsample7", "afterUpsample11", "image"];
let failed = false;
for (const name of CHECKS) {
  const got = name === "image" ? image : trace[name];
  if (!got) {
    console.log(`  ${name.padEnd(16)} MISSING from the port's trace`);
    failed = true;
    break;
  }
  const s = stats(got, f32(name));
  const ok = s.relRms <= 2e-5;
  console.log(`  ${name.padEnd(16)} rel-RMS ${s.relRms.toExponential(3)}  abs max ${s.absMax.toExponential(3)}  ${ok ? "ok" : "MISMATCH"}`);
  if (!ok) {
    failed = true;
    console.log(`\nFirst mismatch at ${name}. Everything after it is downstream of the same cause.`);
    break;
  }
}

if (!failed) {
  const out = fileURLToPath(new URL("../fixtures/vae-check.png", import.meta.url));
  writeFileSync(out, encodePng(image, golden.latent * 8, golden.latent * 8));
  console.log(`\nOK: the decoder matches the model. Wrote ${out}`);
  console.log("The image is noise — the golden decodes a random latent, not a sampled one.");
}
process.exit(failed ? 1 : 0);
