/**
 * The resident DiT, against the same golden the CPU port is held to.
 *
 * Issue #170 stage 3. Compared against the model, not against the CPU port —
 * two ports that drift the same way agree with each other and with nothing
 * else.
 *
 * Issue #170 stage 2's notes: Intermediates are compared **in order and it stops at the
 * first disagreement**: a 52-block stack that reports only its last tensor says
 * that something is wrong and nothing about where.
 *
 *     npx tsx examples/anima/src/verify-forward.ts --dit ~/anima-q8
 */
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { type AnimaConfig, type AnimaTrace } from "./dit.js";
import { animaForwardResident, releaseAnimaWeights, type AnimaResidentStats } from "./dit-resident.js";
import { createResidentDevice } from "../../../harness/resident.js";
import { ditKernels } from "../../zimage/src/kernels-node.js";
import { permuteForRope, ropeAxisDims } from "./block.js";
import { type AnimaWeights, loadAnimaSubset } from "./weights-node.js";

interface Golden {
  config: Record<string, number | boolean | string>;
  latent: number;
  contextSeq: number;
  quantizationCost: { relativeRms: number };
  tensors: { name: string; offset: number; length: number }[];
}

function stats(got: Float32Array, want: Float32Array): { absMax: number; relRms: number } {
  let absMax = 0, sqErr = 0, sqWant = 0;
  for (let i = 0; i < want.length; i += 1) {
    const d = got[i]! - want[i]!;
    absMax = Math.max(absMax, Math.abs(d));
    sqErr += d * d;
    sqWant += want[i]! * want[i]!;
  }
  return { absMax, relRms: Math.sqrt(sqErr) / Math.sqrt(sqWant) };
}

const flag = process.argv.indexOf("--dit");
const dir = flag >= 0 ? process.argv[flag + 1] : process.env.ANIMA_DIT_DIR;
if (!dir) {
  console.error("verify-forward: pass --dit <dir>, the output of tools/convert_dit.py.");
  process.exit(2);
}

const fixtures = new URL("../fixtures/", import.meta.url);
const golden = JSON.parse(readFileSync(fileURLToPath(new URL("forward.manifest.json", fixtures)), "utf8")) as Golden;
const raw = readFileSync(fileURLToPath(new URL("forward.bin", fixtures)));
const blob = new Float32Array(raw.buffer, raw.byteOffset, raw.byteLength / 4);
const get = (name: string): Float32Array => {
  const e = golden.tensors.find((t) => t.name === name);
  if (!e) throw new Error(`golden has no tensor ${name}`);
  return blob.subarray(e.offset, e.offset + e.length);
};

const c = golden.config as unknown as {
  num_blocks: number; model_channels: number; num_heads: number; adaln_lora_dim: number;
  in_channels: number; out_channels: number; patch_spatial: number; patch_temporal: number;
  crossattn_emb_channels: number; concat_padding_mask: boolean;
  rope_t_extrapolation_ratio: number; rope_h_extrapolation_ratio: number; rope_w_extrapolation_ratio: number;
};
const cfg: AnimaConfig = {
  numBlocks: c.num_blocks,
  modelChannels: c.model_channels,
  numHeads: c.num_heads,
  adalnLoraDim: c.adaln_lora_dim,
  inChannels: c.in_channels,
  outChannels: c.out_channels,
  patchSpatial: c.patch_spatial,
  patchTemporal: c.patch_temporal,
  crossattnEmbChannels: c.crossattn_emb_channels,
  concatPaddingMask: c.concat_padding_mask,
  maxPeriod: 10000,
  normEps: 1e-6,
  ropeExtrapolation: {
    t: c.rope_t_extrapolation_ratio,
    h: c.rope_h_extrapolation_ratio,
    w: c.rope_w_extrapolation_ratio,
  },
};

console.log(`reading the DiT from ${dir}`);
const source = loadAnimaSubset(dir);
const headDim = cfg.modelChannels / cfg.numHeads;
const axisDims = ropeAxisDims(headDim);
console.log(`  ${cfg.numBlocks} blocks, dim ${cfg.modelChannels}, rope axes t/h/w = ${axisDims.join("/")}`);

/**
 * The weights, with Q/K permuted into `ops/rope`'s channel order.
 *
 * Done here rather than inside the block so it happens once per tensor rather
 * than once per call — and only for **self-attention**, since the cross
 * attention never sees a rotation (`predict2.py:184`). The QK-Norm weight goes
 * through the same permutation: it is `[headDim]` and multiplies channel by
 * channel inside a head, so leaving it alone scales each channel by another's.
 */
/**
 * The weight source the resident path takes.
 *
 * `packedQ8` returns `null` for the self-attention Q/K projections: those are
 * permuted into `ops/rope`'s channel order on the host, and a permuted tensor
 * is no longer the packed one on disk. Everything else — 90% of the weights by
 * bytes — still goes through `matmulQ8` untouched.
 */
const permuted = new Map<string, Float32Array>();
const isPermuted = (name: string): boolean =>
  /\.self_attn\.(q|k)_(proj|norm)\.weight$/.test(name);

const gpuWeights = {
  has: (name: string) => source.has(name),
  shapeOf: (name: string) => source.shapeOf(name),
  // A permuted tensor is not the packed one any more, so those fall back to the
  // dense path. It is 4 tensors per block out of 20.
  packedQ8: (name: string) => (isPermuted(name) ? null : source.packedQ8(name)),
  get: (name: string): Float32Array => weights.get(name),
};

const weights = {
  has: (name: string) => source.has(name),
  shapeOf: (name: string) => source.shapeOf(name),
  get: (name: string): Float32Array => {
    const cached = permuted.get(name);
    if (cached) return cached;
    const raw = source.get(name);
    const isSelfQK = /\.self_attn\.(q|k)_proj\.weight$/.test(name);
    const isSelfNorm = /\.self_attn\.(q|k)_norm\.weight$/.test(name);
    if (!isSelfQK && !isSelfNorm) return raw;
    const out = isSelfQK
      ? permuteForRope(raw, cfg.numHeads, headDim, cfg.modelChannels)
      : permuteForRope(raw, 1, headDim, 1);
    permuted.set(name, out);
    return out;
  },
};

/**
 * Every name the forward reads, and every prefix it announced first.
 *
 * The browser hydrates its heap from `onBeforePrefix` and reads synchronously
 * afterwards, so a tensor whose name no announced prefix covers is a run-time
 * "read before it was preloaded" — in the page, on someone else's machine,
 * after a 4.7 GB download. Checked here instead, where it costs one run.
 */
const announced: string[] = [];
const read = new Set<string>();
// Wraps `gpuWeights`, not `source`. Delegating `packedQ8` straight to the raw
// source takes the fast path past the rope permutation — the first version of
// this wrapper did, and read 4.814e-1 instead of 1.968e-5.
const watched = {
  has: (name: string) => { read.add(name); return gpuWeights.has(name); },
  shapeOf: (name: string) => gpuWeights.shapeOf(name),
  get: (name: string) => { read.add(name); return gpuWeights.get(name); },
  packedQ8: (name: string) => { read.add(name); return gpuWeights.packedQ8(name); },
};

const device = await createResidentDevice();
if (!device) {
  console.error("verify-forward-gpu: no WebGPU adapter available.");
  process.exit(2);
}
const held = new Map<string, GPUBuffer>();
const blank = (): AnimaResidentStats =>
  ({ dispatches: 0, submits: 0, poolSlots: 0, poolBytes: 0, weightBuffers: 0, uploadedBytes: 0 });
const input = { latent: get("x"), T: 1, H: golden.latent, W: golden.latent, t: get("t")[0]!, context: get("context") };

const first = blank();
const started = Date.now();
const trace: AnimaTrace = {};
const out = await animaForwardResident(
  device, ditKernels(), cfg, watched, input, first, held, trace, undefined,
  async (prefix) => { announced.push(prefix); },
);
const firstSecs = (Date.now() - started) / 1000;

// A second forward with the weights already resident — what steps 2..N cost.
const second = blank();
const again = Date.now();
await animaForwardResident(device, ditKernels(), cfg, watched, input, second, held);

const uncovered = [...read].filter((name) => !announced.some((prefix) => name.startsWith(prefix)));
if (uncovered.length > 0) {
  console.error(
    `\nverify-forward-gpu: ${uncovered.length} tensors are read but no onBeforePrefix announced them:\n` +
      uncovered.slice(0, 8).map((n) => `  ${n}`).join("\n") +
      "\nThe browser hydrates its heap from those prefixes, so this is a run-time failure there.",
  );
  process.exit(1);
}
console.log(`  ${read.size} tensors read, all covered by ${announced.length} announced prefixes`);
console.log(`  forward 1: ${firstSecs.toFixed(1)}s, uploaded ${(first.uploadedBytes / 1e9).toFixed(2)} GB`);
console.log(
  `  forward 2: ${((Date.now() - again) / 1000).toFixed(1)}s, uploaded ` +
    `${(second.uploadedBytes / 1e9).toFixed(3)} GB  <- weights already resident`,
);
console.log(`  ${first.dispatches} dispatches, ${first.submits} submits, pool ${(first.poolBytes / 1e9).toFixed(2)} GB`);

const checkpoints: [string, Float32Array | undefined][] = [
  ["tEmbed", trace.tEmbed],
  ["afterBlock0", trace.afterBlock0],
  ["afterBlocks", trace.afterBlocks],
  ["output", out],
];

// The port's own error on one block is 3.221e-7. Over 52 blocks f32
// accumulation drifts further, so the bar is 1e-4 — far below the 4.018e-2 that
// q8 costs, which is what makes a failure here mean "the port".
const bar = 1e-4;
let failed = false;
console.log("");
for (const [name, got] of checkpoints) {
  if (!got) {
    console.error(`  ${name}: not captured`);
    failed = true;
    break;
  }
  const s = stats(got, get(name));
  const denseName = `${name}Dense`;
  const dense = golden.tensors.some((t) => t.name === denseName) ? stats(got, get(denseName)) : null;
  const verdict = s.relRms < bar ? "ok" : "MISMATCH";
  console.log(
    `  ${name.padEnd(14)} vs quantized ${s.relRms.toExponential(3)}  ` +
      `vs dense ${dense ? dense.relRms.toExponential(3) : "   n/a   "}  ${verdict}`,
  );
  if (s.relRms >= bar) {
    console.error(`\nFirst mismatch at ${name}. Everything after it is downstream of the same cause.`);
    failed = true;
    break;
  }
}

releaseAnimaWeights(held);
if (failed) process.exit(1);
console.log(
  `\nOK: the full forward matches the model.\n` +
    `The "vs dense" column is what q8 costs over ${cfg.numBlocks} blocks, ` +
    `measured independently in torch as ${golden.quantizationCost.relativeRms.toExponential(3)}.`,
);
void (undefined as AnimaWeights | undefined);
