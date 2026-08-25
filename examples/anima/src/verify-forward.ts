/**
 * Runs Anima's whole DiT on the shipped weights and compares it to the model.
 *
 * Issue #170 stage 2. Intermediates are compared **in order and it stops at the
 * first disagreement**: a 52-block stack that reports only its last tensor says
 * that something is wrong and nothing about where.
 *
 *     npx tsx examples/anima/src/verify-forward.ts --dit ~/anima-q8
 */
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { type AnimaConfig, type AnimaTrace, animaForward } from "./dit.js";
import { setQuantizationMode, type QuantizationMode } from "./quantization-probe.js";
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
const permuted = new Map<string, Float32Array>();
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
 * `--quant weights|w8a8` simulates a quantized matmul in the CPU port.
 *
 * Issue #184. The only `subgroup-matrix` configurations this GPU offers are
 * `i8 x i8 -> i32`, so reaching a tensor core means quantizing the
 * *activation* as well as the weight. What that costs is measurable here,
 * before any WGSL exists.
 *
 * `weights` is the calibration, not a result: it simulates what the GPU path
 * already does, and the answer is known to be 4.018e-2 against the dense
 * golden. If this run does not reproduce that, the `w8a8` number below it is
 * worth nothing.
 */
const quantAt = process.argv.indexOf("--quant");
const quant = (quantAt >= 0 ? process.argv[quantAt + 1] : "off") as QuantizationMode;
if (!["off", "weights", "w8a8"].includes(quant)) {
  console.error(`verify-forward: --quant must be off, weights or w8a8 (got ${quant})`);
  process.exit(2);
}
setQuantizationMode(quant);
console.log(`  matmul simulation: ${quant}`);

const started = Date.now();
const trace: AnimaTrace = {};
const out = animaForward(
  cfg,
  weights,
  { latent: get("x"), T: 1, H: golden.latent, W: golden.latent, t: get("t")[0]!, context: get("context") },
  trace,
);
console.log(`  forward in ${((Date.now() - started) / 1000).toFixed(1)}s (CPU)`);

const checkpoints: [string, Float32Array | undefined][] = [
  ["tEmbed", trace.tEmbed],
  ["afterBlock0", trace.afterBlock0],
  ["afterBlocks", trace.afterBlocks],
  ["output", out],
];

// The port's own error on one block is 3.221e-7. Over 52 blocks f32
// accumulation drifts further, so the bar is 1e-4 — far below the 4.018e-2 that
// q8 costs, which is what makes a failure here mean "the port".
// Only meaningful with `--quant off`: the other modes are asking how far a
// quantization moves the answer, so a "mismatch" is the measurement.
const bar = quant === "off" ? 1e-4 : Infinity;
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

if (failed) process.exit(1);
console.log(
  `\nOK: the full forward matches the model.\n` +
    `The "vs dense" column is what q8 costs over ${cfg.numBlocks} blocks, ` +
    `measured independently in torch as ${golden.quantizationCost.relativeRms.toExponential(3)}.`,
);
void (undefined as AnimaWeights | undefined);
