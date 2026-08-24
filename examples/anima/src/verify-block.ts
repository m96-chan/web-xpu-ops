/**
 * Runs `net.blocks.0` on the shipped weights and compares it to the model.
 *
 * Issue #170's first checkpoint. Two comparisons, because two different things
 * can be wrong: against the golden's **quantized** run, where an honest port
 * should agree to f32 rounding, and against its full-precision run, where the
 * gap is what q8 costs and is reported rather than asserted.
 *
 * Not a vitest file: it needs the converted checkpoint, which CI does not have.
 *
 *     npx tsx examples/anima/src/verify-block.ts --dit ~/anima-q8
 */
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { type AnimaBlockConfig, type AnimaBlockWeights, animaBlock } from "./block.js";
import { loadAnimaSubset } from "./weights-node.js";

const WEIGHT_NAMES = [
  "self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight",
  "self_attn.output_proj.weight", "self_attn.q_norm.weight", "self_attn.k_norm.weight",
  "cross_attn.q_proj.weight", "cross_attn.k_proj.weight", "cross_attn.v_proj.weight",
  "cross_attn.output_proj.weight", "cross_attn.q_norm.weight", "cross_attn.k_norm.weight",
  "mlp.layer1.weight", "mlp.layer2.weight",
  "adaln_modulation_self_attn.1.weight", "adaln_modulation_self_attn.2.weight",
  "adaln_modulation_cross_attn.1.weight", "adaln_modulation_cross_attn.2.weight",
  "adaln_modulation_mlp.1.weight", "adaln_modulation_mlp.2.weight",
] as const;

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
  console.error("verify-block: pass --dit <dir>, the output of tools/convert_dit.py.");
  process.exit(2);
}

const fixtures = new URL("../fixtures/", import.meta.url);
const golden = JSON.parse(readFileSync(fileURLToPath(new URL("block.manifest.json", fixtures)), "utf8")) as {
  config: AnimaBlockConfig;
  quantizationCost: { relativeRms: number };
  tensors: { name: string; offset: number; length: number }[];
};
const raw = readFileSync(fileURLToPath(new URL("block.bin", fixtures)));
const blob = new Float32Array(raw.buffer, raw.byteOffset, raw.byteLength / 4);
const get = (name: string): Float32Array => {
  const e = golden.tensors.find((t) => t.name === name);
  if (!e) throw new Error(`golden has no tensor ${name}`);
  return blob.subarray(e.offset, e.offset + e.length);
};

console.log(`loading net.blocks.0 from ${dir}`);
const started = Date.now();
const source = loadAnimaSubset(dir, (name) => name.startsWith("net.blocks.0."));
const weights: Record<string, Float32Array> = {};
for (const name of WEIGHT_NAMES) weights[name.replace(/\./g, "_")] = source.get(`net.blocks.0.${name}`);
console.log(`  ${WEIGHT_NAMES.length} tensors in ${((Date.now() - started) / 1000).toFixed(2)}s`);

const ran = Date.now();
const got = animaBlock(
  golden.config,
  weights as unknown as AnimaBlockWeights,
  get("x"), get("emb"), get("lora"), get("context"),
);
console.log(`  block ran in ${((Date.now() - ran) / 1000).toFixed(2)}s (CPU)`);

const vsQuantized = stats(got, get("output"));
const vsDense = stats(got, get("outputDense"));
console.log("");
console.log("port vs model, same quantized weights (the port's own error):");
console.log(`  abs max ${vsQuantized.absMax.toExponential(3)}  rel-RMS ${vsQuantized.relRms.toExponential(3)}`);
console.log("port vs model at full precision (adds what q8 costs):");
console.log(`  abs max ${vsDense.absMax.toExponential(3)}  rel-RMS ${vsDense.relRms.toExponential(3)}`);
console.log(`  of which quantization alone, measured in torch: ${golden.quantizationCost.relativeRms.toExponential(3)}`);

// f32 accumulation over K=2048 and K=8192 in a different order than torch uses.
// 1e-4 is loose for that and far tighter than the 2e-2 q8 costs, so this fails
// on a porting mistake and not on the format.
const bar = 1e-4;
if (!(vsQuantized.relRms < bar)) {
  console.error(`\nFAIL: rel-RMS ${vsQuantized.relRms.toExponential(3)} against the quantized model exceeds ${bar}.`);
  process.exit(1);
}
console.log(`\nOK: the port matches the model on real weights to ${vsQuantized.relRms.toExponential(3)} relative RMS.`);
