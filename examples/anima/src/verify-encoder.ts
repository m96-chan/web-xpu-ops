/**
 * Runs Anima's conditioning path on the shipped weights and compares it to the
 * model.
 *
 * Issue #170 stage 4. Checkpoints are compared **in order and it stops at the
 * first disagreement**: 28 encoder layers followed by 6 adapter blocks reported
 * only at the end says that something is wrong and nothing about where.
 *
 *     npx tsx examples/anima/src/verify-encoder.ts \
 *       --encoder ~/anima-src/qwen_3_06b_base.safetensors --dit ~/anima-q8
 *
 * The encoder is read dense out of its own safetensors and the adapter out of
 * the q8 DiT — the pair actually shipped. `tools/gen_encoder_golden.py` records
 * both a dense and a q8 answer for the adapter so a porting mistake can be told
 * apart from what the format costs.
 */
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { qwen3Encode, type Qwen3LayerWeights, type Qwen3Weights, permuteLayerForRope } from "../../zimage/src/text-encoder.js";
import { SafetensorsFile } from "../../zimage/src/safetensors.js";
import {
  ADAPTER,
  type AdapterBlockWeights,
  type AdapterWeights,
  QWEN3_06B,
  llmAdapterForward,
  permuteAdapterBlock,
} from "./text-encoder.js";
import { loadAnimaSubset } from "./weights-node.js";

interface Entry {
  name: string;
  shape: number[];
  dtype: "f32" | "i32";
  offset: number;
}
interface Golden {
  prompt: string;
  encoder: string;
  dit: string;
  tensors: Entry[];
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

const arg = (name: string): string | undefined => {
  const at = process.argv.indexOf(name);
  return at >= 0 ? process.argv[at + 1] : undefined;
};
const encoderPath = arg("--encoder") ?? process.env.ANIMA_ENCODER;
const ditDir = arg("--dit") ?? process.env.ANIMA_DIT_DIR;
if (!encoderPath || !ditDir) {
  console.error("verify-encoder: pass --encoder <qwen_3_06b_base.safetensors> --dit <convert_dit.py output dir>");
  process.exit(2);
}

// --- the golden ---
const path = fileURLToPath(new URL("../fixtures/encoder-golden.bin", import.meta.url));
const raw = readFileSync(path);
const headerLength = Number(raw.readBigUInt64LE(0));
const golden = JSON.parse(raw.subarray(8, 8 + headerLength).toString("utf8")) as Golden;
const base = 8 + headerLength;
const find = (name: string): Entry => {
  const e = golden.tensors.find((t) => t.name === name);
  if (!e) throw new Error(`golden has no tensor ${name}`);
  return e;
};
const f32 = (name: string): Float32Array => {
  const e = find(name);
  const count = e.shape.reduce((a, b) => a * b, 1);
  return new Float32Array(raw.buffer.slice(raw.byteOffset + base + e.offset, raw.byteOffset + base + e.offset + count * 4));
};
const i32 = (name: string): Int32Array => {
  const e = find(name);
  const count = e.shape.reduce((a, b) => a * b, 1);
  return new Int32Array(raw.buffer.slice(raw.byteOffset + base + e.offset, raw.byteOffset + base + e.offset + count * 4));
};

const qwenIds = i32("qwenIds");
const t5Ids = i32("t5Ids");
console.log(`prompt: ${JSON.stringify(golden.prompt)}`);
console.log(`  ${qwenIds.length} qwen ids, ${t5Ids.length} t5 ids`);

// --- the encoder, dense from its own file ---
const encoderFile = new SafetensorsFile(encoderPath);
const layerCache = new Map<number, Qwen3LayerWeights>();
const encoderWeights: Qwen3Weights = {
  numLayers: QWEN3_06B.numLayers,
  embed(ids) {
    const rows = new Float32Array(ids.length * QWEN3_06B.hiddenSize);
    const table = encoderFile.read("model.embed_tokens.weight");
    for (let i = 0; i < ids.length; i += 1) {
      rows.set(
        table.subarray(ids[i]! * QWEN3_06B.hiddenSize, (ids[i]! + 1) * QWEN3_06B.hiddenSize),
        i * QWEN3_06B.hiddenSize,
      );
    }
    return rows;
  },
  layer(index) {
    const cached = layerCache.get(index);
    if (cached) return cached;
    const p = `model.layers.${index}.`;
    const built = permuteLayerForRope({
      input_layernorm: encoderFile.read(`${p}input_layernorm.weight`),
      q_proj: encoderFile.read(`${p}self_attn.q_proj.weight`),
      k_proj: encoderFile.read(`${p}self_attn.k_proj.weight`),
      v_proj: encoderFile.read(`${p}self_attn.v_proj.weight`),
      o_proj: encoderFile.read(`${p}self_attn.o_proj.weight`),
      q_norm: encoderFile.read(`${p}self_attn.q_norm.weight`),
      k_norm: encoderFile.read(`${p}self_attn.k_norm.weight`),
      post_attention_layernorm: encoderFile.read(`${p}post_attention_layernorm.weight`),
      gate_proj: encoderFile.read(`${p}mlp.gate_proj.weight`),
      up_proj: encoderFile.read(`${p}mlp.up_proj.weight`),
      down_proj: encoderFile.read(`${p}mlp.down_proj.weight`),
    }, QWEN3_06B);
    layerCache.set(index, built);
    return built;
  },
  // `layer="last"` is after `model.norm` — see `Qwen3Weights.finalNorm`.
  finalNorm: encoderFile.read("model.norm.weight"),
};

// --- the adapter, out of the q8 DiT ---
const dit = loadAnimaSubset(ditDir);
const blockCache = new Map<number, AdapterBlockWeights>();
const adapterWeights: AdapterWeights = {
  embed(ids) {
    const table = dit.get("net.llm_adapter.embed.weight");
    const rows = new Float32Array(ids.length * ADAPTER.targetDim);
    for (let i = 0; i < ids.length; i += 1) {
      rows.set(table.subarray(ids[i]! * ADAPTER.targetDim, (ids[i]! + 1) * ADAPTER.targetDim), i * ADAPTER.targetDim);
    }
    return rows;
  },
  block(index) {
    const cached = blockCache.get(index);
    if (cached) return cached;
    const p = `net.llm_adapter.blocks.${index}.`;
    const built = permuteAdapterBlock({
      norm_self_attn: dit.get(`${p}norm_self_attn.weight`),
      self_attn_q_proj: dit.get(`${p}self_attn.q_proj.weight`),
      self_attn_k_proj: dit.get(`${p}self_attn.k_proj.weight`),
      self_attn_v_proj: dit.get(`${p}self_attn.v_proj.weight`),
      self_attn_o_proj: dit.get(`${p}self_attn.o_proj.weight`),
      self_attn_q_norm: dit.get(`${p}self_attn.q_norm.weight`),
      self_attn_k_norm: dit.get(`${p}self_attn.k_norm.weight`),
      norm_cross_attn: dit.get(`${p}norm_cross_attn.weight`),
      cross_attn_q_proj: dit.get(`${p}cross_attn.q_proj.weight`),
      cross_attn_k_proj: dit.get(`${p}cross_attn.k_proj.weight`),
      cross_attn_v_proj: dit.get(`${p}cross_attn.v_proj.weight`),
      cross_attn_o_proj: dit.get(`${p}cross_attn.o_proj.weight`),
      cross_attn_q_norm: dit.get(`${p}cross_attn.q_norm.weight`),
      cross_attn_k_norm: dit.get(`${p}cross_attn.k_norm.weight`),
      norm_mlp: dit.get(`${p}norm_mlp.weight`),
      mlp_0_weight: dit.get(`${p}mlp.0.weight`),
      mlp_0_bias: dit.get(`${p}mlp.0.bias`),
      mlp_2_weight: dit.get(`${p}mlp.2.weight`),
      mlp_2_bias: dit.get(`${p}mlp.2.bias`),
    });
    blockCache.set(index, built);
    return built;
  },
  out_proj: dit.get("net.llm_adapter.out_proj.weight"),
  out_proj_bias: dit.get("net.llm_adapter.out_proj.bias"),
  norm: dit.get("net.llm_adapter.norm.weight"),
};

// --- run ---
const started = Date.now();
const source = qwen3Encode(QWEN3_06B, encoderWeights, qwenIds);
console.log(`  encoder in ${((Date.now() - started) / 1000).toFixed(1)}s`);

const trace = new Map<string, Float32Array>();
const context = llmAdapterForward(adapterWeights, source, qwenIds.length, t5Ids, (index, hidden) => {
  if (index === 0) trace.set("adapterBlock0", hidden.slice());
});

// --- compare, in order ---
const CHECKS: { name: string; got: Float32Array; tolerance: number }[] = [
  { name: "source", got: source, tolerance: 2e-5 },
  { name: "adapterBlock0", got: trace.get("adapterBlock0")!, tolerance: 5e-2 },
  { name: "context", got: context, tolerance: 5e-2 },
];

let failed = false;
console.log();
for (const check of CHECKS) {
  const want = f32(check.name);
  const dense = f32(`${check.name}Dense`);
  const q = stats(check.got, want);
  const d = stats(check.got, dense);
  const ok = q.relRms <= check.tolerance;
  console.log(
    `  ${check.name.padEnd(14)} vs quantized ${q.relRms.toExponential(3)}` +
      `  vs dense ${d.relRms.toExponential(3)}  ${ok ? "ok" : "MISMATCH"}`,
  );
  if (!ok) {
    failed = true;
    console.log(`\nFirst mismatch at ${check.name}. Everything after it is downstream of the same cause.`);
    break;
  }
}

if (!failed) {
  console.log("\nOK: the conditioning path matches the model.");
  console.log('The "vs dense" column is what q8 on the adapter costs; the encoder is not quantized.');
}
process.exit(failed ? 1 : 0);
