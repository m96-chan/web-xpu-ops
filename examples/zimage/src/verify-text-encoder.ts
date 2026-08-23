/**
 * Runs Z-Image's text encoder and compares it to the model.
 *
 * Issue #166 stage 3. The weights are read straight out of the checkpoint's
 * `.safetensors` at full precision — **not** quantized — on purpose: stages 1
 * and 2 showed how easily a porting mistake and a format's cost get attributed
 * to each other, and the cheapest way to keep them apart here is to have no
 * format in the picture yet. Quantizing the encoder is a separate step with a
 * separate number.
 *
 *     npx tsx examples/zimage/src/verify-text-encoder.ts
 *
 * Reads the checkpoint through `tools/models.py`'s resolution order, so
 * `--model-dir` / `ZIMAGE_MODEL_DIR` works the same as everywhere else.
 */
import { readFileSync, readdirSync, existsSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { ShardedSafetensors, SafetensorsFile } from "./safetensors.js";
import {
  type Qwen3Config,
  type Qwen3LayerWeights,
  type Qwen3Weights,
  permuteLayerForRope,
  qwen3Encode,
} from "./text-encoder.js";

interface Golden {
  validTokens: number;
  keptTokens: number;
  hiddenStatesIndex: { resolvedLayer: number };
  config: Omit<Qwen3Config, "stopAfterLayer">;
  tensors: { name: string; shape: number[]; offset: number; length: number }[];
}

/** The same resolution `tools/models.py` does, for the one component this needs. */
function resolveTextEncoder(): string {
  const explicit = process.env.ZIMAGE_MODEL_DIR;
  if (explicit) {
    const sub = join(explicit, "text_encoder");
    if (!existsSync(sub)) throw new Error(`${sub} does not exist.`);
    return sub;
  }
  const hub = join(process.env.HF_HOME ?? join(homedir(), ".cache", "huggingface"), "hub");
  const repo = join(hub, "models--Tongyi-MAI--Z-Image", "snapshots");
  if (!existsSync(repo)) {
    throw new Error(
      `No text encoder found. Run tools/gen_text_encoder_golden.py once to fetch it, ` +
        `or set ZIMAGE_MODEL_DIR to a directory laid out like the repository.`,
    );
  }
  const snapshot = readdirSync(repo)[0]!;
  return join(repo, snapshot, "text_encoder");
}

function stats(got: Float32Array, want: Float32Array): { absMax: number; relRms: number } {
  let absMax = 0;
  let sqErr = 0;
  let sqWant = 0;
  for (let i = 0; i < want.length; i += 1) {
    const d = got[i]! - want[i]!;
    absMax = Math.max(absMax, Math.abs(d));
    sqErr += d * d;
    sqWant += want[i]! * want[i]!;
  }
  return { absMax, relRms: Math.sqrt(sqErr) / Math.sqrt(sqWant) };
}

function main(): void {
  const fixtures = new URL("../fixtures/", import.meta.url);
  const golden = JSON.parse(
    readFileSync(fileURLToPath(new URL("text-encoder.manifest.json", fixtures)), "utf8"),
  ) as Golden;
  const raw = readFileSync(fileURLToPath(new URL("text-encoder.bin", fixtures)));
  const blob = new Float32Array(raw.buffer, raw.byteOffset, raw.byteLength / 4);
  const get = (name: string): Float32Array => {
    const entry = golden.tensors.find((t) => t.name === name);
    if (!entry) throw new Error(`golden has no tensor ${name}`);
    return blob.subarray(entry.offset, entry.offset + entry.length);
  };

  const dir = resolveTextEncoder();
  console.log(`reading the text encoder from ${dir}`);
  const indexPath = join(dir, "model.safetensors.index.json");
  const single = join(dir, "model.safetensors");
  let read: (name: string) => Float32Array;
  let close: () => void;
  if (existsSync(indexPath)) {
    const weightMap = (JSON.parse(readFileSync(indexPath, "utf8")) as { weight_map: Record<string, string> })
      .weight_map;
    const shards = new ShardedSafetensors(dir, weightMap);
    read = (name) => shards.read(name);
    close = () => shards.close();
  } else {
    const file = new SafetensorsFile(single);
    read = (name) => file.read(name);
    close = () => file.close();
  }

  const cfg: Qwen3Config = { ...golden.config, stopAfterLayer: golden.hiddenStatesIndex.resolvedLayer };

  // The golden stores the padded run's ids; the real ones are its prefix.
  const idsF32 = get("inputIds");
  const valid = golden.validTokens;
  const ids = Int32Array.from([...idsF32.subarray(0, valid)].map((v) => Math.round(v)));

  const weights: Qwen3Weights = {
    numLayers: cfg.numLayers,
    // 15 rows of a 151936-row table, read as 15 reads rather than one 1.55 GB
    // allocation. `gather` is the op for the GPU path; here the file is the
    // table.
    embed: (wanted) => {
      const table = read("model.embed_tokens.weight");
      const out = new Float32Array(wanted.length * cfg.hiddenSize);
      for (let i = 0; i < wanted.length; i += 1) {
        out.set(table.subarray(wanted[i]! * cfg.hiddenSize, (wanted[i]! + 1) * cfg.hiddenSize), i * cfg.hiddenSize);
      }
      return out;
    },
    layer: (index) => {
      const p = `model.layers.${index}.`;
      const raw: Qwen3LayerWeights = {
        input_layernorm: read(`${p}input_layernorm.weight`),
        q_proj: read(`${p}self_attn.q_proj.weight`),
        k_proj: read(`${p}self_attn.k_proj.weight`),
        v_proj: read(`${p}self_attn.v_proj.weight`),
        o_proj: read(`${p}self_attn.o_proj.weight`),
        q_norm: read(`${p}self_attn.q_norm.weight`),
        k_norm: read(`${p}self_attn.k_norm.weight`),
        post_attention_layernorm: read(`${p}post_attention_layernorm.weight`),
        gate_proj: read(`${p}mlp.gate_proj.weight`),
        up_proj: read(`${p}mlp.up_proj.weight`),
        down_proj: read(`${p}mlp.down_proj.weight`),
      };
      return permuteLayerForRope(raw, cfg);
    },
  };

  console.log(`  ${valid} tokens, layers 0..${cfg.stopAfterLayer} of ${cfg.numLayers}`);
  const started = Date.now();
  const got = qwen3Encode(cfg, weights, ids);
  console.log(`  encoded in ${((Date.now() - started) / 1000).toFixed(1)}s (CPU)`);
  close();

  const want = get("capFeats").subarray(0, valid * cfg.hiddenSize);
  const s = stats(got.subarray(0, valid * cfg.hiddenSize), want);

  // The golden is f32 and so is this, so the only difference should be
  // accumulation order over 35 layers. 1e-4 is loose for that and far tighter
  // than a one-layer-off mistake, which would be O(1).
  const bar = 1e-4;
  console.log("");
  console.log(`  capFeats  rel-RMS ${s.relRms.toExponential(3)}  abs max ${s.absMax.toExponential(3)}  ` +
    `${s.relRms < bar ? "ok" : "MISMATCH"}`);

  if (!(s.relRms < bar)) {
    console.error(`\nFAIL: rel-RMS ${s.relRms.toExponential(3)} exceeds ${bar}.`);
    process.exit(1);
  }
  console.log(
    `\nOK: the encoder matches the model on the shipped weights, and the padded golden's first ` +
      `${valid} rows match an unpadded run — the causal argument for skipping the 512-token pad holds.`,
  );
}

main();
