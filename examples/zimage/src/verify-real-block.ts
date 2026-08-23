/**
 * Runs `layers.0` with the shipped weights and compares it to the model.
 *
 * This is issue #166's first checkpoint, and the thing `block.test.ts` cannot
 * do: that one proves the composition matches Z-Image's algebra, at 64 channels
 * with random weights, because a fixture with real weights would be gigabytes.
 * Here the weights come off `convert_dit.py`'s output and the golden carries
 * only the inputs and outputs, so the full width — `dim=3840`, `head_dim=128`,
 * RoPE axes `[32, 48, 48]` — is exercised against the real tensors.
 *
 * Two comparisons, and they answer different questions:
 *
 *   - **against `outputQ4`** — the model run with weights put through the same
 *     quantization. Disagreement here is a porting or loader mistake, and the
 *     bar is f32 rounding.
 *   - **against `outputDense`** — the model run with its own weights. The gap
 *     is what 4-bit costs, not what the port gets wrong, and it is reported
 *     rather than asserted.
 *
 * Not a vitest file on purpose: it needs the 3.3 GB blob, which CI does not
 * have. `block.test.ts` is the part that runs everywhere.
 *
 *     npx tsx examples/zimage/src/verify-real-block.ts --dit ~/zimage-q4
 */
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { type BlockConfig, type BlockWeights, zimageBlock } from "./block.js";
import { layerSelector, loadDitSubset } from "./weights-node.js";

interface GoldenManifest {
  config: BlockConfig & { adalnEmbedDim: number };
  ids: number[][];
  quantizedWeights: Record<string, string>;
  quantizationCost: { relativeRms: number; cosine: number };
  tensors: { name: string; shape: number[]; offset: number; length: number }[];
}

const WEIGHT_NAMES = [
  "attention.to_q.weight",
  "attention.to_k.weight",
  "attention.to_v.weight",
  "attention.to_out.0.weight",
  "attention.norm_q.weight",
  "attention.norm_k.weight",
  "feed_forward.w1.weight",
  "feed_forward.w2.weight",
  "feed_forward.w3.weight",
  "attention_norm1.weight",
  "ffn_norm1.weight",
  "attention_norm2.weight",
  "ffn_norm2.weight",
  "adaLN_modulation.0.weight",
  "adaLN_modulation.0.bias",
] as const;

function stats(got: Float32Array, want: Float32Array): { absMax: number; relRms: number; cosine: number } {
  let absMax = 0;
  let sqErr = 0;
  let sqWant = 0;
  let dot = 0;
  let sqGot = 0;
  for (let i = 0; i < want.length; i += 1) {
    const d = got[i]! - want[i]!;
    absMax = Math.max(absMax, Math.abs(d));
    sqErr += d * d;
    sqWant += want[i]! * want[i]!;
    sqGot += got[i]! * got[i]!;
    dot += got[i]! * want[i]!;
  }
  return {
    absMax,
    relRms: Math.sqrt(sqErr) / Math.sqrt(sqWant),
    cosine: dot / (Math.sqrt(sqGot) * Math.sqrt(sqWant)),
  };
}

function main(): void {
  const ditFlag = process.argv.indexOf("--dit");
  const dit = ditFlag >= 0 ? process.argv[ditFlag + 1] : process.env.ZIMAGE_DIT_DIR;
  if (!dit) {
    console.error(
      "verify-real-block: pass --dit <dir>, the output of tools/convert_dit.py (or set ZIMAGE_DIT_DIR).\n" +
        "It is not downloaded automatically: converting the checkpoint is a 12 GB read and a deliberate act.",
    );
    process.exit(2);
  }

  const fixtures = new URL("../fixtures/", import.meta.url);
  const golden = JSON.parse(
    readFileSync(fileURLToPath(new URL("real-block.manifest.json", fixtures)), "utf8"),
  ) as GoldenManifest;
  const raw = readFileSync(fileURLToPath(new URL("real-block.bin", fixtures)));
  const blob = new Float32Array(raw.buffer, raw.byteOffset, raw.byteLength / 4);
  const get = (name: string): Float32Array => {
    const entry = golden.tensors.find((t) => t.name === name);
    if (!entry) throw new Error(`golden has no tensor ${name}`);
    return blob.subarray(entry.offset, entry.offset + entry.length);
  };

  console.log(`loading layers.0 from ${dit}`);
  const started = Date.now();
  const weights = loadDitSubset(dit, layerSelector(0));
  const collected = weights.block(0, WEIGHT_NAMES) as unknown as BlockWeights;
  console.log(`  ${WEIGHT_NAMES.length} tensors in ${((Date.now() - started) / 1000).toFixed(2)}s`);

  const cfg = golden.config;
  const positions = Int32Array.from(golden.ids.flat());
  const ran = Date.now();
  const got = zimageBlock(cfg, collected, get("x"), get("adalnInput"), positions);
  console.log(`  block ran in ${((Date.now() - ran) / 1000).toFixed(2)}s (CPU)`);

  const vsQ4 = stats(got, get("outputQ4"));
  const vsDense = stats(got, get("outputDense"));
  console.log("");
  console.log("port vs model, same quantized weights (this is the port's own error):");
  console.log(`  abs max ${vsQ4.absMax.toExponential(3)}  rel-RMS ${vsQ4.relRms.toExponential(3)}  cos ${vsQ4.cosine.toFixed(9)}`);
  console.log("port vs model at full precision (adds what quantization costs):");
  console.log(`  abs max ${vsDense.absMax.toExponential(3)}  rel-RMS ${vsDense.relRms.toExponential(3)}  cos ${vsDense.cosine.toFixed(6)}`);
  console.log(
    `  of which quantization alone, measured in torch: rel-RMS ` +
      `${golden.quantizationCost.relativeRms.toExponential(3)}  cos ${golden.quantizationCost.cosine.toFixed(6)}`,
  );

  // The bar is f32 accumulation over K=3840 and K=10240 in a different order
  // than torch uses, not agreement to the bit. 1e-4 relative is loose enough
  // for that and far tighter than the 1.8e-2 quantization costs — so this
  // fails on a porting mistake and does not fail on the format.
  const bar = 1e-4;
  if (!(vsQ4.relRms < bar)) {
    console.error(`\nFAIL: rel-RMS ${vsQ4.relRms.toExponential(3)} against the quantized model exceeds ${bar}.`);
    process.exit(1);
  }
  console.log(`\nOK: the port matches the model on real weights to ${vsQ4.relRms.toExponential(3)} relative RMS.`);
}

main();
