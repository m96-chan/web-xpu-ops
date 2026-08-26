/**
 * The GPU conditioner against the hidden state Qwen3-VL's own code produced.
 *
 * Issue #212. A script rather than a test: it holds **26.27 GB** of weights on
 * the device, which is not something to do inside a vitest worker beside 2,000
 * other tests. `examples/h3-dit/src/verify-forward.ts` is the same arrangement.
 *
 *     npx tsx examples/h3-ref2v/src/verify-conditioner.ts \
 *       --dir ~/h3-cond-gpu --golden ~/h3-cond-real
 */
import { closeSync, openSync, readFileSync, readSync, statSync } from "node:fs";
import { createResidentDevice } from "../../../harness/resident.js";
import { ConditionerGpu, type ConditionerManifest } from "./conditioner-gpu.js";
import { qwen3vlPositionGrid } from "./text-encoder.js";
import { conditionerKernels } from "./kernels-node.js";
import { patchify, type ProcessorConfig } from "./processor.js";
import type { Grid } from "./vision.js";

const arg = (name: string): string | undefined => {
  const at = process.argv.indexOf(name);
  return at >= 0 ? process.argv[at + 1] : undefined;
};
const dir = arg("--dir");
const goldenDir = arg("--golden");
if (!dir || !goldenDir) {
  console.error("verify-conditioner: --dir and --golden are required");
  process.exit(2);
}

const f32 = (path: string): Float32Array => {
  const buffer = readFileSync(path);
  return new Float32Array(buffer.buffer, buffer.byteOffset, buffer.byteLength / 4);
};

const manifest = JSON.parse(readFileSync(`${dir}/conditioner.manifest.json`, "utf8")) as ConditionerManifest & {
  processor: ProcessorConfig;
};
const golden = JSON.parse(readFileSync(`${goldenDir}/golden.json`, "utf8")) as {
  imageSize: number; tokenIds: number[]; visualRuns: [number, number][];
  grid: Grid; seq: number; hidden: [number, number]; layer: number; stages?: number[];
  tokenIds: number[]; positionIds: { t: number[]; h: number[]; w: number[] };
};

// `--layers N` truncates the stack and compares against `hidden.N.bin`, so a
// divergence can be bisected without a 26 GB run per guess.
const layersArg = arg("--layers");
const stage = layersArg === undefined ? null : Number(layersArg);
if (stage !== null) {
  manifest.layers = stage;
  if (!golden.stages?.includes(stage)) {
    console.error(`the golden has no hidden.${stage}.bin (it has ${golden.stages?.join(", ")})`);
    process.exit(2);
  }
}

if (stage === null && golden.layer !== manifest.textEncoderLayer) {
  console.error(`the golden read layer ${golden.layer} and this conversion holds through ${manifest.textEncoderLayer}`);
  process.exit(2);
}

const weightsPath = `${dir}/${manifest.dtype === "q8" ? "conditioner.q8.bin" : "conditioner.bin"}`;
const fd = openSync(weightsPath, "r");
const read = (offsetBytes: number, byteLength: number): Uint8Array => {
  const bytes = Buffer.allocUnsafe(byteLength);
  const got = readSync(fd, bytes, 0, byteLength, offsetBytes);
  if (got !== byteLength) throw new Error(`${weightsPath}: read ${got} of ${byteLength}`);
  return new Uint8Array(bytes.buffer, bytes.byteOffset, byteLength);
};

/**
 * The two tables the forward reads on the host.
 *
 * `embed_tokens` is q8 in the conversion, so it is dequantised here: a gather
 * of 76 rows out of 151,936 is not worth a GPU pass, and quantising the table
 * and then reading it back through its scales is what the conversion already
 * decided.
 */
function hostTable(name: string, rows: number, cols: number): Float32Array {
  const entry = manifest.tensors.find((t) => t.name === name);
  if (!entry) throw new Error(`no tensor named "${name}"`);
  if (entry.kind !== "q8") {
    const bytes = read(entry.offset * 4, entry.count * 4);
    return new Float32Array(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
  }
  const scaleEntry = manifest.tensors.find((t) => t.name === `${name}.scale`);
  if (!scaleEntry) throw new Error(`no scale for "${name}"`);
  const words = new Uint32Array(
    (() => { const b = read(entry.offset * 4, entry.count * 4); return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength); })(),
  );
  const scaleBytes = read(scaleEntry.offset * 4, scaleEntry.count * 4);
  const scales = new Float32Array(scaleBytes.buffer.slice(scaleBytes.byteOffset, scaleBytes.byteOffset + scaleBytes.byteLength));
  const perRow = Math.ceil(cols / 4);
  const out = new Float32Array(rows * cols);
  for (let r = 0; r < rows; r += 1) {
    for (let c = 0; c < cols; c += 1) {
      const word = words[r * perRow + (c >> 2)]!;
      // Four codes a word, least-significant byte first, sign-extended.
      const code = ((word >> ((c & 3) * 8)) & 0xff) << 24 >> 24;
      out[r * cols + c] = code * scales[r]!;
    }
  }
  return out;
}

const t = manifest.textConfig;
const v = manifest.visionConfig;
console.log(
  `weights ${(statSync(weightsPath).size / 1e9).toFixed(2)} GB (${manifest.dtype}), ` +
    `${manifest.layers} text layers of ${t.hidden_size}, ${v.depth} vision blocks`,
);

const embedTokens = hostTable("embed_tokens.weight", 151936, t.hidden_size);
const posEmbed = hostTable("visual.pos_embed.weight", v.num_position_embeddings, v.hidden_size);

// The reference, through this port's own processor.
const pixelBytes = readFileSync(`${goldenDir}/pixels.bin`);
const pixels = new Uint8Array(pixelBytes.buffer, pixelBytes.byteOffset, pixelBytes.byteLength);
const patches = patchify([pixels], golden.imageSize, golden.imageSize, manifest.processor);
if (patches.grid[0] !== golden.grid[0] || patches.grid[1] !== golden.grid[1] || patches.grid[2] !== golden.grid[2]) {
  console.error(`the patchify produced grid ${patches.grid} and the golden says ${golden.grid}`);
  process.exit(1);
}

// The grid the model builds for itself, rebuilt here and **held to the
// golden's own** — because a wrong one is a working model conditioned on the
// wrong geometry, and it reads as quantisation noise.
const modalities = golden.tokenIds.map((id) => (id === 151655 ? 1 : id === 151656 ? 2 : 0));
const positions = qwen3vlPositionGrid(
  modalities, [{ grid: golden.grid, modality: 1 }], manifest.processor.mergeSize);
for (const axis of ["t", "h", "w"] as const) {
  const mine = positions[axis];
  const theirs = golden.positionIds[axis];
  const at = mine.findIndex((value, i) => value !== theirs[i]);
  if (at >= 0) {
    console.error(`the ${axis} grid differs at token ${at}: ${mine[at]} against ${theirs[at]}`);
    process.exit(1);
  }
}
console.log(`position grid matches the model's own on all three axes`);

const device = await createResidentDevice();
if (!device) {
  console.error("verify-conditioner: no adapter");
  process.exit(2);
}

const uploadStart = performance.now();
const conditioner = await ConditionerGpu.create(
  device, conditionerKernels(), manifest, read, { embedTokens, posEmbed });
closeSync(fd);
console.log(`uploaded in ${((performance.now() - uploadStart) / 1000).toFixed(1)} s`);

const started = performance.now();
const got = await conditioner.forward({
  tokenIds: Int32Array.from(golden.tokenIds),
  positions,
  patches: patches.pixelValues,
  grids: [golden.grid],
  visualRuns: golden.visualRuns,
});
const took = performance.now() - started;

console.log(
  `${golden.seq} tokens in ${took.toFixed(0)} ms — ${conditioner.dispatches} dispatches, ` +
    `${conditioner.submitMs.toFixed(0)} ms in the queue, ${conditioner.readbackMs.toFixed(0)} ms reading back, ` +
    `${conditioner.recordMs.toFixed(0)} ms recording`,
);

const want = f32(stage === null ? `${goldenDir}/hidden.bin` : `${goldenDir}/hidden.${stage}.bin`);
if (got.length !== want.length) {
  console.error(`length ${got.length} against ${want.length}`);
  process.exit(1);
}
let worst = 0;
let sum = 0;
let peak = 0;
let nonFinite = 0;
for (let i = 0; i < want.length; i += 1) {
  if (!Number.isFinite(got[i]!)) nonFinite += 1;
  const d = Math.abs(got[i]! - want[i]!);
  sum += d * d;
  if (d > worst) worst = d;
  peak = Math.max(peak, Math.abs(want[i]!));
}
// Counted, never assumed away: `Math.abs(NaN - x) > worst` is false, so a
// wholly non-finite output reports a perfect score. #210 printed exactly that.
if (nonFinite) {
  console.error(`${nonFinite} of ${want.length} values are not finite (first: ${got[0]})`);
  process.exit(1);
}
console.log(
  `hidden_states[${stage ?? golden.layer}]: worst ${worst.toExponential(3)}  rms ${Math.sqrt(sum / want.length).toExponential(3)}  ` +
    `signal peak ${peak.toFixed(1)}  -> ${((worst / peak) * 100).toFixed(2)}% of peak`,
);

conditioner.destroy();
device.destroy();
