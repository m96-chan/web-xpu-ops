/**
 * The GPU conditioner against the hidden state Qwen3-VL's own code produced.
 *
 * Issue #212. A script rather than a test: it holds **25.78 GB** of weights on
 * the device, which is not something to do inside a vitest worker beside 2,000
 * other tests. `examples/h3-dit/src/verify-forward.ts` is the same arrangement.
 *
 *     npx tsx examples/h3-ref2v/src/verify-conditioner.ts \
 *       --dir ~/h3-cond-gpu --golden ~/h3-cond-real
 */
import { closeSync, openSync, readFileSync, readSync, statSync, writeFileSync } from "node:fs";
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
  imageSize: number; visualRuns: [number, number][];
  grid: Grid; seq: number; hidden: [number, number]; layer: number; stages?: number[];
  quantised?: boolean;
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

// **`hidden_states[k]` is layer `k`'s input**, so `k` layers run to produce it,
// not `k + 1`. A conversion that keeps one layer too many still produces a
// well-formed activation of the right shape, and against a stack whose last
// rows are massive activations it reads as quantisation noise: the run that
// found this reported 96% of peak either way, and only the *median row* moved
// -- 1.1% against 24.9%.
if (stage === null && manifest.layers !== golden.layer) {
  console.error(
    `this conversion evaluates ${manifest.layers} text layers and hidden_states[${golden.layer}] is the ` +
      `input to layer ${golden.layer}, which ${golden.layer} layers produce — reconvert`,
  );
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
console.log(
  golden.quantised
    ? "golden: the released weights round-tripped through this converter's int8"
    : "golden: the released weights in bf16 — see the README on which number this can and cannot settle",
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

/**
 * The same divergence per row, split by what the row is.
 *
 * Worst-over-peak is one number about one element, and in this stack that
 * element is a **massive activation**: from layer 43 a few visual tokens grow
 * by a factor of a hundred, and which tokens is a near-tie. A row-wise figure
 * says whether the other seventy-five rows are right, which is the question.
 */
const hidden = t.hidden_size;
const isVisual = new Set<number>();
for (const [start, length] of golden.visualRuns) {
  for (let i = 0; i < length; i += 1) isVisual.add(start + i);
}
const norms: number[] = [];
const relative = (visual: boolean): { median: number; max: number } => {
  const rows: number[] = [];
  for (let r = 0; r < golden.seq; r += 1) {
    if (isVisual.has(r) !== visual) continue;
    let dn = 0;
    let wn = 0;
    for (let c = 0; c < hidden; c += 1) {
      const d = got[r * hidden + c]! - want[r * hidden + c]!;
      dn += d * d;
      wn += want[r * hidden + c]! ** 2;
    }
    rows.push(Math.sqrt(dn / wn));
  }
  rows.sort((a, b) => a - b);
  return { median: rows[Math.floor(rows.length / 2)] ?? 0, max: rows[rows.length - 1] ?? 0 };
};
for (let r = 0; r < golden.seq; r += 1) {
  let n = 0;
  for (let c = 0; c < hidden; c += 1) n += got[r * hidden + c]! ** 2;
  norms.push(Math.sqrt(n));
}
const percent = (x: number): string => `${(x * 100).toFixed(3)}%`;
for (const visual of [false, true]) {
  const { median, max } = relative(visual);
  console.log(`  ${visual ? "visual" : "text  "} rows: median ${percent(median)}, worst row ${percent(max)}`);
}
const ranked = norms.map((n, r) => [r, n] as const).sort((a, b) => b[1] - a[1]).slice(0, 3);
console.log(`  the port's largest rows: ${ranked.map(([r, n]) => `${r}:${n.toFixed(0)}`).join("  ")}`);

// `--dump` writes what the port produced, so the same activation can be held
// to the *other* reference without a second 26 GB upload.
const dump = arg("--dump");
if (dump) {
  writeFileSync(dump, Buffer.from(got.buffer, got.byteOffset, got.byteLength));
  console.log(`wrote ${dump}`);
}

conditioner.destroy();
device.destroy();
