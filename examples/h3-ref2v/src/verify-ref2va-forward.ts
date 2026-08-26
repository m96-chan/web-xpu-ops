/**
 * The GPU DiT on a **`ref2va`** packed sequence, against `transformer_ref`'s own
 * velocity.
 *
 * Issue #212. `examples/h3-dit/src/verify-forward.ts` is the `t2va` half and
 * this is the same comparison for the layout that has reference rows — which
 * `t2va` cannot exercise, because everything that only `ref2va` does is
 * downstream of `num_condition_video_rows` being nonzero:
 *
 * - `[text | reference blocks | target audio | target video]`, so the video
 *   rows are not one contiguous generated run.
 * - **Three or four distinct noise levels** rather than one or two, since the
 *   anchors sit at `max(t, 0.999)`. A conversion built for `t2va` indexes past
 *   the end of its own modulation table on the first step that needs a third,
 *   and the failure is a bind group whose offset is larger than the buffer.
 * - A reference's vision block is tagged **video** among the text rows.
 *
 *     npx tsx examples/h3-ref2v/src/verify-ref2va-forward.ts \
 *       --dir ~/h3-work/h3-ref-gpu-4 --golden ~/h3-work/h3-ref2va-forward
 */
import { closeSync, openSync, readFileSync, readSync, statSync } from "node:fs";
import { createResidentDevice } from "../../../harness/resident.js";
import { DitGpu, type DitManifest } from "../../h3-dit/src/model-gpu.js";
import { ditKernels } from "../../h3-dit/src/kernels-node.js";
import { buildRef2vaRowTimesteps, buildRef2vaSequence } from "./layout.js";

const arg = (name: string): string | undefined => {
  const at = process.argv.indexOf(name);
  return at >= 0 ? process.argv[at + 1] : undefined;
};
const dir = arg("--dir");
const goldenDir = arg("--golden");
if (!dir || !goldenDir) {
  console.error("verify-ref2va-forward: --dir (convert_dit.py --workflow ref2va output) and --golden (gen_real_ref2va_forward_golden.py output) are required");
  process.exit(2);
}

const f32 = (path: string): Float32Array => {
  const buffer = readFileSync(path);
  return new Float32Array(buffer.buffer, buffer.byteOffset, buffer.byteLength / 4);
};

const manifest = JSON.parse(readFileSync(`${dir}/dit.manifest.json`, "utf8")) as DitManifest;
// `--layers N` truncates the stack, for bisecting a divergence against a golden
// generated with the same `--layers`.
const layersArg = arg("--layers");
if (layersArg) manifest.layers = Number(layersArg);
// `--max-workgroups N` lowers the grid ceiling so the chunking runs at a
// sequence length that fits in memory. Issue #211: 576x320 needs 75,600
// workgroups for one feed-forward dispatch and the device allows 65,535, but
// reproducing that honestly needs 1,350 rows and 24 GB of weights. Squeezing
// the ceiling instead exercises the same split against the same golden.
const maxWorkgroupsArg = arg("--max-workgroups");
const golden = JSON.parse(readFileSync(`${goldenDir}/golden.json`, "utf8")) as {
  layers: number;
  steps: number;
  stepIndex: number;
  layout: {
    numTextTokens: number; textTokenTags: number[];
    referenceFrames: number; referenceSize: number;
    numLatentFrames: number; latentHeight: number; latentWidth: number;
    numAudioLatents: number; seq: number;
    numConditionVideoRows: number; numConditionAudioRows: number;
    tokenTags: number[]; timestep: number[]; timestepIndices: number[];
    videoIndices: number[]; audioIndices: number[]; textIndices: number[];
  };
};

if (golden.layers !== manifest.layers) {
  console.error(`the golden ran ${golden.layers} blocks and the conversion holds ${manifest.layers}`);
  process.exit(2);
}

const layout = buildRef2vaSequence({
  numTextTokens: golden.layout.numTextTokens,
  textTokenTags: golden.layout.textTokenTags,
  references: [{ kind: "image", hasAudio: false }],
  visualGeometry: [[golden.layout.referenceFrames, golden.layout.referenceSize, golden.layout.referenceSize]],
  audioRowCounts: [],
  numLatentFrames: golden.layout.numLatentFrames,
  latentHeight: golden.layout.latentHeight,
  latentWidth: golden.layout.latentWidth,
  numAudioLatents: golden.layout.numAudioLatents,
  patchSize: manifest.config.patch_size,
});
if (layout.numReferenceVideoRows !== golden.layout.numConditionVideoRows) {
  console.error(
    `this port reserved ${layout.numReferenceVideoRows} reference video rows and the golden has ` +
      `${golden.layout.numConditionVideoRows}`,
  );
  process.exit(1);
}

// The layout is rebuilt here rather than read from the golden, so a divergence
// between this port's `buildPackedSequence` and upstream's shows up as a
// mismatch here instead of as a wrong velocity forty blocks later.
const same = (a: ArrayLike<number>, b: ArrayLike<number>): boolean =>
  a.length === b.length && Array.from(a).every((v, i) => v === b[i]);
if (layout.seq !== golden.layout.seq || !same(layout.tokenTags, golden.layout.tokenTags)
  || !same(layout.videoIndices, golden.layout.videoIndices)) {
  console.error("the rebuilt layout disagrees with the golden's");
  process.exit(1);
}

const schedule = manifest.schedules[String(golden.steps)];
if (!schedule) {
  console.error(`this conversion has no tables for ${golden.steps} steps (it has ${manifest.stepCounts.join(", ")})`);
  process.exit(2);
}
const plan = buildRef2vaRowTimesteps(
  layout, layout.numReferenceVideoRows, layout.numReferenceAudioRows,
  schedule.video[golden.stepIndex]!, schedule.audio[golden.stepIndex]!,
);
if (!same(plan.timestepIndices, golden.layout.timestepIndices)) {
  console.error("the rebuilt per-row timesteps disagree with the golden's");
  process.exit(1);
}
if (!same(plan.timestep, golden.layout.timestep.map((t) => Math.fround(t)))) {
  console.error(`this port's noise levels are [${[...plan.timestep]}] and the golden's are [${golden.layout.timestep}]`);
  process.exit(1);
}

// **The conversion's own table, not this port's ordering.** `torch.unique`
// produces indices into the levels *this step* carries; the tables were
// evaluated at conversion time over a fixed set, and a level nobody built a
// table for has to be an error rather than an index into whatever follows.
const declared = schedule.levels;
if (!declared) {
  console.error("this conversion predates the `levels` field — re-run convert_dit.py with --workflow ref2va");
  process.exit(2);
}
const toTable = Array.from(plan.timestep, (level) => {
  const at = declared[golden.stepIndex]!.findIndex((candidate) => Math.fround(candidate) === level);
  if (at < 0) {
    console.error(`no modulation table at t=${level}; this conversion has [${declared[golden.stepIndex]!.join(", ")}]`);
    process.exit(1);
  }
  return at;
});
layout.timestepIndices = Int32Array.from(plan.timestepIndices, (own) => toTable[own]!);

const device = await createResidentDevice();
if (!device) {
  console.error("verify-ref2va-forward: no adapter");
  process.exit(2);
}

const openReader = (path: string) => {
  const fd = openSync(path, "r");
  return {
    // Bytes, never a `Float32Array`: q8 words would go through f32's NaN
    // canonicalisation on the way to the device.
    read: (offsetBytes: number, byteLength: number): Uint8Array => {
      const bytes = Buffer.allocUnsafe(byteLength);
      const got = readSync(fd, bytes, 0, byteLength, offsetBytes);
      if (got !== byteLength) throw new Error(`${path}: read ${got} bytes where ${byteLength} were wanted`);
      return new Uint8Array(bytes.buffer, bytes.byteOffset, byteLength);
    },
    close: () => closeSync(fd),
  };
};

const weightsPath = `${dir}/${manifest.dtype === "q8" ? "dit.q8.bin" : "dit.bin"}`;
console.log(
  `weights ${(statSync(weightsPath).size / 1e9).toFixed(2)} GB (${manifest.dtype}), ` +
    `tables ${(statSync(`${dir}/adaln.bin`).size / 1e9).toFixed(2)} GB, ${manifest.layers} blocks`,
);
const weights = openReader(weightsPath);
const tables = openReader(`${dir}/adaln.bin`);

const uploadStart = performance.now();
const dit = await DitGpu.create(device, ditKernels(), manifest, weights.read, tables.read);
if (maxWorkgroupsArg) {
  dit.maxWorkgroupsPerDimension = Number(maxWorkgroupsArg);
  console.log(`grid ceiling lowered to ${dit.maxWorkgroupsPerDimension} workgroups — the chunked path`);
}
weights.close();
tables.close();
console.log(`uploaded in ${((performance.now() - uploadStart) / 1000).toFixed(1)} s`);

const started = performance.now();
const got = await dit.forward({
  video: f32(`${goldenDir}/input.video.bin`),
  audio: f32(`${goldenDir}/input.audio.bin`),
  text: f32(`${goldenDir}/input.text.bin`),
  layout,
  steps: golden.steps,
  stepIndex: golden.stepIndex,
});
const took = performance.now() - started;

console.log(
  `one step over ${layout.seq} rows in ${took.toFixed(0)} ms — ` +
    `${dit.dispatches} dispatches, ${dit.submitMs.toFixed(0)} ms in the queue, ` +
    `${dit.readbackMs.toFixed(0)} ms in the final submit and readback, ${dit.recordMs.toFixed(0)} ms recording`,
);

function compare(name: string, got: Float32Array, want: Float32Array): number {
  if (got.length !== want.length) {
    console.error(`${name}: ${got.length} values against ${want.length}`);
    process.exit(1);
  }
  let worst = 0;
  let sum = 0;
  let peak = 0;
  // **Counted, not assumed away.** `Math.abs(NaN - x) > worst` is *false*, so a
  // wholly NaN output reports a worst difference of exactly zero -- which this
  // script printed once, next to an RMS of NaN, and it read like a pass. A
  // comparison that cannot say "these are not numbers" is not a comparison.
  let nonFinite = 0;
  for (let i = 0; i < want.length; i += 1) {
    if (!Number.isFinite(got[i]!)) nonFinite += 1;
    const d = Math.abs(got[i]! - want[i]!);
    sum += d * d;
    if (d > worst) worst = d;
    peak = Math.max(peak, Math.abs(want[i]!));
  }
  if (nonFinite) {
    console.error(`${name}: ${nonFinite} of ${want.length} values are not finite (first: ${got[0]})`);
    process.exit(1);
  }
  console.log(
    `${name}: worst ${worst.toExponential(3)}  rms ${Math.sqrt(sum / want.length).toExponential(3)}  ` +
      `signal peak ${peak.toFixed(4)}`,
  );
  return worst / peak;
}

const videoRelative = compare("video velocity", got.video, f32(`${goldenDir}/output.video.bin`));
const audioRelative = compare("audio velocity", got.audio, f32(`${goldenDir}/output.audio.bin`));
console.log(`relative to peak: video ${(videoRelative * 100).toFixed(2)}%, audio ${(audioRelative * 100).toFixed(2)}%`);

// The same inputs twice, and the two must agree **exactly**. Scratch buffers
// are pooled, so the second forward is the first one to see a *used* buffer --
// anything relied on being zero rather than written is right the first time and
// wrong after. `examples/h3-video` found exactly that with its cls token.
const second = await dit.forward({
  video: f32(`${goldenDir}/input.video.bin`),
  audio: f32(`${goldenDir}/input.audio.bin`),
  text: f32(`${goldenDir}/input.text.bin`),
  layout,
  steps: golden.steps,
  stepIndex: golden.stepIndex,
});
let repeat = 0;
for (let i = 0; i < second.video.length; i += 1) repeat = Math.max(repeat, Math.abs(second.video[i]! - got.video[i]!));
console.log(`same inputs twice: worst difference ${repeat.toExponential(3)}`);
if (repeat !== 0) {
  console.error("the second forward disagreed with the first — a pooled buffer is being read before it is written");
  process.exit(1);
}

dit.destroy();
device.destroy();
