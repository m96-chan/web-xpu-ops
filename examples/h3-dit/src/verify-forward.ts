/**
 * The GPU DiT against the velocity MiniMax-H3's own code produced.
 *
 * Issue #210. A script rather than a test: it holds **20.08 GB** of weights on
 * the device, which is not something to do inside a vitest worker beside 1,800
 * other tests (#107 — real-model-scale work in that process crashes the Dawn
 * binding). `examples/h3-video/src/verify-decode.ts` is the same arrangement.
 *
 *     npx tsx examples/h3-dit/src/verify-forward.ts --dir ~/h3-dit-gpu --golden ~/h3-dit-real
 */
import { closeSync, openSync, readFileSync, readSync, statSync } from "node:fs";
import { createResidentDevice } from "../../../harness/resident.js";
import { buildPackedSequence, buildRowTimesteps } from "./layout.js";
import { DitGpu, type DitManifest } from "./model-gpu.js";
import { ditKernels } from "./kernels-node.js";

const arg = (name: string): string | undefined => {
  const at = process.argv.indexOf(name);
  return at >= 0 ? process.argv[at + 1] : undefined;
};
const dir = arg("--dir");
const goldenDir = arg("--golden");
if (!dir || !goldenDir) {
  console.error("verify-forward: --dir (convert_dit.py output) and --golden (gen_real_forward_golden.py output) are required");
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
    numTextTokens: number; numLatentFrames: number; latentHeight: number; latentWidth: number;
    numAudioLatents: number; seq: number;
    tokenTags: number[]; timestepIndices: number[];
    videoIndices: number[]; audioIndices: number[]; textIndices: number[];
  };
};

if (golden.layers !== manifest.layers) {
  console.error(`the golden ran ${golden.layers} blocks and the conversion holds ${manifest.layers}`);
  process.exit(2);
}

const layout = buildPackedSequence({
  numTextTokens: golden.layout.numTextTokens,
  numLatentFrames: golden.layout.numLatentFrames,
  latentHeight: golden.layout.latentHeight,
  latentWidth: golden.layout.latentWidth,
  numAudioLatents: golden.layout.numAudioLatents,
  patchSize: manifest.config.patch_size,
});

// The layout is rebuilt here rather than read from the golden, so a divergence
// between this port's `buildPackedSequence` and upstream's shows up as a
// mismatch here instead of as a wrong velocity forty blocks later.
const same = (a: ArrayLike<number>, b: ArrayLike<number>): boolean =>
  a.length === b.length && [...a].every((v, i) => v === b[i]);
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
const { timestepIndices } = buildRowTimesteps(
  layout, schedule.video[golden.stepIndex]!, schedule.audio[golden.stepIndex]!,
);
if (!same(timestepIndices, golden.layout.timestepIndices)) {
  console.error("the rebuilt per-row timesteps disagree with the golden's");
  process.exit(1);
}
layout.timestepIndices = timestepIndices;

const device = await createResidentDevice();
if (!device) {
  console.error("verify-forward: no adapter");
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
