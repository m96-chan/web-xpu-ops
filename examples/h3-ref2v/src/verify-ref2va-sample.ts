/**
 * A whole `ref2va` sampling loop against the one upstream ran.
 *
 * Issue #216. `verify-ref2va-forward.ts` holds **one forward** to the model and
 * this port matches it to 0.68% of peak; that is not the same claim as fifteen
 * of them matching. The loop adds the scheduler, a row-timestep plan that
 * changes every step, and the rule that the conditioning rows are re-imposed by
 * never being written — and R2V's output flickers at 2.7x `t2va`'s with a
 * latent 31% hot, which everything a single forward can see has failed to
 * explain.
 *
 * Both sides start from the **same** initial state, read out of the golden: a
 * port that starts somewhere else is not being compared to anything.
 *
 *     npx tsx examples/h3-ref2v/src/verify-ref2va-sample.ts \
 *       --dir ~/h3-work/h3-ref-gpu-4 --golden ~/h3-work/h3-ref2va-sample
 */
import { closeSync, openSync, readFileSync, readSync, statSync } from "node:fs";
import { createResidentDevice } from "../../../harness/resident.js";
import { DitGpu, type DitManifest } from "../../h3-dit/src/model-gpu.js";
import { ditKernels } from "../../h3-dit/src/kernels-node.js";
import { buildRef2vaRowTimesteps, buildRef2vaSequence } from "./layout.js";
import { setTimesteps, step as schedulerStep } from "../../h3-dit/src/scheduler.js";

const arg = (name: string): string | undefined => {
  const at = process.argv.indexOf(name);
  return at >= 0 ? process.argv[at + 1] : undefined;
};
const dir = arg("--dir");
const goldenDir = arg("--golden");
if (!dir || !goldenDir) {
  console.error("verify-ref2va-sample: --dir and --golden (gen_real_ref2va_sample_golden.py output) are required");
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
  layout: {
    numTextTokens: number; textTokenTags: number[];
    referenceFrames: number; referenceSize: number;
    numLatentFrames: number; latentHeight: number; latentWidth: number;
    numAudioLatents: number; seq: number;
    numConditionVideoRows: number; numConditionAudioRows: number;
    tokenTags: number[];
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
const declared = schedule.levels;
if (!declared) {
  console.error("this conversion predates the `levels` field — re-run convert_dit.py with --workflow ref2va");
  process.exit(2);
}

const device = await createResidentDevice();
if (!device) {
  console.error("verify-ref2va-sample: no adapter");
  process.exit(2);
}

const openReader = (path: string) => {
  const fd = openSync(path, "r");
  return {
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
const dit = await DitGpu.create(device, ditKernels(), manifest, weights.read, tables.read);
weights.close();
tables.close();

const video = setTimesteps(golden.steps, 12);
const audio = setTimesteps(golden.steps, 3);
if (video.timesteps.length !== schedule.video.length) {
  console.error("the schedule this port builds is a different length from the one the tables were built for");
  process.exit(1);
}

// **The same initial state upstream started from**, not one this port drew.
let videoRows = f32(`${goldenDir}/input.video.bin`);
let audioRows = f32(`${goldenDir}/input.audio.bin`);
const text = f32(`${goldenDir}/input.text.bin`);
const perRow = manifest.config.in_channels
  * manifest.config.patch_size[0] * manifest.config.patch_size[1] * manifest.config.patch_size[2];
const anchored = layout.numReferenceVideoRows * perRow;
const anchorsIn = videoRows.slice(0, anchored);

const started = performance.now();
for (let i = 0; i < video.timesteps.length; i += 1) {
  const plan = buildRef2vaRowTimesteps(
    layout, layout.numReferenceVideoRows, layout.numReferenceAudioRows,
    video.timesteps[i]!, audio.timesteps[i]!);
  const toTable = Array.from(plan.timestep, (level) => {
    const at = declared[i]!.findIndex((candidate) => Math.fround(candidate) === level);
    if (at < 0) {
      console.error(`step ${i}: no modulation table at t=${level}`);
      process.exit(1);
    }
    return at;
  });
  layout.timestepIndices = Int32Array.from(plan.timestepIndices, (own) => toTable[own]!);
  const velocity = await dit.forward({
    video: videoRows, audio: audioRows, text, layout, steps: golden.steps, stepIndex: i,
  });
  const stepped = schedulerStep(
    video, velocity.video.subarray(anchored), video.timesteps[i]!, videoRows.subarray(anchored), i);
  const next = new Float32Array(videoRows.length);
  next.set(videoRows.subarray(0, anchored), 0);
  next.set(stepped, anchored);
  videoRows = next;
  audioRows = schedulerStep(audio, velocity.audio, audio.timesteps[i]!, audioRows, i);
  const rms = Math.sqrt(videoRows.reduce((s, v) => s + v * v, 0) / videoRows.length);
  console.log(`  step ${i + 1}/${video.timesteps.length}  t=${video.timesteps[i]!.toFixed(4)}  rows ${rms.toFixed(4)}`);
}
console.log(`sampled in ${((performance.now() - started) / 1000).toFixed(1)} s`);

// The anchors must be exactly what went in — the rule the loop is built on.
let anchorDrift = 0;
for (let i = 0; i < anchored; i += 1) {
  anchorDrift = Math.max(anchorDrift, Math.abs(videoRows[i]! - anchorsIn[i]!));
}
console.log(`anchor drift: ${anchorDrift.toExponential(3)}`);

const got = { video: videoRows, audio: audioRows };

/**
 * The temporal roughness of the generated rows, in this port and in upstream's
 * own output.
 *
 * Issue #216: R2V's frames flicker at 2.7x `t2va`'s, and upstream's own output
 * does *not* get rougher when a reference is added — measured at -2.3%. So the
 * number to compare is not the worst element, it is how much the latent moves
 * between frames.
 */
{
  const perRowValues = perRow;
  const generated = (x: Float32Array): number[] => {
    const rows = (x.length - anchored) / perRowValues;
    const frames = golden.layout.numLatentFrames;
    const perFrame = (rows / frames) * perRowValues;
    const out: number[] = [];
    for (let f = 1; f < frames; f += 1) {
      let sum = 0;
      for (let i = 0; i < perFrame; i += 1) {
        sum += Math.abs(x[anchored + f * perFrame + i]! - x[anchored + (f - 1) * perFrame + i]!);
      }
      out.push(sum / perFrame);
    }
    return out;
  };
  const rmsOf = (x: Float32Array): number => {
    let s = 0;
    for (let i = anchored; i < x.length; i += 1) s += x[i]! * x[i]!;
    return Math.sqrt(s / (x.length - anchored));
  };
  const wantVideo = f32(`${goldenDir}/output.video.bin`);
  const mine = generated(videoRows);
  const theirs = generated(wantVideo);
  const mean = (a: number[]): number => a.reduce((s, v) => s + v, 0) / a.length;
  console.log(
    `latent frame-to-frame: this port ${(mean(mine) / rmsOf(videoRows)).toFixed(4)} relative, ` +
      `upstream ${(mean(theirs) / rmsOf(wantVideo)).toFixed(4)} ` +
      `(${((mean(mine) / rmsOf(videoRows)) / (mean(theirs) / rmsOf(wantVideo)) * 100 - 100).toFixed(1)}%)`,
  );
}

function compare(name: string, gotValues: Float32Array, want: Float32Array): number {
  if (gotValues.length !== want.length) {
    console.error(`${name}: ${gotValues.length} values against ${want.length}`);
    process.exit(1);
  }
  let worst = 0;
  let sum = 0;
  let peak = 0;
  let nonFinite = 0;
  for (let i = 0; i < want.length; i += 1) {
    if (!Number.isFinite(gotValues[i]!)) nonFinite += 1;
    const d = Math.abs(gotValues[i]! - want[i]!);
    sum += d * d;
    if (d > worst) worst = d;
    peak = Math.max(peak, Math.abs(want[i]!));
  }
  if (nonFinite) {
    console.error(`${name}: ${nonFinite} of ${want.length} values are not finite`);
    process.exit(1);
  }
  const wantRms = Math.sqrt(want.reduce((s, v) => s + v * v, 0) / want.length);
  const gotRms = Math.sqrt(gotValues.reduce((s, v) => s + v * v, 0) / gotValues.length);
  console.log(
    `${name}: worst ${worst.toExponential(3)}  rms ${Math.sqrt(sum / want.length).toExponential(3)}  ` +
      `peak ${peak.toFixed(4)}  |  rms got ${gotRms.toFixed(4)} against ${wantRms.toFixed(4)} ` +
      `(${((gotRms / wantRms - 1) * 100).toFixed(1)}%)`,
  );
  return worst / peak;
}

const videoRelative = compare("video velocity", got.video, f32(`${goldenDir}/output.video.bin`));
const audioRelative = compare("audio velocity", got.audio, f32(`${goldenDir}/output.audio.bin`));
console.log(`relative to peak: video ${(videoRelative * 100).toFixed(2)}%, audio ${(audioRelative * 100).toFixed(2)}%`);

dit.destroy();
device.destroy();
