/**
 * What the decoder does to a latent that does not change in time.
 *
 * Issue #216's second half. R2V's decoded frames carry a colour fringe that is
 * absent from the first frame and grows with the frame index — the same in
 * every run measured, and present before `59ac5b1` as well as after, so it is
 * not the encoder chunking.
 *
 * **The decoder is held to the model at 4.530e-6 on a golden with two latent
 * frames.** A generation uses twelve. Two is where a temporal stride error is
 * degenerate: the frame stride and the channel stride cannot be told apart, an
 * off-by-one in a causal window has nowhere to land, and every accumulating
 * drift has one step to accumulate over. It is the same shape of blind spot
 * that let the encoder's eight-frame golden miss `encode_temporal` entirely.
 *
 * So this asks a question the golden cannot: **if the latent is identical for
 * every frame, is the output?** It has to be, up to the causal padding at the
 * very start — nothing in a convolution over a constant signal may drift. No
 * golden is needed to say that, and no reference either; the invariant is the
 * test.
 *
 *     npx tsx examples/h3-video/src/probe-temporal.ts --dir ~/h3-work/h3-video-q8 --frames 12
 */
import { closeSync, openSync, readFileSync, readSync } from "node:fs";
import { createResidentDevice } from "../../../harness/resident.js";
import { VideoDecoderGpu, denormalise, type VideoDecoderManifest } from "./decoder-gpu.js";
import { videoKernels } from "./kernels-node.js";

const arg = (name: string): string | undefined => {
  const at = process.argv.indexOf(name);
  return at >= 0 ? process.argv[at + 1] : undefined;
};
const dir = arg("--dir");
if (!dir) {
  console.error("probe-temporal: --dir is required (a converted decoder)");
  process.exit(2);
}
const stem = arg("--stem") ?? "decoder.q8";
const latentFrames = Number(arg("--frames") ?? 12);
const size = Number(arg("--size") ?? 16);

const manifest = JSON.parse(
  readFileSync(`${dir}/${stem}.manifest.json`, "utf8"),
) as VideoDecoderManifest;
const z = manifest.config.in_channels;

const device = await createResidentDevice();
if (!device) {
  console.error("probe-temporal: no adapter");
  process.exit(2);
}
const fd = openSync(`${dir}/${stem}.bin`, "r");
const read = (offsetBytes: number, byteLength: number): Uint8Array => {
  const bytes = Buffer.allocUnsafe(byteLength);
  const got = readSync(fd, bytes, 0, byteLength, offsetBytes);
  if (got !== byteLength) throw new Error(`${stem}.bin: read ${got} of ${byteLength}`);
  return new Uint8Array(bytes.buffer, bytes.byteOffset, byteLength);
};
const decoder = await VideoDecoderGpu.create(device, videoKernels(), manifest, read);
closeSync(fd);

// One frame's worth of latent, repeated. Structured rather than constant in
// space, so the picture is something a spatial bug could also disturb — the
// invariant being checked is only about *time*.
const plane = size * size;
const one = new Float32Array(z * plane);
for (let c = 0; c < z; c += 1) {
  for (let i = 0; i < plane; i += 1) {
    one[c * plane + i] = Math.sin((i % size) * 0.7 + c) * 0.8 + Math.cos(Math.floor(i / size) * 0.5 - c) * 0.6;
  }
}
const latent = new Float32Array(z * latentFrames * plane);
for (let c = 0; c < z; c += 1) {
  for (let t = 0; t < latentFrames; t += 1) {
    latent.set(one.subarray(c * plane, (c + 1) * plane), (c * latentFrames + t) * plane);
  }
}

const raw = await decoder.decode(latent, [latentFrames, size, size]);
const shown = denormalise(raw, manifest.config.out_channels, manifest.pixelMean, manifest.pixelStd);
const channels = manifest.config.out_channels;
const height = size * 16;
const width = size * 16;
const frames = shown.length / (channels * height * width);
const perFrame = channels * height * width;
console.log(`latent ${z}x${latentFrames}x${size}x${size}, identical every frame -> ${frames} pixel frames`);

/** Mean absolute difference between frame `t` and frame `t - 1`, per channel. */
const step = (t: number, channel: number): number => {
  let sum = 0;
  const at = t * perFrame + channel * height * width;
  const before = (t - 1) * perFrame + channel * height * width;
  for (let i = 0; i < height * width; i += 1) sum += Math.abs(shown[at + i]! - shown[before + i]!);
  return (sum / (height * width)) * 255;
};

console.log(`${"frame".padStart(6)}  ${"R".padStart(8)}  ${"G".padStart(8)}  ${"B".padStart(8)}`);
let worst = 0;
for (let t = 1; t < frames; t += 1) {
  const d = [0, 1, 2].map((c) => step(t, c));
  worst = Math.max(worst, ...d);
  console.log(
    `${String(t).padStart(6)}  ${d.map((v) => v.toFixed(3).padStart(8)).join("  ")}` +
      (t < 4 ? "   <- inside the causal padding" : ""),
  );
}
console.log(
  `\nworst frame-to-frame difference past the padding: ${worst.toFixed(3)} of 255 levels.\n` +
    "A constant latent must decode to constant frames; anything here is the decoder's own.",
);

decoder.destroy();
device.destroy();
