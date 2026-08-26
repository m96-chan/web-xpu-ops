/**
 * The GPU encoder against `EncoderFCN3D` and `quant_conv` — the model's own
 * output, not this repository's CPU port.
 *
 * Issue #214. `verify-encode.ts` holds the CPU version to the same golden and
 * takes **120.5 s** on the 8x32x32 clip; this is the same comparison against
 * the same `moments.bin`. Held to the *reference*, never to the other kernel:
 * two ports that agree with each other and not with the model are two wrong
 * answers.
 *
 * A script rather than a test because it needs a device and 0.72 GB of
 * weights, the same arrangement `verify-encode.ts` and
 * `examples/h3-video/src/verify-decode.ts` use.
 *
 *     npx tsx examples/h3-encoder/src/verify-encode-gpu.ts --dir ~/h3-work/h3-encoder-whole
 */
import { closeSync, openSync, readFileSync, readSync, statSync } from "node:fs";
import { createResidentDevice } from "../../../harness/resident.js";
import { EncoderGpu, channelPlanOf, latentMeanOf, type EncoderGpuManifest } from "./encoder-gpu.js";
import { encoderKernels } from "./kernels-node.js";

const arg = (name: string): string | undefined => {
  const at = process.argv.indexOf(name);
  return at >= 0 ? process.argv[at + 1] : undefined;
};
const dir = arg("--dir");
if (!dir) {
  console.error("verify-encode-gpu: --dir is required (a directory gen_resnet_golden.py --whole wrote)");
  process.exit(2);
}

const f32 = (path: string): Float32Array => {
  const buffer = readFileSync(path);
  return new Float32Array(buffer.buffer, buffer.byteOffset, buffer.byteLength / 4);
};

const manifest = JSON.parse(readFileSync(`${dir}/encoder.manifest.json`, "utf8")) as EncoderGpuManifest & {
  video: [number, number, number];
  latent: [number, number, number];
};

const fd = openSync(`${dir}/encoder.bin`, "r");
const read = (offsetBytes: number, byteLength: number): Uint8Array => {
  const bytes = Buffer.allocUnsafe(byteLength);
  const got = readSync(fd, bytes, 0, byteLength, offsetBytes);
  if (got !== byteLength) throw new Error(`encoder.bin: read ${got} of ${byteLength}`);
  return new Uint8Array(bytes.buffer, bytes.byteOffset, byteLength);
};

const c = manifest.config;
const { blockMid } = channelPlanOf(c);
console.log(
  `weights ${(statSync(`${dir}/encoder.bin`).size / 1e9).toFixed(2)} GB, levels ${blockMid.join(" ")}`,
);

const device = await createResidentDevice();
if (!device) {
  console.error("verify-encode-gpu: no adapter");
  process.exit(2);
}

const uploadStart = performance.now();
const encoder = await EncoderGpu.create(device, encoderKernels(), manifest, read);
closeSync(fd);
console.log(`uploaded in ${((performance.now() - uploadStart) / 1000).toFixed(1)} s`);

const [T, H, W] = manifest.video;
const started = performance.now();
const got = await encoder.encode(f32(`${dir}/video.bin`), T, H, W);
const took = performance.now() - started;
console.log(
  `video ${T}x${H}x${W} -> moments ${got.C}x${got.D}x${got.H}x${got.W} in ${(took / 1000).toFixed(2)} s — ` +
    `${encoder.dispatches} dispatches, ${encoder.submitMs.toFixed(0)} ms in the queue, ` +
    `${encoder.readbackMs.toFixed(0)} ms reading back, ${encoder.recordMs.toFixed(0)} ms recording`,
);

const [lT, lH, lW] = manifest.latent;
if (got.D !== lT || got.H !== lH || got.W !== lW || got.C !== 2 * c.z_channels) {
  console.error(`shape ${got.C}x${got.D}x${got.H}x${got.W} against ${2 * c.z_channels}x${lT}x${lH}x${lW}`);
  process.exit(1);
}

const want = f32(`${dir}/moments.bin`);
if (got.data.length !== want.length) {
  console.error(`length ${got.data.length} against ${want.length}`);
  process.exit(1);
}
let worst = 0;
let sum = 0;
let peak = 0;
let nonFinite = 0;
for (let i = 0; i < want.length; i += 1) {
  if (!Number.isFinite(got.data[i]!)) nonFinite += 1;
  const d = Math.abs(got.data[i]! - want[i]!);
  sum += d * d;
  if (d > worst) worst = d;
  peak = Math.max(peak, Math.abs(want[i]!));
}
// Counted, never assumed away: `Math.abs(NaN - x) > worst` is false, so a wholly
// non-finite output reports a perfect score. #210 printed exactly that.
if (nonFinite) {
  console.error(`${nonFinite} of ${want.length} values are not finite (first: ${got.data[0]})`);
  process.exit(1);
}
console.log(
  `worst ${worst.toExponential(3)}  rms ${Math.sqrt(sum / want.length).toExponential(3)}  ` +
    `moments peak ${peak.toFixed(4)}  -> ${((worst / peak) * 100).toFixed(3)}% of peak`,
);

// The mean is the **first** half of the channels, which is what a decoder is
// given. Checked rather than assumed: the second half is the log-variance, and
// a decoder handed it would produce something plausible and wrong.
const mean = latentMeanOf(got, c.z_channels);
if (mean.length !== c.z_channels * lT * lH * lW || mean[0] !== got.data[0]) {
  console.error("the mean is not the first half of the channels");
  process.exit(1);
}
console.log(`mean ${c.z_channels}x${lT}x${lH}x${lW}, which is what the decoder takes`);

encoder.destroy();
device.destroy();
