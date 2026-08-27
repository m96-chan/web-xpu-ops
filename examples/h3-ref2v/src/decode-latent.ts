/**
 * Decode a latent that came out of a golden, so upstream's own output can be
 * looked at and measured.
 *
 * Issue #216. The flicker ladder — mean absolute difference between consecutive
 * frames — has only ever been measured on *this port's* output, so it could
 * never say whether the model flickers or the port makes it flicker. The two
 * are opposite conclusions and the ladder cannot tell them apart on its own.
 *
 * `gen_real_ref2va_sample_golden.py` writes upstream's own `output.video.bin`.
 * This runs it through the same decode `generate-r2v.ts` runs, so the number
 * measured on it is comparable, and writes `frames.rgb`/`frames.json` in the
 * layout `tools/measure_flicker.py` reads.
 *
 * The decoder is this port's — the one held to the model at 4.530e-6 — because
 * a golden decoder for a video does not exist here and the question is about
 * the latent, not the VAE.
 *
 *     npx tsx examples/h3-ref2v/src/decode-latent.ts \
 *       --latent ~/h3-work/h3-real-sample/output.video.bin \
 *       --golden ~/h3-work/h3-real-sample \
 *       --vae ~/h3-work/h3-video-q8 --out ~/h3-work/h3-decoded-upstream
 */
import { closeSync, mkdirSync, openSync, readFileSync, readSync, writeFileSync } from "node:fs";
import { createResidentDevice } from "../../../harness/resident.js";
import { unpatchifyVideoLatents } from "../../h3-dit/src/layout.js";
import {
  VideoDecoderGpu, denormalise, unnormaliseLatent, type VideoDecoderManifest,
} from "../../h3-video/src/decoder-gpu.js";
import { videoKernels } from "../../h3-video/src/kernels-node.js";

const arg = (name: string): string | undefined => {
  const at = process.argv.indexOf(name);
  return at >= 0 ? process.argv[at + 1] : undefined;
};
const latentPath = arg("--latent");
const goldenDir = arg("--golden");
const vaeDir = arg("--vae");
// `decoder.q8.manifest.json` for a quantised conversion, `decoder.manifest.json`
// for a float one -- the same pair `generate-r2v.ts` picks between.
const vaeStem = arg("--vae-stem") ?? "decoder.q8";
const outDir = arg("--out");
if (!latentPath || !goldenDir || !vaeDir || !outDir) {
  console.error("decode-latent: --latent, --golden, --vae and --out are required");
  process.exit(2);
}

const f32 = (path: string): Float32Array => {
  const buffer = readFileSync(path);
  return new Float32Array(buffer.buffer, buffer.byteOffset, buffer.byteLength / 4);
};

const golden = JSON.parse(readFileSync(`${goldenDir}/golden.json`, "utf8")) as {
  layout: {
    numLatentFrames: number; latentHeight: number; latentWidth: number;
    numConditionVideoRows: number;
  };
};
const vaeManifest = JSON.parse(
  readFileSync(`${vaeDir}/${vaeStem}.manifest.json`, "utf8"),
) as VideoDecoderManifest;

const rows = f32(latentPath);
const { numLatentFrames, latentHeight, latentWidth, numConditionVideoRows } = golden.layout;
// The latent's channel count is the decoder's *input*, which the manifest
// calls `in_channels` — not `out_channels`, which is the picture's three.
const z = vaeManifest.config.in_channels;
// **The anchor rows come off first.** They lead the packed sequence and are
// not part of the generated video; decoding them alongside it would prepend
// the reference to the output and quietly change every frame-to-frame
// difference measured after it.
const perRow = z * 1 * 2 * 2;
const generatedRows = rows.subarray(numConditionVideoRows * perRow);

const latent = unpatchifyVideoLatents(
  generatedRows, z, numLatentFrames, latentHeight, latentWidth, [1, 2, 2],
);
const rms = (x: ArrayLike<number>): number => {
  let s = 0;
  for (let i = 0; i < x.length; i += 1) s += (x[i] as number) ** 2;
  return Math.sqrt(s / x.length);
};
console.log(
  `${generatedRows.length / perRow} generated rows -> latent ${z}x${numLatentFrames}x${latentHeight}x${latentWidth}, ` +
    `rms ${rms(latent).toFixed(4)}`,
);

const device = await createResidentDevice();
if (!device) {
  console.error("decode-latent: no adapter");
  process.exit(2);
}
const fd = openSync(`${vaeDir}/${vaeStem}.bin`, "r");
const read = (offsetBytes: number, byteLength: number): Uint8Array => {
  const bytes = Buffer.allocUnsafe(byteLength);
  const got = readSync(fd, bytes, 0, byteLength, offsetBytes);
  if (got !== byteLength) throw new Error(`${vaeStem}.bin: read ${got} of ${byteLength}`);
  return new Uint8Array(bytes.buffer, bytes.byteOffset, byteLength);
};
const decoder = await VideoDecoderGpu.create(device, videoKernels(), vaeManifest, read);
closeSync(fd);

// **`unnormaliseLatent` before the decode.** The DiT's latent space is not the
// decoder's: leaving it out gives a blurred, grid-textured picture that looks
// like a quantisation problem and is not one.
const raw = await decoder.decode(
  unnormaliseLatent(latent, vaeManifest), [numLatentFrames, latentHeight, latentWidth],
);
const shown = denormalise(raw, vaeManifest.config.out_channels, vaeManifest.pixelMean, vaeManifest.pixelStd);

const height = latentHeight * 16;
const width = latentWidth * 16;
const frames = shown.length / (3 * height * width);
const bytes = new Uint8Array(shown.length);
let low = Infinity;
let high = -Infinity;
for (let i = 0; i < shown.length; i += 1) {
  low = Math.min(low, shown[i]!);
  high = Math.max(high, shown[i]!);
  bytes[i] = Math.max(0, Math.min(255, Math.round(shown[i]! * 255)));
}

mkdirSync(outDir, { recursive: true });
writeFileSync(`${outDir}/frames.rgb`, Buffer.from(bytes));
writeFileSync(`${outDir}/frames.json`, `${JSON.stringify({
  source: latentPath,
  golden: goldenDir,
  frames, height, width, channels: 3,
  layout: "[frame][channel][row][col] u8",
  range: [low, high],
}, null, 1)}\n`);
console.log(
  `${frames} frames of ${width}x${height} -> ${outDir}/frames.rgb  ` +
    `denormalised range [${low.toFixed(4)}, ${high.toFixed(4)}]`,
);

decoder.destroy();
device.destroy();
