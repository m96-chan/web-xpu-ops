/**
 * How many latent frames the GPU encoder returns, and how, for a real reference.
 *
 * Issue #216. `examples/h3-encoder` is held to `EncoderFCN3D` + `quant_conv`,
 * which is `AutoencoderKLLegacy.encode`. But `encode_base` does not call
 * `encode` for a multi-frame input — it calls `encode_temporal`, which pads the
 * clip up to a multiple of `clip_length`, encodes each 17-frame chunk
 * **independently**, and drops `token_drop` latent frames off the end. The
 * released checkpoint ships `vae_clip_length 17`, `vae_token_drop 3`.
 *
 * `tools/measure_temporal_chunking.py` measures the two torch paths against
 * each other: at 48 frames — a 2-second reference, the shortest the model card
 * allows — they return the *same shape* and differ by 17.9% rms. This script
 * asks the remaining question: which of the two does the GPU encoder do?
 *
 * It answers with shapes rather than values, because the shapes already
 * separate the hypotheses at 51 frames (13 latent frames the plain way, 12 the
 * chunked way) and needs no golden to do it.
 *
 *     npx tsx examples/h3-encoder/src/measure-chunking-gpu.ts --dir ~/h3-work/h3-encoder-whole
 */
import { readFileSync } from "node:fs";
import { createResidentDevice } from "../../../harness/resident.js";
import { EncoderGpu, type EncoderGpuManifest } from "./encoder-gpu.js";
import { encoderKernels } from "./kernels-node.js";

const arg = (name: string): string | undefined => {
  const at = process.argv.indexOf(name);
  return at >= 0 ? process.argv[at + 1] : undefined;
};
const dir = arg("--dir");
if (!dir) {
  console.error("measure-chunking-gpu: --dir is required");
  process.exit(2);
}

const manifest = JSON.parse(readFileSync(`${dir}/encoder.manifest.json`, "utf8")) as EncoderGpuManifest;
const weights = readFileSync(`${dir}/encoder.bin`);

const device = await createResidentDevice();
if (!device) {
  console.error("no GPU adapter");
  process.exit(2);
}

const encoder = await EncoderGpu.create(device, encoderKernels(), manifest, (at: number, length: number) =>
  new Uint8Array(weights.buffer, weights.byteOffset + at, length));

/** `clip_length` and `token_drop` from the released `video_vae` config. */
const CLIP = 17;
const DROP = 3;
/** `5 * ceil(T / 17) - 3`: what `encode_temporal` returns for `T` pixel frames. */
const chunked = (frames: number): number =>
  Math.ceil(frames / CLIP) * Math.ceil(CLIP / 4) - DROP;

/**
 * Held to `encode_temporal`'s own output, when a golden is there.
 *
 * Shapes alone cannot close this: at 48 and 68 frames the two paths return the
 * same shape and differ by 17.9% and 19.0% rms, which is the whole reason
 * nothing downstream ever complained. So the numbers get compared too.
 */
const goldenDir = arg("--golden");
if (goldenDir) {
  const meta = JSON.parse(readFileSync(`${goldenDir}/conditioning.json`, "utf8")) as {
    video: [number, number, number]; latent: number[];
  };
  const f32 = (path: string): Float32Array => {
    const buffer = readFileSync(path);
    return new Float32Array(buffer.buffer, buffer.byteOffset, buffer.byteLength / 4);
  };
  const [T, H, W] = meta.video;
  const got = await encoder.encodeConditioning(f32(`${goldenDir}/video.bin`), T, H, W);
  const want = f32(`${goldenDir}/conditioning.bin`);
  const shape = `${got.C}x${got.D}x${got.H}x${got.W}`;
  console.log(`held to encode_temporal: ${T}x${H}x${W} -> ${shape}, want ${meta.latent.join("x")}`);
  if (got.data.length !== want.length) {
    console.error(`  ${got.data.length} values against ${want.length}`);
    process.exit(1);
  }
  let worst = 0;
  let sum = 0;
  let peak = 0;
  for (let i = 0; i < want.length; i += 1) {
    worst = Math.max(worst, Math.abs(got.data[i]! - want[i]!));
    sum += (got.data[i]! - want[i]!) ** 2;
    peak = Math.max(peak, Math.abs(want[i]!));
  }
  console.log(
    `  worst ${worst.toExponential(3)}  rms ${Math.sqrt(sum / want.length).toExponential(3)}  ` +
      `peak ${peak.toFixed(4)}  -> ${((worst / peak) * 100).toFixed(4)}% of peak`,
  );
}

const SIZE = 32;
console.log(`${"frames".padStart(7)}  ${"got".padStart(5)}  ${"chunked".padStart(8)}  agrees with`);
for (const frames of [8, 17, 22, 48, 51, 68]) {
  const pixels = new Float32Array(3 * frames * SIZE * SIZE);
  for (let i = 0; i < pixels.length; i += 1) pixels[i] = Math.sin(i * 0.017);
  const moments = await encoder.encodeConditioning(pixels, frames, SIZE, SIZE);
  const want = chunked(frames);
  console.log(
    `${String(frames).padStart(7)}  ${String(moments.D).padStart(5)}  ${String(want).padStart(8)}  ` +
      (moments.D === want ? "encode_temporal" : "AutoencoderKLLegacy.encode (no chunking)"),
  );
}

encoder.destroy();
device.destroy();
