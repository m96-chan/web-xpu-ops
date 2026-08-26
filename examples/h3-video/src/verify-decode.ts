/**
 * The GPU decoder against the pixels MiniMax-H3's own code produced.
 *
 * Issue #200. A script rather than a test: it holds 9.69 GB of weights on the
 * device, which is not something to do inside a vitest worker beside 1,700
 * other tests (#107 — real-model-scale work in that process crashes the Dawn
 * binding). `examples/anima` verifies its forward the same way.
 *
 *     npx tsx examples/h3-video/src/verify-decode.ts --dir ~/h3-video-web
 */
import { existsSync, readFileSync, statSync, openSync, readSync, closeSync } from "node:fs";
import { createResidentDevice } from "../../../harness/resident.js";
import { VideoDecoderGpu, denormalise, type VideoDecoderManifest } from "./decoder-gpu.js";
import { videoKernels } from "./kernels-node.js";

const arg = (name: string): string | undefined => {
  const at = process.argv.indexOf(name);
  return at >= 0 ? process.argv[at + 1] : undefined;
};
const dir = arg("--dir");
if (!dir) {
  console.error("verify-decode: --dir is required (a directory tools/convert_decoder.py wrote)");
  process.exit(2);
}

const f32 = (path: string): Float32Array => {
  const buffer = readFileSync(path);
  return new Float32Array(buffer.buffer, buffer.byteOffset, buffer.byteLength / 4);
};

const manifest = JSON.parse(
  readFileSync(
    existsSync(`${dir}/decoder.q8.manifest.json`) ? `${dir}/decoder.q8.manifest.json` : `${dir}/decoder.manifest.json`,
    "utf8",
  ),
) as VideoDecoderManifest;

/**
 * `--bench T,H,W` times a random latent instead of comparing to a golden.
 *
 * The arithmetic does not depend on the shape — that is what the comparison
 * establishes — but the *time* does, and a demo should not offer a size nobody
 * has run. Reported as a time, never as a correctness claim: this mode has
 * nothing to compare against and says so.
 */
const bench = arg("--bench");
const golden = bench
  ? (() => {
      const [T, H, W] = bench.split(",").map(Number) as [number, number, number];
      const c = manifest.config;
      return { dims: [T, H, W] as [number, number, number], frames: T * c.patch_size_t, height: H * c.patch_size, width: W * c.patch_size };
    })()
  : (JSON.parse(readFileSync(`${dir}/golden.json`, "utf8")) as {
      dims: [number, number, number];
      frames: number;
      height: number;
      width: number;
    });

// Read the 9.69 GB one tensor at a time rather than mapping it whole: a single
// `Float32Array` over it would be 2.4 billion elements, past what a typed array
// is guaranteed to hold, and every tensor is uploaded and then dropped anyway.
const bin = `${dir}/${manifest.dtype === "q8" ? "decoder.q8.bin" : "decoder.bin"}`;
console.log(`weights ${(statSync(bin).size / 1e9).toFixed(2)} GB, ${manifest.tensors.length} tensors, ${manifest.dtype}`);
const fd = openSync(bin, "r");
// Bytes: a `q8` tensor is packed int8 in u32 words, and nothing here should
// interpret them.
const read = (offsetBytes: number, byteLength: number): Uint8Array => {
  const bytes = Buffer.allocUnsafe(byteLength);
  const got = readSync(fd, bytes, 0, byteLength, offsetBytes);
  if (got !== byteLength) throw new Error(`${bin}: read ${got} bytes where ${byteLength} were wanted`);
  return new Uint8Array(bytes.buffer, bytes.byteOffset, byteLength);
};

const device = await createResidentDevice();
if (!device) {
  console.error("verify-decode: no adapter");
  process.exit(2);
}

const uploadStart = performance.now();
const decoder = await VideoDecoderGpu.create(device, videoKernels(), manifest, read);
decoder.blocksPerSubmit = Number(arg("--blocks-per-submit") ?? 1);
closeSync(fd);
console.log(`uploaded in ${((performance.now() - uploadStart) / 1000).toFixed(1)} s`);

const [T, H, W] = golden.dims;
const latent = bench
  ? Float32Array.from({ length: manifest.config.in_channels * T * H * W }, (_, i) => Math.sin(i * 0.37) * 0.5)
  : f32(`${dir}/latent.bin`);
const want = bench ? null : f32(`${dir}/pixels.bin`);

const started = performance.now();
const got = await decoder.decode(latent, golden.dims);
const took = performance.now() - started;

console.log(`latent ${golden.dims.join("x")} -> ${golden.frames}x${golden.height}x${golden.width} in ${took.toFixed(0)} ms`);
console.log(
  `  ${decoder.dispatches} dispatches: ${decoder.submitMs.toFixed(0)} ms in the queue, ` +
    `${decoder.readbackMs.toFixed(0)} ms in the final submit and readback, ` +
    `${decoder.recordMs.toFixed(0)} ms recording (${((decoder.recordMs / decoder.dispatches) * 1000).toFixed(0)} us each)`,
);

if (!want) {
  console.log(`(--bench: nothing to compare against — this is a time, not a correctness claim)`);
  decoder.destroy();
  device.destroy();
  process.exit(0);
}

if (got.length !== want.length) {
  console.error(`length ${got.length} against ${want.length}`);
  process.exit(1);
}
let worst = 0;
let sum = 0;
let peak = 0;
for (let i = 0; i < want.length; i += 1) {
  const d = Math.abs(got[i]! - want[i]!);
  sum += d * d;
  if (d > worst) worst = d;
  peak = Math.max(peak, Math.abs(want[i]!));
}
console.log(`worst ${worst.toExponential(3)}  rms ${Math.sqrt(sum / want.length).toExponential(3)}  signal peak ${peak.toFixed(4)}`);

// The same latent twice, and the two must agree **exactly**.
//
// Not a formality. Scratch buffers are pooled, so the second decode is the
// first one that sees a *used* buffer — and anything the decoder relies on
// being zero, rather than writing, is right the first time and wrong after.
// The cls token is exactly that: `ViT3DDecoder` builds it as
// `torch.zeros_like(...)`, and deleting the explicit clear leaves a
// single-decode check green.
const secondStart = performance.now();
const second = await decoder.decode(latent, golden.dims);
console.log(
  `second decode: ${(performance.now() - secondStart).toFixed(0)} ms ` +
    `(${decoder.submitMs.toFixed(0)} ms in the queue) — the scratch pool is warm by now`,
);
let repeat = 0;
for (let i = 0; i < got.length; i += 1) repeat = Math.max(repeat, Math.abs(second[i]! - got[i]!));
console.log(`same latent twice: worst difference ${repeat.toExponential(3)}`);
if (repeat !== 0) {
  console.error("the second decode disagreed with the first — a pooled buffer is being read before it is written");
  process.exit(1);
}

// The same difference in the units a viewer sees.
//
// A worst element in the model's normalised space says nothing on its own: the
// denormalisation multiplies by a per-channel std of about 0.22, and the result
// is shown as 8-bit levels. **Quoting the normalised number alone is how a
// quantisation gets called accurate or inaccurate without either being
// established.**
const shownGot = denormalise(got, manifest.config.out_channels, manifest.pixelMean, manifest.pixelStd);
const shownWant = denormalise(want, manifest.config.out_channels, manifest.pixelMean, manifest.pixelStd);
let worstLevels = 0;
let sumLevels = 0;
for (let i = 0; i < shownGot.length; i += 1) {
  const a = Math.max(0, Math.min(255, Math.round(shownGot[i]! * 255)));
  const b = Math.max(0, Math.min(255, Math.round(shownWant[i]! * 255)));
  const d = Math.abs(a - b);
  sumLevels += d * d;
  if (d > worstLevels) worstLevels = d;
}
console.log(
  `as 8-bit pixels: worst ${worstLevels} levels of 255, RMS ${Math.sqrt(sumLevels / shownGot.length).toFixed(3)}`,
);

const shown = shownGot;
let low = Infinity;
let high = -Infinity;
for (const v of shown) {
  low = Math.min(low, v);
  high = Math.max(high, v);
}
console.log(`denormalised range [${low.toFixed(4)}, ${high.toFixed(4)}]`);

decoder.destroy();
device.destroy();
