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
import { readFileSync, statSync, openSync, readSync, closeSync } from "node:fs";
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

const manifest = JSON.parse(readFileSync(`${dir}/decoder.manifest.json`, "utf8")) as VideoDecoderManifest;
const golden = JSON.parse(readFileSync(`${dir}/golden.json`, "utf8")) as {
  dims: [number, number, number];
  frames: number;
  height: number;
  width: number;
};

// Read the 9.69 GB one tensor at a time rather than mapping it whole: a single
// `Float32Array` over it would be 2.4 billion elements, past what a typed array
// is guaranteed to hold, and every tensor is uploaded and then dropped anyway.
const bin = `${dir}/decoder.bin`;
console.log(`weights ${(statSync(bin).size / 1e9).toFixed(2)} GB, ${manifest.tensors.length} tensors`);
const fd = openSync(bin, "r");
const read = (offset: number, count: number): Float32Array => {
  const bytes = Buffer.allocUnsafe(count * 4);
  const got = readSync(fd, bytes, 0, count * 4, offset * 4);
  if (got !== count * 4) throw new Error(`decoder.bin: read ${got} bytes where ${count * 4} were wanted`);
  return new Float32Array(bytes.buffer, bytes.byteOffset, count);
};

const device = await createResidentDevice();
if (!device) {
  console.error("verify-decode: no adapter");
  process.exit(2);
}

const uploadStart = performance.now();
const decoder = new VideoDecoderGpu(device, videoKernels(), manifest, read);
closeSync(fd);
console.log(`uploaded in ${((performance.now() - uploadStart) / 1000).toFixed(1)} s`);

const latent = f32(`${dir}/latent.bin`);
const want = f32(`${dir}/pixels.bin`);

const started = performance.now();
const got = await decoder.decode(latent, golden.dims);
const took = performance.now() - started;

console.log(`latent ${golden.dims.join("x")} -> ${golden.frames}x${golden.height}x${golden.width} in ${took.toFixed(0)} ms`);

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
const second = await decoder.decode(latent, golden.dims);
let repeat = 0;
for (let i = 0; i < got.length; i += 1) repeat = Math.max(repeat, Math.abs(second[i]! - got[i]!));
console.log(`same latent twice: worst difference ${repeat.toExponential(3)}`);
if (repeat !== 0) {
  console.error("the second decode disagreed with the first — a pooled buffer is being read before it is written");
  process.exit(1);
}

const shown = denormalise(got, manifest.config.out_channels, manifest.pixelMean, manifest.pixelStd);
let low = Infinity;
let high = -Infinity;
for (const v of shown) {
  low = Math.min(low, v);
  high = Math.max(high, v);
}
console.log(`denormalised range [${low.toFixed(4)}, ${high.toFixed(4)}]`);

decoder.destroy();
device.destroy();
