/**
 * The whole encoder against `EncoderFCN3D` and `quant_conv`.
 *
 * Issue #200. A script rather than a test: the reference walks six levels of
 * twelve `ResnetBlock3D`s on the CPU and takes minutes, well past
 * `scripts/test.mjs`'s per-file limit. `examples/anima` verifies its forward the
 * same way, and so does `examples/h3-video/src/verify-decode.ts`.
 *
 * `encoder-block.test.ts` holds the *arithmetic* to the model; this holds the
 * **assembly** — six levels, the strided downsamples between them, and the
 * projection at the end. The two fail differently: a wrong block moves every
 * value, a wrong assembly usually changes a shape or shifts the picture.
 *
 *     npx tsx examples/h3-encoder/src/verify-encode.ts --dir ~/h3-encoder-whole
 */
import { closeSync, openSync, readFileSync, readSync } from "node:fs";
import { channelPlan, h3Encode, latentMean, type EncoderConfig } from "./encoder.js";

const arg = (name: string): string | undefined => {
  const at = process.argv.indexOf(name);
  return at >= 0 ? process.argv[at + 1] : undefined;
};
const dir = arg("--dir");
if (!dir) {
  console.error("verify-encode: --dir is required (a directory gen_resnet_golden.py --whole wrote)");
  process.exit(2);
}

const f32 = (path: string): Float32Array => {
  const buffer = readFileSync(path);
  return new Float32Array(buffer.buffer, buffer.byteOffset, buffer.byteLength / 4);
};

const manifest = JSON.parse(readFileSync(`${dir}/encoder.manifest.json`, "utf8")) as {
  config: {
    ch: number; ch_mult: number[]; num_res_blocks: number;
    space_down: number[]; time_down: number[]; in_channels: number;
    z_channels: number; groups: number; norm_eps: number;
  };
  video: [number, number, number];
  latent: [number, number, number];
  tensors: { name: string; offset: number; count: number }[];
  elements: number;
};

// One tensor at a time: `encoder.bin` is 721 MB, which `readFileSync` would hold
// entirely for no reason.
const fd = openSync(`${dir}/encoder.bin`, "r");
const w = (name: string): Float32Array => {
  const entry = manifest.tensors.find((t) => t.name === name);
  if (!entry) throw new Error(`encoder.bin has no tensor named "${name}"`);
  const bytes = Buffer.allocUnsafe(entry.count * 4);
  const got = readSync(fd, bytes, 0, entry.count * 4, entry.offset * 4);
  if (got !== entry.count * 4) throw new Error(`${name}: read ${got} bytes where ${entry.count * 4} were wanted`);
  return new Float32Array(bytes.buffer, bytes.byteOffset, entry.count);
};

const c = manifest.config;
const cfg: EncoderConfig = {
  ch: c.ch, chMult: c.ch_mult, numResBlocks: c.num_res_blocks,
  spaceDown: c.space_down, timeDown: c.time_down,
  inChannels: c.in_channels, zChannels: c.z_channels,
  groups: c.groups, normEps: c.norm_eps,
};
const { blockIn, blockMid } = channelPlan(cfg);
console.log(`levels ${blockMid.join(" ")}  (first block of each takes ${blockIn.join(" ")})`);

const [T, H, W] = manifest.video;
const started = performance.now();
const got = h3Encode(cfg, w, f32(`${dir}/video.bin`), T, H, W);
closeSync(fd);
console.log(`video ${T}x${H}x${W} -> latent ${got.C}x${got.D}x${got.H}x${got.W} in ${((performance.now() - started) / 1000).toFixed(1)} s`);

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
for (let i = 0; i < want.length; i += 1) {
  const d = Math.abs(got.data[i]! - want[i]!);
  sum += d * d;
  if (d > worst) worst = d;
  peak = Math.max(peak, Math.abs(want[i]!));
}
console.log(`worst ${worst.toExponential(3)}  rms ${Math.sqrt(sum / want.length).toExponential(3)}  moments peak ${peak.toFixed(4)}`);

// The mean is the **first** half of the channels, which is what a decoder is
// given. Checked rather than assumed: the second half is the log-variance, and
// a decoder handed it would produce something plausible and wrong.
const mean = latentMean(got, c.z_channels);
if (mean.length !== c.z_channels * lT * lH * lW || mean[0] !== got.data[0]) {
  console.error("the mean is not the first half of the channels");
  process.exit(1);
}
console.log(`mean ${c.z_channels}x${lT}x${lH}x${lW}, which is what the decoder takes`);
