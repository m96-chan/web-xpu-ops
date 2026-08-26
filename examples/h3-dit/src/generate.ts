/**
 * A prompt embedding in, video frames out — MiniMax-H3 end to end.
 *
 * Issue #210. The sampling loop over `examples/h3-dit`'s DiT, then
 * `examples/h3-video`'s VAE decoder on the latent it produced.
 *
 * **The two models do not fit on the device at once**, and `destroy()` alone is
 * not enough to make them: 20.08 GB of DiT plus 0.58 GB of tables plus 2.43 GB
 * of decoder is 23.1 GB before any scratch, on a 32 GB card a browser may
 * already be using a third of. Destroying the DiT and uploading the decoder in
 * the same process **failed** -- Dawn handed back invalid buffers, which is
 * what this device does when it is out of memory.
 *
 * **`destroy()` schedules the freeing; it does not do it.** Dawn releases on
 * its next tick and it ticks on GPU work, not on a timer, so the fix is to ask
 * it for some: `device.reclaim()` submits four round trips and the memory is
 * back. `harness/reclaim.ts` has the measurement and `harness/verify-reclaim.ts`
 * runs both halves of it. Issue #213.
 *
 * The phases can still be run as separate **processes** -- `--phase sample`
 * writes the latent and exits, `--phase decode` reads it back -- which is what
 * this file did before there was a release it could rely on.
 *
 * **`--phase both` at 256x256 over 22 frames does not exercise that**, and the
 * measurement says so: 20.66 GB of DiT and 2.43 GB of decoder is 23.1 GB, which
 * fits on this card at once. Removing the `reclaim()` below and re-running it
 * changed nothing -- the mutation does not bite here. What it guards is the
 * geometry where the two do not fit and the case the paragraph above hit, and
 * the thing that measures it is `harness/verify-reclaim.ts`, which runs both
 * halves at 25.78 GB and fails without the call.
 *
 * **There is no text encoder here.** Qwen3-VL-32B is 66.7 GB and runs offline,
 * in `tools/encode_prompt.py`; this takes the embedding it wrote. A prompt
 * nobody has encoded cannot be generated, which is the whole shape of the
 * limitation and not a detail.
 *
 * The checkpoint is **guidance-distilled**: one forward per step, no negative
 * prompt, no unconditional branch.
 *
 *     npx tsx examples/h3-dit/src/generate.ts \
 *       --dit ~/h3-dit-gpu --vae ~/h3-video-q8 --prompts ~/h3-prompts \
 *       --prompt 0 --frames 22 --height 256 --width 256 --steps 16 --out ~/h3-out
 */
import { closeSync, mkdirSync, openSync, readFileSync, readSync, writeFileSync } from "node:fs";
import { createResidentDevice } from "../../../harness/resident.js";
import {
  VideoDecoderGpu, denormalise, unnormaliseLatent, type VideoDecoderManifest,
} from "../../h3-video/src/decoder-gpu.js";
import { videoKernels } from "../../h3-video/src/kernels-node.js";
import {
  AUDIO_CHANNELS,
  alignNumFrames,
  audioLatentNumFrames,
  buildPackedSequence,
  buildRowTimesteps,
  patchifyVideoLatents,
  unpatchifyVideoLatents,
  videoLatentNumFrames,
} from "./layout.js";
import { DitGpu, type DitManifest } from "./model-gpu.js";
import { ditKernels } from "./kernels-node.js";
import { setTimesteps, step as schedulerStep } from "./scheduler.js";

const arg = (name: string): string | undefined => {
  const at = process.argv.indexOf(name);
  return at >= 0 ? process.argv[at + 1] : undefined;
};
const number = (name: string, fallback: number): number => Number(arg(name) ?? fallback);

const ditDir = arg("--dit");
const vaeDir = arg("--vae");
const promptsDir = arg("--prompts");
const outDir = arg("--out") ?? "h3-out";
const phase = arg("--phase") ?? "both";
if (!["sample", "decode", "both"].includes(phase)) {
  console.error(`generate: --phase must be sample, decode or both, not ${phase}`);
  process.exit(2);
}
if ((phase !== "decode" && (!ditDir || !promptsDir)) || (phase !== "sample" && !vaeDir)) {
  console.error("generate: --dit and --prompts are required to sample, --vae to decode");
  process.exit(2);
}

const frames = alignNumFrames(number("--frames", 22));
const height = number("--height", 256);
const width = number("--width", 256);
const steps = number("--steps", 16);
const seed = number("--seed", 0);
const promptIndex = number("--prompt", 0);

const f32 = (path: string): Float32Array => {
  const buffer = readFileSync(path);
  return new Float32Array(buffer.buffer, buffer.byteOffset, buffer.byteLength / 4);
};

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

/** xorshift128+ and Box-Muller, as `examples/anima` uses. */
function gaussianNoise(count: number, seedValue: number): Float32Array {
  let s0 = (seedValue ^ 0x9e3779b9) >>> 0 || 1;
  let s1 = (seedValue * 0x85ebca6b + 0xc2b2ae35) >>> 0 || 2;
  const next = (): number => {
    let x = s0;
    const y = s1;
    s0 = y;
    x ^= x << 23;
    x ^= x >>> 17;
    x ^= y ^ (y >>> 26);
    s1 = x >>> 0;
    return ((s0 + s1) >>> 0) / 4294967296;
  };
  const out = new Float32Array(count);
  // **Not torch's Philox**, so a seed reproduces this port's own runs and not
  // MiniMax's for the same number.
  for (let i = 0; i < count; i += 2) {
    const u = Math.max(next(), Number.MIN_VALUE);
    const v = next();
    const r = Math.sqrt(-2 * Math.log(u));
    out[i] = r * Math.cos(2 * Math.PI * v);
    if (i + 1 < count) out[i + 1] = r * Math.sin(2 * Math.PI * v);
  }
  return out;
}

const ditManifest = JSON.parse(readFileSync(`${ditDir!}/dit.manifest.json`, "utf8")) as DitManifest;
const prompts = JSON.parse(readFileSync(`${promptsDir!}/prompts.json`, "utf8")) as {
  layer: number;
  prompts: { prompt: string; file: string; tokens: number; dim: number }[];
};
const chosen = prompts.prompts[promptIndex];
if (!chosen) {
  console.error(`generate: --prompt ${promptIndex} is outside the ${prompts.prompts.length} encoded`);
  process.exit(2);
}
if (chosen.dim !== ditManifest.config.text_dim) {
  console.error(`the embedding is ${chosen.dim}-wide and the DiT reads ${ditManifest.config.text_dim}`);
  process.exit(2);
}
if (!ditManifest.stepCounts.includes(steps)) {
  console.error(
    `this conversion holds modulation tables for ${ditManifest.stepCounts.join(", ")} steps, not ${steps}. ` +
      `Re-run convert_dit.py with --steps ${steps}.`,
  );
  process.exit(2);
}

const c = ditManifest.config;
const vaeStride = 16;
if (height % (vaeStride * c.patch_size[1]) || width % (vaeStride * c.patch_size[2])) {
  console.error(`generate: ${width}x${height} must be a multiple of ${vaeStride * c.patch_size[2]}`);
  process.exit(2);
}
const latentFrames = videoLatentNumFrames(frames);
const latentHeight = height / vaeStride;
const latentWidth = width / vaeStride;
const audioLatents = audioLatentNumFrames(frames);

const layout = buildPackedSequence({
  numTextTokens: chosen.tokens,
  numLatentFrames: latentFrames,
  latentHeight,
  latentWidth,
  numAudioLatents: audioLatents,
  patchSize: c.patch_size,
});
console.log(
  `"${chosen.prompt}"\n  ${frames} frames of ${width}x${height} -> latent ${latentFrames}x${latentHeight}x${latentWidth}, ` +
    `${audioLatents} audio latents, ${layout.seq} packed rows, ${steps} steps`,
);

const device = await createResidentDevice();
if (!device) {
  console.error("generate: no adapter");
  process.exit(2);
}

// Phase 1: sample.
let latent: Float32Array;
if (phase === "decode") {
  latent = f32(`${outDir}/latent.bin`);
  console.log(`  read ${outDir}/latent.bin, ${latent.length} values`);
} else {
  const video = setTimesteps(steps, 12);
  const audio = setTimesteps(steps, 3);
  if (video.timesteps.length !== ditManifest.schedules[String(steps)]!.video.length) {
    console.error("the schedule this port builds is a different length from the one the tables were built for");
    process.exit(1);
  }

  let videoRows = patchifyVideoLatents(
    gaussianNoise(c.in_channels * latentFrames * latentHeight * latentWidth, seed),
    c.in_channels, latentFrames, latentHeight, latentWidth, c.patch_size,
  );
  let audioRows = gaussianNoise(audioLatents * AUDIO_CHANNELS * c.audio_in_channels, seed + 1);
  const text = f32(`${promptsDir!}/${chosen.file}`).slice(0, chosen.tokens * c.text_dim);

  const weights = openReader(`${ditDir!}/${ditManifest.dtype === "q8" ? "dit.q8.bin" : "dit.bin"}`);
  const tables = openReader(`${ditDir!}/adaln.bin`);
  let at = performance.now();
  const dit = await DitGpu.create(device, ditKernels(), ditManifest, weights.read, tables.read);
  weights.close();
  tables.close();
  console.log(`  DiT uploaded in ${((performance.now() - at) / 1000).toFixed(1)} s`);

  at = performance.now();
  for (let i = 0; i < video.timesteps.length; i += 1) {
    const { timestepIndices } = buildRowTimesteps(layout, video.timesteps[i]!, audio.timesteps[i]!);
    layout.timestepIndices = timestepIndices;
    const stepStart = performance.now();
    const velocity = await dit.forward({ video: videoRows, audio: audioRows, text, layout, steps, stepIndex: i });
    videoRows = schedulerStep(video, velocity.video, video.timesteps[i]!, videoRows, i);
    audioRows = schedulerStep(audio, velocity.audio, audio.timesteps[i]!, audioRows, i);
    console.log(
      `  step ${i + 1}/${video.timesteps.length}  t=${video.timesteps[i]!.toFixed(4)}  ` +
        `${(performance.now() - stepStart).toFixed(0)} ms`,
    );
  }
  const sampleMs = performance.now() - at;
  dit.destroy();
  // **Not enough on its own** -- see this file's own doc. Unmeasurable at this
  // geometry, where both models fit at once; load-bearing where they do not.
  await device.reclaim();
  console.log(
    `  sampled in ${(sampleMs / 1000).toFixed(1)} s ` +
      `(${(sampleMs / video.timesteps.length).toFixed(0)} ms a step)`,
  );

  latent = unpatchifyVideoLatents(
    videoRows, c.in_channels, latentFrames, latentHeight, latentWidth, c.patch_size,
  );
  mkdirSync(outDir, { recursive: true });
  writeFileSync(`${outDir}/latent.bin`, Buffer.from(latent.buffer, latent.byteOffset, latent.byteLength));
  writeFileSync(`${outDir}/audio-latent.bin`, Buffer.from(audioRows.buffer, audioRows.byteOffset, audioRows.byteLength));
  if (phase === "sample") {
    console.log(`  wrote ${outDir}/latent.bin — run again with --phase decode`);
    device.destroy();
    process.exit(0);
  }
}

// Phase 2: decode. The DiT is gone and reclaimed by now — see this file's own doc.
const vaeManifest = JSON.parse(readFileSync(`${vaeDir!}/decoder.q8.manifest.json`, "utf8")) as VideoDecoderManifest;
const vaeWeights = openReader(`${vaeDir!}/decoder.q8.bin`);
let decodeAt = performance.now();
const decoder = await VideoDecoderGpu.create(device, videoKernels(), vaeManifest, vaeWeights.read);
vaeWeights.close();
console.log(`  VAE uploaded in ${((performance.now() - decodeAt) / 1000).toFixed(1)} s`);

decodeAt = performance.now();
// **The DiT works in the normalised latent space and this decoder does not.**
// Handing it the sampler's output verbatim produced a blurred picture with a
// grid over it, and that was read as what int8 costs until `ref2va` needed the
// same transform in the other direction. Issue #212.
const pixels = await decoder.decode(
  unnormaliseLatent(latent, vaeManifest), [latentFrames, latentHeight, latentWidth]);
console.log(`  decoded in ${((performance.now() - decodeAt) / 1000).toFixed(1)} s`);

const shown = denormalise(pixels, vaeManifest.config.out_channels, vaeManifest.pixelMean, vaeManifest.pixelStd);
const outFrames = latentFrames * vaeManifest.config.patch_size_t;
const outHeight = latentHeight * vaeManifest.config.patch_size;
const outWidth = latentWidth * vaeManifest.config.patch_size;
const bytes = new Uint8Array(shown.length);
let low = Infinity;
let high = -Infinity;
for (let i = 0; i < shown.length; i += 1) {
  bytes[i] = Math.max(0, Math.min(255, Math.round(shown[i]! * 255)));
  low = Math.min(low, shown[i]!);
  high = Math.max(high, shown[i]!);
}
writeFileSync(`${outDir}/frames.rgb`, Buffer.from(bytes));
writeFileSync(`${outDir}/frames.json`, `${JSON.stringify({
  prompt: chosen.prompt,
  frames: outFrames,
  height: outHeight,
  width: outWidth,
  channels: vaeManifest.config.out_channels,
  layout: "[frame][channel][row][col] u8",
  steps,
  seed,
  range: [low, high],
}, null, 1)}\n`);
console.log(
  `  ${outFrames} frames of ${outWidth}x${outHeight} -> ${outDir}/frames.rgb, denormalised range ` +
    `[${low.toFixed(4)}, ${high.toFixed(4)}]`,
);

decoder.destroy();
device.destroy();
