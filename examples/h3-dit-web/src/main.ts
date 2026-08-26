/**
 * MiniMax-H3 generating video in a browser: a prompt in, frames on a canvas.
 *
 * Issue #210. Nothing here is new arithmetic — `examples/h3-dit/src/model-gpu.ts`
 * is the DiT and `examples/h3-video/src/decoder-gpu.ts` is the decoder, both
 * checked against the model's own output — and this is the wiring plus the two
 * things a page needs that a script does not: weights from a folder the visitor
 * picked, and a device that survives between generations.
 *
 * **About 23 GB stays on the GPU**: 20.08 GB of int8 DiT, 0.58 GB of
 * precomputed modulation tables and 2.43 GB of decoder. The Node script keeps
 * them in *separate processes* because a machine sharing its card with a
 * browser runs out; a page has no such option, so it holds both and says so.
 *
 * **The prompt list is fixed.** MiniMax-H3 conditions on Qwen3-VL-32B, 66.7 GB,
 * which is not going into a tab — `tools/encode_prompt.py` runs it offline and
 * this reads the embeddings. That is a limitation of the demo, not of the port,
 * and the page says it rather than hiding it behind a disabled text field.
 */
import { DEFAULT_WEIGHTS_BASE, HttpByteSource, type ByteSource } from "../../web-common/src/byte-source.js";
import {
  requireBoundFolder, wireChangeFolder, type GateElements, type GateOptions,
} from "../../web-common/src/gate.js";
import { createBrowserResidentDevice } from "../../web-common/src/browser-resident.js";
import {
  VideoDecoderGpu, denormalise, unnormaliseLatent, type VideoDecoderManifest,
} from "../../h3-video/src/decoder-gpu.js";
import { DitGpu, type DitManifest } from "../../h3-dit/src/model-gpu.js";
import {
  AUDIO_CHANNELS,
  alignNumFrames,
  audioLatentNumFrames,
  buildPackedSequence,
  buildRowTimesteps,
  patchifyVideoLatents,
  unpatchifyVideoLatents,
  videoLatentNumFrames,
} from "../../h3-dit/src/layout.js";
import { setTimesteps, step as schedulerStep } from "../../h3-dit/src/scheduler.js";
import { ditKernels, videoKernels } from "./kernels-web.js";

declare const BUILD_STAMP: string;

const WEIGHTS_BASE = new URLSearchParams(location.search).get("weights") ?? DEFAULT_WEIGHTS_BASE;
/**
 * Everything the page reads. Both models and the prompt embeddings, because a
 * generation needs all three and a folder missing one is better refused at the
 * gate than forty seconds into an upload.
 */
const WEIGHT_FILES = [
  "dit.manifest.json", "dit.q8.bin", "adaln.bin",
  "decoder.q8.manifest.json", "decoder.q8.bin",
  "prompts.json",
];

const $ = <T extends HTMLElement>(id: string): T => {
  const element = document.getElementById(id);
  if (!element) throw new Error(`no element #${id}`);
  return element as T;
};

const status = $<HTMLParagraphElement>("status");
const detail = $<HTMLParagraphElement>("detail");
const goButton = $<HTMLButtonElement>("go");
const promptSelect = $<HTMLSelectElement>("prompt");
const sizeSelect = $<HTMLSelectElement>("size");
const stepsSelect = $<HTMLSelectElement>("steps");
const seedInput = $<HTMLInputElement>("seed");
const canvas = $<HTMLCanvasElement>("frames");
const timing = $<HTMLDivElement>("timing");
const progressBar = $<HTMLDivElement>("progress").firstElementChild as HTMLDivElement;
const playButton = $<HTMLButtonElement>("play");
const slider = $<HTMLInputElement>("frame-slider");
const frameLabel = $<HTMLSpanElement>("frame-label");

function say(message: string, extra = "", fraction: number | null = null): void {
  status.textContent = message;
  detail.textContent = extra;
  progressBar.style.width = fraction === null ? "0" : `${(fraction * 100).toFixed(1)}%`;
}

$<HTMLElement>("build").textContent = BUILD_STAMP;

const gate: GateOptions = {
  elements: {
    dialog: $<HTMLDialogElement>("gate"),
    title: $<HTMLHeadingElement>("gate-title"),
    body: $<HTMLParagraphElement>("gate-body"),
    action: $<HTMLButtonElement>("gate-action"),
    dismiss: $<HTMLButtonElement>("gate-dismiss"),
    progress: $<HTMLParagraphElement>("gate-progress"),
    bar: $<HTMLDivElement>("gate-bar"),
    barFill: $<HTMLDivElement>("gate-bar").firstElementChild as HTMLDivElement,
    why: $<HTMLParagraphElement>("gate-why"),
  } satisfies GateElements,
  files: WEIGHT_FILES,
  weightsBase: WEIGHTS_BASE,
  downloadSize: "23.1 GB",
  licence:
    "Powered by MiniMax H3. The model is licensed under the MiniMax H3 Community License Agreement, " +
    "not this page's MIT, and nothing here redistributes it.",
};

/** xorshift128+ and Box-Muller, the generator the other demos seed with. */
function gaussian(count: number, seedValue: number): Float32Array {
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
  // Not torch's Philox: a seed reproduces **this page's** runs, not MiniMax's.
  for (let i = 0; i < count; i += 2) {
    const u = Math.max(next(), Number.MIN_VALUE);
    const v = next();
    const r = Math.sqrt(-2 * Math.log(u));
    out[i] = r * Math.cos(2 * Math.PI * v);
    if (i + 1 < count) out[i + 1] = r * Math.sin(2 * Math.PI * v);
  }
  return out;
}

interface Clip {
  pixels: Float32Array;
  frames: number;
  height: number;
  width: number;
}

let clip: Clip | null = null;
let playing = false;

function draw(index: number): void {
  if (!clip) return;
  const { pixels, frames, height, width } = clip;
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext("2d");
  if (!context) throw new Error("no 2d context");
  const image = context.createImageData(width, height);
  const perFrame = height * width;
  const perChannel = frames * perFrame;
  for (let i = 0; i < perFrame; i += 1) {
    for (let c = 0; c < 3; c += 1) {
      const v = pixels[c * perChannel + index * perFrame + i]!;
      image.data[i * 4 + c] = Math.max(0, Math.min(255, Math.round(v * 255)));
    }
    image.data[i * 4 + 3] = 255;
  }
  context.putImageData(image, 0, 0);
  slider.value = String(index);
  frameLabel.textContent = `${index + 1} / ${frames}`;
}

function loop(): void {
  if (!playing || !clip) return;
  const next = (Number(slider.value) + 1) % clip.frames;
  draw(next);
  // 24 fps, the rate the model generates at.
  setTimeout(() => requestAnimationFrame(loop), 1000 / 24);
}

slider.oninput = () => draw(Number(slider.value));
playButton.onclick = () => {
  playing = !playing;
  playButton.textContent = playing ? "Pause" : "Play";
  if (playing) loop();
};

/**
  * `?serve=<base>` reads the weights straight off an HTTP host instead of a folder.
  *
  * Not a convenience: the folder gate needs `showDirectoryPicker`, which needs
  * a *user gesture* and a native dialog, so a headless browser cannot reach the
  * page's actual work — and "the in-browser run is unmeasured" is how the
  * neighbouring demo's README still has to end. With `server.mjs` answering
  * ranges, this path is the same reads over `fetch`, and it can be driven.
  *
  * It is **not** the arrangement for a real visitor: nothing is cached, so
  * every reload re-reads 23 GB. The gate is still the default.
  */
async function main(): Promise<void> {
  say("checking this browser …");
  const serveBase = new URLSearchParams(location.search).get("serve");
  let source: ByteSource;
  if (serveBase) {
    source = new HttpByteSource(serveBase);
    say("reading over HTTP …", `${serveBase} — no folder, nothing cached`);
  } else {
    const bound = await requireBoundFolder(gate);
    if (!bound) return;
    source = bound;
    wireChangeFolder(gate, $<HTMLButtonElement>("bind"), source);
  }

  $<HTMLDivElement>("folder-row").hidden = false;
  $<HTMLSpanElement>("folder-state").textContent = source.describe;

  const readJson = async <T>(name: string): Promise<T> =>
    JSON.parse(new TextDecoder().decode(await source.read(name, 0, await source.size(name)))) as T;

  say("reading the manifests …");
  const ditManifest = await readJson<DitManifest>("dit.manifest.json");
  const vaeManifest = await readJson<VideoDecoderManifest>("decoder.q8.manifest.json");
  const prompts = await readJson<{ prompts: { prompt: string; file: string; tokens: number; dim: number }[] }>(
    "prompts.json",
  );

  for (const [index, entry] of prompts.prompts.entries()) {
    const option = document.createElement("option");
    option.value = String(index);
    option.textContent = entry.prompt;
    promptSelect.append(option);
  }
  // Only the step counts the conversion holds tables for. Offering others would
  // be offering a button that throws — `adaln_proj` is not resident, so a step
  // count with no table has no modulation at all.
  for (const count of ditManifest.stepCounts) {
    const option = document.createElement("option");
    option.value = String(count);
    option.textContent = `${count} steps`;
    stepsSelect.append(option);
  }

  say("starting the device …");
  const device = await createBrowserResidentDevice();

  // **Printed before anything is uploaded, not after something fails.** The
  // first run of this page spent 21 s uploading 23 GB and then reported
  // `Instance dropped`; the cause was a limit that could have been read in the
  // first second. `ops/matmul` declares `@compute @workgroup_size(512)`, so a
  // device whose `maxComputeWorkgroupSizeX` is the spec's 256 cannot build the
  // pipeline every projection here dispatches. Issue #211.
  const adapter = await (navigator as Navigator & { gpu: GPU }).gpu.requestAdapter();
  const limits = adapter?.limits;
  const workgroupX = limits?.maxComputeWorkgroupSizeX ?? 0;
  const limitLine =
    `workgroupX ${workgroupX}, invocations ${limits?.maxComputeInvocationsPerWorkgroup}, ` +
    `workgroup storage ${limits?.maxComputeWorkgroupStorageSize}, ` +
    `buffer ${((limits?.maxBufferSize ?? 0) / 1e9).toFixed(2)} GB, ` +
    `binding ${((limits?.maxStorageBufferBindingSize ?? 0) / 1e9).toFixed(2)} GB`;
  $<HTMLDivElement>("limits").textContent = limitLine;
  if (workgroupX < 512) {
    say(
      "this browser cannot run the matmul this model needs.",
      `maxComputeWorkgroupSizeX is ${workgroupX} and ops/matmul declares 512 — ${limitLine}`,
    );
    // Stopped here rather than uploading 23 GB first. See issue #211.
    return;
  }

  const bytes = async (file: string) =>
    async (offsetBytes: number, byteLength: number): Promise<Uint8Array> =>
      new Uint8Array(await source.read(file, offsetBytes, byteLength));

  const started = performance.now();
  let painted = 0;
  const report = (what: string) => async (done: number, total: number) => {
    // Yielded every 200 MB so the progress bar paints; a yield per tensor costs
    // more than the reads do.
    if (done - painted < 200e6 && done < total) return;
    painted = done;
    say(`uploading the ${what} …`, `${(done / 1e9).toFixed(2)} of ${(total / 1e9).toFixed(2)} GB`, done / total);
    await new Promise((resolve) => setTimeout(resolve, 0));
  };

  const dit = await DitGpu.create(
    device, ditKernels, ditManifest,
    await bytes(ditManifest.dtype === "q8" ? "dit.q8.bin" : "dit.bin"),
    await bytes("adaln.bin"),
    report("DiT"),
  );
  painted = 0;
  const decoder = await VideoDecoderGpu.create(
    device, videoKernels, vaeManifest, await bytes("decoder.q8.bin"), report("decoder"),
  );
  const uploadSeconds = (performance.now() - started) / 1000;

  goButton.disabled = false;
  // **Bytes uploaded, not device memory measured.** WebGPU exposes no way to
  // ask how much the device actually took, and on this machine `nvidia-smi`
  // never moved while this line said 23.10 GB -- so calling it "resident" was
  // a claim the page cannot support (rule 9).
  const uploadedGb = (ditManifest.residentBytes + ditManifest.tables.reduce((s, t) => s + t.count * 4, 0)
    + vaeManifest.elements * 4) / 1e9;
  say("ready.", `${uploadedGb.toFixed(2)} GB uploaded in ${uploadSeconds.toFixed(1)} s`);

  goButton.onclick = async () => {
    goButton.disabled = true;
    playing = false;
    playButton.textContent = "Play";
    try {
      const c = ditManifest.config;
      const chosen = prompts.prompts[Number(promptSelect.value)]!;
      const [requestedFrames, height, width] = sizeSelect.value.split(",").map(Number) as [number, number, number];
      const steps = Number(stepsSelect.value);
      const seed = Number(seedInput.value) | 0;

      const frames = alignNumFrames(requestedFrames);
      const vaeStride = vaeManifest.config.patch_size;
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

      const text = new Float32Array(
        await source.read(chosen.file, 0, chosen.tokens * c.text_dim * 4),
      );
      const video = setTimesteps(steps, 12);
      const audio = setTimesteps(steps, 3);
      let videoRows = patchifyVideoLatents(
        gaussian(c.in_channels * latentFrames * latentHeight * latentWidth, seed),
        c.in_channels, latentFrames, latentHeight, latentWidth, c.patch_size,
      );
      let audioRows = gaussian(audioLatents * AUDIO_CHANNELS * c.audio_in_channels, seed + 1);

      const sampleStart = performance.now();
      for (let i = 0; i < video.timesteps.length; i += 1) {
        layout.timestepIndices = buildRowTimesteps(layout, video.timesteps[i]!, audio.timesteps[i]!).timestepIndices;
        const velocity = await dit.forward({ video: videoRows, audio: audioRows, text, layout, steps, stepIndex: i });
        videoRows = schedulerStep(video, velocity.video, video.timesteps[i]!, videoRows, i);
        audioRows = schedulerStep(audio, velocity.audio, audio.timesteps[i]!, audioRows, i);
        say(
          "sampling …",
          `step ${i + 1} of ${video.timesteps.length}, ${layout.seq} packed rows`,
          (i + 1) / video.timesteps.length,
        );
        // Yielded so the progress line paints between steps; without it the tab
        // is frozen for the whole run and the count arrives all at once.
        await new Promise((resolve) => setTimeout(resolve, 0));
      }
      const sampleMs = performance.now() - sampleStart;

      say("decoding …", `${latentFrames * latentHeight * latentWidth} latent tokens`);
      const decodeStart = performance.now();
      const latent = unpatchifyVideoLatents(
        videoRows, c.in_channels, latentFrames, latentHeight, latentWidth, c.patch_size,
      );
      const raw = await decoder.decode(
      // **The DiT's latent space is not the decoder's** — issue #212. Without
      // this the page drew a blurred frame with a grid over it and called it
      // int8.
      unnormaliseLatent(latent, vaeManifest), [latentFrames, latentHeight, latentWidth]);
      const decodeMs = performance.now() - decodeStart;

      const pixels = denormalise(raw, vaeManifest.config.out_channels, vaeManifest.pixelMean, vaeManifest.pixelStd);
      clip = {
        pixels,
        frames: latentFrames * vaeManifest.config.patch_size_t,
        height: latentHeight * vaeStride,
        width: latentWidth * vaeStride,
      };
      slider.max = String(clip.frames - 1);
      draw(0);

      timing.innerHTML =
        "<table>" +
        `<tr><td>sampling, ${video.timesteps.length} steps</td><td>${(sampleMs / 1000).toFixed(1)} s</td></tr>` +
        `<tr><td>a step</td><td>${(sampleMs / video.timesteps.length).toFixed(0)} ms</td></tr>` +
        `<tr><td>decode</td><td>${decodeMs.toFixed(0)} ms</td></tr>` +
        `<tr><td>frames</td><td>${clip.frames} at ${clip.width}x${clip.height}</td></tr>` +
        `<tr><td>packed rows</td><td>${layout.seq.toLocaleString()}</td></tr>` +
        "</table>";
      say("done.", "press Play, or drag the slider");
    } catch (error) {
      // Named rather than swallowed: a generation that fails silently leaves
      // the page looking like it is still working.
      say("that generation failed.", String(error));
      throw error;
    } finally {
      goButton.disabled = false;
    }
  };
}

void main();
