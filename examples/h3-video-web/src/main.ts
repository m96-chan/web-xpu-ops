/**
 * MiniMax-H3's video decoder in a browser: a latent in, frames on a canvas.
 *
 * Issue #200. Nothing here is new arithmetic —
 * `examples/h3-video/src/decoder-gpu.ts` is the decoder, checked against the
 * model's own pixels to 5.841e-6 — and this is the wiring plus the two things a
 * page needs that a script does not: weights from a folder the visitor picked,
 * and a device that survives between decodes.
 *
 * **9.69 GB of f32 stays on the GPU.** That is the honest requirement and the
 * page says it. Quantising to int8 would be 2.4 GB and is not done yet: it is an
 * accuracy question, and this library's rule is that a wrong kernel is worth
 * less than none.
 */
import { DEFAULT_WEIGHTS_BASE } from "../../web-common/src/byte-source.js";
import {
  requireBoundFolder, wireChangeFolder, type GateElements, type GateOptions,
} from "../../web-common/src/gate.js";
import { createBrowserResidentDevice } from "../../anima-web/src/browser-resident.js";
import { VideoDecoderGpu, denormalise, type VideoDecoderManifest } from "../../h3-video/src/decoder-gpu.js";
import { videoKernels } from "./kernels-web.js";

declare const BUILD_STAMP: string;

/**
 * Where the weights come from.
 *
 * **There is no published default for this model.** Unlike Anima and Z-Image,
 * its licence has not been read through (issue #190), so nothing has been
 * mirrored; the shared base is a fallback so the field is never undefined, and
 * converting your own copy is the documented path. `?weights=` overrides it.
 */
const WEIGHTS_BASE = new URLSearchParams(location.search).get("weights") ?? DEFAULT_WEIGHTS_BASE;
const WEIGHT_FILES = ["decoder.manifest.json", "decoder.bin"];

const $ = <T extends HTMLElement>(id: string): T => {
  const element = document.getElementById(id);
  if (!element) throw new Error(`no element #${id}`);
  return element as T;
};

const status = $<HTMLParagraphElement>("status");
const detail = $<HTMLParagraphElement>("detail");
const goButton = $<HTMLButtonElement>("go");
const sizeSelect = $<HTMLSelectElement>("size");
const sourceSelect = $<HTMLSelectElement>("source");
const seedInput = $<HTMLInputElement>("seed");
const scaleInput = $<HTMLInputElement>("scale");
const canvas = $<HTMLCanvasElement>("frames");
const timing = $<HTMLDivElement>("timing");
const playButton = $<HTMLButtonElement>("play");
const slider = $<HTMLInputElement>("frame-slider");
const frameLabel = $<HTMLSpanElement>("frame-label");

function say(message: string, extra = ""): void {
  status.textContent = message;
  detail.textContent = extra;
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
  downloadSize: "9.69 GB",
  // Written from the same place the page acts on: another demo copied this
  // sentence out of its neighbour's markup once and inherited the wrong size
  // and the wrong licence with it.
  licence:
    "The model is MiniMax's, under the MiniMax H3 Community License Agreement — not this page's MIT, " +
    "and nothing here redistributes it.",
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
  for (let i = 0; i < count; i += 2) {
    const u = Math.max(next(), Number.MIN_VALUE);
    const v = next();
    const r = Math.sqrt(-2 * Math.log(u));
    out[i] = r * Math.cos(2 * Math.PI * v);
    if (i + 1 < count) out[i + 1] = r * Math.sin(2 * Math.PI * v);
  }
  return out;
}

/**
 * A latent `[C, T, H, W]` to hand the decoder.
 *
 * The prior is what the model's KL term trains towards, so it is the honest
 * default. The other two exist because a latent of independent draws decodes to
 * a field of independent noise, and the DiT this normally receives from does
 * not emit that either — a *correlated* latent is the more interesting thing to
 * look at, and `smooth` correlates along time so the frames move together.
 */
function makeLatent(kind: string, C: number, T: number, H: number, W: number, seed: number, scale: number): Float32Array {
  const perFrame = H * W;
  const raw = gaussian(C * T * perFrame, seed);
  const out = new Float32Array(raw.length);
  if (kind === "smooth") {
    const a = 0.8;
    for (let c = 0; c < C; c += 1) {
      for (let p = 0; p < perFrame; p += 1) {
        let state = 0;
        for (let t = 0; t < T; t += 1) {
          const at = (c * T + t) * perFrame + p;
          state = a * state + (1 - a) * raw[at]!;
          out[at] = (state / Math.sqrt((1 - a) / (1 + a))) * scale;
        }
      }
    }
    return out;
  }
  if (kind === "drift") {
    const other = gaussian(raw.length, seed ^ 0x5bf03635);
    for (let c = 0; c < C; c += 1) {
      for (let t = 0; t < T; t += 1) {
        const w = T === 1 ? 0 : t / (T - 1);
        for (let p = 0; p < perFrame; p += 1) {
          const at = (c * T + t) * perFrame + p;
          out[at] = ((1 - w) * raw[at]! + w * other[at]!) * scale;
        }
      }
    }
    return out;
  }
  for (let i = 0; i < out.length; i += 1) out[i] = raw[i]! * scale;
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

/** One frame of `[3, T, H, W]` onto the canvas, clamped to [0, 1]. */
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
      // The decoder's output is denormalised into roughly [0, 1]; the clamp is
      // for the tails, not a tone curve.
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
  // 24 fps, which is the rate the model generates at.
  setTimeout(() => requestAnimationFrame(loop), 1000 / 24);
}

slider.oninput = () => draw(Number(slider.value));
playButton.onclick = () => {
  playing = !playing;
  playButton.textContent = playing ? "Pause" : "Play";
  if (playing) loop();
};

async function main(): Promise<void> {
  say("checking this browser …");
  const source = await requireBoundFolder(gate);
  if (!source) return;

  wireChangeFolder(gate, $<HTMLButtonElement>("bind"), source);
  $<HTMLDivElement>("folder-row").hidden = false;
  $<HTMLSpanElement>("folder-state").textContent = source.describe;

  say("reading the manifest …");
  const manifestBytes = await source.read("decoder.manifest.json", 0, await source.size("decoder.manifest.json"));
  const manifest = JSON.parse(new TextDecoder().decode(manifestBytes)) as VideoDecoderManifest;

  say("starting the device …");
  const device = await createBrowserResidentDevice();

  // One tensor at a time, out of the folder and straight into a GPU buffer.
  // Nothing holds the whole file: a single `Float32Array` over 9.69 GB would be
  // 2.4 billion elements, past what a typed array is guaranteed to hold.
  const started = performance.now();
  let painted = 0;
  const decoder = await VideoDecoderGpu.create(
    device,
    videoKernels,
    manifest,
    async (offset, count) => new Float32Array(await source.read("decoder.bin", offset * 4, count * 4)),
    async (done, total) => {
      // Yielded every 200 MB so the progress line paints. Without a yield, 657
      // reads is a tab that looks frozen for half a minute; with one per
      // tensor, the yields cost more than the reads.
      if (done - painted < 200e6 && done < total) return;
      painted = done;
      say("uploading the decoder …", `${(done / 1e9).toFixed(2)} of ${(total / 1e9).toFixed(2)} GB`);
      await new Promise((resolve) => setTimeout(resolve, 0));
    },
  );
  const uploadSeconds = (performance.now() - started) / 1000;

  goButton.disabled = false;
  say("ready.", `${manifest.tensors.length} tensors, ${(manifest.elements * 4 / 1e9).toFixed(2)} GB, in ${uploadSeconds.toFixed(1)} s`);

  goButton.onclick = async () => {
    goButton.disabled = true;
    playing = false;
    playButton.textContent = "Play";
    try {
      const [T, H, W] = sizeSelect.value.split(",").map(Number) as [number, number, number];
      const c = manifest.config;
      const latent = makeLatent(
        sourceSelect.value, c.in_channels, T, H, W,
        Number(seedInput.value) | 0, Number(scaleInput.value) || 1,
      );

      say("decoding …", `${T * H * W} tokens`);
      const at = performance.now();
      const raw = await decoder.decode(latent, [T, H, W]);
      const took = performance.now() - at;

      const pixels = denormalise(raw, c.out_channels, manifest.pixelMean, manifest.pixelStd);
      clip = { pixels, frames: T * c.patch_size_t, height: H * c.patch_size, width: W * c.patch_size };
      slider.max = String(clip.frames - 1);
      draw(0);

      timing.innerHTML =
        "<table>" +
        `<tr><td>decode</td><td>${took.toFixed(0)} ms</td></tr>` +
        `<tr><td>frames</td><td>${clip.frames} at ${clip.width}x${clip.height}</td></tr>` +
        `<tr><td>tokens</td><td>${(T * H * W).toLocaleString()}</td></tr>` +
        "</table>";
      say("done.", "press Play, or drag the slider");
    } catch (error) {
      // Named rather than swallowed: a decode that fails silently leaves the
      // page looking like it is still working.
      say("that decode failed.", String(error));
      throw error;
    } finally {
      goButton.disabled = false;
    }
  };
}

void main();
