/**
 * MiniMax-H3's audio decoder in a browser: a latent in, a waveform out.
 *
 * Issue #200. Nothing here is new arithmetic — `examples/h3-audio/decoder-gpu.ts`
 * is the decoder, checked against the model's own Python to 5.007e-6, and this
 * is the wiring plus the two things a page needs that a script does not:
 * weights from a folder the visitor picked, and a device that survives between
 * decodes.
 *
 * **There is no prompt, and that is stated on the page rather than implied.**
 * The transformer that would write the latent is 20B and the encoder that would
 * read a prompt is 32B — 27 GB between them at the smallest quantisation
 * published, against 260 MB for this. A latent sampled from the prior is what a
 * VAE decoder is built to receive, so that is what it is given. A page that let
 * you type a prompt and then ignored it would be worse than one that says so.
 */
import { DEFAULT_WEIGHTS_BASE } from "../../web-common/src/byte-source.js";
import {
  requireBoundFolder, wireChangeFolder, type GateElements, type GateOptions,
} from "../../web-common/src/gate.js";
import { createBrowserResidentDevice } from "../../web-common/src/browser-resident.js";
import { AudioVaeWeights, type AudioVaeManifest } from "../../h3-audio/src/decoder.js";
import { AudioVaeGpu } from "../../h3-audio/src/decoder-gpu.js";
import { audioKernels } from "./kernels-web.js";

declare const BUILD_STAMP: string;

/**
 * Where the weights come from.
 *
 * `?weights=` overrides it for anyone pointing at their own copy or a local
 * server. **There is no published default for this model**: unlike Anima and
 * Z-Image, its licence has not been read through (issue #190), so nothing has
 * been mirrored and the base falls back to the shared one only so the field is
 * never undefined. Converting it yourself is the documented path — see the
 * README beside this file.
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
const sourceSelect = $<HTMLSelectElement>("source");
const secondsInput = $<HTMLInputElement>("seconds");
const seedInput = $<HTMLInputElement>("seed");
const scaleInput = $<HTMLInputElement>("scale");
const canvas = $<HTMLCanvasElement>("wave");
const player = $<HTMLAudioElement>("player");
const timing = $<HTMLDivElement>("timing");

function say(message: string, extra = ""): void {
  status.textContent = message;
  detail.textContent = extra;
}

$<HTMLElement>("build").textContent = BUILD_STAMP;

const gateElements: GateElements = {
  dialog: $<HTMLDialogElement>("gate"),
  title: $<HTMLHeadingElement>("gate-title"),
  body: $<HTMLParagraphElement>("gate-body"),
  action: $<HTMLButtonElement>("gate-action"),
  dismiss: $<HTMLButtonElement>("gate-dismiss"),
  progress: $<HTMLParagraphElement>("gate-progress"),
  bar: $<HTMLDivElement>("gate-bar"),
  barFill: $<HTMLDivElement>("gate-bar").firstElementChild as HTMLDivElement,
  why: $<HTMLParagraphElement>("gate-why"),
};

const gate: GateOptions = {
  elements: gateElements,
  files: WEIGHT_FILES,
  weightsBase: WEIGHTS_BASE,
  downloadSize: "260 MB",
  // Written here, from the same place the page acts on, because the first
  // version of another demo copied this sentence out of its neighbour's markup
  // and inherited the wrong size and the wrong licence with it.
  // "Powered by MiniMax H3" is the agreement's own wording for a product that
  // uses the model, not a courtesy. Its territory clause -- worldwide except
  // the EU, the UK, South Korea and the USA -- is why nothing here mirrors the
  // weights: a public host cannot honour it.
  licence:
    "Powered by MiniMax H3. The model is licensed under the MiniMax H3 Community License Agreement, " +
    "not this page's MIT, and nothing here redistributes it.",
};

/** xorshift128+ and Box-Muller, the same generator the other demos seed with. */
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
 * A latent `[C, T]` to hand the decoder.
 *
 * Three ways of drawing one, and none of them is a prompt. The prior is what
 * the model's KL term trains towards, so `prior` is the honest default. The
 * other two exist because 40 Hz of independent noise is 40 Hz of independent
 * noise, and a decoder's response to a *correlated* latent is the more
 * interesting thing to listen to — the DiT it normally receives from does not
 * emit white noise either.
 */
function makeLatent(kind: string, C: number, T: number, seed: number, scale: number): Float32Array {
  const raw = gaussian(C * T, seed);
  const out = new Float32Array(C * T);
  if (kind === "smooth") {
    // A one-pole filter along time, per channel, renormalised so the scale
    // control still means what it says.
    const a = 0.85;
    for (let c = 0; c < C; c += 1) {
      let state = 0;
      for (let t = 0; t < T; t += 1) {
        state = a * state + (1 - a) * raw[c * T + t]!;
        out[c * T + t] = (state / Math.sqrt((1 - a) / (1 + a))) * scale;
      }
    }
    return out;
  }
  if (kind === "tone") {
    // Two draws, crossfaded end to end: the latent moves slowly and coherently
    // rather than jumping every 25 ms.
    const other = gaussian(C * T, seed ^ 0x5bf03635);
    for (let c = 0; c < C; c += 1) {
      for (let t = 0; t < T; t += 1) {
        const w = T === 1 ? 0 : t / (T - 1);
        out[c * T + t] = ((1 - w) * raw[c * T + t]! + w * other[c * T + t]!) * scale;
      }
    }
    return out;
  }
  for (let i = 0; i < out.length; i += 1) out[i] = raw[i]! * scale;
  return out;
}

/** The waveform, as a filled envelope. One column per pixel, min and max. */
function draw(samples: Float32Array): void {
  const context = canvas.getContext("2d");
  if (!context) throw new Error("no 2d context");
  const { width, height } = canvas;
  context.fillStyle = "#1c1f26";
  context.fillRect(0, 0, width, height);
  context.strokeStyle = "#2c313b";
  context.beginPath();
  context.moveTo(0, height / 2);
  context.lineTo(width, height / 2);
  context.stroke();

  context.fillStyle = "#4f7cff";
  const per = Math.max(1, Math.floor(samples.length / width));
  for (let x = 0; x < width; x += 1) {
    let low = 1;
    let high = -1;
    const start = x * per;
    for (let i = start; i < Math.min(start + per, samples.length); i += 1) {
      const v = samples[i]!;
      if (v < low) low = v;
      if (v > high) high = v;
    }
    if (low > high) continue;
    const top = ((1 - high) / 2) * height;
    const bottom = ((1 - low) / 2) * height;
    context.fillRect(x, top, 1, Math.max(1, bottom - top));
  }
}

/**
 * A 16-bit mono WAV, because `<audio>` cannot be handed a Float32Array.
 *
 * Written out rather than routed through `AudioContext.decodeAudioData`: the
 * samples are already decoded, and a WAV blob is what makes the result
 * something the visitor can save.
 */
function wav(samples: Float32Array, sampleRate: number): Blob {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);
  const ascii = (at: number, text: string): void => {
    for (let i = 0; i < text.length; i += 1) view.setUint8(at + i, text.charCodeAt(i));
  };
  ascii(0, "RIFF");
  view.setUint32(4, 36 + samples.length * 2, true);
  ascii(8, "WAVEfmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  ascii(36, "data");
  view.setUint32(40, samples.length * 2, true);
  for (let i = 0; i < samples.length; i += 1) {
    // The decoder already clamps to [-1, 1]; this is the 16-bit conversion,
    // not a second limiter.
    view.setInt16(44 + i * 2, Math.round(samples[i]! * 32767), true);
  }
  return new Blob([buffer], { type: "audio/wav" });
}

async function main(): Promise<void> {
  say("checking this browser …");
  const source = await requireBoundFolder(gate);
  if (!source) return;

  wireChangeFolder(gate, $<HTMLButtonElement>("bind"), source);
  $<HTMLDivElement>("folder-row").hidden = false;
  $<HTMLSpanElement>("folder-state").textContent = source.describe;

  say("reading the decoder …");
  const manifestBytes = await source.read("decoder.manifest.json", 0, await source.size("decoder.manifest.json"));
  const manifest = JSON.parse(new TextDecoder().decode(manifestBytes)) as AudioVaeManifest;
  const binBytes = await source.read("decoder.bin", 0, await source.size("decoder.bin"));
  const weights = new AudioVaeWeights(manifest, new Float32Array(binBytes));

  say("starting the device …");
  const device = await createBrowserResidentDevice();
  const decoder = new AudioVaeGpu(device, audioKernels, manifest, weights);

  goButton.disabled = false;
  say(
    "ready.",
    `${manifest.latentChannels} channels at ${manifest.sampleRate / manifest.hopLength} Hz, ` +
      `${manifest.sampleRate / 1000} kHz out.`,
  );

  goButton.onclick = async () => {
    goButton.disabled = true;
    try {
      const seconds = Math.max(0.2, Number(secondsInput.value) || 2);
      const T = Math.max(1, Math.round((seconds * manifest.sampleRate) / manifest.hopLength));
      const latent = makeLatent(
        sourceSelect.value,
        manifest.latentChannels,
        T,
        Number(seedInput.value) | 0,
        Number(scaleInput.value) || 1,
      );

      say("decoding …", `${T} latent frames`);
      const started = performance.now();
      const samples = await decoder.decode(latent, T);
      const took = performance.now() - started;

      draw(samples);
      player.src = URL.createObjectURL(wav(samples, manifest.sampleRate));
      const audioSeconds = samples.length / manifest.sampleRate;
      timing.innerHTML =
        "<table>" +
        `<tr><td>decode</td><td>${took.toFixed(0)} ms</td></tr>` +
        `<tr><td>audio</td><td>${audioSeconds.toFixed(2)} s</td></tr>` +
        `<tr><td>faster than real time</td><td>${((audioSeconds * 1000) / took).toFixed(1)}x</td></tr>` +
        "</table>";
      say("done.", `${samples.length.toLocaleString()} samples`);
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
