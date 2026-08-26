/**
 * MiniMax-H3's R2V in a browser: references and a prompt in, video out.
 *
 * Issue #212. What separates this from `examples/h3-dit-web` is that **the
 * conditioner cannot be precomputed**. There, a fixed prompt list is baked
 * offline and the page reads embeddings. Here the reference *is* the input, so
 * Qwen3-VL runs in the tab.
 *
 * **Three models, one at a time.** The conditioner is 25.78 GB of int8, the DiT
 * 20.08 and the VAE decoder 2.43 — 48.7 GB, which fits on no card this page
 * will meet. Each is uploaded, used, and dropped before the next, the same
 * staging `examples/h3-dit/src/generate.ts` uses across processes and for the
 * same measured reason.
 *
 * Every stage below is held to the model's own output somewhere in
 * `examples/h3-ref2v` — the layout, the presentation, the text stack, the
 * vision tower and the patchify. The one step that is **not** is the resize:
 * upstream uses PIL's bicubic and a browser has `drawImage`. The page says so.
 */
import { DEFAULT_WEIGHTS_BASE, HttpByteSource, type ByteSource } from "../../web-common/src/byte-source.js";
import {
  requireBoundFolder, wireChangeFolder, type GateElements, type GateOptions,
} from "../../web-common/src/gate.js";
import { createBrowserResidentDevice } from "../../web-common/src/browser-resident.js";
import { buildRef2vaSequence, type Reference } from "../../h3-ref2v/src/layout.js";
import { buildPresentation, type VisionSpecials } from "../../h3-ref2v/src/presentation.js";
import { visionTokenCount, type ProcessorConfig } from "../../h3-ref2v/src/processor.js";
import { alignNumFrames, audioLatentNumFrames, videoLatentNumFrames } from "../../h3-dit/src/layout.js";
import { ByteLevelBpeTokenizer, type BpeVocab } from "../../../llm/tokenizer-bpe.js";
import { loadReference, type LoadedReference } from "./references.js";

declare const BUILD_STAMP: string;

const WEIGHTS_BASE = new URLSearchParams(location.search).get("weights") ?? DEFAULT_WEIGHTS_BASE;
/** The three models and the tokenizer, because a generation needs all of them. */
const WEIGHT_FILES = [
  "conditioner.manifest.json", "conditioner.q8.bin",
  "dit.manifest.json", "dit.q8.bin", "adaln.bin",
  "decoder.q8.manifest.json", "decoder.q8.bin",
  "tokenizer.json",
];

const $ = <T extends HTMLElement>(id: string): T => {
  const element = document.getElementById(id);
  if (!element) throw new Error(`no element #${id}`);
  return element as T;
};

const status = $<HTMLParagraphElement>("status");
const detail = $<HTMLParagraphElement>("detail");
const goButton = $<HTMLButtonElement>("go");
const promptInput = $<HTMLTextAreaElement>("prompt");
const sizeSelect = $<HTMLSelectElement>("size");
const stepsSelect = $<HTMLSelectElement>("steps");
const seedInput = $<HTMLInputElement>("seed");
const dropZone = $<HTMLDivElement>("drop");
const fileInput = $<HTMLInputElement>("files");
const referenceList = $<HTMLDivElement>("references");
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
  downloadSize: "48.7 GB",
  licence:
    "Powered by MiniMax H3. The model is licensed under the MiniMax H3 Community License Agreement, " +
    "not this page's MIT, and nothing here redistributes it.",
};

/** The references, in the order they were dropped — which is packed order. */
const references: LoadedReference[] = [];

function renderReferences(): void {
  referenceList.replaceChildren();
  for (const [index, reference] of references.entries()) {
    const figure = document.createElement("figure");
    const media = document.createElement(reference.kind === "video" ? "video" : "img");
    media.src = reference.preview;
    if (media instanceof HTMLVideoElement) media.muted = true;
    const caption = document.createElement("figcaption");
    const [t, h, w] = reference.patches.grid;
    caption.textContent = `${index + 1}. ${t}x${h}x${w}`;
    caption.title = `${reference.name} — ${reference.width}x${reference.height}`;
    figure.append(media, caption);
    // Click to remove: the order is packed order, so being able to drop one
    // matters more than being able to reorder.
    figure.onclick = () => {
      URL.revokeObjectURL(reference.preview);
      references.splice(index, 1);
      renderReferences();
    };
    referenceList.append(figure);
  }
  dropZone.textContent = references.length
    ? `${references.length} reference${references.length > 1 ? "s" : ""} — drop more, or click one to remove it`
    : "Drop images or video here, or click to choose";
}

interface Clip { pixels: Float32Array; frames: number; height: number; width: number }
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
  draw((Number(slider.value) + 1) % clip.frames);
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

  // **The limits, before anything is uploaded.** `examples/h3-dit-web` spent
  // 21 s on 23 GB to learn one number; issue #211.
  const adapter = await (navigator as Navigator & { gpu?: GPU }).gpu?.requestAdapter();
  const limits = adapter?.limits;
  const workgroupX = limits?.maxComputeWorkgroupSizeX ?? 0;
  const limitLine =
    `workgroupX ${workgroupX}, invocations ${limits?.maxComputeInvocationsPerWorkgroup}, ` +
    `workgroup storage ${limits?.maxComputeWorkgroupStorageSize}, ` +
    `buffer ${((limits?.maxBufferSize ?? 0) / 1e9).toFixed(2)} GB`;
  $<HTMLDivElement>("limits").textContent = limitLine;
  if (workgroupX < 512) {
    say("this browser cannot run the matmul this model needs.",
      `maxComputeWorkgroupSizeX is ${workgroupX} and ops/matmul declares 512 — ${limitLine}`);
    return;
  }

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
  const conditioner = await readJson<{
    processor: ProcessorConfig;
    specials: VisionSpecials;
    videoSampleFps: number;
    textEncoderLayer: number;
    stepCounts?: number[];
  }>("conditioner.manifest.json");
  const dit = await readJson<{ stepCounts: number[] }>("dit.manifest.json");

  for (const count of dit.stepCounts) {
    const option = document.createElement("option");
    option.value = String(count);
    option.textContent = `${count} steps`;
    stepsSelect.append(option);
  }

  // Dropping references works before the models are up — loading one is
  // browser work, and making the visitor wait 90 s to find out their video will
  // not decode would be the wrong order.
  const accept = async (files: FileList | null): Promise<void> => {
    if (!files) return;
    for (const file of Array.from(files)) {
      try {
        say(`reading ${file.name} …`);
        references.push(await loadReference(file, conditioner.processor, conditioner.videoSampleFps));
        renderReferences();
        say("ready when you are.", `${references.length} reference(s)`);
      } catch (error) {
        say(`could not read ${file.name}.`, String(error));
      }
    }
  };
  dropZone.onclick = () => fileInput.click();
  fileInput.onchange = () => void accept(fileInput.files);
  dropZone.ondragover = (event) => { event.preventDefault(); dropZone.classList.add("over"); };
  dropZone.ondragleave = () => dropZone.classList.remove("over");
  dropZone.ondrop = (event) => {
    event.preventDefault();
    dropZone.classList.remove("over");
    void accept(event.dataTransfer?.files ?? null);
  };
  renderReferences();

  // **The tokenizer this repository already has.** `llm/tokenizer-bpe.ts` is
  // Qwen's byte-level BPE and its vocabulary is committed; whether it agrees
  // with H3's own tokenizer is measured in
  // `examples/h3-ref2v/src/tokenizer.test.ts`, on every segment the
  // presentation can produce. It does.
  say("reading the tokenizer …");
  const tokenizer = new ByteLevelBpeTokenizer(await readJson<BpeVocab>("tokenizer.json"));

  say("starting the device …");
  const device = await createBrowserResidentDevice();
  void device;

  goButton.disabled = false;
  say("ready.", "drop a reference, then Generate");

  goButton.onclick = () => {
    if (references.length === 0) {
      say("no references.", "R2V conditions on at least one image or video — that is what makes it R2V.");
      return;
    }
    try {
      const [requestedFrames, height, width] = sizeSelect.value.split(",").map(Number) as [number, number, number];
      const images = references.filter((r) => r.kind === "image");
      const videos = references.filter((r) => r.kind === "video");
      const merge = conditioner.processor.mergeSize;

      const presentation = buildPresentation({
        tokenize: (text) => tokenizer.encode(text),
        specials: conditioner.specials,
        prompt: promptInput.value,
        references: references.map((r) => ({ kind: r.kind, hasAudio: false })) as Reference[],
        imageTokenCounts: images.map((r) => visionTokenCount(r.patches.grid, merge)),
        // Per **block**, so one frame's worth rather than the whole reference.
        videoBlockTokenCounts: videos.map(
          (r) => visionTokenCount([1, r.patches.grid[1], r.patches.grid[2]], merge)),
        videoBlockTimestamps: videos.map((r) => r.blockTimestamps),
      });

      // The reference geometry the layout reads is the shape the **VAE
      // encoder** will produce, not the vision tower's: the tower feeds the
      // conditioner and the encoder feeds the sequence. At 16x spatial
      // compression and this port's still-image handling that is one latent
      // frame per reference.
      const visualGeometry = references.map((r) => [1, r.height / 16, r.width / 16] as [number, number, number]);
      const layout = buildRef2vaSequence({
        numTextTokens: presentation.tokenIds.length,
        textTokenTags: presentation.tokenTags,
        references: references.map((r) => ({ kind: r.kind, hasAudio: false })) as Reference[],
        visualGeometry,
        audioRowCounts: [],
        numLatentFrames: videoLatentNumFrames(alignNumFrames(requestedFrames)),
        latentHeight: height / 16,
        latentWidth: width / 16,
        numAudioLatents: audioLatentNumFrames(alignNumFrames(requestedFrames)),
        patchSize: [1, 2, 2],
      });

      timing.innerHTML =
        "<table>" +
        `<tr><td>references</td><td>${images.length} image, ${videos.length} video</td></tr>` +
        `<tr><td>presentation</td><td>${presentation.tokenIds.length} tokens, ` +
        `${presentation.tokenTags.filter((t) => t === 0).length} tagged video</td></tr>` +
        `<tr><td>reference rows</td><td>${layout.numReferenceVideoRows}</td></tr>` +
        `<tr><td>packed sequence</td><td>${layout.seq.toLocaleString()} rows</td></tr>` +
        `<tr><td>steps</td><td>${stepsSelect.value}, seed ${seedInput.value}</td></tr>` +
        "</table>";

      // **Everything that decides *what* the models are given is here and is
      // held to the model.** What is not here is the running of them: the
      // conditioner and the vision tower have CPU references and no GPU path,
      // and no converter has written a `conditioner.q8.bin`. Said plainly
      // rather than dressed as a failure.
      say(
        "the request is built; the models are not wired yet.",
        `${layout.seq.toLocaleString()} packed rows, of which ${layout.numReferenceVideoRows} are references — ` +
          "the conditioner still needs a GPU path and a conversion. Issue #212.",
      );
    } catch (error) {
      say("that request could not be built.", String(error));
    }
  };
}

void main();
