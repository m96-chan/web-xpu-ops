/**
 * A reference in, video out — MiniMax-H3's `ref2va`, end to end.
 *
 * Issue #212. `examples/h3-dit/src/generate.ts` is the `t2va` half and reads a
 * prompt embedding somebody baked offline. Here **the reference is the input**,
 * so nothing can be precomputed and four models run, one at a time:
 *
 * | | |
 * | --- | --- |
 * | the visual VAE **encoder** | 0.72 GB — the reference becomes latents |
 * | the **conditioner** (Qwen3-VL) | 25.78 GB — `hidden_states[50]` |
 * | `transformer_ref` | 20.66 GB — a different checkpoint from `t2va`'s |
 * | the visual VAE **decoder** | 2.43 GB |
 *
 * **48.9 GB, which fits on no card.** Each is uploaded, used, dropped and
 * *reclaimed* before the next: `destroy()` only schedules the freeing and Dawn
 * releases on GPU work rather than on a timer, so `device.reclaim()` sits at
 * every boundary. Issue #213 has that measurement.
 *
 * ## Three things `t2va` does not do
 *
 * - **The reference latents are noised before they are packed**, at
 *   `keyframe_noise_aug = 0.999` rather than clean, because the released model
 *   was trained with its anchors very slightly noised. One noise draw per
 *   reference, drawn *before* the target's — upstream's order, and it says the
 *   order is part of what a generator reproduces.
 * - **Four noise levels per forward, not two.** `buildRef2vaRowTimesteps`.
 * - **The anchors are re-imposed by not being touched.** The scheduler writes
 *   only the rows past the reference ones; there is no mask and no
 *   recomposition, and a port that stepped every row would drift the reference
 *   away over sixteen steps and produce something that looks conditioned.
 *
 * **The seeds do not reproduce MiniMax's.** `gaussianNoise` is xorshift128+ and
 * Box-Muller, not torch's Philox, the same trade `generate.ts` documents.
 *
 *     npx tsx examples/h3-ref2v/src/generate-r2v.ts \
 *       --conditioner ~/h3-work/h3-cond-gpu --encoder ~/h3-work/h3-encoder-whole \
 *       --dit ~/h3-work/h3-ref-gpu --vae ~/h3-work/h3-video-q8 \
 *       --vae-config ~/h3-work/vae-config/Ref2VA/video_vae/config.json \
 *       --reference ~/h3-work/h3-cond-real/pixels.bin --reference-size 256 \
 *       --prompt "the reference, moving" --out ~/h3-out-r2v
 */
import { closeSync, mkdirSync, openSync, readFileSync, readSync, writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { createResidentDevice, type ResidentDevice } from "../../../harness/resident.js";
import { ByteLevelBpeTokenizer, type BpeVocab } from "../../../llm/tokenizer-bpe.js";
import { EncoderGpu, type EncoderGpuManifest } from "../../h3-encoder/src/encoder-gpu.js";
import { encoderKernels } from "../../h3-encoder/src/kernels-node.js";
import {
  AUDIO_CHANNELS, alignNumFrames, audioLatentNumFrames, patchifyVideoLatents,
  unpatchifyVideoLatents, videoLatentNumFrames,
} from "../../h3-dit/src/layout.js";
import { DitGpu, type DitManifest } from "../../h3-dit/src/model-gpu.js";
import { ditKernels } from "../../h3-dit/src/kernels-node.js";
import { scaleNoise, setTimesteps, step as schedulerStep } from "../../h3-dit/src/scheduler.js";
import {
  VideoDecoderGpu, denormalise, unnormaliseLatent, type VideoDecoderManifest,
} from "../../h3-video/src/decoder-gpu.js";
import { videoKernels } from "../../h3-video/src/kernels-node.js";
import { ConditionerGpu, type ConditionerManifest } from "./conditioner-gpu.js";
import { conditionerKernels } from "./kernels-node.js";
import { buildRef2vaRowTimesteps, buildRef2vaSequence } from "./layout.js";
import { buildPresentation, sampleVideoConditionFrames, type VisionSpecials } from "./presentation.js";
import { patchify, visionTokenCount, type ProcessorConfig } from "./processor.js";
import type { Reference } from "./layout.js";
import { qwen3vlPositionGrid } from "./text-encoder.js";

/** `MiniMaxH3ModularPipeline.keyframe_noise_aug` — 0.999, just short of clean. */
const KEYFRAME_NOISE_AUG = 0.999;
/** The pixel convention the visual VAE was trained under: ImageNet's. */
const PIXEL_MEAN = [0.485, 0.456, 0.406];
const PIXEL_STD = [0.229, 0.224, 0.225];

const arg = (name: string): string | undefined => {
  const at = process.argv.indexOf(name);
  return at >= 0 ? process.argv[at + 1] : undefined;
};
const number = (name: string, fallback: number): number => Number(arg(name) ?? fallback);

const conditionerDir = arg("--conditioner");
const encoderDir = arg("--encoder");
const ditDir = arg("--dit");
const vaeDir = arg("--vae");
const vaeConfigPath = arg("--vae-config");
const outDir = arg("--out") ?? "h3-out-r2v";

/**
 * `--reference image:PATH:W:H` or `--reference video:PATH:W:H:FRAMES`,
 * repeatable, **in packed order**.
 *
 * Raw `uint8` RGB, already at a size `smartResize` would have asked for —
 * `patchify` refuses anything else rather than cropping it quietly, and a
 * browser would do the resampling with `drawImage`. A video's frames are the
 * ones the conditioner reads, sampled at `videoSampleFps`.
 */
interface ReferenceInput {
  kind: "image" | "video";
  path: string;
  width: number;
  height: number;
  /** 1 for a still; the sampled count for a video. */
  frames: number;
}
const referenceInputs: ReferenceInput[] = [];
for (let i = 0; i < process.argv.length; i += 1) {
  if (process.argv[i] !== "--reference") continue;
  const spec = process.argv[i + 1] ?? "";
  // Split from the right: a path may contain colons, and every field after it
  // is a number.
  const parts = spec.split(":");
  const kind = parts.shift();
  if (kind !== "image" && kind !== "video") {
    console.error(`--reference must start with image: or video:, not "${spec}"`);
    process.exit(2);
  }
  const numbers = kind === "image" ? 2 : 3;
  if (parts.length < numbers + 1) {
    console.error(`--reference ${kind}:PATH:W:H${kind === "video" ? ":FRAMES" : ""} — got "${spec}"`);
    process.exit(2);
  }
  const tail = parts.splice(parts.length - numbers, numbers).map(Number);
  referenceInputs.push({
    kind, path: parts.join(":"),
    width: tail[0]!, height: tail[1]!, frames: kind === "video" ? tail[2]! : 1,
  });
}
if (!conditionerDir || !encoderDir || !ditDir || !vaeDir || !vaeConfigPath || referenceInputs.length === 0) {
  console.error(
    "generate-r2v: --conditioner, --encoder, --dit, --vae, --vae-config and at least one --reference are required",
  );
  process.exit(2);
}
const prompt = arg("--prompt") ?? "the reference, moving";
/**
 * The released model's own specification, from `MiniMaxAI/MiniMax-H3`'s card
 * and the `ref2va` request in its `scripts/readme/`.
 *
 * **These are defaults and limits, not suggestions.** Every one of them was
 * violated by hand on the first real run of this script — a 1.2-second target
 * at 256 pixels from a 25.9-second reference and a six-word prompt — and each
 * violation looked like a property of the model until the card was read. The
 * numbers live here so that asking for something outside them takes saying so.
 */
const SPEC = {
  /** "Output duration | 4–15 seconds". */
  minSeconds: 4,
  maxSeconds: 15,
  /** The official `ref2va` request's own `duration_seconds`. */
  defaultSeconds: 5,
  /** "The shorter side is set to 768 pixels by default." */
  defaultShortEdge: 768,
  /** "Output frame rate | 24 FPS". */
  fps: 24,
  /** Ref2VA: "Videos: ≤ 3 clips; each clip must be 2–15 seconds long; total duration ≤ 15 seconds". */
  maxVideoReferences: 3,
  minReferenceSeconds: 2,
  maxReferenceSeconds: 15,
  maxTotalReferenceSeconds: 15,
  /** "Images: ≤ 9 images" and "Maximum number of files across all input types is 12". */
  maxImageReferences: 9,
  maxReferences: 12,
  /**
   * The six sections `docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md` defines for a
   * full-reference rewrite. The card calls H3-Context-IR "critical to the
   * quality of the final output" and its own example request is 5,650 prompt
   * tokens; a short prompt is not a small version of that input.
   */
  promptSections: [
    "subject_definitions", "summary", "retention_analysis",
    "detailed_description", "overall_soundscape", "non_diegetic_music",
  ],
} as const;

/** Everything out-of-spec is refused unless this is passed. */
const outOfSpec = process.argv.includes("--out-of-spec");
const complain = (what: string): void => {
  if (outOfSpec) {
    console.warn(`  out of spec, and --out-of-spec was given: ${what}`);
    return;
  }
  console.error(`generate-r2v: ${what}\n  pass --out-of-spec to do it anyway`);
  process.exit(2);
};

/**
 * The reference limits, checked before 26 GB of weights are uploaded.
 *
 * A video's seconds are its sampled frame count over the conditioner's own
 * sample rate — the frames handed to this script are already the ones it reads.
 */
const videoRefs = referenceInputs.filter((r) => r.kind === "video");
const imageRefs = referenceInputs.filter((r) => r.kind === "image");
if (referenceInputs.length > SPEC.maxReferences) {
  complain(`${referenceInputs.length} references and the model takes ${SPEC.maxReferences}`);
}
if (videoRefs.length > SPEC.maxVideoReferences) {
  complain(`${videoRefs.length} video references and the model takes ${SPEC.maxVideoReferences}`);
}
if (imageRefs.length > SPEC.maxImageReferences) {
  complain(`${imageRefs.length} image references and the model takes ${SPEC.maxImageReferences}`);
}

const seconds = Number(arg("--seconds") ?? SPEC.defaultSeconds);
if (seconds < SPEC.minSeconds || seconds > SPEC.maxSeconds) {
  complain(`the model generates ${SPEC.minSeconds}–${SPEC.maxSeconds} s and this asks for ${seconds}`);
}
// `--frames` still overrides, for bisecting against a golden at a fixed count.
const frames = alignNumFrames(number("--frames", Math.round(seconds * SPEC.fps)));
const steps = number("--steps", 16);
const seed = number("--seed", 0);

/**
 * The target canvas, from a short edge and an aspect ratio rather than two
 * pixel counts.
 *
 * Rounded to a multiple of the VAE's 16 and the DiT's 2 — 32 — because
 * everything downstream refuses anything else, and the short edge is kept
 * exact so `--short-edge 768` means 768.
 */
const aspect = arg("--aspect") ?? "1:1";
const shortEdge = number("--short-edge", SPEC.defaultShortEdge);
if (shortEdge !== SPEC.defaultShortEdge) {
  complain(`the model's short edge is ${SPEC.defaultShortEdge} by default and this asks for ${shortEdge}`);
}
const ratio = aspect.split(":").map(Number);
if (ratio.length !== 2 || !ratio.every((v) => Number.isFinite(v) && v > 0)) {
  console.error(`generate-r2v: --aspect wants W:H, not "${aspect}"`);
  process.exit(2);
}
const BLOCK = 32;
const round32 = (v: number): number => Math.max(BLOCK, Math.round(v / BLOCK) * BLOCK);
const portrait = ratio[0]! < ratio[1]!;
const width = number("--width", portrait ? shortEdge : round32(shortEdge * (ratio[0]! / ratio[1]!)));
const height = number("--height", portrait ? round32(shortEdge * (ratio[1]! / ratio[0]!)) : shortEdge);
if (width % BLOCK || height % BLOCK) {
  console.error(`generate-r2v: ${width}x${height} must be a multiple of ${BLOCK}`);
  process.exit(2);
}

const f32 = (path: string): Float32Array => {
  const buffer = readFileSync(path);
  return new Float32Array(buffer.buffer, buffer.byteOffset, buffer.byteLength / 4);
};

/** A file read in ranges, so a 25 GB blob is never held whole. */
const openReader = (path: string) => {
  const fd = openSync(path, "r");
  return {
    read: (offsetBytes: number, byteLength: number): Uint8Array => {
      const bytes = Buffer.allocUnsafe(byteLength);
      const got = readSync(fd, bytes, 0, byteLength, offsetBytes);
      if (got !== byteLength) throw new Error(`${path}: read ${got} of ${byteLength}`);
      return new Uint8Array(bytes.buffer, bytes.byteOffset, byteLength);
    },
    close: () => closeSync(fd),
  };
};

/**
 * xorshift128+ and Box-Muller, as `examples/h3-dit/src/generate.ts` uses.
 *
 * **Not torch's Philox**, so a seed reproduces this port's own runs and not
 * MiniMax's for the same number. Drawn from one stream in upstream's order —
 * the conditioning noise first, one draw per reference, then the target's —
 * because that order is what a generator reproduces even when the generator is
 * a different one.
 */
function noiseStream(seedValue: number): (count: number) => Float32Array {
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
  return (count: number): Float32Array => {
    const out = new Float32Array(count);
    for (let i = 0; i < count; i += 2) {
      const u = Math.max(next(), Number.MIN_VALUE);
      const v = next();
      const r = Math.sqrt(-2 * Math.log(u));
      out[i] = r * Math.cos(2 * Math.PI * v);
      if (i + 1 < count) out[i + 1] = r * Math.sin(2 * Math.PI * v);
    }
    return out;
  };
}

const draw = noiseStream(seed);

/**
 * The RMS of a stage's output, printed at every boundary.
 *
 * A four-model pipeline can run to completion and hand back zeros — this one
 * did, on its first end-to-end run, and every stage reported a plausible
 * duration on the way. A magnitude per stage is what turns that into a
 * bisection instead of a guess.
 */
const rms = (x: Float32Array): number => {
  let sum = 0;
  for (let i = 0; i < x.length; i += 1) sum += x[i]! * x[i]!;
  return Math.sqrt(sum / Math.max(1, x.length));
};

// ---------------------------------------------------------------- the request

const conditionerManifest = JSON.parse(
  readFileSync(`${conditionerDir}/conditioner.manifest.json`, "utf8"),
) as ConditionerManifest & {
  processor: ProcessorConfig; specials: VisionSpecials; videoSampleFps: number;
};
const ditManifest = JSON.parse(readFileSync(`${ditDir}/dit.manifest.json`, "utf8")) as DitManifest;
const encoderManifest = JSON.parse(
  readFileSync(`${encoderDir}/encoder.manifest.json`, "utf8"),
) as EncoderGpuManifest;
const vaeConfig = JSON.parse(readFileSync(vaeConfigPath, "utf8")) as {
  latents_mean: number[]; latents_std: number[]; latent_channels: number;
};

if (!ditManifest.stepCounts.includes(steps)) {
  console.error(
    `this conversion holds modulation tables for ${ditManifest.stepCounts.join(", ")} steps, not ${steps}`,
  );
  process.exit(2);
}
const c = ditManifest.config;
if (c.text_dim !== conditionerManifest.textConfig.hidden_size) {
  console.error(
    `the DiT reads ${c.text_dim}-wide conditioning and the conditioner produces ` +
      `${conditionerManifest.textConfig.hidden_size}`,
  );
  process.exit(2);
}
const vaeStride = 16;
if (height % (vaeStride * c.patch_size[1]) || width % (vaeStride * c.patch_size[2])) {
  console.error(`generate-r2v: ${width}x${height} must be a multiple of ${vaeStride * c.patch_size[2]}`);
  process.exit(2);
}

/** Every reference's frames, and what the tower made of them. */
const references = referenceInputs.map((input) => {
  const raw = readFileSync(input.path);
  const bytes = new Uint8Array(raw.buffer, raw.byteOffset, raw.byteLength);
  const perFrame = input.width * input.height * 3;
  if (bytes.length !== perFrame * input.frames) {
    console.error(
      `${input.path} holds ${bytes.length} bytes and ${input.frames}x${input.width}x${input.height} RGB is ${perFrame * input.frames}`,
    );
    process.exit(2);
  }
  const frames = Array.from({ length: input.frames }, (_, f) => bytes.subarray(f * perFrame, (f + 1) * perFrame));
  return { input, frames, bytes };
});

// The tokenizer this repository already has: Qwen's byte-level BPE, held to
// H3's own on every segment a presentation can produce — `tokenizer.test.ts`.
const tokenizer = new ByteLevelBpeTokenizer(JSON.parse(readFileSync(
  fileURLToPath(new URL("../../../llm/data/qwen-qwen3-4b.bpe-vocab.json", import.meta.url)), "utf8",
)) as BpeVocab);

// The reference durations, now that the conditioner's own sample rate is known.
{
  let total = 0;
  for (const r of videoRefs) {
    const length = r.frames / conditionerManifest.videoSampleFps;
    total += length;
    if (length < SPEC.minReferenceSeconds || length > SPEC.maxReferenceSeconds) {
      complain(
        `${r.path.split("/").pop()} is ${length.toFixed(1)} s and a reference clip is ` +
          `${SPEC.minReferenceSeconds}–${SPEC.maxReferenceSeconds} s`,
      );
    }
  }
  if (total > SPEC.maxTotalReferenceSeconds) {
    complain(`${total.toFixed(1)} s of reference video and the model takes ${SPEC.maxTotalReferenceSeconds}`);
  }
}

/**
 * The prompt's shape.
 *
 * H3-Context-IR is a hosted service and is not in the open release, so this
 * cannot build the rewrite — but it can refuse to pretend a sentence is one.
 * Measured on one request, everything else held: replacing a six-word prompt
 * with the six sections took the port's seam figure from 1.51 to 1.14 and put
 * the reference's own printed shirt on the subject.
 */
{
  const missing = SPEC.promptSections.filter((section) => !prompt.includes(`${section}:`));
  if (missing.length) {
    complain(
      `the prompt has no ${missing.join(", ")} — the model expects H3-Context-IR's six-section rewrite, ` +
        "see docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md in the model repository",
    );
  }
}

const processor = conditionerManifest.processor;
const merge = processor.mergeSize;
const towered = references.map(({ input, frames }) =>
  ({ input, ...patchify(frames, input.height, input.width, processor) }));
for (const t of towered) {
  console.log(`  ${t.input.kind} ${t.input.path.split("/").pop()}: ${t.input.frames} frame(s) of ` +
    `${t.input.width}x${t.input.height} -> grid ${t.grid.join("x")}, ` +
    `${visionTokenCount(t.grid, merge)} vision tokens`);
}

const packedReferences: Reference[] = towered.map((t) => ({ kind: t.input.kind, hasAudio: false }));
// **A video is one vision block per merged frame group**, each with its own
// timestamp, so its token count is one group's worth and not the whole clip's.
// The frames here are already the ones the conditioner reads, so the sampler is
// asked for a stride of one and only its block timestamps are used.
const videoBlocks = towered.filter((t) => t.input.kind === "video").map((t) =>
  sampleVideoConditionFrames(
    t.input.frames, conditionerManifest.videoSampleFps, conditionerManifest.videoSampleFps,
    processor.temporalPatchSize,
  ).blockTimestamps);

const presentation = buildPresentation({
  tokenize: (text) => tokenizer.encode(text),
  specials: conditionerManifest.specials,
  prompt,
  references: packedReferences,
  imageTokenCounts: towered.filter((t) => t.input.kind === "image").map((t) => visionTokenCount(t.grid, merge)),
  videoBlockTokenCounts: towered.filter((t) => t.input.kind === "video")
    .map((t) => visionTokenCount([1, t.grid[1], t.grid[2]], merge)),
  videoBlockTimestamps: videoBlocks,
});

const latentFrames = videoLatentNumFrames(frames);
const latentHeight = height / vaeStride;
const latentWidth = width / vaeStride;
const audioLatents = audioLatentNumFrames(frames);
console.log(
  `"${prompt}" + ${towered.length} reference(s)\n` +
    `  presentation ${presentation.tokenIds.length} tokens\n` +
    `  ${frames} frames of ${width}x${height} -> latent ${latentFrames}x${latentHeight}x${latentWidth}, ${steps} steps`,
);

const device = await createResidentDevice();
if (!device) {
  console.error("generate-r2v: no adapter");
  process.exit(2);
}

/**
 * Drop a model and wait until the card has the memory back.
 *
 * `destroy()` schedules the freeing; Dawn does it on its next tick and it ticks
 * on GPU work. Without this the next stage gets an *invalid* buffer, which does
 * not throw. Issue #213.
 */
async function handOver(label: string, resident: ResidentDevice): Promise<void> {
  const at = performance.now();
  await resident.reclaim();
  console.log(`  ${label} dropped and reclaimed in ${((performance.now() - at) / 1000).toFixed(2)} s`);
}

// ------------------------------------------------- stage 1: the VAE encoder

let referenceLatents: Float32Array;
const referenceGeometry: [number, number, number][] = [];
{
  const weights = openReader(`${encoderDir}/encoder.bin`);
  let at = performance.now();
  const encoder = await EncoderGpu.create(device, encoderKernels(), encoderManifest, weights.read);
  weights.close();
  console.log(`  VAE encoder uploaded in ${((performance.now() - at) / 1000).toFixed(1)} s`);

  /**
   * How much the references are shrunk **before the VAE**, and why there is a
   * knob for it at all.
   *
   * The tower's view of a reference and its anchor rows are separate things:
   * one is vision tokens in the presentation, the other is latents in the
   * packed sequence. Only the second competes with the target for the
   * transformer's attention, and at the model's own geometry it barely does —
   * five seconds at a short edge of 768 is 38,184 target rows against a few
   * hundred of reference, **about 2%**. At 256x256 over 22 frames the target is
   * 448 rows and the same references are 552, which is **123%**: the references
   * are some fifty times more dominant than anything the model was trained on.
   *
   * `--anchor-scale N` box-filters the reference by N before encoding it, so a
   * small target can be given a proportionate anchor without giving up what the
   * tower sees.
   */
  const anchorScale = Math.max(1, Math.round(number("--anchor-scale", 1)));
  const shrink = (bytes: Uint8Array, w: number, h: number, f: number, n: number): {
    bytes: Uint8Array; width: number; height: number;
  } => {
    if (n === 1) return { bytes, width: w, height: h };
    const W = Math.max(BLOCK, Math.floor(w / n / BLOCK) * BLOCK);
    const H = Math.max(BLOCK, Math.floor(h / n / BLOCK) * BLOCK);
    const sx = w / W;
    const sy = h / H;
    const out = new Uint8Array(f * W * H * 3);
    for (let t = 0; t < f; t += 1) {
      for (let y = 0; y < H; y += 1) {
        const y0 = Math.floor(y * sy);
        const y1 = Math.max(y0 + 1, Math.floor((y + 1) * sy));
        for (let x = 0; x < W; x += 1) {
          const x0 = Math.floor(x * sx);
          const x1 = Math.max(x0 + 1, Math.floor((x + 1) * sx));
          for (let c = 0; c < 3; c += 1) {
            let sum = 0;
            let n2 = 0;
            for (let yy = y0; yy < y1; yy += 1) {
              for (let xx = x0; xx < x1; xx += 1) {
                sum += bytes[((t * h + yy) * w + xx) * 3 + c]!;
                n2 += 1;
              }
            }
            out[((t * H + y) * W + x) * 3 + c] = Math.round(sum / n2);
          }
        }
      }
    }
    return { bytes: out, width: W, height: H };
  };

  const packed: Float32Array[] = [];
  for (const { input, bytes: raw } of references) {
    const small = shrink(raw, input.width, input.height, input.frames, anchorScale);
    const bytes = small.bytes;
    // `encode_vae_condition`: ImageNet-normalised `u8 / 255`, channel-major
    // over the whole clip.
    const plane = small.width * small.height;
    const voxels = plane * input.frames;
    const normalised = new Float32Array(3 * voxels);
    for (let f = 0; f < input.frames; f += 1) {
      for (let i = 0; i < plane; i += 1) {
        for (let ch = 0; ch < 3; ch += 1) {
          normalised[(ch * input.frames + f) * plane + i] =
            (bytes[(f * plane + i) * 3 + ch]! / 255 - PIXEL_MEAN[ch]!) / PIXEL_STD[ch]!;
        }
      }
    }
    at = performance.now();
    // **`encodeConditioning`, not `encode`** (issue #216). `encode` is
    // `AutoencoderKLLegacy.encode`, which `encode_base` calls only for a single
    // image; a frame stack goes through `encode_temporal`'s independent
    // 17-frame chunks. Measured on the released weights, the two differ by
    // 17.9% rms at a two-second reference and 21.5% at a three-and-a-half
    // second one -- with the same shape, which is why nothing complained.
    const moments = await encoder.encodeConditioning(normalised, input.frames, small.height, small.width);
    console.log(
      `  ${input.kind} encoded in ${((performance.now() - at) / 1000).toFixed(2)} s -> ` +
        `moments ${moments.C}x${moments.D}x${moments.H}x${moments.W}`,
    );

  // `DiagonalGaussianDistribution(moments).sample()`: mean is the first half of
  // the channels and log-variance the second, and the conditioning is
  // **sampled**, not taken at the mode. Upstream draws it under its own
  // generator seeded 42, independently of the request — a different stream, so
  // this one is too.
  const z = vaeConfig.latent_channels;
  const per = moments.D * moments.H * moments.W;
  const sampleNoise = noiseStream(42)(z * per);
  const sampled = new Float32Array(z * per);
  for (let i = 0; i < sampled.length; i += 1) {
    // **The log-variance is clamped to [-30, 20]** before the exponential, as
    // `DiagonalGaussianDistribution` does. Inert on every reference measured
    // here — a 256x256 photograph runs -11.8 to -3.1 — but the guard is one
    // line and what it guards against is `exp(10)` multiplying a noise sample.
    const logvar = Math.min(20, Math.max(-30, moments.data[z * per + i]!));
    sampled[i] = moments.data[i]! + Math.exp(0.5 * logvar) * sampleNoise[i]!;
  }
  // **Rounded to float16 and back**, which upstream does explicitly: about
  // eleven bits of every conditioning latent, and leaving it out is a
  // conditioning the model was not given.
  const half = new Float16Array(sampled);
  const normalisedLatents = new Float32Array(sampled.length);
  for (let ch = 0; ch < z; ch += 1) {
    const mean = vaeConfig.latents_mean[ch]!;
    const std = vaeConfig.latents_std[ch]!;
    for (let i = 0; i < per; i += 1) {
      normalisedLatents[ch * per + i] = (half[ch * per + i]! - mean) / std;
    }
  }

  // **The anchors are noised.** `scale_noise(x, 0.999, noise)` is
  // `0.999 * x + 0.001 * noise` in H3's `t` convention, and the draw comes
  // first, before the target's.
    const anchorNoise = draw(normalisedLatents.length);
    const noised = scaleNoise(normalisedLatents, KEYFRAME_NOISE_AUG, anchorNoise);
    referenceGeometry.push([moments.D, moments.H, moments.W]);
    packed.push(patchifyVideoLatents(noised, z, moments.D, moments.H, moments.W, c.patch_size));
    console.log(
      `    moments rms ${rms(moments.data).toFixed(4)}, sampled ${rms(sampled).toFixed(4)}, ` +
        `normalised ${rms(normalisedLatents).toFixed(4)}, noised ${rms(noised).toFixed(4)}`,
    );
  }
  const total = packed.reduce((n, x) => n + x.length, 0);
  referenceLatents = new Float32Array(total);
  let at2 = 0;
  for (const x of packed) { referenceLatents.set(x, at2); at2 += x.length; }
  encoder.destroy();
  await handOver("VAE encoder", device);
}

// **The layout is built from what the encoder produced**, not from what the
// caller asked for: a reference's rows are its own latent geometry, which only
// the encoder knows — a video's frame count goes through the causal temporal
// compression on the way.
// `--flat-text-tags` is a bisect handle for issue #216: it marks every text row
// as text, where the presentation marks a reference's vision block as *video*.
// Upstream does the latter and this port agrees with it exactly, so a
// difference here is a fact about the model, not a fix.
const flatTags = process.argv.includes("--flat-text-tags");
const layout = buildRef2vaSequence({
  numTextTokens: presentation.tokenIds.length,
  textTokenTags: flatTags ? presentation.tokenTags.map(() => 1) : presentation.tokenTags,
  references: packedReferences,
  visualGeometry: referenceGeometry,
  audioRowCounts: [],
  numLatentFrames: latentFrames,
  latentHeight,
  latentWidth,
  numAudioLatents: audioLatents,
  patchSize: c.patch_size,
});
const perRowLatents = c.in_channels * c.patch_size[0] * c.patch_size[1] * c.patch_size[2];
if (referenceLatents.length !== layout.numReferenceVideoRows * perRowLatents) {
  console.error(
    `the layout reserved ${layout.numReferenceVideoRows} reference rows and the encoder produced ` +
      `${referenceLatents.length / perRowLatents}`,
  );
  process.exit(1);
}
console.log(
  `  ${layout.seq} packed rows, of which ${layout.numReferenceVideoRows} are the references ` +
    `(${referenceGeometry.map((g) => g.join("x")).join(", ")})`,
);
/**
 * What this port has actually been timed at, said before the wait rather than
 * after it.
 *
 * Attention is quadratic in the packed length, so the model's own defaults —
 * five seconds at a short edge of 768 — are tens of thousands of rows and are
 * not the same order of request as the ones below. Measured on an RTX 5090,
 * int8, 16 steps: 2,036 rows at 4.5 s a step, 3,227 at 7.8, 4,768 at 12.2.
 */
const MEASURED_ROWS = 4768;
if (layout.seq > MEASURED_ROWS) {
  console.log(
    `  note: the longest sequence this port has been timed at is ${MEASURED_ROWS} rows, at 12.2 s a step. ` +
      `This is ${layout.seq}, and attention is quadratic in it.`,
  );
}

// ------------------------------------------------- stage 2: the conditioner

let conditioning: Float32Array;
/**
 * `--conditioning FILE` reads `hidden_states[50]` instead of computing it.
 *
 * A bisect handle, not a shortcut: the conditioner's own massive-activation row
 * differs from the bf16 reference's, and what that costs downstream was
 * recorded as unmeasured for a long time. Feeding this port the reference's own
 * conditioning and changing nothing else is how that gets a number.
 */
const conditioningFile = arg("--conditioning");
if (conditioningFile) {
  conditioning = f32(conditioningFile);
  console.log(
    `  conditioning read from ${conditioningFile} — ${conditioning.length / conditionerManifest.textConfig.hidden_size} rows`,
  );
  if (conditioning.length !== presentation.tokenIds.length * conditionerManifest.textConfig.hidden_size) {
    console.error(
      `it holds ${conditioning.length / conditionerManifest.textConfig.hidden_size} rows and this presentation is ` +
        `${presentation.tokenIds.length} tokens`,
    );
    process.exit(2);
  }
} else {
  const weightsPath =
    `${conditionerDir}/${conditionerManifest.dtype === "q8" ? "conditioner.q8.bin" : "conditioner.bin"}`;
  const reader = openReader(weightsPath);

  /** The two tables the conditioner reads on the host. */
  const hostTable = (name: string, rows: number, cols: number): Float32Array => {
    const entry = conditionerManifest.tensors.find((t) => t.name === name)!;
    if (entry.kind !== "q8") {
      const bytes = reader.read(entry.offset * 4, entry.count * 4);
      return new Float32Array(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
    }
    const scaleEntry = conditionerManifest.tensors.find((t) => t.name === `${name}.scale`)!;
    const b = reader.read(entry.offset * 4, entry.count * 4);
    const words = new Uint32Array(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength));
    const sb = reader.read(scaleEntry.offset * 4, scaleEntry.count * 4);
    const scales = new Float32Array(sb.buffer.slice(sb.byteOffset, sb.byteOffset + sb.byteLength));
    const perRow = Math.ceil(cols / 4);
    const table = new Float32Array(rows * cols);
    for (let r = 0; r < rows; r += 1) {
      for (let col = 0; col < cols; col += 1) {
        const word = words[r * perRow + (col >> 2)]!;
        table[r * cols + col] = (((word >> ((col & 3) * 8)) & 0xff) << 24 >> 24) * scales[r]!;
      }
    }
    return table;
  };

  const t = conditionerManifest.textConfig;
  const v = conditionerManifest.visionConfig;
  const embedTokens = hostTable("embed_tokens.weight", 151936, t.hidden_size);
  const posEmbed = hostTable("visual.pos_embed.weight", v.num_position_embeddings, v.hidden_size);

  let at = performance.now();
  const conditioner = await ConditionerGpu.create(
    device, conditionerKernels(), conditionerManifest, reader.read, { embedTokens, posEmbed });
  reader.close();
  console.log(`  conditioner uploaded in ${((performance.now() - at) / 1000).toFixed(1)} s`);

  // The rows a vision block occupies are its **interior**: the markers are
  // tagged video too and the tower produces no token for them.
  const modalities = presentation.tokenIds.map(
    (id) => (id === conditionerManifest.specials.imagePad ? 1 : id === conditionerManifest.specials.videoPad ? 2 : 0));
  const runs: [number, number][] = [];
  for (let i = 0; i < modalities.length; i += 1) {
    if (modalities[i] === 0) continue;
    const start = i;
    while (i < modalities.length && modalities[i] !== 0) i += 1;
    runs.push([start, i - start]);
  }
  /**
   * One entry per **visual run**, not per reference.
   *
   * An image is one block; a video is one block per merged frame group, each
   * separated by its own `<|vision_start|>` and `<|vision_end|>`. The rotary
   * clock advances per block, so the grid the position builder is given has to
   * be a block's, `[1, h, w]`, and there have to be as many of them as there
   * are runs.
   */
  const blocks: { grid: [number, number, number]; modality: 1 | 2 }[] = [];
  for (const t of towered) {
    if (t.input.kind === "image") {
      blocks.push({ grid: t.grid, modality: 1 });
    } else {
      for (let f = 0; f < t.grid[0]; f += 1) blocks.push({ grid: [1, t.grid[1], t.grid[2]], modality: 2 });
    }
  }
  if (blocks.length !== runs.length) {
    console.error(`the presentation has ${runs.length} vision blocks and the references make ${blocks.length}`);
    process.exit(1);
  }
  const positions = qwen3vlPositionGrid(modalities, blocks, merge);

  // The tower takes every reference's patches at once, in packed order, and
  // its pooled tokens come back in the same order — merge-block order is
  // frame-major, so a video's groups land in the order its blocks were emitted.
  const allPatches = new Float32Array(towered.reduce((n, t) => n + t.pixelValues.length, 0));
  let patchAt = 0;
  for (const t of towered) { allPatches.set(t.pixelValues, patchAt); patchAt += t.pixelValues.length; }

  at = performance.now();
  conditioning = await conditioner.forward({
    tokenIds: Int32Array.from(presentation.tokenIds),
    positions,
    patches: allPatches,
    grids: towered.map((t) => t.grid),
    visualRuns: runs,
  });
  console.log(
    `  conditioned in ${((performance.now() - at) / 1000).toFixed(2)} s — ` +
      `hidden_states[${conditionerManifest.textEncoderLayer}], ${conditioning.length / t.hidden_size} rows`,
  );
  console.log(`  conditioning rms ${rms(conditioning).toFixed(4)}`);
  conditioner.destroy();
  await handOver("conditioner", device);
}

// ------------------------------------------------- stage 3: transformer_ref

let latent: Float32Array;
{
  const video = setTimesteps(steps, 12);
  const audio = setTimesteps(steps, 3);
  const z = c.in_channels;
  const perRow = z * c.patch_size[0] * c.patch_size[1] * c.patch_size[2];

  // **The target's noise comes after the anchor's**, which is upstream's order.
  const targetRows = patchifyVideoLatents(
    draw(z * latentFrames * latentHeight * latentWidth),
    z, latentFrames, latentHeight, latentWidth, c.patch_size);
  let videoRows = new Float32Array(referenceLatents.length + targetRows.length);
  videoRows.set(referenceLatents, 0);
  videoRows.set(targetRows, referenceLatents.length);
  let audioRows = draw(audioLatents * AUDIO_CHANNELS * c.audio_in_channels);

  const weights = openReader(`${ditDir}/${ditManifest.dtype === "q8" ? "dit.q8.bin" : "dit.bin"}`);
  const tables = openReader(`${ditDir}/adaln.bin`);
  let at = performance.now();
  const dit = await DitGpu.create(device, ditKernels(), ditManifest, weights.read, tables.read);
  weights.close();
  tables.close();
  console.log(`  transformer_ref uploaded in ${((performance.now() - at) / 1000).toFixed(1)} s`);

  /**
   * This run's distinct noise levels, mapped onto the ones the conversion
   * built modulation tables for.
   *
   * `buildRef2vaRowTimesteps` returns indices into **its own** sorted unique,
   * which is what upstream's transformer would project live. Nothing here
   * projects: `adaln_proj` is 39.3% of the checkpoint and the tables were
   * evaluated at conversion time, so the two lists have to be reconciled —
   * and a level the conversion never evaluated has to be an error rather than
   * an index into whatever follows the table.
   */
  const declared = ditManifest.schedules[String(steps)]!.levels;
  const remap = (own: Float32Array, stepIndex: number): Int32Array => {
    if (!declared) {
      throw new Error(
        "this conversion predates the `levels` field, so it was built for t2va's two noise levels — " +
          "re-run convert_dit.py with --workflow ref2va",
      );
    }
    const table = declared[stepIndex]!;
    return Int32Array.from(own, (level) => {
      const at = table.findIndex((candidate) => Math.fround(candidate) === level);
      if (at < 0) {
        throw new Error(
          `step ${stepIndex} needs a modulation table at t=${level} and this conversion has ` +
            `[${table.join(", ")}] — re-run convert_dit.py with --workflow ref2va`,
        );
      }
      return at;
    });
  };

  const anchored = layout.numReferenceVideoRows * perRow;
  at = performance.now();
  for (let i = 0; i < video.timesteps.length; i += 1) {
    const plan = buildRef2vaRowTimesteps(
      layout, layout.numReferenceVideoRows, layout.numReferenceAudioRows,
      video.timesteps[i]!, audio.timesteps[i]!);
    const toTable = remap(plan.timestep, i);
    layout.timestepIndices = Int32Array.from(plan.timestepIndices, (own) => toTable[own]!);
    const stepStart = performance.now();
    const velocity = await dit.forward({ video: videoRows, audio: audioRows, text: conditioning, layout, steps, stepIndex: i });

    // **Only the generated rows are ever written.** The anchors are re-imposed
    // by construction, not by a mask — stepping them would drift the reference
    // away over sixteen steps.
    const stepped = schedulerStep(
      video, velocity.video.subarray(anchored), video.timesteps[i]!, videoRows.subarray(anchored), i);
    const next = new Float32Array(videoRows.length);
    next.set(videoRows.subarray(0, anchored), 0);
    next.set(stepped, anchored);
    videoRows = next;
    audioRows = schedulerStep(audio, velocity.audio, audio.timesteps[i]!, audioRows, i);
    console.log(
      `  step ${i + 1}/${video.timesteps.length}  t=${video.timesteps[i]!.toFixed(4)}  ` +
        `${(performance.now() - stepStart).toFixed(0)} ms  ` +
        `velocity rms ${rms(velocity.video).toFixed(4)}, rows ${rms(videoRows).toFixed(4)}`,
    );
  }
  const sampleMs = performance.now() - at;
  console.log(
    `  sampled in ${(sampleMs / 1000).toFixed(1)} s (${(sampleMs / video.timesteps.length).toFixed(0)} ms a step)`,
  );

  latent = unpatchifyVideoLatents(
    videoRows.slice(anchored), z, latentFrames, latentHeight, latentWidth, c.patch_size);
  console.log(`  latent rms ${rms(latent).toFixed(4)} over ${latent.length} values`);

  // **The anchors must come out exactly as they went in.** They are re-imposed
  // by not being written, so any drift means the scheduler reached rows it
  // should not have — and a drifted anchor is a generation conditioned on
  // something the reference never said.
  const anchorsOut = videoRows.subarray(0, anchored);
  let anchorDrift = 0;
  for (let i = 0; i < anchored; i += 1) {
    anchorDrift = Math.max(anchorDrift, Math.abs(anchorsOut[i]! - referenceLatents[i]!));
  }
  console.log(`  anchor drift after ${video.timesteps.length} steps: ${anchorDrift.toExponential(3)}`);
  if (anchorDrift !== 0) {
    console.error("the reference rows changed during sampling — they are meant to be untouched");
    process.exit(1);
  }
  dit.destroy();
  await handOver("transformer_ref", device);
}

// ------------------------------------------------- stage 4: the VAE decoder

const vaeManifest = JSON.parse(
  readFileSync(`${vaeDir}/decoder.q8.manifest.json`, "utf8"),
) as VideoDecoderManifest;
const vaeWeights = openReader(`${vaeDir}/${vaeManifest.dtype === "q8" ? "decoder.q8.bin" : "decoder.bin"}`);
let at = performance.now();
const decoder = await VideoDecoderGpu.create(device, videoKernels(), vaeManifest, vaeWeights.read);
vaeWeights.close();
console.log(`  VAE decoder uploaded in ${((performance.now() - at) / 1000).toFixed(1)} s`);

at = performance.now();
// **Out of the DiT's latent space and into the decoder's.** The reference went
// the other way at stage 1; both directions are the same statistics.
const raw = await decoder.decode(
  unnormaliseLatent(latent, vaeManifest), [latentFrames, latentHeight, latentWidth]);
console.log(`  decoded in ${((performance.now() - at) / 1000).toFixed(2)} s`);
console.log(`  decoder output rms ${rms(raw).toFixed(4)}`);
const shown = denormalise(raw, vaeManifest.config.out_channels, vaeManifest.pixelMean, vaeManifest.pixelStd);

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
// **A run that produced nothing is a failure, not a black video.** Counted
// rather than assumed away: every stage above reported a plausible duration on
// the run that first hit this.
if (high - low < 1e-6) {
  console.error(`the decoded frames are constant at ${low} — see the per-stage rms above for where it went`);
  process.exit(1);
}
mkdirSync(outDir, { recursive: true });
writeFileSync(`${outDir}/frames.rgb`, Buffer.from(bytes));
// The latent as well as the frames: a drift that is already in the latent is
// the sampler's and one that appears at the decode is the decoder's, and only
// having both says which.
writeFileSync(`${outDir}/latent.bin`, Buffer.from(latent.buffer, latent.byteOffset, latent.byteLength));
writeFileSync(`${outDir}/frames.json`, `${JSON.stringify({
  workflow: "ref2va",
  prompt,
  references: referenceInputs.map((r) => ({ kind: r.kind, path: r.path, frames: r.frames, width: r.width, height: r.height })),
  referenceLatents: referenceGeometry,
  referenceRows: layout.numReferenceVideoRows,
  packedRows: layout.seq,
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
  `  ${outFrames} frames of ${outWidth}x${outHeight} -> ${outDir}/frames.rgb, ` +
    `denormalised range [${low.toFixed(4)}, ${high.toFixed(4)}]`,
);

decoder.destroy();
device.destroy();
