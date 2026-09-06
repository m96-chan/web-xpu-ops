/**
 * A dropped file into the patches the vision tower reads.
 *
 * Issue #212. This is the only part of R2V that is a *browser* problem rather
 * than a port of upstream: getting pixels out of a `File`, at the size
 * `smartResize` asked for, in the layout `patchify` takes.
 *
 * **The resize is the browser's, not PIL's**, and that is stated on the page.
 * `Qwen2VLImageProcessor` resamples with bicubic; `drawImage` uses whatever the
 * browser's `imageSmoothingQuality` gives it. Everything downstream is held to
 * the model's own output; this step is **unmeasured**.
 *
 * Video frames are sampled at H3's own 24 fps and then read at the
 * conditioner's 2 fps by `sampleVideoConditionFrames`, so a reference's own
 * frame rate has to be known — `HTMLVideoElement` gives duration but not fps,
 * so this seeks to the timestamps the sampler asks for instead of decoding
 * everything and dropping most of it.
 */
import { patchify, smartResize, type PatchifyResult, type ProcessorConfig } from "../../h3-ref2v/src/processor.js";
import { sampleVideoConditionFrames } from "../../h3-ref2v/src/presentation.js";

/** H3 normalises every reference onto its own clock before anything reads it. */
export const H3_FPS = 24;

export interface LoadedReference {
  kind: "image" | "video";
  name: string;
  /** An object URL for the thumbnail; the caller revokes it. */
  preview: string;
  patches: PatchifyResult;
  /** One timestamp per merged vision block; empty for an image. */
  blockTimestamps: number[];
  width: number;
  height: number;
}

/** `[height, width]` the model wants this image at, and the canvas to get there. */
function targetSize(height: number, width: number, cfg: ProcessorConfig): [number, number] {
  return smartResize(height, width, cfg.patchSize * cfg.mergeSize, cfg.minPixels, cfg.maxPixels);
}

/**
 * Draws `source` at `[height, width]` and returns its bytes.
 *
 * `imageSmoothingQuality: "high"` because the default is nearest-ish on a large
 * downscale, which is further from bicubic than the browser can get. Still not
 * bicubic.
 */
function drawToBytes(
  source: CanvasImageSource, height: number, width: number,
): Uint8Array {
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext("2d", { willReadFrequently: true });
  if (!context) throw new Error("references: no 2d context");
  context.imageSmoothingEnabled = true;
  context.imageSmoothingQuality = "high";
  context.drawImage(source, 0, 0, width, height);
  const data = context.getImageData(0, 0, width, height).data;
  // RGBA out, RGB in: the alpha channel is dropped rather than composited,
  // because upstream converts to RGB and a transparent PNG would otherwise
  // arrive as whatever the canvas cleared to.
  const out = new Uint8Array(width * height * 3);
  for (let i = 0, at = 0; i < data.length; i += 4) {
    out[at++] = data[i]!;
    out[at++] = data[i + 1]!;
    out[at++] = data[i + 2]!;
  }
  return out;
}

async function loadImage(file: File, cfg: ProcessorConfig): Promise<LoadedReference> {
  const preview = URL.createObjectURL(file);
  const bitmap = await createImageBitmap(file);
  const [height, width] = targetSize(bitmap.height, bitmap.width, cfg);
  const bytes = drawToBytes(bitmap, height, width);
  bitmap.close();
  return {
    kind: "image", name: file.name, preview,
    // One frame in; `patchify` repeats it to fill the temporal patch.
    patches: patchify([bytes], height, width, cfg),
    blockTimestamps: [], height, width,
  };
}

async function loadVideo(file: File, cfg: ProcessorConfig, sampleFps: number): Promise<LoadedReference> {
  const preview = URL.createObjectURL(file);
  const video = document.createElement("video");
  video.src = preview;
  video.muted = true;
  await new Promise<void>((resolve, reject) => {
    video.onloadedmetadata = () => resolve();
    video.onerror = () => reject(new Error(`references: ${file.name} did not decode`));
  });

  // H3 resamples every reference onto 24 fps first, so the frame count the
  // sampler reasons about is the duration at *that* rate — not the file's own.
  const normalisedFrames = Math.max(1, Math.round(video.duration * H3_FPS));
  const { indices, blockTimestamps } = sampleVideoConditionFrames(
    normalisedFrames, H3_FPS, sampleFps, cfg.temporalPatchSize);

  const [height, width] = targetSize(video.videoHeight, video.videoWidth, cfg);
  const frames: Uint8Array[] = [];
  for (const index of indices) {
    // Seeking rather than decoding everything: a 15-second reference is 360
    // frames at 24 fps and the conditioner reads 30 of them.
    await new Promise<void>((resolve, reject) => {
      video.onseeked = () => resolve();
      video.onerror = () => reject(new Error(`references: ${file.name} failed while seeking`));
      video.currentTime = index / H3_FPS;
    });
    frames.push(drawToBytes(video, height, width));
  }
  return {
    kind: "video", name: file.name, preview,
    patches: patchify(frames, height, width, cfg),
    blockTimestamps, height, width,
  };
}

/** One dropped file, loaded and patchified. */
export async function loadReference(
  file: File, cfg: ProcessorConfig, sampleFps: number,
): Promise<LoadedReference> {
  if (file.type.startsWith("video/")) return loadVideo(file, cfg, sampleFps);
  if (file.type.startsWith("image/")) return loadImage(file, cfg);
  throw new Error(`references: ${file.name} is neither an image nor a video (${file.type || "no type"})`);
}
