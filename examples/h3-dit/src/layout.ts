/**
 * MiniMax-H3's packed sequence — the layout the transformer refuses to build.
 *
 * Issue #210. `MiniMaxH3Transformer3DModel.forward` takes the row order, the
 * modality tags, the noise level of every row and the `(t, h, w)` rotary grid
 * as *arguments*. Which means every one of them is a free choice this port
 * could make differently, and **not one of them changes a shape**: a wrong
 * layout generates a video, just not the one the weights were trained for.
 *
 * Ported from `diffusers.modular_pipelines.minimax_h3.before_denoise` (rule 7).
 * Six things it decides:
 *
 * - **Video rows are frame-major, then row-major within a frame**, and each row
 *   holds one `(t, h, w)` patch of every channel.
 * - **The spatial grid is aspect-normalised**, then scaled by 32. A square
 *   canvas spans `[0, 32)` on both axes; a wide one spans `[0, 32)` in width
 *   and a *centred sub-interval* in height. On a square canvas the
 *   normalisation is the identity, which is where a bug hides.
 * - It is built with **`np.linspace(..., endpoint=False)`**, i.e.
 *   `start + arange(n) * (stop - start) / n` — not `torch.linspace`, and not
 *   the same rounding.
 * - **Latent frames are spaced `5/3 * (1, 4, 4, 4, 4)` in rotary time.** The
 *   VAE's first latent covers one pixel frame and the rest cover four, so the
 *   spacing is non-uniform and repeats with period five.
 * - **The media clock starts after the text**, so the prompt's length moves the
 *   video's rotary time.
 * - **Audio rows are channel-major**, carry no height coordinate at all, and
 *   are pinned to the two *extremes* of the width grid.
 */
import type { PackedLayout } from "./model.js";

export const VIDEO_TAG = 0;
export const TEXT_TAG = 1;
export const AUDIO_TAG = 2;

export const FPS = 24;
export const AUDIO_LATENTS_PER_SECOND = 40;
export const AUDIO_CHANNELS = 2;

/** `[0, 32)` on the long axis: the rotary span of a canvas. */
const ROPE_SPATIAL_SCALE = 32;
/** 24 fps against the audio's 40 latents/s. */
const ROPE_FRAME_RESCALE = 5 / 3;
/** Pixel frames each latent frame covers: the first one, then four apiece. */
const ROPE_FRAMES_PER_LATENT = [1, 4, 4, 4, 4];

/**
 * `numFrames` snapped up to the next `17n + 5` the video VAE can encode.
 *
 * Not a rounding convenience: the VAE encodes in chunks of 17 pixel frames
 * into 5 latent frames, plus a 5-frame head, and any other count is rejected.
 */
export function alignNumFrames(numFrames: number, framesPerChunk = 17, latentsPerChunk = 5): number {
  if (numFrames < 1) throw new Error(`alignNumFrames: numFrames must be positive, got ${numFrames}`);
  let n = numFrames;
  while (n % framesPerChunk !== latentsPerChunk) n += 1;
  return n;
}

/** Latent frames for an aligned frame count: `5n + 2`. */
export function videoLatentNumFrames(numFrames: number, framesPerChunk = 17, latentsPerChunk = 5): number {
  if (numFrames % framesPerChunk !== latentsPerChunk) {
    throw new Error(`videoLatentNumFrames: needs ${framesPerChunk}n + ${latentsPerChunk}, got ${numFrames}`);
  }
  return Math.floor((numFrames - latentsPerChunk) / framesPerChunk) * latentsPerChunk + 2;
}

/**
 * Python's `round`: **half to even**, not half up.
 *
 * `Math.round(22.5)` is 23 and `round(22.5)` is 22, and that is not a corner
 * case here — it is the default 720p canvas. 1280x720 at a 32 multiple asks
 * for `round(720 / 32) = round(22.5)`, so half-up returns a **736**-pixel
 * canvas where the model wants 704. Nothing downstream would object.
 */
function roundHalfToEven(x: number): number {
  const floor = Math.floor(x);
  const rest = x - floor;
  if (rest > 0.5) return floor + 1;
  if (rest < 0.5) return floor;
  return floor % 2 === 0 ? floor : floor + 1;
}

/** Audio latents covering `numFrames` video frames, rounded onto the latent grid. */
export function audioLatentNumFrames(numFrames: number, fps = FPS, latentsPerSecond = AUDIO_LATENTS_PER_SECOND): number {
  return roundHalfToEven((numFrames / fps) * latentsPerSecond);
}

/**
 * The canvas H3 generates on when the caller names none: 16:9 at `shortEdge`,
 * capped at `maxPixels`, both axes rounded to `multiple`.
 */
export function resolveCanvasSize(
  aspectWidth: number,
  aspectHeight: number,
  multiple: number,
  shortEdge: number,
  maxPixels: number,
  minAspectRatio = 1 / 4,
  maxAspectRatio = 4,
): [number, number] {
  if (aspectWidth <= 0 || aspectHeight <= 0) {
    throw new Error(`resolveCanvasSize: aspect must be positive, got ${aspectWidth}x${aspectHeight}`);
  }
  const ratio = aspectWidth / aspectHeight;
  if (!(minAspectRatio <= ratio && ratio <= maxAspectRatio)) {
    throw new Error(`resolveCanvasSize: aspect ratio ${ratio} is outside [${minAspectRatio}, ${maxAspectRatio}]`);
  }
  let width = ratio >= 1 ? shortEdge * ratio : shortEdge;
  let height = ratio >= 1 ? shortEdge : shortEdge / ratio;
  const area = width * height;
  if (area > maxPixels) {
    const scale = Math.sqrt(maxPixels / area);
    width *= scale;
    height *= scale;
  }
  return [
    Math.max(multiple, roundHalfToEven(height / multiple) * multiple),
    Math.max(multiple, roundHalfToEven(width / multiple) * multiple),
  ];
}

/**
 * `[C, F, H, W]` latents to `[rows, C * pt * ph * pw]`, frame-major then row-major.
 *
 * The channel is the **outer** axis inside a row and the patch offsets the
 * inner ones — `permute(0, 2, 4, 6, 1, 3, 5, 7)` upstream — so a row reads
 * `channel 0's patch, channel 1's patch, …`, not `pixel 0's channels, …`.
 */
export function patchifyVideoLatents(
  latents: Float32Array,
  channels: number,
  frames: number,
  height: number,
  width: number,
  patchSize: [number, number, number],
): Float32Array {
  const [pt, ph, pw] = patchSize;
  if (frames % pt || height % ph || width % pw) {
    throw new Error(`patchifyVideoLatents: ${frames}x${height}x${width} is not divisible by ${patchSize}`);
  }
  if (latents.length !== channels * frames * height * width) {
    throw new Error(`patchifyVideoLatents: ${latents.length} values for ${channels}x${frames}x${height}x${width}`);
  }
  const fp = frames / pt;
  const hp = height / ph;
  const wp = width / pw;
  const rowWidth = channels * pt * ph * pw;
  const out = new Float32Array(fp * hp * wp * rowWidth);
  let at = 0;
  for (let f = 0; f < fp; f += 1) {
    for (let y = 0; y < hp; y += 1) {
      for (let x = 0; x < wp; x += 1) {
        for (let c = 0; c < channels; c += 1) {
          for (let dt = 0; dt < pt; dt += 1) {
            for (let dy = 0; dy < ph; dy += 1) {
              for (let dx = 0; dx < pw; dx += 1) {
                const frame = f * pt + dt;
                const row = y * ph + dy;
                const col = x * pw + dx;
                out[at] = latents[((c * frames + frame) * height + row) * width + col]!;
                at += 1;
              }
            }
          }
        }
      }
    }
  }
  return out;
}

/**
 * One aspect-normalised spatial axis, `dim / patch` coordinates scaled by 32.
 *
 * `np.linspace(left, left + ratio, n, endpoint=False)` is
 * `left + arange(n) * ratio / n` — the right endpoint is *excluded*, so a
 * square canvas spans `[0, 32)` rather than `[0, 32]`.
 */
function spatialPositionGrid(dim: number, patch: number, sqrtArea: number): Float64Array {
  const ratio = dim / sqrtArea;
  const left = (1 - ratio) / 2;
  const n = dim / patch;
  const grid = new Float64Array(n);
  for (let i = 0; i < n; i += 1) grid[i] = (left + (i * ratio) / n) * ROPE_SPATIAL_SCALE;
  return grid;
}

/** The rotary time of every latent frame, starting at `origin`. */
function temporalPositionGrid(numLatentFrames: number, origin: number): Float64Array {
  const out = new Float64Array(numLatentFrames);
  let time = origin;
  for (let i = 0; i < numLatentFrames; i += 1) {
    out[i] = time;
    time += ROPE_FRAME_RESCALE * ROPE_FRAMES_PER_LATENT[i % ROPE_FRAMES_PER_LATENT.length]!;
  }
  return out;
}

export interface PackedSequenceArgs {
  numTextTokens: number;
  numLatentFrames: number;
  latentHeight: number;
  latentWidth: number;
  numAudioLatents: number;
  patchSize: [number, number, number];
  audioChannels?: number;
}

/**
 * The `[text | audio | video]` layout of the `t2va` task — no keyframe anchors.
 *
 * Conditioning rows (`fl2va`, `ref2va`) are **not** built here. They sit between
 * the text and the audio and are tagged as video, and leaving them out is a
 * missing feature rather than a simplification of this one.
 */
export function buildPackedSequence(args: PackedSequenceArgs): PackedLayout {
  const { numTextTokens, numLatentFrames, latentHeight, latentWidth, numAudioLatents } = args;
  const [, patchH, patchW] = args.patchSize;
  const audioChannels = args.audioChannels ?? AUDIO_CHANNELS;

  if (latentHeight % patchH || latentWidth % patchW) {
    throw new Error(`buildPackedSequence: ${latentHeight}x${latentWidth} is not a whole number of ${patchH}x${patchW} patches`);
  }
  const rowsPerFrame = (latentHeight / patchH) * (latentWidth / patchW);
  const numAudioRows = numAudioLatents * audioChannels;
  const numVideoRows = numLatentFrames * rowsPerFrame;
  const seq = numTextTokens + numAudioRows + numVideoRows;

  const audioStart = numTextTokens;
  const videoStart = audioStart + numAudioRows;

  const positionIds = new Float32Array(seq * 3);
  const wide = new Float64Array(seq * 3);

  // Text rows sit on the time axis at their row index; the media clock starts
  // where the text ends.
  for (let i = 0; i < numTextTokens; i += 1) wide[i * 3] = i;

  const sqrtArea = Math.sqrt(latentHeight * latentWidth);
  const heightGrid = spatialPositionGrid(latentHeight, patchH, sqrtArea);
  const widthGrid = spatialPositionGrid(latentWidth, patchW, sqrtArea);

  // Audio: channel-major, one rotary unit per latent, no height, and pinned to
  // the two extremes of the width grid — the first channel to the left edge and
  // everything after it to the right.
  for (let i = 0; i < numAudioRows; i += 1) {
    const row = audioStart + i;
    wide[row * 3] = numTextTokens + (i % numAudioLatents);
    wide[row * 3 + 2] = i < numAudioLatents ? widthGrid[0]! : widthGrid[widthGrid.length - 1]!;
  }

  const frameTime = temporalPositionGrid(numLatentFrames, numTextTokens);
  for (let f = 0; f < numLatentFrames; f += 1) {
    for (let y = 0; y < latentHeight / patchH; y += 1) {
      for (let x = 0; x < latentWidth / patchW; x += 1) {
        const row = videoStart + f * rowsPerFrame + y * (latentWidth / patchW) + x;
        wide[row * 3] = frameTime[f]!;
        wide[row * 3 + 1] = heightGrid[y]!;
        wide[row * 3 + 2] = widthGrid[x]!;
      }
    }
  }
  // The grid is built in f64 — upstream is explicit about that — and narrowed
  // once, at the end, rather than accumulated in f32.
  for (let i = 0; i < wide.length; i += 1) positionIds[i] = wide[i]!;

  const tokenTags = new Int32Array(seq).fill(VIDEO_TAG);
  for (let i = 0; i < numTextTokens; i += 1) tokenTags[i] = TEXT_TAG;
  for (let i = audioStart; i < videoStart; i += 1) tokenTags[i] = AUDIO_TAG;

  return {
    seq,
    tokenTags,
    // Filled by `buildRowTimesteps`; a layout on its own carries no noise level.
    timestepIndices: new Int32Array(seq),
    positionIds,
    videoIndices: Int32Array.from({ length: numVideoRows }, (_, i) => videoStart + i),
    audioIndices: Int32Array.from({ length: numAudioRows }, (_, i) => audioStart + i),
    textIndices: Int32Array.from({ length: numTextTokens }, (_, i) => i),
  };
}

/**
 * A noise level per row, reduced to the `(timestep, timestepIndices)` pair.
 *
 * One forward serves rows at different noise levels: the video and audio rows
 * step down their own schedules — H3 runs the audio schedule at `shift = 3`
 * and the video's at 12, so they are *not* the same number at the same step.
 * **Text rows never reach an output head and inherit the video timestep.**
 */
export function buildRowTimesteps(
  layout: PackedLayout,
  videoTimestep: number,
  audioTimestep: number,
): { timestep: Float32Array; timestepIndices: Int32Array } {
  const rowTimesteps = new Float32Array(layout.seq).fill(Math.fround(videoTimestep));
  for (const row of layout.audioIndices) rowTimesteps[row] = Math.fround(audioTimestep);

  // `torch.unique(sorted=True, return_inverse=True)`.
  const distinct = [...new Set(rowTimesteps)].sort((a, b) => a - b);
  const at = new Map(distinct.map((v, i) => [v, i]));
  const timestepIndices = Int32Array.from(rowTimesteps, (v) => at.get(v)!);
  return { timestep: Float32Array.from(distinct), timestepIndices };
}
