/**
 * MiniMax-H3's `ref2va` packed sequence — the layout R2V conditions through.
 *
 * Issue #212. `[text | reference blocks | target audio | target video]`. What
 * makes this more than `examples/h3-dit`'s `t2va` layout with extra rows is
 * that **the references advance a shared rotary clock**: where the generated
 * video sits in rotary time depends on how many references came before it and
 * how long each one was. That is the layout, not a detail of the presentation.
 *
 * Ported from `MiniMaxH3Ref2VAPrepareLayoutStep.build_ref2va_packed_sequence`
 * (rule 7), and every one of the six decisions below returns a well-formed
 * sequence when it is wrong:
 *
 * - **An image takes exactly one rotary slot**, not a latent frame's `5/3`.
 * - **A video reference's soundtrack rows are packed immediately *before* its
 *   video rows**, sharing their origin — the same alignment the generated audio
 *   and video have.
 * - A video reference advances the clock by `max(audioLatents, videoSpan)`.
 * - That span is summed **sequentially**, which is deliberately *not* how the
 *   `t2va` layout's "last frame" anchor sums the same series: that one
 *   reproduces numpy's pairwise sum, and the two orders differ in the last ulp
 *   from 16 latent frames onwards. Upstream keeps both, one per call site, so
 *   this does too.
 * - A **standalone** audio reference is pinned to the *target's* width grid; a
 *   video reference's soundtrack is pinned to *its own*.
 * - Reference rows are tagged **video (0)** — including, upstream, the
 *   vision-block rows that sit inside the text range.
 *
 * The geometry of a reference is the shape of what its encoder produced, never
 * a separate argument, so the two cannot disagree.
 */
import type { PackedLayout } from "../../h3-dit/src/model.js";
import { AUDIO_CHANNELS, AUDIO_TAG, TEXT_TAG, VIDEO_TAG } from "../../h3-dit/src/layout.js";

/** `[0, 32)` on the long axis, and 24 fps against the audio's 40 latents/s. */
const ROPE_SPATIAL_SCALE = 32;
const ROPE_FRAME_RESCALE = 5 / 3;
const ROPE_FRAMES_PER_LATENT = [1, 4, 4, 4, 4];

export type ReferenceKind = "image" | "video" | "audio";

export interface Reference {
  kind: ReferenceKind;
  /** Only a video reference can carry one, and only then are audio rows packed. */
  hasAudio?: boolean;
}

export interface Ref2vaArgs {
  numTextTokens: number;
  /**
   * The modality tag of every text row, `[numTextTokens]`.
   *
   * **Not a constant.** MiniMax-H3 tags the rows of a reference's vision block
   * **video (0)** even though they sit in the text range, so this is an
   * argument upstream takes and a port that fills in `TEXT_TAG` is wrong for
   * every request that has a reference — which, in `ref2va`, is all of them.
   * Defaults to all-text so a bare prompt still works.
   */
  textTokenTags?: Int32Array | number[];
  references: Reference[];
  /** `[latentFrames, latentHeight, latentWidth]` per image and video reference, in packed order. */
  visualGeometry: [number, number, number][];
  /** Rows per audio-bearing reference, in packed order — `latents * audioChannels`. */
  audioRowCounts: number[];
  numLatentFrames: number;
  latentHeight: number;
  latentWidth: number;
  numAudioLatents: number;
  patchSize: [number, number, number];
  audioChannels?: number;
}

export interface Ref2vaLayout extends PackedLayout {
  /** How many leading video rows are references rather than generated. */
  numReferenceVideoRows: number;
  numReferenceAudioRows: number;
}

/**
 * One aspect-normalised spatial axis — `np.linspace(..., endpoint=False)`.
 *
 * The same function `examples/h3-dit/src/layout.ts` has. Not imported: it is
 * private there, and a reference block's grid is built from *its own* height
 * and width, which is the thing this file exists to get right.
 */
function spatialPositionGrid(dim: number, patch: number, sqrtArea: number): Float64Array {
  const ratio = dim / sqrtArea;
  const left = (1 - ratio) / 2;
  const n = dim / patch;
  const grid = new Float64Array(n);
  for (let i = 0; i < n; i += 1) grid[i] = (left + (i * ratio) / n) * ROPE_SPATIAL_SCALE;
  return grid;
}

/** The `(h, w)` grid of one latent frame, row-major, and the width axis it came from. */
function frameGrid(height: number, width: number, patchH: number, patchW: number): {
  grid: Float64Array; rows: number; widthGrid: Float64Array;
} {
  const sqrtArea = Math.sqrt(height * width);
  const heightGrid = spatialPositionGrid(height, patchH, sqrtArea);
  const widthGrid = spatialPositionGrid(width, patchW, sqrtArea);
  const rows = heightGrid.length * widthGrid.length;
  const grid = new Float64Array(rows * 2);
  let at = 0;
  for (const h of heightGrid) {
    for (const w of widthGrid) {
      grid[at * 2] = h;
      grid[at * 2 + 1] = w;
      at += 1;
    }
  }
  return { grid, rows, widthGrid };
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

/**
 * How much rotary time a video reference occupies, **summed sequentially**.
 *
 * The `t2va` layout's "last frame" keyframe anchor sums the same series with
 * numpy's *pairwise* summation, and the two disagree in the last ulp from 16
 * latent frames onwards. Upstream keeps both rather than picking one, so a port
 * that shares an implementation between them is wrong at exactly one of the two
 * call sites — with no visible symptom, since a video is still produced.
 */
export function videoReferenceSpan(numLatentFrames: number): number {
  let span = 0;
  for (let i = 0; i < numLatentFrames; i += 1) {
    span += ROPE_FRAME_RESCALE * ROPE_FRAMES_PER_LATENT[i % ROPE_FRAMES_PER_LATENT.length]!;
  }
  return span;
}

/**
 * One channel-major audio block.
 *
 * Audio rows carry **no height coordinate** and are pinned to the two extremes
 * of the width grid of *their own* block — the target's grid for a standalone
 * reference, the video's for a soundtrack.
 */
function fillAudioPositions(
  wide: Float64Array,
  start: number,
  numAudioLatents: number,
  rotaryTime: number,
  widthGrid: Float64Array,
  audioChannels: number,
): void {
  const left = widthGrid[0]!;
  const right = widthGrid[widthGrid.length - 1]!;
  for (let i = 0; i < numAudioLatents * audioChannels; i += 1) {
    const row = start + i;
    wide[row * 3] = rotaryTime + (i % numAudioLatents);
    wide[row * 3 + 2] = i < numAudioLatents ? left : right;
  }
}

export function buildRef2vaSequence(args: Ref2vaArgs): Ref2vaLayout {
  const { numTextTokens, references, visualGeometry, audioRowCounts } = args;
  const { numLatentFrames, latentHeight, latentWidth, numAudioLatents } = args;
  const [, patchH, patchW] = args.patchSize;
  const audioChannels = args.audioChannels ?? AUDIO_CHANNELS;

  const target = frameGrid(latentHeight, latentWidth, patchH, patchW);
  const numTargetVideoRows = numLatentFrames * target.rows;
  const numTargetAudioRows = numAudioLatents * audioChannels;

  let numReferenceVideoRows = 0;
  for (const [frames, height, width] of visualGeometry) {
    numReferenceVideoRows += frames * (height / patchH) * (width / patchW);
  }
  const numReferenceAudioRows = audioRowCounts.reduce((sum, rows) => sum + rows, 0);

  const seq = numTextTokens + numReferenceVideoRows + numReferenceAudioRows
    + numTargetAudioRows + numTargetVideoRows;

  // Built in f64 — upstream is explicit — and narrowed once at the end, because
  // `MiniMaxH3RotaryPosEmbed.forward` opens with `position_ids.to(float32)`.
  const wide = new Float64Array(seq * 3);
  for (let i = 0; i < numTextTokens; i += 1) wide[i * 3] = i;

  const videoRuns: { start: number; count: number }[] = [];
  const audioRuns: { start: number; count: number }[] = [];
  let cursor = numTextTokens;
  let rotaryTime = numTextTokens;
  // Both lists skip the references they do not apply to, so they are consumed
  // as cursors alongside the reference list rather than indexed by it.
  let visualAt = 0;
  let audioAt = 0;

  for (const reference of references) {
    if (reference.kind === "image") {
      const geometry = visualGeometry[visualAt];
      if (!geometry) throw new Error("buildRef2vaSequence: an image reference has no encoded geometry");
      visualAt += 1;
      const [frames, height, width] = geometry;
      const grid = frameGrid(height, width, patchH, patchW);
      const count = frames * grid.rows;
      for (let i = 0; i < count; i += 1) {
        wide[(cursor + i) * 3] = rotaryTime;
        wide[(cursor + i) * 3 + 1] = grid.grid[(i % grid.rows) * 2]!;
        wide[(cursor + i) * 3 + 2] = grid.grid[(i % grid.rows) * 2 + 1]!;
      }
      videoRuns.push({ start: cursor, count });
      cursor += count;
      // **One slot, not 5/3.** An image is a single frame and takes a single
      // integer rotary slot.
      rotaryTime += 1;
    } else if (reference.kind === "audio") {
      const rows = audioRowCounts[audioAt];
      if (rows === undefined) throw new Error("buildRef2vaSequence: an audio reference has no encoded rows");
      audioAt += 1;
      const latents = rows / audioChannels;
      // A standalone soundtrack sits on the **target's** width grid.
      fillAudioPositions(wide, cursor, latents, rotaryTime, target.widthGrid, audioChannels);
      audioRuns.push({ start: cursor, count: rows });
      cursor += rows;
      rotaryTime += latents;
    } else if (reference.kind === "video") {
      const rows = reference.hasAudio ? audioRowCounts[audioAt] : 0;
      if (rows === undefined) throw new Error("buildRef2vaSequence: a video reference has no encoded audio rows");
      if (reference.hasAudio) audioAt += 1;
      const geometry = visualGeometry[visualAt];
      if (!geometry) throw new Error("buildRef2vaSequence: a video reference has no encoded geometry");
      visualAt += 1;
      const [frames, height, width] = geometry;
      const grid = frameGrid(height, width, patchH, patchW);
      const videoRows = frames * grid.rows;
      const latents = rows / audioChannels;

      // **Audio first, then video**, sharing an origin — the same alignment the
      // generated pair has.
      fillAudioPositions(wide, cursor, latents, rotaryTime, grid.widthGrid, audioChannels);
      audioRuns.push({ start: cursor, count: rows });
      const videoStart = cursor + rows;

      const frameTime = temporalPositionGrid(frames, rotaryTime);
      for (let f = 0; f < frames; f += 1) {
        for (let r = 0; r < grid.rows; r += 1) {
          const row = videoStart + f * grid.rows + r;
          wide[row * 3] = frameTime[f]!;
          wide[row * 3 + 1] = grid.grid[r * 2]!;
          wide[row * 3 + 2] = grid.grid[r * 2 + 1]!;
        }
      }
      videoRuns.push({ start: videoStart, count: videoRows });
      cursor = videoStart + videoRows;
      rotaryTime += Math.max(latents, videoReferenceSpan(frames));
    } else {
      throw new Error(`buildRef2vaSequence: a reference is an image, a video or an audio, not ${String(reference.kind)}`);
    }
  }

  // The generated rows, on the origin the references left behind.
  const audioStart = cursor;
  const videoStart = audioStart + numTargetAudioRows;
  fillAudioPositions(wide, audioStart, numAudioLatents, rotaryTime, target.widthGrid, audioChannels);
  const frameTime = temporalPositionGrid(numLatentFrames, rotaryTime);
  for (let f = 0; f < numLatentFrames; f += 1) {
    for (let r = 0; r < target.rows; r += 1) {
      const row = videoStart + f * target.rows + r;
      wide[row * 3] = frameTime[f]!;
      wide[row * 3 + 1] = target.grid[r * 2]!;
      wide[row * 3 + 2] = target.grid[r * 2 + 1]!;
    }
  }
  videoRuns.push({ start: videoStart, count: numTargetVideoRows });
  audioRuns.push({ start: audioStart, count: numTargetAudioRows });

  const flatten = (runs: { start: number; count: number }[]): Int32Array => {
    const out = new Int32Array(runs.reduce((sum, run) => sum + run.count, 0));
    let at = 0;
    for (const run of runs) for (let i = 0; i < run.count; i += 1) out[at++] = run.start + i;
    return out;
  };
  const videoIndices = flatten(videoRuns);
  const audioIndices = flatten(audioRuns);
  const textIndices = Int32Array.from({ length: numTextTokens }, (_, i) => i);

  // Text rows carry the tags the caller gave them; the reference and target
  // rows are tagged by index.
  //
  // Upstream assigns text, then audio, then video, and **the order is not
  // observable**: the three index sets are disjoint by construction, so no row
  // is written twice. Recorded rather than tested — a mutation that swaps the
  // last two survives, and it survives because it cannot matter.
  const textTags = args.textTokenTags;
  if (textTags && textTags.length !== numTextTokens) {
    throw new Error(`buildRef2vaSequence: ${textTags.length} text tags for ${numTextTokens} text rows`);
  }
  const tokenTags = new Int32Array(seq);
  for (let i = 0; i < numTextTokens; i += 1) tokenTags[i] = textTags ? textTags[i]! : TEXT_TAG;
  for (const row of audioIndices) tokenTags[row] = AUDIO_TAG;
  for (const row of videoIndices) tokenTags[row] = VIDEO_TAG;

  const positionIds = new Float32Array(seq * 3);
  for (let i = 0; i < wide.length; i += 1) positionIds[i] = wide[i]!;

  return {
    seq,
    tokenTags,
    timestepIndices: new Int32Array(seq),
    positionIds,
    videoIndices,
    audioIndices,
    textIndices,
    numReferenceVideoRows,
    numReferenceAudioRows,
  };
}
