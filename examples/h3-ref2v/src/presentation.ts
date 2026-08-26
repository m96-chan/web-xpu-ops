/**
 * How a `ref2va` request is announced to the conditioner.
 *
 * Issue #212. Before a reference's pixels reach anything, the reference is
 * *labelled* and a run of pad tokens is put where its vision tokens will go.
 * That announcement is what produces the `textTokenTags` `./layout.ts` takes,
 * because **a vision block's rows are tagged video (0) while sitting among the
 * text rows**.
 *
 * Ported from `MiniMaxH3Ref2VATextEncoderStep` (rule 7). Five decisions, each
 * of which yields a well-formed token list when it is wrong:
 *
 * - `"<Picture i>: "`, `"<Audio j>: "`, `"<Video k>: "`, numbered **per
 *   modality** — the second image is `<Picture 2>` even if four references came
 *   between.
 * - A video that carries sound is labelled `"<Audio j>: "` **before**
 *   `"<Video k>: "`, mirroring the order its rows are packed in.
 * - A video gets **one timestamped vision block per merged frame group**.
 * - The timestamp is rendered with Python's `"{:.1f}"`, which rounds **half to
 *   even**. The mean of a 2 fps pair is `0.25` and renders `"<0.2 seconds>"`;
 *   JavaScript's `toFixed(1)` gives `"0.3"`, a different token, a different
 *   embedding, and no other symptom.
 * - The prompt follows **verbatim**: no chat template, no special tokens.
 *
 * The tokenizer is an argument. What this file is responsible for is the
 * assembly — order, counts, tags, timestamps — and holding that to upstream
 * without a BPE implementation in the way is the point.
 */
import { TEXT_TAG, VIDEO_TAG } from "../../h3-dit/src/layout.js";
import type { Reference } from "./layout.js";

/** The four ids the vision blocks are built from. */
export interface VisionSpecials {
  visionStart: number;
  visionEnd: number;
  imagePad: number;
  videoPad: number;
}

/**
 * Python's `"{:.1f}"` — **round half to even**, on one decimal.
 *
 * `(0.25).toFixed(1)` is `"0.3"` in JavaScript and `"0.2"` in Python, and 0.25
 * is exactly the mean of a 2 fps pair — the first block of every reference
 * video. Getting it wrong changes a token, which changes an embedding, and
 * changes nothing else that anyone would see.
 */
export function formatSeconds(value: number): string {
  // **The tie has to be detected on the exact value, not on `value * 10`.**
  // 0.15 is really 0.1499999999999999944…, so Python renders "0.1" — but
  // multiplying by ten rounds it to exactly 1.5, which reads as a tie and gives
  // "0.2". That is not hypothetical: it is the third block of a reference read
  // at 30 fps, and it was wrong here until the fixture covered one.
  //
  // `toFixed(20)` is the exact decimal expansion, correctly rounded far past
  // where a tie could hide. A genuine tie is a 5 followed by nothing else.
  const exact = value.toFixed(20);
  const tail = exact.slice(exact.indexOf(".") + 2);
  if (!/^50*$/.test(tail)) return value.toFixed(1);
  // A true tie is `(2k + 1) / 20`, which scales by ten exactly, so this
  // multiply is safe in *this* branch and only here.
  const floor = Math.floor(value * 10);
  const rounded = floor % 2 === 0 ? floor : floor + 1;
  return (rounded / 10).toFixed(1);
}

export interface SampledVideo {
  /** Frame indices the conditioner reads, into the 24 fps normalised video. */
  indices: number[];
  /** One timestamp per merged block — what `"<{t} seconds>"` renders. */
  blockTimestamps: number[];
}

/**
 * Which frames the conditioner sees, and how its blocks are labelled.
 *
 * Read at `sampleFps`: every `fps / sampleFps`-th frame, **deduplicated** — at
 * a stride below 1 the same frame would otherwise be taken twice. Qwen3-VL then
 * merges the sampled frames in groups of `temporalPatch`, repeating the last
 * one when the count does not divide, and a group is labelled with the **mean**
 * of its timestamps.
 *
 * Pure arithmetic on counts: no pixels reach this.
 */
export function sampleVideoConditionFrames(
  numFrames: number,
  fps: number,
  sampleFps: number,
  temporalPatch: number,
): SampledVideo {
  const stride = fps / sampleFps;
  const indices: number[] = [];
  let cursor = 0;
  // `round` is Python's, so half goes to even. At a stride of 12 it never
  // matters; at 24 / 5 it does.
  const round = (x: number): number => {
    const floor = Math.floor(x);
    const rest = x - floor;
    if (rest > 0.5) return floor + 1;
    if (rest < 0.5) return floor;
    return floor % 2 === 0 ? floor : floor + 1;
  };
  while (round(cursor) < numFrames) {
    const at = round(cursor);
    if (indices.length === 0 || at > indices[indices.length - 1]!) indices.push(at);
    cursor += stride;
  }
  if (indices.length < temporalPatch) {
    const minimum = round((temporalPatch - 1) * stride) + 1;
    throw new Error(
      `sampleVideoConditionFrames: a reference read at ${sampleFps} fps and merged in groups of ${temporalPatch} ` +
        `needs at least ${minimum} frames at ${fps} fps, got ${numFrames}`,
    );
  }

  const timestamps = indices.map((_, i) => i / sampleFps);
  // `-n % patch` is non-negative in Python and can be negative in JavaScript.
  const pad = ((-timestamps.length % temporalPatch) + temporalPatch) % temporalPatch;
  for (let i = 0; i < pad; i += 1) timestamps.push(timestamps[timestamps.length - 1]!);

  const blockTimestamps: number[] = [];
  for (let i = 0; i < timestamps.length; i += temporalPatch) {
    blockTimestamps.push((timestamps[i]! + timestamps[i + temporalPatch - 1]!) / 2);
  }
  return { indices, blockTimestamps };
}

export interface PresentationArgs {
  /** The conditioner's tokenizer, as a function. `add_special_tokens=False`. */
  tokenize: (text: string) => number[];
  specials: VisionSpecials;
  prompt: string;
  references: Reference[];
  /** Vision tokens in each image reference's block, in image order. */
  imageTokenCounts: number[];
  /** Vision tokens per block of each video reference, in video order. */
  videoBlockTokenCounts: number[];
  /** The timestamp of every block, per video reference. */
  videoBlockTimestamps: number[][];
}

/** The token ids of the presentation, and the modality tag of every one. */
export function buildPresentation(args: PresentationArgs): { tokenIds: number[]; tokenTags: number[] } {
  const { tokenize, specials, prompt, references } = args;
  const tokenIds: number[] = [];
  const tokenTags: number[] = [];

  const emitText = (text: string): void => {
    for (const id of tokenize(text)) {
      tokenIds.push(id);
      tokenTags.push(TEXT_TAG);
    }
  };
  const emitVision = (pad: number, count: number): void => {
    // `[vision_start] + [pad] * count + [vision_end]`, **all tagged video** —
    // including the two markers, which is why the tag cannot be derived from
    // "is this a pad token".
    tokenIds.push(specials.visionStart);
    tokenTags.push(VIDEO_TAG);
    for (let i = 0; i < count; i += 1) {
      tokenIds.push(pad);
      tokenTags.push(VIDEO_TAG);
    }
    tokenIds.push(specials.visionEnd);
    tokenTags.push(VIDEO_TAG);
  };

  const counts = { image: 0, video: 0, audio: 0 };
  for (const reference of references) {
    // **Before the kind check**, so a video with sound is announced as audio
    // first — the order its rows are packed in.
    if (reference.hasAudio) {
      counts.audio += 1;
      emitText(`<Audio ${counts.audio}>: `);
    }
    if (reference.kind === "image") {
      counts.image += 1;
      emitText(`<Picture ${counts.image}>: `);
      const tokens = args.imageTokenCounts[counts.image - 1];
      if (tokens === undefined) throw new Error(`buildPresentation: image ${counts.image} has no vision token count`);
      emitVision(specials.imagePad, tokens);
    } else if (reference.kind === "video") {
      counts.video += 1;
      emitText(`<Video ${counts.video}>: `);
      const stamps = args.videoBlockTimestamps[counts.video - 1];
      const tokens = args.videoBlockTokenCounts[counts.video - 1];
      if (!stamps || tokens === undefined) {
        throw new Error(`buildPresentation: video ${counts.video} has no blocks`);
      }
      for (const stamp of stamps) {
        emitText(`<${formatSeconds(stamp)} seconds>`);
        emitVision(specials.videoPad, tokens);
      }
    } else if (reference.kind !== "audio") {
      throw new Error(`buildPresentation: a reference is an image, a video or an audio, not ${String(reference.kind)}`);
    }
  }
  // Verbatim, last.
  emitText(prompt);
  return { tokenIds, tokenTags };
}
