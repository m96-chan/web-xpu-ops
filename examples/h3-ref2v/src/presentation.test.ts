/**
 * The `ref2va` presentation against diffusers' own block.
 *
 * Issue #212. The fixture carries the token ids **and a map from each text
 * segment to its ids**, so this holds the *assembly* — order, counts, tags,
 * timestamps — without a BPE implementation in the way. The tokenizer is a
 * separate concern and will get its own golden.
 */
import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { TEXT_TAG, VIDEO_TAG } from "../../h3-dit/src/layout.js";
import { buildPresentation, formatSeconds, sampleVideoConditionFrames } from "./presentation.js";
import type { Reference } from "./layout.js";

const golden = JSON.parse(readFileSync(new URL("../fixtures/presentation.json", import.meta.url), "utf8")) as {
  visionStart: number; visionEnd: number; imagePad: number; videoPad: number;
  temporalPatch: number;
  segments: Record<string, number[]>;
  cases: {
    name: string;
    references: { kind: string; hasAudio: boolean }[];
    imageTokenCounts: number[];
    videoBlockTokenCounts: number[];
    videoBlockTimestamps: number[][];
    prompt: string;
    tokenIds: number[];
    tokenTags: number[];
  }[];
  sampler: {
    numFrames: number; fps: number; sampleFps: number;
    numSampled: number; blockTimestamps: number[]; rendered: string[];
  }[];
};

const specials = {
  visionStart: golden.visionStart, visionEnd: golden.visionEnd,
  imagePad: golden.imagePad, videoPad: golden.videoPad,
};

/** The tokenizer, as the fixture recorded it. An unseen segment is a failure. */
const tokenize = (text: string): number[] => {
  const ids = golden.segments[text];
  if (!ids) throw new Error(`the fixture has no tokens for ${JSON.stringify(text)}`);
  return ids;
};

const build = (c: (typeof golden.cases)[number]) => buildPresentation({
  tokenize,
  specials,
  prompt: c.prompt,
  references: c.references.map((r) => ({ kind: r.kind, hasAudio: r.hasAudio })) as Reference[],
  imageTokenCounts: c.imageTokenCounts,
  videoBlockTokenCounts: c.videoBlockTokenCounts,
  videoBlockTimestamps: c.videoBlockTimestamps,
});

describe("h3 ref2v / presentation", () => {
  it("covers the arrangements that differ", () => {
    expect(golden.cases.length).toBeGreaterThan(4);
    expect(golden.cases.some((c) => c.references.filter((r) => r.kind === "image").length > 1)).toBe(true);
    expect(golden.cases.some((c) => c.references.some((r) => r.kind === "video" && r.hasAudio))).toBe(true);
    expect(golden.cases.some((c) => c.references.some((r) => r.kind === "video" && !r.hasAudio))).toBe(true);
    expect(golden.cases.some((c) => c.references.some((r) => r.kind === "audio"))).toBe(true);
  });

  it("reproduces the token ids and their tags", () => {
    for (const c of golden.cases) {
      const got = build(c);
      expect(got.tokenIds, c.name).toEqual(c.tokenIds);
      expect(got.tokenTags, c.name).toEqual(c.tokenTags);
    }
  });

  it("tags a vision block's markers as video, not just its pads", () => {
    // `[vision_start] + [pad] * n + [vision_end]`, **all** video — so the tag
    // cannot be derived from "is this a pad token", which is the shortcut that
    // reads correctly and is wrong at both ends of every block.
    const c = golden.cases.find((x) => x.name === "one image")!;
    const got = build(c);
    const start = got.tokenIds.indexOf(golden.visionStart);
    expect(start).toBeGreaterThan(-1);
    expect(got.tokenTags[start]).toBe(VIDEO_TAG);
    const end = got.tokenIds.indexOf(golden.visionEnd);
    expect(got.tokenTags[end]).toBe(VIDEO_TAG);
    expect(got.tokenTags[start - 1]).toBe(TEXT_TAG);
  });

  it("labels a video that carries sound as audio first", () => {
    const c = golden.cases.find((x) => x.name === "a video with sound is labelled Audio first")!;
    const got = build(c);
    const audio = tokenize("<Audio 1>: ");
    const video = tokenize("<Video 1>: ");
    // The presentation opens with the audio label, then the video's.
    expect(got.tokenIds.slice(0, audio.length)).toEqual(audio);
    expect(got.tokenIds.slice(audio.length, audio.length + video.length)).toEqual(video);
  });

  it("numbers per modality, not per reference", () => {
    // **A video before an image**, so the image is still `<Picture 1>`. With
    // every image first, a counter that summed the modalities would have agreed
    // in every other case — which it did, until this one was added.
    const c = golden.cases.find((x) => x.name === "a video then an image, still Picture 1")!;
    const got = build(c);
    const picture1 = tokenize("<Picture 1>: ").join(",");
    expect(got.tokenIds.join(",")).toContain(picture1);
    // And *not* `<Picture 2>`, which a per-reference counter would emit here.
    // The fixture does hold that segment — another case has two images — so
    // this has to look in the produced tokens, not in the segment map.
    expect(got.tokenIds.join(",")).not.toContain(tokenize("<Picture 2>: ").join(","));
  });

  it("deduplicates sampled frames when the stride is below one", () => {
    // `video_sample_fps` is a public argument, so a stride under 1 is
    // reachable even though the shipped 2 fps never gets near it — and it is
    // the only arrangement where the deduplication does anything.
    const s = golden.sampler.find((x) => x.sampleFps > x.fps)!;
    const got = sampleVideoConditionFrames(s.numFrames, s.fps, s.sampleFps, golden.temporalPatch);
    expect(got.indices.length).toBe(s.numSampled);
    expect(new Set(got.indices).size).toBe(got.indices.length);
    // Strictly increasing, which is what the deduplication guarantees.
    for (let i = 1; i < got.indices.length; i += 1) {
      expect(got.indices[i]!).toBeGreaterThan(got.indices[i - 1]!);
    }
  });

  it("gives a video one block per merged group", () => {
    const c = golden.cases.find((x) => x.name === "a silent video, three blocks")!;
    const got = build(c);
    const starts = got.tokenIds.filter((id) => id === golden.visionStart).length;
    expect(starts).toBe(c.videoBlockTimestamps[0]!.length);
  });

  it("renders a timestamp with Python's round-half-to-even", () => {
    // The mean of a 2 fps pair is exactly 0.25. `(0.25).toFixed(1)` is "0.3"
    // in JavaScript and "0.2" in Python — a different token, a different
    // embedding, and no other symptom.
    expect(formatSeconds(0.25)).toBe("0.2");
    expect((0.25).toFixed(1)).toBe("0.3");
    expect(formatSeconds(0.75)).toBe("0.8");
    expect(formatSeconds(1.25)).toBe("1.2");
    expect(formatSeconds(2)).toBe("2.0");
    // **0.15 is not a tie**, whatever it looks like: the double is
    // 0.1499999999999999944…, so Python renders "0.1". Detecting the tie on
    // `value * 10` rounds it to exactly 1.5 and gives "0.2" — which is what
    // this did until a 30 fps sampler case put a 0.15 in front of it.
    expect(formatSeconds(0.15)).toBe("0.1");
    expect(0.15 * 10).toBe(1.5);
    expect(formatSeconds(0.35)).toBe("0.3");
    // And every timestamp the sampler produced, as upstream rendered it.
    for (const s of golden.sampler) {
      expect(s.blockTimestamps.map(formatSeconds), `${s.numFrames}@${s.fps}`).toEqual(s.rendered);
    }
  });

  it("reproduces the frame sampling", () => {
    for (const s of golden.sampler) {
      const got = sampleVideoConditionFrames(s.numFrames, s.fps, s.sampleFps, golden.temporalPatch);
      expect(got.indices.length, `${s.numFrames}@${s.fps}`).toBe(s.numSampled);
      expect(got.blockTimestamps.length, `${s.numFrames}@${s.fps}`).toBe(s.blockTimestamps.length);
      for (let i = 0; i < s.blockTimestamps.length; i += 1) {
        expect(got.blockTimestamps[i], `${s.numFrames}@${s.fps}[${i}]`).toBeCloseTo(s.blockTimestamps[i]!, 9);
      }
    }
  });

  it("refuses a video too short to fill one merged group", () => {
    // Upstream raises rather than merging a single frame with itself, and says
    // how many frames would do.
    expect(() => sampleVideoConditionFrames(6, 24, 2, 2)).toThrow(/at least 13 frames/);
  });

  it("refuses a reference whose vision features are missing", () => {
    expect(() => buildPresentation({
      tokenize, specials, prompt: "x", references: [{ kind: "image" }],
      imageTokenCounts: [], videoBlockTokenCounts: [], videoBlockTimestamps: [],
    })).toThrow(/no vision token count/);
    expect(() => buildPresentation({
      tokenize, specials, prompt: "x", references: [{ kind: "video" }],
      imageTokenCounts: [], videoBlockTokenCounts: [], videoBlockTimestamps: [],
    })).toThrow(/no blocks/);
  });
});
