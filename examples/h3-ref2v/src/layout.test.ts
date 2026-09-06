/**
 * The `ref2va` packed layout against diffusers' own pipeline block.
 *
 * Issue #212. Arithmetic on shapes, so the fixture is committed and this runs
 * everywhere. Five cases, chosen so each of the layout's decisions is the only
 * thing separating two of them: one image; an image and a *silent* video; a
 * video *with* sound then an image; a standalone soundtrack between two images;
 * and a video reference past the sixteen latent frames where the two summation
 * orders diverge.
 */
import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { AUDIO_TAG, TEXT_TAG, VIDEO_TAG } from "../../h3-dit/src/layout.js";
import { buildRef2vaSequence, videoReferenceSpan, type Reference } from "./layout.js";

interface Case {
  name: string;
  references: { kind: string; hasAudio: boolean }[];
  visual: [number, number, number][];
  audio: [number, number][];
  numTextTokens: number;
  textTokenTags: number[];
  numLatentFrames: number;
  latentHeight: number;
  latentWidth: number;
  numAudioLatents: number;
  seq: number;
  positionIds: number[];
  tokenTags: number[];
  videoIndices: number[];
  audioIndices: number[];
  textIndices: number[];
  numReferenceVideoRows: number;
  numReferenceAudioRows: number;
}

const golden = JSON.parse(readFileSync(new URL("../fixtures/layout.json", import.meta.url), "utf8")) as {
  patchSize: [number, number, number];
  cases: Case[];
};

const build = (c: Case) => buildRef2vaSequence({
  numTextTokens: c.numTextTokens,
  textTokenTags: c.textTokenTags,
  references: c.references.map((r) => ({ kind: r.kind, hasAudio: r.hasAudio })) as Reference[],
  visualGeometry: c.visual,
  audioRowCounts: c.audio.map(([rows]) => rows),
  numLatentFrames: c.numLatentFrames,
  latentHeight: c.latentHeight,
  latentWidth: c.latentWidth,
  numAudioLatents: c.numAudioLatents,
  patchSize: golden.patchSize,
});

describe("h3 ref2v / layout", () => {
  it("covers each reference arrangement", () => {
    // Without this the loops below would pass over whatever happens to be in
    // the fixture — including an empty list.
    expect(golden.cases.length).toBeGreaterThan(4);
    const kinds = new Set(golden.cases.flatMap((c) => c.references.map((r) => r.kind)));
    expect(kinds).toEqual(new Set(["image", "video", "audio"]));
    expect(golden.cases.some((c) => c.references.some((r) => r.kind === "video" && r.hasAudio))).toBe(true);
    expect(golden.cases.some((c) => c.references.some((r) => r.kind === "video" && !r.hasAudio))).toBe(true);
  });

  it("reproduces the row order, the tags and the rotary grid", () => {
    let worst = 0;
    for (const c of golden.cases) {
      const got = build(c);
      expect(got.seq, c.name).toBe(c.seq);
      expect([...got.tokenTags], c.name).toEqual(c.tokenTags);
      expect([...got.videoIndices], c.name).toEqual(c.videoIndices);
      expect([...got.audioIndices], c.name).toEqual(c.audioIndices);
      expect([...got.textIndices], c.name).toEqual(c.textIndices);
      expect(got.numReferenceVideoRows, c.name).toBe(c.numReferenceVideoRows);
      expect(got.numReferenceAudioRows, c.name).toBe(c.numReferenceAudioRows);
      for (let i = 0; i < c.positionIds.length; i += 1) {
        // The golden is upstream's f64 grid; the comparison narrows it, because
        // `MiniMaxH3RotaryPosEmbed.forward` opens with `to(torch.float32)`.
        worst = Math.max(worst, Math.abs(got.positionIds[i]! - Math.fround(c.positionIds[i]!)));
      }
    }
    console.log(`h3 ref2v layout: position worst ${worst.toExponential(3)}`);
    expect(worst).toBe(0);
  });

  it("gives an image one rotary slot, not a latent frame's 5/3", () => {
    const c = golden.cases.find((x) => x.name === "one image")!;
    const got = build(c);
    // The image block starts where the text ends…
    const imageStart = c.numTextTokens;
    expect(got.positionIds[imageStart * 3]).toBeCloseTo(c.numTextTokens, 5);
    // …and the target video starts exactly one slot later. At 5/3 it would be
    // 1.667 later, which is a video with its conditioning in the wrong place
    // and no other symptom.
    const targetStart = c.videoIndices[c.videoIndices.length - c.numLatentFrames * 4]!;
    expect(got.positionIds[targetStart * 3]).toBeCloseTo(c.numTextTokens + 1, 5);
  });

  it("packs a video reference's soundtrack before its video rows", () => {
    const c = golden.cases.find((x) => x.name === "a video with sound, then an image")!;
    const got = build(c);
    const firstAudio = c.audioIndices[0]!;
    const firstVideo = c.videoIndices[0]!;
    // Audio first, and both starting at the same rotary time.
    expect(firstAudio).toBeLessThan(firstVideo);
    expect(got.positionIds[firstAudio * 3]).toBeCloseTo(got.positionIds[firstVideo * 3]!, 5);
    // A soundtrack has no height coordinate at all.
    for (const row of c.audioIndices) expect(got.positionIds[row * 3 + 1]).toBe(0);
  });

  it("sums a video reference's span sequentially, not pairwise", () => {
    // 5/3 * (1, 4, 4, 4, 4) repeating. The two orders agree below 16 latent
    // frames and differ in the last ulp above it, which is why the fixture has
    // an 18-frame case and why this function is not shared with the `t2va`
    // layout's keyframe anchor.
    expect(videoReferenceSpan(1)).toBeCloseTo(5 / 3, 12);
    expect(videoReferenceSpan(5)).toBeCloseTo((5 / 3) * 17, 12);
    let sequential = 0;
    for (let i = 0; i < 18; i += 1) sequential += (5 / 3) * [1, 4, 4, 4, 4][i % 5]!;
    // Bit-for-bit, not close: the whole point is which order the additions run
    // in, and `toBeCloseTo` cannot see a last-ulp difference.
    expect(videoReferenceSpan(18)).toBe(sequential);
  });

  it("pins a standalone soundtrack to the target's width grid", () => {
    // A video reference's soundtrack uses **its own** grid; a standalone one
    // uses the target's. In the fixture the two canvases differ (4x4 and 4x8
    // references against a 4x8 target), so swapping them is visible.
    const c = golden.cases.find((x) => x.name === "a standalone soundtrack between two images")!;
    const got = build(c);
    const widths = new Set(c.audioIndices.map((row) => got.positionIds[row * 3 + 2]!));
    // Two extremes, whichever grid they came from — the golden pins which.
    expect(widths.size).toBe(2);
  });

  it("tags every reference row as video, and every soundtrack row as audio", () => {
    for (const c of golden.cases) {
      const got = build(c);
      for (const row of got.videoIndices) expect(got.tokenTags[row], c.name).toBe(VIDEO_TAG);
      for (const row of got.audioIndices) expect(got.tokenTags[row], c.name).toBe(AUDIO_TAG);
      // **Not** always TEXT_TAG: a reference's vision block sits in the text
      // range and is tagged video. The golden pins which rows those are.
      for (const row of got.textIndices) expect(got.tokenTags[row], c.name).toBe(c.textTokenTags[row]);
      // Every row is accounted for exactly once.
      expect(got.videoIndices.length + got.audioIndices.length + got.textIndices.length, c.name).toBe(got.seq);
    }
  });

  it("carries a vision block's tag through the text rows", () => {
    // MiniMax-H3 tags a reference's vision block **video (0)** while it sits
    // among the text rows. A port that fills in TEXT_TAG is wrong for every
    // request with a reference — which in ref2va is all of them.
    const c = golden.cases.find((x) => x.name === "a vision block inside the text rows")!;
    expect(new Set(c.textTokenTags)).toEqual(new Set([TEXT_TAG, VIDEO_TAG]));
    const got = build(c);
    expect([...got.tokenTags].slice(0, c.numTextTokens)).toEqual(c.textTokenTags);
    // And the default is still all-text, for a bare prompt.
    expect(() => buildRef2vaSequence({
      numTextTokens: 3, references: [], visualGeometry: [], audioRowCounts: [],
      numLatentFrames: 2, latentHeight: 4, latentWidth: 4, numAudioLatents: 2, patchSize: golden.patchSize,
    })).not.toThrow();
    expect(() => buildRef2vaSequence({
      numTextTokens: 3, textTokenTags: [1, 1], references: [], visualGeometry: [], audioRowCounts: [],
      numLatentFrames: 2, latentHeight: 4, latentWidth: 4, numAudioLatents: 2, patchSize: golden.patchSize,
    })).toThrow(/2 text tags for 3 text rows/);
  });

  it("refuses a reference whose encoder produced nothing", () => {
    const base = golden.cases[0]!;
    expect(() => buildRef2vaSequence({
      numTextTokens: 2, references: [{ kind: "image" }], visualGeometry: [], audioRowCounts: [],
      numLatentFrames: 2, latentHeight: 4, latentWidth: 4, numAudioLatents: 2, patchSize: golden.patchSize,
    })).toThrow(/no encoded geometry/);
    expect(() => buildRef2vaSequence({
      numTextTokens: 2, references: [{ kind: "audio" }], visualGeometry: [], audioRowCounts: [],
      numLatentFrames: 2, latentHeight: 4, latentWidth: 4, numAudioLatents: 2, patchSize: golden.patchSize,
    })).toThrow(/no encoded rows/);
    expect(() => buildRef2vaSequence({
      ...{
        numTextTokens: base.numTextTokens, visualGeometry: base.visual,
        audioRowCounts: base.audio.map(([rows]) => rows), numLatentFrames: base.numLatentFrames,
        latentHeight: base.latentHeight, latentWidth: base.latentWidth,
        numAudioLatents: base.numAudioLatents, patchSize: golden.patchSize,
      },
      references: [{ kind: "text" as unknown as Reference["kind"] }],
    })).toThrow(/image, a video or an audio/);
  });
});
