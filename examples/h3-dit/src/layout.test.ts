/**
 * The packed sequence layout against diffusers' own pipeline blocks.
 *
 * Issue #210. Arithmetic on shapes, so the fixture is committed and this runs
 * everywhere. Three canvases: **square** — where the aspect normalisation is
 * the identity and hides a bug — plus a wide and a tall one.
 */
import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import {
  alignNumFrames,
  audioLatentNumFrames,
  buildPackedSequence,
  buildRowTimesteps,
  patchifyVideoLatents,
  resolveCanvasSize,
  videoLatentNumFrames,
} from "./layout.js";

interface Case {
  numTextTokens: number;
  numLatentFrames: number;
  latentHeight: number;
  latentWidth: number;
  numAudioLatents: number;
  positionIds: number[];
  tokenTags: number[];
  videoIndices: number[];
  audioIndices: number[];
  textIndices: number[];
  uniqueTimesteps: number[];
  timestepIndices: number[];
  patchify: { channels: number; rows: number; cols: number; values: number[] };
}

const golden = JSON.parse(readFileSync(new URL("../fixtures/layout.json", import.meta.url), "utf8")) as {
  patchSize: [number, number, number];
  cases: Case[];
  frames: { requested: number; aligned: number; latentFrames: number; audioLatents: number }[];
  canvases: { aspectWidth: number; aspectHeight: number; shortEdge: number; maxPixels: number; multiple: number; size: [number, number] }[];
};

describe("h3 dit / layout", () => {
  it("has cases at more than one aspect ratio", () => {
    // On a square canvas the aspect normalisation is the identity, so a case
    // list of squares would pass with the normalisation deleted.
    expect(golden.cases.length).toBeGreaterThan(2);
    expect(new Set(golden.cases.map((c) => c.latentHeight === c.latentWidth)).size).toBe(2);
  });

  it("reproduces the row order, the tags and the rotary grid", () => {
    let worst = 0;
    for (const c of golden.cases) {
      const got = buildPackedSequence({
        numTextTokens: c.numTextTokens,
        numLatentFrames: c.numLatentFrames,
        latentHeight: c.latentHeight,
        latentWidth: c.latentWidth,
        numAudioLatents: c.numAudioLatents,
        patchSize: golden.patchSize,
      });
      expect(got.seq).toBe(c.tokenTags.length);
      expect([...got.tokenTags]).toEqual(c.tokenTags);
      expect([...got.videoIndices]).toEqual(c.videoIndices);
      expect([...got.audioIndices]).toEqual(c.audioIndices);
      expect([...got.textIndices]).toEqual(c.textIndices);
      for (let i = 0; i < c.positionIds.length; i += 1) {
        // The golden is upstream's **f64** grid; the comparison narrows it.
        // `MiniMaxH3RotaryPosEmbed.forward` opens with
        // `position_ids.to(torch.float32)`, so f32 is what the model rotates
        // by and the f64 tail is never read. Comparing against the unnarrowed
        // number would demand precision the model itself discards.
        worst = Math.max(worst, Math.abs(got.positionIds[i]! - Math.fround(c.positionIds[i]!)));
      }
    }
    console.log(`h3 layout: position worst ${worst.toExponential(3)}`);
    expect(worst).toBe(0);
  });

  it("spaces latent frames 5/3 * (1, 4, 4, 4, 4) and starts the clock after the text", () => {
    const c = golden.cases.find((x) => x.numLatentFrames >= 6)!;
    const got = buildPackedSequence({
      numTextTokens: c.numTextTokens,
      numLatentFrames: c.numLatentFrames,
      latentHeight: c.latentHeight,
      latentWidth: c.latentWidth,
      numAudioLatents: c.numAudioLatents,
      patchSize: golden.patchSize,
    });
    const rowsPerFrame = (c.latentHeight / golden.patchSize[1]) * (c.latentWidth / golden.patchSize[2]);
    const videoStart = c.videoIndices[0]!;
    const time = (frame: number): number => got.positionIds[(videoStart + frame * rowsPerFrame) * 3]!;
    // The first latent covers one pixel frame and the next four cover four
    // each, so the gaps are 5/3 then 20/3 — uniform spacing is the plausible
    // wrong answer and it drifts the whole clip.
    expect(time(0)).toBeCloseTo(c.numTextTokens, 5);
    expect(time(1) - time(0)).toBeCloseTo(5 / 3, 5);
    expect(time(2) - time(1)).toBeCloseTo(20 / 3, 5);
    expect(time(5) - time(4)).toBeCloseTo(20 / 3, 5);
    // Period five: frame 5 restarts the pattern.
    expect(time(6) - time(5)).toBeCloseTo(5 / 3, 5);
  });

  it("gives audio rows no height and pins them to the width extremes", () => {
    const c = golden.cases[1]!;
    const got = buildPackedSequence({
      numTextTokens: c.numTextTokens,
      numLatentFrames: c.numLatentFrames,
      latentHeight: c.latentHeight,
      latentWidth: c.latentWidth,
      numAudioLatents: c.numAudioLatents,
      patchSize: golden.patchSize,
    });
    const widths = new Set<number>();
    for (const row of got.audioIndices) {
      expect(got.positionIds[row * 3 + 1]).toBe(0);
      widths.add(got.positionIds[row * 3 + 2]!);
    }
    // Two values, not a spread: the left edge for the first channel and the
    // right edge for the second.
    expect(widths.size).toBe(2);
  });

  it("reproduces patchify", () => {
    for (const c of golden.cases) {
      const { channels, rows, cols, values } = c.patchify;
      const latents = Float32Array.from(
        { length: channels * c.numLatentFrames * c.latentHeight * c.latentWidth },
        (_, i) => i,
      );
      const got = patchifyVideoLatents(
        latents, channels, c.numLatentFrames, c.latentHeight, c.latentWidth, golden.patchSize,
      );
      expect(got.length).toBe(rows * cols);
      // Every element names its own coordinate, so these are exact integers
      // and a transposed axis shows up in the digits.
      expect([...got]).toEqual(values);
    }
  });

  it("reproduces the frame alignment and the two latent counts", () => {
    for (const f of golden.frames) {
      expect(alignNumFrames(f.requested)).toBe(f.aligned);
      expect(videoLatentNumFrames(f.aligned)).toBe(f.latentFrames);
      expect(audioLatentNumFrames(f.aligned)).toBe(f.audioLatents);
    }
    // The alignment rounds **up** and 346 is the case upstream calls out: it
    // becomes 362, which is 15.083 s and past the 15 s ceiling.
    expect(alignNumFrames(346)).toBe(362);
    expect(videoLatentNumFrames(362)).toBe(107);
  });

  it("reproduces the default canvas", () => {
    for (const c of golden.canvases) {
      expect(resolveCanvasSize(c.aspectWidth, c.aspectHeight, c.multiple, c.shortEdge, c.maxPixels)).toEqual(c.size);
    }
    expect(() => resolveCanvasSize(16, 1, 32, 480, 1e6)).toThrow(/aspect ratio/);
  });

  it("gives the audio rows their own noise level", () => {
    const c = golden.cases[0]!;
    const layout = buildPackedSequence({
      numTextTokens: c.numTextTokens,
      numLatentFrames: c.numLatentFrames,
      latentHeight: c.latentHeight,
      latentWidth: c.latentWidth,
      numAudioLatents: c.numAudioLatents,
      patchSize: golden.patchSize,
    });
    const { timestep, timestepIndices } = buildRowTimesteps(layout, 0.25, 0.5);
    expect([...timestep]).toEqual(c.uniqueTimesteps);
    expect([...timestepIndices]).toEqual(c.timestepIndices);
    // Text inherits the *video* timestep — it never reaches an output head, so
    // giving it its own would add a third table row that nothing needs.
    for (const row of layout.textIndices) {
      expect(timestep[timestepIndices[row]!]).toBe(timestep[timestepIndices[layout.videoIndices[0]!]!]);
    }
  });
});
