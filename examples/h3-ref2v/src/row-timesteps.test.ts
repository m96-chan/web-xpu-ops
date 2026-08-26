/**
 * `ref2va`'s per-row noise levels, against upstream's own static method.
 *
 * Issue #212. One forward serves every row of the packed sequence and they are
 * **not all at the same noise level**. `examples/h3-dit`'s `buildRowTimesteps`
 * knows two — the video schedule and the audio one. `ref2va` adds the two the
 * conditioning rows sit at, and the four collapse into fewer whenever any of
 * them coincide, which is the part a port gets wrong quietly.
 *
 * The fixture is `tools/gen_row_timesteps_golden.py`, which **calls**
 * `MiniMaxH3SetTimestepsStep.build_row_timesteps` rather than reproducing it.
 * Indices and timesteps only — no weights, so this runs in CI anywhere.
 */
import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { buildRef2vaRowTimesteps } from "./layout.js";

interface Case {
  name: string;
  numTextTokens: number;
  videoIndices: number[];
  audioIndices: number[];
  numConditionVideoRows: number;
  numConditionAudioRows: number;
  videoTimestep: number;
  audioTimestep: number;
  timestep: number[];
  timestepIndices: number[];
}

const golden = JSON.parse(
  readFileSync(new URL("../fixtures/row-timesteps.json", import.meta.url), "utf8"),
) as { keyframeNoiseAug: number; conditionAudioTimestep: number; cases: Case[] };

const run = (c: Case): { timestep: Float32Array; timestepIndices: Int32Array } =>
  buildRef2vaRowTimesteps(
    {
      seq: c.numTextTokens + c.videoIndices.length + c.audioIndices.length,
      videoIndices: Int32Array.from(c.videoIndices),
      audioIndices: Int32Array.from(c.audioIndices),
    },
    c.numConditionVideoRows,
    c.numConditionAudioRows,
    c.videoTimestep,
    c.audioTimestep,
  );

describe("h3 ref2v / row timesteps", () => {
  it("reproduces every level and every row's index into them", () => {
    for (const c of golden.cases) {
      const got = run(c);
      expect([...got.timestep], c.name).toEqual(c.timestep.map((t) => Math.fround(t)));
      expect([...got.timestepIndices], c.name).toEqual(c.timestepIndices);
    }
  });

  it("pins the conditioning video rows near clean and the reference audio rows at one", () => {
    // The case with all four levels live: the anchors are the *only* reason
    // there are more than two, so reading them off the result is what says the
    // rule was applied and not merely that the two schedules differ.
    const c = golden.cases.find((x) => x.name === "audio reference rows")!;
    const { timestep, timestepIndices } = run(c);
    const levelOf = (row: number): number => timestep[timestepIndices[row]!]!;
    expect(levelOf(c.videoIndices[0]!)).toBe(Math.fround(golden.keyframeNoiseAug));
    expect(levelOf(c.videoIndices[c.numConditionVideoRows]!)).toBe(Math.fround(c.videoTimestep));
    expect(levelOf(c.audioIndices[0]!)).toBe(Math.fround(golden.conditionAudioTimestep));
    expect(levelOf(c.audioIndices[c.numConditionAudioRows]!)).toBe(Math.fround(c.audioTimestep));
  });

  it("lets the video schedule overtake the anchor rather than pinning it", () => {
    // `max(videoTimestep, 0.999)`, not a constant 0.999. Past the anchor the
    // conditioning rows collapse onto the generated ones and there are two
    // levels where every other case has three — a port that hardcoded the
    // anchor agrees everywhere else and disagrees here.
    const c = golden.cases.find((x) => x.name === "video timestep above the anchor")!;
    const { timestep, timestepIndices } = run(c);
    expect(timestep.length).toBe(2);
    expect(timestep[timestepIndices[c.videoIndices[0]!]!]).toBe(Math.fround(c.videoTimestep));
  });

  it("text rows inherit the video timestep, never an anchor", () => {
    // They never reach an output head, so nothing downstream would show it.
    for (const c of golden.cases) {
      const { timestep, timestepIndices } = run(c);
      for (let row = 0; row < c.numTextTokens; row += 1) {
        expect(timestep[timestepIndices[row]!], `${c.name} text row ${row}`).toBe(Math.fround(c.videoTimestep));
      }
    }
  });
});
