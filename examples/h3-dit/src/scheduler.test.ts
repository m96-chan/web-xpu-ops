/**
 * The sampling schedule against diffusers' own `MiniMaxH3Scheduler`.
 *
 * Issue #210. The schedule is arithmetic on `num_inference_steps` and `shift`,
 * so `tools/gen_scheduler_golden.py` writes a 120 KB JSON with no weights in
 * it and no model licence on it, and this runs everywhere.
 *
 * Both shipped shifts are covered: `scheduler/` is **12.0** and
 * `audio_scheduler/` is **3.0**.
 */
import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { scaleNoise, setTimesteps, step } from "./scheduler.js";

interface Case {
  shift: number;
  numInferenceSteps: number;
  sigmas: number[];
  timesteps: number[];
  sample: number[];
  velocity: number[];
  steps: { index: number; timestep: number; prevSample: number[] }[];
  scaleNoiseAt0_3: number[];
}

const golden = JSON.parse(
  readFileSync(new URL("../fixtures/scheduler.json", import.meta.url), "utf8"),
) as { cases: Case[] };

describe("h3 dit / scheduler", () => {
  it("has cases at both shipped shifts", () => {
    // Without this the loops below would pass over an empty list.
    expect(golden.cases.length).toBeGreaterThan(4);
    expect(new Set(golden.cases.map((c) => c.shift))).toEqual(new Set([12, 3]));
  });

  it("reproduces the sigma grid", () => {
    let worst = 0;
    for (const c of golden.cases) {
      const got = setTimesteps(c.numInferenceSteps, c.shift);
      expect(got.sigmas.length).toBe(c.sigmas.length);
      // The terminal sigma is exactly zero, not nearly: `step` divides by
      // `sigmas[index]` and the last blend is `ratio = 0`, which is what makes
      // the final sample the denoised one rather than a mixture.
      expect(got.sigmas[got.sigmas.length - 1]).toBe(0);
      for (let i = 0; i < c.sigmas.length; i += 1) worst = Math.max(worst, Math.abs(got.sigmas[i]! - c.sigmas[i]!));
      for (let i = 0; i < c.timesteps.length; i += 1) {
        worst = Math.max(worst, Math.abs(got.timesteps[i]! - c.timesteps[i]!));
      }
    }
    console.log(`h3 scheduler: sigma/timestep worst ${worst.toExponential(3)}`);
    // The grid is built in f32 on both sides, so this is exact rather than close.
    expect(worst).toBe(0);
  });

  it("puts t = 1 at clean and gives the terminal sigma no evaluation", () => {
    const s = setTimesteps(8, 12);
    // `timesteps = 1 - sigmas[:-1]`, so eight grid points drive **seven**
    // forwards. Counting them as eight walks off the end of the sigma grid.
    expect(s.timesteps.length).toBe(s.sigmas.length - 1);
    // sigma starts at 1, so the first timestep is 0 — the *noisiest* end. The
    // DDPM convention most schedulers here use is the other way round.
    expect(s.sigmas[0]).toBe(1);
    expect(s.timesteps[0]).toBe(0);
  });

  it("reproduces one Euler step", () => {
    let worst = 0;
    for (const c of golden.cases) {
      const schedule = setTimesteps(c.numInferenceSteps, c.shift);
      for (const want of c.steps) {
        const got = step(
          schedule,
          Float32Array.from(c.velocity),
          want.timestep,
          Float32Array.from(c.sample),
          want.index,
        );
        for (let i = 0; i < want.prevSample.length; i += 1) {
          worst = Math.max(worst, Math.abs(got[i]! - want.prevSample[i]!));
        }
      }
    }
    console.log(`h3 scheduler: step worst ${worst.toExponential(3)}`);
    expect(worst).toBeLessThan(1e-6);
  });

  it("reproduces scale_noise", () => {
    let worst = 0;
    for (const c of golden.cases) {
      const got = scaleNoise(Float32Array.from(c.sample), 0.3, Float32Array.from(c.velocity));
      for (let i = 0; i < got.length; i += 1) worst = Math.max(worst, Math.abs(got[i]! - c.scaleNoiseAt0_3[i]!));
    }
    expect(worst).toBeLessThan(1e-6);
    // `t = 1` is clean: the sample comes back untouched, which is what makes it
    // usable for un-noised conditioning anchors.
    const sample = Float32Array.from([1, -2, 3]);
    expect([...scaleNoise(sample, 1, Float32Array.from([9, 9, 9]))]).toEqual([1, -2, 3]);
  });

  it("collapses the f32 collisions the shift creates near sigma = 1", () => {
    // `torch.unique_consecutive` is in upstream's `set_timesteps` and **no
    // realistic step count reaches it**: measured against torch, a shift of 12
    // collapses nothing at 100,000 grid points and 42,208 at 1,000,000, while
    // a shift of 3 collapses nothing at either. So the branch is unreachable in
    // use and would sit unexercised -- which is how a port drops it and nobody
    // notices.
    expect(setTimesteps(1_000_000, 12).sigmas.length).toBe(957_792);
    expect(setTimesteps(1_000_000, 3).sigmas.length).toBe(1_000_000);
    expect(setTimesteps(100_000, 12).sigmas.length).toBe(100_000);
  });

  it("refuses a schedule it cannot step", () => {
    expect(() => setTimesteps(1)).toThrow(/at least 2/);
    const s = setTimesteps(4, 12);
    // The last sigma has no successor; stepping there would read past the grid
    // and, with a zero denominator, hand back NaN rather than an error.
    expect(() => step(s, new Float32Array(3), 0.5, new Float32Array(3), s.sigmas.length - 1)).toThrow(/no next sigma/);
  });
});
