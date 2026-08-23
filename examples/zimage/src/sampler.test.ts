import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { eulerStep, flowSchedule, modelTimestep } from "./sampler.js";

/**
 * The schedule, against Z-Image's own scheduler.
 *
 * Every number here came out of `zimage_utils.get_timesteps_sigmas` and
 * `zimage_utils.step` (`tools/gen_sampler_golden.py`), not out of reading this
 * port and agreeing with it. The schedule is short enough to write from memory,
 * which is exactly why it is not: the four ways to get it plausibly wrong are
 * listed in the generator, and each produces images rather than errors.
 */
const golden = JSON.parse(
  readFileSync(fileURLToPath(new URL("../fixtures/sampler.golden.json", import.meta.url)), "utf8"),
) as {
  steps: number;
  shift: number;
  numTrainTimesteps: number;
  timesteps: number[];
  sigmas: number[];
  modelTimesteps: number[];
  step: { sample: number[]; modelOutput: number[]; results: number[][] };
};

describe("flowSchedule", () => {
  const got = flowSchedule(golden.steps, golden.shift, golden.numTrainTimesteps);

  it("produces one timestep per step", () => {
    expect(got.timesteps).toHaveLength(golden.steps);
  });

  it("produces one more sigma than steps, ending at zero", () => {
    // The appended 0 is what lets the last step land on a clean latent. A port
    // that omitted it would run seven steps and stop short of the image.
    expect(got.sigmas).toHaveLength(golden.steps + 1);
    expect(got.sigmas[golden.steps]).toBe(0);
  });

  it("matches the model's timesteps", () => {
    got.timesteps.forEach((t, i) => expect(t).toBeCloseTo(golden.timesteps[i]!, 4));
  });

  it("matches the model's sigmas", () => {
    got.sigmas.forEach((s, i) => expect(s).toBeCloseTo(golden.sigmas[i]!, 6));
  });

  it("starts at sigma 1", () => {
    // `linspace(1000, 1, steps+1)[:-1]` starts at exactly 1000, so the first
    // sigma is exactly 1 whatever the shift — the shift maps 1 to 1. A port
    // using `linspace(1000, 1, steps)` would start somewhere else and every
    // number after it would differ.
    expect(got.sigmas[0]).toBe(1);
  });

  it("is monotonically decreasing", () => {
    for (let i = 1; i < got.sigmas.length; i += 1) {
      expect(got.sigmas[i]!).toBeLessThan(got.sigmas[i - 1]!);
    }
  });

  it("changes with the shift", () => {
    // Otherwise a port that ignored `shift` entirely would pass every check
    // above that happens to use the default.
    const other = flowSchedule(golden.steps, 1.0, golden.numTrainTimesteps);
    expect(other.sigmas[1]).not.toBeCloseTo(got.sigmas[1]!, 4);
    // shift = 1 is the identity: sigma = 1*s/(1 + 0*s) = s.
    expect(other.sigmas[1]).toBeCloseTo(other.timesteps[1]! / golden.numTrainTimesteps, 6);
  });
});

describe("modelTimestep", () => {
  it("reverses the timestep the way the generator does", () => {
    // `(1000 - t)/1000`. The model then scales by 1000 again internally, so a
    // port that skipped either would still be handed a number in a plausible
    // range and denoise for the wrong amount of time.
    golden.timesteps.forEach((t, i) => {
      expect(modelTimestep(t, golden.numTrainTimesteps)).toBeCloseTo(golden.modelTimesteps[i]!, 6);
    });
  });

  it("maps the first timestep to zero and never to one", () => {
    expect(modelTimestep(1000, 1000)).toBe(0);
    expect(modelTimestep(0, 1000)).toBe(1);
  });
});

describe("eulerStep", () => {
  const sample = Float32Array.from(golden.step.sample);
  const modelOutput = Float32Array.from(golden.step.modelOutput);

  it("matches the model's step at every sigma", () => {
    golden.step.results.forEach((want, i) => {
      const got = eulerStep(sample, modelOutput, golden.sigmas, i);
      got.forEach((v, j) => expect(v).toBeCloseTo(want[j]!, 5));
    });
  });

  it("leaves the sample alone when the interval is empty", () => {
    const got = eulerStep(sample, modelOutput, [0.5, 0.5], 0);
    expect([...got]).toEqual([...sample]);
  });

  it("moves against the model output, because dt is negative", () => {
    // `dt = sigma_next - sigma` and sigma decreases, so the update subtracts.
    // A port that wrote `sigma - sigma_next` would denoise in reverse and
    // produce noise that looks like a plausible failure of the model.
    const got = eulerStep(Float32Array.from([0]), Float32Array.from([1]), [1, 0.5], 0);
    expect(got[0]).toBeCloseTo(-0.5, 6);
  });

  it("refuses a step index with no next sigma", () => {
    expect(() => eulerStep(sample, modelOutput, golden.sigmas, golden.sigmas.length - 1)).toThrow(/sigma/i);
  });
});
