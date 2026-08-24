/**
 * The sampler against ComfyUI's own, from `fixtures/sampler-golden.json`.
 *
 * Issue #170 stage 5. This runs on the CPU with no weights and no GPU, so it is
 * part of `npm test` rather than a script — the schedule and the stepper are
 * the two halves of the pipeline that can be checked for free, and they are
 * also the two whose mistakes look like "the model is a bit worse" rather than
 * like a crash.
 */
import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import {
  LATENT,
  SAMPLING,
  applyCfg,
  betaPpf,
  betaSchedule,
  cfgEnabled,
  calculateDenoised,
  flowSigmas,
  latentToVae,
  timestepOf,
  noiseScaling,
  resMultistep,
  rint,
  vaeToLatent,
} from "./sampler.js";

interface Golden {
  shift: number;
  multiplier: number;
  modelSigmas: { count: number; first: number; last: number; every100: number[] };
  betaSchedules: Record<string, number[]>;
  trajectory: { sigmas: number[]; x0: number[]; steps: number[][]; final: number[] };
  noiseScaling: { sigma: number; noise: number[]; image: number[]; scaled: number[] }[];
}

const golden = JSON.parse(
  readFileSync(fileURLToPath(new URL("../fixtures/sampler-golden.json", import.meta.url)), "utf8"),
) as Golden;

/** The toy denoiser `gen_sampler_golden.py` sampled with, to the letter. */
function toyDenoise(x: Float32Array, sigma: number): Float32Array {
  const out = new Float32Array(x.length);
  for (let i = 0; i < x.length; i += 1) {
    out[i] = x[i]! * 0.85 - Math.sin(x[i]!) * sigma + 0.1 * sigma * sigma;
  }
  return out;
}

function maxAbs(got: ArrayLike<number>, want: ArrayLike<number>): number {
  expect(got.length).toBe(want.length);
  let worst = 0;
  for (let i = 0; i < want.length; i += 1) worst = Math.max(worst, Math.abs(got[i]! - want[i]!));
  return worst;
}

describe("the flow sigma table", () => {
  const sigmas = flowSigmas();

  it("has the length and the ends ComfyUI builds", () => {
    expect(sigmas.length).toBe(golden.modelSigmas.count);
    expect(sigmas[0]).toBeCloseTo(golden.modelSigmas.first, 9);
    expect(sigmas[sigmas.length - 1]).toBeCloseTo(golden.modelSigmas.last, 9);
  });

  it("matches through the middle, not only at the ends", () => {
    // A `time_snr_shift` with the shift applied to the wrong side of the
    // fraction still gives sigma(0)≈0 and sigma(1)=1. The interior is where the
    // two disagree, which is why the golden samples it.
    const sampled = golden.modelSigmas.every100.map((_, i) => sigmas[i * 100]!);
    expect(maxAbs(sampled, golden.modelSigmas.every100)).toBeLessThan(1e-6);
  });

  it("uses shift 3.0 and multiplier 1.0, not the class defaults", () => {
    expect(SAMPLING.shift).toBe(golden.shift);
    expect(SAMPLING.multiplier).toBe(golden.multiplier);
  });

  it("puts the multiplier where it is actually observable", () => {
    // Not in the table: `sigma(t) = time_snr_shift(shift, t / multiplier)` is
    // called with `t = (i / timesteps) * multiplier`, so it cancels and the
    // table is identical for 1 and for 1000. Asserted rather than assumed,
    // because the natural place to look for a wrong multiplier is here and
    // looking here would find nothing.
    expect(Array.from(flowSigmas(3.0, 1000))).toEqual(Array.from(flowSigmas(3.0, 1.0)));

    // It survives in the timestep the DiT is handed. Anima's 1.0 passes sigma
    // through; the class default would pass 1000x that.
    expect(timestepOf(0.7)).toBeCloseTo(0.7, 9);
    expect(timestepOf(0.7, 1000)).toBeCloseTo(700, 6);
  });
});

describe("the beta schedule", () => {
  const modelSigmas = flowSigmas();

  it.each(Object.keys(golden.betaSchedules))("matches at %s steps", (steps) => {
    const want = golden.betaSchedules[steps]!;
    const got = betaSchedule(modelSigmas, Number(steps));
    // The length is asserted separately: `beta_scheduler` drops consecutive
    // duplicate indices, so a port that returns `steps + 1` sigmas has already
    // diverged even if every value it did produce is right.
    expect(got.length).toBe(want.length);
    expect(maxAbs(got, want)).toBeLessThan(1e-6);
  });

  it("inverts the CDF, not something merely monotone", () => {
    // Beta(0.6, 0.6) is symmetric, so the median is exactly 0.5 — a property
    // no amount of curve-fitting to the endpoints reproduces by accident.
    expect(betaPpf(0.5)).toBeCloseTo(0.5, 9);
    expect(betaPpf(0.25) + betaPpf(0.75)).toBeCloseTo(1.0, 9);
  });

  it("drops indices that round to the same table entry", () => {
    // Not reachable at any step count the workflow uses: below 200 steps every
    // rounded index is distinct, so the branch that drops duplicates never
    // fires and a port missing it passes. 200 steps collapse to 199 sigmas and
    // 1000 to 848, both recorded in the golden above — the `it.each` over
    // `betaSchedules` is what actually fails, and this states why those two
    // step counts are in a fixture for a model nobody runs at 1000 steps.
    expect(betaSchedule(modelSigmas, 200).length).toBeLessThan(201);
    expect(betaSchedule(modelSigmas, 1000).length).toBeLessThan(1001);
    // And below that the collapse must *not* happen, or the port is dropping
    // sigmas it should keep.
    expect(betaSchedule(modelSigmas, 40).length).toBe(41);
  });

  it("rounds halves the way numpy does", () => {
    // Asserted directly rather than through a schedule. Anima's schedules only
    // ever produce 499.5, where round-half-to-even and round-half-up agree, so
    // nothing downstream can observe this — and a rule that is unobservable is
    // exactly the one that quietly rots.
    expect(rint(2.5)).toBe(2);
    expect(rint(3.5)).toBe(4);
    expect(rint(-2.5)).toBe(-2);
    expect(rint(499.5)).toBe(500);
    expect(rint(2.4)).toBe(2);
    expect(rint(2.6)).toBe(3);
  });

  it("ends at zero and starts at the top of the table", () => {
    const got = betaSchedule(modelSigmas, 40);
    expect(got[0]).toBeCloseTo(1.0, 9);
    expect(got[got.length - 1]).toBe(0);
  });
});

describe("res_multistep", () => {
  const { sigmas, x0, steps, final } = golden.trajectory;

  it("reproduces the whole trajectory, step by step", () => {
    const seen: number[][] = [];
    const out = resMultistep(toyDenoise, Float32Array.from(x0), sigmas, {
      onStep: (_i, _n, x) => seen.push(Array.from(x)),
    });
    // Step by step rather than only at the end: a second-order coefficient that
    // is wrong early can be walked back by later steps, and a final-latent-only
    // check would call that a pass.
    expect(seen.length).toBe(steps.length);
    for (let i = 0; i < steps.length; i += 1) {
      expect(maxAbs(seen[i]!, steps[i]!)).toBeLessThan(2e-6);
    }
    expect(maxAbs(out, final)).toBeLessThan(2e-6);
  });

  it("is not Euler", () => {
    // The guard for the mistake this file is really about. Forcing every step
    // down the first-order branch must break the golden — if it does not, the
    // second-order branch is never being reached and the test proves nothing.
    const euler = resMultistep(
      (x, sigma) => {
        const d = toyDenoise(x, sigma);
        return d;
      },
      Float32Array.from(x0),
      // Two sigmas: one step, which always takes the Euler branch.
      [sigmas[0]!, sigmas[sigmas.length - 1]!],
    );
    expect(maxAbs(euler, final)).toBeGreaterThan(1e-3);
  });

  it("refuses a schedule it cannot step", () => {
    expect(() => resMultistep(toyDenoise, new Float32Array(4), [1.0])).toThrow(/at least two sigmas/);
  });
});

describe("CFG", () => {
  it("interpolates from the unconditional, not from the conditional", () => {
    const cond = Float32Array.from([2, 4]);
    const uncond = Float32Array.from([1, 1]);
    // `uncond + 3 * (cond - uncond)` = [4, 10]. The other convention that keeps
    // turning up, `cond + 3 * (cond - uncond)`, gives [5, 13] — the same family
    // of answers one unit of scale further along, which is why it is invisible
    // except by comparing against the reference.
    expect(Array.from(applyCfg(cond, uncond, 3))).toEqual([4, 10]);
    expect(Array.from(applyCfg(cond, uncond, 1))).toEqual(Array.from(cond));
    expect(Array.from(applyCfg(cond, uncond, 0))).toEqual(Array.from(uncond));
  });

  it("commutes with the denoising, which is why it runs before it", () => {
    // ComfyUI applies CFG to the denoised predictions; this port applies it to
    // the raw model outputs. `x - v * sigma` is affine in `v`, so the two agree
    // — asserted rather than argued, because the port depends on it.
    const x = Float32Array.from([0.5, -0.25]);
    const cond = Float32Array.from([2, 4]);
    const uncond = Float32Array.from([1, 1]);
    const sigma = 0.4, scale = 3;
    const before = calculateDenoised(sigma, applyCfg(cond, uncond, scale), x);
    const after = applyCfg(
      calculateDenoised(sigma, cond, x),
      calculateDenoised(sigma, uncond, x),
      scale,
    );
    expect(maxAbs(before, after)).toBeLessThan(1e-6);
  });

  it("is off at or below 1.0", () => {
    expect(cfgEnabled(1.0)).toBe(false);
    expect(cfgEnabled(0)).toBe(false);
    expect(cfgEnabled(8)).toBe(true);
  });
});

describe("the latent conventions", () => {
  it.each(golden.noiseScaling.map((c, i) => [i, c] as const))(
    "builds latent %i as ComfyUI does",
    (_i, c) => {
      // The second case is the one that means anything. Text-to-image starts at
      // sigma 1.0 against a zero image, where `sigma * noise + (1 - sigma) *
      // image` reduces to `noise` and a port that dropped either coefficient
      // still passes. A sigma inside the schedule with a non-zero image is what
      // separates them.
      const got = noiseScaling(c.sigma, Float32Array.from(c.noise), Float32Array.from(c.image));
      expect(maxAbs(got, c.scaled)).toBeLessThan(1e-6);
    },
  );

  it("denoises with the flow convention, not eps", () => {
    // `x - v * sigma`. `EPS` would be `x - v * sigma` over a different
    // parameterisation and `X0` would return `v` untouched; the arithmetic is
    // asserted directly because there is no golden that distinguishes them
    // without a model.
    const x = Float32Array.from([1, 2, 3]);
    const v = Float32Array.from([0.5, 0.5, 0.5]);
    expect(maxAbs(calculateDenoised(0.4, v, x), [1 - 0.2, 2 - 0.2, 3 - 0.2])).toBeLessThan(1e-6);
  });

  it("scales the latent per channel, and round-trips", () => {
    const latent = Float32Array.from({ length: LATENT.channels * 4 }, (_, i) => Math.sin(i));
    const back = vaeToLatent(latentToVae(latent));
    expect(maxAbs(back, latent)).toBeLessThan(1e-5);
    // Per channel, not one global scale: channel 7's mean is 1.5508 and
    // channel 2's is -0.9113, so a global constant cannot satisfy both.
    const vae = latentToVae(new Float32Array(LATENT.channels * 4));
    expect(vae[7 * 4]).toBeCloseTo(LATENT.mean[7]!, 5);
    expect(vae[2 * 4]).toBeCloseTo(LATENT.mean[2]!, 5);
  });
});
