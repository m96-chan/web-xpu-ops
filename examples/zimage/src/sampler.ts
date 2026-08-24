/**
 * Z-Image's rectified-flow sampler.
 *
 * Six lines of arithmetic, which is why it is worth being careful with: a port
 * written from memory produces images, just not the right ones. Correctness is
 * `fixtures/sampler.golden.json`, baked from `zimage_utils` itself.
 *
 * The four places this differs from what one would guess:
 *
 *  - `linspace(1000, 1, steps + 1)` with the **last point dropped**, not
 *    `linspace` over `steps` points. Only the size of the first interval gives
 *    it away.
 *  - The shift is applied to sigma, not to the timestep, as
 *    `shift * s / (1 + (shift - 1) * s)`, and the generator's default is
 *    **3.0**. (`BASE_SHIFT`/`MAX_SHIFT` in `zimage_config` belong to a
 *    resolution-dependent path the image generator does not take.)
 *  - A final `sigma = 0` is appended, so the last step lands on a clean latent
 *    rather than stopping one interval short.
 *  - The model is called with the timestep **reversed**, `(1000 - t) / 1000`,
 *    and its prediction is **negated** before the step: Z-Image predicts
 *    negative noise (`zimage_generate_image.py:604`). Those two cancel in any
 *    test that compares a port against itself, so both are pinned against
 *    upstream.
 */

/** `zimage_config.DEFAULT_SCHEDULER_NUM_TRAIN_TIMESTEPS`. */
export const NUM_TRAIN_TIMESTEPS = 1000;

/** The image generator's `--flow_shift` default. */
export const DEFAULT_FLOW_SHIFT = 3.0;

/** `zimage_config.DEFAULT_INFERENCE_STEPS`. */
export const DEFAULT_STEPS = 8;

export interface FlowSchedule {
  /** `[steps]`, in `[0, numTrainTimesteps]`, descending. */
  timesteps: number[];
  /** `[steps + 1]`, in `[0, 1]`, descending and ending at exactly 0. */
  sigmas: number[];
}

/** The timesteps and sigmas for one run. */
export function flowSchedule(
  steps: number,
  shift = DEFAULT_FLOW_SHIFT,
  numTrainTimesteps = NUM_TRAIN_TIMESTEPS,
): FlowSchedule {
  if (steps < 1) throw new Error(`flowSchedule: steps must be at least 1, got ${steps}`);

  const sigmas: number[] = [];
  const timesteps: number[] = [];
  // `np.linspace(numTrainTimesteps, 1, steps + 1)[:-1]` — `steps + 1` points
  // spanning down to 1, with the last dropped.
  const span = numTrainTimesteps - 1;
  for (let i = 0; i < steps; i += 1) {
    const raw = (numTrainTimesteps - (span * i) / steps) / numTrainTimesteps;
    const sigma = (shift * raw) / (1 + (shift - 1) * raw);
    sigmas.push(sigma);
    timesteps.push(sigma * numTrainTimesteps);
  }
  sigmas.push(0);
  return { timesteps, sigmas };
}

/**
 * The timestep the model is actually called with.
 *
 * Reversed and normalised. The DiT scales it back up by `t_scale = 1000`
 * internally, so a port that skipped either half still hands the model a number
 * in a plausible range — and denoises for the wrong amount of time.
 */
export function modelTimestep(timestep: number, numTrainTimesteps = NUM_TRAIN_TIMESTEPS): number {
  return (numTrainTimesteps - timestep) / numTrainTimesteps;
}

/**
 * One Euler update: `x + (sigma_next - sigma) * v`.
 *
 * `v` is the **negated** model output — see the note at the top. `dt` is
 * negative because sigma decreases, so this subtracts; writing the difference
 * the other way round denoises in reverse and looks like the model failing.
 */
export function eulerStep(
  sample: Float32Array,
  modelOutput: Float32Array,
  sigmas: readonly number[],
  stepIndex: number,
): Float32Array {
  if (stepIndex + 1 >= sigmas.length) {
    throw new Error(`eulerStep: step ${stepIndex} has no next sigma (${sigmas.length} sigmas).`);
  }
  if (sample.length !== modelOutput.length) {
    throw new Error(`eulerStep: sample has ${sample.length} values and the model output ${modelOutput.length}.`);
  }
  const dt = sigmas[stepIndex + 1]! - sigmas[stepIndex]!;
  const out = new Float32Array(sample.length);
  for (let i = 0; i < sample.length; i += 1) out[i] = sample[i]! + dt * modelOutput[i]!;
  return out;
}

/**
 * Classifier-free guidance: `cond + scale * (cond - uncond)`.
 *
 * **Which model this is decides whether it is used at all**, and getting that
 * wrong is not subtle. `Tongyi-MAI/Z-Image` is the *Base* model — undistilled,
 * 28–50 steps, CFG 3.0–5.0 — while `Z-Image-Turbo` is the distilled one at 8
 * steps with no CFG. `zimage_config.py`'s `DEFAULT_INFERENCE_STEPS = 8` and
 * `DEFAULT_GUIDANCE_SCALE = 0.0` are Turbo's numbers; running Base on them
 * produces an image that has not converged and barely follows the prompt.
 *
 * Upstream's threshold is `guidance_scale > 1.0` (`zimage_generate_image.py:577`),
 * not `> 0`, so a scale of exactly 1 is "off" rather than "identity" — the same
 * arithmetic either way, but one fewer forward pass.
 */
export function applyCfg(conditional: Float32Array, unconditional: Float32Array, scale: number): Float32Array {
  if (conditional.length !== unconditional.length) {
    throw new Error(
      `applyCfg: conditional has ${conditional.length} values and unconditional ${unconditional.length}.`,
    );
  }
  const out = new Float32Array(conditional.length);
  for (let i = 0; i < out.length; i += 1) {
    out[i] = conditional[i]! + scale * (conditional[i]! - unconditional[i]!);
  }
  return out;
}

/** Upstream's own test for whether CFG runs at all. */
export function cfgEnabled(scale: number): boolean {
  return scale > 1.0;
}

/**
 * What each published checkpoint wants, from its own model card.
 *
 * Kept here rather than left to a caller's memory, because the two differ in
 * every number that matters and the failure mode of mixing them is an image
 * that looks like a bug in the port.
 */
export const VARIANTS = {
  base: { steps: 30, guidance: 4.0, note: "Tongyi-MAI/Z-Image — undistilled, 28-50 steps, CFG 3.0-5.0" },
  turbo: { steps: 8, guidance: 0.0, note: "Tongyi-MAI/Z-Image-Turbo — distilled, 8 steps, no CFG" },
} as const;

export type Variant = keyof typeof VARIANTS;

/**
 * The latent the VAE decoder wants: `latent / scalingFactor + shiftFactor`.
 *
 * `zimage_utils.shift_scale_latents_for_decode`. The two constants are the
 * VAE's, not the sampler's, and getting them backwards produces an image with
 * the right structure and the wrong colours.
 */
export function scaleLatentsForDecode(
  latents: Float32Array,
  scalingFactor: number,
  shiftFactor: number,
): Float32Array {
  const out = new Float32Array(latents.length);
  for (let i = 0; i < latents.length; i += 1) out[i] = latents[i]! / scalingFactor + shiftFactor;
  return out;
}
