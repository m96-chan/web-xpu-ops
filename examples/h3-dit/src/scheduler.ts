/**
 * MiniMax-H3's sampling schedule — rectified flow with an exponential shift.
 *
 * Issue #210. Ported from diffusers' `MiniMaxH3Scheduler` (rule 7: the
 * conventions here have several plausible spellings and upstream decides).
 * Four of them, each of which produces a video rather than an error when it is
 * got wrong:
 *
 * - **`t = 1 - sigma`, and `t = 1` is *clean*.** The opposite of the
 *   DDPM convention most schedulers here use, and the transformer is
 *   conditioned on `t`, not on sigma.
 * - **The terminal sigma has no model evaluation.** `num_inference_steps` grid
 *   points drive `num_inference_steps - 1` forwards.
 * - **`step` recovers its sigma from the *timestep*, not from the grid.**
 *   Upstream keeps the two apart on purpose: for `sigma < 0.5` the f32 round
 *   trip `1 - (1 - sigma)` is not exact.
 * - **Despite the reference class being named "euler ancestral", `eta` is 0.**
 *   No noise is ever re-injected.
 *
 * The video scheduler ships `shift = 12.0` and the audio one `shift = 3.0`.
 */

export interface Schedule {
  /** `num_inference_steps` grid points, strictly decreasing, ending at exactly 0. */
  sigmas: Float32Array;
  /** `1 - sigmas[:-1]` — one per model evaluation, so one shorter than `sigmas`. */
  timesteps: Float32Array;
}

/**
 * `linspace(1, 0, steps)` pushed through the exponential shift, duplicates collapsed.
 *
 * The shift maps 0 to exactly 0, so the terminal point is already on the grid.
 */
export function setTimesteps(numInferenceSteps: number, shift = 12.0): Schedule {
  if (!Number.isInteger(numInferenceSteps) || numInferenceSteps < 2) {
    throw new Error(`setTimesteps: needs at least 2 grid points, got ${numInferenceSteps}`);
  }
  const raw: number[] = [];
  // `torch.linspace(1, 0, n)` in f32, reproduced element for element.
  //
  // **Three details, all measured against torch rather than reasoned about,
  // because getting any of them wrong shifts sigmas by one f32 ulp and no
  // sampler would ever look wrong.** The grid decides which timesteps the
  // transformer is conditioned on, and `unique_consecutive` below decides how
  // many there are.
  //
  // 1. The step is rounded to **f32 first**, then used.
  // 2. The first half counts up from `start` and the second half counts **down
  //    from `end`** -- ATen splits at `steps / 2` so the terminal value is
  //    exactly `end` rather than an accumulated approximation of it.
  // 3. Each element is one **fused** multiply-add: the product is not rounded
  //    before the addition. `Math.fround(1 + step * i)` is that, since JS
  //    multiplies in f64.
  //
  // Naive `1 - i / (n - 1)` disagrees with torch at 4 of 50 points, and
  // rounding the product first disagrees at a different 4.
  const step = Math.fround(-1 / (numInferenceSteps - 1));
  const halfway = Math.floor(numInferenceSteps / 2);
  for (let i = 0; i < numInferenceSteps; i += 1) {
    const base = i < halfway
      ? Math.fround(1 + step * i)
      : Math.fround(-step * (numInferenceSteps - i - 1));
    // `shift * base`, `(shift - 1) * base` and the sum are three separate torch
    // ops on f32 tensors, so each rounds on its own -- not one expression.
    const numerator = Math.fround(shift * base);
    const denominator = Math.fround(1 + Math.fround((shift - 1) * base));
    raw.push(Math.fround(numerator / denominator));
  }
  // The shift compresses the grid near sigma = 1, and f32 collides there.
  // `torch.unique_consecutive`: only *adjacent* duplicates go.
  const sigmas: number[] = [];
  for (const s of raw) if (sigmas.length === 0 || sigmas[sigmas.length - 1] !== s) sigmas.push(s);

  const timesteps = Float32Array.from(sigmas.slice(0, -1), (s) => Math.fround(1 - s));
  return { sigmas: Float32Array.from(sigmas), timesteps };
}

/**
 * The forward process in H3's `t` convention: `x_t = t * x_0 + (1 - t) * noise`.
 *
 * `t = 1` returns the sample unchanged. Used to noise conditioning anchors,
 * where `t` is a `noise_aug` level rather than a schedule entry, so it is taken
 * at face value and **not** looked up in the schedule.
 */
export function scaleNoise(sample: Float32Array, timestep: number, noise: Float32Array): Float32Array {
  if (sample.length !== noise.length) {
    throw new Error(`scaleNoise: sample has ${sample.length} values and noise ${noise.length}`);
  }
  const out = new Float32Array(sample.length);
  for (let i = 0; i < sample.length; i += 1) out[i] = timestep * sample[i]! + (1 - timestep) * noise[i]!;
  return out;
}

/**
 * One Euler step, written as an `x_t` / `x_0` blend.
 *
 * `modelOutput` is the data-ward velocity the transformer returns at
 * `timestep`; `stepIndex` addresses the sigma grid. The two sources of sigma
 * are deliberately different — see the note at the top of this file.
 */
export function step(
  schedule: Schedule,
  modelOutput: Float32Array,
  timestep: number,
  sample: Float32Array,
  stepIndex: number,
): Float32Array {
  if (stepIndex < 0 || stepIndex + 1 >= schedule.sigmas.length) {
    throw new Error(`step: index ${stepIndex} has no next sigma in a grid of ${schedule.sigmas.length}`);
  }
  if (sample.length !== modelOutput.length) {
    throw new Error(`step: sample has ${sample.length} values and the model output ${modelOutput.length}`);
  }
  const sigmaFromTimestep = 1 - timestep;
  const ratio = schedule.sigmas[stepIndex + 1]! / schedule.sigmas[stepIndex]!;
  const out = new Float32Array(sample.length);
  for (let i = 0; i < sample.length; i += 1) {
    const denoised = sample[i]! + sigmaFromTimestep * modelOutput[i]!;
    out[i] = ratio * sample[i]! + (1 - ratio) * denoised;
  }
  return out;
}
