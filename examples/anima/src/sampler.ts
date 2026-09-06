/**
 * Anima's sampler: the schedule, the stepper, and the two latent conventions
 * around them.
 *
 * Nothing here is `examples/zimage`'s sampler with different numbers. Z-Image
 * takes Euler steps down a linear rectified-flow schedule; Anima's released
 * workflow asks for `res_multistep` down a `beta` schedule, and the settings
 * come from three separate places, none of them a default:
 *
 * | | where |
 * | --- | --- |
 * | `shift` 3.0, `multiplier` 1.0 | `supported_models.py:1136` |
 * | `beta` schedule, 40 steps, CFG 8 | the released `workflow_2_pass.json` |
 * | `res_multistep`, `eta = 0` | `sample_res_multistep`, `sampling.py:1459` |
 *
 * **`multiplier` is 1.0, not 1000.** `ModelSamplingDiscreteFlow` defaults it to
 * 1000 and Z-Image's port multiplies its timestep by 1000 for that reason;
 * Anima's model reads sigma *as* the timestep. Carrying the habit across gives
 * a plausible image conditioned on the wrong point of the trajectory — the
 * shape of mistake that cost a whole debugging session on Z-Image, where Turbo
 * step counts on the Base model produced images that looked like a broken VAE.
 *
 * Every constant and both formulas are pinned by `fixtures/sampler-golden.json`,
 * baked from ComfyUI's own code by `tools/gen_sampler_golden.py`.
 */

/** `supported_models.Anima.sampling_settings`. */
export const SAMPLING = { shift: 3.0, multiplier: 1.0, timesteps: 1000 } as const;

/** The released workflow's first pass — `['beta', 40, 1]` with CFG 8. */
export const DEFAULTS = { steps: 40, guidance: 8.0, width: 832, height: 1216 } as const;

/**
 * `latent_formats.Wan21` — 16 channels with a per-channel mean and standard
 * deviation, not Z-Image's single scale and shift. `scale_factor` is 1.0 here,
 * so it is left out of the arithmetic rather than multiplied by one.
 */
export const LATENT = {
  channels: 16,
  mean: Float32Array.from([
    -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
    0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921,
  ]),
  std: Float32Array.from([
    2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
    3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160,
  ]),
  /**
   * `Wan21.latent_rgb_factors`, `[16, 3]` row-major, with its bias.
   *
   * The linear 16-to-3 projection ComfyUI shows as a live preview while
   * sampling. It is **not** a decoder: one eighth of the resolution and a
   * matrix multiply, against a VAE with 194 tensors. Useful for seeing that a
   * latent has structure at all, and for nothing that needs to look right.
   */
  rgbFactors: Float32Array.from([
    -0.1299, -0.1692, 0.2932,
    0.0671, 0.0406, 0.0442,
    0.3568, 0.2548, 0.1747,
    0.0372, 0.2344, 0.142,
    0.0313, 0.0189, -0.0328,
    0.0296, -0.0956, -0.0665,
    -0.3477, -0.4059, -0.2925,
    0.0166, 0.1902, 0.1975,
    -0.0412, 0.0267, -0.1364,
    -0.1293, 0.074, 0.1636,
    0.068, 0.3019, 0.1128,
    0.0032, 0.0581, 0.0639,
    -0.1251, 0.0927, 0.1699,
    0.006, -0.0633, 0.0005,
    0.3477, 0.2275, 0.295,
    0.1984, 0.0913, 0.1861,
  ]),
  rgbBias: Float32Array.from([-0.1835, -0.0868, -0.336]),
} as const;

/** `time_snr_shift` (`model_sampling.py:279`). */
function timeSnrShift(alpha: number, t: number): number {
  return alpha === 1.0 ? t : (alpha * t) / (1 + (alpha - 1) * t);
}

/**
 * `ModelSamplingDiscreteFlow.set_parameters`' table: `timesteps` entries, from
 * `sigma(1/timesteps * multiplier)` up to `sigma(multiplier)`.
 *
 * The schedule below indexes this table rather than evaluating a formula, so
 * the table's length and its rounding are part of the answer.
 */
export function flowSigmas(
  shift = SAMPLING.shift,
  multiplier: number = SAMPLING.multiplier,
  timesteps: number = SAMPLING.timesteps,
): Float64Array {
  const out = new Float64Array(timesteps);
  for (let i = 1; i <= timesteps; i += 1) {
    // `sigma(timestep) = time_snr_shift(shift, timestep / multiplier)`, called
    // with `timestep = (i / timesteps) * multiplier`. **The multiplier cancels
    // here** — the table is the same whether it is 1 or 1000, so no test of
    // this function can catch a wrong one. `timestepOf` is where it survives,
    // and that is where it is pinned.
    out[i - 1] = timeSnrShift(shift, i / timesteps);
  }
  void multiplier;
  return out;
}

/**
 * `ModelSamplingDiscreteFlow.timestep` — `sigma * multiplier`, and the one
 * place the multiplier is observable.
 *
 * With Anima's 1.0 the DiT is handed sigma itself. With the class default of
 * 1000 — which Z-Image uses, and which a port copied across from it would
 * inherit — it would be handed a number a thousand times larger, and the
 * timestep embedding would be evaluated somewhere else entirely. Nothing
 * crashes; the image is just conditioned on the wrong point of the trajectory.
 */
export function timestepOf(sigma: number, multiplier: number = SAMPLING.multiplier): number {
  return sigma * multiplier;
}

/**
 * `ln B(a, b)` by Lanczos, the standard approximation. Only ever called with
 * a = b = 0.6 here, but written generally because a special-cased constant
 * would hide which function this is.
 */
function lnGamma(z: number): number {
  const C = [
    76.18009172947146, -86.50532032941677, 24.01409824083091,
    -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5,
  ];
  let y = z;
  const x = z;
  let tmp = x + 5.5;
  tmp -= (x + 0.5) * Math.log(tmp);
  let ser = 1.000000000190015;
  for (const c of C) ser += c / ++y;
  return -tmp + Math.log((2.5066282746310005 * ser) / x);
}

/**
 * The regularized incomplete beta `I_x(a, b)`, by the continued fraction in
 * *Numerical Recipes* §6.4 with Lentz's algorithm.
 *
 * This is what `scipy.stats.beta.cdf` computes, and inverting it is what
 * `beta_scheduler` needs. Writing it out is unavoidable: the schedule is not a
 * closed form, and approximating it with something smoother would move every
 * sigma a little, which is invisible until the images are subtly worse.
 */
function incompleteBeta(x: number, a: number, b: number): number {
  if (x <= 0) return 0;
  if (x >= 1) return 1;
  const front = Math.exp(lnGamma(a + b) - lnGamma(a) - lnGamma(b) + a * Math.log(x) + b * Math.log(1 - x));
  // The fraction converges quickly only on one side of the mean; the identity
  // `I_x(a,b) = 1 - I_{1-x}(b,a)` moves the other side onto the fast one.
  if (x > (a + 1) / (a + b + 2)) return 1 - incompleteBeta(1 - x, b, a);

  const TINY = 1e-30;
  let f = 1, c = 1, d = 0;
  for (let i = 0; i <= 300; i += 1) {
    const m = Math.floor(i / 2);
    let numerator: number;
    if (i === 0) numerator = 1;
    else if (i % 2 === 0) numerator = (m * (b - m) * x) / ((a + 2 * m - 1) * (a + 2 * m));
    else numerator = -(((a + m) * (a + b + m) * x) / ((a + 2 * m) * (a + 2 * m + 1)));

    d = 1 + numerator * d;
    if (Math.abs(d) < TINY) d = TINY;
    d = 1 / d;
    c = 1 + numerator / c;
    if (Math.abs(c) < TINY) c = TINY;
    const delta = c * d;
    f *= delta;
    if (Math.abs(1 - delta) < 1e-14) return (front * (f - 1)) / a;
  }
  throw new Error(`incompleteBeta: did not converge at x=${x}, a=${a}, b=${b}`);
}

/**
 * `scipy.stats.beta.ppf` — the inverse CDF, by bisection.
 *
 * Bisection rather than Newton because `Beta(0.6, 0.6)` has infinite density at
 * both ends, where a Newton step overshoots out of `[0, 1]`. 200 halvings take
 * the bracket below 1e-60; the result is then multiplied by 999 and rounded, so
 * anything under about 1e-4 is already exact for this purpose.
 */
export function betaPpf(p: number, a = 0.6, b = 0.6): number {
  if (p <= 0) return 0;
  if (p >= 1) return 1;
  let low = 0, high = 1;
  for (let i = 0; i < 200; i += 1) {
    const mid = (low + high) / 2;
    if (incompleteBeta(mid, a, b) < p) low = mid;
    else high = mid;
  }
  return (low + high) / 2;
}

/**
 * `beta_scheduler` (`samplers.py:696`), which is not "sigmas from a formula".
 *
 * It walks `steps` points of the *inverse* Beta CDF, rounds each onto the
 * 1000-entry table, **drops consecutive duplicates**, and appends a zero. So
 * asking for 40 steps can return fewer than 41 sigmas — a port that assumes
 * `steps + 1` and pre-sizes an array is wrong on exactly the schedules where
 * two rounded indices collide.
 */
export function betaSchedule(
  modelSigmas: Float64Array,
  steps: number,
  alpha = 0.6,
  beta = 0.6,
): Float64Array {
  const total = modelSigmas.length - 1;
  const sigmas: number[] = [];
  let lastT = -1;
  for (let i = 0; i < steps; i += 1) {
    // `1 - numpy.linspace(0, 1, steps, endpoint=False)` — `endpoint=False`
    // makes the spacing `1/steps`, so the first point is 1.0 and the last is
    // `1/steps`, never 0.
    const t = 1 - i / steps;
    // `numpy.rint` is round-half-to-**even**, not JavaScript's round-half-up.
    //
    // Measured, because the difference is easy to assume and turns out not to
    // bite here: over 40, 200 and 1000 steps the only exact half `beta.ppf(t) *
    // 999` produces is 499.5, whose floor is odd, and both conventions round it
    // up to 500. So no schedule this model uses can tell them apart. `rint` is
    // still numpy's, because matching the reference where it is currently
    // unobservable is cheaper than discovering later that it became observable.
    const index = rint(betaPpf(t, alpha, beta) * total);
    if (index !== lastT) sigmas.push(modelSigmas[index]!);
    lastT = index;
  }
  sigmas.push(0.0);
  return Float64Array.from(sigmas);
}

/** `numpy.rint`: halves go to the even neighbour. */
export function rint(x: number): number {
  const floor = Math.floor(x);
  const frac = x - floor;
  if (frac > 0.5) return floor + 1;
  if (frac < 0.5) return floor;
  return floor % 2 === 0 ? floor : floor + 1;
}

/**
 * `CONST.noise_scaling` — how the first latent is built.
 *
 * `sigma * noise + (1 - sigma) * image`, with the image all zeros for
 * text-to-image, which makes it `sigma * noise` and `sigma` is 1.0 at the top
 * of a `beta` schedule. Written in full anyway: the day someone adds
 * image-to-image, the term that was silently zero has to already be here.
 */
export function noiseScaling(
  sigma: number,
  noise: Float32Array,
  latentImage?: Float32Array | null,
): Float32Array {
  const out = new Float32Array(noise.length);
  for (let i = 0; i < noise.length; i += 1) {
    out[i] = sigma * noise[i]! + (1 - sigma) * (latentImage ? latentImage[i]! : 0);
  }
  return out;
}

/**
 * `CONST.calculate_denoised` — `x - v * sigma`, where `v` is what the DiT
 * returns. This is the flow-matching convention; `EPS` and `V_PREDICTION`
 * differ, and `model_base.py:120` is where Anima is pinned to this one.
 */
export function calculateDenoised(sigma: number, modelOutput: Float32Array, modelInput: Float32Array): Float32Array {
  const out = new Float32Array(modelOutput.length);
  for (let i = 0; i < out.length; i += 1) out[i] = modelInput[i]! - modelOutput[i]! * sigma;
  return out;
}

/**
 * `uncond + scale * (cond - uncond)` — `cfg_function` (`samplers.py:598`).
 *
 * **Not** `cond + scale * (cond - uncond)`, which `examples/zimage`'s port uses
 * for a model whose own repository defines it that way. The two differ by
 * exactly one unit of scale — `cond + s*d == uncond + (s+1)*d` — so the
 * workflow's CFG 8 would silently be run at 9. Nothing looks broken; the images
 * are just over-guided.
 *
 * Order against `calculateDenoised` does not matter and the port relies on it:
 * ComfyUI applies CFG to the *denoised* predictions, this applies it to the raw
 * model outputs, and since `x - v * sigma` is affine in `v` the two agree
 * exactly. Applying it to the raw outputs halves what has to be kept per step.
 */
export function applyCfg(cond: Float32Array, uncond: Float32Array, scale: number): Float32Array {
  const out = new Float32Array(cond.length);
  for (let i = 0; i < out.length; i += 1) out[i] = uncond[i]! + scale * (cond[i]! - uncond[i]!);
  return out;
}

/** Below this the workflow's CFG is off and the unconditional pass is skipped. */
export function cfgEnabled(scale: number): boolean {
  return scale > 1.0;
}

export interface ResMultistepOptions {
  /** Called with the latent at the start of each step, as ComfyUI's callback is. */
  onStep?: (index: number, total: number, x: Float32Array, sigma: number) => void;
}

/**
 * `sample_res_multistep` (`sampling.py:1459`) — `res_multistep` with `eta = 0`.
 *
 * `eta = 0` collapses `get_ancestral_step` to `(sigma_to, 0)`, so `sigma_down`
 * is just the next sigma and no noise is ever added. What is left is a
 * second-order exponential multistep (arXiv 2308.02157) that falls back to
 * Euler on the first step and on the last, where `sigma_down` reaches zero.
 *
 * `old_sigma_down` is therefore `sigmas[i]` when step `i` reads it, which makes
 * `t_old === t` and `c2 = (t_prev - t) / h`. The port keeps the variable rather
 * than folding that in — the day someone wants the ancestral variant, `eta` is
 * the only thing that has to change.
 *
 * `denoise(x, sigma)` must return the **denoised** latent, not the model's raw
 * output: that is `calculateDenoised`'s job, and CFG happens before it.
 */
export function resMultistep(
  denoise: (x: Float32Array, sigma: number, index: number) => Float32Array,
  x0: Float32Array,
  sigmas: Float64Array | number[],
  options: ResMultistepOptions = {},
): Float32Array {
  const sigmaList = Array.from(sigmas);
  const steps = sigmaList.length - 1;
  if (steps < 1) throw new Error(`resMultistep: need at least two sigmas, got ${sigmaList.length}`);

  let x = Float32Array.from(x0);
  let oldDenoised: Float32Array | null = null;
  let oldSigmaDown = 0;

  const tFn = (sigma: number): number => -Math.log(sigma);
  // `phi1(t) = expm1(t)/t` and `phi2(t) = (phi1(t) - 1)/t`. Both are removable
  // singularities at 0, and both are only ever called with `-h != 0` because a
  // schedule with two equal sigmas cannot occur — `beta_scheduler` drops
  // consecutive duplicates for exactly this reason.
  const expm1 = (t: number): number => (Math.abs(t) < 1e-8 ? t + (t * t) / 2 : Math.exp(t) - 1);
  const phi1 = (t: number): number => expm1(t) / t;
  const phi2 = (t: number): number => (phi1(t) - 1.0) / t;

  for (let i = 0; i < steps; i += 1) {
    const sigma = sigmaList[i]!;
    const sigmaDown = sigmaList[i + 1]!;
    options.onStep?.(i, steps, x, sigma);

    const denoised = denoise(x, sigma, i);

    if (sigmaDown === 0 || oldDenoised === null) {
      // Euler. `to_d(x, sigma, denoised) = (x - denoised) / sigma`.
      const dt = sigmaDown - sigma;
      const next = new Float32Array(x.length);
      for (let j = 0; j < x.length; j += 1) next[j] = x[j]! + ((x[j]! - denoised[j]!) / sigma) * dt;
      x = next;
    } else {
      const t = tFn(sigma);
      const tOld = tFn(oldSigmaDown);
      const tNext = tFn(sigmaDown);
      const tPrev = tFn(sigmaList[i - 1]!);
      const h = tNext - t;
      const c2 = (tPrev - tOld) / h;

      const phi1Val = phi1(-h);
      const phi2Val = phi2(-h);
      // `torch.nan_to_num(..., nan=0.0)`: `c2` is zero when two sigmas repeat,
      // which the schedule prevents — kept so a hand-built schedule that does
      // repeat degrades the way ComfyUI's does instead of poisoning the latent.
      const b1 = Number.isFinite(phi1Val - phi2Val / c2) ? phi1Val - phi2Val / c2 : 0;
      const b2 = Number.isFinite(phi2Val / c2) ? phi2Val / c2 : 0;

      // `sigma_fn(h) = h.neg().exp()`.
      const decay = Math.exp(-h);
      const next = new Float32Array(x.length);
      for (let j = 0; j < x.length; j += 1) {
        next[j] = decay * x[j]! + h * (b1 * denoised[j]! + b2 * oldDenoised[j]!);
      }
      x = next;
    }

    oldDenoised = denoised;
    oldSigmaDown = sigmaDown;
  }
  return x;
}

/** `Wan21.process_out` — latent to VAE input, per channel. */
export function latentToVae(latent: Float32Array, channels = LATENT.channels): Float32Array {
  const perChannel = latent.length / channels;
  const out = new Float32Array(latent.length);
  for (let c = 0; c < channels; c += 1) {
    for (let i = 0; i < perChannel; i += 1) {
      out[c * perChannel + i] = latent[c * perChannel + i]! * LATENT.std[c]! + LATENT.mean[c]!;
    }
  }
  return out;
}

/** `Wan21.process_in` — the inverse, for image-to-image. */
export function vaeToLatent(latent: Float32Array, channels = LATENT.channels): Float32Array {
  const perChannel = latent.length / channels;
  const out = new Float32Array(latent.length);
  for (let c = 0; c < channels; c += 1) {
    for (let i = 0; i < perChannel; i += 1) {
      out[c * perChannel + i] = (latent[c * perChannel + i]! - LATENT.mean[c]!) / LATENT.std[c]!;
    }
  }
  return out;
}
