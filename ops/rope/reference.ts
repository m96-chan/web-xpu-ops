/**
 * Rotary position embedding, with optional NTK and YaRN context scaling.
 *
 * Each adjacent pair within a head is rotated by an angle that depends on the
 * token's position and on how far into the head dimension the pair sits.
 *
 * `posOffset` exists for KV-cache continuation: during decoding the tensor
 * holds one token but its true position is however many came before.
 *
 * Note for anyone comparing a backend against this: GPU `sin` and `cos` are
 * markedly less accurate than the f64 ones here — measured at up to 1.86e-4 of
 * absolute error against 1.2e-7 for f32 epsilon — and this op calls both per
 * element. Tolerances have to reflect the hardware, not the arithmetic.
 *
 * ## Scaling
 *
 * NTK and YaRN both let a model read further than it was trained to. Neither
 * changes the rotation: they change **which frequency each pair rotates at**,
 * and YaRN additionally scales the magnitude. So they are parameters of this
 * op, not ops of their own.
 *
 * There is no PyTorch definition to follow here — `torch` ships no RoPE — so
 * rule 7's fallback is the de-facto reference pair, which agree with each
 * other and with the paper:
 *
 *   - the YaRN authors' own code, `jquesnelle/yarn`
 *     (`scaled_rope/modeling_llama_yarn.py`, `scaled_rope/LlamaNTKScaledRotaryEmbedding.py`)
 *   - `transformers/modeling_rope_utils.py`
 *     (`_compute_yarn_parameters`, `_compute_dynamic_ntk_parameters`)
 *
 * Both are transcribed below with the papers' symbols named in the comments.
 * Where the two differ, the divergence is called out and a side is chosen.
 */

/** Ⓐ NTK-aware interpolation (YaRN paper §3.1) and Ⓑ YaRN (§3.4). */
export type RoPEScaling =
  | {
      kind: "ntk";
      /**
       * The paper's `s` (`alpha` in `LlamaNTKScaledRotaryEmbedding`): how much
       * further than the trained context you intend to read.
       *
       * `transformers` also ships a *dynamic* NTK that recomputes `s` from the
       * running sequence length each step. That is a scheduling decision made
       * once per forward pass, above a kernel; feed the result in here.
       */
      factor: number;
    }
  | {
      kind: "yarn";
      /** The paper's `s` — target context divided by trained context. */
      factor: number;
      /** The paper's `L`: the context the model was trained on. */
      originalContextLength: number;
      /** The paper's `β_fast`, 32 in the paper and in both implementations. */
      betaFast?: number;
      /** The paper's `β_slow`, 1 in the paper and in both implementations. */
      betaSlow?: number;
      /**
       * The paper's `√(1/t)` — attention temperature, `mscale` in the code.
       * Defaults to `0.1·ln(s) + 1`, which is what the paper prescribes.
       * Override only to reproduce a checkpoint that was fine-tuned with a
       * different one (DeepSeek-V3 is the usual case).
       */
      attentionFactor?: number;
    };

export interface RoPEArgs {
  input: Float32Array;
  /** Tokens. */
  N: number;
  numHeads: number;
  headDim: number;
  /** Position of the first token, for KV-cache continuation. */
  posOffset: number;
  /** 10000 conventionally; some models use 500000. */
  thetaBase: number;
  /** Omitted means plain RoPE, unchanged in every bit. */
  scaling?: RoPEScaling;
}

/**
 * The whole of NTK and YaRN, reduced to five numbers that do not depend on
 * position or on the tensor.
 *
 * Everything expensive and fiddly about the two schemes — a log, a floor, a
 * ceiling, two clamps, a singularity guard — is a function of the head
 * geometry alone. Computing it here rather than per element is what lets one
 * unbranched kernel serve all three schemes; see `wgsl/kernel.wgsl`.
 */
export interface RoPEFrequencyParams {
  /** The base actually raised to `-2i/D`. NTK is nothing but a change here. */
  effectiveBase: number;
  /** YaRN's `s`, dividing the interpolated frequencies. `1` for the others. */
  interpolationFactor: number;
  /** Pair index where YaRN's ramp leaves pure extrapolation. */
  rampLow: number;
  /** Pair index where YaRN's ramp reaches pure interpolation. */
  rampHigh: number;
  /** YaRN's `√(1/t)`, multiplying `cos` and `sin`. `1` for the others. */
  attentionFactor: number;
}

/**
 * Plain RoPE, stated as the degenerate case of the scaled one.
 *
 * `interpolationFactor: 1` makes `interpolation - extrapolation` exactly zero
 * in IEEE arithmetic, so the ramp multiplies a true zero and the frequency is
 * the unscaled one bit for bit — not merely to within a tolerance. The ramp
 * bounds are then arbitrary; `0`/`1` keeps the division finite.
 */
const UNSCALED: Omit<RoPEFrequencyParams, "effectiveBase"> = {
  interpolationFactor: 1,
  rampLow: 0,
  rampHigh: 1,
  attentionFactor: 1,
};

export function ropeFrequencyParams(
  headDim: number,
  thetaBase: number,
  scaling?: RoPEScaling,
): RoPEFrequencyParams {
  if (!scaling) return { ...UNSCALED, effectiveBase: thetaBase };

  if (scaling.kind === "ntk") {
    // YaRN paper §3.1: b' = b · s^(D/(D-2)). Verbatim from
    // `LlamaNTKScaledRotaryEmbedding`: `base = base * alpha ** (dim/(dim-2))`.
    //
    // The exponent is D/(D-2) rather than 1 so that the *lowest* frequency
    // stretches by exactly s while the highest barely moves — the whole point
    // of NTK over plain positional interpolation.
    return {
      ...UNSCALED,
      effectiveBase: thetaBase * Math.pow(scaling.factor, headDim / (headDim - 2)),
    };
  }

  const { factor, originalContextLength, betaFast = 32, betaSlow = 1 } = scaling;

  /**
   * The paper's inverse of "how many rotations does pair i complete over L":
   * i = D·ln(L / (2π·r)) / (2·ln b). Identical in both implementations.
   */
  const correctionDim = (rotations: number) =>
    (headDim * Math.log(originalContextLength / (rotations * 2 * Math.PI))) /
    (2 * Math.log(thetaBase));

  // Both implementations floor the low end and ceil the high end before
  // clamping. `transformers` makes that optional (`truncate`) and defaults it
  // on, so on-by-default is the behaviour the two agree about.
  //
  // The clamp to `headDim - 1` is looser than it looks: `correctionDim`
  // returns a *pair* index, of which there are only D/2. Kept anyway, because
  // matching the reference implementations matters more than tightening a
  // bound that never binds for real geometries.
  const rampLow = Math.max(Math.floor(correctionDim(betaFast)), 0);
  let rampHigh = Math.min(Math.ceil(correctionDim(betaSlow)), headDim - 1);

  // Both implementations nudge here rather than divide by zero. Only the
  // clamps can produce it: β_fast > β_slow makes the raw bounds strictly
  // ordered, and floor/ceil cannot close a strict gap. See `reference.test.ts`
  // for the degenerate context that does reach it.
  if (rampHigh === rampLow) rampHigh += 0.001;

  return {
    effectiveBase: thetaBase,
    interpolationFactor: factor,
    rampLow,
    rampHigh,
    // §3.4: √(1/t) = 0.1·ln(s) + 1, and 1 when there is nothing to extend.
    attentionFactor:
      scaling.attentionFactor ?? (factor <= 1 ? 1 : 0.1 * Math.log(factor) + 1),
  };
}

export function rope({
  input,
  N,
  numHeads,
  headDim,
  posOffset,
  thetaBase,
  scaling,
}: RoPEArgs): Float32Array {
  const output = new Float32Array(input.length);
  const halfDim = headDim / 2;
  const { effectiveBase, interpolationFactor, rampLow, rampHigh, attentionFactor } =
    ropeFrequencyParams(headDim, thetaBase, scaling);

  for (let token = 0; token < N; token += 1) {
    for (let head = 0; head < numHeads; head += 1) {
      for (let pair = 0; pair < halfDim; pair += 1) {
        // The reference implementations write these as `1/pos_freqs` and
        // `1/(s·pos_freqs)` with `pos_freqs = b^(2i/D)`. Written as a negative
        // exponent instead, so that the no-scaling path is the same expression
        // this file has always evaluated, down to the last bit.
        const extrapolation = Math.pow(effectiveBase, (-2 * pair) / headDim);
        const interpolation = extrapolation / interpolationFactor;

        // γ(i) in the paper: 0 below `rampLow` (extrapolate — these pairs turn
        // fast enough to have been seen at every phase during training), 1
        // above `rampHigh` (interpolate — these have not completed a rotation
        // even once), linear between.
        const ramp = Math.min(Math.max((pair - rampLow) / (rampHigh - rampLow), 0), 1);
        const invFreq = extrapolation + (interpolation - extrapolation) * ramp;

        const theta = (token + posOffset) * invFreq;
        // The attention temperature is folded into cos/sin, exactly as both
        // implementations do (`emb.cos() * self.mscale`). Scaling q and k
        // before the dot product would be the same thing one step later.
        const cos = Math.cos(theta) * attentionFactor;
        const sin = Math.sin(theta) * attentionFactor;

        const base = (token * numHeads + head) * headDim + pair * 2;
        const x0 = input[base]!;
        const x1 = input[base + 1]!;
        output[base] = x0 * cos - x1 * sin;
        output[base + 1] = x0 * sin + x1 * cos;
      }
    }
  }
  return output;
}
