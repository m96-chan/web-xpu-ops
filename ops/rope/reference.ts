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
 * ## The head range, and which axis it is on
 *
 * `headOffset` / `headCount` rotate a contiguous run of heads and copy the rest
 * through unchanged. **The range is over heads — the `H` of `[N, H, Dh]` — and
 * not over the channels within a head.** Both are called "half-RoPE" in the
 * wild and they are different ops:
 *
 *   - **this one**, `x.chunk(2, dim=-2)` in Irodori-TTS's `_apply_rotary_half`:
 *     some heads see position and the others cannot see order at all, which is
 *     what a block attending jointly over positioned latents and unpositioned
 *     conditioning wants.
 *   - **partial rotary** (`rotaryDim`, GPT-NeoX and Phi): *every* head rotates,
 *     but only its first `rotaryDim` channels. Not implemented here, and
 *     deliberately not: nothing in front of this library needs it, and a
 *     parameter added on speculation is one every kernel variant has to honour.
 *
 * Heads outside the range are **copied**, not left at zero. `rope` returns a
 * fresh array, so an uninitialised passthrough head is silent garbage flowing
 * into attention rather than an error.
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
  /**
   * First head to rotate. Defaults to 0.
   *
   * Over the **head** axis, not over channels — see the note at the top of this
   * file, which is there because "half-RoPE" means both things in the wild.
   */
  headOffset?: number;
  /**
   * How many heads to rotate, starting at `headOffset`. Defaults to all of
   * them, which is the behaviour this op had before the range existed.
   *
   * Heads outside `[headOffset, headOffset + headCount)` are copied unchanged.
   */
  headCount?: number;
  /**
   * Precomputed angles from `ropeCache`. Omitted means every angle is computed
   * where it is used, which is what this op did before caching existed.
   *
   * A cache that was not built from this call's `headDim`, `thetaBase` and
   * `scaling` is rejected rather than used — see `ropeCache`.
   */
  cache?: RoPECache;
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

/**
 * The frequency pair `i` rotates at, from the five scalars.
 *
 * Shared by `rope` and `ropeCache` deliberately. A table built by a second copy
 * of this expression would agree with the fallback beside it only for as long
 * as nobody edited one of the two.
 */
function invFreq(
  { effectiveBase, interpolationFactor, rampLow, rampHigh }: RoPEFrequencyParams,
  headDim: number,
  pair: number,
): number {
  // The reference implementations write these as `1/pos_freqs` and
  // `1/(s·pos_freqs)` with `pos_freqs = b^(2i/D)`. Written as a negative
  // exponent instead, so that the no-scaling path is the same expression this
  // file has always evaluated, down to the last bit.
  const extrapolation = Math.pow(effectiveBase, (-2 * pair) / headDim);
  const interpolation = extrapolation / interpolationFactor;

  // γ(i) in the paper: 0 below `rampLow` (extrapolate — these pairs turn fast
  // enough to have been seen at every phase during training), 1 above
  // `rampHigh` (interpolate — these have not completed a rotation even once),
  // linear between.
  const ramp = Math.min(Math.max((pair - rampLow) / (rampHigh - rampLow), 0), 1);
  return extrapolation + (interpolation - extrapolation) * ramp;
}

export interface RoPECacheArgs {
  headDim: number;
  /** 10000 conventionally; some models use 500000. */
  thetaBase: number;
  /** How many positions to tabulate. Valid for `0 <= pos < positions`. */
  positions: number;
  /** Omitted means plain RoPE. */
  scaling?: RoPEScaling;
}

/**
 * Precomputed `cos`/`sin` for every (position, pair), so that decoding does not
 * recompute them.
 *
 * The angles depend on position and pair only — not on the head, not on the
 * tensor — so one table serves every head of every step. That is where the
 * saving comes from: an uncached pass spends `N · numHeads · headDim/2` of each
 * of `pow`, `sin` and `cos`, and the table costs `positions · headDim/2` of
 * each, once. Measured in `reference.test.ts` and on the GPU; see the PR.
 *
 * ## Past the end of the table
 *
 * A table covers `positions` positions and decoding runs past it. Three things
 * are possible and only one of them is correct:
 *
 *   - **grow** it — impossible from inside a dispatch; there is no host there
 *   - **wrap** it — `pos % positions` is silently the wrong angle, which comes
 *     back as a plausible-looking tensor rather than as an error
 *   - **fall back** to computing the angle where it is used
 *
 * So this op falls back. Past the end it is exactly the uncached op — the same
 * expression, at the same cost, giving the same answer — and the only thing a
 * short table costs is the saving. `rope` and `wgsl/kernel.wgsl` both do this,
 * and both are tested at the boundary with a deliberately wrong table, which is
 * the only way to tell "read the table" from "recomputed anyway" apart.
 *
 * ## A table built for other parameters
 *
 * The same failure by another route: a table built at one `thetaBase` or one
 * `scaling` and handed to a call using another holds angles that are wrong in
 * the same quiet way. `rope` refuses it. The five scalars the table was built
 * from travel with it in `freq` for exactly that comparison — two schemes that
 * agree in all five are the same rotation, so this is not conservative, it is
 * exact. The kernel cannot check anything of the sort, since it receives the
 * scalars already reduced; the host is where this has to be caught.
 */
export interface RoPECache {
  /**
   * `[positions, headDim/2, 2]` — `cos` then `sin`, YaRN's gain already folded
   * in. f32 because that is what the GPU reads, and the reference holds the
   * same rounding so that the two agree about more than the angle.
   */
  table: Float32Array;
  /** Positions `0 <= pos < positions` are in the table. */
  positions: number;
  headDim: number;
  /** What the table was built from. `rope` refuses a cache that disagrees. */
  freq: RoPEFrequencyParams;
}

export function ropeCache({ headDim, thetaBase, positions, scaling }: RoPECacheArgs): RoPECache {
  const freq = ropeFrequencyParams(headDim, thetaBase, scaling);
  const halfDim = headDim / 2;
  const table = new Float32Array(positions * halfDim * 2);

  for (let pos = 0; pos < positions; pos += 1) {
    for (let pair = 0; pair < halfDim; pair += 1) {
      const theta = pos * invFreq(freq, headDim, pair);
      // Interleaved, `cos` first, so that the two a thread needs are adjacent —
      // the kernel walks `pair` fastest, which makes the whole row one burst.
      const at = (pos * halfDim + pair) * 2;
      table[at] = Math.cos(theta) * freq.attentionFactor;
      table[at + 1] = Math.sin(theta) * freq.attentionFactor;
    }
  }
  return { table, positions, headDim, freq };
}

export function rope({
  input,
  N,
  numHeads,
  headDim,
  posOffset,
  thetaBase,
  scaling,
  cache,
  headOffset = 0,
  headCount = numHeads,
}: RoPEArgs): Float32Array {
  const output = new Float32Array(input.length);
  const halfDim = headDim / 2;
  const freq = ropeFrequencyParams(headDim, thetaBase, scaling);
  const { attentionFactor } = freq;

  // A table built for another rotation holds angles that are wrong without
  // looking wrong. Compared against the five scalars rather than against
  // `thetaBase` and `scaling` directly, because the scalars are what the
  // rotation is: two schemes agreeing in all five rotate identically, so this
  // rejects every mismatch and no matching pair.
  if (cache) {
    if (cache.headDim !== headDim) {
      throw new Error(`rope: cache holds headDim ${cache.headDim}, called with ${headDim}`);
    }
    for (const key of Object.keys(freq) as (keyof RoPEFrequencyParams)[]) {
      if (cache.freq[key] !== freq[key]) {
        throw new Error(
          `rope: cache was built with ${key}=${cache.freq[key]}, called with ${key}=${freq[key]}`,
        );
      }
    }
  }

  // A range past the last head is a caller bug, and the quiet reading of it —
  // rotate the heads that exist and drop the rest — comes back as a tensor
  // rather than as an error. Same argument as the cache check above.
  if (headOffset < 0 || headCount < 0 || headOffset + headCount > numHeads) {
    throw new Error(
      `rope: head range [${headOffset}, ${headOffset + headCount}) does not fit ${numHeads} heads`,
    );
  }

  for (let token = 0; token < N; token += 1) {
    for (let head = 0; head < numHeads; head += 1) {
      // Outside the range the row is copied. Not skipped: the output is a
      // fresh array, so a head nobody writes stays zero and takes attention
      // with it.
      if (head < headOffset || head >= headOffset + headCount) {
        const from = (token * numHeads + head) * headDim;
        for (let i = 0; i < headDim; i += 1) output[from + i] = input[from + i]!;
        continue;
      }

      for (let pair = 0; pair < halfDim; pair += 1) {
        const pos = token + posOffset;

        let cos: number;
        let sin: number;
        if (cache && pos < cache.positions) {
          const at = (pos * halfDim + pair) * 2;
          cos = cache.table[at]!;
          sin = cache.table[at + 1]!;
        } else {
          const theta = pos * invFreq(freq, headDim, pair);
          // The attention temperature is folded into cos/sin, exactly as both
          // implementations do (`emb.cos() * self.mscale`). Scaling q and k
          // before the dot product would be the same thing one step later.
          cos = Math.cos(theta) * attentionFactor;
          sin = Math.sin(theta) * attentionFactor;
        }

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

/**
 * Multi-axis RoPE: the head split into blocks of channels, each turned by its
 * own position.
 *
 * `rope` above gives a token one position. A patch in an image transformer has
 * several — Z-Image's DiT gives every token a `(t, y, x)` triple and rotates
 * channels `[0, 32)` by `t`, `[32, 80)` by `y` and `[80, 128)` by `x`. That is
 * this op. It is not `rope` called three times: the three blocks live in one
 * head and go into one attention dot product, so splitting them across calls
 * would mean three passes over the same tensor to write disjoint thirds of it.
 *
 * ## What this follows
 *
 * `torch` has no RoPE at all, let alone a multi-axis one, so rule 7's fallback
 * is the implementation this op exists to run:
 *
 *   Tongyi-MAI/Z-Image @ 26f23eda626ffadda020b04ff79488e1d72004cd
 *   `src/zimage/transformer.py` — `RopeEmbedder`, `apply_rotary_emb`
 *   `src/config/model.py` — `ROPE_THETA = 256.0`, `ROPE_AXES_DIMS = [32, 48, 48]`
 *
 * `ops/rope/axes-cases.ts` holds outputs produced by importing those two
 * functions and running them; `ops/rope/tools/gen_axes_fixture.py` regenerates
 * it. Three things had to be decided to write this op, and upstream decides all
 * three — none of them is a guess:
 *
 * **1. Where the positions come from.** Upstream's `RopeEmbedder.__call__`
 * takes `ids` of shape `[tokens, axes]` and asserts exactly that, so the
 * positions are *given*, not derived from a patch grid. `positions` here is the
 * same array flattened token-major, and this op stays a kernel rather than
 * acquiring an opinion about how a caller lays out its patches.
 *
 * **2. One `thetaBase` or one per axis.** Upstream has a single `theta` and
 * gives each axis its own **denominator**: `1 / theta ** (arange(0, d, 2) / d)`
 * with `d` the *axis's* channel count, not the head's. So every axis sweeps the
 * same frequency range — 1 down to about `1/theta` — however many channels it
 * was given, and a 32-channel axis is a coarser sampling of that range than a
 * 48-channel one rather than a truncation of it. Dividing by `headDim` instead
 * is the plausible wrong answer and `axes-cases.ts`'s `uneven` case is there to
 * catch it. A per-axis base is deliberately not a parameter: upstream has one
 * base, and a parameter added on speculation is one every kernel variant has to
 * honour ever after.
 *
 * **3. Which channels pair up.** `torch.view_as_complex(x.reshape(*, -1, 2))`
 * pairs **adjacent** channels `2i`/`2i+1` — the same convention `rope` above
 * uses, and *not* HF Llama's `rotate_half`, which pairs `i` with `i + D/2` and
 * is why `llm/weights.ts#permuteRopeChannels` exists. A Z-Image checkpoint
 * therefore needs no channel permutation on the way in, and a Llama-style one
 * would need the same permutation as for `rope`. This is stated in the README
 * too, because getting it wrong produces a tensor rather than an error.
 *
 * ## Positions are computed here, not looked up
 *
 * Upstream tabulates each axis to `axes_lens` (`[1536, 512, 512]`) and indexes
 * the table, so a position past the end is an `IndexError` and a *negative* one
 * silently wraps to the far end of the table, Python-style. This op computes
 * the angle where it is used, so `axes_lens` is not a parameter of it:
 *
 *   - inside upstream's table the two agree to one f32 ulp — measured at
 *     2.384e-7, and that is what the fixture checks
 *   - past it this op returns the rotation that position actually implies,
 *     where upstream raises
 *   - a negative position turns the other way, which is the rotation `-p`
 *     means, rather than becoming position `axes_lens[a] - p`
 *
 * The last one is a real divergence and is chosen rather than inherited: there
 * is no table here for `-1` to be the last row *of*, and a numbering accident of
 * `torch` indexing is not part of what RoPE is. `Int32Array` for the same
 * reason `gather` uses it — a negative position stays negative instead of
 * becoming a huge positive one that nothing can tell from a real one.
 *
 * ## Odd axis dims throw
 *
 * An axis of 3 channels has no third pair to rotate, and upstream cannot
 * express one either: `arange(0, 3, 2)` gives it 2 frequencies for a block that
 * holds 1.5 pairs, and with `axes_dims = [3, 3]` upstream fails with
 * `The size of tensor a (3) must match the size of tensor b (4)` — measured,
 * torch 2.10.0. So there is no convention to follow, only conventions to
 * invent, and this throws instead of picking one.
 *
 * ## What is not here
 *
 * No `scaling`, no head range, no angle cache. Z-Image uses none of the three,
 * and each is a parameter every future variant of this kernel would have to
 * honour. The cache is the one most likely to be wanted (`ropeCacheAxes`, a
 * `[axis, pos, pair, cos/sin]` table); nothing here blocks it, because the
 * angle is a pure function of `(axis, position, pair)` exactly as `ropeCache`'s
 * is of `(position, pair)` — but it is not written until a measurement asks
 * for it (#148, rule 8).
 */
export interface RoPEAxesArgs {
  /** `[N, numHeads, headDim]`, with `headDim` the sum of `axisDims`. */
  input: Float32Array;
  /** Tokens. */
  N: number;
  numHeads: number;
  /**
   * Channels each axis owns, in order. Axis `a` owns the contiguous block
   * starting after every earlier axis's block, which is what upstream's
   * `torch.cat(result, dim=-1)` builds.
   *
   * Every entry must be positive and **even**; the sum is the head dim, which
   * is therefore not passed separately — two ways to say the same number is one
   * way to say two different ones.
   */
  axisDims: readonly number[];
  /**
   * `[N, axisDims.length]`, token-major: `positions[token * axes + axis]`.
   * Upstream's `ids`, flattened.
   *
   * **Not necessarily integers.** Z-Image indexes tokens by their grid
   * coordinate, so its positions are whole numbers; MiniMax-H3's visual VAE
   * normalises each axis to `(-1, 1)` — `2 * (i + 0.5) / n - 1` — so its are
   * fractional and its angles are those times `2π` (issue #200). A rotation by
   * a fractional angle is the same rotation; nothing in the arithmetic below
   * ever needed the position to be whole, and requiring it would have meant a
   * second kernel that differs by a binding type.
   *
   * `Float32Array` is the general one and `Int32Array` still fits: every
   * integer up to 2^24 is exact in f32, and no model here indexes that far. A
   * negative position stays negative and turns the rotation the other way,
   * which is the property the `i32` binding was chosen for.
   */
  positions: Float32Array | Int32Array;
  /** Shared by every axis. Z-Image uses 256; 1-D RoPE conventionally 10000. */
  thetaBase: number;
}

export function ropeAxes({ input, N, numHeads, axisDims, positions, thetaBase }: RoPEAxesArgs): Float32Array {
  const axes = axisDims.length;
  if (axes === 0) {
    throw new Error("ropeAxes: axisDims is empty; a head has to belong to at least one axis");
  }
  for (const [axis, dim] of axisDims.entries()) {
    // Rejected rather than rounded down: half a pair is not a rotation, and
    // upstream cannot express one either — see the note above.
    if (!Number.isInteger(dim) || dim <= 0 || dim % 2 !== 0) {
      throw new Error(`ropeAxes: axis ${axis} has ${dim} channels; every axis needs a positive even count`);
    }
  }

  const headDim = axisDims.reduce((sum, dim) => sum + dim, 0);
  // The head dim is derived, so this is the only place an `axisDims` that does
  // not describe the tensor can be caught. Reading on regardless would rotate
  // the wrong channels by the wrong angle and hand back a tensor rather than an
  // error, which is the failure this whole file is written against.
  if (input.length !== N * numHeads * headDim) {
    throw new Error(
      `ropeAxes: input holds ${input.length} values, but ${N} × ${numHeads} × ${headDim} is ${N * numHeads * headDim}`,
    );
  }
  if (positions.length !== N * axes) {
    throw new Error(
      `ropeAxes: positions holds ${positions.length} values, but ${N} tokens on ${axes} axes need ${N * axes}`,
    );
  }

  const output = new Float32Array(input.length);
  for (let token = 0; token < N; token += 1) {
    for (let head = 0; head < numHeads; head += 1) {
      // Where this axis's channels start — every earlier axis's block, in
      // order, which is what upstream's `torch.cat(..., dim=-1)` produces.
      let from = 0;
      for (let axis = 0; axis < axes; axis += 1) {
        const dim = axisDims[axis]!;
        const pos = positions[token * axes + axis]!;

        for (let pair = 0; pair < dim / 2; pair += 1) {
          // `dim`, not `headDim`: upstream's `theta ** (arange(0, d, 2) / d)`
          // normalises by the axis's own channel count, so each axis sweeps the
          // whole frequency range. Written as a negative exponent for the same
          // reason `invFreq` above is — it is then the identical expression
          // `rope` evaluates, which is what makes the one-axis case agree with
          // it bit for bit rather than merely closely.
          const theta = pos * Math.pow(thetaBase, (-2 * pair) / dim);
          const base = (token * numHeads + head) * headDim + from + pair * 2;
          const x0 = input[base]!;
          const x1 = input[base + 1]!;
          // Adjacent channels, `2i`/`2i+1`, as `torch.view_as_complex` pairs
          // them upstream and as `rope` pairs them here — not `rotate_half`.
          const cos = Math.cos(theta);
          const sin = Math.sin(theta);
          output[base] = x0 * cos - x1 * sin;
          output[base + 1] = x0 * sin + x1 * cos;
        }
        from += dim;
      }
    }
  }
  return output;
}


/**
 * MiniMax-H3's rotary channel order, as a permutation into `ropeAxes`'s.
 *
 * Issue #200. Both of H3's models rotate the same way and neither matches this
 * library's `axes` entry, so the conversion is written once here rather than
 * twice in two examples.
 *
 * H3 builds `rotDim` angles as `[axis0, axis1, axis2]` — `rotDim / 2 / axes`
 * frequencies each — and then **tiles the whole block twice**, so channel `c`
 * rotates against `c + rotDim / 2`. `ropeAxes` gives each axis a contiguous
 * block and rotates **adjacent** pairs inside it. Channel `c` of H3 therefore
 * belongs at `axis * (2 * perAxis) + 2 * freq + half`.
 *
 * Channels past `rotDim` are not rotated by H3 (`rope_dim_ratio < 1` in the
 * visual VAE, `2 * axes * rope_freq_dim < head_dim` in the DiT) and are left
 * where they are; an axis pinned at position 0 covers them, because a rotation
 * by zero is the identity.
 *
 * **Apply it to the weights, not the activations.** `permuteForRope` does the
 * same for Anima: permuting the rows of the Q and K projections costs nothing
 * per forward, and the dot product `q · k` is unchanged when both sides are
 * permuted alike. V is never rotated and must not be touched.
 *
 * `h3-cases.ts`'s generated table is what this is checked against.
 */
export function h3RopePermutation(headDim: number, rotDim: number, axes = 3): number[] {
  if (rotDim % (2 * axes) !== 0) {
    throw new Error(`h3RopePermutation: ${rotDim} rotated channels do not divide into 2 x ${axes} halves`);
  }
  if (rotDim > headDim) {
    throw new Error(`h3RopePermutation: ${rotDim} rotated channels exceed a head's ${headDim}`);
  }
  const perAxis = rotDim / 2 / axes;
  const permutation = new Array<number>(headDim);
  for (let c = 0; c < rotDim; c += 1) {
    const half = Math.floor(c / (rotDim / 2));
    const rest = c % (rotDim / 2);
    const axis = Math.floor(rest / perAxis);
    const freq = rest % perAxis;
    permutation[axis * (2 * perAxis) + 2 * freq + half] = c;
  }
  for (let c = rotDim; c < headDim; c += 1) permutation[c] = c;
  return permutation;
}

/**
 * The `positions` buffer `wgsl/axes.wgsl` binds, with the slack it expects.
 *
 * **`Float32Array`, and that is the whole point of this function existing.**
 * The binding was `array<i32>` until `ropeAxes` learned fractional positions
 * (#206) and is `array<f32>` now. Uploading an `Int32Array` to it is not an
 * error anywhere: WebGPU copies the bytes, and a small integer's bit pattern
 * read as a float is a denormal — so every angle becomes zero, every token gets
 * the identity rotation, and the model returns a plausible tensor with no
 * positional information in it. `examples/anima` shipped that for eighteen
 * commits and drew a single flat colour.
 *
 * Four axes of slack past the end, filled with something visible, for the same
 * reason `testing.ts` does it: a lane that ran past `count` reads there, and
 * zeros would look like the identity rotation rather than like a bug.
 */
export function ropeAxisPositionBuffer(
  positions: ArrayLike<number>,
  numAxes: number,
  slackValue = 9999,
): Float32Array {
  const out = new Float32Array(positions.length + numAxes * 4).fill(slackValue);
  out.set(Array.from(positions), 0);
  return out;
}
