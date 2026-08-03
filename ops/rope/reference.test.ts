import { describe, expect, it } from "vitest";
import { rope, ropeCache, ropeFrequencyParams, type RoPEScaling } from "./reference.js";

/**
 * The parts of RoPE scaling a kernel cannot be asked about.
 *
 * `ropeFrequencyParams` collapses NTK and YaRN into five position-independent
 * scalars, and the kernel receives those scalars already computed. So the log,
 * the floor, the ceiling, the clamps and the singularity guard never reach a
 * shader — comparing the kernel against the reference cannot exercise any of
 * them, and this file is where they are pinned instead.
 *
 * Every expected value below is written out from the papers' formulas rather
 * than copied from a run, so agreeing with it means agreeing with the paper.
 *
 * Sources, which agree with each other:
 *   - YaRN, arXiv:2309.00071, §3.1 (NTK-aware) and §3.4 (YaRN)
 *   - jquesnelle/yarn — `scaled_rope/modeling_llama_yarn.py`,
 *     `scaled_rope/LlamaNTKScaledRotaryEmbedding.py`
 *   - huggingface/transformers — `modeling_rope_utils.py`
 *
 * Touches no GPU, so it costs the suite nothing.
 */
describe("rope / reference", () => {
  const headDim = 16;
  const thetaBase = 10000;

  it("leaves plain RoPE identical to the last bit when no scaling is asked for", () => {
    // The expression this file evaluated before scaling existed, inlined so
    // that the scaled path cannot quietly redefine the unscaled one. Exact
    // equality, not a tolerance: `interpolationFactor === 1` makes
    // `interpolation - extrapolation` a true IEEE zero, and `attentionFactor
    // === 1` an exact multiply, so there is nothing to round.
    const plain = (input: Float32Array, N: number, numHeads: number, posOffset: number) => {
      const output = new Float32Array(input.length);
      for (let token = 0; token < N; token += 1) {
        for (let head = 0; head < numHeads; head += 1) {
          for (let pair = 0; pair < headDim / 2; pair += 1) {
            const theta = (token + posOffset) * Math.pow(thetaBase, (-2 * pair) / headDim);
            const base = (token * numHeads + head) * headDim + pair * 2;
            const x0 = input[base]!;
            const x1 = input[base + 1]!;
            output[base] = x0 * Math.cos(theta) - x1 * Math.sin(theta);
            output[base + 1] = x0 * Math.sin(theta) + x1 * Math.cos(theta);
          }
        }
      }
      return output;
    };

    const [N, numHeads] = [5, 3];
    const input = Float32Array.from({ length: N * numHeads * headDim }, (_, i) => Math.sin(i * 0.29) * 2);
    for (const posOffset of [0, 7, 200]) {
      expect(rope({ input, N, numHeads, headDim, posOffset, thetaBase })).toEqual(
        plain(input, N, numHeads, posOffset),
      );
    }
    expect(ropeFrequencyParams(headDim, thetaBase)).toEqual({
      effectiveBase: thetaBase,
      interpolationFactor: 1,
      rampLow: 0,
      rampHigh: 1,
      attentionFactor: 1,
    });
  });

  it("gives NTK a new base and nothing else", () => {
    // §3.1: b' = b · s^(D/(D-2)). Nothing about NTK is per-position or
    // per-pair, so every other field has to come back at its identity.
    const factor = 8;
    expect(ropeFrequencyParams(headDim, thetaBase, { kind: "ntk", factor })).toEqual({
      effectiveBase: 10000 * Math.pow(8, 16 / 14),
      interpolationFactor: 1,
      rampLow: 0,
      rampHigh: 1,
      attentionFactor: 1,
    });

    // The exponent is D/(D-2), not 1. Getting that wrong is the classic NTK
    // bug and it is invisible unless the two are compared: at D = 16 they are
    // apart by more than 40%.
    const naive = 10000 * 8;
    expect(Math.abs(10000 * Math.pow(8, 16 / 14) - naive) / naive).toBeGreaterThan(0.3);
  });

  it("places YaRN's ramp where the inverse-rotation formula says", () => {
    // §3.4: pair i completes L/(2π·b^(2i/D)) rotations over a context of L, so
    // the pair that completes exactly r of them is
    //   i(r) = D·ln(L / (2π·r)) / (2·ln b)
    // β_fast = 32 rotations bounds the pairs that are safe to extrapolate,
    // β_slow = 1 the pairs that must be interpolated. Both implementations
    // floor the low end, ceil the high end, then clamp into [0, D-1].
    const originalContextLength = 64;
    const i = (r: number) =>
      (headDim * Math.log(originalContextLength / (r * 2 * Math.PI))) / (2 * Math.log(thetaBase));

    // Written out so the clamp and the truncation are both visible: at this
    // geometry i(32) = -0.994 and i(1) = 2.016.
    expect(i(32)).toBeCloseTo(-0.994, 3);
    expect(i(1)).toBeCloseTo(2.016, 3);

    const params = ropeFrequencyParams(headDim, thetaBase, {
      kind: "yarn",
      factor: 8,
      originalContextLength,
    });
    expect(params.rampLow).toBe(Math.max(Math.floor(i(32)), 0)); // 0
    expect(params.rampHigh).toBe(Math.min(Math.ceil(i(1)), headDim - 1)); // 3
    expect([params.rampLow, params.rampHigh]).toEqual([0, 3]);
    // YaRN interpolates by dividing the frequency, and leaves the base alone —
    // that division is the whole difference from NTK.
    expect(params.effectiveBase).toBe(thetaBase);
    expect(params.interpolationFactor).toBe(8);
  });

  it("truncates the correction range outwards, not to the nearest", () => {
    // The case above cannot see this: at D = 16, L = 64 the low bound rounds
    // and floors to the same number, so `Math.round` there passes everything.
    // Verified by mutation, not assumed.
    //
    // D = 128, L = 4096 — Llama-2's own geometry — separates all three:
    //   low  20.944 → floor 20, round 21
    //   high 45.027 → ceil  46, round 45, floor 45
    // Widening outwards is what both implementations do, and it matters in the
    // direction that is safe: a wider ramp blends more pairs rather than
    // extrapolating one it should not have.
    const params = ropeFrequencyParams(128, thetaBase, {
      kind: "yarn",
      factor: 8,
      originalContextLength: 4096,
    });
    expect([params.rampLow, params.rampHigh]).toEqual([20, 46]);
  });

  it("takes β_fast and β_slow as the paper's defaults when they are omitted", () => {
    const scaling = { kind: "yarn", factor: 8, originalContextLength: 64 } as const;
    expect(ropeFrequencyParams(headDim, thetaBase, scaling)).toEqual(
      ropeFrequencyParams(headDim, thetaBase, { ...scaling, betaFast: 32, betaSlow: 1 }),
    );
  });

  it("sets YaRN's attention temperature to 0.1·ln(s) + 1, and drops it below s = 1", () => {
    // §3.4: √(1/t) = 0.1·ln(s) + 1. It multiplies cos and sin, so it survives
    // into the output as a plain gain — which is why a test that only looks at
    // magnitudes can pass while every angle is wrong.
    const at = (factor: number) =>
      ropeFrequencyParams(headDim, thetaBase, { kind: "yarn", factor, originalContextLength: 64 })
        .attentionFactor;
    expect(at(8)).toBe(0.1 * Math.log(8) + 1);
    expect(at(32)).toBe(0.1 * Math.log(32) + 1);
    // Both implementations short-circuit at s ≤ 1 rather than letting ln go
    // negative and shrink the logits.
    expect(at(1)).toBe(1);
    expect(at(0.5)).toBe(1);

    // An explicit override wins, which is how a checkpoint fine-tuned with a
    // different temperature (DeepSeek-V3) is reproduced.
    expect(
      ropeFrequencyParams(headDim, thetaBase, {
        kind: "yarn",
        factor: 8,
        originalContextLength: 64,
        attentionFactor: 1.5,
      }).attentionFactor,
    ).toBe(1.5);
  });

  it("nudges the ramp apart when the clamps collapse it, instead of dividing by zero", () => {
    // The only way to reach it: a trained context short enough that both
    // correction dimensions fall at or below zero and clamp together. β_fast >
    // β_slow makes the raw bounds strictly ordered, and floor/ceil cannot close
    // a strict gap, so nothing else can.
    const params = ropeFrequencyParams(headDim, thetaBase, {
      kind: "yarn",
      factor: 8,
      originalContextLength: 6, // below 2π, so i(1) is already negative
    });
    expect(params.rampLow).toBe(0);
    expect(params.rampHigh).toBe(0.001); // both implementations add exactly this

    // The point of the nudge is that the frequencies stay finite.
    const input = Float32Array.from({ length: 16 }, (_, i) => i + 1);
    const out = rope({
      input, N: 1, numHeads: 1, headDim, posOffset: 9, thetaBase,
      scaling: { kind: "yarn", factor: 8, originalContextLength: 6 },
    });
    expect(Array.from(out).every(Number.isFinite)).toBe(true);
  });

  it("actually rides the ramp — moving β_slow alone moves the output", () => {
    // Without this, a "YaRN" that interpolated every pair equally, or none of
    // them, would still pass everything above: `rampLow`/`rampHigh` would be
    // checked as numbers but never used. β_slow only moves `rampHigh`, so a
    // difference here can only have come through the ramp.
    const shared = { N: 2, numHeads: 1, headDim, posOffset: 200, thetaBase };
    const input = Float32Array.from({ length: 2 * headDim }, (_, i) => Math.sin(i * 0.29) * 2);
    const wide = rope({ ...shared, input, scaling: { kind: "yarn", factor: 8, originalContextLength: 64, betaSlow: 0.25 } });
    const narrow = rope({ ...shared, input, scaling: { kind: "yarn", factor: 8, originalContextLength: 64, betaSlow: 1 } });

    // β_slow = 0.25 pushes rampHigh from 3 to 4, so pairs 1..3 blend
    // differently and pairs 0 and 4.. are untouched.
    expect(
      ropeFrequencyParams(headDim, thetaBase, { kind: "yarn", factor: 8, originalContextLength: 64, betaSlow: 0.25 }).rampHigh,
    ).toBe(4);
    expect(Math.max(...Array.from(wide, (v, k) => Math.abs(v - narrow[k]!)))).toBeGreaterThan(0.1);
  });

  it("separates the three schemes far past the trained context", () => {
    // This is what makes the GPU tests worth running. If plain, NTK and YaRN
    // were within tolerance of each other at the positions under test, a
    // kernel that ignored `scaling` outright would agree with all three.
    //
    // Measured max absolute divergence at these settings, f64 against f64,
    // against the 1e-3 the WGSL tests allow:
    //   positions 200..202: NTK vs plain 4.86, YaRN vs plain 3.75,
    //                       YaRN vs NTK 4.59
    // Three to four orders of magnitude of headroom.
    const shared = { N: 3, numHeads: 2, headDim, posOffset: 200, thetaBase };
    const input = Float32Array.from({ length: 3 * 2 * headDim }, (_, i) => Math.sin(i * 0.29) * 2);
    const gap = (a: Float32Array, b: Float32Array) =>
      Math.max(...Array.from(a, (v, k) => Math.abs(v - b[k]!)));

    const plain = rope({ ...shared, input });
    const ntk = rope({ ...shared, input, scaling: { kind: "ntk", factor: 8 } });
    const yarn = rope({ ...shared, input, scaling: { kind: "yarn", factor: 8, originalContextLength: 64 } });

    expect(gap(ntk, plain)).toBeGreaterThan(1);
    expect(gap(yarn, plain)).toBeGreaterThan(1);
    expect(gap(yarn, ntk)).toBeGreaterThan(1);
  });
});

/**
 * The head range: rotating a subset of the heads and copying the rest.
 *
 * The axis is the one thing worth being careful about here — see `reference.ts`.
 * The range is over **heads**, not over channels within a head, so every test
 * below asserts on whole head rows.
 *
 * Touches no GPU. `wgsl-heads.test.ts` runs the same shapes against the kernel.
 */
describe("rope / head range", () => {
  const headDim = 16;
  const thetaBase = 10000;
  /**
   * Nowhere zero, on purpose — `+ 3` puts every element in [1, 5].
   *
   * A passthrough head that was never written back reads as zero, so an input
   * that is itself zero somewhere cannot tell "copied" from "left alone". The
   * plain wave used elsewhere in this file is exactly 0 at index 0.
   */
  const live = (n: number) => Float32Array.from({ length: n }, (_, i) => Math.sin(i * 0.29) * 2 + 3);

  const [N, numHeads, posOffset] = [4, 8, 5];
  const base = { N, numHeads, headDim, posOffset, thetaBase };
  const input = live(N * numHeads * headDim);
  const row = (a: Float32Array, token: number, head: number) =>
    Array.from(a.slice((token * numHeads + head) * headDim, (token * numHeads + head + 1) * headDim));

  it("leaves the defaults byte-identical to a call that never heard of a range", () => {
    // Exact equality, not a tolerance: the default range has to be the same
    // arithmetic in the same order, not the same answer to within rounding.
    for (const scaling of [undefined, { kind: "ntk", factor: 8 } as const]) {
      expect(rope({ ...base, input, scaling, headOffset: 0, headCount: numHeads })).toEqual(
        rope({ ...base, input, scaling }),
      );
    }
  });

  it("rotates the heads inside the range and copies the ones outside it", () => {
    // Heads 2, 3, 4 rotate; 0, 1 and 5, 6, 7 pass through. Both ends of the
    // range are tested by that: an implementation that ignored `headOffset`
    // would rotate 0..2 and one that ignored `headCount` would rotate 2..7.
    const [headOffset, headCount] = [2, 3];
    const out = rope({ ...base, input, headOffset, headCount });
    const rotated = rope({ ...base, input });

    for (let token = 0; token < N; token += 1) {
      for (let head = 0; head < numHeads; head += 1) {
        const inside = head >= headOffset && head < headOffset + headCount;
        const label = `token ${token} head ${head}`;
        // Exact both ways. Inside the range it is the same expression on the
        // same numbers; outside it is a copy, and a copy that rounds is not one.
        expect(row(out, token, head), label).toEqual(
          inside ? row(rotated, token, head) : row(input, token, head),
        );
        if (inside) {
          // Otherwise "copied everything" would satisfy the line above at
          // position 0, where the rotation is the identity.
          expect(row(out, token, head), label).not.toEqual(row(input, token, head));
        } else {
          // The condition the issue names: zeros pass the equality above only
          // if the input was zero too, so the input must not be.
          expect(row(out, token, head).every((v) => v !== 0), label).toBe(true);
        }
      }
    }
  });

  it("copies the whole tensor when the range is empty", () => {
    // The degenerate end. `headCount: 0` is a caller who wants no rotation at
    // all in this block, and what comes back has to be the input rather than a
    // fresh zeroed buffer — `rope` allocates its own output, so nothing else
    // would fill it.
    const out = rope({ ...base, input, headOffset: 3, headCount: 0 });
    expect(Array.from(out)).toEqual(Array.from(input));
    expect(out.every((v) => v !== 0)).toBe(true);
  });

  it("still reads the cache for the heads it rotates", () => {
    // The two features compose: the table is indexed by position and pair and
    // by nothing else, so a head range does not change which entry a rotated
    // head reads. Poisoned table, so that "read the table" is observable.
    const cache = ropeCache({ headDim, thetaBase, positions: N + posOffset });
    const poisoned = { ...cache, table: cache.table.map((v) => v * 0.5) };
    const out = rope({ ...base, input, cache: poisoned, headOffset: 0, headCount: 1 });
    const halved = rope({ ...base, input, cache: poisoned });

    expect(row(out, 1, 0)).toEqual(row(halved, 1, 0));
    expect(row(out, 1, 1)).toEqual(row(input, 1, 1));
  });

  it("refuses a range that runs past the last head", () => {
    // Rotating heads 6..9 of 8 is a caller bug, and the quiet reading of it —
    // rotate 6 and 7, drop the rest — returns a tensor that is wrong in a way
    // no assertion downstream would catch.
    expect(() => rope({ ...base, input, headOffset: 6, headCount: 4 })).toThrow(/head range/);
    expect(() => rope({ ...base, input, headOffset: -1, headCount: 2 })).toThrow(/head range/);
    expect(() => rope({ ...base, input, headOffset: 0, headCount: -1 })).toThrow(/head range/);
    // The exact fit is not an error.
    expect(() => rope({ ...base, input, headOffset: 4, headCount: 4 })).not.toThrow();
  });
});

/**
 * The rotary cache: what the table holds, what happens past its end, and what
 * the whole thing actually saves.
 *
 * The last one is a measurement rather than a claim. `Math.sin`, `Math.cos` and
 * `Math.pow` are counted here for real, because "precomputing saves
 * transcendentals" is the kind of statement that is obviously true and
 * quantitatively wrong as often as not.
 *
 * Touches no GPU. `wgsl.test.ts` runs the same boundary against the kernel.
 */
describe("rope / cache", () => {
  const headDim = 16;
  const halfDim = headDim / 2;
  const thetaBase = 10000;
  const wave = (n: number) => Float32Array.from({ length: n }, (_, i) => Math.sin(i * 0.29) * 2);
  const gap = (a: Float32Array, b: ArrayLike<number>) =>
    Math.max(...Array.from(a, (v, k) => Math.abs(v - b[k]!)));

  const schemes: [string, RoPEScaling | undefined][] = [
    ["plain", undefined],
    ["ntk", { kind: "ntk", factor: 8 }],
    ["yarn", { kind: "yarn", factor: 8, originalContextLength: 64 }],
  ];

  it("tabulates the angles the uncached path computes, for every scheme", () => {
    // Written out from the papers' formulas rather than read back from a run,
    // and separately per scheme — NTK moves the base and YaRN moves the ramp
    // and the gain, so a table that ignored `scaling` would agree with plain
    // RoPE and with nothing else. That is the failure this covers: the table
    // depends on the frequencies, so scaling has to reach it.
    for (const [name, scaling] of schemes) {
      const p = ropeFrequencyParams(headDim, thetaBase, scaling);
      const cache = ropeCache({ headDim, thetaBase, positions: 5, scaling });
      expect(cache.positions, name).toBe(5);
      expect(cache.table.length, name).toBe(5 * halfDim * 2);

      for (let pos = 0; pos < 5; pos += 1) {
        for (let pair = 0; pair < halfDim; pair += 1) {
          const extrapolation = Math.pow(p.effectiveBase, (-2 * pair) / headDim);
          const interpolation = extrapolation / p.interpolationFactor;
          const ramp = Math.min(Math.max((pair - p.rampLow) / (p.rampHigh - p.rampLow), 0), 1);
          const theta = pos * (extrapolation + (interpolation - extrapolation) * ramp);
          // Exact, not close: the table is f32 and this is the f64 value it was
          // rounded from, so `Math.fround` is the entire difference between the
          // two. Anything else here would be a different angle.
          const at = (pos * halfDim + pair) * 2;
          expect(cache.table[at], `${name} cos pos=${pos} pair=${pair}`).toBe(
            Math.fround(Math.cos(theta) * p.attentionFactor),
          );
          expect(cache.table[at + 1], `${name} sin pos=${pos} pair=${pair}`).toBe(
            Math.fround(Math.sin(theta) * p.attentionFactor),
          );
        }
      }
    }
  });

  it("puts cos before sin, and the position stride outside the pair stride", () => {
    // Layout, pinned without repeating the formula. Position 0 has theta = 0 at
    // every pair, so its whole row is (gain, 0) repeated — which is true of the
    // cos slots and of no other arrangement of the same numbers. Swap cos and
    // sin, or transpose the two strides, and this row stops looking like that.
    const scaling = { kind: "yarn", factor: 8, originalContextLength: 64 } as const;
    const gain = ropeFrequencyParams(headDim, thetaBase, scaling).attentionFactor;
    expect(gain).toBeGreaterThan(1.2); // so that "cos at pos 0" is not just 1

    const cache = ropeCache({ headDim, thetaBase, positions: 3, scaling });
    expect(Array.from(cache.table.slice(0, halfDim * 2))).toEqual(
      Array.from({ length: halfDim * 2 }, (_, k) => (k % 2 === 0 ? Math.fround(gain) : 0)),
    );
    // And position 1 starts right after position 0, rather than the pairs of
    // position 0 being strided across the table.
    expect(cache.table[halfDim * 2]).toBe(Math.fround(Math.cos(1) * gain));
  });

  it("agrees with the uncached path when every position is in the table", () => {
    // The done-when condition from the issue — and on its own it is a weak
    // test, which is worth saying out loud: an implementation that ignored the
    // cache entirely would pass it. The two tests below are what separate "read
    // the table" from "recomputed anyway". This one is here to catch the
    // opposite mistake, a table that is used but built wrong.
    //
    // Not exact. The table is f32 and the uncached path keeps cos and sin in
    // f64, so the cached result is the same angle rounded one step earlier.
    // Measured max absolute difference, the same 2.384e-7 for all three
    // schemes on inputs of magnitude 2 — one f32 ulp at that size, which is
    // where the rounding was moved to and nothing else.
    const [N, numHeads, posOffset] = [6, 3, 4];
    const input = wave(N * numHeads * headDim);
    for (const [name, scaling] of schemes) {
      const cache = ropeCache({ headDim, thetaBase, positions: N + posOffset, scaling });
      const cached = rope({ input, N, numHeads, headDim, posOffset, thetaBase, scaling, cache });
      const uncached = rope({ input, N, numHeads, headDim, posOffset, thetaBase, scaling });
      expect(gap(cached, uncached), name).toBeLessThan(1e-6);
    }
  });

  it("reads the table below its length and computes directly past it", () => {
    // The boundary, observed with a table that is deliberately wrong: every
    // entry halved. Nothing else can tell the two paths apart, because when the
    // table is right they agree by construction.
    //
    // Positions 2..7 against a table of 4, so the run starts inside the table
    // and leaves it: tokens 0..1 must come back halved, tokens 2..5 must come
    // back as the uncached op. An off-by-one at the boundary moves exactly one
    // token across that line and fails both halves at once.
    const [N, numHeads, posOffset, positions] = [6, 2, 2, 4];
    const perToken = numHeads * headDim;
    const input = wave(N * perToken);
    const built = ropeCache({ headDim, thetaBase, positions });
    const poisoned = { ...built, table: built.table.map((v) => v * 0.5) };

    const out = rope({ input, N, numHeads, headDim, posOffset, thetaBase, cache: poisoned });
    const uncached = rope({ input, N, numHeads, headDim, posOffset, thetaBase });

    for (let token = 0; token < N; token += 1) {
      const from = token * perToken;
      const mine = out.slice(from, from + perToken);
      const theirs = uncached.slice(from, from + perToken);
      if (token + posOffset < positions) {
        // Halved: the rotation is linear in cos and sin, so halving both halves
        // the output. Close to half, not merely different from the uncached
        // value, so that "used some other table entry" fails too.
        expect(gap(mine, Array.from(theirs, (v) => v * 0.5)), `token ${token}`).toBeLessThan(1e-6);
        expect(gap(mine, theirs), `token ${token}`).toBeGreaterThan(0.5);
      } else {
        // Past the end it is the uncached op exactly — same expression, same
        // f64 arithmetic, so equality rather than a tolerance.
        expect(Array.from(mine), `token ${token}`).toEqual(Array.from(theirs));
      }
    }
  });

  it("does not wrap past the end of the table", () => {
    // Wrapping is the failure the issue names: `pos % positions` is a real
    // angle for some other position, so the output is a plausible tensor rather
    // than an error. With a table of 4, position 4 would wrap to position 0,
    // whose angles are cos = gain, sin = 0 — the identity rotation. So a
    // wrapping implementation returns that token unrotated.
    //
    // The halved table makes the counterfactual concrete: wrapping would return
    // exactly half the input for this token, and it must not.
    const [numHeads, posOffset, positions] = [2, 4, 4];
    const perToken = numHeads * headDim;
    const input = wave(perToken);
    const built = ropeCache({ headDim, thetaBase, positions });
    const poisoned = { ...built, table: built.table.map((v) => v * 0.5) };

    const out = rope({ input, N: 1, numHeads, headDim, posOffset, thetaBase, cache: poisoned });
    expect(gap(out, Array.from(input, (v) => v * 0.5))).toBeGreaterThan(0.5);
    // What it does instead: rotate by position 4, like the uncached op.
    expect(Array.from(out)).toEqual(
      Array.from(rope({ input, N: 1, numHeads, headDim, posOffset, thetaBase })),
    );
  });

  it("refuses a table built for a different rotation", () => {
    // The other way to get plausible garbage. A table is just numbers; nothing
    // about it says which base or which scaling produced it, so a caller who
    // keeps two models in one process can hand over the wrong one and get a
    // tensor back. The five scalars travel with the table so that this is an
    // error instead.
    const args = { input: wave(headDim), N: 1, numHeads: 1, headDim, posOffset: 0, thetaBase };
    const cache = ropeCache({ headDim, thetaBase, positions: 4 });

    expect(() => rope({ ...args, cache })).not.toThrow();
    // A different base: the same call shape, a table of different angles.
    expect(() => rope({ ...args, thetaBase: 500000, cache })).toThrow(/effectiveBase/);
    // Scaling asked for, table built without it. This is the one the kernel
    // cannot catch and the one #22 made reachable.
    expect(() => rope({ ...args, cache, scaling: { kind: "ntk", factor: 8 } })).toThrow(
      /effectiveBase/,
    );
    expect(() =>
      rope({ ...args, cache, scaling: { kind: "yarn", factor: 8, originalContextLength: 64 } }),
    ).toThrow(/interpolationFactor/);
    // Wrong geometry, which would also read past the end of the table.
    expect(() => rope({ ...args, headDim: 8, cache })).toThrow(/headDim/);
  });

  it("saves a measured number of transcendental calls, not an assumed one", () => {
    // Counted, per rule 9. `Math.sin`, `Math.cos` and `Math.pow` are wrapped and
    // the calls tallied, so these numbers come from the code rather than from
    // the loop bounds read off the page.
    //
    // The shape is a decode: 64 steps of one token each, at Llama-2's geometry
    // (8 heads, head_dim 128). Plain RoPE, so `ropeFrequencyParams` itself
    // contributes nothing to the count and the tally is all rotation.
    const [steps, numHeads, decodeDim] = [64, 8, 128];
    const input = wave(numHeads * decodeDim);
    const step = { input, N: 1, numHeads, headDim: decodeDim, thetaBase };

    const count = (body: () => void) => {
      const real = { sin: Math.sin, cos: Math.cos, pow: Math.pow };
      let calls = 0;
      Math.sin = (x: number) => (calls += 1, real.sin(x));
      Math.cos = (x: number) => (calls += 1, real.cos(x));
      Math.pow = (x: number, y: number) => (calls += 1, real.pow(x, y));
      try {
        body();
      } finally {
        Object.assign(Math, real);
      }
      return calls;
    };

    const uncached = count(() => {
      for (let pos = 0; pos < steps; pos += 1) rope({ ...step, posOffset: pos });
    });
    let cache!: ReturnType<typeof ropeCache>;
    const build = count(() => {
      cache = ropeCache({ headDim: decodeDim, thetaBase, positions: steps });
    });
    const reads = count(() => {
      for (let pos = 0; pos < steps; pos += 1) rope({ ...step, posOffset: pos, cache });
    });

    // Three calls — pow, cos, sin — per (token, head, pair), every step.
    expect(uncached).toBe(steps * numHeads * (decodeDim / 2) * 3);
    expect(uncached).toBe(98304);
    // The table is indexed by position and pair and by nothing else, so the
    // heads are where the saving is: one table serves all eight of them.
    expect(build).toBe(steps * (decodeDim / 2) * 3);
    expect(build).toBe(12288);
    // And the steps themselves become free.
    expect(reads).toBe(0);
    expect(uncached / (build + reads)).toBe(numHeads);
  });
});
