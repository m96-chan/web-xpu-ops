import { describe, expect, it } from "vitest";
import { rope, ropeFrequencyParams } from "./reference.js";

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
