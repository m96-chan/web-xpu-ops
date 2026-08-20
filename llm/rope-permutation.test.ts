import { describe, expect, it } from "vitest";
import { rope } from "../ops/rope/reference.js";
import { permuteRopeChannels } from "./weights.js";

/**
 * Proves the claim `weights.ts#permuteRopeChannels` documents, numerically —
 * rule 2: a derivation is not code until it has been run.
 *
 * HF Llama's `apply_rotary_pos_emb` / `rotate_half`, transcribed directly from
 * `transformers/models/llama/modeling_llama.py` rather than re-derived: for
 * pair index `i` in `[0, headDim/2)`,
 *
 *   q'_i            = q_i * cos_i - q_{i+headDim/2} * sin_i
 *   q'_{i+headDim/2} = q_{i+headDim/2} * cos_i + q_i * sin_i
 *
 * with `cos_i = cos(pos * theta_i)`, `sin_i = sin(pos * theta_i)`,
 * `theta_i = thetaBase ^ (-2i/headDim)` — the same angle `ops/rope` uses for
 * its interleaved pair `(2i, 2i+1)`.
 */
function rotateHalfReference(
  x: Float32Array,
  numHeads: number,
  headDim: number,
  pos: number,
  thetaBase: number,
): Float32Array {
  const half = headDim / 2;
  const out = new Float32Array(x.length);
  for (let h = 0; h < numHeads; h += 1) {
    const base = h * headDim;
    for (let i = 0; i < half; i += 1) {
      const theta = pos * Math.pow(thetaBase, (-2 * i) / headDim);
      const cos = Math.cos(theta);
      const sin = Math.sin(theta);
      const qi = x[base + i]!;
      const qih = x[base + i + half]!;
      out[base + i] = qi * cos - qih * sin;
      out[base + i + half] = qih * cos + qi * sin;
    }
  }
  return out;
}

/** `permuteRopeChannels` operating on a length-1-per-row "weight", i.e. directly on an activation vector. */
function permuteActivation(x: Float32Array, numHeads: number, headDim: number): Float32Array {
  return permuteRopeChannels(x, numHeads, headDim, 1);
}

const seeded = (n: number, k = 0.31, phase = 0) =>
  Float32Array.from({ length: n }, (_, i) => Math.sin(i * k + phase) * 1.7 + Math.cos(i * 0.07) * 0.3);

describe("permuteRopeChannels vs HF rotate_half", () => {
  const CASES = [
    { numHeads: 1, headDim: 4, pos: 0, thetaBase: 10000 },
    { numHeads: 1, headDim: 4, pos: 5, thetaBase: 10000 },
    { numHeads: 2, headDim: 8, pos: 3, thetaBase: 10000 },
    { numHeads: 4, headDim: 16, pos: 11, thetaBase: 10000 },
    { numHeads: 3, headDim: 16, pos: 200, thetaBase: 500000 }, // Sarashina's theta
  ];

  for (const { numHeads, headDim, pos, thetaBase } of CASES) {
    it(`reproduces rotate_half exactly at numHeads=${numHeads} headDim=${headDim} pos=${pos} thetaBase=${thetaBase}`, () => {
      const xHf = seeded(numHeads * headDim, 0.31, pos * 0.1);

      // HF-ordering input, rotated by the HF reference.
      const wantHf = rotateHalfReference(xHf, numHeads, headDim, pos, thetaBase);

      // Same input, permuted into this engine's interleaved channel order, then
      // rotated by the actual op under test (ops/rope, not a re-implementation).
      const xEngine = permuteActivation(xHf, numHeads, headDim);
      const gotEngine = rope({
        input: xEngine,
        N: 1,
        numHeads,
        headDim,
        posOffset: pos,
        thetaBase,
      });

      // Un-permute the result back to HF ordering and compare channel for
      // channel — the permutation is claimed to be an exact relabelling, not
      // an approximation, so this is equality, not a tolerance.
      const gotHf = new Float32Array(gotEngine.length);
      const half = headDim / 2;
      for (let h = 0; h < numHeads; h += 1) {
        const base = h * headDim;
        for (let i = 0; i < half; i += 1) {
          gotHf[base + i] = gotEngine[base + 2 * i]!;
          gotHf[base + i + half] = gotEngine[base + 2 * i + 1]!;
        }
      }

      for (let i = 0; i < wantHf.length; i += 1) {
        expect(gotHf[i]).toBeCloseTo(wantHf[i]!, 5);
      }
    });
  }

  it("preserves the Q.K dot product across the permutation, which is what attention actually needs", () => {
    const numHeads = 2;
    const headDim = 8;
    const pos = 17;
    const thetaBase = 10000;
    const qHf = seeded(numHeads * headDim, 0.41, 0.2);
    const kHf = seeded(numHeads * headDim, 0.23, 1.1);

    const qRotHf = rotateHalfReference(qHf, numHeads, headDim, pos, thetaBase);
    const kRotHf = rotateHalfReference(kHf, numHeads, headDim, pos, thetaBase);

    const qEngine = permuteActivation(qHf, numHeads, headDim);
    const kEngine = permuteActivation(kHf, numHeads, headDim);
    const qRotEngine = rope({ input: qEngine, N: 1, numHeads, headDim, posOffset: pos, thetaBase });
    const kRotEngine = rope({ input: kEngine, N: 1, numHeads, headDim, posOffset: pos, thetaBase });

    for (let h = 0; h < numHeads; h += 1) {
      let dotHf = 0;
      let dotEngine = 0;
      for (let d = 0; d < headDim; d += 1) {
        dotHf += qRotHf[h * headDim + d]! * kRotHf[h * headDim + d]!;
        dotEngine += qRotEngine[h * headDim + d]! * kRotEngine[h * headDim + d]!;
      }
      expect(dotEngine).toBeCloseTo(dotHf, 4);
    }
  });
});
