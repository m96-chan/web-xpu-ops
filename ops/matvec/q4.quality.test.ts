import { describe, expect, it } from "vitest";
import { quantize } from "../quantize/reference.js";
import {
  Q4_CODE_MAX,
  Q4_GROUP_SIZE,
  matvec,
  matvecQ4G128,
  matvecQ8,
  packQ4,
  packQ8,
  quantizeQ4G128,
} from "./reference.js";

/**
 * What the group axis actually buys, measured rather than asserted from the
 * issue that asked for it.
 *
 * Issue #137 chose group-wise scale on **voxshot's** measurements of a real
 * model (MioTTS-0.6B), which this repository cannot re-run: it has no weights
 * and no logits. What it can do is show the *mechanism* those measurements
 * are explained by, on a matrix constructed to have the structure real weights
 * have and iid noise does not — columns whose magnitude varies by orders of
 * magnitude *within a row*. This file is where README's "The q4 format"
 * numbers come from; run it with `npm run test:file ops/matvec/q4.quality.test.ts`
 * and the table prints.
 *
 * CPU only, deliberately: nothing here needs a GPU, so nothing here can pass
 * vacuously on a machine without an adapter (`gpuTest`'s early return, the
 * failure mode this repository has already been bitten by twice).
 */

/** Deterministic LCG so every number in the README is reproducible exactly. */
function lcg(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state * 1664525 + 1013904223) >>> 0;
    return state / 4294967296;
  };
}

/**
 * Approximately-gaussian noise (sum of 12 uniforms, the standard trick), with
 * a per-128-column magnitude band of 1x / 10x / 100x. The bands are the point:
 * a per-row scale has to cover the largest of them, which leaves the smallest
 * with a fraction of a code step.
 */
function noiseMatrix(N: number, K: number, seed: number, mode: "iid" | "outlier" | "banded"): Float32Array {
  const rand = lcg(seed);
  const out = new Float32Array(N * K);
  for (let i = 0; i < N * K; i += 1) {
    let sum = 0;
    for (let j = 0; j < 12; j += 1) sum += rand();
    const value = sum - 6;
    const col = i % K;
    if (mode === "iid") out[i] = value;
    // One column in 64 is 20x the rest — an "outlier channel", the thing
    // per-channel quantization schemes are usually motivated by. Every group
    // of 128 contains two of them, so no group escapes: this case is here to
    // show that the group axis buys nothing when the structure is uniform.
    else if (mode === "outlier") out[i] = col % 64 === 0 ? value * 20 : value;
    else out[i] = value * Math.pow(10, Math.floor(col / Q4_GROUP_SIZE) % 3);
  }
  return out;
}

const bandedMatrix = (N: number, K: number, seed: number) => noiseMatrix(N, K, seed, "banded");

/**
 * Per-**row** q4 — the format this op deliberately does not have, written here
 * (and only here, in a test) so that the comparison the choice rests on can be
 * run rather than cited.
 */
function quantizeQ4PerRow(input: Float32Array, N: number, K: number) {
  const codes = new Int32Array(N * K);
  const scales = new Float32Array(N);
  for (let row = 0; row < N; row += 1) {
    let absmax = 0;
    for (let col = 0; col < K; col += 1) absmax = Math.max(absmax, Math.abs(input[row * K + col]!));
    scales[row] = absmax === 0 ? 1 : absmax / Q4_CODE_MAX;
    const inverse = absmax === 0 ? 0 : Q4_CODE_MAX / absmax;
    for (let col = 0; col < K; col += 1) {
      const value = Math.round(input[row * K + col]! * inverse);
      codes[row * K + col] = Math.max(-Q4_CODE_MAX, Math.min(Q4_CODE_MAX, value));
    }
  }
  return { codes, scales };
}

/** `sqrt(sum (got - want)^2 / sum want^2)` — 1.0 means "as wrong as returning zeros". */
function rmsRelative(got: ArrayLike<number>, want: ArrayLike<number>): number {
  let num = 0;
  let den = 0;
  for (let i = 0; i < want.length; i += 1) {
    num += (got[i]! - want[i]!) ** 2;
    den += want[i]! ** 2;
  }
  return Math.sqrt(num / den);
}

/** Worst absolute error as a fraction of the largest reference magnitude. */
function peakRelative(got: ArrayLike<number>, want: ArrayLike<number>): number {
  let peak = 0;
  let worst = 0;
  for (let i = 0; i < want.length; i += 1) peak = Math.max(peak, Math.abs(want[i]!));
  for (let i = 0; i < want.length; i += 1) worst = Math.max(worst, Math.abs(got[i]! - want[i]!));
  return worst / peak;
}

const N = 256;
const K = 2560;

describe("q4 format quality", () => {
  const matrix = bandedMatrix(N, K, 12345);
  const vector = Float32Array.from({ length: K }, (_, i) => Math.sin(i * 0.11) * 1.5);
  const groupsPerRow = Math.ceil(K / Q4_GROUP_SIZE);

  const q8 = quantize({ input: matrix, N, D: K });
  const q4Row = quantizeQ4PerRow(matrix, N, K);
  const q4g = quantizeQ4G128({ input: matrix, N, K });

  const dequantPerRow = (codes: Int32Array, scales: Float32Array) => {
    const out = new Float32Array(N * K);
    for (let row = 0; row < N; row += 1) {
      for (let col = 0; col < K; col += 1) out[row * K + col] = codes[row * K + col]! * scales[row]!;
    }
    return out;
  };
  const q8Dequant = dequantPerRow(q8.output, q8.scales);
  const q4RowDequant = dequantPerRow(q4Row.codes, q4Row.scales);
  const q4gDequant = new Float32Array(N * K);
  for (let row = 0; row < N; row += 1) {
    for (let col = 0; col < K; col += 1) {
      q4gDequant[row * K + col] =
        q4g.codes[row * K + col]! * q4g.scales[row * groupsPerRow + Math.floor(col / Q4_GROUP_SIZE)]!;
    }
  }

  /** The columns whose magnitude is 1x — two orders below their own row's peak. */
  const smallBandOnly = (source: Float32Array) => {
    const picked: number[] = [];
    for (let row = 0; row < N; row += 1) {
      for (let col = 0; col < K; col += 1) {
        if (Math.floor(col / Q4_GROUP_SIZE) % 3 === 0) picked.push(source[row * K + col]!);
      }
    }
    return Float32Array.from(picked);
  };

  it("prints the numbers README quotes", () => {
    const row = (label: string, values: number[]) =>
      `${label.padEnd(24)} ${values.map((v) => v.toExponential(3)).join("  ")}`;
    console.log(`\nN=${N} K=${K}, columns are q8-row / q4-row / q4-g128`);
    for (const mode of ["iid", "outlier", "banded"] as const) {
      const input = noiseMatrix(N, K, 12345, mode);
      const exact = matvec({ matrix: input, vector, M: N, K });
      const eight = quantize({ input, N, D: K });
      const four = quantizeQ4PerRow(input, N, K);
      const grouped = quantizeQ4G128({ input, N, K });

      const eightDequant = dequantPerRow(eight.output, eight.scales);
      const fourDequant = dequantPerRow(four.codes, four.scales);
      const groupedDequant = new Float32Array(N * K);
      for (let r = 0; r < N; r += 1) {
        for (let c = 0; c < K; c += 1) {
          groupedDequant[r * K + c] =
            grouped.codes[r * K + c]! * grouped.scales[r * groupsPerRow + Math.floor(c / Q4_GROUP_SIZE)]!;
        }
      }

      const eightOut = matvecQ8({
        weight: packQ8({ codes: eight.output, N, K }),
        scale: eight.scales,
        vector,
        N,
        K,
      });
      const fourOut = matvec({ matrix: fourDequant, vector, M: N, K });
      const groupedOut = matvecQ4G128({
        weight: packQ4({ codes: grouped.codes, N, K }),
        scale: grouped.scales,
        vector,
        N,
        K,
      });

      console.log(`\n  ${mode}`);
      console.log(
        row("  weight RMS rel", [
          rmsRelative(eightDequant, input),
          rmsRelative(fourDequant, input),
          rmsRelative(groupedDequant, input),
        ]),
      );
      console.log(
        row("  GEMV peak rel", [
          peakRelative(eightOut, exact),
          peakRelative(fourOut, exact),
          peakRelative(groupedOut, exact),
        ]),
      );
      if (mode === "banded") {
        console.log(
          row("  1x columns only", [
            rmsRelative(smallBandOnly(eightDequant), smallBandOnly(input)),
            rmsRelative(smallBandOnly(fourDequant), smallBandOnly(input)),
            rmsRelative(smallBandOnly(groupedDequant), smallBandOnly(input)),
          ]),
        );
      }
    }
    // Nothing is asserted here beyond the run completing — the claims are the
    // three tests below. This one exists so the table can be regenerated.
    expect(vector).toHaveLength(K);
  });

  it("annihilates the small columns at per-row q4, and does not at group-128", () => {
    // The mechanism, stated as a number: a per-row scale covers the 100x band,
    // so a 1x value is at most 0.07 of a code step and every one of them
    // rounds to zero — RMS-relative error of exactly 1, the score of returning
    // zeros. The group scale sees only its own 128 columns and is unaffected.
    const base = smallBandOnly(matrix);
    expect(rmsRelative(smallBandOnly(q4RowDequant), base)).toBe(1);
    expect(rmsRelative(smallBandOnly(q4gDequant), base)).toBeLessThan(0.2);
    // Not a free win for bit width: q8's per-row scale loses these columns too,
    // just less completely. The axis matters, not only the code size.
    expect(rmsRelative(smallBandOnly(q8Dequant), base)).toBeGreaterThan(0.5);
  });

  it("costs 4.25 bits per weight, measured from the buffers it produces", () => {
    // K is a multiple of 128 here, so this is the format's steady-state cost:
    // 4 bits of code plus one f32 scale per 128 weights = 4 + 32/128.
    const packed = packQ4({ codes: q4g.codes, N, K });
    const bits = ((packed.byteLength + q4g.scales.byteLength) * 8) / (N * K);
    expect(bits).toBeCloseTo(4.25, 10);
    // Against the alternatives, on the same matrix: per-row q4 is 4 + 32/K,
    // q8 is 8 + 32/K. The group axis costs a quarter of a bit.
    const q8Packed = packQ8({ codes: q8.output, N, K });
    expect(((q8Packed.byteLength + q8.scales.byteLength) * 8) / (N * K)).toBeCloseTo(8 + 32 / K, 10);
    expect(bits - (4 + 32 / K)).toBeCloseTo(0.25 - 32 / K, 10);
  });

  it("beats per-row q4 on the whole matrix too, not only on the small columns", () => {
    // Weaker than the band-restricted claim, and deliberately kept: a global
    // RMS-relative figure is dominated by the largest columns, so this is the
    // number that would be quoted by anyone comparing formats casually — and
    // it moves in the same direction, just far less.
    expect(rmsRelative(q4gDequant, matrix)).toBeLessThan(rmsRelative(q4RowDequant, matrix));
  });
});
