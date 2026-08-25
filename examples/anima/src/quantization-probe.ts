/**
 * Simulating a quantized matmul in the CPU port, to find out what it costs.
 *
 * Issue #184. The only `subgroup-matrix` configurations this GPU offers are
 * `i8 x i8 -> i32` (M16 N16 K32 and M16 N8 K32) — no f32, no f16. Reaching a
 * tensor core therefore means **W8A8**: the weight is already int8, and the
 * *activation* would have to be too. Whether that is usable is a question about
 * numbers, not about WGSL, and it can be answered before a kernel exists.
 *
 * So `linear()` in `dit.ts` and `block.ts` consults this. Quantize-dequantize
 * both operands per row and multiply in f32, and you get **exactly** what an
 * `i8 x i8 -> i32` GEMM computes: the accumulator is integral, and at K = 8192
 * the largest possible sum is `8192 * 127 * 127 = 1.3e8`, comfortably inside
 * i32. There is no accumulation error to model — only the quantization.
 *
 * **The mode is off unless a probe turns it on**, and the tests that hold the
 * CPU port to its golden run with it off, so this cannot quietly change what
 * the reference means.
 *
 * The probe is `tools/w8a8.ts`, and it exists to be able to say *no*: if the
 * answer is that W8A8 is unusable, that is a kernel not written.
 */

/**
 * `"off"` is the shipped reference. `"weights"` quantizes the weight only —
 * what the GPU path already does, and the mode whose answer is **known**
 * (4.018e-2 over 52 blocks, measured independently in torch), so it is the
 * check on this simulation rather than a result from it. `"w8a8"` quantizes
 * both.
 */
export type QuantizationMode = "off" | "weights" | "w8a8";

let mode: QuantizationMode = "off";

export function setQuantizationMode(next: QuantizationMode): void {
  mode = next;
}

export function quantizationMode(): QuantizationMode {
  return mode;
}

/**
 * One row's worth of `quantize`'s per-row absmax, applied and undone.
 *
 * `ops/quantize`'s convention exactly — `[-127, 127]`, `127/absmax` formed as a
 * reciprocal, `floor(x) + (frac >= 0.5)` — because a probe that rounds
 * differently from the op it is predicting is predicting a different op.
 */
function roundTrip(values: Float32Array, rows: number, cols: number): Float32Array {
  const out = new Float32Array(values.length);
  for (let r = 0; r < rows; r += 1) {
    const base = r * cols;
    let absmax = 0;
    for (let c = 0; c < cols; c += 1) absmax = Math.max(absmax, Math.abs(values[base + c]!));
    if (absmax === 0) continue;
    const inverse = 127 / absmax;
    const scale = absmax / 127;
    for (let c = 0; c < cols; c += 1) {
      const scaled = values[base + c]! * inverse;
      const floored = Math.floor(scaled);
      const code = Math.max(-127, Math.min(127, floored + (scaled - floored >= 0.5 ? 1 : 0)));
      out[base + c] = code * scale;
    }
  }
  return out;
}

/** The activation as the current mode would present it to the matmul. */
export function activationFor(x: Float32Array, rows: number, inDim: number): Float32Array {
  return mode === "w8a8" ? roundTrip(x, rows, inDim) : x;
}

/**
 * Quantized weights, cached by identity.
 *
 * A weight does not change between calls and the MLP's is 16.7M elements, so
 * re-quantizing it per block turns a slow probe into an unusable one. Keyed on
 * the array itself, so a caller that hands back a different array — the rope
 * permutation does — gets its own entry rather than a stale one.
 */
const quantizedWeights = new WeakMap<Float32Array, Float32Array>();

/**
 * The weight as the current mode would present it.
 *
 * `[outDim, inDim]` row-major, and the rows are the output channels — which is
 * what `quantize`'s per-row scale is per, and what `convert_dit.py` writes.
 */
export function weightFor(weight: Float32Array, outDim: number, inDim: number): Float32Array {
  if (mode === "off") return weight;
  const cached = quantizedWeights.get(weight);
  if (cached) return cached;
  const out = roundTrip(weight, outDim, inDim);
  quantizedWeights.set(weight, out);
  return out;
}
