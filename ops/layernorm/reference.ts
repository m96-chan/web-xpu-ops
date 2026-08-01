/**
 * LayerNorm: `(x_i - mean) / sqrt(var + eps) * w_i + b_i`
 *
 * The definition of correct for this op. Every backend is measured against it,
 * and it is deliberately the slowest, plainest expression of the maths — its
 * job is to be obviously right, not fast.
 *
 * Conventions follow `torch.nn.functional.layer_norm`, verified against
 * PyTorch 2.10 rather than read off the documentation:
 *
 *   - The variance is **biased** (divided by `D`, not `D - 1`). Checked by
 *     computing both against `F.layer_norm(x, (4,), w, b, 1e-5)` in float64:
 *     the `1/D` form matched to the last bit, the `1/(D-1)` form did not.
 *   - `eps` sits **inside** the square root, `sqrt(var + eps)`, not beside it.
 *     Same check: `x / (sqrt(var) + eps)` disagreed in the sixth digit.
 *   - `weight` and `bias` are one value per normalised element and shared
 *     across rows, matching PyTorch's `normalized_shape` of `(D,)`. PyTorch
 *     treats both as optional and defaults them to 1 and 0; here they are
 *     required, because the kernel needs the bindings either way and a caller
 *     that wants the defaults can pass them.
 *
 * ## Why the variance is computed in two passes
 *
 * The tidy identity `var = E[x²] - E[x]²` is one pass and one accumulator, and
 * it is wrong exactly where LayerNorm is interesting. When the mean is large
 * relative to the spread, both terms are near `mean²` and the subtraction
 * cancels away every significant digit the spread had: at `x ≈ 8192 ± 4` in
 * f32, `x²` lands past 2^26 where the spacing between representable values is
 * 8, so each square is rounded by up to 4 while the true variance is about 5.
 * The answer is then noise, and it is noise that only appears on some inputs.
 *
 * Two passes — mean first, then the mean of the squared deviations — never
 * subtracts two large nearly-equal numbers, so nothing cancels. Welford's
 * single-pass update is equally stable and would be the choice if the data
 * arrived streaming, but it does not: the row is already in memory, and a
 * reference is read far more often than it is run. Two passes is the least
 * clever thing that is also stable, which is what this file is for.
 */
export interface LayerNormArgs {
  input: Float32Array;
  /** One value per column, shared across rows. */
  weight: Float32Array;
  /** One value per column, shared across rows. */
  bias: Float32Array;
  /** Rows. */
  N: number;
  /** Columns; the axis that is normalised. */
  D: number;
  eps: number;
}

export function layernorm({ input, weight, bias, N, D, eps }: LayerNormArgs): Float32Array {
  const output = new Float32Array(N * D);
  for (let row = 0; row < N; row += 1) {
    let sum = 0;
    for (let col = 0; col < D; col += 1) {
      sum += input[row * D + col]!;
    }
    const mean = sum / D;

    let sumSquaredDeviations = 0;
    for (let col = 0; col < D; col += 1) {
      const deviation = input[row * D + col]! - mean;
      sumSquaredDeviations += deviation * deviation;
    }
    const variance = sumSquaredDeviations / D;

    const scale = 1 / Math.sqrt(variance + eps);
    for (let col = 0; col < D; col += 1) {
      output[row * D + col] = (input[row * D + col]! - mean) * scale * weight[col]! + bias[col]!;
    }
  }
  return output;
}
