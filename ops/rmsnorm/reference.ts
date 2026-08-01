/**
 * RMSNorm: `x_i * w_i / sqrt(mean(x²) + eps)`
 *
 * The definition of correct for this op. Every backend is measured against it,
 * and it is deliberately the slowest, plainest expression of the maths — its
 * job is to be obviously right, not fast.
 */
export interface RMSNormArgs {
  input: Float32Array;
  weight: Float32Array;
  /** Rows. */
  N: number;
  /** Columns; `weight` is one value per column, shared across rows. */
  D: number;
  eps: number;
}

export function rmsnorm({ input, weight, N, D, eps }: RMSNormArgs): Float32Array {
  const output = new Float32Array(N * D);
  for (let row = 0; row < N; row += 1) {
    let sumSquares = 0;
    for (let col = 0; col < D; col += 1) {
      const value = input[row * D + col]!;
      sumSquares += value * value;
    }
    const scale = 1 / Math.sqrt(sumSquares / D + eps);
    for (let col = 0; col < D; col += 1) {
      output[row * D + col] = input[row * D + col]! * scale * weight[col]!;
    }
  }
  return output;
}
