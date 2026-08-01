/**
 * Row-wise softmax.
 *
 * The maximum is subtracted before exponentiating. That is not an optimisation:
 * `exp(800)` is Infinity in f32, and the whole row becomes NaN. Any
 * implementation that skips it is wrong on real logits, so the tests include a
 * row that would overflow.
 */
export function softmax({ input, N, D }: { input: Float32Array; N: number; D: number }): Float32Array {
  const output = new Float32Array(N * D);
  for (let row = 0; row < N; row += 1) {
    let max = -Infinity;
    for (let col = 0; col < D; col += 1) max = Math.max(max, input[row * D + col]!);

    let sum = 0;
    for (let col = 0; col < D; col += 1) sum += Math.exp(input[row * D + col]! - max);

    for (let col = 0; col < D; col += 1) {
      output[row * D + col] = Math.exp(input[row * D + col]! - max) / sum;
    }
  }
  return output;
}
