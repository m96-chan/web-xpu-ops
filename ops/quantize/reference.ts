/**
 * Per-row absmax quantization to int8.
 *
 * Each row gets its own scale, so a row of small values keeps its resolution
 * instead of collapsing into a couple of levels.
 *
 * The range is [-127, 127], not [-128, 127]: symmetry matters more than the one
 * extra level, since an asymmetric range biases the dequantized result.
 *
 * An all-zero row is the case worth being careful about — `absmax` is 0, and the
 * reciprocal has to be forced to 0 rather than divided.
 */
export function quantize({
  input,
  N,
  D,
}: {
  input: Float32Array;
  N: number;
  D: number;
}): { output: Int32Array; scales: Float32Array } {
  const output = new Int32Array(N * D);
  const scales = new Float32Array(N);

  for (let row = 0; row < N; row += 1) {
    let absmax = 0;
    for (let col = 0; col < D; col += 1) absmax = Math.max(absmax, Math.abs(input[row * D + col]!));

    scales[row] = absmax === 0 ? 1 : absmax / 127;
    const inverse = absmax === 0 ? 0 : 127 / absmax;

    for (let col = 0; col < D; col += 1) {
      const value = Math.round(input[row * D + col]! * inverse);
      output[row * D + col] = Math.max(-127, Math.min(127, value));
    }
  }
  return { output, scales };
}
