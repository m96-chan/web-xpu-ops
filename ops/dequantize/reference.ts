/**
 * Turns an integer accumulator back into floats.
 *
 * Both scales are applied because the product came from quantized weights and
 * quantized activations: the accumulator carries the product of both errors, and
 * undoing one without the other leaves the result off by a constant factor.
 */
export function dequantize({
  input,
  weightScale,
  inputScale,
}: {
  input: Int32Array;
  weightScale: number;
  inputScale: number;
}): Float32Array {
  const output = new Float32Array(input.length);
  for (let i = 0; i < input.length; i += 1) output[i] = input[i]! * weightScale * inputScale;
  return output;
}
