/** Which non-linearity to apply. The numbers are the shader's `activation_type`. */
export const ACTIVATION = { relu2: 0, silu: 1 } as const;
export type ActivationKind = (typeof ACTIVATION)[keyof typeof ACTIVATION];

/**
 * Elementwise activation.
 *
 * `relu2` is `max(0, x)²`; `silu` is `x * sigmoid(x)`. Both appear in gated
 * feed-forward blocks, which is why they share a kernel and a switch rather
 * than living apart.
 */
export function activation({
  input,
  kind,
}: {
  input: Float32Array;
  kind: ActivationKind;
}): Float32Array {
  const output = new Float32Array(input.length);
  for (let i = 0; i < input.length; i += 1) {
    const x = input[i]!;
    output[i] = kind === ACTIVATION.relu2 ? Math.max(0, x) ** 2 : x / (1 + Math.exp(-x));
  }
  return output;
}
