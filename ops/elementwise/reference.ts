/** Which binary operation to apply. The numbers are the shader's `op`. */
export const ELEMENTWISE = { add: 0, multiply: 1 } as const;
export type ElementwiseKind = (typeof ELEMENTWISE)[keyof typeof ELEMENTWISE];

/** Elementwise binary op over two equally sized arrays. */
export function elementwise({
  a,
  b,
  kind,
}: {
  a: Float32Array;
  b: Float32Array;
  kind: ElementwiseKind;
}): Float32Array {
  if (a.length !== b.length) throw new Error(`length mismatch: ${a.length} vs ${b.length}`);
  const output = new Float32Array(a.length);
  for (let i = 0; i < a.length; i += 1) {
    output[i] = kind === ELEMENTWISE.add ? a[i]! + b[i]! : a[i]! * b[i]!;
  }
  return output;
}
