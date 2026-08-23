/**
 * bf16 to f32.
 *
 * Its own module because the browser needs it and `safetensors.ts` — the only
 * other place it belongs — opens file descriptors at import time. A pure
 * function with no dependencies is the cheapest thing to move.
 *
 * bf16 **is** f32's top 16 bits — same exponent width, mantissa truncated — so
 * this is a shift, not a conversion, and every bf16 value is exactly
 * representable in f32. Reading the same bits as f16 would give numbers of an
 * entirely different magnitude while still looking like numbers.
 */
export function bf16ToF32(values: Uint16Array): Float32Array {
  const out = new Float32Array(values.length);
  const view = new DataView(new ArrayBuffer(4));
  for (let i = 0; i < values.length; i += 1) {
    view.setUint32(0, values[i]! << 16, true);
    out[i] = view.getFloat32(0, true);
  }
  return out;
}
