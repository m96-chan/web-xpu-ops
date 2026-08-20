/**
 * `codes[outFeatures, inFeatures]` int8 (row-scale quantized, `ops/quantize`'s
 * per-row absmax convention) -> `[inFeatures, outFeatures]` f32, dequantized
 * and transposed in one pass: `output[col, row] = codes[row, col] * scale[row]`.
 *
 * Exists for issue #117's GPU-resident LLM prefill
 * (`llm/engine-q8-resident.ts#runPrefillResident`), which needs exactly this
 * — `matmul`'s `b` operand is `[K, N]` (`ops/matmul/wgsl/kernel.wgsl`), while
 * every quantized weight this engine holds is `[outFeatures, inFeatures]`,
 * `matvec`'s own `[M, K]` convention (`llm/weights.ts`'s doc) — but as a GPU
 * dispatch rather than the three CPU passes (`packInt8Rows` then
 * `dequantizePackedQ8` then `transposeRowMajor`) `LlamaEngineQ8#project`'s
 * prefill branch takes. Measured on Sarashina2.2-1B's shapes (RTX 5090,
 * Chrome): those three CPU passes cost ~100ms/layer (~2.5s for 24 layers),
 * on the critical path of *every* prefill call regardless of prompt length
 * — moving the dequant+transpose itself to the GPU (`ops/dequant_transpose/wgsl/kernel.wgsl`)
 * and only packing the raw codes on the CPU (a plain byte copy, no
 * arithmetic — ~170ms for the same 24 layers) is most of that gap closed.
 *
 * Not the inverse of `dequantize()`-then-`quantize()`, same reasoning
 * `weights-q8.ts#dequantizePackedQ8`'s own doc gives: this reads the
 * **packed** representation the engine actually keeps resident.
 */
export interface DequantTransposeArgs {
  /** `[outFeatures, ceil(inFeatures / 4)]` u32, row-major — `ops/matvec`'s `q8` packed wire format (`packQ8`). */
  weight: Uint32Array;
  /** `[outFeatures]`, one absmax-derived scale per row. */
  scale: Float32Array;
  outFeatures: number;
  inFeatures: number;
}

/** Sign-extends the byte at `lane` (0..3) of a packed word — `ops/matvec/reference.ts#unpackI8`, copied rather than imported to keep this op's dependency graph to itself (rule 2: same convention, checked, not re-derived). */
function unpackI8(word: number, lane: number): number {
  const byte = (word >>> (lane * 8)) & 0xff;
  return byte >= 128 ? byte - 256 : byte;
}

/** Returns `[inFeatures, outFeatures]` f32, row-major. */
export function dequantTranspose({ weight, scale, outFeatures, inFeatures }: DequantTransposeArgs): Float32Array {
  const wordsPerRow = Math.ceil(inFeatures / 4);
  if (weight.length !== outFeatures * wordsPerRow) {
    throw new Error(`dequantTranspose(): expected ${outFeatures * wordsPerRow} packed words, got ${weight.length}`);
  }
  if (scale.length !== outFeatures) {
    throw new Error(`dequantTranspose(): expected ${outFeatures} scales, got ${scale.length}`);
  }
  const output = new Float32Array(inFeatures * outFeatures);
  for (let row = 0; row < outFeatures; row += 1) {
    const s = scale[row]!;
    const rowWordOffset = row * wordsPerRow;
    for (let col = 0; col < inFeatures; col += 1) {
      const word = weight[rowWordOffset + (col >> 2)]!;
      output[col * outFeatures + row] = unpackI8(word, col & 3) * s;
    }
  }
  return output;
}
