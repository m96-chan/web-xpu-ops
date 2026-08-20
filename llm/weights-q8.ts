import { packQ8 } from "../ops/matvec/index.js";
import type { LlamaConfig } from "./config.js";

/**
 * A single per-row absmax-quantized linear weight (issue #105's converted
 * weight format): `codes` is `[N, K]` row-major, `ops/quantize/reference.ts#quantize`'s
 * own int8 convention (symmetric `[-127, 127]`) — but **unpacked**, one code
 * per array element, not yet `matvecQ8`'s packed `u32` wire format. `scale`
 * is `[N]`, one absmax-derived factor per row, `quantize`'s own convention.
 *
 * Kept unpacked here rather than pre-packed: packing (`packInt8Rows` below)
 * is `matvecQ8`'s storage detail, needed only for `LlamaEngineQ8`'s decode
 * path, while the prefill path (`dequantizePackedQ8`) and the CPU-side
 * embedding gather (`gatherDequantRows`) both want to address individual
 * rows/columns directly — `llm/tools/convert_weights.py` writes exactly this
 * shape (one int8 byte per code, `[N, K]` row-major) so nothing downstream
 * has to un-pack a wire format the converter never had a reason to produce.
 */
export interface QuantizedLinear {
  /** `[N, K]` row-major, one signed int8 code per element. */
  codes: Int8Array;
  /** `[N]`, one absmax-derived scale per row. */
  scale: Float32Array;
}

/**
 * `LlamaLayerWeights` (`weights.ts`), with every Linear projection quantized.
 * Norms stay f32 — issue #105's scope: "norm重みはf32" — and `wq`/`wk` carry
 * the same `permuteRopeChannels` row permutation `LlamaLayerWeights.wq`/`.wk`
 * document, applied **before** quantization by the converter (permuting rows
 * and quantizing per row commute — reordering rows does not change any row's
 * own absmax — so the order is a convenience, not a correctness requirement,
 * and the converter does it in that order because it already has the f32
 * weight in hand at that point).
 */
export interface LlamaLayerWeightsQ8 {
  attnNorm: Float32Array;
  wq: QuantizedLinear;
  wk: QuantizedLinear;
  wv: QuantizedLinear;
  wo: QuantizedLinear;
  ffnNorm: Float32Array;
  wGate: QuantizedLinear;
  wUp: QuantizedLinear;
  wDown: QuantizedLinear;
}

/**
 * `LlamaWeights`, quantized. `embedTokens` is quantized too (issue #105:
 * "埋め込みテーブル: int8+行スケール") but is never packed or fed to
 * `matvecQ8` — `LlamaEngineQ8` gathers and dequantizes only the rows a
 * forward call actually needs (`gatherDequantRows`), entirely on the CPU,
 * rather than materializing the whole `[vocabSize, hiddenSize]` table as f32
 * (102,400 x 1,792 for Sarashina2.2-1B — 733 MiB dequantized, for a table a
 * single forward call touches at most `maxSeqLen` rows of).
 */
export interface LlamaWeightsQ8 {
  /** `[vocabSize, hiddenSize]` */
  embedTokens: QuantizedLinear;
  layers: LlamaLayerWeightsQ8[];
  /** `[hiddenSize]` */
  finalNorm: Float32Array;
  /** `[vocabSize, hiddenSize]` */
  lmHead: QuantizedLinear;
}

/** Shape assertions for `LlamaWeightsQ8`, mirroring `weights.ts#assertWeightShapes`. */
export function assertWeightShapesQ8(config: LlamaConfig, weights: LlamaWeightsQ8): void {
  const { hiddenSize, numHeads, numKvHeads, headDim, ffnHidden, vocabSize, numLayers } = config;

  const expectVec = (name: string, actual: number, wanted: number) => {
    if (actual !== wanted) throw new Error(`LlamaWeightsQ8: ${name} has ${actual} elements, expected ${wanted}`);
  };
  const expectLinear = (name: string, lin: QuantizedLinear, n: number, k: number) => {
    expectVec(`${name}.codes`, lin.codes.length, n * k);
    expectVec(`${name}.scale`, lin.scale.length, n);
  };

  expectLinear("embedTokens", weights.embedTokens, vocabSize, hiddenSize);
  expectVec("finalNorm", weights.finalNorm.length, hiddenSize);
  expectLinear("lmHead", weights.lmHead, vocabSize, hiddenSize);
  if (weights.layers.length !== numLayers) {
    throw new Error(`LlamaWeightsQ8: ${weights.layers.length} layers, expected ${numLayers}`);
  }
  weights.layers.forEach((layer, i) => {
    expectVec(`layers[${i}].attnNorm`, layer.attnNorm.length, hiddenSize);
    expectLinear(`layers[${i}].wq`, layer.wq, numHeads * headDim, hiddenSize);
    expectLinear(`layers[${i}].wk`, layer.wk, numKvHeads * headDim, hiddenSize);
    expectLinear(`layers[${i}].wv`, layer.wv, numKvHeads * headDim, hiddenSize);
    expectLinear(`layers[${i}].wo`, layer.wo, hiddenSize, numHeads * headDim);
    expectVec(`layers[${i}].ffnNorm`, layer.ffnNorm.length, hiddenSize);
    expectLinear(`layers[${i}].wGate`, layer.wGate, ffnHidden, hiddenSize);
    expectLinear(`layers[${i}].wUp`, layer.wUp, ffnHidden, hiddenSize);
    expectLinear(`layers[${i}].wDown`, layer.wDown, hiddenSize, ffnHidden);
  });
}

/**
 * Concatenates per-row scale arrays end to end, in order — the scale-array
 * counterpart to `reshape.ts#concatRowsInt8`. Fusing Q/K/V (or gate/up) into
 * one projection stacks their *rows*; each row keeps the scale it already
 * had (`quantize`'s scale is per row, not per matrix), so the fused
 * projection's scale array is simply its sources' scale arrays laid end to
 * end in the same order the rows themselves were stacked — nothing is
 * recomputed.
 */
export function concatScales(scales: Float32Array[]): Float32Array {
  const total = scales.reduce((sum, s) => sum + s.length, 0);
  const output = new Float32Array(total);
  let offset = 0;
  scales.forEach((s) => {
    output.set(s, offset);
    offset += s.length;
  });
  return output;
}

/**
 * Packs unpacked `[N, K]` int8 `codes` into `matvecQ8`'s wire format
 * (`ops/matvec/reference.ts#packQ8`) without materializing a 4x-larger
 * `Int32Array` copy first.
 *
 * `packQ8`'s own signature takes `Int32Array` codes because that is what
 * `ops/quantize`'s reference returns; this repository's other quantize
 * callers already hold their codes that way. `LlamaEngineQ8` does not — its
 * codes come from `llm/tools/convert_weights.py` as one int8 byte per code,
 * kept that way through `concatRowsInt8` — and building a full `Int32Array`
 * just to call `packQ8` would transiently quadruple a resident weight's
 * memory for every projection at construction time, exactly the kind of
 * cost issue #105 flags as worth avoiding for a model this size.
 *
 * `packQ8`'s row/word loop only ever does `codes[i] & 0xff` — plain
 * indexed reads, no `Int32Array`-specific behaviour — so a same-valued
 * `Int8Array` produces byte-identical packed words; `weights-q8.test.ts`
 * checks this against a real `Int32Array` rather than assuming it (rule 2).
 */
export function packInt8Rows(codes: Int8Array, N: number, K: number): Uint32Array {
  return packQ8({ codes: codes as unknown as Int32Array, N, K });
}

/**
 * The inverse of `packInt8Rows` composed with quantization's scale: unpacks
 * a `matvecQ8`-format packed weight and applies each row's scale, producing
 * the full `[N, K]` f32 matrix `LlamaEngineQ8`'s prefill path feeds to
 * `runMatMul` (after `transposeRowMajor`).
 *
 * The unpacking — sign-extend the low byte of each packed word, four codes
 * per word, least-significant byte first — is `ops/matvec/reference.ts#matvecQ8`'s
 * own `unpackI8`, copied rather than re-derived (rule 2), so this and the GPU
 * kernel agree on what a given packed byte means by construction, not by
 * coincidence.
 *
 * Deliberately not the inverse of `dequantize()`-then-`quantize()`: this
 * reads the **packed** representation the engine actually keeps resident
 * (see `weights-q8.ts` module doc / `llm/engine-q8.ts`), not a permanently-held
 * unpacked copy — issue #105's real-model memory concern is exactly why
 * `LlamaEngineQ8` keeps only the packed form after construction and
 * reconstructs f32 on demand, once per `forward` call that needs it (prefill),
 * rather than once per token.
 */
export function dequantizePackedQ8(weight: Uint32Array, scale: Float32Array, N: number, K: number): Float32Array {
  const wordsPerRow = Math.ceil(K / 4);
  const output = new Float32Array(N * K);
  for (let row = 0; row < N; row += 1) {
    const rowWordOffset = row * wordsPerRow;
    const s = scale[row]!;
    for (let col = 0; col < K; col += 1) {
      const word = weight[rowWordOffset + (col >> 2)]!;
      const byte = (word >>> ((col & 3) * 8)) & 0xff;
      const signed = byte >= 128 ? byte - 256 : byte;
      output[row * K + col] = signed * s;
    }
  }
  return output;
}

/**
 * Copies a `QuantizedLinear`'s `codes` and `scale` into fresh, independent
 * buffers, decoupled from whatever buffer the source views were backed by.
 *
 * A manifest loader (`weights-q8-io.ts#buildLlamaWeightsQ8`) hands back
 * every quantized weight as a **view** into one shared buffer covering the
 * *entire* checkpoint's codes (and another for scales) — cheap to build, but
 * a `TypedArray` view keeps its whole backing `ArrayBuffer` alive, not just
 * the slice it reads. `LlamaEngineQ8` only needs `embedTokens` to outlive
 * construction (every per-layer Linear's codes are consumed once, while
 * building the packed `matvecQ8` form `project` actually reads); without
 * this copy, retaining `embedTokens`'s view alone would pin the whole
 * checkpoint's raw codes resident — ~1.4 GiB for Sarashina2.2-1B, for a
 * ~183 MiB table — for the engine's entire lifetime. Measured, not assumed
 * (rule 2): a construction-only timing run showed resident memory roughly
 * double what `LlamaEngineQ8`'s packed weights alone need, traced to exactly
 * this retention.
 */
export function cloneQuantizedLinear(source: QuantizedLinear): QuantizedLinear {
  return { codes: source.codes.slice(), scale: source.scale.slice() };
}

/**
 * Dequantizes only the requested rows of a `QuantizedLinear`, in request
 * order (duplicates gather and dequantize twice, same as `ops/gather`'s own
 * `torch.index_select` convention) — the CPU-side embedding gather issue #105
 * calls for: `LlamaEngineQ8.forward` calls this with `tokens` as `indices`
 * instead of dispatching `runGather` against a dequantized-in-full embedding
 * table.
 */
export function gatherDequantRows(table: QuantizedLinear, indices: number[], D: number): Float32Array {
  const rows = table.scale.length;
  const output = new Float32Array(indices.length * D);
  indices.forEach((rowIndex, i) => {
    // Same convention as ops/gather: indices outside [0, rows) gather zeros
    // (the output starts zeroed). Without this, an out-of-vocab token id from
    // an externally encoded prompt reads undefined and fills the row with
    // NaN, which propagates through every layer and still argmaxes to a
    // plausible-looking token.
    if (rowIndex < 0 || rowIndex >= rows) return;
    const s = table.scale[rowIndex]!;
    const base = rowIndex * D;
    for (let c = 0; c < D; c += 1) output[i * D + c] = table.codes[base + c]! * s;
  });
  return output;
}
