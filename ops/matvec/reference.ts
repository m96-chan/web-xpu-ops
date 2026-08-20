import { activation, ACTIVATION } from "../activation/reference.js";

/**
 * matvec (GEMV): `out[i] = sum_k matrix[i, k] * vector[k]`
 *
 * The definition of correct for this op. Every backend is measured against it,
 * and it is deliberately the slowest, plainest expression of the maths — its job
 * is to be obviously right, not fast.
 *
 * ## Which convention this follows
 *
 * There are two live conventions and they disagree, so this does not pick
 * quietly (rule 7):
 *
 * - **BLAS `sgemv`** is `y := alpha * op(A) * x + beta * y`, with a transpose
 *   flag, two scalars, and an accumulate-into-`y` semantics.
 * - **PyTorch `torch.mv(mat, vec)`** is `out = mat @ vec`, no scalars, no
 *   transpose flag, no accumulation. `mat` is `(n, m)`, `vec` is `(m,)`, and the
 *   result is `(n,)`.
 *
 * This follows **PyTorch `torch.mv`**: `matrix` is `[M, K]` row-major, `vector`
 * is `[K]`, and the output is `[M]`. No alpha, no beta, no transpose, and no
 * bias — `torch.mv` has none of them either, and `torch.nn.functional.linear`
 * puts the bias in a separate argument rather than folding it in here.
 *
 * The row-major `[M, K]` layout is the load-bearing part, not a detail. One
 * output element is one contiguous run of `K` weights, which is exactly the
 * access pattern a bandwidth-bound kernel wants: each weight is read once, in
 * address order, and there is no reuse to tile for. A `[K, M]` layout would
 * compute the same numbers and stream them in the wrong order.
 *
 * Accumulation here is plain left-to-right in f64. The kernels accumulate in
 * f32 and in a different order, so they are compared by agreement rather than
 * equality — see `harness/agree.ts`.
 */
export interface MatVecArgs {
  /** `[M, K]` row-major: one contiguous row of `K` weights per output element. */
  matrix: Float32Array;
  /** `[K]`, shared by every row. */
  vector: Float32Array;
  /** Rows of `matrix`, and the length of the output. */
  M: number;
  /** Columns of `matrix`, and the length of `vector`. */
  K: number;
}

export function matvec({ matrix, vector, M, K }: MatVecArgs): Float32Array {
  const output = new Float32Array(M);
  for (let row = 0; row < M; row += 1) {
    let sum = 0;
    for (let col = 0; col < K; col += 1) {
      sum += matrix[row * K + col]! * vector[col]!;
    }
    output[row] = sum;
  }
  return output;
}

/**
 * `matvecQ8`: the same GEMV, with the weight held as int8 instead of f32
 * (W8A32 — quantized weight, f32 activation).
 *
 * Issue #97's motivation is bandwidth: decode-time GEMV is memory-bound, and an
 * int8 weight is a quarter the bytes of f32 for the same values. `matvecQ8`
 * halves that again versus a naive int8-as-i32-per-lane layout by packing four
 * codes into one `u32` word — the wire format this op actually reads, not just
 * a storage convenience.
 *
 * ## Weight format
 *
 * `weight` is `[N, ceil(K/4)]` `u32`, row-major: each row is its own run of
 * packed words, independent of every other row (`packQ8` below produces
 * exactly this). Within a word the four codes are **least-significant byte
 * first** — code `4*w + 0` sits in bits 0..7, `4*w + 1` in bits 8..15, and so
 * on. That is the byte order a little-endian host gets for free by viewing an
 * `Int8Array`'s buffer as a `Uint32Array`, which is the whole reason to pick
 * it: it is the one order that needs no shuffling on the host that produced
 * the codes in the first place, only on the GPU that has to unpack them.
 *
 * A code is stored as its two's-complement byte — `-1` is `0xff`, not `255` —
 * so unpacking has to sign-extend the low byte, not just mask it.
 *
 * ## Scale
 *
 * `scale` is `[N]`, one factor per row, applied after the dot product
 * (`sum_k code[n,k] * vector[k]) * scale[n]` — equivalent to scaling every term
 * first because the scale does not depend on `k`, and cheaper by `K - 1`
 * multiplies per row. It is the *same* per-row absmax scale `quantize`
 * produces (symmetric `[-127, 127]`, see `ops/quantize/reference.ts`), not a
 * new convention invented here: `matvecQ8(weight: packQ8(quantize(w).output, N, K),
 * scale: quantize(w).scales, ...)` is the intended pipeline, and rule 7 says
 * not to pick a second one quietly.
 *
 * `K` beyond the packed row is never read — the last word of a row can carry
 * up to three unused high bytes when `K % 4 != 0`, and those lanes are simply
 * never asked for, the same way `matvec`'s strided loop leaves a partial final
 * pass rather than branching on it.
 */
export interface MatVecQ8Args {
  /** `[N, ceil(K/4)]` u32, row-major, 4 codes per word, least-significant byte first. */
  weight: Uint32Array;
  /** `[N]`, one absmax-derived scale per row — `quantize`'s convention. */
  scale: Float32Array;
  /** `[K]`, shared by every row. */
  vector: Float32Array;
  /** Rows of `weight`, and the length of the output. */
  N: number;
  /** Columns before packing, and the length of `vector`. */
  K: number;
}

/** Sign-extends the low byte of a u32: `0xff` is `-1`, not `255`. */
function unpackI8(byte: number): number {
  return byte >= 128 ? byte - 256 : byte;
}

export function matvecQ8({ weight, scale, vector, N, K }: MatVecQ8Args): Float32Array {
  const wordsPerRow = Math.ceil(K / 4);
  const output = new Float32Array(N);
  for (let row = 0; row < N; row += 1) {
    const rowOffset = row * wordsPerRow;
    let sum = 0;
    for (let col = 0; col < K; col += 1) {
      const word = weight[rowOffset + (col >> 2)]!;
      const byte = (word >>> ((col & 3) * 8)) & 0xff;
      sum += unpackI8(byte) * vector[col]!;
    }
    output[row] = sum * scale[row]!;
  }
  return output;
}

/**
 * `matvecQ8Ffn`: `silu(matvecQ8(weightGate, x)) * matvecQ8(weightUp, x)` — one
 * row, two int8 weights, in a single logical op instead of four (issue #111).
 *
 * ## Why this exists
 *
 * `llm/engine-q8-resident.ts`'s decode step computes exactly this shape every
 * layer, as four separate GPU dispatches: `matvecQ8(gate)`, `matvecQ8(up)`,
 * `activation(silu)`, `elementwise(multiply)`. Issue #110 already made one
 * decode step pay exactly one `queue.submit`/readback, so what four
 * dispatches cost past that point is per-dispatch encoding and pass-boundary
 * overhead, not round-trip latency — fusing them into one halves the
 * weight-reading dispatches and removes the two elementwise ones entirely,
 * without changing the arithmetic: `silu` and `*` are still applied in f32,
 * in the same order, to the same two dot products this reference computes
 * separately below. (PR #127 review, item 5: an earlier version of this doc
 * cited this issue's own opening "prefill at 76 vs. 365 tokens costs about
 * the same wall time" observation as the reason dispatch count dominates
 * here — that observation is real, but it describes *prefill*'s own fixed
 * cost, which runs through `matmul`, never `matvecQ8`, and this PR's own
 * measurement (README, "Fused decode kernels (issue #111)") shows prefill
 * is untouched by these two fused kernels, as expected. The actual, smaller
 * measured effect — 7.2% lower decode latency from cutting 411 dispatches
 * to 291 per token — is what motivates this op; see that README section
 * for the numbers rather than the refuted inference.)
 *
 * ## Reference shape, not reference *independence*
 *
 * This composes `matvecQ8` (already the correctness definition for a single
 * quantized GEMV row, `ops/matvec/q8.wgsl.test.ts`) with `ops/activation`'s
 * own `activation({ kind: ACTIVATION.silu })` (rule 7 — imported, not
 * copied or re-derived a second time; same precedent as `ops/gqa/reference.ts`
 * importing `ops/attention/reference.ts#resolveMask` rather than restating
 * its mask logic) rather than re-deriving the packed-int8 unpacking a third
 * time. Rule 8 asks this reference to be "obviously right", and composing
 * three already-obviously-right pieces is more obviously right than a fresh
 * from-scratch loop would be — the risk this op actually carries is in the
 * *fusion* (does the WGSL kernel's shared single-pass-over-`vector` unpacking
 * of two weights agree with computing them apart?), which this reference,
 * built from the already-verified parts, is positioned to catch. (PR #127
 * review, item 8 and item 1: an earlier version of this function copied
 * `silu`'s formula inline instead of importing it, and had no CPU-only test
 * independent of this file's own composition — see
 * `ops/matvec/reference.test.ts`'s `matvecQ8Ffn`/`matvecQ8Residual`
 * `describe` blocks for the hand-computed ground truth that catches a
 * gate/up mix-up this function's own kernel-vs-reference tests alone could
 * not.)
 */
export interface MatVecQ8FfnArgs {
  /** `[N, ceil(K/4)] u32` — `matvecQ8`'s packed format, gate projection. */
  weightGate: Uint32Array;
  /** `[N]` — gate's per-row scale. */
  scaleGate: Float32Array;
  /** `[N, ceil(K/4)] u32` — up projection, same wire format as `weightGate`. */
  weightUp: Uint32Array;
  /** `[N]` — up's per-row scale. */
  scaleUp: Float32Array;
  /** `[K]`, shared by both projections. */
  vector: Float32Array;
  N: number;
  K: number;
}

export function matvecQ8Ffn({ weightGate, scaleGate, weightUp, scaleUp, vector, N, K }: MatVecQ8FfnArgs): Float32Array {
  const gate = matvecQ8({ weight: weightGate, scale: scaleGate, vector, N, K });
  const up = matvecQ8({ weight: weightUp, scale: scaleUp, vector, N, K });
  // `ops/activation`'s own silu, not a copied formula (rule 7 — PR #127
  // review, item 8).
  const gated = activation({ input: gate, kind: ACTIVATION.silu });
  const output = new Float32Array(N);
  for (let row = 0; row < N; row += 1) {
    output[row] = gated[row]! * up[row]!;
  }
  return output;
}

/**
 * `matvecQ8Residual`: `residual + matvecQ8(weight, x)` — one row instead of
 * `matvecQ8` followed by an `elementwise(add)` dispatch (issue #111).
 *
 * `llm/engine-q8-resident.ts`'s decode step uses this for both
 * post-attention (`o_proj`) and post-FFN (`down_proj`) residual adds: each
 * currently pays a `matvecQ8` dispatch and then a separate `elementwise`
 * dispatch to add the pre-projection residual stream back in. The residual
 * add does not depend on any column of `weight`, so it can be applied once
 * per row, after that row's reduction — the same "scale is per-row, applied
 * once after the dot product, not per term" shape `matvecQ8`'s own doc
 * already establishes for `scale`; `residual` rides along the same way.
 */
export interface MatVecQ8ResidualArgs {
  /** `[N, ceil(K/4)] u32` — `matvecQ8`'s packed format. */
  weight: Uint32Array;
  /** `[N]` — one absmax-derived scale per row. */
  scale: Float32Array;
  /** `[K]`. */
  vector: Float32Array;
  /** `[N]`, added to this row's `matvecQ8` output after the scale. */
  residual: Float32Array;
  N: number;
  K: number;
}

export function matvecQ8Residual({ weight, scale, vector, residual, N, K }: MatVecQ8ResidualArgs): Float32Array {
  const projected = matvecQ8({ weight, scale, vector, N, K });
  const output = new Float32Array(N);
  for (let row = 0; row < N; row += 1) {
    output[row] = residual[row]! + projected[row]!;
  }
  return output;
}

/**
 * Packs per-row int8 codes (`quantize`'s output: values in `[-127, 127]`,
 * `[N, K]` row-major, one code per array element) into `matvecQ8`'s wire
 * format: `[N, ceil(K/4)]` `u32`, four codes per word, least-significant byte
 * first.
 *
 * Kept apart from `quantize` rather than folded into it: `quantize` is a
 * general activation/weight quantizer whose output (`Int32Array`, one code per
 * lane) is what a compute kernel wants to read, not what a weight wants to sit
 * in VRAM as. Packing is specific to `matvecQ8`'s storage layout, so it lives
 * beside the op that defines that layout.
 *
 * A row's trailing lanes — when `K % 4 != 0` — are left `0` rather than
 * carrying anything from a neighbouring row; `matvecQ8` never reads them, but
 * a defined value beats an uninitialised one for anyone who inspects the
 * buffer directly.
 *
 * (PR #127 review, item 6: an earlier version of this PR left this doc
 * comment stranded above `matvecQ8Ffn` instead — inserted before it rather
 * than before `packQ8`, so it silently attached to the wrong declaration
 * and `packQ8` itself lost its hover/`.d.ts` doc entirely. Moved back to
 * directly precede the function it actually documents.)
 */
export function packQ8({ codes, N, K }: { codes: Int32Array; N: number; K: number }): Uint32Array {
  const wordsPerRow = Math.ceil(K / 4);
  const packed = new Uint32Array(N * wordsPerRow);
  for (let row = 0; row < N; row += 1) {
    for (let word = 0; word < wordsPerRow; word += 1) {
      let bits = 0;
      for (let lane = 0; lane < 4; lane += 1) {
        const col = word * 4 + lane;
        if (col >= K) break;
        const byte = codes[row * K + col]! & 0xff;
        bits |= byte << (lane * 8);
      }
      packed[row * wordsPerRow + word] = bits >>> 0;
    }
  }
  return packed;
}
