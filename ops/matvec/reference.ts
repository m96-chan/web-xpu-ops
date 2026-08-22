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

/**
 * How many contiguous columns share one scale in the q4 format (issue #137).
 *
 * 128, and not a parameter. Issue #137's own measurement — on MioTTS-0.6B, in
 * voxshot, not in this repository — is that `g64` buys about 10% on weight RMS
 * error and nothing the output side can resolve: better on one prompt, worse
 * on the other, with the only argmax flip in the whole comparison landing on
 * `g64` itself. What it costs is 0.25 more bits per weight. A knob whose two
 * settings cannot be told apart by the thing they are meant to improve is a
 * knob that guarantees two callers will pick differently and get different
 * numbers (rule 7), so this format has one setting.
 *
 * The number is load-bearing in three places that have to agree: the scale
 * array's shape (`[N, ceil(K/128)]`), the packing (`128 % 8 == 0`, so a packed
 * word never straddles two groups and a kernel can look the scale up once per
 * word), and the tile loop of `matmulQ4G128` (`ops/matmul`). Changing it means
 * changing all three and re-measuring, which is the point of writing it down
 * once.
 */
export const Q4_GROUP_SIZE = 128;

/** The symmetric code range of the q4 format: `[-7, 7]`, not `[-8, 7]`. */
export const Q4_CODE_MAX = 7;

/**
 * Per-**group** absmax quantization to 4-bit codes — the weight side of
 * `matvecQ4G128` / `matmulQ4G128`.
 *
 * `input` is `[N, K]` row-major. Each run of `Q4_GROUP_SIZE` contiguous
 * columns *within one row* gets its own scale, and the last group of a row is
 * short when `K % 128 != 0` rather than reaching into the next row.
 *
 * ## Why this is not `ops/quantize` with a different bit count
 *
 * `ops/quantize/reference.ts#quantize` is per-**row** absmax int8, and it stays
 * that. Issue #137 measured what happens if q4 inherits that axis: on
 * MioTTS-0.6B (voxshot's measurement, not this repository's) per-row q4 puts
 * the prompt logits at 4.5e-1 peak-relative error — ten to thirty-five times
 * q8's — and cuts the greedy-agreement length from 6 tokens to 1. Group-wise
 * scale brings that back by about 3x for 0.22 extra bits per weight. Four bits
 * of resolution simply cannot absorb the dynamic range of a whole row.
 *
 * So the group axis exists because of q4, its size has to match the packing,
 * and both belong beside the op that defines the wire format — the same
 * reasoning `packQ8`'s own doc gives for living here rather than inside
 * `quantize`.
 *
 * ## Range: `[-7, 7]`, and why not `[-8, 7]`
 *
 * `scale = absmax / 7`, codes clamped to `[-7, 7]`. The obvious alternative is
 * llama.cpp's Q4_0 range `[-8, 7]`, which uses all sixteen levels and — issue
 * #137 measured this — comes out **best of every configuration tried on weight
 * RMS error** (6.26e-3–1.50e-2). It also flipped the argmax in 4 of 4
 * case×configuration pairs and failed to reach EOS within 128 tokens where the
 * f32 model needed 84. Clipping only one tail turns the error from something
 * that looks like noise into a systematic bias, and a bias accumulates
 * coherently along the contracted dimension in a way independent noise does
 * not (peak error 1.115e-1 against `[-7, 7]`'s 7.14e-2). Symmetric it is —
 * the same call `quantize` already makes for int8, for the same reason, and
 * the reason this op does **not** claim to be "Q4_0 compatible": block 128 vs
 * 32, `[-7, 7]` vs `[-8, 7]`, f32 scale vs fp16.
 *
 * ## Which reciprocal
 *
 * `code = round(value * (7 / absmax))` — the reciprocal is formed from
 * `absmax` in f64, **not** as `1 / f32(scale)`. This is the convention
 * `ops/quantize/reference.ts` and `llm/tools/quant_common.py` already use
 * (`inverse = 127 / absmax`), and it is written down because the two disagree:
 * issue #137 measured q8 prompt-logit peak-relative error moving 4.778e-2 →
 * 4.695e-2 between them, because values sitting on a rounding boundary land on
 * different sides. llama.cpp uses the other one. Neither is more correct;
 * picking silently is what rule 7 forbids.
 *
 * Rounding is JavaScript's `Math.round` — ties toward `+Infinity`, so `2.5`
 * gives `3` and `-2.5` gives `-2`. Not "half away from zero", not banker's
 * rounding, both of which `llm/tools/quant_common.py`'s own module doc walks
 * through for the int8 case; a converter that wants to produce these codes
 * offline has to reproduce this exactly, the same way that module does.
 *
 * An all-zero group takes `scale = 1` rather than dividing by zero, matching
 * `quantize`'s own guard: the codes are all zero either way, and a scale of 1
 * dequantizes them back to zero without producing a NaN.
 */
export function quantizeQ4G128({
  input,
  N,
  K,
}: {
  input: Float32Array;
  N: number;
  K: number;
}): { codes: Int32Array; scales: Float32Array } {
  const groupsPerRow = Math.ceil(K / Q4_GROUP_SIZE);
  const codes = new Int32Array(N * K);
  const scales = new Float32Array(N * groupsPerRow);

  for (let row = 0; row < N; row += 1) {
    for (let group = 0; group < groupsPerRow; group += 1) {
      const start = group * Q4_GROUP_SIZE;
      const end = Math.min(start + Q4_GROUP_SIZE, K);

      let absmax = 0;
      for (let col = start; col < end; col += 1) {
        absmax = Math.max(absmax, Math.abs(input[row * K + col]!));
      }

      scales[row * groupsPerRow + group] = absmax === 0 ? 1 : absmax / Q4_CODE_MAX;
      // Formed from `absmax`, not from the stored f32 scale — see the doc above.
      const inverse = absmax === 0 ? 0 : Q4_CODE_MAX / absmax;

      for (let col = start; col < end; col += 1) {
        const value = Math.round(input[row * K + col]! * inverse);
        codes[row * K + col] = Math.max(-Q4_CODE_MAX, Math.min(Q4_CODE_MAX, value));
      }
    }
  }
  return { codes, scales };
}

/**
 * Packs `quantizeQ4G128`'s codes into `matvecQ4G128`'s wire format:
 * `[N, ceil(K/8)]` `u32`, **eight** codes per word, least-significant nibble
 * first — code `8*w + 0` in bits 0..3, `8*w + 1` in bits 4..7, up to
 * `8*w + 7` in bits 28..31.
 *
 * The nibble order is `packQ8`'s byte order at half the width, deliberately:
 * issue #137 asked for it in as many words, because the consumer that will
 * write these buffers already emits `packQ8`'s layout, and a format differing
 * only in field width is a change of constants rather than of code.
 *
 * A code is stored as its two's-complement **nibble** — `-1` is `0xf`, `-7` is
 * `0x9` — so unpacking has to sign-extend four bits, not mask them. Masking
 * would turn every negative weight into a large positive one, which is exactly
 * the failure `packQ8`'s own test guards against one width up.
 *
 * A row's trailing lanes, when `K % 8 != 0`, are left `0`. Nothing reads them
 * (`matvecQ4G128` bounds its loop by `K`), but a defined value beats an
 * uninitialised one for anyone inspecting the buffer.
 *
 * Kept apart from `quantizeQ4G128` for the reason `packQ8` is kept apart from
 * `quantize`: the code array is what a caller can inspect and test against,
 * the packed words are what VRAM should hold, and folding the two together
 * would leave no place to stand between them.
 */
export function packQ4({ codes, N, K }: { codes: Int32Array; N: number; K: number }): Uint32Array {
  const wordsPerRow = Math.ceil(K / 8);
  const packed = new Uint32Array(N * wordsPerRow);
  for (let row = 0; row < N; row += 1) {
    for (let word = 0; word < wordsPerRow; word += 1) {
      let bits = 0;
      for (let lane = 0; lane < 8; lane += 1) {
        const col = word * 8 + lane;
        if (col >= K) break;
        const nibble = codes[row * K + col]! & 0xf;
        bits |= nibble << (lane * 4);
      }
      packed[row * wordsPerRow + word] = bits >>> 0;
    }
  }
  return packed;
}

/** Sign-extends a 4-bit field: `0xf` is `-1` and `0x9` is `-7`, not 15 and 9. */
function unpackI4(nibble: number): number {
  return nibble >= 8 ? nibble - 16 : nibble;
}

/**
 * `matvecQ4G128`: `matvecQ8`'s GEMV with the weight held as 4-bit codes and one
 * scale per group of `Q4_GROUP_SIZE` columns (W4A32 — quantized weight, f32
 * activation).
 *
 * `out[n] = sum_k unpack(weight[n, k]) * scale[n, k / 128] * vector[k]`.
 *
 * ## Layout
 *
 * - `weight`: `[N, ceil(K/8)]` `u32`, row-major, eight two's-complement 4-bit
 *   codes per word, least-significant nibble first (`packQ4` above).
 * - `scale`: `[N, ceil(K/128)]` f32, row-major — one absmax-derived factor per
 *   group, `quantizeQ4G128`'s output unchanged. **Not** `[N]`: the whole
 *   reason this op exists rather than a 4-bit `matvecQ8` is that a single
 *   scale per row loses too much (see `quantizeQ4G128`'s doc).
 * - `vector`: `[K]`, shared by every row. `output`: `[N]`.
 *
 * ## Why the scale multiplies each term here and not each group
 *
 * The kernel hoists it: 128 is a multiple of 8, so a packed word lies entirely
 * inside one group, and `ops/matvec/wgsl/q4_g128.wgsl` looks the scale up once
 * per word and applies it to that word's eight terms together. That is
 * algebraically the same sum and materially fewer multiplies. This reference
 * does the opposite — one lookup and one multiply per column, in the order the
 * definition is written — because its job is to be obviously the definition,
 * not to be the kernel a second time (rule 8). The kernel is then measured
 * against it, and a mistake in the hoisting has something to fail against.
 *
 * `K` beyond the packed row is never read: the last word of a row can carry up
 * to seven unused nibbles when `K % 8 != 0`, and the loop never asks for them.
 */
export interface MatVecQ4G128Args {
  /** `[N, ceil(K/8)]` u32, row-major, 8 codes per word, least-significant nibble first. */
  weight: Uint32Array;
  /** `[N, ceil(K/128)]` f32, row-major — one absmax-derived scale per group of 128 columns. */
  scale: Float32Array;
  /** `[K]`, shared by every row. */
  vector: Float32Array;
  /** Rows of `weight`, and the length of the output. */
  N: number;
  /** Columns before packing, and the length of `vector`. */
  K: number;
}

export function matvecQ4G128({ weight, scale, vector, N, K }: MatVecQ4G128Args): Float32Array {
  const wordsPerRow = Math.ceil(K / 8);
  const groupsPerRow = Math.ceil(K / Q4_GROUP_SIZE);
  const output = new Float32Array(N);
  for (let row = 0; row < N; row += 1) {
    let sum = 0;
    for (let col = 0; col < K; col += 1) {
      const word = weight[row * wordsPerRow + (col >> 3)]!;
      const nibble = (word >>> ((col & 7) * 4)) & 0xf;
      const groupScale = scale[row * groupsPerRow + Math.floor(col / Q4_GROUP_SIZE)]!;
      sum += unpackI4(nibble) * groupScale * vector[col]!;
    }
    output[row] = sum;
  }
  return output;
}
