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
