/**
 * Matmul (GEMM): `C = A @ B`, with `A` `[M, K]`, `B` `[K, N]`, `C` `[M, N]`.
 *
 * The definition of correct for this op. Deliberately the naive triple loop —
 * its job is to be obviously right, not fast. A tiled kernel is measured
 * against this and against nothing else.
 *
 * Conventions follow PyTorch's `torch.mm` / 2-D `torch.matmul` (rule 7):
 *
 * - **Row-major, no transpose flags.** `torch.mm(a, b)` takes both operands in
 *   the layout they are stored in and has no `transa` / `transb`; BLAS `sgemm`
 *   does, and that is the other plausible convention. Transposing is a separate
 *   op here (`primitive/transpose`), so this one does not grow flags.
 * - **No `alpha` / `beta` accumulate.** BLAS GEMM computes
 *   `C = alpha*A@B + beta*C`; `torch.mm` computes `A@B` and writes it. Following
 *   PyTorch keeps `C` write-only, which also means the output buffer never has
 *   to be initialised by the caller.
 * - **Empty `K` gives zeros.** `torch.mm` on a `[M, 0] @ [0, N]` returns an
 *   `[M, N]` tensor of zeros rather than erroring; the empty sum is 0.
 *
 * Accumulation is in f64 because JavaScript numbers are f64. The kernels
 * accumulate in f32, so agreement — not equality — is the test.
 */
export interface MatmulArgs {
  /** `[M, K]`, row-major. */
  a: Float32Array;
  /** `[K, N]`, row-major. */
  b: Float32Array;
  /** Rows of `A` and of `C`. */
  M: number;
  /** Columns of `B` and of `C`. */
  N: number;
  /** The contracted dimension: columns of `A`, rows of `B`. */
  K: number;
}

export function matmul({ a, b, M, N, K }: MatmulArgs): Float32Array {
  const output = new Float32Array(M * N);
  for (let row = 0; row < M; row += 1) {
    for (let col = 0; col < N; col += 1) {
      let sum = 0;
      for (let k = 0; k < K; k += 1) {
        sum += a[row * K + k]! * b[k * N + col]!;
      }
      output[row * N + col] = sum;
    }
  }
  return output;
}

/**
 * `matmulQ8`: `matmul`, with the right-hand operand held as an int8 weight
 * instead of f32 (W8A32 — quantized weight, f32 activation) — `matmul`'s own
 * `C = A @ B` with `B` supplied pre-quantized, not a different contraction.
 *
 * Issue #128's motivation: `LlamaEngineQ8Resident.runPrefillResident`
 * (`llm/engine-q8-resident.ts`) used to dequantize-and-transpose every
 * projection's weight (`ops/dequant_transpose`) into a `[K, M]` f32 buffer
 * once per `forward()` call, purely so plain `matmul` had an operand to read
 * — a full write of `K * M` f32 values immediately followed by a full read
 * of the same, on *every* prefill regardless of prompt length (issue #128's
 * own background: measured 76-token and 365-token prefill at effectively the
 * same ~1.2s, so the cost was not the matmul or the prompt, it was this weight
 * prep). `matmulQ8` removes that pass entirely: the kernel reads the packed
 * int8 weight directly, in the `[M, K]` row-major layout the engine already
 * keeps resident for `matvecQ8` — no transpose, no intermediate f32 buffer.
 *
 * ## Weight format
 *
 * `weight` is **exactly** `matvecQ8`'s wire format (`ops/matvec/reference.ts`'s
 * own doc has the full byte-layout story): `[M, ceil(K/4)]` `u32`, row-major,
 * four two's-complement int8 codes packed per word, least-significant byte
 * first. `scale` is `[M]`, one absmax-derived factor per row, applied once
 * per output element after the dot product — the same "scale does not depend
 * on the contracted index" reasoning `matvecQ8`'s own doc gives, carried over
 * unchanged because the contraction itself is unchanged; `matmulQ8` is `N`
 * independent `matvecQ8` calls against the same weight, not a new kernel with
 * a new convention (rule 7: not picked quietly).
 *
 * `a` is `[N, K]` f32, row-major — `matmul`'s own `A` operand, unchanged.
 * `output` is `[N, M]` f32, row-major — one row per input row, one column
 * per weight row, matching `matmul({ a, b, M: N, N: M, K })`'s own shape once
 * `b` is `weight` dequantized-and-transposed (which is exactly how this
 * reference is defined below, per rule 8: composed from already-correct
 * pieces, not a new derivation).
 */
export interface MatmulQ8Args {
  /** `[N, K]` f32, row-major — `matmul`'s own `a`. */
  a: Float32Array;
  /** `[M, ceil(K/4)]` u32, row-major, 4 codes per word, least-significant byte first — `matvecQ8`'s own wire format. */
  weight: Uint32Array;
  /** `[M]`, one absmax-derived scale per row — `quantize`'s convention, `matvecQ8`'s own `scale`. */
  scale: Float32Array;
  /** Rows of `a`, and of `output`. */
  N: number;
  /** Rows of `weight`/`scale`, and columns of `output`. */
  M: number;
  /** Columns of `a` before packing, and the contracted dimension. */
  K: number;
}

/**
 * Sign-extends the low byte of a u32: `0xff` is `-1`, not `255`.
 *
 * Copied from `ops/matvec/reference.ts#unpackI8` rather than imported —
 * `ops/dequant_transpose/reference.ts#unpackI8` makes the identical choice
 * for the identical reason (rule 2: same convention, checked, not
 * re-derived; this op's own dependency graph stays self-contained).
 */
function unpackI8(byte: number): number {
  return byte >= 128 ? byte - 256 : byte;
}

/**
 * `weight`/`scale` dequantized into `matmul`'s `[K, M]` `b` operand, then
 * `matmul` itself — the "definition of correct" rule 8 asks for: this is not
 * a new algorithm, it is the old one (`matmul`) fed a `b` built from the
 * packed weight the same way `ops/dequant_transpose/reference.ts#dequantTranspose`
 * already does (that function is not called directly, to keep this op's
 * production-code dependency graph to itself — `ops/matmul/reference.ts`
 * importing `ops/dequant_transpose/reference.ts` for one function would tie
 * two independently-versioned ops together for no reason a caller can see).
 */
export function matmulQ8({ a, weight, scale, N, M, K }: MatmulQ8Args): Float32Array {
  const wordsPerRow = Math.ceil(K / 4);
  const b = new Float32Array(K * M);
  for (let row = 0; row < M; row += 1) {
    const s = scale[row]!;
    const rowOffset = row * wordsPerRow;
    for (let k = 0; k < K; k += 1) {
      const word = weight[rowOffset + (k >> 2)]!;
      const byte = (word >>> ((k & 3) * 8)) & 0xff;
      b[k * M + row] = unpackI8(byte) * s;
    }
  }
  return matmul({ a, b, M: N, N: M, K });
}
