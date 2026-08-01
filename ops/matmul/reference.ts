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
