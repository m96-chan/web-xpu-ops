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
