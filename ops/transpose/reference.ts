/**
 * Transpose of a row-major 2D matrix: `output[c][r] = input[r][c]`.
 *
 * The definition of correct for this op. Every backend is measured against it,
 * and it is deliberately the slowest, plainest expression of the move — its job
 * is to be obviously right, not fast.
 *
 * Matches PyTorch's `torch.transpose(x, 0, 1).contiguous()` (equivalently
 * `x.t().contiguous()`) for a 2D f32 tensor. PyTorch's own `transpose` returns a
 * *view* that only swaps strides and copies nothing; there is no such thing as a
 * strided view here, so this materialises the result the way `.contiguous()`
 * would. That is the only difference, and it is a difference in what a tensor
 * is, not in which value ends up where.
 *
 * No arithmetic is involved, so the output is a permutation of the input and
 * every element is bit-identical to its source.
 */
export interface TransposeArgs {
  /** Row-major `[rows, cols]`. */
  input: Float32Array;
  rows: number;
  cols: number;
}

/** Returns a row-major `[cols, rows]` matrix. */
export function transpose({ input, rows, cols }: TransposeArgs): Float32Array {
  const output = new Float32Array(rows * cols);
  for (let r = 0; r < rows; r += 1) {
    for (let c = 0; c < cols; c += 1) {
      output[c * rows + r] = input[r * cols + c]!;
    }
  }
  return output;
}
