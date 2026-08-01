/**
 * Gather: `output[n, :] = table[indices[n], :]`
 *
 * The definition of correct for this op. Every backend is measured against it,
 * and it is deliberately the slowest, plainest expression of the maths — its
 * job is to be obviously right, not fast.
 *
 * ## Which "gather" this is
 *
 * PyTorch spells two different things with names close enough to confuse, and
 * this is the first of them:
 *
 * - **This op** matches `torch.index_select(table, 0, indices)`, equivalently
 *   `torch.nn.functional.embedding(indices, table)` — a 1-D index picks whole
 *   *rows* of a 2-D table, and the output is `[indices.length, D]`.
 * - It is **not** `torch.gather(input, dim, index)`, where `index` has the same
 *   rank as `input` and selects one element per output position.
 *
 * Row selection is the one chosen because embedding lookup is why this op
 * exists, and embedding lookup is exactly `index_select` along dim 0. Anyone
 * wanting the element-wise form is asking for a different op, not a flag on
 * this one.
 *
 * ## Out-of-range indices
 *
 * Indices outside `[0, rows)` — including negative ones — produce a **row of
 * zeros**. PyTorch raises `IndexError` here, and a GPU kernel cannot; of the
 * behaviours that are actually available, this is the one chosen, because:
 *
 * - Clamping to the nearest valid row returns a real embedding for a token that
 *   was never in the vocabulary. It looks plausible all the way to the loss
 *   curve, which is the worst possible failure mode.
 * - Wrapping (`row % rows`) has the same problem and additionally invents a
 *   mapping nobody asked for.
 * - Zeros are what PyTorch itself already produces for a `padding_idx` row of
 *   `nn.Embedding`, so downstream code that handles padding already handles it.
 *
 * Negative indices are **not** Python-style indexing from the end; neither
 * `index_select` nor `embedding` accepts them, so `-1` is out of range and
 * yields zeros like any other out-of-range value.
 *
 * ## Index dtype
 *
 * `Int32Array`, not `Uint32Array`, even though valid indices are non-negative.
 * WGSL has no 64-bit integer, so PyTorch's `int64` index is out of reach either
 * way; `i32` is chosen over `u32` so that a negative index stays negative
 * instead of becoming a large positive one that no bounds check can tell apart
 * from a genuine — if wrong — index.
 */
export interface GatherArgs {
  /** Row-major `[rows, D]` lookup table — the embedding matrix. */
  table: Float32Array;
  /** One table row index per output row; length is the output row count. */
  indices: Int32Array;
  /** Rows in `table`. Indices outside `[0, rows)` gather zeros. */
  rows: number;
  /** Columns in `table`, and therefore in each output row. */
  D: number;
}

export function gather({ table, indices, rows, D }: GatherArgs): Float32Array {
  const output = new Float32Array(indices.length * D);
  for (let n = 0; n < indices.length; n += 1) {
    const row = indices[n]!;
    // Out of range gathers zeros, and the buffer starts zeroed.
    if (row < 0 || row >= rows) continue;
    for (let d = 0; d < D; d += 1) {
      output[n * D + d] = table[row * D + d]!;
    }
  }
  return output;
}
