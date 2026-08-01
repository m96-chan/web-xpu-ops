/**
 * Scatter: indexed writes into rows, `output[row][index[row][slot]] += src[row][slot]`.
 *
 * The definition of correct for this op. Every backend is measured against it,
 * and it is deliberately the slowest, plainest expression of the maths — its
 * job is to be obviously right, not fast.
 *
 * ## Colliding indices accumulate. This is the decision.
 *
 * When two slots in a row name the same target, their values are **added**.
 * Not "the last one wins", not "undefined":
 *
 * - *Last write wins* cannot be implemented on a GPU and cannot be tested. The
 *   order in which threads reach a slot is unspecified, so "last" has no
 *   meaning — it is undefined behaviour wearing a reassuring name, and callers
 *   would build on whichever answer their first device happened to give.
 * - *Undefined* would be honest, but it makes the op useless for the three
 *   things scatter is actually for — gradient accumulation, MoE dispatch and
 *   bincount — all of which collide by construction.
 *
 * Accumulation is the only rule that gives the same answer for every possible
 * thread ordering, so it is the only one a caller can depend on. The kernel
 * pays for it with an atomic per write.
 *
 * ## What this matches in PyTorch
 *
 * `torch.zeros(N, D).scatter_add_(1, index, src)` — that is, `scatter_add_`
 * along `dim=1`, **not** `scatter_`. PyTorch documents `scatter_` itself as
 * non-deterministic when indices collide, which is exactly the behaviour this
 * op refuses to expose.
 *
 * Three deliberate deviations from PyTorch, none of them silent:
 *
 * 1. **`self` is implicitly zero.** PyTorch scatters in place onto an existing
 *    tensor. A dispatch cannot: seeding the output from a base tensor and
 *    accumulating into it within the same dispatch would race, there being no
 *    barrier across workgroups. Callers who need `base + scatter` add the base
 *    in a second dispatch.
 * 2. **Out-of-range indices are ignored** rather than raising. WGSL has no way
 *    to raise, and an unchecked out-of-range write is worse than useless: the
 *    address stays inside the flat output buffer and silently lands in a
 *    *different row*. Skipping is what makes the behaviour defined at all.
 *    PyTorch's `IndexError` is a host-side luxury.
 * 3. **`index` is i32**, where PyTorch uses int64. WGSL has no 64-bit integer.
 *    Negative indices are out of range, as they are in PyTorch — `scatter_add_`
 *    does not take Python-style negative indexing.
 *
 * ## What is still order-dependent
 *
 * The *value* of a collided sum, in the last bits. Addition of f32 is not
 * associative, so a slot summed in a different order can differ by a rounding
 * step per collision. The result is deterministic in exact arithmetic and
 * agrees with the reference to within measured tolerance; it is not bit-exact.
 * See `wgsl.test.ts` for the measured figure.
 */
export interface ScatterArgs {
  /** `[N, S]` — the values being written. */
  src: Float32Array;
  /** `[N, S]` — for each value, which column of its row to add it into. */
  index: Int32Array;
  /** Rows. */
  N: number;
  /** Slots per row: how many writes each row performs. */
  S: number;
  /** Columns of the output row; valid indices are `[0, D)`. */
  D: number;
}

export function scatter({ src, index, N, S, D }: ScatterArgs): Float32Array {
  const output = new Float32Array(N * D);
  for (let row = 0; row < N; row += 1) {
    for (let slot = 0; slot < S; slot += 1) {
      const target = index[row * S + slot]!;
      // Out of range is dropped, not clamped and not wrapped: clamping would
      // corrupt a real element, which is the failure mode this check exists
      // to prevent.
      if (target < 0 || target >= D) continue;
      // Accumulation in f32, one step at a time, matching what the atomic on
      // the GPU does — only the order differs.
      output[row * D + target] = output[row * D + target]! + src[row * S + slot]!;
    }
  }
  return output;
}
