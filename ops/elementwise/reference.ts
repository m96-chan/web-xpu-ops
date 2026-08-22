/** Which binary operation to apply. The numbers are the shader's `op`. */
export const ELEMENTWISE = { add: 0, multiply: 1 } as const;
export type ElementwiseKind = (typeof ELEMENTWISE)[keyof typeof ELEMENTWISE];

/** Elementwise binary op over two equally sized arrays. */
export function elementwise({
  a,
  b,
  kind,
}: {
  a: Float32Array;
  b: Float32Array;
  kind: ElementwiseKind;
}): Float32Array {
  if (a.length !== b.length) throw new Error(`length mismatch: ${a.length} vs ${b.length}`);
  const output = new Float32Array(a.length);
  for (let i = 0; i < a.length; i += 1) {
    output[i] = kind === ELEMENTWISE.add ? a[i]! + b[i]! : a[i]! * b[i]!;
  }
  return output;
}

/**
 * `elementwiseRows`: the same two operations, with `b` broadcast **along the
 * last dimension** — `out[s, d] = a[s, d] ⊕ b[d]` for `a` of shape `[S, D]`,
 * row-major, and `b` of shape `[D]` (issue #150).
 *
 * The definition of correct for this op. Every backend is measured against it,
 * and it is deliberately the slowest, plainest expression of the maths — its
 * job is to be obviously right, not fast.
 *
 * Two callers, one mechanism, which is why they are one function and not two:
 *
 *   - `kind: ELEMENTWISE.add` is **biasAdd** — a `Linear`'s bias, added to
 *     every row of its `[S, D]` output. This is exactly what
 *     `torch.nn.functional.linear(x, w, bias)` folds in, and what `ops/matvec`
 *     deliberately leaves out (see its doc: `torch.mv` has no bias, and
 *     `F.linear` keeps it a separate argument rather than a third GEMV column).
 *   - `kind: ELEMENTWISE.multiply` is **rowwiseAffine** — AdaLN's per-channel
 *     scale, the `× scale[D]` half of `x * (1 + scale) + shift`.
 *
 * ## Why this is a separate function and not `elementwise` growing a case
 *
 * `elementwise({a, b})` could have inferred the broadcast from the lengths:
 * when `b.length` divides `a.length`, treat it as a row. That was rejected.
 *
 * The lengths do not carry the intent. A `[3, 3]` activation and a `[3]` vector
 * have lengths 9 and 3, and *two* different broadcasts fit — down the rows and
 * across the columns — with nothing in the arguments to say which. They give
 * different answers, both well-formed, both finite: measured against PyTorch
 * 2.10, `torch.arange(9).reshape(3,3) + torch.tensor([1,2,3])` is
 * `[[1,3,5],[4,6,8],[7,9,11]]` while the same tensor plus
 * `c.unsqueeze(1)` is `[[1,2,3],[5,6,7],[9,10,11]]`. An inferring `elementwise`
 * would pick one silently, and a caller who meant the other would get numbers,
 * not an error. That is issue #143's class exactly — a wrong argument that
 * returns a plausible value instead of throwing — and the reason this op takes
 * `S` and `D` explicitly: the caller *states* the shape, so a mistake in it is
 * a contradiction the validation below can see rather than an ambiguity the
 * library has to resolve on its own.
 *
 * Keeping it a separate function also keeps the promise the issue actually
 * cares about: `elementwise` is byte-for-byte what it was, so the existing
 * callers (`llm/engine.ts` and `llm/engine-q8-resident.ts`: the residual add
 * and the SwiGLU multiply) cannot have changed behaviour. An extra branch
 * inside `elementwise` would have needed an argument to prove that; a function
 * that was not touched does not.
 *
 * ## Which broadcast, and only that one
 *
 * The last dimension, following NumPy/PyTorch's rule that shapes align from the
 * right: `[S, D]` against `[D]` broadcasts, `[S, D]` against `[S]` does not.
 * Verified rather than recalled (rule 2, rule 7) — PyTorch 2.10 raises "The
 * size of tensor a (3) must match the size of tensor b (6) at non-singleton
 * dimension 1" for a `[2,3]` plus a `[6]`, so `b.length === a.length` is *not*
 * a broadcast under the convention this library follows; it is a mistake, and
 * it throws here too.
 *
 * General N-dimensional broadcasting is deliberately out of scope (issue #150):
 * the requirement is the last dimension, and a shape rule that only handles the
 * case it was tested on is safer than one that quietly handles more.
 *
 * `S = 1` degenerates to plain `elementwise` and `D = 1` broadcasts a single
 * scalar over everything; both are kept legal rather than special-cased,
 * because a caller assembling `[S, D]` from a model config should not have to
 * branch on the sizes it happens to get.
 */
export interface ElementwiseRowsArgs {
  /** `[S, D]` row-major: `S` contiguous rows of `D`. */
  a: Float32Array;
  /** `[D]`, shared by every row. */
  b: Float32Array;
  /** Rows of `a`. */
  S: number;
  /** Columns of `a`, and the length of `b`. */
  D: number;
  kind: ElementwiseKind;
}

export function elementwiseRows({ a, b, S, D, kind }: ElementwiseRowsArgs): Float32Array {
  // Message format follows `ops/gqa/reference.ts` (rule 7). The dimensions are
  // checked before the lengths so that a missing one — issue #143's reported
  // mechanism, where `undefined` makes every loop run zero times and an empty
  // array comes back with no error — names the argument that is actually wrong
  // rather than a length that only looks wrong because of it.
  if (!Number.isInteger(S) || S < 1) {
    throw new Error(`elementwiseRows(): S must be a positive integer, got ${S}`);
  }
  if (!Number.isInteger(D) || D < 1) {
    throw new Error(`elementwiseRows(): D must be a positive integer, got ${D}`);
  }
  if (b.length !== D) {
    throw new Error(`elementwiseRows(): b must have D=${D} elements, got ${b.length}`);
  }
  // Exact, not "at least". A longer `a` than `S * D` is a caller who disagrees
  // with themselves about the shape, and the whole point of taking `S` and `D`
  // is to hear about that instead of silently using a prefix.
  if (a.length !== S * D) {
    throw new Error(`elementwiseRows(): a must have S*D=${S * D} elements for S=${S}, D=${D}, got ${a.length}`);
  }

  const output = new Float32Array(S * D);
  for (let row = 0; row < S; row += 1) {
    for (let col = 0; col < D; col += 1) {
      const i = row * D + col;
      output[i] = kind === ELEMENTWISE.add ? a[i]! + b[col]! : a[i]! * b[col]!;
    }
  }
  return output;
}
