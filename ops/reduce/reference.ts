/** Which reduction to apply. The numbers are the shader's `kind`. */
export const REDUCE = { sum: 0, max: 1, min: 2, mean: 3 } as const;
export type ReduceKind = (typeof REDUCE)[keyof typeof REDUCE];

export interface ReduceArgs {
  input: Float32Array;
  /** Product of the dimensions before the reduced axis. */
  outer: number;
  /** Length of the reduced axis; this is what gets collapsed. */
  axis: number;
  /** Product of the dimensions after the reduced axis. */
  inner: number;
  kind: ReduceKind;
}

/**
 * Reduction along one axis of a tensor, viewed as `[outer, axis, inner]`.
 *
 * Any rank and any axis collapse to that view: `outer` is the product of the
 * dimensions before the axis, `inner` the product of those after. Reducing the
 * last axis is `inner = 1`, which is the shape rmsnorm and softmax already
 * reduce over. Output is `[outer, inner]`, i.e. `torch.sum(x, dim=d)` with
 * `keepdim=False`.
 *
 * The definition of correct for this op — deliberately the slowest, plainest
 * expression of it. Its job is to be obviously right, not fast.
 *
 * ## Conventions (rule 7: follow PyTorch)
 *
 * Measured against torch 2.10.0+cu128 rather than read off the docs:
 *
 * - **Empty axis, sum** → `0`. The additive identity. `torch.sum(torch.empty(0))`
 *   is `tensor(0.)`.
 * - **Empty axis, mean** → `NaN`. PyTorch computes `sum / count` without
 *   guarding, so an empty axis is `0 / 0`; `torch.mean(torch.empty(0))` is
 *   `tensor(nan)`. It is not `0`.
 * - **Empty axis, max / min** → *throws*. There is no finite identity that is
 *   also a valid answer, and PyTorch refuses rather than inventing one:
 *   `torch.amax(torch.empty(0), dim=0)` raises `IndexError: amax(): Expected
 *   reduction dim 0 to have non-zero size.` This reference throws to match. It
 *   deliberately does *not* return `-Infinity` — that would silently hand back
 *   a value PyTorch considers undefined, and the two libraries would then
 *   disagree on whether the caller has a bug.
 * - **mean's denominator** is the axis length, always. Not the number of finite
 *   elements — dividing by that is `nanmean`, a different function. So a single
 *   NaN anywhere on the axis makes the whole mean NaN.
 * - **NaN propagates** through all four here, as it does in PyTorch: `sum`,
 *   `mean`, `amax` and `amin` of `[1, nan, 3]` are all `nan`. Note that the
 *   WGSL kernel does *not* match on `max` / `min`; see `wgsl/kernel.wgsl` for
 *   the measurement and why it is left that way.
 */
export function reduce({ input, outer, axis, inner, kind }: ReduceArgs): Float32Array {
  if (input.length !== outer * axis * inner) {
    throw new Error(`expected ${outer * axis * inner} elements, got ${input.length}`);
  }
  if (axis === 0 && (kind === REDUCE.max || kind === REDUCE.min)) {
    // Matches PyTorch, which raises here rather than returning an identity.
    throw new Error(`reduce(): expected reduction axis to have non-zero size`);
  }

  const output = new Float32Array(outer * inner);
  for (let o = 0; o < outer; o += 1) {
    for (let i = 0; i < inner; i += 1) {
      let acc = kind === REDUCE.max ? -Infinity : kind === REDUCE.min ? Infinity : 0;
      for (let a = 0; a < axis; a += 1) {
        const value = input[(o * axis + a) * inner + i]!;
        if (kind === REDUCE.max) acc = Math.max(acc, value);
        else if (kind === REDUCE.min) acc = Math.min(acc, value);
        else acc += value;
      }
      output[o * inner + i] = kind === REDUCE.mean ? acc / axis : acc;
    }
  }
  return output;
}
