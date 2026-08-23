/**
 * axpy: `out[i] = y[i] + a * x[i]`, with `a` a **scalar**.
 *
 * The definition of correct for this op. Every backend is measured against it,
 * and it is deliberately the slowest, plainest expression of the maths.
 *
 * ## Why an op rather than two calls to `ops/elementwise`
 *
 * A rectified-flow sampler's whole scheduler step is this line: `latent +=
 * dt * velocity`, once per diffusion step, over a latent of a few hundred
 * thousand elements. `ops/elementwise` cannot express it — it takes two
 * **equally sized** arrays and throws otherwise, so a scalar coefficient has
 * to be materialised as a full-length buffer of copies of `a` and then run
 * through two dispatches:
 *
 * ```
 * scaled = elementwise(velocity, fill(dt, N), multiply)   // dispatch 1
 * latent = elementwise(latent, scaled, add)               // dispatch 2
 * ```
 *
 * That is 5N floats of traffic (read velocity, read the fill buffer, write
 * scaled, read scaled, read latent, write latent — 6N, minus whatever the
 * cache keeps) against this op's 3N, plus an N-element buffer of a repeated
 * constant that has to be uploaded or filled before the first dispatch can
 * run. On a bandwidth-bound elementwise kernel that ratio *is* the runtime.
 * Row broadcast — `[S,D] + [D]` — is a different problem and is issue #150;
 * this op's only broadcast is the scalar.
 *
 * ## Which convention this follows
 *
 * Two live conventions, and they disagree about operand order and about
 * in-placeness, so this does not pick quietly (rule 7):
 *
 * - **BLAS `saxpy`** is `y := a*x + y` — in-place in `y`, and the name this op
 *   borrows.
 * - **PyTorch `torch.add(input, other, *, alpha)`** is `input + alpha * other`,
 *   out-of-place, with `Tensor.add_` as the separate in-place spelling.
 *
 * The arithmetic follows **PyTorch `torch.add`**: `y` is `input`, `x` is
 * `other`, `a` is `alpha`. That is not merely a naming choice — see the
 * rounding section, where the two conventions genuinely differ in the bits
 * they return.
 *
 * This function is **out-of-place**, and BLAS's in-place spelling lives only
 * in the kernel (`wgsl/inplace.wgsl`), not here. A JS in-place reference would
 * be a one-line alias with nothing to check: the interesting question about
 * in-place is a GPU question (can one buffer be read and written by the same
 * dispatch, and does that survive `harness/resident.ts`'s bind-group reuse),
 * and it is answered where it can be observed, in `ops/axpy/inplace.wgsl.test.ts`.
 *
 * ## Rounding: this rounds **once**, and that is a decision
 *
 * `y[i] + a * x[i]` here is computed in f64 and rounded exactly once, on the
 * store into the `Float32Array`. The two-dispatch spelling above rounds
 * **twice** — once when `scaled` is written to memory as f32, once on the add
 * — and the two answers are not always the same number.
 *
 * They can differ by everything, not by an ulp. At `a = f32(0.1)`, `x = 3`,
 * `y = -f32(0.3)`: the exact product 0.3000000044703484 rounds to f32(0.3), so
 * multiply-then-add cancels to exactly **0**, while a single rounding gives
 * **-2^-27**. `ops/axpy/reference.test.ts` pins that case.
 *
 * Which one is right is not this library's call to make (rule 7), and it was
 * measured rather than assumed: `torch 2.10.0+cu128` returns the **single-
 * rounded** value, on CPU and on CUDA alike — `torch.add` contracts into an
 * FMA. Over 100,000 random f32 pairs at `a = 0.37`, this f64-then-store
 * reference matched `torch.add`'s output bit for bit on every element; the
 * multiply-then-add spelling differed on 20,732 of them (max abs 2.4e-7).
 *
 * So the fused kernel is not merely faster than the two dispatches it
 * replaces, it is the spelling that agrees with PyTorch — provided the GPU
 * contracts too. Whether this device's WGSL compiler does is measured in
 * `ops/axpy/wgsl.test.ts`, which runs both paths on the GPU and compares them
 * element by element rather than trusting either.
 *
 * ## `a` is an f64 here and an f32 on the GPU
 *
 * `a` is a plain JS number, so a value like `0.37` is the f64 nearest it,
 * while the uniform the kernel reads holds the f32 nearest it — a relative
 * difference of up to 6e-8 in every output element, which is inside the
 * default tolerance but not inside a bit-for-bit comparison. The kernel tests
 * pass `Math.fround(a)` for exactly that reason.
 */
export interface AxpyArgs {
  /** `[N]`: the vector scaled by `a` — `other` in `torch.add`, the velocity field in a sampler step. */
  x: Float32Array;
  /** `[N]`: the vector added to — `input` in `torch.add`, the latent in a sampler step. Not written; see the doc above. */
  y: Float32Array;
  /** The scalar coefficient — `alpha` in `torch.add`, `dt` in a sampler step. Changes every step, which is what shapes the kernel's uniform. */
  a: number;
}

export function axpy({ x, y, a }: AxpyArgs): Float32Array {
  if (x.length !== y.length) throw new Error(`length mismatch: ${x.length} vs ${y.length}`);
  const output = new Float32Array(x.length);
  for (let i = 0; i < x.length; i += 1) {
    output[i] = y[i]! + a * x[i]!;
  }
  return output;
}
