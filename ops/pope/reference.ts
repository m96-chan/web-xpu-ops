/**
 * PoPE — Legendre Orthogonal Polynomial Position Encoding.
 *
 * Aggarwal, "PoPE: Legendre Orthogonal Polynomials Based Position Encoding for
 * Large Language Models", arXiv:2405.04585v1 (29 Apr 2024).
 *
 * PoPE is an **absolute** position encoding, so unlike `rope` (which rotates Q
 * and K) and unlike `alibi` (which biases the scores) it produces a table that
 * is added to the token embeddings, exactly where Vaswani's sinusoidal encoding
 * goes. This op is that table. Adding it is `elementwise.add`: the table and
 * the embeddings are both `[N, dModel]`, so there is nothing for a fused kernel
 * to save and no reason to write one.
 *
 * ## The definition, and where it is under-specified
 *
 * Section 3.2, Equation (14), verbatim:
 *
 *     PE(pos, i) = P_pos(x_i)
 *
 *     "Where, x_i are equal interval samples in [-1, 1] with interval
 *      length = 2i/d_model."
 *
 * So the polynomial **order** is the token position and the **argument** is the
 * feature index swept across the polynomial's domain — the opposite assignment
 * from the sinusoidal scheme, where the feature index picks a frequency and the
 * position is the argument.
 *
 * Two readings had to be pinned down, and both are recorded here rather than
 * chosen silently:
 *
 *   1. `x_i`. Read as `x_i = -1 + 2i/dModel` for `i` in `[0, dModel)`, which is
 *      what "equal interval samples ... with interval length 2i/d_model" means
 *      if the intervals are to be equal at all. The samples therefore start at
 *      `-1` and stop one step short of `+1`.
 *   2. Whether `pos` starts at 0 or 1. The paper does not say. It matters: at
 *      `pos = 0` the order-0 polynomial is the constant 1, so the first token
 *      would carry no positional information at all, while Section 2.1 indexes
 *      the sequence as `{w_i}` for `i = 1..N`. Rather than pick, `posOffset` is
 *      a required argument — as it is in `rope`, where it also serves KV-cache
 *      continuation. Pass `1` for the paper's 1-based indexing; pass `0` for
 *      the 0-based convention the rest of this repository uses.
 *
 * There is no PyTorch equivalent to defer to (rule 7): PoPE is not in torch,
 * and `scipy.special.eval_legendre` is only the polynomial, not the encoding.
 * The polynomial itself is standard and unambiguous, and is evaluated by the
 * three-term recurrence the paper quotes as Equation (15).
 */

/**
 * Legendre polynomial `P_n(x)` by the three-term recurrence, which is
 * Equation (15) of the paper rearranged:
 *
 *     (2n + 1) x P_n(x) = n P_{n-1}(x) + (n + 1) P_{n+1}(x)
 *     =>  P_{n+1}(x) = ((2n + 1) x P_n(x) - n P_{n-1}(x)) / (n + 1)
 *
 * Rodrigues' formula (Equation 12) is the compact definition but is not a way
 * to compute anything; the explicit coefficient sum overflows and cancels for
 * even modest `n`. The recurrence stays inside `[-1, 1]` for every term because
 * `|P_n(x)| <= 1` on the domain, which is the reason it is the standard method
 * and the reason an f32 kernel can follow it at all.
 */
export function legendre(n: number, x: number): number {
  if (n === 0) return 1;
  let previous = 1; // P_0
  let current = x; // P_1
  for (let k = 1; k < n; k += 1) {
    const next = ((2 * k + 1) * x * current - k * previous) / (k + 1);
    previous = current;
    current = next;
  }
  return current;
}

export interface PoPEArgs {
  /** Tokens. */
  N: number;
  /** Model width; also the number of samples the polynomial domain is cut into. */
  dModel: number;
  /**
   * Position of the first row. `1` reproduces the paper's 1-based token
   * indexing; `0` matches `rope` and the rest of this repository, and also
   * serves KV-cache continuation. Required rather than defaulted because the
   * paper does not state which it means.
   */
  posOffset: number;
}

/** The `[N, dModel]` encoding table, row-major. */
export function pope({ N, dModel, posOffset }: PoPEArgs): Float32Array {
  const output = new Float32Array(N * dModel);
  for (let token = 0; token < N; token += 1) {
    const order = token + posOffset;
    for (let i = 0; i < dModel; i += 1) {
      const x = -1 + (2 * i) / dModel;
      output[token * dModel + i] = legendre(order, x);
    }
  }
  return output;
}
