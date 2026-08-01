/**
 * ALiBi — Attention with Linear Biases.
 *
 * Press, Smith and Lewis, "Train Short, Test Long: Attention with Linear Biases
 * Enables Input Length Extrapolation", ICLR 2022 (arXiv:2108.12409).
 *
 * Unlike `rope`, nothing here rotates Q or K. ALiBi leaves the projections
 * alone and adds a per-head linear penalty to the attention **scores**, so it
 * runs on a different tensor at a different point in the layer and cannot share
 * `rope`'s shape at all.
 *
 * Two things live in this file, deliberately apart:
 *
 *   `alibiSlopes` — the per-head geometric sequence `m`. This is the part that
 *                   implementations quietly disagree on, so it is written as
 *                   the paper's own algorithm and tested on its own.
 *   `alibi`       — the application, `scores + m * (j - i)`.
 */

/**
 * The per-head slopes, transcribed from the reference implementation published
 * with the paper (`get_slopes` in `ofirpress/attention_with_linear_biases`, and
 * identical to `build_alibi_tensor` in HuggingFace `transformers`' BLOOM):
 *
 *     def get_slopes(n):
 *         def get_slopes_power_of_2(n):
 *             start = (2**(-2**-(math.log2(n)-3)))
 *             ratio = start
 *             return [start*ratio**i for i in range(n)]
 *         if math.log2(n).is_integer():
 *             return get_slopes_power_of_2(n)
 *         else:
 *             closest_power_of_2 = 2**math.floor(math.log2(n))
 *             return (get_slopes_power_of_2(closest_power_of_2)
 *                     + get_slopes(2*closest_power_of_2)[0::2][:n-closest_power_of_2])
 *
 * The two branches are the whole reason this is tested separately. For a power
 * of two the slopes are the clean geometric run `2^-8/n ... 2^-8`. For anything
 * else the paper does **not** simply interpolate: it takes the full run for the
 * closest smaller power of two and then appends every other slope of the run
 * one size up. The appended ones are *larger* than the ones before them, so the
 * sequence is not monotonic — an implementation that sorts, interleaves, or
 * truncates a longer run instead will pass a power-of-two test and be wrong for
 * every other head count. 12 heads is the shape that catches it.
 *
 * Kept as the recursion rather than the closed form on purpose: this is the
 * definition, and the closed form the kernel uses (`exp2` of a rational
 * exponent) is a derivation that has to be checked against something.
 */
export function alibiSlopes(numHeads: number): Float64Array {
  if (!Number.isInteger(numHeads) || numHeads < 1) {
    throw new Error(`numHeads must be a positive integer, got ${numHeads}`);
  }
  return Float64Array.from(getSlopes(numHeads));
}

function getSlopesPowerOfTwo(n: number): number[] {
  const start = 2 ** -(2 ** -(Math.log2(n) - 3));
  const ratio = start;
  return Array.from({ length: n }, (_, i) => start * ratio ** i);
}

function getSlopes(n: number): number[] {
  if (Number.isInteger(Math.log2(n))) return getSlopesPowerOfTwo(n);
  const closest = 2 ** Math.floor(Math.log2(n));
  const extra = getSlopes(2 * closest)
    .filter((_, i) => i % 2 === 0)
    .slice(0, n - closest);
  return [...getSlopesPowerOfTwo(closest), ...extra];
}

export interface ALiBiArgs {
  /** Attention scores, `[numHeads, M, N]` row-major. */
  scores: Float32Array;
  numHeads: number;
  /** Query rows. */
  M: number;
  /** Key columns. */
  N: number;
  /**
   * Position of the first query row, for KV-cache continuation — the same need
   * `rope`'s `posOffset` serves. While decoding, `M` is 1 and its true position
   * is however many tokens came before it.
   */
  posOffset: number;
  /** Defaults to `alibiSlopes(numHeads)`; injectable so the two halves can be tested apart. */
  slopes?: ArrayLike<number>;
}

/**
 * Adds the linear bias to a score matrix.
 *
 *     out[h, i, j] = scores[h, i, j] + m_h * (j - (i + posOffset))
 *
 * The sign and the origin follow the paper, which writes the bias applied to
 * query `i` as `m * [-(i-1), ..., -2, -1, 0]` over the keys — that is, zero on
 * the diagonal and growing more negative the further back the key sits. So the
 * quantity is the **relative** distance `j - i`, not the key's absolute index.
 *
 * HuggingFace's BLOOM implementation uses `m * j` instead. The two differ by
 * `m * i`, which is constant along a row and therefore vanishes in the softmax
 * that follows, so both train the same model — but they are not the same
 * tensor, and this op hands back the tensor. The relative form is the one the
 * paper defines, so that is the one here.
 *
 * Masking is **not** part of this op. ALiBi is defined for causal attention,
 * where `j > i` never survives to the softmax; those entries are computed here
 * as the plain linear extension (positive, since the key is ahead of the
 * query), and the caller's mask is expected to remove them. Writing `-inf`
 * here instead would fold a masking policy into a bias op and make the result
 * unusable for the bidirectional variants that mask differently.
 */
export function alibi({ scores, numHeads, M, N, posOffset, slopes }: ALiBiArgs): Float32Array {
  const expected = numHeads * M * N;
  if (scores.length !== expected) {
    throw new Error(`scores: expected ${expected} elements, got ${scores.length}`);
  }
  const m = slopes ?? alibiSlopes(numHeads);
  if (m.length !== numHeads) {
    throw new Error(`slopes: expected ${numHeads} elements, got ${m.length}`);
  }

  const output = new Float32Array(expected);
  for (let head = 0; head < numHeads; head += 1) {
    for (let query = 0; query < M; query += 1) {
      for (let key = 0; key < N; key += 1) {
        const index = (head * M + query) * N + key;
        output[index] = scores[index]! + m[head]! * (key - (query + posOffset));
      }
    }
  }
  return output;
}
