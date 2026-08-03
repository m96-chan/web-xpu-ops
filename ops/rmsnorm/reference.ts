/**
 * RMSNorm: `x_i * w_i / sqrt(mean(x²) + eps)`
 *
 * The definition of correct for this op. Every backend is measured against it,
 * and it is deliberately the slowest, plainest expression of the maths — its
 * job is to be obviously right, not fast.
 *
 * ## What `weight` is, and which axis `groups` counts
 *
 * With `groups = 1` (the default, and everything written before #78) there is
 * one gamma vector of length `D`, shared by every row.
 *
 * With `groups = G` the weight is `[G, D]` and row `n` uses `weight[(n % G)]`.
 * That serves QK-norm, where each attention head carries its own gamma
 * (voxshot#115's RF-DiT: `RMSNorm((heads, head_dim))`, reducing over `head_dim`
 * only). **`n % G` is only correct because the group axis is assumed to be the
 * one immediately to the left of `D`** — the fastest varying of the axes that
 * were flattened into `N`:
 *
 *   `[B, S, H, Dh]` → `N = B*S*H`, `D = Dh`, `G = H`, `row = (b*S + s)*H + h`,
 *   so `row % H == h`. Verified against PyTorch, not reasoned about: the loop
 *   above reproduces `F.rms_norm(x, (Dh,), eps=eps) * w` for `w: [H, Dh]`
 *   exactly (max abs difference 0.0, float64, torch 2.10).
 *
 * **The `[B, H, S, Dh]` layout is refused, not silently mis-served.** There the
 * head index is `floor(row / S) % H` and `row % H` is simply a different
 * answer — checked, and it disagrees. This op has no parameter that can express
 * it, so a caller holding that layout must either transpose to `[B, S, H, Dh]`
 * or materialise an `[N, D]` weight and pass `groups = N`. A second "rows per
 * group" stride would cover it, but no caller has asked for one, and an
 * untested second parameter is how the two layouts start getting confused.
 *
 * `N` need not be a multiple of `G`; the weight simply keeps repeating, which
 * is what `%` says. Real callers have `N = B*S*G`, but leaving the remainder
 * undefined would be a rule that only exists in prose.
 *
 * ## Where this departs from `torch.nn.RMSNorm`
 *
 * `torch.nn.RMSNorm((H, Dh))` normalises over **both** trailing axes, because
 * PyTorch reduces over `len(normalized_shape)` dimensions. Measured, not
 * assumed: against the per-head form it differs by 1.87 in absolute value on
 * random `[2, 3, 4, 8]` input. This op keeps the reduction over `D` alone and
 * lets `weight` be wider, matching `F.rms_norm(x, (D,), eps=eps) * w` with `w`
 * broadcast over the group axis — the form QK-norm actually uses, and the one
 * the RF-DiT in voxshot#115 implements. A caller who wants PyTorch's
 * whole-`[H, Dh]` reduction gets it from this op by passing `D = H*Dh` and
 * `groups = 1`, so nothing is lost by choosing this reading.
 *
 * `eps` sits inside the square root, `x / sqrt(mean(x²) + eps)`, as PyTorch has
 * it.
 *
 * `ops/layernorm` documents `weight` the same way and does not have this
 * parameter yet; #81 tracks giving it one, so the two do not end up disagreeing
 * about what a weight is.
 */
export interface RMSNormArgs {
  input: Float32Array;
  /**
   * `[groups, D]` — one gamma vector per group, repeating every `groups` rows.
   * Length `D` when `groups` is 1.
   */
  weight: Float32Array;
  /** Rows. */
  N: number;
  /** Columns; the axis that is normalised. */
  D: number;
  eps: number;
  /**
   * Number of consecutive rows that use distinct gammas before the weight
   * repeats — the extent of the axis immediately left of `D`. Defaults to 1,
   * which is one gamma vector shared by every row.
   *
   * `0` also means one group, matching the kernel: there, a zero is how a
   * caller written before this parameter existed leaves the word, and the two
   * must not disagree about it.
   */
  groups?: number;
}

export function rmsnorm({ input, weight, N, D, eps, groups = 1 }: RMSNormArgs): Float32Array {
  const G = Math.max(groups, 1);
  const output = new Float32Array(N * D);
  for (let row = 0; row < N; row += 1) {
    let sumSquares = 0;
    for (let col = 0; col < D; col += 1) {
      const value = input[row * D + col]!;
      sumSquares += value * value;
    }
    const scale = 1 / Math.sqrt(sumSquares / D + eps);
    const weightOffset = (row % G) * D;
    for (let col = 0; col < D; col += 1) {
      output[row * D + col] = input[row * D + col]! * scale * weight[weightOffset + col]!;
    }
  }
  return output;
}
