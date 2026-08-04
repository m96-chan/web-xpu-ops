/**
 * GroupNorm: `(x - mean_g) / sqrt(var_g + eps) * w_c + b_c`, where the
 * statistics are pooled over a **group of channels** and the affine transform
 * is per channel.
 *
 * The definition of correct for this op. Every backend is measured against it,
 * and it is deliberately the slowest, plainest expression of the maths — its
 * job is to be obviously right, not fast.
 *
 * Conventions follow `torch.nn.GroupNorm` / `torch.nn.functional.group_norm`,
 * verified against PyTorch 2.10 in float64 rather than read off the
 * documentation:
 *
 *   - The variance is **biased** (divided by the element count, not by count
 *     minus one). Against torch's own answer: the biased form matched to
 *     4.44e-16, the unbiased form was out by 1.62e-1.
 *   - `eps` sits **inside** the square root, `sqrt(var + eps)`. Placing it
 *     outside moved the answer by 4.32e-6 — small, but four thousand f32 ulps
 *     and not noise.
 *   - `eps` defaults to **1e-5** in `torch.nn.GroupNorm`. Required here rather
 *     than defaulted, so a caller states the number their checkpoint trained
 *     with instead of inheriting one framework's habit.
 *   - `C % G != 0` **throws**, as torch does ("Expected number of channels in
 *     input to be divisible by num_groups"). Distributing a remainder would be
 *     a different op that happens to run.
 *
 * ## What is normalised, and why that is the whole point
 *
 * For `[N, C, L]` the channels split into `G` contiguous groups of `C / G`, and
 * each of the `N × G` groups reduces `(C / G) × L` values to one mean and one
 * variance. The reduction covers the group's channels **and** every spatial
 * element in them.
 *
 * That is what separates this from what the library already had. `layernorm`
 * over `[N·C, L]` gives one mean per channel; GroupNorm pools `C / G` channels
 * into each. Same input, same output shape, no error either way, different
 * numbers — which is why this is an op rather than something a caller
 * assembles. `G = 1` reduces the whole sample; `G = C` is per-channel
 * (InstanceNorm), and is exactly `layernorm` over `[N·C, L]`.
 *
 * `weight` and `bias` are `[C]` — **per channel, not per group**. Channels
 * inside one group share statistics and do not share their affine transform,
 * and a kernel that indexed the weight by group would be caught only by a
 * weight that varies inside a group.
 *
 * ## The layout is `[N, C, L]`, and the arithmetic only works for that one
 *
 * Contiguous, channel-major — what a 1-D codec stage carries, and what
 * `ops/snake` already assumes. Stated because it is not deducible from the
 * arguments: a caller holding `[N, L, C]` gets a well-formed wrong answer.
 *
 * A convenience of that layout, used by the kernel rather than by this file:
 * a group's elements are one contiguous span, `(C / G) × L` long, so the
 * reduction is over a run rather than a stride.
 *
 * ## Why the variance is computed in two passes
 *
 * Same reason as `layernorm`, and the same trade. `var = E[x²] - E[x]²` is one
 * pass, and it subtracts two large near-equal numbers whenever the mean is
 * large relative to the spread — at `x ≈ 8192 ± 4` in f32 the squares land past
 * 2^26, where the representable values step by 8, and the answer is noise. Two
 * passes over data already in cache never subtracts two large numbers at all.
 */
export interface GroupNormArgs {
  /** `[N, C, L]`, contiguous. */
  input: Float32Array;
  /** `[C]`: one value per channel, shared across batch and length. */
  weight: Float32Array;
  /** `[C]`: one value per channel, shared across batch and length. */
  bias: Float32Array;
  /** Batch. */
  N: number;
  /** Channels. Must be divisible by {@link GroupNormArgs.G}. */
  C: number;
  /** Length; every trailing spatial axis flattens into this one. */
  L: number;
  /** Groups. The statistics are pooled over `C / G` channels at a time. */
  G: number;
  /** Inside the square root. `torch.nn.GroupNorm` defaults it to 1e-5. */
  eps: number;
}

export function groupNorm({ input, weight, bias, N, C, L, G, eps }: GroupNormArgs): Float32Array {
  if (G <= 0 || C % G !== 0) {
    throw new Error(
      `group_norm: expected number of channels (${C}) to be divisible by num_groups (${G})`,
    );
  }

  const output = new Float32Array(N * C * L);
  const channelsPerGroup = C / G;
  const count = channelsPerGroup * L;

  for (let n = 0; n < N; n += 1) {
    for (let g = 0; g < G; g += 1) {
      // A group is one contiguous run: channel `g * channelsPerGroup` onwards,
      // every length element of each.
      const start = (n * C + g * channelsPerGroup) * L;

      let sum = 0;
      for (let i = 0; i < count; i += 1) {
        sum += input[start + i]!;
      }
      const mean = sum / count;

      let sumSquaredDeviations = 0;
      for (let i = 0; i < count; i += 1) {
        const deviation = input[start + i]! - mean;
        sumSquaredDeviations += deviation * deviation;
      }
      const variance = sumSquaredDeviations / count;

      const scale = 1 / Math.sqrt(variance + eps);
      for (let i = 0; i < count; i += 1) {
        // The affine transform is per channel, so the channel has to come back
        // out of the flat offset rather than being the group.
        const channel = g * channelsPerGroup + Math.floor(i / L);
        output[start + i] = (input[start + i]! - mean) * scale * weight[channel]! + bias[channel]!;
      }
    }
  }
  return output;
}
