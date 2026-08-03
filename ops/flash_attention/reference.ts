import { defaultScale, resolveMask, type AttentionArgs } from "../attention/reference.js";

/**
 * Attention computed the way a fused kernel has to compute it: one tile of keys
 * at a time, with a running max and a running sum, so the `L x S` score matrix
 * is never held anywhere.
 *
 * The arithmetic result is the same function as `ops/attention` — same
 * conventions, same `scale` default, same `queryOffset` mask, all inherited
 * from `AttentionArgs` rather than restated here so the two cannot drift. What
 * differs is only *how*, and "how" is the entire reason this op exists (#19):
 * `attention` materialises `[B, H, L, S]` between its two dispatches, and at any
 * sequence length worth having a fused kernel for, that buffer is the largest
 * thing in the computation.
 *
 * So this reference is deliberately **not** the naive nest of loops — that one
 * already exists next door and is imported by the tests to check this against.
 * Its job is to be the plainest possible statement of the *online* recurrence,
 * because that recurrence is what the kernel implements and what can be got
 * subtly wrong:
 *
 *   m_new = max(m, max(tile))            the running maximum
 *   corr  = exp(m - m_new)               what everything accumulated so far is
 *                                        worth once the maximum moves
 *   l     = l * corr + sum(exp(s - m_new))
 *   acc   = acc * corr + sum(exp(s - m_new) * v)
 *   out   = acc / l                      normalised once, at the end
 *
 * `corr` is the whole trick and the thing tests have to be able to see: it is
 * exactly 1 whenever the maximum does not move, so a kernel that drops it is
 * correct on every input whose largest score happens to land in the first tile.
 *
 * Two departures from `ops/attention/reference.ts`, both deliberate:
 *
 * - **No `probs` output.** There is nothing to return; it is never formed. The
 *   comparison against the unfused implementation is on `output` alone.
 * - **The probabilities are never rounded to f32.** `attention` rounds them,
 *   because its two dispatches really do pass them through an f32 buffer, and a
 *   reference should model what its kernel does. Here they live in registers
 *   from the moment they are computed until they are summed, so rounding them
 *   would be modelling a step that does not happen. The residual difference
 *   between the two references is one f32 rounding per probability, and the
 *   tests measure it rather than assume it.
 *
 * Masked entries are `-Infinity`, as in `ops/attention`; the kernel uses
 * `-FLT_MAX` because WGSL cannot write an infinity, and the two agree for every
 * input this op accepts.
 *
 * `mask` — the additive attention bias, PyTorch's float `attn_mask` — is
 * `ops/attention`'s in every respect, arrives through `AttentionArgs`, and is
 * documented there rather than restated here. Two things it changes about this
 * file specifically:
 *
 * - **A fully masked row is now reachable** (#77), and this recurrence reaches
 *   it as `l = 0` and `acc = 0`, so the final normalisation would be `0 / 0`.
 *   That row returns zeros instead, matching `aten::_safe_softmax` and hence
 *   torch's SDPA. The condition is `l == 0` and it is exact: whichever key holds
 *   the final maximum contributed `exp(0) = 1`, so `l` is 0 if and only if no
 *   score was ever finite. It is also the reason `m` is not consulted — the
 *   kernel's running maximum starts at `-FLT_MAX` and cannot report `-Infinity`.
 * - **`mNew === -Infinity` skips the fold.** With every score so far masked
 *   there is nothing to rescale and `exp(-Inf - -Inf)` is NaN. The kernel does
 *   not need the equivalent, because its `m` starts finite.
 */
export interface FlashAttentionArgs extends AttentionArgs {
  /**
   * Keys folded in per step. Defaults to `DEFAULT_TILE`.
   *
   * The answer does not depend on it — that is a property worth having a knob
   * for, since it is the cheapest way to catch a recurrence that only works when
   * everything lands in one tile. The kernel's tile is fixed at compile time;
   * this one is a parameter so the tests can sweep it.
   */
  tile?: number;
}

export interface FlashAttentionResult {
  /** `[B, H, L, Dv]`, row-major. The only thing this op produces. */
  output: Float32Array;
}

/**
 * The tile the WGSL kernel uses, and this reference's default.
 *
 * Exported so `wgsl.test.ts` can choose sequence lengths that fall below it, on
 * it, and across it without writing 64 out a second time.
 */
export const DEFAULT_TILE = 64;

export function flashAttention(args: FlashAttentionArgs): FlashAttentionResult {
  const { q, k, v, B, H, L, S, D, Dv } = args;
  const causal = args.causal ?? false;
  const queryOffset = args.queryOffset ?? 0;
  const scale = args.scale ?? defaultScale(D);
  const tile = args.tile ?? DEFAULT_TILE;
  const { at: bias } = resolveMask(args, "flashAttention");

  const output = new Float32Array(B * H * L * Dv);

  for (let head = 0; head < B * H; head += 1) {
    const b = Math.floor(head / H);
    const h = head % H;
    const qHead = head * L * D;
    const kHead = head * S * D;
    const vHead = head * S * Dv;
    const oHead = head * L * Dv;

    for (let i = 0; i < L; i += 1) {
      let m = -Infinity; // running maximum over every score seen so far
      let l = 0; // running sum of exp(score - m), rescaled as m moves
      const acc = new Float64Array(Dv); // running sum of exp(score - m) * v

      for (let base = 0; base < S; base += tile) {
        const limit = Math.min(tile, S - base);

        // The one tile of scores that is ever alive. Everything below reads it
        // and then it is gone; nothing accumulates into an [L, S] anything.
        const scores = new Float64Array(limit);
        for (let s = 0; s < limit; s += 1) {
          const j = base + s;
          if (causal && j > i + queryOffset) {
            scores[s] = -Infinity;
            continue;
          }
          let dot = 0;
          for (let d = 0; d < D; d += 1) dot += q[qHead + i * D + d]! * k[kHead + j * D + d]!;
          scores[s] = dot * scale + bias(b, h, i, j);
        }

        let tileMax = -Infinity;
        for (let s = 0; s < limit; s += 1) tileMax = Math.max(tileMax, scores[s]!);

        // Fold the tile in. When the maximum does not move, `corr` is exp(0) = 1
        // and this is an ordinary running sum; when it does, every earlier term
        // is worth `corr` times what it was.
        const mNew = Math.max(m, tileMax);
        // Nothing finite has been seen yet, in this tile or any before it, so
        // there is nothing to rescale and nothing to add — and `exp(-Inf - -Inf)`
        // is NaN, which would poison `l` for the rest of the row. Reachable only
        // through `mask` (#77).
        if (mNew === -Infinity) continue;
        const corr = Math.exp(m - mNew);
        l *= corr;
        for (let c = 0; c < Dv; c += 1) acc[c] = acc[c]! * corr;
        for (let s = 0; s < limit; s += 1) {
          const p = Math.exp(scores[s]! - mNew);
          l += p;
          const j = base + s;
          for (let c = 0; c < Dv; c += 1) acc[c] = acc[c]! + p * v[vHead + j * Dv + c]!;
        }
        m = mNew;
      }

      // The single normalisation, after every key has been seen. `l >= 1`
      // whenever any score was finite: the key holding the final maximum
      // contributed exp(0). `l === 0` is the fully masked row, and it gets
      // zeros rather than 0/0 — see the header.
      if (l === 0) continue;
      for (let c = 0; c < Dv; c += 1) output[oHead + i * Dv + c] = acc[c]! / l;
    }
  }

  return { output };
}
