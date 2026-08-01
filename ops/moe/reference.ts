/**
 * MoE routing: router, dispatch, gather.
 *
 * The definition of correct for these three ops. Every backend is measured
 * against them, and they are deliberately the slowest, plainest expression of
 * the maths — their job is to be obviously right, not fast.
 *
 * Three functions rather than one because they are three dispatches, but they
 * are one op: a router that picks experts, a dispatch that lays tokens out in
 * per-expert runs so an expert's weights can be applied to a contiguous block,
 * and a gather that puts the results back in token order. Between dispatch and
 * gather the caller runs the expert FFNs; nothing here does.
 *
 * ```
 *   logits [T, E] ──router──▶ expert [T, k], gate [T, k]
 *   x      [T, D] ──dispatch(expert)──▶ buffer [E, C, D], pos [T, k]
 *                                  (caller runs the experts: buffer ──▶ y)
 *   y  [E, C, D] ──gather(expert, pos, gate)──▶ out [T, D]
 * ```
 *
 * `moeGather` is not `ops/gather`. That one is `index_select` over an embedding
 * table; this one is the combine half of a MoE layer, and the names collide only
 * because the issue that asked for it calls it gather. GShard calls the pair
 * "dispatch" and "combine", which is the less confusing naming and the reason
 * that word appears throughout the comments below.
 *
 * ## Conventions this follows, and why they had to be chosen
 *
 * Every one of these has more than one defensible answer in the literature, so
 * none of them is picked silently.
 *
 * ### Softmax before top-k, then renormalised over the k — Mixtral
 *
 * Matches `MixtralSparseMoeBlock` in HuggingFace `transformers`:
 *
 * ```python
 * routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
 * routing_weights, selected_experts = torch.topk(routing_weights, top_k, dim=-1)
 * routing_weights /= routing_weights.sum(dim=-1, keepdim=True)   # norm_topk_prob
 * ```
 *
 * Softmax-over-all then renormalise-over-k is algebraically identical to
 * softmax-over-the-k-selected-logits, because the `1/Z` of the full softmax
 * cancels. So the "before or after" argument only bites when the renormalisation
 * is *absent*, and there the two really differ: the Switch Transformer takes
 * `k = 1` and uses the full softmax probability as the gate, which is a number
 * strictly less than 1 and is what makes its gradient reach the router at all.
 * Renormalising at `k = 1` would make every gate exactly 1.0 and delete that.
 *
 * Both are therefore reachable, through `normalize`:
 *
 * - `normalize: true` — gates sum to 1 over the k. Mixtral, and Qwen2-MoE /
 *   DeepSeek-MoE with `norm_topk_prob=True`.
 * - `normalize: false` — gates are the full-softmax probabilities of the chosen
 *   experts and sum to less than 1. Switch Transformer, GShard, and
 *   `norm_topk_prob=False`.
 *
 * There is no default. A caller who does not know which one they want has a
 * bug either way, and half of them would get the wrong one from any default
 * this could pick.
 *
 * ### Ties in top-k go to the lower expert index
 *
 * `torch.topk` explicitly does not guarantee anything here — the documentation
 * says the order of equal elements is undefined. A kernel cannot leave it
 * undefined: on a GPU "whichever thread got there first" changes with the
 * scheduler, and two runs of the same model would route the same token to
 * different experts. So this is specified rather than inherited: **among equal
 * logits the lower expert index ranks first**, which is also what CPU
 * `torch.topk` happens to produce today.
 *
 * Exact ties are not a curiosity here. A freshly initialised router, a padded
 * batch, or any masking scheme that writes a constant into several logits
 * produces them immediately.
 *
 * ### Capacity overflow drops by rank first, then by token index — GShard
 *
 * An expert has room for `C` tokens. When more than `C` assignments name it,
 * the ones that fit are chosen in this order:
 *
 * 1. lower rank `r` first — a token's first-choice expert beats another token's
 *    second choice;
 * 2. then lower token index.
 *
 * Everything past `C` is **dropped**: it is copied nowhere, the expert never
 * sees it, and it contributes nothing to that token's output. A token whose
 * every choice was dropped comes back as a row of zeros, which in a transformer
 * block means the residual passes through unchanged.
 *
 * This is GShard's rule ("the top-1 expert has priority over the top-2"), the
 * same one the Switch Transformer and fairseq's `top2gating` implement as a
 * `cumsum` per rank. The alternative — flattening `(token, rank)` in row-major
 * order, so token 0's second choice outranks token 1's first — is what a
 * stable sort by expert id gives you, and it is what Megablocks does; but
 * Megablocks is dropless, so it never has to answer this question at all.
 *
 * Dropping by *arrival* — an atomic counter, first thread to reach the expert
 * wins — is the one option that is not available, for the same reason
 * `ops/scatter` refuses "last write wins": nothing orders the threads, so the
 * set of dropped tokens would vary run to run on the same input and the same
 * device.
 *
 * ### The gate is applied in gather, once
 *
 * Not in dispatch, and not in both. Weighting on the way in would put the gate
 * inside the expert FFN, which is wrong for any non-linear expert, and applying
 * it at both ends squares it — a result that stays plausible (gates are near 1
 * when normalised and `k` is small) while being wrong everywhere. Dispatch
 * copies token rows verbatim; `moeGather` multiplies exactly once.
 *
 * ### Nothing divides by a token count
 *
 * An expert receiving zero tokens is ordinary, not exceptional — with 64
 * experts and a short sequence most of them get nothing — so no step here
 * averages over an expert's tokens. The only divisions are by the softmax
 * denominator, which contains `exp(0) = 1` and so is at least 1, and by the sum
 * of the k gates, which contains the largest of them and so is positive.
 */

/** Which experts a token was routed to, and with what weight. */
export interface RouterResult {
  /** `[T, k]` expert ids, best first. `-1` where no expert was available. */
  expert: Int32Array;
  /** `[T, k]` gate weights, aligned with `expert`. `0` where `expert` is -1. */
  gate: Float32Array;
}

export interface RouterArgs {
  /** Row-major `[T, E]` router logits. */
  logits: Float32Array;
  /** Tokens. */
  T: number;
  /** Experts. */
  E: number;
  /** Experts per token. `k > E` fills the surplus slots with `-1`. */
  k: number;
  /** Renormalise the k gates to sum to 1 (Mixtral) or leave them as the full
   * softmax probabilities (Switch / GShard). See the module comment; there is
   * no default because both are in use. */
  normalize: boolean;
}

export function router({ logits, T, E, k, normalize }: RouterArgs): RouterResult {
  const expert = new Int32Array(T * k).fill(-1);
  const gate = new Float32Array(T * k);

  for (let t = 0; t < T; t += 1) {
    const row = t * E;

    // Softmax over all E experts, max-subtracted. exp(800) is Infinity in f32
    // and takes the row to NaN; router logits are not small.
    let max = -Infinity;
    for (let e = 0; e < E; e += 1) max = Math.max(max, logits[row + e]!);
    let denominator = 0;
    for (let e = 0; e < E; e += 1) denominator += Math.exp(logits[row + e]! - max);

    // Top-k over the *logits*, not over the probabilities. Same ordering —
    // softmax is monotonic — but exp() in f32 can round two distinct logits to
    // the same probability, which would turn a well-defined order into a tie
    // and hand it to the index rule below for no reason.
    //
    // Selection sort, one pass per rank, keeping the previous winner as the
    // ceiling. Ranking is by (logit descending, index ascending), so the
    // ceiling is a pair rather than a value.
    let ceilingValue = Infinity;
    let ceilingIndex = -1;
    for (let r = 0; r < k; r += 1) {
      let bestValue = 0;
      let bestIndex = -1;
      for (let e = 0; e < E; e += 1) {
        const value = logits[row + e]!;
        // Strictly below the previous winner in the (value, index) order.
        const eligible = value < ceilingValue || (value === ceilingValue && e > ceilingIndex);
        if (!eligible) continue;
        // `>` and not `>=`, scanning e upwards: on a tie the first one seen
        // wins, which is the lowest index.
        if (bestIndex < 0 || value > bestValue) {
          bestValue = value;
          bestIndex = e;
        }
      }
      // Only when k > E: every expert has already been taken. The slot stays
      // -1 / 0 and dispatch drops it.
      if (bestIndex < 0) break;
      expert[t * k + r] = bestIndex;
      gate[t * k + r] = Math.exp(bestValue - max) / denominator;
      ceilingValue = bestValue;
      ceilingIndex = bestIndex;
    }

    if (normalize) {
      let sum = 0;
      for (let r = 0; r < k; r += 1) sum += gate[t * k + r]!;
      // Positive whenever any expert was chosen: it contains the row maximum,
      // whose probability is at least 1/E.
      for (let r = 0; r < k; r += 1) gate[t * k + r] = gate[t * k + r]! / sum;
    }
  }

  return { expert, gate };
}

/** Where each token landed, and the per-expert buffer it landed in. */
export interface DispatchResult {
  /**
   * `[E, C, D]`. Capacity slots that no token reached are **left as zero**, and
   * `moeGather` never reads them — `pos` cannot point at one. A backend is free
   * to leave whatever was already in the buffer there.
   */
  buffer: Float32Array;
  /** `[T, k]` position within its expert's run, or `-1` if dropped. */
  pos: Int32Array;
}

export interface DispatchArgs {
  /** Row-major `[T, D]` token features. */
  x: Float32Array;
  /** `[T, k]` expert ids from the router. Outside `[0, E)` is dropped. */
  expert: Int32Array;
  T: number;
  k: number;
  E: number;
  /** Capacity: rows each expert has room for. Usually
   * `ceil(capacity_factor * T * k / E)`; the surplus is dropped. */
  C: number;
  D: number;
}

export function moeDispatch({ x, expert, T, k, E, C, D }: DispatchArgs): DispatchResult {
  const buffer = new Float32Array(E * C * D);
  const pos = new Int32Array(T * k).fill(-1);

  // Rank-major, then token order — GShard's priority rule, spelled out as the
  // loop nest that produces it rather than hidden in a cumsum. `filled` is the
  // running count per expert, which is exactly the position the next token
  // takes.
  const filled = new Int32Array(E);
  for (let r = 0; r < k; r += 1) {
    for (let t = 0; t < T; t += 1) {
      const e = expert[t * k + r]!;
      // Out of range is dropped, not clamped: clamping routes a token to an
      // expert it did not choose, which is invisible downstream.
      if (e < 0 || e >= E) continue;
      const slot = filled[e]!;
      // Over capacity. The count still advances so that the rule stays "the
      // first C in this order", not "whoever came after the drop".
      filled[e] = slot + 1;
      if (slot >= C) continue;
      pos[t * k + r] = slot;
      for (let d = 0; d < D; d += 1) {
        // Verbatim. The gate is applied in moeGather, once.
        buffer[(e * C + slot) * D + d] = x[t * D + d]!;
      }
    }
  }

  return { buffer, pos };
}

export interface MoeGatherArgs {
  /** `[E, C, D]` expert outputs, in the layout `moeDispatch` produced. */
  y: Float32Array;
  /** `[T, k]` expert ids. */
  expert: Int32Array;
  /** `[T, k]` positions from `moeDispatch`. `-1` contributes nothing. */
  pos: Int32Array;
  /** `[T, k]` gate weights. Applied here, and only here. */
  gate: Float32Array;
  T: number;
  k: number;
  E: number;
  C: number;
  D: number;
}

/**
 * `out[t, :] = Σ_r gate[t, r] * y[expert[t, r], pos[t, r], :]`
 *
 * A slot with `pos < 0`, or an expert outside `[0, E)`, contributes nothing —
 * so a token every one of whose choices was dropped comes back as zeros rather
 * than as a scaled copy of somebody else's row.
 */
export function moeGather({ y, expert, pos, gate, T, k, E, C, D }: MoeGatherArgs): Float32Array {
  const out = new Float32Array(T * D);
  for (let t = 0; t < T; t += 1) {
    for (let r = 0; r < k; r += 1) {
      const slot = pos[t * k + r]!;
      const e = expert[t * k + r]!;
      if (slot < 0 || slot >= C || e < 0 || e >= E) continue;
      const weight = gate[t * k + r]!;
      for (let d = 0; d < D; d += 1) {
        out[t * D + d] = out[t * D + d]! + weight * y[(e * C + slot) * D + d]!;
      }
    }
  }
  return out;
}
