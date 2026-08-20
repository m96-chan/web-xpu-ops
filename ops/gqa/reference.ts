import { resolveMask, type MaskShape } from "../attention/reference.js";

/**
 * Grouped-query and multi-query attention: SDPA where several query heads read
 * the same K and V.
 *
 * One op covers both. MQA is GQA with `kvHeads = 1`, and plain MHA is GQA with
 * `kvHeads = H`, so there is nothing to switch on — the KV head index is a
 * function of the query head index and the two head counts, and the three named
 * configurations are three points on it. Writing MQA separately would mean two
 * implementations of the same indexing, one of which is exercised by fewer
 * tests.
 *
 * ## The head mapping
 *
 * Query head `h` reads KV head `h / (H / kvHeads)`, integer division: the query
 * heads are cut into `kvHeads` **contiguous** groups of `H / kvHeads`.
 *
 * That is the whole op, and it has an equally plausible alternative — strided
 * groups, `h % kvHeads` — which is wrong, and wrong in the way that is hardest
 * to notice: it agrees with the contiguous mapping at both ends. At
 * `kvHeads = H` both are the identity. At `kvHeads = 1` both are constant 0. A
 * GQA implementation can be tested at MHA and MQA, pass, and still be wrong at
 * every configuration anyone actually ships.
 *
 * Contiguous is what PyTorch does, and it was measured rather than read
 * (rule 7). On torch 2.10, `scaled_dot_product_attention(..., enable_gqa=True)`
 * at `H=6, kvHeads=3` reproduces `k.repeat_interleave(2, dim=1)` exactly (max
 * abs difference 0.0) and differs from `k.repeat(1, 2, 1, 1)` by 2.86 — which is
 * to say the two conventions are not close, and picking the wrong one is not a
 * tolerance question.
 *
 * ## `H % kvHeads != 0` is an error
 *
 * Rejected, not rounded. Also measured on torch 2.10: `enable_gqa=True` with
 * `H=6` accepts `kvHeads` of 1, 2, 3 and 6 and raises `RuntimeError("Number of
 * heads in key and value must divide the number of heads in query")` for 4, 5
 * and 7. Matching that is rule 7, but it would be the right answer anyway —
 * every way of handling a ragged split (clamp the last group, distribute the
 * remainder, wrap) is a silent guess about which query heads lose their own KV,
 * and the caller has a model architecture that already knows.
 *
 * Note this also rejects `kvHeads > H`, which is not a grouping at all.
 *
 * ## Everything else is `attention` (#18)
 *
 * `scale`, the causal convention and `queryOffset` are unchanged, and this file
 * deliberately does not restate why they are what they are — see
 * `ops/attention/reference.ts`, which measured them. Two facts worth recording
 * because they were checked here rather than assumed:
 *
 * - the default `scale` is still `1/sqrt(D)` from the query's head dim under
 *   `enable_gqa=True`; sharing KV does not change it;
 * - causal masking is applied after the KV head is chosen, so `is_causal=True`
 *   with GQA equals `is_causal=True` on the expanded K and V. Verified at
 *   `H=4, kvHeads=2, L=3, S=7`.
 *
 * ## `mask` is `attention`'s, unchanged (#77)
 *
 * The additive attention bias, its broadcast shape, its rejection of `causal`
 * and its zeros for a fully masked row are all `ops/attention`'s and are
 * documented there. This op imports the check rather than restating it: three
 * copies of a shape contract are three chances for the three attention ops to
 * disagree about the shape of the same tensor, and a caller who has to remember
 * which op wants which layout has been given the choice the mask exists to take
 * away.
 *
 * `maskHeads` is over the **query** heads (`H`), not `kvHeads`. Verified against
 * torch 2.10: `enable_gqa=True` with a `[B, 1, 1, S]` mask reproduces the same
 * mask applied to the expanded K and V to 4.4e-16, so the mask is applied after
 * the KV head is chosen — as causal masking already was.
 *
 * The contract that changed: `queryOffset >= 0` no longer implies no row is
 * fully masked, because `mask` can mask every key. See `ops/attention` for what
 * that row now returns.
 */

export interface GroupedAttentionArgs {
  /** `[B, H, L, D]`, row-major. */
  q: Float32Array;
  /** `[B, kvHeads, S, D]`, row-major. */
  k: Float32Array;
  /** `[B, kvHeads, S, Dv]`, row-major. */
  v: Float32Array;
  /** Batch size. Shared by Q, K and V — only the head count differs. */
  B: number;
  /** Query heads. */
  H: number;
  /**
   * Key / value heads. Must divide `H`. `kvHeads === H` is plain MHA,
   * `kvHeads === 1` is MQA, and anything between is GQA.
   */
  kvHeads: number;
  /** Query positions. */
  L: number;
  /** Key / value positions — the KV cache length, which may exceed `L`. */
  S: number;
  /** Head dim of Q and K. */
  D: number;
  /** Head dim of V, and of the output. May differ from `D`. */
  Dv: number;
  /** Apply the causal mask. Default `false`. Throws if combined with `mask`. */
  causal?: boolean;
  /** Absolute position of query row 0 within the key sequence. Default `0`. Must be `>= 0`. */
  queryOffset?: number;
  /** Overrides the `1/sqrt(D)` default, like PyTorch's `scale=`. */
  scale?: number;
  /**
   * Additive attention bias, `[maskBatch, maskHeads, maskRows, S]` row-major —
   * PyTorch's float `attn_mask`. See `ops/attention/reference.ts`.
   */
  mask?: Float32Array;
  /**
   * `[maskBatch, maskHeads, maskRows]`, each `1` or the full dimension. Default
   * `[B, 1, 1]`. `maskHeads` counts **query** heads, not `kvHeads`.
   */
  maskShape?: MaskShape;
  /**
   * The number of key/value positions actually resident — issue #117.
   * Default `S`, the whole cache, unchanged from every caller before this
   * field existed.
   *
   * `S` stays the address stride (`k`/`v` are still `[B, kvHeads, S, D/Dv]`
   * buffers, and `kvHeadOf`'s per-head offset is still `kvFlat * S * D`);
   * `sEff` only bounds *which* of those `S` positions the softmax scans.
   * Positions `[sEff, S)` are treated exactly like a fully-masked key: not
   * summed into the row max, not summed into the softmax denominator, not
   * read from `v`. That is a real difference from setting those positions'
   * `mask` to `-Infinity` — a masked-but-in-range key still costs a dot
   * product and a wasted term in the sum; an `sEff`-excluded one costs
   * nothing, which is the entire point (a resident decode/prefill engine
   * addresses its KV cache at `maxSeqLen` but has only filled the first
   * `position` entries of it, and re-scanning the unfilled remainder every
   * step is issue #116's own measured cost, 672 MiB/token on Sarashina2.2-1B's
   * shape).
   *
   * A caller can only shrink `sEff` where doing so is provably safe, so this
   * is checked rather than trusted (rule 8):
   *
   * - `causal` false: `sEff` must equal `S`. Without a causal bound this op
   *   has no way to know which excluded positions were actually irrelevant —
   *   `mask` is an opaque function of `(b, h, i, j)`, so unlike the causal
   *   case below there is no formula here to check against.
   * - `causal` true: every query row `i` (`0 <= i < L`) may attend up to key
   *   `i + queryOffset`, so the row with the largest reach is the last one,
   *   `L - 1 + queryOffset`. `sEff` must be at least `min(S, L + queryOffset)`
   *   — anything smaller silently drops a key that row is entitled to see.
   */
  sEff?: number;
}

export interface GroupedAttentionResult {
  /** `[B, H, L, S]` — the attention matrix, one row per *query* head. */
  probs: Float32Array;
  /** `[B, H, L, Dv]`, row-major. */
  output: Float32Array;
}

/**
 * The `scale` PyTorch uses when none is given: `1/sqrt(D)`, from the query's
 * head dim. Same value as `attention`'s, restated here so this op's kernels and
 * tests have one definition to share rather than reaching across ops for it.
 */
export function defaultScale(D: number): number {
  return 1 / Math.sqrt(D);
}

/**
 * Bytes one K + V cache occupies.
 *
 * Here because the reason this op exists is a memory number, and a number that
 * only appears in a README drifts from the code. `kvHeads` is the only term a
 * caller changes to shrink it, and the saving is exactly linear in it: an
 * 8-group GQA layer holds a quarter of what a 32-head MHA layer holds, not
 * "less".
 *
 * Counts the elements the cache actually stores — `B * kvHeads * S * (D + Dv)`
 * of them — times `bytesPerElement`. Q is not cached, the attention matrix is
 * not cached, and neither is counted.
 */
export function kvCacheBytes(args: {
  B: number;
  kvHeads: number;
  S: number;
  D: number;
  Dv: number;
  /** Default 4, this library's f32 storage. Pass 2 for an f16 cache, 1 for int8. */
  bytesPerElement?: number;
}): number {
  const { B, kvHeads, S, D, Dv } = args;
  return B * kvHeads * S * (D + Dv) * (args.bytesPerElement ?? 4);
}

/**
 * KV head read by query head `h`. Exported because the kernels have to compute
 * the same thing in WGSL and this is the one place the convention is written
 * down in prose.
 */
export function kvHeadOf(h: number, H: number, kvHeads: number): number {
  return Math.floor(h / (H / kvHeads));
}

export function groupedAttention(args: GroupedAttentionArgs): GroupedAttentionResult {
  const { q, k, v, B, H, kvHeads, L, S, D, Dv } = args;
  const causal = args.causal ?? false;
  const queryOffset = args.queryOffset ?? 0;
  const scale = args.scale ?? defaultScale(D);
  const sEff = args.sEff ?? S;
  const { at: bias } = resolveMask(args, "groupedAttention");

  if (kvHeads < 1 || !Number.isInteger(kvHeads)) {
    throw new Error(`groupedAttention(): kvHeads must be a positive integer, got ${kvHeads}`);
  }
  if (H % kvHeads !== 0) {
    throw new Error(
      `groupedAttention(): kvHeads=${kvHeads} must divide H=${H}; a ragged split has no defensible grouping`,
    );
  }
  // sEff's own contract (see the field's doc): checked, not trusted, because
  // a wrong value here does not fail loudly on its own — it just silently
  // drops a key some query row was entitled to see.
  if (!Number.isInteger(sEff) || sEff < 1 || sEff > S) {
    throw new Error(`groupedAttention(): sEff must be an integer in [1, S=${S}], got ${sEff}`);
  }
  if (!causal && sEff !== S) {
    throw new Error(
      `groupedAttention(): sEff=${sEff} < S=${S} requires causal=true — without a causal bound there is no way to prove the excluded positions were unreachable`,
    );
  }
  if (causal) {
    const needed = Math.min(S, L + queryOffset);
    if (sEff < needed) {
      throw new Error(
        `groupedAttention(): sEff=${sEff} is too small for L=${L}, queryOffset=${queryOffset} — the last query row can attend up to key ${L - 1 + queryOffset}, so sEff must be at least ${needed}`,
      );
    }
  }
  // Shape checks, because the whole point of this op is that K and V are
  // *smaller* than Q, and the easiest mistake a caller makes when adopting it is
  // handing over MHA-shaped tensors that the index arithmetic will happily read
  // the wrong corner of.
  if (q.length !== B * H * L * D) {
    throw new Error(`groupedAttention(): expected ${B * H * L * D} query elements, got ${q.length}`);
  }
  if (k.length !== B * kvHeads * S * D) {
    throw new Error(`groupedAttention(): expected ${B * kvHeads * S * D} key elements, got ${k.length}`);
  }
  if (v.length !== B * kvHeads * S * Dv) {
    throw new Error(`groupedAttention(): expected ${B * kvHeads * S * Dv} value elements, got ${v.length}`);
  }

  const probs = new Float32Array(B * H * L * S);
  const output = new Float32Array(B * H * L * Dv);

  // The naive nest, exactly as in `attention`, with one line changed: which head
  // of K and V this query head reads. Rule 8 — this is the definition of
  // correct, so it is written to be read.
  for (let b = 0; b < B; b += 1) {
    for (let h = 0; h < H; h += 1) {
      const qFlat = b * H + h;
      const kvFlat = b * kvHeads + kvHeadOf(h, H, kvHeads);

      const qHead = qFlat * L * D;
      const pHead = qFlat * L * S;
      const oHead = qFlat * L * Dv;
      const kHead = kvFlat * S * D;
      const vHead = kvFlat * S * Dv;

      for (let i = 0; i < L; i += 1) {
        // `sEff`, not `S`: positions `[sEff, S)` are excluded from the scan
        // entirely, not masked to `-Infinity` and then summed anyway — see
        // `sEff`'s own doc for why that distinction is the whole point.
        const row = new Float64Array(sEff);
        for (let j = 0; j < sEff; j += 1) {
          if (causal && j > i + queryOffset) {
            row[j] = -Infinity;
            continue;
          }
          let dot = 0;
          for (let d = 0; d < D; d += 1) dot += q[qHead + i * D + d]! * k[kHead + j * D + d]!;
          row[j] = dot * scale + bias(b, h, i, j);
        }

        let max = -Infinity;
        for (let j = 0; j < sEff; j += 1) max = Math.max(max, row[j]!);
        // Every key masked: zeros, as `ops/attention` explains and torch's
        // `aten::_safe_softmax` does. Reachable only through `mask`.
        if (max === -Infinity) continue;
        let sum = 0;
        for (let j = 0; j < sEff; j += 1) sum += Math.exp(row[j]! - max);
        for (let j = 0; j < sEff; j += 1) probs[pHead + i * S + j] = Math.exp(row[j]! - max) / sum;

        for (let c = 0; c < Dv; c += 1) {
          let acc = 0;
          for (let j = 0; j < sEff; j += 1) acc += probs[pHead + i * S + j]! * v[vHead + j * Dv + c]!;
          output[oHead + i * Dv + c] = acc;
        }
      }
    }
  }

  return { probs, output };
}
