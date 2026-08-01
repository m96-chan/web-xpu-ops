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
 * Contract, inherited unchanged: `queryOffset >= 0`, so no row is ever fully
 * masked and neither the reference nor the kernels guard against one.
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
  /** Apply the causal mask. Default `false`. */
  causal?: boolean;
  /** Absolute position of query row 0 within the key sequence. Default `0`. Must be `>= 0`. */
  queryOffset?: number;
  /** Overrides the `1/sqrt(D)` default, like PyTorch's `scale=`. */
  scale?: number;
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

  if (kvHeads < 1 || !Number.isInteger(kvHeads)) {
    throw new Error(`groupedAttention(): kvHeads must be a positive integer, got ${kvHeads}`);
  }
  if (H % kvHeads !== 0) {
    throw new Error(
      `groupedAttention(): kvHeads=${kvHeads} must divide H=${H}; a ragged split has no defensible grouping`,
    );
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
        const row = new Float64Array(S);
        for (let j = 0; j < S; j += 1) {
          if (causal && j > i + queryOffset) {
            row[j] = -Infinity;
            continue;
          }
          let dot = 0;
          for (let d = 0; d < D; d += 1) dot += q[qHead + i * D + d]! * k[kHead + j * D + d]!;
          row[j] = dot * scale;
        }

        let max = -Infinity;
        for (let j = 0; j < S; j += 1) max = Math.max(max, row[j]!);
        let sum = 0;
        for (let j = 0; j < S; j += 1) sum += Math.exp(row[j]! - max);
        for (let j = 0; j < S; j += 1) probs[pHead + i * S + j] = Math.exp(row[j]! - max) / sum;

        for (let c = 0; c < Dv; c += 1) {
          let acc = 0;
          for (let j = 0; j < S; j += 1) acc += probs[pHead + i * S + j]! * v[vHead + j * Dv + c]!;
          output[oHead + i * Dv + c] = acc;
        }
      }
    }
  }

  return { probs, output };
}
