/**
 * Scaled dot-product attention, unfused: `softmax(mask(scale * Q @ K^T)) @ V`.
 *
 * The definition of correct for this op, and the thing `flash_attention` (#19)
 * has to be measured against. Deliberately the naive nest of loops — the fused
 * kernel is allowed to be clever, this is not.
 *
 * Conventions follow `torch.nn.functional.scaled_dot_product_attention`
 * (rule 7). Each of the following was checked against torch 2.10 rather than
 * read off the documentation, because every one of them has a second plausible
 * answer that is silently wrong:
 *
 * - **`scale` defaults to `1/sqrt(D)`, where `D` is the *query's* head dim.**
 *   When `V` has a different head dim (`Dv`), PyTorch still divides by the
 *   query's. Measured: `1/sqrt(Dv)` does not reproduce its output.
 * - **`causal` is upper-left aligned**, like PyTorch's `is_causal=True`: the
 *   mask is `tril` of an `L x S` matrix taken from the top-left corner, so
 *   query `i` may attend to key `j` iff `j <= i` — *even when `S > L`*.
 *   Measured at `L=3, S=7`: PyTorch keeps keys 0..i, not the last `i+1` keys.
 *   That is a trap for a KV cache, which is why `queryOffset` exists.
 * - **`Dv != D` is allowed** and the output is `[B, H, L, Dv]`.
 *
 * `queryOffset` is this library's addition and the reason is worth stating,
 * because the follow-up ops depend on it. It is the absolute position of query
 * row 0 in the key sequence: row `i` sits at `i + queryOffset`, so the mask is
 * `j <= i + queryOffset`. It reaches both of PyTorch's conventions from one
 * knob, with no second boolean:
 *
 *   - `queryOffset = 0`     — `is_causal=True`. The default.
 *   - `queryOffset = S - L` — `torch.nn.attention.bias.causal_lower_right`,
 *     which is what a KV-cache decode step actually wants: the query rows are
 *     the *newest* positions, sitting at the end of the cache. Verified equal
 *     to `causal_lower_right(L, S)` on torch 2.10.
 *
 * Contract: `queryOffset >= 0`. Every row then attends to at least key 0, so no
 * row is ever fully masked. Nothing in the kernels guards against a fully
 * masked row, deliberately — it is unreachable, and rule 1 says code that
 * cannot be made to fail is code that should not be written. (For the record,
 * PyTorch fills a fully masked row with zeros; this reference would give NaN
 * and the kernel a uniform row. All three disagree, and all three are outside
 * the contract.)
 *
 * Sums are in f64 because JavaScript numbers are f64; the kernels sum in f32,
 * so agreement — not equality — is the test. `probs` is stored as f32 on
 * purpose: the two-dispatch kernel rounds the attention matrix to f32 on its
 * way between dispatches, and the reference should model that rather than
 * carry f64 probabilities into the second matmul.
 */
export interface AttentionArgs {
  /** `[B, H, L, D]`, row-major. */
  q: Float32Array;
  /** `[B, H, S, D]`, row-major. */
  k: Float32Array;
  /** `[B, H, S, Dv]`, row-major. */
  v: Float32Array;
  /** Batch size. */
  B: number;
  /** Attention heads. Shared by Q, K and V today; GQA / MQA (#24) is the op that splits them. */
  H: number;
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

export interface AttentionResult {
  /**
   * `[B, H, L, S]` — the materialised attention matrix, after masking and
   * softmax. Exposed because it is the seam between the two dispatches, so each
   * half can be tested on its own. The fused kernel (#19) never forms it and
   * will be compared on `output` alone.
   */
  probs: Float32Array;
  /** `[B, H, L, Dv]`, row-major. */
  output: Float32Array;
}

/**
 * The `scale` PyTorch uses when none is given: `1/sqrt(D)`, from the query's
 * head dim. Exported so the tests and the fused kernel cannot drift from it by
 * each writing the formula out again.
 */
export function defaultScale(D: number): number {
  return 1 / Math.sqrt(D);
}

export function attention(args: AttentionArgs): AttentionResult {
  const { q, k, v, B, H, L, S, D, Dv } = args;
  const causal = args.causal ?? false;
  const queryOffset = args.queryOffset ?? 0;
  const scale = args.scale ?? defaultScale(D);

  const probs = new Float32Array(B * H * L * S);
  const output = new Float32Array(B * H * L * Dv);

  for (let head = 0; head < B * H; head += 1) {
    const qHead = head * L * D;
    const kHead = head * S * D;
    const vHead = head * S * Dv;
    const pHead = head * L * S;
    const oHead = head * L * Dv;

    for (let i = 0; i < L; i += 1) {
      // Scores for one query row. Masked entries become -Infinity, which is
      // literally what PyTorch adds as `attn_bias`.
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

      // Row softmax, max-subtracted. Not an optimisation: attention logits grow
      // with D and exp() overflows f32 long before the inputs look unusual.
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

  return { probs, output };
}
