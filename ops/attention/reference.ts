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
 * ## `mask` — the additive attention bias (#77)
 *
 * `mask` is **added to the scores before the softmax**, which is PyTorch's
 * *float* `attn_mask`: `softmax(Q @ K^T * scale + mask) @ V`. Measured on torch
 * 2.10 rather than read — `scaled_dot_product_attention(q, k, v, attn_mask=m)`
 * reproduces that expression to 6.7e-16 in f64, and disagrees with the
 * subtractive reading by 1.06.
 *
 * There is exactly **one** mask type here, and it is this one. The choice is
 * worth stating because torch's `attn_mask` accepts two and means different
 * things by them:
 *
 *   - **float** — added to the scores. `-Infinity` masks, `0` does not.
 *   - **bool** — `True` means *attend*, `False` means *masked* (measured, not
 *     read: a bool mask keeping keys 0..2 equals a float mask of `-inf` on keys
 *     3..). Note this is the **opposite polarity** to `key_padding_mask` in
 *     `nn.MultiheadAttention`, where `True` means *ignore*. That reversal is the
 *     single most common way to get this wrong.
 *
 * The additive float form is the one implemented, for a reason specific to this
 * library: `ops/alibi` already produces an additive score bias and says
 * "masking is the caller's". With the same representation the two compose by
 * addition and no new op is needed to combine them. A boolean buffer would also
 * cost a second storage type in every kernel that reads it.
 *
 * `keyPaddingBias` converts the boolean form to this one. It is a converter, not
 * a second mask type — nothing downstream of it knows booleans exist.
 *
 * ### Broadcasting
 *
 * Torch broadcasts `attn_mask` against `[B, H, L, S]`. Both shapes the issue
 * named are supported, and so is anything between: `maskShape` is
 * `[maskBatch, maskHeads, maskRows]` and each entry must be either `1` or the
 * full dimension. `[B, 1, 1]` (the default) is a key-padding mask; `[B, H, L]`
 * is a full per-element bias; `[1, H, L]` is what `ops/alibi` hands back, so
 * that composition needs no tiling. The last axis is always `S` — a mask that
 * does not name every key is not a key mask.
 *
 * Not supported: an implicit leading broadcast, torch's `[L, S]`. Dropping axes
 * makes the buffer's shape depend on how many were dropped, and the kernel is
 * handed a flat buffer with no shape at all. Pass `maskShape: [1, 1, L]`.
 *
 * ### `causal` and `mask` together throw
 *
 * As torch does: `RuntimeError: _scaled_dot_product_attention: Explicit
 * attn_mask should not be set when is_causal=True` (measured on 2.10). Nobody
 * has asked for an intersection, and inventing one means every caller who
 * passes both by accident gets a plausible answer instead of an error.
 *
 * A caller who wants both writes the causal triangle into the mask, which is
 * exactly what torch's own `attn_bias` does internally.
 *
 * ### A fully masked row
 *
 * `queryOffset >= 0` used to guarantee every row saw key 0, so no row could be
 * fully masked and none of the kernels guarded against one. **`mask` breaks that
 * guarantee**: a sample whose every key is `-Infinity` is now reachable, and it
 * is reachable by accident — a zero-length sequence in a batch is one padded row
 * away from it.
 *
 * So it is guarded rather than declared out of contract, because a NaN row is a
 * wrong answer that looks like a result. **A fully masked row produces zeros**,
 * in `probs` and in `output`, which is what torch does: `scaled_dot_product_
 * attention` routes through `aten::_safe_softmax`, not `torch.softmax`.
 * Measured, and the distinction is real — on the same all-`-inf` row
 * `torch.softmax` gives `[nan, nan, nan, nan]` while `aten::_safe_softmax` gives
 * `[0, 0, 0, 0]`, and SDPA's output is all zeros with no NaN.
 *
 * "Fully masked" means the row maximum is `-Infinity`, and nothing weaker. A row
 * of `-FLT_MAX` is **not** fully masked: torch treats it as an ordinary finite
 * score and returns a uniform row (measured — output `[-0.662, -0.211, 0.556]`,
 * the mean of V, not zeros). The kernels have to tell the two apart because
 * their own causal mask writes `-FLT_MAX` — WGSL cannot write an infinity
 * literal — and they do it on the softmax denominator rather than the row
 * maximum: whichever key holds the maximum contributes `exp(0) = 1`, so the sum
 * is 0 exactly when no score was finite, while the maximum cannot report `-inf`
 * at all once the reduction starts from `-FLT_MAX`.
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
  /** Apply the causal mask. Default `false`. Throws if combined with `mask`. */
  causal?: boolean;
  /** Absolute position of query row 0 within the key sequence. Default `0`. Must be `>= 0`. */
  queryOffset?: number;
  /** Overrides the `1/sqrt(D)` default, like PyTorch's `scale=`. */
  scale?: number;
  /**
   * Additive attention bias, `[maskBatch, maskHeads, maskRows, S]` row-major —
   * PyTorch's **float** `attn_mask`. Added to the scaled scores before the
   * softmax, so `-Infinity` masks a key and `0` leaves it alone.
   */
  mask?: Float32Array;
  /**
   * The broadcast shape of `mask`, `[maskBatch, maskHeads, maskRows]`. Each
   * entry must be `1` or the full dimension (`B`, `H`, `L`). Default `[B, 1, 1]`,
   * a key-padding mask.
   */
  maskShape?: MaskShape;
}

/** `[maskBatch, maskHeads, maskRows]`; the last axis of a mask is always `S`. */
export type MaskShape = readonly [number, number, number];

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

/**
 * Turns the boolean key mask people actually hold into the additive bias this
 * library takes.
 *
 * `keep` is `[B, S]` and follows torch's **`attn_mask`** polarity, which is the
 * one named in the parameter: truthy means *attend to this key*, falsy means
 * *masked*. It is the reverse of `nn.MultiheadAttention`'s `key_padding_mask`,
 * so a caller holding one of those passes `!padding`.
 *
 * Returns `[B, 1, 1, S]` — the default `maskShape` — with `0` where a key is
 * kept and `-Infinity` where it is not. `0` and not a small number: an additive
 * bias is only a mask when it is `-Infinity`, and "very negative" is a different
 * op that leaks a little probability.
 *
 * A converter, not a second mask type. Nothing downstream of it takes booleans.
 */
export function keyPaddingBias(B: number, S: number, keep: ArrayLike<boolean | number>): Float32Array {
  if (keep.length !== B * S) {
    throw new Error(`keyPaddingBias(): expected ${B * S} flags, got ${keep.length}`);
  }
  const bias = new Float32Array(B * S);
  for (let index = 0; index < B * S; index += 1) bias[index] = keep[index] ? 0 : -Infinity;
  return bias;
}

/**
 * Validates a mask against the attention shape and returns how to index it.
 *
 * One definition, shared by `attention`, `gqa` and `flash_attention`, and the
 * one place the mask contract is enforced. Three copies of it would be three
 * chances for the three ops to disagree about the shape of the same tensor,
 * which is precisely the failure the mask exists to prevent.
 *
 * The returned `at` is `(b, h, i, j) => number`, and is `() => 0` when there is
 * no mask — the additive identity, which is also literally what torch's own
 * reference does (`attn_bias = torch.zeros(L, S)`, added unconditionally).
 */
export function resolveMask(
  args: {
    mask?: Float32Array;
    maskShape?: MaskShape;
    causal?: boolean;
    B: number;
    H: number;
    L: number;
    S: number;
  },
  op: string,
): { shape: MaskShape; at: (b: number, h: number, i: number, j: number) => number } {
  const { mask, B, H, L, S } = args;
  const shape = args.maskShape ?? ([B, 1, 1] as const);
  if (!mask) {
    if (args.maskShape) throw new Error(`${op}(): maskShape given without a mask`);
    return { shape, at: () => 0 };
  }
  if (args.causal) {
    // torch 2.10: "RuntimeError: _scaled_dot_product_attention: Explicit
    // attn_mask should not be set when is_causal=True". Rejected rather than
    // intersected, so a caller who passes both by accident is told rather than
    // handed a plausible answer.
    throw new Error(`${op}(): causal and mask are exclusive, as torch rejects is_causal with attn_mask`);
  }
  const [mb, mh, mr] = shape;
  const full = { maskBatch: [mb, B], maskHeads: [mh, H], maskRows: [mr, L] } as const;
  for (const [name, [got, want]] of Object.entries(full)) {
    if (got !== 1 && got !== want) {
      throw new Error(`${op}(): ${name} must be 1 or ${want}, got ${got}`);
    }
  }
  const expected = mb * mh * mr * S;
  if (mask.length !== expected) {
    throw new Error(`${op}(): mask ${mb}x${mh}x${mr}x${S} needs ${expected} elements, got ${mask.length}`);
  }
  return {
    shape,
    at: (b, h, i, j) =>
      mask[(((mb === 1 ? 0 : b) * mh + (mh === 1 ? 0 : h)) * mr + (mr === 1 ? 0 : i)) * S + j]!,
  };
}

export function attention(args: AttentionArgs): AttentionResult {
  const { q, k, v, B, H, L, S, D, Dv } = args;
  const causal = args.causal ?? false;
  const queryOffset = args.queryOffset ?? 0;
  const scale = args.scale ?? defaultScale(D);
  const { at: bias } = resolveMask(args, "attention");

  const probs = new Float32Array(B * H * L * S);
  const output = new Float32Array(B * H * L * Dv);

  for (let b = 0; b < B; b += 1) {
    for (let h = 0; h < H; h += 1) {
      const head = b * H + h;
      const qHead = head * L * D;
      const kHead = head * S * D;
      const vHead = head * S * Dv;
      const pHead = head * L * S;
      const oHead = head * L * Dv;

      for (let i = 0; i < L; i += 1) {
        // Scores for one query row. Causally masked entries become -Infinity,
        // which is literally what PyTorch adds as `attn_bias`; `mask` is added
        // to the survivors, which is what PyTorch does with a float `attn_mask`.
        const row = new Float64Array(S);
        for (let j = 0; j < S; j += 1) {
          if (causal && j > i + queryOffset) {
            row[j] = -Infinity;
            continue;
          }
          let dot = 0;
          for (let d = 0; d < D; d += 1) dot += q[qHead + i * D + d]! * k[kHead + j * D + d]!;
          row[j] = dot * scale + bias(b, h, i, j);
        }

        // Row softmax, max-subtracted. Not an optimisation: attention logits
        // grow with D and exp() overflows f32 long before the inputs look
        // unusual.
        let max = -Infinity;
        for (let j = 0; j < S; j += 1) max = Math.max(max, row[j]!);

        // Every key masked. Reachable only through `mask`, and zeros rather than
        // the NaN the lines below would produce — `aten::_safe_softmax`, which is
        // what SDPA calls, not `torch.softmax`, which returns NaN here. `output`
        // follows because the whole probability row is 0.
        if (max === -Infinity) continue;

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
