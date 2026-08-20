import type { LlamaConfig } from "./config.js";

/**
 * One decoder block's weights, f32, each `[outFeatures, inFeatures]`
 * row-major — `torch.nn.Linear.weight`'s own layout, and `matvec`'s `[M, K]`
 * convention, so a checkpoint's tensor needs no reshaping to become `matvec`'s
 * `matrix` directly. `llm/reshape.ts#transposeRowMajor` derives the transposed
 * copy `matmul`'s prefill path wants; that derivation happens once at engine
 * construction, not in this type.
 *
 * `wq` and `wk` are **not** the checkpoint's raw rows. See
 * `permuteRopeChannels` below for why, and where the permutation happens.
 */
export interface LlamaLayerWeights {
  /** `[hiddenSize]` */
  attnNorm: Float32Array;
  /** `[numHeads*headDim, hiddenSize]`, rows permuted per `permuteRopeChannels`. */
  wq: Float32Array;
  /** `[numKvHeads*headDim, hiddenSize]`, rows permuted per `permuteRopeChannels`. */
  wk: Float32Array;
  /** `[numKvHeads*headDim, hiddenSize]` — V is never rotated, so no permutation. */
  wv: Float32Array;
  /** `[hiddenSize, numHeads*headDim]` */
  wo: Float32Array;
  /** `[hiddenSize]` */
  ffnNorm: Float32Array;
  /** `[ffnHidden, hiddenSize]` */
  wGate: Float32Array;
  /** `[ffnHidden, hiddenSize]` */
  wUp: Float32Array;
  /** `[hiddenSize, ffnHidden]` */
  wDown: Float32Array;
}

export interface LlamaWeights {
  /** `[vocabSize, hiddenSize]` */
  embedTokens: Float32Array;
  layers: LlamaLayerWeights[];
  /** `[hiddenSize]` */
  finalNorm: Float32Array;
  /**
   * `[vocabSize, hiddenSize]`. Pass the same array as `embedTokens` when
   * `config.tieEmbeddings` is `true`; this type does not tie them for the
   * caller, because a `LlamaWeights` is data, not policy — the target model
   * (Sarashina2.2-1B) does not tie them at all.
   */
  lmHead: Float32Array;
}

/**
 * Reorders a Q or K projection's output rows from HF Llama's "rotate-half"
 * channel order into `ops/rope`'s interleaved-pair order — the permutation a
 * checkpoint's `q_proj.weight` / `k_proj.weight` needs before this engine's
 * RoPE dispatch can use it, and the reason `LlamaLayerWeights.wq` / `.wk` are
 * documented as *not* the raw checkpoint tensor.
 *
 * ## Why the two disagree
 *
 * `ops/rope` rotates **adjacent** channel pairs within a head — `(x[2i], x[2i+1])`
 * turn together, for `i` in `[0, headDim/2)` (see `ops/rope/reference.ts`).
 * HF's Llama (`apply_rotary_pos_emb`, `rotate_half`) instead pairs a channel
 * with the one **half a head away**: `(x[i], x[i + headDim/2])`. Both are
 * genuine RoPE — same angle per pair, `theta_i = pos * base^(-2i/headDim)` —
 * they just number the channels within a head differently. A weight trained
 * (or, here, randomly initialised) against one numbering produces nonsense
 * fed through a kernel that assumes the other.
 *
 * ## Why permuting the projection weight is enough
 *
 * Relabel channel `i` of HF's ordering as this engine's channel `2i`, and
 * channel `i + headDim/2` as `2i + 1`. Under that relabelling the two rotations
 * are **the same formula, verbatim** — `rotate_half`'s
 * `q'_i = q_i·cos_i − q_{i+headDim/2}·sin_i`, `q'_{i+headDim/2} = q_{i+headDim/2}·cos_i + q_i·sin_i`
 * is exactly `ops/rope`'s `out[2i] = x[2i]·cos_i − x[2i+1]·sin_i`,
 * `out[2i+1] = x[2i]·sin_i + x[2i+1]·cos_i` with `x[2i] := q_i`,
 * `x[2i+1] := q_{i+headDim/2}`. So relabelling — reordering the rows that
 * *produce* those channels, i.e. this permutation — reproduces the identical
 * rotated values, not merely an equivalent-up-to-tolerance one.
 *
 * And because the relabelling is applied identically to Q and K, the attention
 * dot product `sum_j q'_j * k'_j` sums the same set of `(q'_i, k'_i)` products
 * in a different order — a permutation invariant, not an approximation. V is
 * never rotated, so it needs no permutation, and the O-projection reads
 * attention's output by (unpermuted) head/channel exactly as before.
 *
 * `llm/rope-permutation.test.ts` checks this claim numerically — a
 * hand-written `rotate_half` reference against `ops/rope`'s own `rope()` on
 * permuted input — rather than trusting the derivation above on its own
 * (rule 2: no guessing).
 *
 * Applied once, at weight-load time (here or in `llm/tools/gen_fixture.py`,
 * which permutes before writing the fixture's weight file), never per
 * forward pass.
 */
export function permuteRopeChannels(weight: Float32Array, heads: number, headDim: number, inFeatures: number): Float32Array {
  if (weight.length !== heads * headDim * inFeatures) {
    throw new Error(
      `permuteRopeChannels: expected ${heads * headDim * inFeatures} elements, got ${weight.length}`,
    );
  }
  const half = headDim / 2;
  const output = new Float32Array(weight.length);
  for (let h = 0; h < heads; h += 1) {
    const headBase = h * headDim * inFeatures;
    for (let i = 0; i < half; i += 1) {
      const fromLow = headBase + i * inFeatures;
      const fromHigh = headBase + (i + half) * inFeatures;
      const toEven = headBase + 2 * i * inFeatures;
      const toOdd = headBase + (2 * i + 1) * inFeatures;
      output.set(weight.subarray(fromLow, fromLow + inFeatures), toEven);
      output.set(weight.subarray(fromHigh, fromHigh + inFeatures), toOdd);
    }
  }
  return output;
}

/** Shape assertions shared by every place a `LlamaWeights` is built (fixtures now, a real converter later). */
export function assertWeightShapes(config: LlamaConfig, weights: LlamaWeights): void {
  const { hiddenSize, numHeads, numKvHeads, headDim, ffnHidden, vocabSize, numLayers } = config;
  const expect = (name: string, actual: number, wanted: number) => {
    if (actual !== wanted) throw new Error(`LlamaWeights: ${name} has ${actual} elements, expected ${wanted}`);
  };
  expect("embedTokens", weights.embedTokens.length, vocabSize * hiddenSize);
  expect("finalNorm", weights.finalNorm.length, hiddenSize);
  expect("lmHead", weights.lmHead.length, vocabSize * hiddenSize);
  if (weights.layers.length !== numLayers) {
    throw new Error(`LlamaWeights: ${weights.layers.length} layers, expected ${numLayers}`);
  }
  weights.layers.forEach((layer, i) => {
    expect(`layers[${i}].attnNorm`, layer.attnNorm.length, hiddenSize);
    expect(`layers[${i}].wq`, layer.wq.length, numHeads * headDim * hiddenSize);
    expect(`layers[${i}].wk`, layer.wk.length, numKvHeads * headDim * hiddenSize);
    expect(`layers[${i}].wv`, layer.wv.length, numKvHeads * headDim * hiddenSize);
    expect(`layers[${i}].wo`, layer.wo.length, hiddenSize * numHeads * headDim);
    expect(`layers[${i}].ffnNorm`, layer.ffnNorm.length, hiddenSize);
    expect(`layers[${i}].wGate`, layer.wGate.length, ffnHidden * hiddenSize);
    expect(`layers[${i}].wUp`, layer.wUp.length, ffnHidden * hiddenSize);
    expect(`layers[${i}].wDown`, layer.wDown.length, hiddenSize * ffnHidden);
  });
}
