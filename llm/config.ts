/**
 * The shape of a llama-architecture decoder-only model, as a plain object.
 *
 * Every number an `llm/engine.ts` forward pass needs and nothing it does not:
 * this is a *config*, not a checkpoint, so it carries no weights. `LlamaEngine`
 * reads it once at construction and every dispatch it builds afterwards is
 * sized from these fields plus the weights passed in beside it.
 *
 * `headDim` is explicit rather than derived as `hiddenSize / numHeads`, because
 * some checkpoints (Sarashina2.2-1B among them: 1792 / 16 = 112, and its
 * `head_dim` in `config.json` is indeed 112 — but not every model divides so
 * cleanly, and some deliberately overprovision it) set `head_dim` independently
 * in `config.json`. Deriving it would be correct until it silently was not, on
 * a checkpoint nobody had tried yet — a guess this repo's rule 2 rules out.
 */
export interface LlamaConfig {
  /** Decoder blocks. */
  numLayers: number;
  /** The residual stream width — `hidden_size` in a HF `config.json`. */
  hiddenSize: number;
  /** Query heads — `num_attention_heads`. */
  numHeads: number;
  /**
   * Key/value heads — `num_key_value_heads`. Equal to `numHeads` is plain MHA;
   * `1` is MQA; anything in between is GQA. Must divide `numHeads` (`ops/gqa`
   * rejects a ragged split, and this engine does not second-guess it).
   */
  numKvHeads: number;
  /** Channels per head, for Q/K/V alike. See the note above on why this is not derived. */
  headDim: number;
  /** Width of the SiLU-gated MLP's hidden layer — `intermediate_size`. */
  ffnHidden: number;
  /** Rows of `embedTokens` and of `lmHead`. */
  vocabSize: number;
  /** RoPE's base, `10000` conventionally; larger bases (`500000`) extend context. */
  ropeTheta: number;
  /** `rms_norm_eps`, inside the square root — see `ops/rmsnorm`. */
  rmsNormEps: number;
  /**
   * Positions the KV cache pre-allocates for. `ops/gqa`'s cache is f32 and
   * fully materialised up front (issue #98's scope: optimising that is later),
   * so this is also the hard ceiling on how many tokens a single `LlamaEngine`
   * can carry — prefill plus every decode step after it.
   */
  maxSeqLen: number;
  /**
   * `true` when `lmHead` is the same matrix as `embedTokens` (weight tying),
   * `false` when the checkpoint carries a separate `lm_head.weight`.
   *
   * Sarashina2.2-1B does **not** tie them (see `SARASHINA_2_2_1B` below), so
   * this defaults to `false` — an engine that silently reused the embedding
   * for a checkpoint that trained a distinct `lm_head` would compute a real
   * matmul against the wrong matrix, not fail loudly.
   */
  tieEmbeddings?: boolean;
}

/**
 * The tiny random model `llm/tools/gen_fixture.py` builds, spelled out as a
 * config so the fixture test constructs its `LlamaEngine` from the same
 * numbers the generator used — one source of truth rather than two files that
 * have to be kept in sync by hand.
 *
 * Chosen to be the smallest shape that still exercises every seam GQA and RoPE
 * have: `numHeads=4`, `numKvHeads=2` is a real 2-groups-of-2 split (not MHA,
 * not MQA), and `headDim=16` (`hiddenSize / numHeads`) is large enough that
 * RoPE rotates more than one pair.
 */
export const TINY_FIXTURE_CONFIG: LlamaConfig = {
  numLayers: 2,
  hiddenSize: 64,
  numHeads: 4,
  numKvHeads: 2,
  headDim: 16,
  ffnHidden: 128,
  vocabSize: 256,
  ropeTheta: 10000,
  rmsNormEps: 1e-5,
  maxSeqLen: 32,
  tieEmbeddings: false,
};

/**
 * Sarashina2.2-1B's real dimensions (issue #96's target model), given here as
 * documentation rather than as a value the engine executes: running the actual
 * checkpoint needs a weight-conversion tool and a tokenizer, both later issues
 * (#96's child issues 3 and 4). This is only proof that `LlamaConfig` has a
 * field for every number that checkpoint's `config.json` carries.
 *
 * `rmsNormEps` and `tieEmbeddings` are the two numbers a `config.json` states
 * explicitly that are easy to get wrong by assuming a Llama default instead —
 * `1e-5` and `false` are Sarashina2.2-1B's own, not merely common ones.
 */
export const SARASHINA_2_2_1B_CONFIG: LlamaConfig = {
  numLayers: 24,
  hiddenSize: 1792,
  numHeads: 16,
  numKvHeads: 8,
  headDim: 112, // 1792 / 16
  ffnHidden: 6272,
  vocabSize: 102400,
  ropeTheta: 500000,
  rmsNormEps: 1e-5,
  maxSeqLen: 4096,
  tieEmbeddings: false,
};
