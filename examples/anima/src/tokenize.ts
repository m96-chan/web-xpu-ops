/**
 * Anima's two tokenizers, over the same prompt.
 *
 * `AnimaTokenizer` (`comfy/text_encoders/anima.py:21`) runs both and hands the
 * results to different places: the Qwen2 byte-level BPE's ids condition the
 * 0.6B, and the T5 unigram's ids index the adapter's own embedding table. They
 * are not two spellings of one thing — see `text-encoder.ts` for why the model
 * wants both.
 *
 * The BPE side is `llm/tokenizer-bpe.ts` unchanged.
 *
 * The unigram side is `llm/tokenizer.ts`'s Viterbi, with T5's normalizer done
 * here first. That class refuses a `Precompiled` charsmap rather than
 * mis-normalizing, and it is right to: the charsmap is a 316 kB blob. But it is
 * a *measurable* blob. `tools/gen_tokenizer_data.py` compares it against NFKC
 * at every codepoint in the BMP and finds exactly **20** that differ, which it
 * emits as a table. So this is `String.normalize("NFKC")` — which the engine
 * already has — plus twenty entries, and the table is derived on every run of
 * the tool rather than transcribed here.
 */
import { type BpeVocab, ByteLevelBpeTokenizer } from "../../../llm/tokenizer-bpe.js";
import { type AddedToken, PieceType, SentencePieceTokenizer, type TokenizerVocab } from "../../../llm/tokenizer.js";

/** The shape `tools/gen_tokenizer_data.py` writes for the T5 side. */
export interface T5Vocab {
  pieces: [string, number][];
  unkId: number;
  byteFallback: boolean;
  addedTokens: AddedToken[];
  normalizer: {
    /** Codepoint (lowercase hex) to what the charsmap gives, where NFKC differs. */
    nfkcExceptions: Record<string, string>;
    stripLeft: boolean;
    stripRight: boolean;
    /** ` {2,}` — a run of two or more spaces. */
    collapsePattern: string;
    /** `▁`. */
    collapseTo: string;
  };
  metaspace: { replacement: string; add_prefix_space?: boolean; prepend_scheme?: string };
  eosId: number;
}

const META_SPACE = "▁";
const SPACE = " ";

/**
 * T5's normalizer sequence, in its own order: the charsmap, then `Strip`
 * (right only), then `Replace` of two-or-more spaces with `▁`.
 *
 * The order matters and is not the obvious one: collapsing runs happens
 * **after** stripping, so `"  both  "` loses its trailing pair entirely and
 * keeps its leading pair as a single `▁`.
 *
 * The collapse *target* turns out not to matter. `Replace` writes `▁` where a
 * space would do just as well, because the `Metaspace` step below rewrites every
 * remaining space to `▁` anyway — measured by changing it and watching all 56
 * fixtures still pass. It is written as the tokenizer writes it, but nobody
 * should read a distinction into it that is not observable.
 */
export function normalizeT5(text: string, config: T5Vocab["normalizer"]): string {
  // The exceptions must survive NFKC, not merely be applied before it. Half of
  // them are characters NFKC *would* fold and the charsmap does not — U+FF5E to
  // an ASCII tilde, U+32FF to 令和 — so substituting them and then normalizing
  // the whole string folds them right back. Running NFKC only over the runs
  // *between* exceptions is what keeps them. (This was the first attempt, and
  // `a～b` came out as `a~b`.)
  //
  // Splicing is safe because every exception is a control, a zero-width, a
  // space-like, or a standalone letter: none is a combining mark and none can
  // compose with a neighbour, so no composition is broken by the seam.
  let out = "";
  let pending = "";
  for (const ch of text) {
    const replacement = config.nfkcExceptions[ch.codePointAt(0)!.toString(16).padStart(4, "0")];
    if (replacement === undefined) {
      pending += ch;
      continue;
    }
    out += pending.normalize("NFKC") + replacement;
    pending = "";
  }
  out += pending.normalize("NFKC");

  if (config.stripLeft) out = out.replace(/^\s+/u, "");
  if (config.stripRight) out = out.replace(/\s+$/u, "");
  out = out.replace(new RegExp(config.collapsePattern, "gu"), config.collapseTo);

  // `Metaspace(replacement="▁", add_prefix_space=true)`, done here rather than
  // by `SentencePieceTokenizer.normalize` because the prefix is **conditional**:
  // HF prepends `▁` only when the text does not already start with one. The
  // unconditional prepend that SentencePiece's own `add_dummy_prefix` performs
  // gives `" leading"` two metaspaces instead of one, which is a different
  // token sequence and not an obviously wrong-looking one.
  // An empty prompt gets no metaspace at all — HF prefixes a *string*, and
  // there is no string here. Prefixing anyway makes `""` encode to one token
  // instead of none, which is a caption the model never saw.
  if (out.length === 0) return "";
  out = out.split(SPACE).join(META_SPACE);
  return out.startsWith(META_SPACE) ? out : META_SPACE + out;
}

/**
 * The T5 unigram, as a `SentencePieceTokenizer` over an already-normalized
 * string.
 *
 * `normalizer.name` is reported as `identity` because `normalizeT5` has already
 * run — the class's guard is against *silently* skipping a normalizer, and this
 * is the opposite of silent. `escapeWhitespaces` and `addDummyPrefix` carry
 * `Metaspace(replacement="▁", add_prefix_space=true)`.
 */
export function t5Tokenizer(vocab: T5Vocab): {
  encode(text: string): number[];
  decode(ids: ArrayLike<number>): string;
} {
  if (vocab.metaspace.replacement !== META_SPACE) {
    throw new Error(`t5Tokenizer: expected a ▁ metaspace, got ${JSON.stringify(vocab.metaspace.replacement)}`);
  }
  if (vocab.metaspace.add_prefix_space === false) {
    throw new Error("t5Tokenizer: normalizeT5 always prefixes; add_prefix_space=false is not implemented");
  }
  const special = new Set(vocab.addedTokens.filter((t) => t.special).map((t) => t.content));
  const inner: TokenizerVocab = {
    vocabSize: vocab.pieces.length,
    byteFallback: vocab.byteFallback,
    unkId: vocab.unkId,
    // T5 has no BOS and its EOS is appended by the post-processor, not by the
    // model — `-1` so nothing here adds one behind `encode`'s back.
    bosId: -1,
    eosId: -1,
    unkSurface: vocab.pieces[vocab.unkId]?.[0] ?? "<unk>",
    // Everything off: `normalizeT5` has already normalized, escaped the
    // whitespace and applied the conditional metaspace prefix. Leaving
    // `escapeWhitespaces` on here would be harmless (no spaces are left) and
    // `addDummyPrefix` would add a second `▁`.
    normalizer: {
      name: "identity",
      addDummyPrefix: false,
      removeExtraWhitespaces: false,
      escapeWhitespaces: false,
    },
    addedTokens: vocab.addedTokens,
    specialTokens: {},
    pieces: vocab.pieces.map(([piece, score], id) => [
      piece,
      score,
      id === vocab.unkId ? PieceType.UNKNOWN : special.has(piece) ? PieceType.CONTROL : PieceType.NORMAL,
    ]),
  };
  const tokenizer = new SentencePieceTokenizer(inner);
  return {
    encode: (text) => tokenizer.encode(normalizeT5(text, vocab.normalizer)),
    decode: (ids) => tokenizer.decode(Array.from(ids)),
  };
}

export interface AnimaTokenizers {
  qwen: ByteLevelBpeTokenizer;
  t5: ReturnType<typeof t5Tokenizer>;
  /** `</s>`, appended by `TemplateProcessing` after every T5 encode. */
  t5EosId: number;
}

export function animaTokenizers(qwenVocab: BpeVocab, t5Vocab: T5Vocab): AnimaTokenizers {
  return { qwen: new ByteLevelBpeTokenizer(qwenVocab), t5: t5Tokenizer(t5Vocab), t5EosId: t5Vocab.eosId };
}

export interface TokenizedPrompt {
  qwenIds: Int32Array;
  t5Ids: Int32Array;
  /**
   * One weight per T5 token, multiplied into the adapter's output by
   * `preprocess_text_embeds`. Plain prompts are all 1.0; the field exists
   * because the model reads it, and because a caller that implements
   * `(emphasis:1.2)` syntax has somewhere to put the result.
   */
  t5Weights: Float32Array;
}

/**
 * A prompt to the ids both halves of the conditioning path want.
 *
 * The Qwen side gets no sentinels at all (`has_start_token=False,
 * has_end_token=False`, `anima.py:10`); the T5 side keeps its `</s>`. Getting
 * that backwards shifts every position by one and is invisible in the output.
 */
export function tokenizePrompt(tokenizers: AnimaTokenizers, text: string): TokenizedPrompt {
  const qwenIds = Int32Array.from(tokenizers.qwen.encode(text));
  const t5 = [...tokenizers.t5.encode(text), tokenizers.t5EosId];
  return {
    qwenIds,
    t5Ids: Int32Array.from(t5),
    t5Weights: new Float32Array(t5.length).fill(1),
  };
}
