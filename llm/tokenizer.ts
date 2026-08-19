/**
 * SentencePiece **unigram** tokenizer: encode (Viterbi) / decode.
 *
 * Ground truth for every behavioral choice below is the real
 * `tokenizer.model` (Sarashina2.2-1B-Instruct), read with Python
 * `sentencepiece` and cross-checked against `transformers` — see issue #101
 * and `llm/tools/export_tokenizer.py` / `llm/tools/gen_fixtures.py` for the
 * scripts that produced these findings. Nothing here is assumed from how
 * SentencePiece models "usually" behave; the common defaults (NFKC
 * normalization, a dummy leading space) do **not** hold for this model and
 * are exactly the kind of thing this file exists to get right rather than
 * infer.
 *
 * ## Normalizer: "identity", not NFKC
 *
 * `normalizer_spec.name` is literally `"identity"`. No Unicode normalization
 * is applied — composed and decomposed forms of the same glyph tokenize
 * differently (verified: U+304C "が" vs U+304B "か" + U+3099 combining voiced
 * sound mark produce different ids). Only two things happen before
 * segmentation, both read from `normalizer_spec`:
 *
 *   - `addDummyPrefix` (false for this model): if true, prepend `▁` before
 *     segmentation.
 *   - `escapeWhitespaces` (true for this model): replace the literal ASCII
 *     space (U+0020) — and *only* that character, not tabs, newlines, NBSP,
 *     or the full-width space — with `▁` (U+2581).
 *
 * `removeExtraWhitespaces` and a non-empty `precompiled_charsmap` /
 * `normalization_rule_tsv` are not implemented; the constructor throws
 * rather than silently mis-normalizing a model that needs them (this
 * model has neither).
 *
 * ## Byte fallback
 *
 * `trainer_spec.byte_fallback` is true: the vocabulary carries 256 BYTE-type
 * pieces (`<0x00>`..`<0xFF>`). A codepoint gets decomposed into these only
 * when the unigram trie has **no** matching piece of any length starting at
 * it — verified empirically (tab `\t` and newline `\n` have no NORMAL piece
 * of their own and always fall back to bytes; astral-plane emoji likewise).
 *
 * The score assigned to that fallback edge is **not** an arbitrary "very
 * negative" sentinel — it has to be small in magnitude, and it has to be
 * exactly one score per *codepoint* (not one per byte). Both were found the
 * hard way, not assumed:
 *
 *   - An early version used a huge per-*byte* constant (`-1e6`, multiplied by
 *     up to 4 for an astral codepoint). That is locally harmless (byte edges
 *     never compete with a real piece at the same position — see below) but
 *     it is not globally harmless: accumulated over dozens of uncovered
 *     codepoints in one string (verified with a 500-character sweep through
 *     the Hiragana block, several of whose codepoints are unassigned/
 *     uncovered), `bestScore` reached magnitudes around -5×10⁷. float32's
 *     precision there is only about ±6 in absolute terms — larger than the
 *     ~1-2 point gaps between competing *real* segmentations later in the
 *     same string — so it silently corrupted unrelated, correct comparisons
 *     for the rest of the string. Reproduced and fixed; see
 *     `llm/tokenizer.test.ts`'s "mixes byte-fallback ... without corrupting"
 *     case and the `repeated_char*`/hiragana-sweep fixtures.
 *   - SentencePiece's actual C++ source (`unigram_model.cc`,
 *     `Lattice::PopulateNodes`) inserts exactly one fallback node per
 *     uncovered *character* — `Insert(begin_pos, 1)` — scored
 *     `min_score() - kUnkPenalty` where `kUnkPenalty = 10.0f` and
 *     `min_score()` is the minimum score over the model's NORMAL pieces (for
 *     this vocabulary, about -18.32, giving a fallback score around -28.32).
 *     `byte_fallback` changes what that one node expands to in the *output*
 *     (the codepoint's UTF-8 bytes instead of literal `<unk>`), not how many
 *     lattice nodes/scores it costs. `UNK_PENALTY` and `byteFallbackScore`
 *     below reproduce that formula.
 *
 * SentencePiece's training semantics guarantee that an uncovered codepoint
 * cannot appear inside *any* vocabulary piece, short or long, so there is
 * never a real competition between "spell this codepoint as bytes" and "use
 * a real piece that happens to contain it" at the *same* position — matching
 * the real formula's magnitude is what keeps that guarantee from being
 * undone by precision loss two or three uncovered codepoints later.
 *
 * ## Special / control tokens are NOT matched by the unigram search
 *
 * `sp.encode("<|system|>")` against the real model does not return the
 * control token's id — it falls through to per-character/byte segmentation
 * of the literal text. CONTROL and UNKNOWN pieces are excluded from the
 * search trie entirely. A real front end (HF `tokenizers`, llama.cpp, ...)
 * recognizes these strings via a *separate* literal-substring
 * pre-tokenization pass over the added-tokens table
 * (`tokenizer_config.json`'s `added_tokens_decoder`) before anything reaches
 * the unigram model — verified by comparing segment boundaries against the
 * real (HF) tokenizer for inputs like `"prefix<|system|>suffix"`. This file
 * reproduces that pass in {@link SentencePieceTokenizer.encode}: added-token
 * content strings are matched left-to-right, longest match first, and the
 * text in between is run through the identity-normalizer + Viterbi path.
 *
 * ## Decoding
 *
 * `spaces_between_special_tokens: false` and `clean_up_tokenization_spaces:
 * false` (`tokenizer_config.json`): decode never inserts a token or word
 * separator that was not in the ids, and applies no post-hoc cleanup pass.
 * It is exactly: substitute each added-token id for its literal content
 * string, decode every other run of ids as SentencePiece would (join piece
 * text, UTF-8-decode grouped byte-fallback runs, then replace `▁` with a
 * literal space), and concatenate — matching the real tokenizer's
 * `decode(encode(text)) === text` round trip observed on every fixture in
 * `llm/data/*.fixtures.json`.
 */

/** `SentencePieceModel.SentencePiece.Type` (protobuf wire values). */
export const PieceType = {
  NORMAL: 1,
  UNKNOWN: 2,
  CONTROL: 3,
  USER_DEFINED: 4,
  UNUSED: 5,
  BYTE: 6,
} as const;
export type PieceType = (typeof PieceType)[keyof typeof PieceType];

/** One row of `pieces` in the exported vocab JSON: `[piece, score, type]`. */
export type PieceTuple = [piece: string, score: number, type: PieceType];

export interface AddedToken {
  id: number;
  content: string;
  special: boolean;
}

export interface NormalizerConfig {
  name: string;
  addDummyPrefix: boolean;
  removeExtraWhitespaces: boolean;
  escapeWhitespaces: boolean;
}

/** Shape written by `llm/tools/export_tokenizer.py`. */
export interface TokenizerVocab {
  vocabSize: number;
  byteFallback: boolean;
  unkId: number;
  bosId: number;
  eosId: number;
  unkSurface: string;
  normalizer: NormalizerConfig;
  addedTokens: AddedToken[];
  specialTokens: Record<string, string>;
  pieces: PieceTuple[];
}

const SPACE = " ";
const META_SPACE = "▁"; // ▁

/**
 * `kUnkPenalty` in SentencePiece's `unigram_model.cc`: the fallback edge for
 * one uncovered codepoint scores `min_score() - 10.0`, not an arbitrarily
 * tiny sentinel. See the module doc for why the magnitude matters (it isn't
 * just "worse than any real piece" — it also has to not blow out float32
 * precision for the rest of the string once several fallback edges
 * accumulate).
 */
const UNK_PENALTY = 10.0;

interface TrieNode {
  children?: Map<string, TrieNode>;
  /** Set when a piece ends exactly here. */
  id?: number;
  score?: number;
}

interface PieceInfo {
  piece: string;
  score: number;
  type: PieceType;
}

/** `[begin, end)` codepoint span, produced by the added-token pre-split. */
type Segment = { kind: "text"; text: string } | { kind: "special"; id: number };

function utf8Bytes(text: string): Uint8Array {
  return new TextEncoder().encode(text);
}

function utf8Decode(bytes: number[]): string {
  return new TextDecoder("utf-8", { fatal: false }).decode(Uint8Array.from(bytes));
}

/** Parses `<0xAB>` -> 0xAB. Returns null if `piece` isn't that exact shape. */
function parseByteHex(piece: string): number | null {
  const m = /^<0x([0-9A-Fa-f]{2})>$/.exec(piece);
  return m ? parseInt(m[1] as string, 16) : null;
}

export class SentencePieceTokenizer {
  private readonly vocab: TokenizerVocab;
  private readonly piecesById: PieceInfo[];
  private readonly trieRoot: TrieNode = {};
  private readonly byteIdByValue: Int32Array = new Int32Array(256).fill(-1);
  private readonly addedTokenById = new Map<number, string>();
  /** Added-token content strings, longest first, for greedy literal matching. */
  private readonly addedTokensByLength: AddedToken[];
  /** `min_score() - kUnkPenalty` — see the module doc's "Byte fallback" section. */
  private readonly byteFallbackScore: number = 0;

  constructor(vocab: TokenizerVocab) {
    this.vocab = vocab;
    const { normalizer } = vocab;
    if (normalizer.name !== "identity") {
      throw new Error(
        `SentencePieceTokenizer: normalizer "${normalizer.name}" is not implemented — ` +
          `only "identity" (no Unicode normalization) has been verified against a real model. ` +
          `Re-check llm/tools/export_tokenizer.py's normalizer_spec dump before adding support.`,
      );
    }
    if (normalizer.removeExtraWhitespaces) {
      throw new Error(
        "SentencePieceTokenizer: removeExtraWhitespaces=true is not implemented " +
          "(no verified model exercises it).",
      );
    }

    this.piecesById = vocab.pieces.map(([piece, score, type]) => ({ piece, score, type }));

    for (const token of vocab.addedTokens) {
      this.addedTokenById.set(token.id, token.content);
    }
    this.addedTokensByLength = [...vocab.addedTokens].sort(
      (a, b) => b.content.length - a.content.length,
    );

    let minNormalScore = Number.POSITIVE_INFINITY;
    for (let id = 0; id < this.piecesById.length; id++) {
      const info = this.piecesById[id] as PieceInfo;
      switch (info.type) {
        case PieceType.NORMAL:
          this.insertTrie(info.piece, id, info.score);
          if (info.score < minNormalScore) minNormalScore = info.score;
          break;
        case PieceType.BYTE: {
          const byte = parseByteHex(info.piece);
          if (byte === null) {
            throw new Error(`SentencePieceTokenizer: malformed BYTE piece ${JSON.stringify(info.piece)}`);
          }
          this.byteIdByValue[byte] = id;
          break;
        }
        case PieceType.CONTROL:
        case PieceType.UNKNOWN:
          // Intentionally excluded from the search trie — see module doc.
          break;
        default:
          throw new Error(
            `SentencePieceTokenizer: unsupported piece type ${info.type} for id ${id} ` +
              `(${JSON.stringify(info.piece)}) — only NORMAL/CONTROL/UNKNOWN/BYTE have been ` +
              `verified against a real model.`,
          );
      }
    }

    if (vocab.byteFallback) {
      for (let b = 0; b < 256; b++) {
        if (this.byteIdByValue[b] === -1) {
          throw new Error(
            `SentencePieceTokenizer: byteFallback=true but no BYTE piece found for 0x${b
              .toString(16)
              .padStart(2, "0")}`,
          );
        }
      }
      if (!Number.isFinite(minNormalScore)) {
        throw new Error("SentencePieceTokenizer: byteFallback=true but the vocab has no NORMAL pieces");
      }
      this.byteFallbackScore = minNormalScore - UNK_PENALTY;
    }
  }

  private insertTrie(piece: string, id: number, score: number): void {
    let node = this.trieRoot;
    for (const ch of piece) {
      node.children ??= new Map();
      let next = node.children.get(ch);
      if (!next) {
        next = {};
        node.children.set(ch, next);
      }
      node = next;
    }
    node.id = id;
    node.score = score;
  }

  /**
   * Splits on literal, longest-match, left-to-right occurrences of added
   * tokens (both "special" and non-"special" — SentencePiece's own search
   * never sees either kind, so both must be pre-split identically). Mirrors
   * `llm/tools/gen_fixtures.py`'s `split_added_tokens`.
   */
  private splitAddedTokens(text: string): Segment[] {
    const segments: Segment[] = [];
    let buffer = "";
    let i = 0;
    const n = text.length;
    outer: while (i < n) {
      for (const token of this.addedTokensByLength) {
        if (token.content.length > 0 && text.startsWith(token.content, i)) {
          if (buffer.length > 0) {
            segments.push({ kind: "text", text: buffer });
            buffer = "";
          }
          segments.push({ kind: "special", id: token.id });
          i += token.content.length;
          continue outer;
        }
      }
      buffer += text[i];
      i += 1;
    }
    if (buffer.length > 0) segments.push({ kind: "text", text: buffer });
    return segments;
  }

  private normalize(text: string): string {
    let out = this.vocab.normalizer.escapeWhitespaces ? text.split(SPACE).join(META_SPACE) : text;
    if (this.vocab.normalizer.addDummyPrefix) out = META_SPACE + out;
    return out;
  }

  /** Unigram Viterbi over one already-normalized plain-text span. */
  private encodeSpan(normalized: string): number[] {
    const cps = Array.from(normalized); // codepoint-safe (surrogate pairs stay whole)
    const n = cps.length;
    if (n === 0) return [];

    // dp[i] = best cumulative score reaching codepoint offset i, plus how we
    // got there (previous offset + the token ids emitted on that edge).
    //
    // Float32Array: piece scores are float32 in the model (`optional float
    // score` in the protobuf) and SentencePiece's own C++ Lattice
    // accumulates them at float32 precision, not float64. That rounding is
    // not cosmetic — verified directly: for `"あ".repeat(30)`, the four
    // pieces [8-char, 8-char, 8-char, 6-char] sum to the *exact same*
    // float64 total regardless of the 6-char piece's position (addition is
    // commutative), but the real tokenizer always places it last.
    // Accumulating in float32 breaks the tie the same way: rounding after
    // each addition makes "...,6-char" score marginally higher than
    // "6-char,..." even though the exact real-number sums are equal,
    // matching the real tokenizer's fixture output. The rounding that
    // actually does this work happens at the `Math.fround(candidate)` calls
    // below, at each comparison — this Float32Array is what makes the
    // *stored* side of each of those comparisons consistent with it (a
    // Float64Array holding the same `fround`ed values would behave
    // identically; storing them pre-rounded, rather than re-deriving the
    // rounding by coincidence, is what keeps the two from drifting apart if
    // this function is ever changed to read `bestScore` from somewhere other
    // than immediately after a `fround`ed write).
    const bestScore = new Float32Array(n + 1).fill(Number.NEGATIVE_INFINITY);
    const backPointer = new Int32Array(n + 1).fill(-1);
    const edgeIds: (number[] | undefined)[] = new Array(n + 1);
    bestScore[0] = 0;

    for (let i = 0; i < n; i++) {
      if (bestScore[i] === Number.NEGATIVE_INFINITY) continue; // unreachable (shouldn't happen)

      let node = this.trieRoot;
      let matchedAny = false;
      for (let j = i; j < n; j++) {
        const child = node.children?.get(cps[j] as string);
        if (!child) break;
        node = child;
        if (node.id !== undefined) {
          matchedAny = true;
          const end = j + 1;
          // Math.fround, not a bare float64 add: bestScore[end] was stored
          // through a Float32Array and is therefore already float32-rounded.
          // Comparing an un-rounded float64 sum against an already-rounded
          // float32 value can flip an exact tie (see the class comment on
          // `bestScore`) in either direction depending on which way the
          // *stored* value happened to round — rounding `candidate` first
          // makes the comparison the same precision on both sides, matching
          // a C++ `float` accumulator where the rounding happens at every
          // assignment, including the one being compared against.
          const candidate = Math.fround((bestScore[i] as number) + (node.score as number));
          if (candidate > (bestScore[end] as number)) {
            bestScore[end] = candidate;
            backPointer[end] = i;
            edgeIds[end] = [node.id];
          }
        }
      }

      if (!matchedAny) {
        // No piece of any length starts here: forced byte-fallback for
        // exactly this one codepoint (see module doc — never in competition
        // with a real piece).
        const bytes = utf8Bytes(cps[i] as string);
        const ids: number[] = [];
        for (const b of bytes) {
          const byteId = this.byteIdByValue[b] as number;
          if (byteId === -1) {
            throw new Error(
              `SentencePieceTokenizer: byte 0x${b.toString(16)} has no BYTE piece and ` +
                `byteFallback is ${this.vocab.byteFallback ? "on" : "off"} — cannot encode ` +
                `codepoint ${JSON.stringify(cps[i])}`,
            );
          }
          ids.push(byteId);
        }
        // One score for the whole codepoint, regardless of how many UTF-8
        // bytes it expands to — matching SentencePiece's real
        // `Insert(begin_pos, 1)` (one node per uncovered *character*, byte
        // count is an output-shape detail, not a per-edge cost). See the
        // module doc's "Byte fallback" section for why this specific
        // magnitude (not just "very negative") is load-bearing.
        const end = i + 1;
        const candidate = Math.fround((bestScore[i] as number) + this.byteFallbackScore);
        if (candidate > (bestScore[end] as number)) {
          bestScore[end] = candidate;
          backPointer[end] = i;
          edgeIds[end] = ids;
        }
      }
    }

    if (bestScore[n] === Number.NEGATIVE_INFINITY) {
      throw new Error("SentencePieceTokenizer: no path reached the end of the span (unreachable)");
    }

    // Backtracking visits edges in reverse chronological order, but each
    // edge's own ids (e.g. the several bytes of one byte-fallback codepoint)
    // are already in correct left-to-right order — unshifting whole edges
    // (rather than pushing ids then reversing the flat array) keeps that
    // internal order intact while still reversing edge order.
    const segments: number[][] = [];
    let pos = n;
    while (pos > 0) {
      const ids = edgeIds[pos] as number[];
      const prev = backPointer[pos] as number;
      segments.unshift(ids);
      pos = prev;
    }
    return segments.flat();
  }

  /**
   * Encodes `text` to token ids. Does **not** add BOS/EOS — this model's
   * `tokenizer_config.json` has `add_bos_token: false` / `add_eos_token:
   * false`, and callers that want the chat-template's literal `<s>`/`</s>`
   * text get them for free through the added-token split.
   */
  encode(text: string): number[] {
    const ids: number[] = [];
    for (const segment of this.splitAddedTokens(text)) {
      if (segment.kind === "special") {
        ids.push(segment.id);
      } else {
        ids.push(...this.encodeSpan(this.normalize(segment.text)));
      }
    }
    return ids;
  }

  /**
   * Decodes token ids back to text. Added-token ids are substituted with
   * their literal content (no separators inserted either side — see module
   * doc); every other id goes through the SentencePiece-style path: piece
   * text is concatenated as-is, byte-fallback runs are grouped and
   * UTF-8-decoded, and `▁` is replaced with a literal space once per
   * contiguous run of non-added-token ids.
   */
  decode(ids: number[]): string {
    const out: string[] = [];
    let spmBuffer = "";
    let byteRun: number[] = [];

    const flushByteRun = () => {
      if (byteRun.length > 0) {
        spmBuffer += utf8Decode(byteRun);
        byteRun = [];
      }
    };
    const flushSpm = () => {
      flushByteRun();
      if (spmBuffer.length > 0) {
        out.push(spmBuffer.split(META_SPACE).join(SPACE));
        spmBuffer = "";
      }
    };

    for (const id of ids) {
      const added = this.addedTokenById.get(id);
      if (added !== undefined) {
        flushSpm();
        out.push(added);
        continue;
      }

      const info = this.piecesById[id];
      if (!info) {
        throw new Error(`SentencePieceTokenizer: id ${id} is out of range (vocabSize=${this.vocab.vocabSize})`);
      }

      if (info.type === PieceType.BYTE) {
        const byte = parseByteHex(info.piece);
        if (byte === null) {
          throw new Error(`SentencePieceTokenizer: malformed BYTE piece ${JSON.stringify(info.piece)}`);
        }
        byteRun.push(byte);
      } else if (info.type === PieceType.NORMAL) {
        flushByteRun();
        spmBuffer += info.piece;
      } else if (id === this.vocab.unkId) {
        flushByteRun();
        spmBuffer += this.vocab.unkSurface;
      } else {
        // CONTROL/UNKNOWN ids not covered by addedTokenById: every such id
        // in the verified vocab *is* covered (control_symbols ==
        // added_tokens_decoder), so this means the vocab JSON and
        // tokenizer_config.json it was exported alongside have drifted.
        throw new Error(
          `SentencePieceTokenizer: id ${id} has type ${info.type} (${JSON.stringify(info.piece)}) ` +
            `but is not in addedTokens — vocab/config mismatch.`,
        );
      }
    }
    flushSpm();

    return out.join("");
  }
}
