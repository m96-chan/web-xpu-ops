/**
 * Byte-level BPE — Qwen's, and GPT-2's before it.
 *
 * A separate module from `tokenizer.ts` because the algorithms have nothing in
 * common beyond their signature. That one is SentencePiece **unigram**: the
 * vocabulary carries a score per piece and encoding is a Viterbi search for the
 * most likely segmentation. This is **BPE**: the vocabulary carries no scores,
 * and encoding replays an ordered list of merges, always applying the
 * highest-ranked pair available. Sharing code between them would mean sharing
 * nothing but the word "tokenizer".
 *
 * Four stages, in this order, because each one's output is the next one's
 * input and swapping any two changes the ids:
 *
 * 1. **NFC** — declared by the checkpoint's own `normalizer`. Without it a
 *    decomposed "é" (`e` + U+0301) tokenises differently from a composed one,
 *    and text from macOS filesystems arrives decomposed.
 * 2. **Pre-tokenisation** — the split regex from the checkpoint, which keeps
 *    contractions, letters, digits, punctuation and whitespace runs apart.
 *    Merges never cross one of these pieces, so this is what stops "dog." from
 *    becoming a token.
 * 3. **Byte level** — each piece is taken as UTF-8 *bytes*, and every byte is
 *    mapped to a printable character. That is the whole trick: the vocabulary
 *    is over this alphabet, so any byte sequence is representable and there is
 *    no unknown token, ever.
 * 4. **Merges** — repeatedly join the adjacent pair with the lowest rank.
 *
 * Correctness is defined by `llm/data/*.bpe-fixtures.json`, produced by running
 * the real `tokenizers` library (`llm/tools/gen_bpe_fixtures.py`), not by
 * agreement with this file's reasoning.
 */

/** The two pieces a BPE needs, plus what the checkpoint says about the rest. */
export interface BpeVocab {
  /** Token string to id. Strings are over the byte-level alphabet, not UTF-8. */
  vocab: Record<string, number>;
  /** Ordered; earlier pairs win. */
  merges: [string, string][] | string[][];
  /** `<|im_start|>` and friends: matched literally, before anything else. */
  addedTokens?: { id: number; content: string; special: boolean }[];
  /** `"NFC"` for Qwen. Anything else is refused rather than ignored. */
  normalizer?: string | null;
  /** The pre-tokeniser's split pattern, as a JS-compatible regex source. */
  splitPattern?: string | null;
}

/**
 * GPT-2's byte-to-character table.
 *
 * Bytes that are already printable ASCII (and two Latin-1 runs) map to
 * themselves; the remaining 68 map to U+0100 and up. The point is that every
 * byte becomes a character no whitespace rule will split and no normaliser will
 * touch — which is what lets stage 3 above run *after* stage 2 without the
 * earlier stages ever seeing a raw byte.
 */
function byteEncoder(): { toChar: string[]; fromChar: Map<string, number> } {
  const printable: number[] = [];
  for (let b = 0x21; b <= 0x7e; b += 1) printable.push(b);
  for (let b = 0xa1; b <= 0xac; b += 1) printable.push(b);
  for (let b = 0xae; b <= 0xff; b += 1) printable.push(b);

  const toChar = new Array<string>(256);
  let next = 0;
  for (let b = 0; b < 256; b += 1) {
    if (printable.includes(b)) {
      toChar[b] = String.fromCodePoint(b);
    } else {
      toChar[b] = String.fromCodePoint(256 + next);
      next += 1;
    }
  }
  const fromChar = new Map<string, number>();
  toChar.forEach((ch, b) => fromChar.set(ch, b));
  return { toChar, fromChar };
}

const BYTES = byteEncoder();

/**
 * The checkpoint's split pattern, translated for JavaScript.
 *
 * `tokenizers` uses Rust's regex crate, whose syntax overlaps JS but is not the
 * same. The two differences that matter here:
 *
 *  - `(?i:...)` is an inline group flag Rust has and JS does not. The group it
 *    applies to is the contraction list, so it is rewritten to spell both
 *    cases out rather than dropped — dropping it would stop `IT'S` from
 *    splitting where `it's` does.
 *  - `\p{L}` and `\p{N}` need the `u` flag in JS. With it, the pattern is
 *    equivalent.
 */
function compileSplit(pattern: string): RegExp {
  const jsSource = pattern.replace(
    /\(\?i:([^)]*)\)/g,
    (_, inner: string) =>
      `(?:${(inner as string)
        .split("|")
        .map((alt) => alt.replace(/[a-z]/g, (c) => `[${c}${c.toUpperCase()}]`))
        .join("|")})`,
  );
  return new RegExp(jsSource, "gu");
}

export class ByteLevelBpeTokenizer {
  readonly #vocab: Map<string, number>;
  readonly #byId: Map<number, string>;
  /** Pair key (`"a b"`) to rank. Lower wins. */
  readonly #ranks: Map<string, number>;
  readonly #split: RegExp;
  readonly #added: { id: number; content: string }[];
  readonly #special = new Set<number>();
  readonly #cache = new Map<string, number[]>();

  constructor(spec: BpeVocab) {
    if (spec.normalizer != null && spec.normalizer !== "NFC") {
      // Refused rather than ignored: a normaliser this does not implement
      // changes the ids, and silently producing different ids is the failure
      // that survives every test written against this file alone.
      throw new Error(
        `ByteLevelBpeTokenizer: checkpoint declares normalizer "${spec.normalizer}", ` +
          `and only NFC is implemented. Ignoring it would silently change the ids.`,
      );
    }

    this.#vocab = new Map(Object.entries(spec.vocab));
    this.#byId = new Map();
    for (const [token, id] of this.#vocab) this.#byId.set(id, token);

    this.#ranks = new Map();
    spec.merges.forEach((pair, rank) => {
      const [a, b] = Array.isArray(pair) ? pair : String(pair).split(" ");
      this.#ranks.set(`${a} ${b}`, rank);
    });

    if (!spec.splitPattern) {
      throw new Error("ByteLevelBpeTokenizer: the vocab carries no splitPattern; re-export it with gen_bpe_fixtures.py");
    }
    this.#split = compileSplit(spec.splitPattern);

    // Longest first, so `<|im_start|>` wins over any shorter added token that
    // happens to be its prefix.
    this.#added = [...(spec.addedTokens ?? [])]
      .map((t) => ({ id: t.id, content: t.content }))
      .sort((x, y) => y.content.length - x.content.length);

    // Added tokens live outside `vocab` in the checkpoint, so the reverse map
    // needs them explicitly. Without this, encoding `<|im_start|>` works and
    // decoding the id it produced throws — the two halves disagreeing about
    // the same token.
    for (const token of this.#added) this.#byId.set(token.id, token.content);
    for (const token of spec.addedTokens ?? []) if (token.special) this.#special.add(token.id);
  }

  get vocabSize(): number {
    return this.#vocab.size;
  }

  /** Applies merges to one pre-tokenised piece, already byte-encoded. */
  #mergePiece(piece: string): number[] {
    const cached = this.#cache.get(piece);
    if (cached) return cached;

    let symbols = [...piece];
    for (;;) {
      let bestRank = Infinity;
      let bestAt = -1;
      for (let i = 0; i + 1 < symbols.length; i += 1) {
        const rank = this.#ranks.get(`${symbols[i]} ${symbols[i + 1]}`);
        if (rank !== undefined && rank < bestRank) {
          bestRank = rank;
          bestAt = i;
        }
      }
      if (bestAt < 0) break;
      symbols = [
        ...symbols.slice(0, bestAt),
        symbols[bestAt]! + symbols[bestAt + 1]!,
        ...symbols.slice(bestAt + 2),
      ];
    }

    const ids: number[] = [];
    for (const symbol of symbols) {
      const id = this.#vocab.get(symbol);
      if (id === undefined) {
        // Unreachable by construction — the byte alphabet is in every
        // byte-level vocabulary, so a single character always resolves. Kept
        // because "unreachable" and "unchecked" should not be the same thing.
        throw new Error(`ByteLevelBpeTokenizer: no id for token ${JSON.stringify(symbol)}`);
      }
      ids.push(id);
    }
    this.#cache.set(piece, ids);
    return ids;
  }

  /** Text to token ids, matching the reference tokenizer. */
  encode(text: string): number[] {
    if (text === "") return [];

    // Added tokens are matched on the raw text, before normalisation: they are
    // declared `normalized: false` in the checkpoint, and NFC could otherwise
    // rewrite one out of existence.
    const out: number[] = [];
    let rest = text;
    while (rest.length > 0) {
      let hitAt = -1;
      let hit: { id: number; content: string } | null = null;
      for (const token of this.#added) {
        const at = rest.indexOf(token.content);
        if (at >= 0 && (hitAt < 0 || at < hitAt)) {
          hitAt = at;
          hit = token;
        }
      }
      if (!hit || hitAt < 0) {
        out.push(...this.#encodePlain(rest));
        break;
      }
      if (hitAt > 0) out.push(...this.#encodePlain(rest.slice(0, hitAt)));
      out.push(hit.id);
      rest = rest.slice(hitAt + hit.content.length);
    }
    return out;
  }

  #encodePlain(text: string): number[] {
    if (text === "") return [];
    const normalised = text.normalize("NFC");
    const ids: number[] = [];
    const utf8 = new TextEncoder();
    for (const match of normalised.matchAll(this.#split)) {
      const piece = match[0];
      if (piece === "") continue;
      let encoded = "";
      for (const byte of utf8.encode(piece)) encoded += BYTES.toChar[byte]!;
      ids.push(...this.#mergePiece(encoded));
    }
    return ids;
  }

  /**
   * Token ids back to text.
   *
   * `skipSpecial` defaults to **true**, matching `tokenizers`' own `decode`
   * (measured, not assumed: on `[151644, 872]` the reference gives `"user"` by
   * default and `"<|im_start|>user"` with `skip_special_tokens=False`). Which
   * one a caller wants depends on the job — rendering a model's output wants
   * the chat scaffolding gone, debugging a prompt wants it visible — so both
   * are reachable, and the default is the one the reference picked.
   */
  decode(ids: readonly number[], options?: { skipSpecial?: boolean }): string {
    const skipSpecial = options?.skipSpecial ?? true;
    const bytes: number[] = [];
    for (const id of ids) {
      if (skipSpecial && this.#special.has(id)) continue;
      const token = this.#byId.get(id);
      if (token === undefined) throw new Error(`ByteLevelBpeTokenizer: unknown id ${id}`);
      for (const ch of token) {
        const byte = BYTES.fromChar.get(ch);
        // An added token like `<|im_start|>` is stored as literal text rather
        // than in the byte alphabet, so its characters are encoded as UTF-8
        // here instead of looked up.
        if (byte === undefined) {
          for (const b of new TextEncoder().encode(ch)) bytes.push(b);
        } else {
          bytes.push(byte);
        }
      }
    }
    return new TextDecoder("utf-8").decode(new Uint8Array(bytes));
  }
}
