/**
 * Tokenizer-agnostic bridge between token ids and text.
 *
 * `llm/constraints` deliberately does not import a tokenizer. Issue #101's
 * tokenizer lives on a separate, not-yet-merged branch, and this ISSUE
 * (#102) is scoped to be independent of it — the constraint machinery only
 * needs *some* mapping between token ids and the strings they decode to,
 * not any particular implementation. A caller wires its own tokenizer
 * through this interface (or a small adapter over one); tests in this
 * directory use a mock.
 */
export interface TokenCodec {
  /** Tokenizes `text` into the id sequence this tokenizer would produce for it. */
  encode(text: string): number[];
  /** The exact string a single token id decodes to. */
  idToToken(id: number): string;
  /** Ids are assumed dense: every `id` with `0 <= id < vocabSize` is valid. */
  readonly vocabSize: number;
}
