import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { type BpeVocab, ByteLevelBpeTokenizer } from "./tokenizer-bpe.js";

/**
 * `llm/tokenizer-bpe.ts` against the real Qwen tokenizer.
 *
 * Every id below came out of the `tokenizers` library, not out of reading this
 * port and agreeing with it (`llm/tools/gen_bpe_fixtures.py`). The cases are
 * the ones the SentencePiece side found to matter (#104) plus the ones specific
 * to BPE: the pre-tokeniser's contraction list, whitespace runs, and the
 * NFC pair that must *not* encode alike.
 */

const data = new URL("./data/", import.meta.url);
const spec = JSON.parse(
  readFileSync(fileURLToPath(new URL("qwen-qwen3-4b.bpe-vocab.json", data)), "utf8"),
) as BpeVocab & { source: string };
const fixtures = JSON.parse(
  readFileSync(fileURLToPath(new URL("qwen-qwen3-4b.bpe-fixtures.json", data)), "utf8"),
) as { cases: { name: string; text: string; ids: number[]; decoded: string }[] };

describe("ByteLevelBpeTokenizer", () => {
  const tok = new ByteLevelBpeTokenizer(spec);

  it("loads the checkpoint's own vocabulary", () => {
    expect(tok.vocabSize).toBe(151643);
  });

  for (const c of fixtures.cases) {
    it(`encodes ${c.name} exactly as the reference tokenizer`, () => {
      expect(tok.encode(c.text)).toEqual(c.ids);
    });
  }

  for (const c of fixtures.cases) {
    it(`decodes ${c.name} back`, () => {
      expect(tok.decode(c.ids)).toBe(c.decoded);
    });
  }

  it("refuses a normalizer it does not implement", () => {
    // Ignoring an unknown normalizer would silently change every id, and no
    // test written against this file alone would notice — the fixtures would
    // have been generated with it applied.
    expect(() => new ByteLevelBpeTokenizer({ ...spec, normalizer: "NFKC" })).toThrow(/only NFC is implemented/);
  });

  it("distinguishes composed from decomposed text", () => {
    // The point of running NFC at all. If it were skipped these would differ;
    // if it were applied twice they would still match, so this pins the
    // direction rather than merely the presence.
    const composed = fixtures.cases.find((c) => c.name === "nfc-composed")!;
    const decomposed = fixtures.cases.find((c) => c.name === "nfd-decomposed")!;
    expect(decomposed.text).not.toBe(composed.text);
    expect(tok.encode(decomposed.text)).toEqual(tok.encode(composed.text));
  });

  it("keeps special tokens whole", () => {
    // `<|im_start|>` is one id, not a merge of `<`, `|`, `im`... A port that
    // dropped added-token matching would still produce ids that decode back to
    // the same string, so the check is on the count.
    const ids = tok.encode("<|im_start|>");
    expect(ids).toHaveLength(1);
    // Round-tripped with `skipSpecial: false`, since the default drops it —
    // that default is the reference's, checked in the fixtures above.
    expect(tok.decode(ids, { skipSpecial: false })).toBe("<|im_start|>");
    expect(tok.decode(ids)).toBe("");
  });
});
