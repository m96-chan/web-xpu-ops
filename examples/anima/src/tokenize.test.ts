/**
 * Both of Anima's tokenizers against the real ones, case by case.
 *
 * Issue #170 stage 5. Correctness is whatever `transformers` and `tokenizers`
 * produce — `tools/gen_tokenizer_data.py` runs the cases through them and
 * commits the ids, so a disagreement here is this port's, not a judgement call.
 */
import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import type { BpeVocab } from "../../../llm/tokenizer-bpe.js";
import { PADDING, type T5Vocab, animaTokenizers, normalizeT5, tokenizePrompt } from "./tokenize.js";

const read = (name: string): unknown =>
  JSON.parse(readFileSync(fileURLToPath(new URL(`../fixtures/${name}`, import.meta.url)), "utf8"));

// Not a fixture of its own: ComfyUI's `qwen25_tokenizer` is byte-for-byte the
// vocabulary this repository already carries for Qwen3-4B — same pieces, merges,
// split pattern and added tokens, checked field by field by
// `tools/gen_tokenizer_data.py`, which refuses to run if they ever diverge.
const qwenVocab = JSON.parse(
  readFileSync(fileURLToPath(new URL("../../../llm/data/qwen-qwen3-4b.bpe-vocab.json", import.meta.url)), "utf8"),
) as BpeVocab;
const t5Vocab = read("t5.unigram-vocab.json") as T5Vocab;
const fixtures = read("tokenizer-fixtures.json") as {
  padding: { qwenMinLength: number; qwenPadToken: number; t5MinLength: number; t5PadToken: number };
  cases: { label: string; text: string; qwen: number[]; t5: number[] }[];
};

const tokenizers = animaTokenizers(qwenVocab, t5Vocab);

// The fixtures come from ComfyUI's `AnimaTokenizer`, which is both backends
// *plus* `SDTokenizer`'s padding — so they are compared against `tokenizePrompt`,
// which is the same whole. Comparing against the bare backends is what missed
// `min_length` and let the pipeline throw on its first empty negative prompt.
describe("the Qwen BPE", () => {
  it.each(fixtures.cases.map((c) => [c.label, c] as const))("encodes %s", (_label, c) => {
    expect(Array.from(tokenizePrompt(tokenizers, c.text).qwenIds)).toEqual(c.qwen);
  });
});

describe("the T5 unigram", () => {
  it.each(fixtures.cases.map((c) => [c.label, c] as const))("encodes %s", (_label, c) => {
    expect(Array.from(tokenizePrompt(tokenizers, c.text).t5Ids)).toEqual(c.t5);
  });
});

describe("SDTokenizer's padding", () => {
  it("matches what ComfyUI's tokenizer was built with", () => {
    expect(PADDING).toEqual(fixtures.padding);
  });

  it("keeps an empty prompt from producing no tokens", () => {
    // What CFG runs against every step. The Qwen side takes no sentinels, so
    // without this it is a zero-length sequence and the encoder throws.
    const out = tokenizePrompt(tokenizers, "");
    expect(Array.from(out.qwenIds)).toEqual([PADDING.qwenPadToken]);
    expect(out.t5Ids.length).toBeGreaterThan(0);
  });
});

describe("T5's normalizer", () => {
  const config = t5Vocab.normalizer;

  it("was measured, not guessed", () => {
    // 51 codepoints across the whole BMP, 31 of them below 0x20. The first
    // measurement started at the space and found 20; the controls are where
    // the charsmap does its most consequential work, mapping tab and newline
    // to a space where NFKC leaves them alone.
    expect(Object.keys(config.nfkcExceptions).length).toBe(51);
    expect(config.nfkcExceptions["0009"]).toBe(" ");
    expect(config.nfkcExceptions["000a"]).toBe(" ");
  });

  it("strips the right end only", () => {
    expect(config.stripLeft).toBe(false);
    expect(config.stripRight).toBe(true);
    // Leading whitespace survives normalization and becomes a metaspace later;
    // trailing whitespace is gone before the tokenizer ever sees it.
    // The output carries the metaspace: `normalizeT5` ends with HF's
    // `Metaspace` step, so a space is a `▁` by the time anyone sees it.
    expect(normalizeT5(" x", config)).toBe("▁x");
    expect(normalizeT5("x ", config)).toBe("▁x");
  });

  it("collapses a run of spaces to one separator", () => {
    expect(normalizeT5("a  b", config)).toBe("\u2581a\u2581b");
    expect(normalizeT5("a   b", config)).toBe("\u2581a\u2581b");
    expect(normalizeT5("a b", config)).toBe("\u2581a\u2581b");
    // All four spellings reach the same string. `Replace` writes U+2581 rather
    // than a space, but `Metaspace` rewrites the remaining spaces to U+2581
    // too, so the target is unobservable — changing it to a space leaves every
    // fixture here passing. Asserted as an equivalence rather than described as
    // a distinction, because it is not one.
    expect(normalizeT5("a    b", config)).toBe(normalizeT5("a b", config));
  });

  it("gives an empty prompt no tokens at all", () => {
    expect(normalizeT5("", config)).toBe("");
    expect(normalizeT5("   ", config)).toBe("");
  });

  it("applies NFKC", () => {
    expect(normalizeT5("\uff71\uff72\uff73", config)).toBe("\u2581\u30a2\u30a4\u30a6");
    expect(normalizeT5("\u2460\u2461", config)).toBe("\u258112");
    expect(normalizeT5("\ufb01ne", config)).toBe("\u2581fine");
    // Decomposed to composed, which is what makes `cafe\u0301` one word either way.
    expect(normalizeT5("cafe\u0301", config)).toBe("\u2581caf\u00e9");
  });

  it("keeps the exceptions NFKC would fold", () => {
    // The measured table earning its place. NFKC turns U+FF5E into `~` and
    // U+32FF into `令和`; T5's older charsmap does neither, and a port that
    // used bare NFKC would tokenize both differently.
    expect(normalizeT5("a～b", config)).toBe("▁a～b");
    expect(normalizeT5("㋿", config)).toBe("▁㋿");
    expect("a～b".normalize("NFKC")).toBe("a~b");
  });

  it("turns zero-widths into spaces rather than dropping them", () => {
    expect(normalizeT5("a\u200bb", config)).toBe("▁a▁b");
    expect(normalizeT5("\ufeffhello", config)).toBe("▁hello");
    // Tab and newline are the same story and were missed by a measurement
    // that started at the space.
    expect(normalizeT5("a\tb", config)).toBe("▁a▁b");
  });
});

describe("tokenizePrompt", () => {
  it("gives the Qwen side no sentinels and the T5 side its </s>", () => {
    const c = fixtures.cases.find((x) => x.label === "the golden's prompt")!;
    const out = tokenizePrompt(tokenizers, c.text);
    expect(Array.from(out.qwenIds)).toEqual(c.qwen);
    expect(Array.from(out.t5Ids)).toEqual(c.t5);
    expect(out.t5Ids[out.t5Ids.length - 1]).toBe(t5Vocab.eosId);
    // Getting these backwards — a sentinel on the Qwen side, none on the T5
    // side — shifts every position by one and changes nothing visible.
    expect(out.qwenIds[0]).not.toBe(t5Vocab.eosId);
  });

  it("weights every token at 1.0", () => {
    const out = tokenizePrompt(tokenizers, "a cat");
    expect(out.t5Weights.length).toBe(out.t5Ids.length);
    expect(Array.from(out.t5Weights).every((w) => w === 1)).toBe(true);
  });

  it("matches the ids the encoder golden was baked with", () => {
    // The join between this file and `verify-encoder.ts`: that golden carries
    // the ids it used, and they must be the ids this produces from the same
    // prompt. Otherwise both halves pass and the pipeline still conditions on
    // something the model never saw.
    const raw = readFileSync(fileURLToPath(new URL("../fixtures/encoder-golden.bin", import.meta.url)));
    const headerLength = Number(raw.readBigUInt64LE(0));
    const header = JSON.parse(raw.subarray(8, 8 + headerLength).toString("utf8")) as {
      prompt: string;
      tensors: { name: string; shape: number[]; offset: number }[];
    };
    const base = 8 + headerLength;
    const ids = (name: string): number[] => {
      const e = header.tensors.find((t) => t.name === name)!;
      const count = e.shape.reduce((a, b) => a * b, 1);
      return Array.from(new Int32Array(raw.buffer.slice(raw.byteOffset + base + e.offset, raw.byteOffset + base + e.offset + count * 4)));
    };
    const out = tokenizePrompt(tokenizers, header.prompt);
    expect(Array.from(out.qwenIds)).toEqual(ids("qwenIds"));
    expect(Array.from(out.t5Ids)).toEqual(ids("t5Ids"));
  });
});
