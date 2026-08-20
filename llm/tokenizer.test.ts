import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { SentencePieceTokenizer, type TokenizerVocab } from "./tokenizer.js";

/**
 * The vocab and fixtures are generated from the real Sarashina2.2-1B-Instruct
 * `tokenizer.model` by `llm/tools/export_tokenizer.py` and
 * `llm/tools/gen_fixtures.py` — see issue #101. Fixture ids/decoded text come
 * from Python `sentencepiece`'s `SentencePieceProcessor` directly, never from
 * `transformers` (which, on this environment, silently converts this UNIGRAM
 * model to an approximate BPE tokenizer when loaded with `use_fast=False` and
 * no `tokenizer.json` is present — verified, and wrong: e.g. it segments
 * "hello" as `['he','ll','o']` where the real unigram model gives the single
 * piece `'hello'`).
 */
function loadJson<T>(relativePath: string): T {
  return JSON.parse(readFileSync(new URL(relativePath, import.meta.url), "utf8")) as T;
}

const vocab = loadJson<TokenizerVocab>("./data/sarashina2.2-1b-instruct.vocab.json");

interface Fixture {
  name: string;
  text: string;
  ids: number[];
  decoded: string;
}
const fixtures = loadJson<Fixture[]>("./data/sarashina2.2-1b-instruct.fixtures.json");

describe("tokenizer / vocab sanity", () => {
  it("loaded the real Sarashina2.2-1B-Instruct vocabulary", () => {
    expect(vocab.vocabSize).toBe(102400);
    expect(vocab.pieces).toHaveLength(102400);
    expect(vocab.byteFallback).toBe(true);
    expect(vocab.normalizer).toEqual({
      name: "identity",
      addDummyPrefix: false,
      removeExtraWhitespaces: false,
      escapeWhitespaces: true,
    });
  });

  it("has at least 50 fixture cases (issue #101 completion condition)", () => {
    expect(fixtures.length).toBeGreaterThanOrEqual(50);
  });
});

describe("tokenizer / fixture parity", () => {
  const tokenizer = new SentencePieceTokenizer(vocab);

  for (const fixture of fixtures) {
    it(`encodes "${fixture.name}" to the exact real-tokenizer id sequence`, () => {
      expect(tokenizer.encode(fixture.text)).toEqual(fixture.ids);
    });

    it(`decodes "${fixture.name}" back to the exact real-tokenizer text`, () => {
      expect(tokenizer.decode(fixture.ids)).toBe(fixture.decoded);
    });
  }
});

describe("tokenizer / documented behaviors", () => {
  const tokenizer = new SentencePieceTokenizer(vocab);

  it("does not apply NFKC or any other Unicode normalization (identity normalizer)", () => {
    // "が" precomposed (U+304C) vs "か" + combining voiced sound mark
    // (U+304B U+3099) — a normalizing tokenizer would merge these to one id.
    const precomposed = tokenizer.encode("が");
    const decomposed = tokenizer.encode("が");
    expect(precomposed).toEqual([306]);
    expect(decomposed).toEqual([22105]);
    expect(precomposed).not.toEqual(decomposed);
  });

  it("replaces only the literal ASCII space with ▁, not other whitespace", () => {
    // Confirmed against the real model: \t and \n have no NORMAL piece and
    // always fall back to bytes — they are never merged into ▁.
    expect(tokenizer.encode(" ")).toEqual([271]); // ▁
    expect(tokenizer.encode("\t")).toEqual([24]); // <0x09>
    expect(tokenizer.encode("\n")).toEqual([25]); // <0x0A>
  });

  it("does not collapse or trim whitespace runs", () => {
    const ids = tokenizer.encode("  double  space  ");
    expect(ids).toEqual([271, 2738, 271, 1272, 271, 271]);
    expect(tokenizer.decode(ids)).toBe("  double  space  ");
  });

  it("adds no implicit leading ▁ (add_dummy_prefix is false for this model)", () => {
    expect(tokenizer.encode("hello")).toEqual([15407]);
    expect(tokenizer.encode(" hello")).toEqual([271, 15407]);
  });

  it("byte-fallbacks an uncovered astral codepoint into its UTF-8 bytes", () => {
    const ids = tokenizer.encode("😀");
    expect(ids).toEqual([255, 174, 167, 143]);
    expect(tokenizer.decode(ids)).toBe("😀");
  });

  it("mixes byte-fallback and a following normal piece without losing either", () => {
    const ids = tokenizer.encode("😀emoji");
    expect(ids).toEqual([255, 174, 167, 143, 44733]);
    expect(tokenizer.decode(ids)).toBe("😀emoji");
  });

  it("does not let many accumulated byte-fallback edges corrupt a later real segmentation choice", () => {
    // Regression test for a real bug found while building this tokenizer: an
    // early version scored each byte-fallback edge at a large constant
    // (-1e6) *per byte*. That never competes locally against a real piece
    // (byte edges only exist where nothing else covers the codepoint at
    // all), but summed over the dozen-plus uncovered codepoints in this
    // sweep it pushed `bestScore` to around -5×10⁷ — a magnitude where
    // float32's own precision (~±6 absolute there) is coarser than the ~1-2
    // point gaps between competing *real* segmentations later in the
    // string, silently flipping them. The fix uses SentencePiece's real
    // formula (`min_score() - 10.0`, once per codepoint, not per byte —
    // verified against `unigram_model.cc`), which keeps accumulated
    // magnitudes small enough that this stays correct.
    //
    // U+3040..U+3099 includes several codepoints with no NORMAL-piece
    // coverage (e.g. U+3040, U+3097, U+3098 are unassigned; U+3099/U+309A
    // are rare combining marks), so this sweep forces repeated byte
    // fallback, then continues into ordinary hiragana ("ただち" etc.) where
    // the real tokenizer's segmentation choice must still come through
    // uncorrupted.
    const text = Array.from({ length: 300 }, (_, i) => String.fromCodePoint(0x3040 + (i % 90))).join(
      "",
    );
    const fixture = fixtures.find((f) => f.name === "byte_fallback_dense_sweep");
    if (!fixture) throw new Error("fixture byte_fallback_dense_sweep not found");
    expect(fixture.text).toBe(text);
    expect(tokenizer.encode(text)).toEqual(fixture.ids);
  });

  it("does not special-case a literal added-token string inside SentencePiece's own search", () => {
    // This is the behavior that makes the added-token pre-split necessary in
    // the first place: absent it, "<|system|>" would decompose byte-by-byte
    // instead of resolving to id 7.
    expect(tokenizer.encode("<|system|>")).toEqual([7]);
  });

  it("splits added tokens out of surrounding text at exact literal boundaries", () => {
    expect(tokenizer.encode("prefix<|system|>suffix")).toEqual([
      14209, // "prefix" alone
      7, // <|system|>
      29607, // "suffix" alone
    ]);
  });

  it("recognizes consecutive added tokens with nothing between them", () => {
    expect(tokenizer.encode("<s><|system|></s>")).toEqual([1, 7, 2]);
  });

  it("round-trips a full chat-template-shaped exchange", () => {
    const text = "<|system|>You are Alibi.</s><|user|>今日の天気は?</s><|assistant|>";
    const ids = tokenizer.encode(text);
    expect(ids[0]).toBe(7); // <|system|>
    expect(ids).toContain(2); // </s>
    expect(ids[ids.length - 1]).toBe(8); // <|assistant|>
    expect(tokenizer.decode(ids)).toBe(text);
  });

  it("inserts no separator around decoded added tokens (spaces_between_special_tokens: false)", () => {
    expect(tokenizer.decode([1, 7, 2])).toBe("<s><|system|></s>");
  });

  it("throws on an out-of-range id rather than silently returning garbage", () => {
    expect(() => tokenizer.decode([999_999])).toThrow(/out of range/);
  });

  it("never produces the <unk> id when encoding arbitrary text (byte fallback covers everything)", () => {
    const stress = "😀🇯🇵👍🏽\t\n　ＡＢＣ";
    expect(tokenizer.encode(stress)).not.toContain(vocab.unkId);
  });
});
