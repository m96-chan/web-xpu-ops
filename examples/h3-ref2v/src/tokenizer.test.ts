/**
 * H3's tokenizer is one this repository already has.
 *
 * Issue #212. `ref2va`'s presentation needs Qwen2's byte-level BPE, and
 * `llm/tokenizer-bpe.ts` is exactly that — written for Qwen3-4B, with its
 * vocabulary already committed at `llm/data/`. Whether the two tokenizers agree
 * is a **measurement**, not an assumption, and this is it: every text segment
 * `tools/gen_presentation_golden.py` recorded from H3's own tokenizer, and the
 * four vision token ids the vision blocks are built from.
 *
 * If they ever diverge — a checkpoint with an extra added token, a different
 * split pattern — this fails here rather than producing an embedding of the
 * wrong thing forty layers later.
 */
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { ByteLevelBpeTokenizer, type BpeVocab } from "../../../llm/tokenizer-bpe.js";

const vocab = JSON.parse(
  readFileSync(fileURLToPath(new URL("../../../llm/data/qwen-qwen3-4b.bpe-vocab.json", import.meta.url)), "utf8"),
) as BpeVocab;
const golden = JSON.parse(
  readFileSync(new URL("../fixtures/presentation.json", import.meta.url), "utf8"),
) as {
  visionStart: number; visionEnd: number; imagePad: number; videoPad: number;
  segments: Record<string, number[]>;
};

const tokenizer = new ByteLevelBpeTokenizer(vocab);

describe("h3 ref2v / tokenizer", () => {
  it("has segments to check", () => {
    // Labels, timestamps and prompts — an empty map would make the loop below
    // a decoration.
    expect(Object.keys(golden.segments).length).toBeGreaterThan(8);
  });

  it("reproduces every segment H3's own tokenizer produced", () => {
    for (const [text, ids] of Object.entries(golden.segments)) {
      expect(tokenizer.encode(text), JSON.stringify(text)).toEqual(ids);
    }
  });

  it("agrees on the four vision token ids", () => {
    // The vision blocks are `[start] + [pad] * n + [end]`, so these four ids
    // are the whole of what a reference contributes to the token stream.
    const lookup = (token: string): number | undefined =>
      vocab.vocab[token] ?? vocab.addedTokens?.find((t) => t.content === token)?.id;
    expect(lookup("<|vision_start|>")).toBe(golden.visionStart);
    expect(lookup("<|vision_end|>")).toBe(golden.visionEnd);
    expect(lookup("<|image_pad|>")).toBe(golden.imagePad);
    expect(lookup("<|video_pad|>")).toBe(golden.videoPad);
  });
});
