import { describe, expect, it } from "vitest";
import { sampleNext } from "../sampler.js";
import { LineFormatConstraint, type LineFormatSpec } from "./line-format.js";
import type { TokenCodec } from "./token-codec.js";

/**
 * A small greedy longest-match tokenizer, standing in for issue #101's real
 * one (not merged into `main` — this ISSUE is independent of it, see the PR
 * body). It exists only so the constraint's token-level forward-matching has
 * something non-trivial to walk: several vocab entries are multi-character
 * ("policy", "al", "low"), so an enum choice like "allow" spans two tokens
 * and genuinely has a "partially matched" state.
 *
 * Token 50 is EOS and decodes to the empty string, so it never shows up in
 * concatenated output.
 */
const VOCAB = [
  "\n", // 0
  " ", // 1
  ":", // 2
  "policy", // 3
  "topic", // 4
  "al", // 5
  "low", // 6
  "de", // 7
  "ny", // 8
  "review", // 9
  "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", // 10-19
  "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", // 20-29
  "u", "v", "w", "x", "y", "z", // 30-35
  "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", // 36-45
  ".", ",", "!", "?", // 46-49
  "", // 50 = EOS
];
const EOS = 50;
const MAX_PIECE_LEN = Math.max(...VOCAB.map((p) => p.length));

function mockCodec(): TokenCodec {
  return {
    vocabSize: VOCAB.length,
    idToToken: (id) => {
      const piece = VOCAB[id];
      if (piece === undefined) throw new Error(`mockCodec: id ${id} out of range`);
      return piece;
    },
    encode: (text) => {
      const ids: number[] = [];
      let i = 0;
      outer: while (i < text.length) {
        for (let len = Math.min(MAX_PIECE_LEN, text.length - i); len >= 1; len -= 1) {
          const piece = text.slice(i, i + len);
          const id = VOCAB.indexOf(piece);
          // VOCAB[50] is "" — indexOf would match it against nothing since
          // `piece` here always has length >= 1, so EOS can never be
          // produced by encode(). That is intentional: EOS is a control
          // token, not text a caller writes out.
          if (id !== -1) {
            ids.push(id);
            i += len;
            continue outer;
          }
        }
        throw new Error(`mockCodec.encode: no token covers ${JSON.stringify(text[i])} at ${i}`);
      }
      return ids;
    },
  };
}

/** `policy: <allow|deny|review>\ntopic: <free text, no newline, max 6 chars>`. */
function policyTopicSpec(): LineFormatSpec {
  return {
    segments: [
      { kind: "literal", text: "policy: " },
      { kind: "enum", choices: ["allow", "deny", "review"] },
      { kind: "literal", text: "\ntopic: " },
      { kind: "freeText", forbiddenChars: ["\n"], maxLength: 6 },
    ],
    eosTokenId: EOS,
  };
}

const decode = (codec: TokenCodec, tokens: readonly number[]) => tokens.map((id) => codec.idToToken(id)).join("");

describe("LineFormatConstraint", () => {
  const codec = mockCodec();

  it("allows only the fixed literal's next token while inside a literal segment", () => {
    const constraint = new LineFormatConstraint(codec, policyTopicSpec());
    // "policy" (3), ":" (2), " " (1) — one token in.
    expect(constraint.nextAllowed([3])).toEqual(new Set([2]));
    expect(constraint.nextAllowed([3, 2])).toEqual(new Set([1]));
  });

  it("offers every enum choice's first token at the start of the enum segment", () => {
    const constraint = new LineFormatConstraint(codec, policyTopicSpec());
    // "allow" -> al(5); "deny" -> de(7); "review" -> review(9) as one token.
    expect(constraint.nextAllowed([3, 2, 1])).toEqual(new Set([5, 7, 9]));
  });

  // (a) the required observation: allowed set after the enum is *partway* matched.
  it("narrows to only the tokens completing the in-progress choice, once the enum is partway matched", () => {
    const constraint = new LineFormatConstraint(codec, policyTopicSpec());
    // Committed "al" (5) — only "allow" continues with "low" (6); "deny" and
    // "review" are no longer reachable from here.
    expect(constraint.nextAllowed([3, 2, 1, 5])).toEqual(new Set([6]));
  });

  it("rejects a prefix that diverges from the required literal text", () => {
    const constraint = new LineFormatConstraint(codec, policyTopicSpec());
    // "policy" (3) is right, but the caller supplied "de" (7) where ":" (2)
    // must come next — a prefix that could only exist if something upstream
    // ignored `nextAllowed`'s mask.
    expect(() => constraint.nextAllowed([3, 7])).toThrow(/diverges/);
  });

  it("moves past the enum into the next literal once a choice is fully matched", () => {
    const constraint = new LineFormatConstraint(codec, policyTopicSpec());
    // "policy: " + "allow" (al, low) — enum just completed.
    expect(constraint.nextAllowed([3, 2, 1, 5, 6])).toEqual(new Set([0])); // "\n"
    // A one-token choice ("review") resolves immediately too.
    expect(constraint.nextAllowed([3, 2, 1, 9])).toEqual(new Set([0]));
  });

  // (b) the required observation: the newline token is excluded in free text.
  it("excludes the forbidden newline token once free text starts", () => {
    const constraint = new LineFormatConstraint(codec, policyTopicSpec());
    // Full prefix through the separator literal: "policy: allow\ntopic: ".
    const prefix = [3, 2, 1, 5, 6, 0, 4, 2, 1];
    const allowed = constraint.nextAllowed(prefix);
    expect(allowed).not.toBeNull();
    expect(allowed!.has(0)).toBe(false); // "\n"
    expect(allowed!.has(10)).toBe(true); // "a" — ordinary free-text token
    expect(allowed!.has(EOS)).toBe(true); // free text may end immediately
  });

  // (c) the required observation: reaching the max length forces termination.
  it("forces EOS once free text reaches its max length", () => {
    const constraint = new LineFormatConstraint(codec, policyTopicSpec());
    const prefix = [3, 2, 1, 5, 6, 0, 4, 2, 1, 10, 11, 12, 13, 14, 15]; // "abcdef" — 6 chars, maxLength=6
    expect(constraint.nextAllowed(prefix)).toEqual(new Set([EOS]));
  });

  it("signals completion (empty allowed set) once EOS has been emitted", () => {
    const constraint = new LineFormatConstraint(codec, policyTopicSpec());
    const prefix = [3, 2, 1, 5, 6, 0, 4, 2, 1, 10, 11, 12, 13, 14, 15, EOS];
    expect(constraint.nextAllowed(prefix)).toEqual(new Set());
  });

  it("excludes a multi-character token whose length would overshoot the remaining free-text budget", () => {
    const constraint = new LineFormatConstraint(codec, policyTopicSpec());
    // 5 of 6 chars used; "low" (3 chars) would overshoot, "a" (1 char) would not.
    const prefix = [3, 2, 1, 5, 6, 0, 4, 2, 1, 10, 11, 12, 13, 14]; // "abcde"
    const allowed = constraint.nextAllowed(prefix);
    expect(allowed!.has(6)).toBe(false); // "low"
    expect(allowed!.has(10)).toBe(true); // "a"
    expect(allowed!.has(EOS)).toBe(true);
  });

  it("rejects a spec whose freeText segment is not last", () => {
    const spec: LineFormatSpec = {
      segments: [
        { kind: "freeText", forbiddenChars: [], maxLength: 4 },
        { kind: "literal", text: "x" },
      ],
      eosTokenId: EOS,
    };
    expect(() => new LineFormatConstraint(codec, spec)).toThrow(/last segment/);
  });

  it("rejects enum choices whose tokenizations are not prefix-free (shorter choice defined first)", () => {
    // A pure single-character tokenizer, so "a" is unambiguously a token
    // prefix of "ab" at the token level, not just the character level.
    const charVocab = ["a", "b", ""];
    const charCodec: TokenCodec = {
      vocabSize: charVocab.length,
      idToToken: (id) => charVocab[id]!,
      encode: (text) => Array.from(text, (ch) => charVocab.indexOf(ch)),
    };
    const spec: LineFormatSpec = {
      segments: [{ kind: "enum", choices: ["a", "ab"] }],
      eosTokenId: 2,
    };
    expect(() => new LineFormatConstraint(charCodec, spec)).toThrow(/prefix/);
  });

  it("rejects enum choices whose tokenizations are not prefix-free (longer choice defined first)", () => {
    // Same ambiguity as above, reached through the other code path: the
    // conflict is only detectable *after* "a"'s single token is consumed and
    // its node turns out to already have a child from "ab".
    const charVocab = ["a", "b", ""];
    const charCodec: TokenCodec = {
      vocabSize: charVocab.length,
      idToToken: (id) => charVocab[id]!,
      encode: (text) => Array.from(text, (ch) => charVocab.indexOf(ch)),
    };
    const spec: LineFormatSpec = {
      segments: [{ kind: "enum", choices: ["ab", "a"] }],
      eosTokenId: 2,
    };
    expect(() => new LineFormatConstraint(charCodec, spec)).toThrow(/prefix/);
  });

  it("rejects an enum segment with no choices", () => {
    const spec: LineFormatSpec = { segments: [{ kind: "enum", choices: [] }], eosTokenId: EOS };
    expect(() => new LineFormatConstraint(codec, spec)).toThrow(/at least one choice/);
  });

  it("rejects an enum choice that encodes to zero tokens", () => {
    const spec: LineFormatSpec = { segments: [{ kind: "enum", choices: [""] }], eosTokenId: EOS };
    expect(() => new LineFormatConstraint(codec, spec)).toThrow(/zero tokens/);
  });

  it("rejects a literal segment that encodes to zero tokens", () => {
    const spec: LineFormatSpec = { segments: [{ kind: "literal", text: "" }], eosTokenId: EOS };
    expect(() => new LineFormatConstraint(codec, spec)).toThrow(/zero tokens/);
  });

  it("rejects a prefix token that matches no enum choice", () => {
    const constraint = new LineFormatConstraint(codec, policyTopicSpec());
    // Past "policy: ", token 10 ("a") starts none of allow/deny/review.
    expect(() => constraint.nextAllowed([3, 2, 1, 10])).toThrow(/matches no enum choice/);
  });

  it("rejects a free-text token outside the allowed set (e.g. the forbidden newline)", () => {
    const constraint = new LineFormatConstraint(codec, policyTopicSpec());
    const prefix = [3, 2, 1, 5, 6, 0, 4, 2, 1, 0]; // "\n" injected into free text
    expect(() => constraint.nextAllowed(prefix)).toThrow(/not allowed in free text/);
  });

  it("forces EOS once a spec with no freeText segment finishes its last segment", () => {
    const spec: LineFormatSpec = { segments: [{ kind: "literal", text: "x" }], eosTokenId: EOS };
    const constraint = new LineFormatConstraint(codec, spec);
    const xTokens = codec.encode("x");
    expect(constraint.nextAllowed(xTokens)).toEqual(new Set([EOS]));
    expect(constraint.nextAllowed([...xTokens, EOS])).toEqual(new Set());
  });

  it("rejects a token appended after a no-freeText spec is already complete", () => {
    const spec: LineFormatSpec = { segments: [{ kind: "literal", text: "x" }], eosTokenId: EOS };
    const constraint = new LineFormatConstraint(codec, spec);
    const xTokens = codec.encode("x");
    expect(() => constraint.nextAllowed([...xTokens, 10])).toThrow(/after the format is complete/);
  });
});

/**
 * Seeded PRNG (mulberry32) — deterministic across runs, unlike `Math.random`.
 * Used only to fabricate logits vectors for the property test below; it has
 * nothing to do with the sampler's own `rng` injection point.
 */
function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

describe("LineFormatConstraint + greedy sampling (property)", () => {
  const codec = mockCodec();
  const format = /^policy: (allow|deny|review)\ntopic: [^\n]{0,6}$/;

  for (const seed of [1, 2, 3, 4, 5, 6, 7, 8]) {
    it(`always produces well-formed output under greedy decoding with masking (seed=${seed})`, () => {
      const rng = mulberry32(seed);
      // One fixed random logits vector reused at every step: the constraint
      // is what has to keep the output well-formed, not a lucky logits shape.
      const logits = Array.from({ length: codec.vocabSize }, () => rng() * 20 - 10);
      const constraint = new LineFormatConstraint(codec, policyTopicSpec());

      const tokens: number[] = [];
      const SAFETY_CAP = 64; // generous: literal(8) + enum(2) + literal(4) + freeText(<=6) + eos
      for (let step = 0; step < SAFETY_CAP; step += 1) {
        const allowed = constraint.nextAllowed(tokens);
        if (allowed !== null && allowed.size === 0) break; // EOS was just emitted
        const next = sampleNext(logits, tokens, { mode: "greedy" }, constraint);
        tokens.push(next);
      }

      expect(tokens[tokens.length - 1]).toBe(EOS);
      const text = decode(codec, tokens);
      expect(text).toMatch(format);
    });
  }
});
