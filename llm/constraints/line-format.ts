import type { Constraint } from "../sampler.js";
import type { TokenCodec } from "./token-codec.js";

/**
 * One stage of a line-format spec.
 *
 * - `literal`: fixed text that must appear verbatim (e.g. `"policy: "`).
 * - `enum`: exactly one of `choices`, chosen by the model.
 * - `freeText`: open text, constrained only by a forbidden-character set and
 *   a maximum length. **Must be the last segment** — see `LineFormatConstraint`.
 */
export type LineFormatSegment =
  | { readonly kind: "literal"; readonly text: string }
  | { readonly kind: "enum"; readonly choices: readonly string[] }
  | { readonly kind: "freeText"; readonly forbiddenChars: readonly string[]; readonly maxLength: number };

export interface LineFormatSpec {
  readonly segments: readonly LineFormatSegment[];
  readonly eosTokenId: number;
}

interface TrieNode {
  readonly children: Map<number, TrieNode>;
  terminal: boolean;
}

/**
 * Builds a token-id trie over `choices`, each tokenized once via
 * `codec.encode`. This is the "candidate string, tokenized, forward-matched"
 * approach from the ISSUE: walking the trie one committed token at a time
 * gives exactly the set of next-token ids consistent with the choices still
 * reachable from the current prefix.
 *
 * Choices must be **token-prefix-free**: no choice's token sequence may be a
 * strict prefix of another's. Without that, reaching a terminal node would
 * be ambiguous — "choice complete, move on" and "still extending a longer
 * choice" would both be true at once, and forward-matching alone cannot
 * tell them apart (that would need lookahead, i.e. picking a token now based
 * on what the *model* will emit two tokens later). Rather than guess, this
 * rejects the spec at construction time.
 */
function buildTrie(codec: TokenCodec, choices: readonly string[]): TrieNode {
  if (choices.length === 0) {
    throw new Error("LineFormatConstraint: an enum segment needs at least one choice");
  }
  const root: TrieNode = { children: new Map(), terminal: false };
  for (const choice of choices) {
    const tokens = codec.encode(choice);
    if (tokens.length === 0) {
      throw new Error(`LineFormatConstraint: enum choice ${JSON.stringify(choice)} encodes to zero tokens`);
    }
    let node = root;
    for (const id of tokens) {
      if (node.terminal) {
        throw new Error(
          `LineFormatConstraint: enum choice ${JSON.stringify(choice)} extends another, already-complete ` +
            "choice's token sequence — ambiguous forward-match. Choices must be token-prefix-free.",
        );
      }
      let next = node.children.get(id);
      if (!next) {
        next = { children: new Map(), terminal: false };
        node.children.set(id, next);
      }
      node = next;
    }
    if (node.children.size > 0) {
      throw new Error(
        `LineFormatConstraint: enum choice ${JSON.stringify(choice)} is a token prefix of another choice — ` +
          "ambiguous forward-match. Choices must be token-prefix-free.",
      );
    }
    node.terminal = true;
  }
  return root;
}

type SegmentPlan =
  | { readonly kind: "literal"; readonly tokens: readonly number[] }
  | { readonly kind: "enum"; readonly trie: TrieNode }
  | { readonly kind: "freeText"; readonly allowed: ReadonlyMap<number, number>; readonly maxLength: number };

/**
 * A `Constraint` (see `llm/sampler.ts`) that enforces a fixed line shape:
 * literal text, an enum choice, more literal text, free text — in the order
 * given by `spec.segments`, ending in `spec.eosTokenId`.
 *
 * `nextAllowed` is stateless by design (it re-derives position from
 * `prefixTokens` on every call, matching the `Constraint` interface) rather
 * than tracking an internal cursor — an instance can be reused concurrently
 * across independent generations without cross-talk.
 */
export class LineFormatConstraint implements Constraint {
  private readonly plan: readonly SegmentPlan[];
  private readonly eosTokenId: number;

  constructor(codec: TokenCodec, spec: LineFormatSpec) {
    this.eosTokenId = spec.eosTokenId;
    this.plan = spec.segments.map((segment, index) => {
      if (segment.kind === "literal") {
        const tokens = codec.encode(segment.text);
        if (tokens.length === 0) {
          throw new Error(`LineFormatConstraint: literal segment ${index} encodes to zero tokens`);
        }
        return { kind: "literal", tokens };
      }
      if (segment.kind === "enum") {
        return { kind: "enum", trie: buildTrie(codec, segment.choices) };
      }
      // freeText: open-ended, so it cannot be followed by anything this
      // state machine still knows how to resume into — it must be last.
      if (index !== spec.segments.length - 1) {
        throw new Error("LineFormatConstraint: a freeText segment must be the last segment");
      }
      const forbidden = new Set(segment.forbiddenChars);
      const allowed = new Map<number, number>();
      for (let id = 0; id < codec.vocabSize; id += 1) {
        if (id === spec.eosTokenId) continue;
        const piece = codec.idToToken(id);
        let ok = true;
        for (const ch of piece) {
          if (forbidden.has(ch)) {
            ok = false;
            break;
          }
        }
        if (ok) allowed.set(id, piece.length);
      }
      return { kind: "freeText", allowed, maxLength: segment.maxLength };
    });
  }

  nextAllowed(prefixTokens: readonly number[]): ReadonlySet<number> | null {
    let pos = 0;
    for (const step of this.plan) {
      if (step.kind === "literal") {
        for (const expected of step.tokens) {
          if (pos >= prefixTokens.length) return new Set([expected]);
          if (prefixTokens[pos] !== expected) {
            throw new Error(
              `LineFormatConstraint: prefix token ${prefixTokens[pos]} at position ${pos} diverges from the required literal (expected ${expected})`,
            );
          }
          pos += 1;
        }
        continue;
      }

      if (step.kind === "enum") {
        let node = step.trie;
        while (!(node.terminal && node.children.size === 0)) {
          if (pos >= prefixTokens.length) return new Set(node.children.keys());
          const id = prefixTokens[pos]!;
          const next = node.children.get(id);
          if (!next) {
            throw new Error(`LineFormatConstraint: token ${id} at position ${pos} matches no enum choice`);
          }
          pos += 1;
          node = next;
        }
        continue;
      }

      // freeText — always the last segment (enforced at construction).
      let length = 0;
      for (; pos < prefixTokens.length; pos += 1) {
        const id = prefixTokens[pos]!;
        if (id === this.eosTokenId) {
          return new Set(); // format complete; nothing may legally follow
        }
        const pieceLength = step.allowed.get(id);
        if (pieceLength === undefined) {
          throw new Error(`LineFormatConstraint: token ${id} at position ${pos} is not allowed in free text`);
        }
        length += pieceLength;
      }
      const remaining = step.maxLength - length;
      const result = new Set<number>();
      if (remaining > 0) {
        for (const [id, pieceLength] of step.allowed) {
          if (pieceLength <= remaining) result.add(id);
        }
      }
      result.add(this.eosTokenId); // free text may always end early, and must end once `remaining <= 0`
      return result;
    }

    // Every segment fully consumed with tokens still left over (only reachable
    // when the spec has no freeText segment) — the format is otherwise done.
    if (pos >= prefixTokens.length) return new Set([this.eosTokenId]);
    if (prefixTokens[pos] === this.eosTokenId) return new Set();
    throw new Error(`LineFormatConstraint: unexpected token ${prefixTokens[pos]} after the format is complete`);
  }
}
