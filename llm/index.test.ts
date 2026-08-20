import { describe, expect, it } from "vitest";
import {
  LineFormatConstraint,
  LlamaEngineQ8,
  loadWeightsQ8FromUrl,
  sampleNext,
  SentencePieceTokenizer,
} from "./index.js";

/**
 * Issue #106's "配線のみ" half: `llm/tokenizer.ts` (#101), `llm/sampler.ts` +
 * `llm/constraints/line-format.ts` (#102) and `llm/browser-weights.ts` (this
 * issue) all landed without ever being re-exported from `llm/index.ts` — the
 * known gap #106's own text calls out. This is a wiring test, not a
 * behavioral one (each module already has its own test file); it exists so
 * that a future PR that forgets to re-export a new `llm/` module fails here
 * instead of silently shipping an engine a caller cannot actually assemble
 * from one import.
 *
 * Every name below is imported from `./index.js`, not from the module it is
 * defined in — that is the entire thing under test. A missing export makes
 * this whole file fail to load (Node's ESM loader throws
 * "does not provide an export named ...", per rule 1: an import-time failure
 * naming the exact missing symbol is Red for the right reason), not merely
 * one assertion inside it.
 */
describe("llm/index.ts wiring", () => {
  it("exports the tokenizer (issue #101)", () => {
    expect(typeof SentencePieceTokenizer).toBe("function");
  });

  it("exports the sampler (issue #102)", () => {
    expect(typeof sampleNext).toBe("function");
    const logits = new Float32Array([0, 5, -1]);
    expect(sampleNext(logits, [], { mode: "greedy" })).toBe(1);
  });

  it("exports line-format constraints (issue #102)", () => {
    expect(typeof LineFormatConstraint).toBe("function");
  });

  it("exports the browser weight loader (this issue)", () => {
    expect(typeof loadWeightsQ8FromUrl).toBe("function");
  });

  it("still exports the int8 engine (issue #105, already wired before this issue)", () => {
    expect(typeof LlamaEngineQ8).toBe("function");
  });
});
