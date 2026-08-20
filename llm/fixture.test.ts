import { describe, expect, it } from "vitest";
import { loadTinyFixture } from "./fixture.js";
import { TINY_FIXTURE_CONFIG } from "./config.js";

describe("loadTinyFixture", () => {
  it("matches llm/config.ts's TINY_FIXTURE_CONFIG", () => {
    const { config } = loadTinyFixture();
    expect(config.numLayers).toBe(TINY_FIXTURE_CONFIG.numLayers);
    expect(config.hiddenSize).toBe(TINY_FIXTURE_CONFIG.hiddenSize);
    expect(config.numHeads).toBe(TINY_FIXTURE_CONFIG.numHeads);
    expect(config.numKvHeads).toBe(TINY_FIXTURE_CONFIG.numKvHeads);
    expect(config.headDim).toBe(TINY_FIXTURE_CONFIG.headDim);
    expect(config.ffnHidden).toBe(TINY_FIXTURE_CONFIG.ffnHidden);
    expect(config.vocabSize).toBe(TINY_FIXTURE_CONFIG.vocabSize);
    expect(config.ropeTheta).toBe(TINY_FIXTURE_CONFIG.ropeTheta);
    expect(config.rmsNormEps).toBe(TINY_FIXTURE_CONFIG.rmsNormEps);
    expect(config.tieEmbeddings).toBe(TINY_FIXTURE_CONFIG.tieEmbeddings);
  });

  it("has one prefill logit row per prompt token and one decode logit row per decode token", () => {
    const f = loadTinyFixture();
    expect(f.prefillLogits).toHaveLength(f.promptTokens.length);
    expect(f.decodeLogits).toHaveLength(f.decodeTokens.length);
    for (const row of [...f.prefillLogits, ...f.decodeLogits]) {
      expect(row).toHaveLength(f.config.vocabSize);
    }
  });

  it("gives every layer its own distinct weight tensors", () => {
    const { weights } = loadTinyFixture();
    // Not merely non-null: two layers sharing an accidental alias would still
    // read back "present", so compare values across layers instead.
    const [l0, l1] = weights.layers as [typeof weights.layers[0], typeof weights.layers[0]];
    expect(Array.from(l0.wq)).not.toEqual(Array.from(l1.wq));
  });

  it("has real (non-zero, finite) weight values", () => {
    const { weights } = loadTinyFixture();
    let sumAbs = 0;
    for (const x of weights.embedTokens) {
      expect(Number.isFinite(x)).toBe(true);
      sumAbs += Math.abs(x);
    }
    expect(sumAbs).toBeGreaterThan(0);
  });
});
