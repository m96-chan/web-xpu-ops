import { describe, expect, it } from "vitest";
import { TINY_FIXTURE_CONFIG } from "./config.js";
import { loadTinyFixtureQ8 } from "./fixture-q8.js";

describe("loadTinyFixtureQ8", () => {
  it("matches llm/config.ts's TINY_FIXTURE_CONFIG", () => {
    const { config } = loadTinyFixtureQ8();
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
    const f = loadTinyFixtureQ8();
    expect(f.prefillLogits).toHaveLength(f.promptTokens.length);
    expect(f.decodeLogits).toHaveLength(f.decodeTokens.length);
    for (const row of [...f.prefillLogits, ...f.decodeLogits]) {
      expect(row).toHaveLength(f.config.vocabSize);
    }
  });

  it("gives every layer its own distinct quantized codes", () => {
    const { weights } = loadTinyFixtureQ8();
    const [l0, l1] = weights.layers as [typeof weights.layers[0], typeof weights.layers[0]];
    expect(Array.from(l0.wq.codes)).not.toEqual(Array.from(l1.wq.codes));
  });

  it("has real (non-zero) int8 codes and finite, positive scales", () => {
    const { weights } = loadTinyFixtureQ8();
    let sumAbs = 0;
    for (const c of weights.embedTokens.codes) sumAbs += Math.abs(c);
    expect(sumAbs).toBeGreaterThan(0);
    for (const s of weights.embedTokens.scale) {
      expect(Number.isFinite(s)).toBe(true);
      expect(s).toBeGreaterThan(0);
    }
  });

  it("keeps every int8 code within quantize()'s [-127, 127] range", () => {
    const { weights } = loadTinyFixtureQ8();
    for (const c of weights.lmHead.codes) {
      expect(c).toBeGreaterThanOrEqual(-127);
      expect(c).toBeLessThanOrEqual(127);
    }
  });
});
