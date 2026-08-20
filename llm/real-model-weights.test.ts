import { existsSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { SARASHINA_2_2_1B_CONFIG } from "./config.js";
import { loadConvertedWeightsQ8 } from "./real-model-weights.js";

/**
 * Smoke-checks `llm/tools/convert_weights.py`'s actual output against
 * `SARASHINA_2_2_1B_CONFIG` — issue #105's completion condition that a real
 * conversion "成立する". Skips when the converted directory is not present,
 * the same posture `harness/suite.ts#useGpu` takes for a missing GPU
 * adapter: this checks an artifact from a manual conversion run (the
 * checkpoint is gitignored, outside this repository, per issue #105's own
 * instructions), not something CI ever produces.
 */

const REAL_MODEL_Q8_DIR = process.env["ALIBI_SARASHINA_Q8_DIR"]
  ?? "/home/m96-chan/project/technologies.moe/alibi-ai/third_party/webgpu-weights/sarashina2.2-1b-alibi-v1-q8";

const present = existsSync(`${REAL_MODEL_Q8_DIR}/manifest.json`);

describe("loadConvertedWeightsQ8 / real Sarashina2.2-1B-alibi-v1 checkpoint", () => {
  it.skipIf(!present)("matches SARASHINA_2_2_1B_CONFIG's dimensions", () => {
    const { config } = loadConvertedWeightsQ8(REAL_MODEL_Q8_DIR, 4096);
    expect(config.numLayers).toBe(SARASHINA_2_2_1B_CONFIG.numLayers);
    expect(config.hiddenSize).toBe(SARASHINA_2_2_1B_CONFIG.hiddenSize);
    expect(config.numHeads).toBe(SARASHINA_2_2_1B_CONFIG.numHeads);
    expect(config.numKvHeads).toBe(SARASHINA_2_2_1B_CONFIG.numKvHeads);
    expect(config.headDim).toBe(SARASHINA_2_2_1B_CONFIG.headDim);
    expect(config.ffnHidden).toBe(SARASHINA_2_2_1B_CONFIG.ffnHidden);
    expect(config.vocabSize).toBe(SARASHINA_2_2_1B_CONFIG.vocabSize);
    expect(config.ropeTheta).toBe(SARASHINA_2_2_1B_CONFIG.ropeTheta);
    expect(config.rmsNormEps).toBe(SARASHINA_2_2_1B_CONFIG.rmsNormEps);
    expect(config.tieEmbeddings).toBe(SARASHINA_2_2_1B_CONFIG.tieEmbeddings);
  });

  it.skipIf(!present)("has real (non-zero, finite) codes, scales and norms", () => {
    const { weights } = loadConvertedWeightsQ8(REAL_MODEL_Q8_DIR, 4096);
    let sumAbs = 0;
    // A full 102,400 x 1,792 sweep is unnecessary to prove the table is real
    // data rather than all-zero padding; the first slice is enough.
    const sampleLength = Math.min(64 * weights.layers[0]!.attnNorm.length, weights.embedTokens.codes.length);
    for (let i = 0; i < sampleLength; i += 1) {
      sumAbs += Math.abs(weights.embedTokens.codes[i]!);
    }
    expect(sumAbs).toBeGreaterThan(0);
    for (const s of weights.embedTokens.scale.subarray(0, 64)) {
      expect(Number.isFinite(s)).toBe(true);
      expect(s).toBeGreaterThan(0);
    }
    for (const n of weights.layers[0]!.attnNorm) {
      expect(Number.isFinite(n)).toBe(true);
    }
  });
});
