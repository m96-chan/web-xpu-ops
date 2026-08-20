import { describe, expect, it } from "vitest";
import {
  buildLlamaWeightsQ8,
  llamaConfigFromManifest,
  type WeightManifestEntry,
} from "./weights-q8-io.js";

/**
 * `weights-q8-io.ts` is the manifest-parsing core shared by `fixture-q8.ts`
 * (the tiny fixture, `tiny_q8.*` filenames, plus logits) and the real-model
 * loader (`llm/tools/convert_weights.py`'s `manifest.json` / `weights.*.bin`,
 * no logits) — issue #105's converter output and its tiny-fixture
 * counterpart share this exact manifest shape (`{name, kind, shape, offsets}`)
 * by design (`convert_weights.py`'s module doc), so one parser serves both
 * rather than two copies that could silently drift apart.
 */

const MANIFEST_CONFIG = {
  numLayers: 1,
  hiddenSize: 4,
  numHeads: 2,
  numKvHeads: 1,
  headDim: 2,
  ffnHidden: 3,
  vocabSize: 5,
  ropeTheta: 10000,
  rmsNormEps: 1e-5,
  tieEmbeddings: false,
};

describe("llamaConfigFromManifest", () => {
  it("carries every manifest config field into LlamaConfig, plus the given maxSeqLen", () => {
    const config = llamaConfigFromManifest(MANIFEST_CONFIG, 64);
    expect(config).toEqual({ ...MANIFEST_CONFIG, maxSeqLen: 64 });
  });
});

describe("buildLlamaWeightsQ8", () => {
  // hiddenSize=4, numHeads*headDim=4 (wq), numKvHeads*headDim=2 (wk/wv),
  // ffnHidden=3, vocabSize=5 — small but every dimension distinct enough that
  // a swapped offset or shape reads the wrong bytes rather than coincidentally
  // matching ones (the gap `engine-q8.wgsl.test.ts`'s wo mutation exposed).
  const { hiddenSize, numHeads, numKvHeads, headDim, ffnHidden, vocabSize } = MANIFEST_CONFIG;
  const qDim = numHeads * headDim;
  const kvDim = numKvHeads * headDim;

  function linearEntry(name: string, n: number, k: number, codesOffset: number, scaleOffset: number): WeightManifestEntry {
    return { name, kind: "quant", shape: [n, k], codesOffset, scaleOffset };
  }
  function normEntry(name: string, n: number, offset: number): WeightManifestEntry {
    return { name, kind: "norm", shape: [n], offset };
  }

  it("reads codes/scale/norm views at the manifest's own offsets, per weight", () => {
    const entries: WeightManifestEntry[] = [
      linearEntry("embedTokens", vocabSize, hiddenSize, 0, 0),
      normEntry("layers.0.attnNorm", hiddenSize, 0),
      linearEntry("layers.0.wq", qDim, hiddenSize, vocabSize * hiddenSize, vocabSize),
      linearEntry("layers.0.wk", kvDim, hiddenSize, vocabSize * hiddenSize + qDim * hiddenSize, vocabSize + qDim),
      linearEntry("layers.0.wv", kvDim, hiddenSize, vocabSize * hiddenSize + (qDim + kvDim) * hiddenSize, vocabSize + qDim + kvDim),
      linearEntry("layers.0.wo", hiddenSize, qDim, vocabSize * hiddenSize + (qDim + 2 * kvDim) * hiddenSize, vocabSize + qDim + 2 * kvDim),
      normEntry("layers.0.ffnNorm", hiddenSize, hiddenSize),
      linearEntry(
        "layers.0.wGate",
        ffnHidden,
        hiddenSize,
        vocabSize * hiddenSize + (qDim + 2 * kvDim) * hiddenSize + hiddenSize * qDim,
        vocabSize + qDim + 2 * kvDim + hiddenSize,
      ),
      linearEntry(
        "layers.0.wUp",
        ffnHidden,
        hiddenSize,
        vocabSize * hiddenSize + (qDim + 2 * kvDim) * hiddenSize + hiddenSize * qDim + ffnHidden * hiddenSize,
        vocabSize + qDim + 2 * kvDim + hiddenSize + ffnHidden,
      ),
      linearEntry(
        "layers.0.wDown",
        hiddenSize,
        ffnHidden,
        vocabSize * hiddenSize + (qDim + 2 * kvDim) * hiddenSize + hiddenSize * qDim + 2 * ffnHidden * hiddenSize,
        vocabSize + qDim + 2 * kvDim + hiddenSize + 2 * ffnHidden,
      ),
      normEntry("finalNorm", hiddenSize, 2 * hiddenSize),
      linearEntry(
        "lmHead",
        vocabSize,
        hiddenSize,
        vocabSize * hiddenSize + (qDim + 2 * kvDim) * hiddenSize + hiddenSize * qDim + 2 * ffnHidden * hiddenSize + hiddenSize * ffnHidden,
        vocabSize + qDim + 2 * kvDim + hiddenSize + 2 * ffnHidden + hiddenSize,
      ),
    ];

    const totalCodes = vocabSize * hiddenSize + (qDim + 2 * kvDim) * hiddenSize + hiddenSize * qDim + 2 * ffnHidden * hiddenSize + hiddenSize * ffnHidden + vocabSize * hiddenSize;
    const codes = Int8Array.from({ length: totalCodes }, (_, i) => ((i * 7) % 255) - 127);
    const totalScale = vocabSize + qDim + 2 * kvDim + hiddenSize + 2 * ffnHidden + hiddenSize + vocabSize;
    const scale = Float32Array.from({ length: totalScale }, (_, i) => i + 1);
    const totalNorm = 3 * hiddenSize;
    const norms = Float32Array.from({ length: totalNorm }, (_, i) => -(i + 1));

    const weights = buildLlamaWeightsQ8(entries, codes.buffer as ArrayBuffer, scale.buffer as ArrayBuffer, norms.buffer as ArrayBuffer, 1);

    // Spot-check a handful of weights land at the right bytes, not just the
    // right total length: `lmHead` reads the *tail* of the codes buffer.
    expect(Array.from(weights.embedTokens.codes)).toEqual(Array.from(codes.subarray(0, vocabSize * hiddenSize)));
    expect(Array.from(weights.lmHead.codes)).toEqual(Array.from(codes.subarray(codes.length - vocabSize * hiddenSize)));
    expect(Array.from(weights.layers[0]!.wq.codes)).toHaveLength(qDim * hiddenSize);
    expect(Array.from(weights.layers[0]!.wq.scale)).toEqual(Array.from(scale.subarray(vocabSize, vocabSize + qDim)));
    expect(Array.from(weights.layers[0]!.attnNorm)).toEqual(Array.from(norms.subarray(0, hiddenSize)));
    expect(Array.from(weights.finalNorm)).toEqual(Array.from(norms.subarray(2 * hiddenSize, 3 * hiddenSize)));
  });

  it("throws a message naming the missing weight when a required entry is absent", () => {
    // numLayers=0 so the first thing missing is embedTokens itself, not a
    // per-layer weight the loop would report first — pins the assertion to
    // one specific, unambiguous name rather than "whichever happens to be
    // checked first".
    expect(() => buildLlamaWeightsQ8([], new ArrayBuffer(0), new ArrayBuffer(0), new ArrayBuffer(0), 0)).toThrow(/embedTokens/);
  });
});
