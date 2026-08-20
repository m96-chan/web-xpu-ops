import type { LlamaConfig } from "./config.js";
import type { LlamaLayerWeightsQ8, LlamaWeightsQ8, QuantizedLinear } from "./weights-q8.js";

/**
 * The manifest shape `llm/tools/convert_weights.py` (the real-model
 * converter) and `llm/tools/gen_fixture_q8.py` (the tiny int8 fixture) both
 * write — a `[N, K]` quantized entry (codes + scale offsets) or an f32 norm
 * entry (a single offset) — and the parsing logic shared by
 * `fixture-q8.ts#loadTinyFixtureQ8` and the real-model loader, so a manifest
 * field renamed in one converter script is a compile error in the other
 * rather than a silent mismatch (rule 7: one convention, not two that can
 * drift).
 */
export interface QuantWeightManifestEntry {
  name: string;
  kind: "quant";
  /** `[N, K]` */
  shape: [number, number];
  /** Offset into the codes file, in int8 elements. */
  codesOffset: number;
  /** Offset into the scales file, in f32 elements. */
  scaleOffset: number;
}

export interface NormWeightManifestEntry {
  name: string;
  kind: "norm";
  shape: [number];
  /** Offset into the norms file, in f32 elements. */
  offset: number;
}

export type WeightManifestEntry = QuantWeightManifestEntry | NormWeightManifestEntry;

/** The `config` object both converter scripts write into their manifest, before `maxSeqLen` (a loader concern, not a conversion one) is added. */
export interface WeightManifestConfig {
  numLayers: number;
  hiddenSize: number;
  numHeads: number;
  numKvHeads: number;
  headDim: number;
  ffnHidden: number;
  vocabSize: number;
  ropeTheta: number;
  rmsNormEps: number;
  tieEmbeddings: boolean;
}

/** `WeightManifestConfig` plus `maxSeqLen`, which the manifest never carries (`convert_weights.py` converts weights, not a sequence-length policy — a caller supplies it, e.g. from `config.json`'s `max_position_embeddings`, or the tiny fixture's own generous headroom). */
export function llamaConfigFromManifest(config: WeightManifestConfig, maxSeqLen: number): LlamaConfig {
  return { ...config, maxSeqLen };
}

function elementsOf(shape: readonly number[]): number {
  return shape.reduce((a, b) => a * b, 1);
}

function f32View(buffer: ArrayBufferLike, offset: number, length: number): Float32Array {
  return new Float32Array(buffer, offset * 4, length);
}

function i8View(buffer: ArrayBufferLike, offset: number, length: number): Int8Array {
  return new Int8Array(buffer, offset, length);
}

/**
 * Builds a `LlamaWeightsQ8` from a manifest's weight entries and the three
 * backing buffers they index into (codes, scales, norms — one contiguous
 * `ArrayBuffer` each, matching `weights.codes.bin` / `weights.scales.bin` /
 * `weights.norms.bin` read whole into memory). Pure and buffer-agnostic —
 * neither this function nor its callers read a file; `fixture-q8.ts` and the
 * real-model loader each do their own `readFileSync` and pass the resulting
 * buffers here, which is what lets this be tested (`weights-q8-io.test.ts`)
 * without any fixture files on disk at all.
 */
export function buildLlamaWeightsQ8(
  entries: readonly WeightManifestEntry[],
  codesBuffer: ArrayBufferLike,
  scalesBuffer: ArrayBufferLike,
  normsBuffer: ArrayBufferLike,
  numLayers: number,
): LlamaWeightsQ8 {
  const quantByName = new Map<string, QuantizedLinear>();
  const normByName = new Map<string, Float32Array>();
  for (const entry of entries) {
    if (entry.kind === "quant") {
      const [n, k] = entry.shape;
      quantByName.set(entry.name, {
        codes: i8View(codesBuffer, entry.codesOffset, n * k),
        scale: f32View(scalesBuffer, entry.scaleOffset, n),
      });
    } else {
      normByName.set(entry.name, f32View(normsBuffer, entry.offset, elementsOf(entry.shape)));
    }
  }
  const getQuant = (name: string): QuantizedLinear => {
    const found = quantByName.get(name);
    if (!found) throw new Error(`buildLlamaWeightsQ8: no quantized weight named ${JSON.stringify(name)}`);
    return found;
  };
  const getNorm = (name: string): Float32Array => {
    const found = normByName.get(name);
    if (!found) throw new Error(`buildLlamaWeightsQ8: no norm weight named ${JSON.stringify(name)}`);
    return found;
  };

  const layers: LlamaLayerWeightsQ8[] = [];
  for (let i = 0; i < numLayers; i += 1) {
    layers.push({
      attnNorm: getNorm(`layers.${i}.attnNorm`),
      wq: getQuant(`layers.${i}.wq`),
      wk: getQuant(`layers.${i}.wk`),
      wv: getQuant(`layers.${i}.wv`),
      wo: getQuant(`layers.${i}.wo`),
      ffnNorm: getNorm(`layers.${i}.ffnNorm`),
      wGate: getQuant(`layers.${i}.wGate`),
      wUp: getQuant(`layers.${i}.wUp`),
      wDown: getQuant(`layers.${i}.wDown`),
    });
  }

  return {
    embedTokens: getQuant("embedTokens"),
    layers,
    finalNorm: getNorm("finalNorm"),
    lmHead: getQuant("lmHead"),
  };
}
