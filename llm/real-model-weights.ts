import { readFileSync } from "node:fs";
import { join } from "node:path";
import type { LlamaConfig } from "./config.js";
import type { LlamaWeightsQ8 } from "./weights-q8.js";
import { assertWeightShapesQ8 } from "./weights-q8.js";
import { buildLlamaWeightsQ8, llamaConfigFromManifest, type WeightManifestConfig, type WeightManifestEntry } from "./weights-q8-io.js";

/**
 * Loads a checkpoint converted by `llm/tools/convert_weights.py` — issue
 * #105's real-model counterpart to `fixture-q8.ts#loadTinyFixtureQ8`, reading
 * the converter's own filenames (`manifest.json`, `weights.codes.bin`,
 * `weights.scales.bin`, `weights.norms.bin`) rather than the tiny fixture's
 * `tiny_q8.*` names. Shares the same manifest-parsing core
 * (`weights-q8-io.ts#buildLlamaWeightsQ8`) as the fixture loader, since
 * `convert_weights.py` and `gen_fixture_q8.py` write the identical
 * `{name, kind, shape, offsets}` shape (see `convert_weights.py`'s module
 * doc).
 *
 * `dirPath` is a plain filesystem path, not a `file://` URL — unlike
 * `loadTinyFixtureQ8`'s `fixtures/` (checked into this repository, resolved
 * relative to this module), a converted checkpoint lives outside the
 * repository entirely (issue #105's own instructions: the converted weights'
 * output directory is `third_party/webgpu-weights/` in a *different* repo,
 * gitignored there), so there is no `import.meta.url`-relative default worth
 * having — a caller always names the directory explicitly.
 */

interface ConvertedManifest {
  config: WeightManifestConfig;
  weights: WeightManifestEntry[];
}

export interface LoadedRealModelQ8 {
  config: LlamaConfig;
  weights: LlamaWeightsQ8;
}

/** A copy of `buffer`'s bytes as a plain `ArrayBuffer`, since `buildLlamaWeightsQ8` indexes from byte 0 of what it is given but a `Buffer`'s `.buffer` is the whole underlying allocation. */
function toArrayBuffer(buffer: Buffer): ArrayBuffer {
  return buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength) as ArrayBuffer;
}

export function loadConvertedWeightsQ8(dirPath: string, maxSeqLen: number): LoadedRealModelQ8 {
  const manifest: ConvertedManifest = JSON.parse(readFileSync(join(dirPath, "manifest.json"), "utf8"));
  const codesBuf = readFileSync(join(dirPath, "weights.codes.bin"));
  const scalesBuf = readFileSync(join(dirPath, "weights.scales.bin"));
  const normsBuf = readFileSync(join(dirPath, "weights.norms.bin"));

  const config = llamaConfigFromManifest(manifest.config, maxSeqLen);
  const weights = buildLlamaWeightsQ8(
    manifest.weights,
    toArrayBuffer(codesBuf),
    toArrayBuffer(scalesBuf),
    toArrayBuffer(normsBuf),
    config.numLayers,
  );
  assertWeightShapesQ8(config, weights);
  return { config, weights };
}
