import type { LlamaConfig } from "./config.js";
import type { LoadedRealModelQ8 } from "./real-model-weights.js";
import type { LlamaWeightsQ8 } from "./weights-q8.js";
import { assertWeightShapesQ8 } from "./weights-q8.js";
import { buildLlamaWeightsQ8, llamaConfigFromManifest, type WeightManifestConfig, type WeightManifestEntry } from "./weights-q8-io.js";

/**
 * Browser counterpart to `real-model-weights.ts#loadConvertedWeightsQ8` —
 * same converted-checkpoint format (`manifest.json` + `weights.codes.bin` +
 * `weights.scales.bin` + `weights.norms.bin`, `llm/tools/convert_weights.py`'s
 * own output, documented in PR #108), loaded via `fetch` instead of
 * `readFileSync` because the checkpoint is served over HTTP by
 * `examples/llm-demo/server.mjs` rather than read off a local path — a
 * browser has no filesystem to read from. The manifest-parsing core
 * (`weights-q8-io.ts#buildLlamaWeightsQ8` / `llamaConfigFromManifest`) is the
 * same function `loadConvertedWeightsQ8` and `fixture-q8.ts` both call, per
 * PR #108's own note that a browser loader "should be a small adapter over
 * the same parsing logic, not a reimplementation".
 *
 * This module is deliberately part of `llm/` (not `examples/llm-demo/`) and
 * exported from `llm/index.ts`: `fetch`, `Response` and `ReadableStream` are
 * standard Web APIs available in every runtime this package already targets
 * (see `tsconfig.build.json`'s note on why `llm/tokenizer.ts` can use
 * `TextEncoder`/`TextDecoder` the same way), so unlike `real-model-weights.ts`
 * (which needs `node:fs`) there is nothing browser-specific enough here to
 * force it out of the package — and keeping it in `llm/` makes it testable
 * with a mocked `fetch`, with no browser and no GPU required.
 */

/** One progress tick for one of the four files this loader fetches. */
export interface WeightFetchProgress {
  /** `"manifest" | "codes" | "scales" | "norms"` — which file this tick is for. */
  file: string;
  loadedBytes: number;
  /** `null` when the response carried no (or an unparsable) `Content-Length`. */
  totalBytes: number | null;
}

interface ConvertedManifest {
  config: WeightManifestConfig;
  weights: WeightManifestEntry[];
}

/**
 * Fetches `url` and returns its bytes, reporting progress as chunks arrive.
 *
 * Reads via the response body's stream rather than `res.arrayBuffer()`
 * so a caller can show a progress bar for the multi-hundred-megabyte
 * `weights.codes.bin` — the whole reason this loader exists rather than one
 * `fetch(...).then(r => r.arrayBuffer())` per file. Falls back to
 * `arrayBuffer()` when a response has no readable body (some test doubles
 * and, per the Fetch spec, `HEAD`/`204` responses), reporting one tick at
 * the end instead of many.
 *
 * `totalBytes` comes only from the `Content-Length` header, never guessed —
 * `examples/llm-demo/server.mjs` is required to set it (issue #106's own
 * instruction: no `Range` support needed, `Content-Length` is what the
 * progress bar reads).
 */
async function fetchWithProgress(
  url: string,
  file: string,
  fetchImpl: typeof fetch,
  onProgress?: (progress: WeightFetchProgress) => void,
): Promise<ArrayBuffer> {
  const res = await fetchImpl(url);
  if (!res.ok) {
    throw new Error(`fetchWithProgress: ${url} responded HTTP ${res.status}`);
  }
  const contentLength = res.headers.get("content-length");
  const totalBytes = contentLength !== null && contentLength !== "" ? Number(contentLength) : null;

  if (!res.body) {
    const buffer = await res.arrayBuffer();
    onProgress?.({ file, loadedBytes: buffer.byteLength, totalBytes: buffer.byteLength });
    return buffer;
  }

  const reader = res.body.getReader();
  const chunks: Uint8Array[] = [];
  let loadedBytes = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loadedBytes += value.byteLength;
    onProgress?.({ file, loadedBytes, totalBytes });
  }

  const out = new Uint8Array(loadedBytes);
  let offset = 0;
  for (const chunk of chunks) {
    out.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return out.buffer;
}

/**
 * Loads a `convert_weights.py`-format checkpoint served under `baseUrl`
 * (`${baseUrl}/manifest.json`, `${baseUrl}/weights.codes.bin`, etc.) over
 * `fetch`. `fetchImpl` defaults to the global `fetch` and is overridable for
 * testing without a network or a browser.
 */
export async function loadWeightsQ8FromUrl(
  baseUrl: string,
  maxSeqLen: number,
  onProgress?: (progress: WeightFetchProgress) => void,
  fetchImpl: typeof fetch = fetch,
): Promise<LoadedRealModelQ8> {
  const base = baseUrl.endsWith("/") ? baseUrl.slice(0, -1) : baseUrl;

  const manifestBytes = await fetchWithProgress(`${base}/manifest.json`, "manifest", fetchImpl, onProgress);
  const manifest: ConvertedManifest = JSON.parse(new TextDecoder("utf-8").decode(manifestBytes));

  const codesBuf = await fetchWithProgress(`${base}/weights.codes.bin`, "codes", fetchImpl, onProgress);
  const scalesBuf = await fetchWithProgress(`${base}/weights.scales.bin`, "scales", fetchImpl, onProgress);
  const normsBuf = await fetchWithProgress(`${base}/weights.norms.bin`, "norms", fetchImpl, onProgress);

  const config: LlamaConfig = llamaConfigFromManifest(manifest.config, maxSeqLen);
  const weights: LlamaWeightsQ8 = buildLlamaWeightsQ8(manifest.weights, codesBuf, scalesBuf, normsBuf, config.numLayers);
  assertWeightShapesQ8(config, weights);
  return { config, weights };
}
