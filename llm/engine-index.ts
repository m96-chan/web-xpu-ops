/**
 * `web-xpu-ops/llm/engine` — the Llama engines and the browser weight cache.
 *
 * Issue #224. `llm/index.ts` is the Node-facing barrel (it reaches
 * `real-model-weights.ts`, which reads the converted checkpoint with
 * `node:fs`); this is the surface a page or a worker can import. Before an
 * engine dispatches anything, the host registers the WGSL it bundled:
 *
 *     registerKernelSources(table);   // { [op]: { [entry]: source } }
 *
 * `LLM_KERNEL_SOURCES` lists every `(op, entry)` the engines will ask for,
 * so a bundler can inline exactly those files from
 * `web-xpu-ops/ops/<op>/wgsl/<entry>.wgsl`.
 */
export { SARASHINA_2_2_1B_CONFIG, TINY_FIXTURE_CONFIG, type LlamaConfig } from "./config.js";
export { argmax, greedyGenerate, LlamaEngine } from "./engine.js";
export { LlamaEngineQ8 } from "./engine-q8.js";
export { createForwardProfile, LlamaEngineQ8Resident, type ForwardProfile } from "./engine-q8-resident.js";
export { LLM_KERNEL_SOURCES, llmKernels, type LlmKernelKey, type LlmKernels } from "./kernels.js";
export { KVCache } from "./kv-cache.js";
export type { LlamaLayerWeights, LlamaWeights } from "./weights.js";
export type { LlamaWeightsQ8 } from "./weights-q8.js";
export {
  buildLlamaWeightsQ8,
  llamaConfigFromManifest,
  type WeightManifestConfig,
  type WeightManifestEntry,
} from "./weights-q8-io.js";
export { loadWeightsQ8FromUrl, type WeightCacheOptions, type WeightFetchProgress } from "./browser-weights.js";
export type { LoadedRealModelQ8 } from "./weights-q8-io.js";
export { InMemoryChunkStore, type ChunkStore } from "./chunk-store.js";
export {
  createIndexedDbChunkStore,
  isIndexedDbSupported,
  DEFAULT_IDB_DATABASE_NAME,
  DEFAULT_IDB_STORE_NAME,
} from "./idb-chunk-store.js";
export { estimateStorageQuota, requestPersistentStorage, type QuotaEstimate } from "./storage-quota.js";
export {
  decideCacheStrategy,
  DEFAULT_CHUNK_SIZE_BYTES,
  type CacheStrategyDecision,
  type CachedFileInfo,
  type CurrentVersionRecord,
} from "./weight-cache.js";
export {
  registerKernelSources,
  type KernelResolver,
  type KernelSources,
  type ResidentDevice,
  type Runner,
} from "../harness/api.js";
