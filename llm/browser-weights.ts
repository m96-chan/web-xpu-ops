import type { ChunkStore } from "./chunk-store.js";
import type { LlamaConfig } from "./config.js";
import { createIndexedDbChunkStore, isIndexedDbSupported } from "./idb-chunk-store.js";
import type { LoadedRealModelQ8 } from "./real-model-weights.js";
import { estimateStorageQuota, requestPersistentStorage, type QuotaEstimate } from "./storage-quota.js";
import {
  decideCacheStrategy,
  DEFAULT_CHUNK_SIZE_BYTES,
  expectedFileByteSizes,
  readCachedFile,
  readCurrentVersion,
  sha256Hex,
  sweepOrphanedChunks,
  writeCachedFile,
  writeCurrentVersion,
  type CachedFileInfo,
  type CurrentVersionRecord,
} from "./weight-cache.js";
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
  const parsedLength = contentLength !== null && contentLength !== "" ? Number(contentLength) : NaN;
  // null, never NaN, for an unparsable header: the type promises
  // `number | null` and NaN is neither — it reads as falsy-but-non-null in
  // `p.totalBytes ? …` and poisons any fraction computed from it. (A
  // Content-Encoding mismatch — decoded bytes counted against an encoded
  // length — is the caller's clamp to make; this only guarantees the value
  // is a real number or absent.)
  const totalBytes = Number.isFinite(parsedLength) && parsedLength >= 0 ? parsedLength : null;

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
 * Options for the persistent IndexedDB weight cache (issue #121, parent #96
 * / alibi-ai's "re-fetches 1.4GB of weights every visit"). Every field is
 * optional and every field's default is the real-browser behavior — a
 * caller that passes nothing gets caching, on, backed by a real
 * `idb-chunk-store.ts#createIndexedDbChunkStore()`, with real
 * `navigator.storage` quota checks. `chunkStore`/`estimateQuota`/`requestPersist`
 * exist to be overridden so `browser-weights.cache.test.ts` can exercise
 * every branch of this file's cache logic in Node, with no IndexedDB and no
 * `navigator` — the same reasoning `weight-cache.ts`'s own injected
 * `ChunkStore` parameter follows.
 */
export interface WeightCacheOptions {
  /** `false` always fetches over HTTP and never touches IndexedDB — this
   * issue's own opt-out ("既定ON・オプトアウト可"). Defaults to `true`. */
  enabled?: boolean;
  /** Overrides the `ChunkStore` this loader caches into — mainly for tests
   * (`InMemoryChunkStore`). When omitted, a real `IndexedDbChunkStore` is
   * opened automatically, and if that is not possible (Node — no
   * `indexedDB` global at all — or a browser where `indexedDB.open()`
   * itself fails, e.g. Safari private mode) this loader transparently falls
   * back to a plain network load: no feature loss, only a persistence
   * difference, exactly this issue's own "フォールバック" requirement. */
  chunkStore?: ChunkStore;
  /** Bytes per chunk written to the store. Defaults to
   * `weight-cache.ts#DEFAULT_CHUNK_SIZE_BYTES`. */
  chunkSizeBytes?: number;
  /** Forwarded to `weight-cache.ts#decideCacheStrategy`. */
  headroomRatio?: number;
  /** Overrides the quota check (`storage-quota.ts#estimateStorageQuota` by
   * default). */
  estimateQuota?: () => Promise<QuotaEstimate | null>;
  /** Overrides the persistence request (`storage-quota.ts#requestPersistentStorage`
   * by default). */
  requestPersist?: () => Promise<boolean>;
}

/** `explicit` first; otherwise a real `IndexedDbChunkStore` if this
 * environment has IndexedDB and opening it actually succeeds, else
 * `undefined` — the single point every "should this load use the cache at
 * all" decision in `loadWeightsQ8FromUrl` reduces to. */
async function resolveChunkStore(explicit: ChunkStore | undefined): Promise<ChunkStore | undefined> {
  if (explicit) return explicit;
  if (!isIndexedDbSupported()) return undefined;
  try {
    return await createIndexedDbChunkStore();
  } catch {
    return undefined;
  }
}

/** `{file, loadedBytes: totalBytes, totalBytes}` — issue #121's own
 * "進捗コールバックは既存のfetch進捗と統一(キャッシュヒット時は即時100%)": a
 * cache hit reports one tick per file, already at 100%, through the exact
 * same `onProgress` shape a network fetch reports many ticks through. */
function reportCacheHit(onProgress: ((p: WeightFetchProgress) => void) | undefined, file: string, totalBytes: number): void {
  onProgress?.({ file, loadedBytes: totalBytes, totalBytes });
}

/**
 * Loads a `convert_weights.py`-format checkpoint served under `baseUrl`
 * (`${baseUrl}/manifest.json`, `${baseUrl}/weights.codes.bin`, etc.) over
 * `fetch`. `fetchImpl` defaults to the global `fetch` and is overridable for
 * testing without a network or a browser.
 *
 * `manifest.json` is always fetched over the network — computing its
 * content hash (issue #121's versioning key) is the only way to know
 * whether a previously-cached `weights.codes.bin`/`.scales.bin`/`.norms.bin`
 * (1.41 GiB combined, the overwhelming majority of this checkpoint) is still
 * current, and the manifest itself is tens of KB, not gigabytes — the "0
 * bytes on reload" this issue's PR proves on real hardware is about the
 * three big binaries, which a cache hit below never requests at all, not
 * about this one small, unavoidable check.
 */
export async function loadWeightsQ8FromUrl(
  baseUrl: string,
  maxSeqLen: number,
  onProgress?: (progress: WeightFetchProgress) => void,
  fetchImpl: typeof fetch = fetch,
  cacheOptions: WeightCacheOptions = {},
): Promise<LoadedRealModelQ8> {
  const base = baseUrl.endsWith("/") ? baseUrl.slice(0, -1) : baseUrl;

  const manifestBytes = await fetchWithProgress(`${base}/manifest.json`, "manifest", fetchImpl, onProgress);
  const manifest: ConvertedManifest = JSON.parse(new TextDecoder("utf-8").decode(manifestBytes));

  const fetchAllFromNetwork = async (): Promise<[ArrayBuffer, ArrayBuffer, ArrayBuffer]> => [
    await fetchWithProgress(`${base}/weights.codes.bin`, "codes", fetchImpl, onProgress),
    await fetchWithProgress(`${base}/weights.scales.bin`, "scales", fetchImpl, onProgress),
    await fetchWithProgress(`${base}/weights.norms.bin`, "norms", fetchImpl, onProgress),
  ];

  const enabled = cacheOptions.enabled ?? true;
  const store = enabled ? await resolveChunkStore(cacheOptions.chunkStore) : undefined;

  let codesBuf: ArrayBuffer;
  let scalesBuf: ArrayBuffer;
  let normsBuf: ArrayBuffer;

  if (!store) {
    [codesBuf, scalesBuf, normsBuf] = await fetchAllFromNetwork();
  } else {
    const manifestHash = await sha256Hex(manifestBytes);

    // Attempt a cache hit first — never gated by quota (review item #1: a
    // read consumes no storage, only a *write* can legitimately be refused
    // for quota reasons; gating the read too meant that once one successful
    // write made `navigator.storage.estimate()`'s own `usage` include this
    // cache's bytes, "free space looks insufficient" became permanent and a
    // valid cache was never read again — every subsequent visit re-downloaded
    // the full checkpoint forever) and never allowed to fail the load
    // (review item #2: a real IndexedDB `get()` can reject — storage
    // pressure closing the connection mid-read, "Clear site data", a stale
    // database from an older cache-store-name layout returning
    // `NotFoundError` — and without a `try`/`catch` here that rejection
    // propagated out of this function itself, turning a load that would
    // have succeeded over plain HTTP into an outright failure; every other
    // way this cache can be broken — a write failure, a corrupted chunk, a
    // hash mismatch — already falls back instead of throwing, and issue
    // #121's own completion condition ("フォールバック動作") promises the
    // same here).
    let hit: { codes: ArrayBuffer; scales: ArrayBuffer; norms: ArrayBuffer } | undefined;
    let current: CurrentVersionRecord | undefined;
    try {
      current = await readCurrentVersion(store, base);
      const cachedInfo = current?.manifestHash === manifestHash ? current.files : undefined;
      hit =
        cachedInfo?.codes && cachedInfo.scales && cachedInfo.norms
          ? await readAllCachedFiles(store, base, manifestHash, cachedInfo as Record<"codes" | "scales" | "norms", CachedFileInfo>)
          : undefined;
    } catch {
      hit = undefined;
    }

    if (hit) {
      ({ codes: codesBuf, scales: scalesBuf, norms: normsBuf } = hit);
      reportCacheHit(onProgress, "codes", codesBuf.byteLength);
      reportCacheHit(onProgress, "scales", scalesBuf.byteLength);
      reportCacheHit(onProgress, "norms", normsBuf.byteLength);

      // Best-effort orphan sweep on a hit too (review item #6): the write
      // path's own sweep below only ever ran at the tail of a *miss* that
      // fully succeeded, so chunks left behind by a write that wrote some
      // chunks (or even this very `current` record) but crashed before
      // reaching its own sweep would otherwise sit unreclaimed for as long
      // as every later visit keeps hitting — which is exactly the visit
      // pattern a working cache produces. `current` is already the record
      // this hit just read, so this is a cheap no-op scan whenever nothing
      // is actually orphaned.
      try {
        await sweepOrphanedChunks(store, base, current);
      } catch {
        // A failed sweep must not turn a successful hit into a failed load.
      }
    } else {
      [codesBuf, scalesBuf, normsBuf] = await fetchAllFromNetwork();

      // Best-effort only, deliberately outside any path that could fail
      // the load itself: the weights are already in memory from the fetch
      // above, so a write failure here (a `QuotaExceededError` mid-write —
      // `estimate()`'s number is a snapshot, not a reservation — or any
      // other IndexedDB error) must never surface as this function
      // throwing. The quota check itself is computed here, not before the
      // hit attempt above, precisely so it only ever gates this write.
      try {
        const sizes = expectedFileByteSizes(manifest.weights);
        const requiredBytes = sizes.codes + sizes.scales + sizes.norms;
        const estimateQuota = cacheOptions.estimateQuota ?? estimateStorageQuota;
        const quota = await estimateQuota();
        const decision = decideCacheStrategy({
          indexedDbSupported: true,
          quota,
          requiredBytes,
          headroomRatio: cacheOptions.headroomRatio,
        });

        if (decision.useCache) {
          const chunkSizeBytes = cacheOptions.chunkSizeBytes ?? DEFAULT_CHUNK_SIZE_BYTES;
          const requestPersist = cacheOptions.requestPersist ?? requestPersistentStorage;
          await requestPersist();
          const codesInfo = await writeCachedFile(store, base, manifestHash, "codes", codesBuf, chunkSizeBytes);
          const scalesInfo = await writeCachedFile(store, base, manifestHash, "scales", scalesBuf, chunkSizeBytes);
          const normsInfo = await writeCachedFile(store, base, manifestHash, "norms", normsBuf, chunkSizeBytes);
          const written: CurrentVersionRecord = {
            manifestHash,
            files: { codes: codesInfo, scales: scalesInfo, norms: normsInfo },
          };
          // Chunks written before the current-version record that points to
          // them, and the record written before the sweep that deletes
          // everything else: a crash between any two of these steps leaves
          // either the previous version still readable (crash before this
          // record lands) or this version readable with some harmless
          // orphaned chunks left over (crash after it lands but before the
          // sweep — reclaimed by review item #6's hit-path sweep above, the
          // next time this cache is read) — never a `current` record
          // pointing at chunks that were never fully written.
          //
          // Known limitation (review item #5, tracked as issue #124): this
          // write is not atomic across the three files. A re-publish that
          // keeps `manifest.json` byte-identical (so `manifestHash` is
          // unchanged) while replacing one or more of the three binaries —
          // not how this repository's own `convert_weights.py` works (it
          // always regenerates `manifest.json` too), but not ruled out for a
          // different future producer of this file layout — interrupted
          // mid-overwrite could leave a mix of old- and new-version chunks
          // under one hash that this cache's own per-file length check
          // cannot distinguish from a correctly written file, silently
          // serving a corrupted checkpoint. The real fix is per-file
          // versioning (folding something like an ETag/Last-Modified into
          // each file's own chunk-key hash, not just the manifest's) rather
          // than one hash shared by all three files; out of scope here.
          await writeCurrentVersion(store, base, written);
          await sweepOrphanedChunks(store, base, written);
        }
      } catch {
        // See the comment above the `try`.
      }
    }
  }

  const config: LlamaConfig = llamaConfigFromManifest(manifest.config, maxSeqLen);
  const weights: LlamaWeightsQ8 = buildLlamaWeightsQ8(manifest.weights, codesBuf, scalesBuf, normsBuf, config.numLayers);
  assertWeightShapesQ8(config, weights);
  return { config, weights };
}

/** Reads `codes`/`scales`/`norms` from `store` under one manifest hash,
 * returning `undefined` (never throwing) the moment any one of them is
 * missing or corrupt (`weight-cache.ts#readCachedFile`'s own contract) — a
 * partially-evicted cache entry is exactly as much a cache miss as no entry
 * at all, since `buildLlamaWeightsQ8` needs all three files to build
 * anything. */
async function readAllCachedFiles(
  store: ChunkStore,
  namespace: string,
  manifestHash: string,
  info: Record<"codes" | "scales" | "norms", CachedFileInfo>,
): Promise<{ codes: ArrayBuffer; scales: ArrayBuffer; norms: ArrayBuffer } | undefined> {
  const codes = await readCachedFile(store, namespace, manifestHash, "codes", info.codes);
  if (!codes) return undefined;
  const scales = await readCachedFile(store, namespace, manifestHash, "scales", info.scales);
  if (!scales) return undefined;
  const norms = await readCachedFile(store, namespace, manifestHash, "norms", info.norms);
  if (!norms) return undefined;
  return { codes, scales, norms };
}
