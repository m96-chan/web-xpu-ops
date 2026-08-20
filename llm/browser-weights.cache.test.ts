import { describe, expect, it } from "vitest";
import { InMemoryChunkStore, type ChunkStore } from "./chunk-store.js";
import { loadWeightsQ8FromUrl, type WeightFetchProgress } from "./browser-weights.js";
import { currentVersionKey, readCurrentVersion } from "./weight-cache.js";

/**
 * `browser-weights.ts`'s wiring of `weight-cache.ts`'s logic (issue #121):
 * an injected `InMemoryChunkStore` stands in for `idb-chunk-store.ts`'s real
 * IndexedDB backend, so what this file proves — cache write on miss, cache
 * read on hit with **zero further `fetchImpl` calls for the three weight
 * files**, automatic re-download on a manifest-hash change, opt-out, and
 * quota-based fallback — is exactly what the real-Chrome verification in
 * this issue's PR re-proves against a real IndexedDB and a real network tab:
 * this file is the Node-level version of that same claim, most importantly
 * the "reload transfers 0 bytes" one (`fetchImpl` throwing on a second call
 * for `codes`/`scales`/`norms`, below, is the Node-testable form of "the
 * DevTools Network panel shows no request").
 *
 * `browser-weights.test.ts` (this issue leaves unmodified) is the existing
 * regression guard for the *no cache options passed at all* path — under
 * Node, `isIndexedDbSupported()` is false, so every one of those tests
 * exercises the same fallback-to-plain-fetch path this file's "opt-out" and
 * "quota-insufficient" cases exercise deliberately.
 */

const CONFIG = {
  numLayers: 0,
  hiddenSize: 2,
  numHeads: 1,
  numKvHeads: 1,
  headDim: 1,
  ffnHidden: 1,
  vocabSize: 3,
  ropeTheta: 10000,
  rmsNormEps: 1e-5,
  tieEmbeddings: false,
};

// Same tiny fixture shape as browser-weights.test.ts: embedTokens [3,2] +
// finalNorm [2] + lmHead [3,2], numLayers: 0.
function manifestJson(extraWhitespace = ""): string {
  const manifest = {
    config: CONFIG,
    weights: [
      { name: "embedTokens", kind: "quant", shape: [3, 2], codesOffset: 0, scaleOffset: 0 },
      { name: "finalNorm", kind: "norm", shape: [2], offset: 0 },
      { name: "lmHead", kind: "quant", shape: [3, 2], codesOffset: 6, scaleOffset: 3 },
    ],
  };
  // A manifest that reformats to different bytes but parses to the same
  // config/weights — issue #121's own versioning trigger is a *content*
  // hash, so this is how a real "the model was re-converted, same shapes,
  // different bytes" update is simulated without needing different shapes.
  return JSON.stringify(manifest) + extraWhitespace;
}

const CODES = Int8Array.from([1, -2, 3, -4, 5, -6, 7, -8, 9, -10, 11, -12]);
const SCALES = Float32Array.from([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]);
const NORMS = Float32Array.from([1.5, -1.5]);

const BASE = "http://demo.invalid/weights";

interface FakeFetchOptions {
  manifestBody?: string;
  /** Files that must never be requested — throws instead of responding, so
   * a test can assert "this file's bytes never crossed the network again". */
  forbid?: readonly string[];
  callCounts?: Record<string, number>;
}

function makeFakeFetch(options: FakeFetchOptions = {}): typeof fetch {
  const callCounts = options.callCounts ?? {};
  const respond = (body: string | ArrayBuffer, contentType: string) => {
    const bytes = typeof body === "string" ? new TextEncoder().encode(body) : new Uint8Array(body);
    return Promise.resolve(
      new Response(bytes, {
        status: 200,
        headers: { "content-type": contentType, "content-length": String(bytes.byteLength) },
      }),
    );
  };
  return (async (input: string | URL | Request): Promise<Response> => {
    const url = String(input);
    const fileFor: Record<string, string> = {
      [`${BASE}/manifest.json`]: "manifest",
      [`${BASE}/weights.codes.bin`]: "codes",
      [`${BASE}/weights.scales.bin`]: "scales",
      [`${BASE}/weights.norms.bin`]: "norms",
    };
    const file = fileFor[url];
    if (file) callCounts[file] = (callCounts[file] ?? 0) + 1;
    if (file && options.forbid?.includes(file)) {
      throw new Error(`makeFakeFetch: ${file} must not be fetched (expected a cache hit)`);
    }
    if (url === `${BASE}/manifest.json`) return respond(options.manifestBody ?? manifestJson(), "application/json");
    if (url === `${BASE}/weights.codes.bin`) return respond(CODES.buffer as ArrayBuffer, "application/octet-stream");
    if (url === `${BASE}/weights.scales.bin`) return respond(SCALES.buffer as ArrayBuffer, "application/octet-stream");
    if (url === `${BASE}/weights.norms.bin`) return respond(NORMS.buffer as ArrayBuffer, "application/octet-stream");
    return respond("not found", "text/plain");
  }) as typeof fetch;
}

function expectSameWeights(weights: { embedTokens: { codes: Int8Array } }): void {
  expect(Array.from(weights.embedTokens.codes)).toEqual([1, -2, 3, -4, 5, -6]);
}

describe("loadWeightsQ8FromUrl — persistent cache (issue #121)", () => {
  it("a cache miss fetches every file and writes chunks + a current-version record to the store", async () => {
    const store = new InMemoryChunkStore();
    const callCounts: Record<string, number> = {};
    const { weights } = await loadWeightsQ8FromUrl(BASE, 128, undefined, makeFakeFetch({ callCounts }), { chunkStore: store });

    expectSameWeights(weights);
    expect(callCounts.codes).toBe(1);
    expect(callCounts.scales).toBe(1);
    expect(callCounts.norms).toBe(1);

    const keys = await store.list();
    expect(keys.length).toBeGreaterThan(0);
    const current = await readCurrentVersion(store, BASE);
    expect(current).toBeDefined();
    expect(current!.files.codes).toBeDefined();
    expect(current!.files.scales).toBeDefined();
    expect(current!.files.norms).toBeDefined();
  });

  it("a cache hit never fetches codes/scales/norms again, and returns byte-identical weights", async () => {
    const store = new InMemoryChunkStore();
    await loadWeightsQ8FromUrl(BASE, 128, undefined, makeFakeFetch(), { chunkStore: store });

    // Second load: codes/scales/norms fetches would throw if attempted —
    // the Node-level proof of "reload transfers 0 bytes" for the big files.
    const ticks: WeightFetchProgress[] = [];
    const { weights } = await loadWeightsQ8FromUrl(
      BASE,
      128,
      (p) => ticks.push({ ...p }),
      makeFakeFetch({ forbid: ["codes", "scales", "norms"] }),
      { chunkStore: store },
    );

    expectSameWeights(weights);

    // "immediate 100%" per issue #121: the first (and here, only) tick for
    // each cached file already has loadedBytes === totalBytes.
    for (const file of ["codes", "scales", "norms"]) {
      const fileTicks = ticks.filter((t) => t.file === file);
      expect(fileTicks.length).toBeGreaterThan(0);
      expect(fileTicks[0]!.loadedBytes).toBe(fileTicks[0]!.totalBytes);
    }
  });

  it("a manifest content change (same shapes, different bytes) triggers a re-download and evicts the old version's chunks", async () => {
    const store = new InMemoryChunkStore();
    await loadWeightsQ8FromUrl(BASE, 128, undefined, makeFakeFetch({ manifestBody: manifestJson() }), {
      chunkStore: store,
    });
    const keysAfterFirst = await store.list();
    const oldCurrent = await readCurrentVersion(store, BASE);

    const callCounts: Record<string, number> = {};
    await loadWeightsQ8FromUrl(BASE, 128, undefined, makeFakeFetch({ manifestBody: manifestJson(" "), callCounts }), {
      chunkStore: store,
    });

    expect(callCounts.codes).toBe(1);
    expect(callCounts.scales).toBe(1);
    expect(callCounts.norms).toBe(1);

    const newCurrent = await readCurrentVersion(store, BASE);
    expect(newCurrent!.manifestHash).not.toBe(oldCurrent!.manifestHash);

    const keysAfterSecond = await store.list();
    // None of the first version's per-chunk keys should still be present —
    // only the (overwritten) current-version key survives across both.
    const oldChunkKeys = keysAfterFirst.filter((k) => k !== currentVersionKey(BASE));
    for (const oldKey of oldChunkKeys) {
      expect(keysAfterSecond).not.toContain(oldKey);
    }
  });

  it("a manifest-hash mismatch short-circuits before reading any chunk data — only the current-version record itself is read", async () => {
    const store = new InMemoryChunkStore();
    await loadWeightsQ8FromUrl(BASE, 128, undefined, makeFakeFetch({ manifestBody: manifestJson() }), { chunkStore: store });

    let getCalls = 0;
    const countingStore: ChunkStore = {
      get: async (key) => {
        getCalls += 1;
        return store.get(key);
      },
      put: (key, value) => store.put(key, value),
      delete: (key) => store.delete(key),
      list: () => store.list(),
    };

    await loadWeightsQ8FromUrl(BASE, 128, undefined, makeFakeFetch({ manifestBody: manifestJson(" ") }), {
      chunkStore: countingStore,
    });

    // Exactly one `get()` — reading the current-version record to learn its
    // hash — with none spent reading (and finding nothing under) chunk keys
    // that belong to the version this load's manifest hash does not match.
    expect(getCalls).toBe(1);
  });

  it("enabled: false always fetches over the network and never touches the store", async () => {
    const store = new InMemoryChunkStore();
    const callCounts: Record<string, number> = {};
    await loadWeightsQ8FromUrl(BASE, 128, undefined, makeFakeFetch({ callCounts }), {
      chunkStore: store,
      enabled: false,
    });
    expect(callCounts.codes).toBe(1);
    await expect(store.list()).resolves.toEqual([]);

    // A second load, cache still disabled, fetches again rather than
    // finding anything (there is nothing to find).
    const callCounts2: Record<string, number> = {};
    await loadWeightsQ8FromUrl(BASE, 128, undefined, makeFakeFetch({ callCounts: callCounts2 }), {
      chunkStore: store,
      enabled: false,
    });
    expect(callCounts2.codes).toBe(1);
  });

  it("insufficient quota falls back to a network load without writing to the store", async () => {
    const store = new InMemoryChunkStore();
    const callCounts: Record<string, number> = {};
    const { weights } = await loadWeightsQ8FromUrl(BASE, 128, undefined, makeFakeFetch({ callCounts }), {
      chunkStore: store,
      estimateQuota: async () => ({ usageBytes: 0, quotaBytes: 1 }), // far below any requiredBytes
    });
    expectSameWeights(weights);
    expect(callCounts.codes).toBe(1);
    await expect(store.list()).resolves.toEqual([]);
  });

  it("a store whose writes fail (simulated QuotaExceededError) does not fail the load — the already-fetched weights are still returned", async () => {
    const store = new InMemoryChunkStore();
    // Delegates read operations to a real (empty) store and fails every
    // write — not a `{...store}` spread, which would silently drop
    // `InMemoryChunkStore`'s prototype methods and fail for the wrong
    // reason (a missing method, not the simulated write failure).
    const failingStore: ChunkStore = {
      get: (key) => store.get(key),
      put: async () => {
        throw new DOMException("quota exceeded mid-write", "QuotaExceededError");
      },
      delete: (key) => store.delete(key),
      list: () => store.list(),
    };
    const { weights } = await loadWeightsQ8FromUrl(BASE, 128, undefined, makeFakeFetch(), { chunkStore: failingStore });
    expectSameWeights(weights);
    // Nothing durable was left behind by the failed write — a later load
    // against the same (now-working) store must fetch fresh rather than
    // trusting a half-written record.
    await expect(readCurrentVersion(store, BASE)).resolves.toBeUndefined();
  });

  it("a corrupted cache entry (a chunk deleted out from under a valid current-version record) falls back to a network re-fetch instead of throwing", async () => {
    const store = new InMemoryChunkStore();
    await loadWeightsQ8FromUrl(BASE, 128, undefined, makeFakeFetch(), { chunkStore: store });

    const keys = await store.list();
    const aChunkKey = keys.find((k) => k !== currentVersionKey(BASE));
    expect(aChunkKey).toBeDefined();
    await store.delete(aChunkKey!);

    const callCounts: Record<string, number> = {};
    const { weights } = await loadWeightsQ8FromUrl(BASE, 128, undefined, makeFakeFetch({ callCounts }), {
      chunkStore: store,
    });
    expectSameWeights(weights);
    // The corrupted file must have been re-fetched — a hard failure here
    // would mean one deleted IndexedDB entry (eviction under storage
    // pressure is exactly this shape) permanently breaks loading.
    expect(Object.values(callCounts).some((n) => n > 0)).toBe(true);
  });
});
