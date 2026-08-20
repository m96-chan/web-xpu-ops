import { describe, expect, it } from "vitest";
import { InMemoryChunkStore } from "./chunk-store.js";
import type { WeightManifestEntry } from "./weights-q8-io.js";
import {
  chunkKey,
  currentVersionKey,
  decideCacheStrategy,
  DEFAULT_CHUNK_SIZE_BYTES,
  expectedFileByteSizes,
  joinChunks,
  readCachedFile,
  readCurrentVersion,
  sha256Hex,
  splitIntoChunks,
  sweepOrphanedChunks,
  writeCachedFile,
  writeCurrentVersion,
  type CurrentVersionRecord,
} from "./weight-cache.js";

/**
 * `weight-cache.ts` — the cache logic issue #121 asks to be Node-testable
 * "storage injected": every function below is either pure (hashing,
 * chunking, key building, the quota decision) or takes a `ChunkStore` as a
 * parameter (`InMemoryChunkStore` here — `chunk-store.test.ts` already
 * covers the double's own fidelity). None of this file touches
 * `indexedDB`/`navigator` — `idb-chunk-store.ts`/`storage-quota.ts` are the
 * only two modules that do, and `browser-weights.cache.test.ts` is what
 * proves this logic and `browser-weights.ts`'s wiring of it fit together.
 */

describe("sha256Hex", () => {
  it("matches a known SHA-256 test vector ('abc')", async () => {
    const bytes = new TextEncoder().encode("abc").buffer as ArrayBuffer;
    await expect(sha256Hex(bytes)).resolves.toBe(
      "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
    );
  });

  it("matches the known SHA-256 of the empty string", async () => {
    await expect(sha256Hex(new ArrayBuffer(0))).resolves.toBe(
      "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    );
  });

  it("is sensitive to every byte: a one-byte difference changes the hash", async () => {
    const a = Uint8Array.from([1, 2, 3]).buffer;
    const b = Uint8Array.from([1, 2, 4]).buffer;
    const [hashA, hashB] = await Promise.all([sha256Hex(a), sha256Hex(b)]);
    expect(hashA).not.toBe(hashB);
  });
});

describe("splitIntoChunks / joinChunks", () => {
  function bufferOf(n: number): ArrayBuffer {
    const arr = new Uint8Array(n);
    for (let i = 0; i < n; i += 1) arr[i] = i % 256;
    return arr.buffer;
  }

  it("splits an exact multiple of chunkSizeBytes into equal chunks", () => {
    const buf = bufferOf(30);
    const chunks = splitIntoChunks(buf, 10);
    expect(chunks.map((c) => c.byteLength)).toEqual([10, 10, 10]);
    expect(joinChunks(chunks).byteLength).toBe(30);
    expect(Array.from(new Uint8Array(joinChunks(chunks)))).toEqual(Array.from(new Uint8Array(buf)));
  });

  it("puts the remainder in a final, smaller chunk", () => {
    const buf = bufferOf(25);
    const chunks = splitIntoChunks(buf, 10);
    expect(chunks.map((c) => c.byteLength)).toEqual([10, 10, 5]);
  });

  it("a buffer smaller than chunkSizeBytes becomes exactly one chunk", () => {
    const buf = bufferOf(3);
    const chunks = splitIntoChunks(buf, 10);
    expect(chunks).toHaveLength(1);
    expect(chunks[0]!.byteLength).toBe(3);
  });

  it("a zero-length buffer becomes exactly one zero-length chunk, not zero chunks", () => {
    // So `writeCachedFile` always has at least one chunk key to write and
    // `readCachedFile` (which reads `chunkCount` chunks) can reconstruct a
    // genuinely empty file rather than treating "0 chunks" as "missing".
    const chunks = splitIntoChunks(new ArrayBuffer(0), 10);
    expect(chunks).toHaveLength(1);
    expect(chunks[0]!.byteLength).toBe(0);
  });

  it("splitIntoChunks throws for a non-positive chunkSizeBytes", () => {
    expect(() => splitIntoChunks(bufferOf(10), 0)).toThrow();
    expect(() => splitIntoChunks(bufferOf(10), -1)).toThrow();
  });

  it("joinChunks of zero chunks is a zero-length buffer", () => {
    expect(joinChunks([]).byteLength).toBe(0);
  });

  it("round-trips DEFAULT_CHUNK_SIZE_BYTES-scale data without corruption (small stand-in, same code path)", () => {
    const buf = bufferOf(1000);
    const chunks = splitIntoChunks(buf, 64);
    const rejoined = joinChunks(chunks);
    expect(Array.from(new Uint8Array(rejoined))).toEqual(Array.from(new Uint8Array(buf)));
  });
});

describe("expectedFileByteSizes", () => {
  it("computes each file's byte size as the max (offset + element count) across its entries, in bytes", () => {
    // Mirrors browser-weights.test.ts's own tiny manifest shape.
    const entries: WeightManifestEntry[] = [
      { name: "embedTokens", kind: "quant", shape: [3, 2], codesOffset: 0, scaleOffset: 0 },
      { name: "finalNorm", kind: "norm", shape: [2], offset: 0 },
      { name: "lmHead", kind: "quant", shape: [3, 2], codesOffset: 6, scaleOffset: 3 },
    ];
    const sizes = expectedFileByteSizes(entries);
    // codes: int8, 1 byte/element — lmHead ends at 6 + 3*2 = 12.
    expect(sizes.codes).toBe(12);
    // scales: f32, 4 bytes/element — lmHead ends at (3 + 3) * 4 = 24.
    expect(sizes.scales).toBe(24);
    // norms: f32, 4 bytes/element — finalNorm ends at (0 + 2) * 4 = 8.
    expect(sizes.norms).toBe(8);
  });

  it("is 0 for a file with no entries of that kind (e.g. a norms-only or quant-only manifest)", () => {
    const entries: WeightManifestEntry[] = [{ name: "finalNorm", kind: "norm", shape: [4], offset: 0 }];
    const sizes = expectedFileByteSizes(entries);
    expect(sizes.codes).toBe(0);
    expect(sizes.scales).toBe(0);
    expect(sizes.norms).toBe(16);
  });
});

describe("chunkKey / currentVersionKey", () => {
  it("never collide across different namespace, manifestHash, file, or index", () => {
    const keys = new Set<string>();
    const variants = [
      chunkKey("ns1", "hashA", "codes", 0),
      chunkKey("ns2", "hashA", "codes", 0),
      chunkKey("ns1", "hashB", "codes", 0),
      chunkKey("ns1", "hashA", "scales", 0),
      chunkKey("ns1", "hashA", "codes", 1),
      currentVersionKey("ns1"),
      currentVersionKey("ns2"),
    ];
    for (const k of variants) keys.add(k);
    expect(keys.size).toBe(variants.length);
  });

  it("currentVersionKey never equals a chunkKey for the same namespace, at any index", () => {
    const cv = currentVersionKey("ns1");
    for (let i = 0; i < 5; i += 1) {
      expect(chunkKey("ns1", "anyhash", "codes", i)).not.toBe(cv);
    }
  });
});

describe("readCurrentVersion / writeCurrentVersion", () => {
  it("returns undefined when nothing has been written yet", async () => {
    const store = new InMemoryChunkStore();
    await expect(readCurrentVersion(store, "ns1")).resolves.toBeUndefined();
  });

  it("round-trips a record written for one namespace, and does not appear under another", async () => {
    const store = new InMemoryChunkStore();
    const record: CurrentVersionRecord = {
      manifestHash: "abc123",
      files: { codes: { totalBytes: 10, chunkSizeBytes: 5, chunkCount: 2 } },
    };
    await writeCurrentVersion(store, "ns1", record);
    await expect(readCurrentVersion(store, "ns1")).resolves.toEqual(record);
    await expect(readCurrentVersion(store, "ns2")).resolves.toBeUndefined();
  });

  it("returns undefined (not a throw) for a corrupt/non-JSON record", async () => {
    const store = new InMemoryChunkStore();
    await store.put(currentVersionKey("ns1"), new TextEncoder().encode("not json").buffer as ArrayBuffer);
    await expect(readCurrentVersion(store, "ns1")).resolves.toBeUndefined();
  });
});

describe("writeCachedFile / readCachedFile", () => {
  it("round-trips a multi-chunk file exactly", async () => {
    const store = new InMemoryChunkStore();
    const original = new Uint8Array(250);
    for (let i = 0; i < original.length; i += 1) original[i] = (i * 7) % 256;
    const info = await writeCachedFile(store, "ns1", "hashA", "codes", original.buffer, 100);
    expect(info.chunkCount).toBe(3);
    expect(info.totalBytes).toBe(250);

    const read = await readCachedFile(store, "ns1", "hashA", "codes", info);
    expect(read).toBeDefined();
    expect(Array.from(new Uint8Array(read!))).toEqual(Array.from(original));
  });

  it("returns undefined when a chunk the info record expects is missing (partial/corrupt cache)", async () => {
    const store = new InMemoryChunkStore();
    const info = await writeCachedFile(store, "ns1", "hashA", "codes", new Uint8Array(30).buffer, 10);
    await store.delete(chunkKey("ns1", "hashA", "codes", 1));
    await expect(readCachedFile(store, "ns1", "hashA", "codes", info)).resolves.toBeUndefined();
  });

  it("returns undefined when the reconstructed length disagrees with info.totalBytes (corrupt bookkeeping)", async () => {
    const store = new InMemoryChunkStore();
    const info = await writeCachedFile(store, "ns1", "hashA", "codes", new Uint8Array(30).buffer, 10);
    const tampered = { ...info, totalBytes: 999 };
    await expect(readCachedFile(store, "ns1", "hashA", "codes", tampered)).resolves.toBeUndefined();
  });

  it("different files under the same manifestHash do not collide", async () => {
    const store = new InMemoryChunkStore();
    const codesInfo = await writeCachedFile(store, "ns1", "hashA", "codes", Uint8Array.from([1, 2, 3]).buffer, 10);
    const scalesInfo = await writeCachedFile(store, "ns1", "hashA", "scales", Uint8Array.from([9, 9, 9]).buffer, 10);
    const codes = await readCachedFile(store, "ns1", "hashA", "codes", codesInfo);
    const scales = await readCachedFile(store, "ns1", "hashA", "scales", scalesInfo);
    expect(Array.from(new Uint8Array(codes!))).toEqual([1, 2, 3]);
    expect(Array.from(new Uint8Array(scales!))).toEqual([9, 9, 9]);
  });
});

describe("sweepOrphanedChunks", () => {
  it("deletes every chunk under the namespace whose manifestHash is not the one to keep, and the current-version key is unaffected by the sweep itself", async () => {
    const store = new InMemoryChunkStore();
    await writeCachedFile(store, "ns1", "oldHash", "codes", new Uint8Array(20).buffer, 10);
    await writeCachedFile(store, "ns1", "newHash", "codes", new Uint8Array(20).buffer, 10);
    const keep: CurrentVersionRecord = {
      manifestHash: "newHash",
      files: { codes: { totalBytes: 20, chunkSizeBytes: 10, chunkCount: 2 } },
    };
    await writeCurrentVersion(store, "ns1", keep);

    await sweepOrphanedChunks(store, "ns1", keep);

    const remaining = await store.list();
    expect(remaining.some((k) => k.includes("oldHash"))).toBe(false);
    expect(remaining.some((k) => k.includes("newHash"))).toBe(true);
    // The current-version record itself, whose key carries no manifestHash,
    // must survive the sweep — it is what the *next* load reads first.
    await expect(readCurrentVersion(store, "ns1")).resolves.toEqual(keep);
  });

  it("never touches another namespace's chunks", async () => {
    const store = new InMemoryChunkStore();
    await writeCachedFile(store, "ns1", "hashA", "codes", new Uint8Array(10).buffer, 10);
    await writeCachedFile(store, "ns2", "hashA", "codes", new Uint8Array(10).buffer, 10);

    await sweepOrphanedChunks(store, "ns1", {
      manifestHash: "some-other-hash",
      files: { codes: { totalBytes: 0, chunkSizeBytes: 10, chunkCount: 0 } },
    });

    const remaining = await store.list();
    expect(remaining.some((k) => k.includes("ns2"))).toBe(true);
  });

  it("with keep undefined, removes every chunk (and the current-version key) under the namespace", async () => {
    const store = new InMemoryChunkStore();
    await writeCachedFile(store, "ns1", "hashA", "codes", new Uint8Array(10).buffer, 10);
    await writeCurrentVersion(store, "ns1", {
      manifestHash: "hashA",
      files: { codes: { totalBytes: 10, chunkSizeBytes: 10, chunkCount: 1 } },
    });

    await sweepOrphanedChunks(store, "ns1", undefined);

    const remaining = await store.list();
    expect(remaining.filter((k) => k.includes("ns1"))).toHaveLength(0);
  });

  it("review item #7: a chunk-size change under the SAME manifestHash leaves no stale higher-index chunks behind", async () => {
    // Repro: first write under hashA with a small chunk size (8 chunks for
    // an 80-byte file), then re-write the *same* file under the *same*
    // hashA with a larger chunk size (1 chunk). A prefix-based sweep
    // (`${namespace} ${hashA} codes `) keeps every one of the 8 old chunk
    // keys too, since they share that prefix with the 1 new one — only an
    // exact per-index check catches that chunks 1..7 are now orphans.
    const store = new InMemoryChunkStore();
    await writeCachedFile(store, "ns1", "hashA", "codes", new Uint8Array(80).buffer, 10);
    for (let i = 0; i < 8; i += 1) {
      await expect(store.get(chunkKey("ns1", "hashA", "codes", i))).resolves.toBeDefined();
    }

    const newInfo = await writeCachedFile(store, "ns1", "hashA", "codes", new Uint8Array(80).buffer, 96);
    expect(newInfo.chunkCount).toBe(1);
    const keep: CurrentVersionRecord = { manifestHash: "hashA", files: { codes: newInfo } };
    await writeCurrentVersion(store, "ns1", keep);

    await sweepOrphanedChunks(store, "ns1", keep);

    await expect(store.get(chunkKey("ns1", "hashA", "codes", 0))).resolves.toBeDefined();
    for (let i = 1; i < 8; i += 1) {
      await expect(store.get(chunkKey("ns1", "hashA", "codes", i))).resolves.toBeUndefined();
    }
  });

  it("keeps only the exact chunk indices info.chunkCount implies, per file, even across multiple files under one hash", async () => {
    const store = new InMemoryChunkStore();
    const codesInfo = await writeCachedFile(store, "ns1", "hashA", "codes", new Uint8Array(25).buffer, 10);
    const scalesInfo = await writeCachedFile(store, "ns1", "hashA", "scales", new Uint8Array(5).buffer, 10);
    const keep: CurrentVersionRecord = { manifestHash: "hashA", files: { codes: codesInfo, scales: scalesInfo } };

    await sweepOrphanedChunks(store, "ns1", keep);

    const remaining = new Set(await store.list());
    expect(remaining.has(chunkKey("ns1", "hashA", "codes", 0))).toBe(true);
    expect(remaining.has(chunkKey("ns1", "hashA", "codes", 1))).toBe(true);
    expect(remaining.has(chunkKey("ns1", "hashA", "codes", 2))).toBe(true);
    expect(remaining.has(chunkKey("ns1", "hashA", "scales", 0))).toBe(true);
  });
});

describe("decideCacheStrategy", () => {
  it("refuses when IndexedDB itself is unsupported, regardless of quota", () => {
    const decision = decideCacheStrategy({
      indexedDbSupported: false,
      quota: { usageBytes: 0, quotaBytes: 1_000_000_000_000 },
      requiredBytes: 100,
    });
    expect(decision.useCache).toBe(false);
    expect(decision.reason).toBe("indexeddb-unavailable");
  });

  it("proceeds optimistically when quota cannot be determined", () => {
    const decision = decideCacheStrategy({ indexedDbSupported: true, quota: null, requiredBytes: 100 });
    expect(decision.useCache).toBe(true);
    expect(decision.reason).toBe("quota-unknown-optimistic");
  });

  it("proceeds when free space comfortably exceeds requiredBytes plus headroom", () => {
    const decision = decideCacheStrategy({
      indexedDbSupported: true,
      quota: { usageBytes: 0, quotaBytes: 1_000_000_000 },
      requiredBytes: 500_000_000,
    });
    expect(decision.useCache).toBe(true);
    expect(decision.reason).toBe("ok");
  });

  it("refuses when free space is short of requiredBytes plus the default headroom", () => {
    const decision = decideCacheStrategy({
      indexedDbSupported: true,
      quota: { usageBytes: 900_000_000, quotaBytes: 1_000_000_000 }, // 100M free
      requiredBytes: 95_000_000, // *1.1 headroom = 104.5M > 100M free
    });
    expect(decision.useCache).toBe(false);
    expect(decision.reason).toBe("quota-insufficient");
  });

  it("honors a caller-supplied headroomRatio", () => {
    const params = {
      indexedDbSupported: true,
      quota: { usageBytes: 0, quotaBytes: 100 },
      requiredBytes: 60,
    };
    // Default headroom (1.1x -> needs 66) fits in 100 free either way; use a
    // large explicit headroom to force the insufficient branch instead.
    expect(decideCacheStrategy({ ...params, headroomRatio: 2 }).useCache).toBe(false);
    expect(decideCacheStrategy({ ...params, headroomRatio: 1 }).useCache).toBe(true);
  });
});

describe("DEFAULT_CHUNK_SIZE_BYTES", () => {
  it("is within the 64-128 MiB range this issue's own spec calls for", () => {
    expect(DEFAULT_CHUNK_SIZE_BYTES).toBeGreaterThanOrEqual(64 * 1024 * 1024);
    expect(DEFAULT_CHUNK_SIZE_BYTES).toBeLessThanOrEqual(128 * 1024 * 1024);
  });
});
