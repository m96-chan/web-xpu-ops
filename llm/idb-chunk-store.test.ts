import { afterEach, describe, expect, it } from "vitest";
import { createIndexedDbChunkStore, isIndexedDbSupported } from "./idb-chunk-store.js";

/**
 * `idb-chunk-store.ts` — the real `ChunkStore` (issue #121) backend. Node has
 * no `indexedDB` global at all, so what this file can prove without a
 * browser is narrower than `chunk-store.test.ts`: not "does IndexedDB work"
 * (real Chrome, the PR's own verification, proves that), but "does this
 * module correctly detect and report IndexedDB's *absence*, rather than
 * throwing something a caller has to guess the meaning of, or — worse —
 * silently returning a store that quietly loses every write."
 *
 * `browser-weights.ts` depends on exactly this behavior: it only calls
 * `createIndexedDbChunkStore()` after `isIndexedDbSupported()` says true, so
 * `isIndexedDbSupported() === false` under Node is what keeps every existing
 * `browser-weights.test.ts` test passing unmodified (this issue's own
 * regression guard, not a coincidence — see `browser-weights.cache.test.ts`).
 */

const originalIndexedDb = (globalThis as { indexedDB?: unknown }).indexedDB;

afterEach(() => {
  (globalThis as { indexedDB?: unknown }).indexedDB = originalIndexedDb;
});

describe("isIndexedDbSupported", () => {
  it("is false under Node (no indexedDB global) — the real, unmocked case", () => {
    expect(isIndexedDbSupported()).toBe(false);
  });

  it("is true once something is installed at globalThis.indexedDB", () => {
    (globalThis as { indexedDB?: unknown }).indexedDB = {};
    expect(isIndexedDbSupported()).toBe(true);
  });
});

describe("createIndexedDbChunkStore", () => {
  it("rejects with a clear error when indexedDB is unavailable, rather than hanging or throwing something unrelated", async () => {
    delete (globalThis as { indexedDB?: unknown }).indexedDB;
    await expect(createIndexedDbChunkStore()).rejects.toThrow(/indexedDB/i);
  });

  it("propagates a synchronous throw from indexedDB.open() as a rejected promise (Safari private-mode's own failure shape)", async () => {
    (globalThis as { indexedDB?: unknown }).indexedDB = {
      open() {
        throw new DOMException("private mode", "SecurityError");
      },
    };
    await expect(createIndexedDbChunkStore()).rejects.toThrow(/private mode/);
  });
});
