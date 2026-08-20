import { describe, expect, it } from "vitest";
import { InMemoryChunkStore, type ChunkStore } from "./chunk-store.js";

/**
 * `InMemoryChunkStore` — the test double for `ChunkStore` (issue #121), the
 * storage interface `llm/weight-cache.ts`'s cache logic is injected with so
 * it can be unit-tested in Node with no IndexedDB. This file checks the
 * double itself behaves like a real key-value store (round-trip, overwrite,
 * delete, list, missing-key), not the cache logic built on top of it —
 * `llm/weight-cache.test.ts` covers that, against this same double.
 */
describe("InMemoryChunkStore", () => {
  it("get() on a missing key returns undefined", async () => {
    const store: ChunkStore = new InMemoryChunkStore();
    await expect(store.get("nope")).resolves.toBeUndefined();
  });

  it("put() then get() round-trips the exact bytes", async () => {
    const store: ChunkStore = new InMemoryChunkStore();
    const bytes = Uint8Array.from([1, 2, 3, 4, 5]).buffer;
    await store.put("a", bytes);
    const got = await store.get("a");
    expect(got).toBeDefined();
    expect(Array.from(new Uint8Array(got!))).toEqual([1, 2, 3, 4, 5]);
  });

  it("put() again with the same key overwrites the previous value", async () => {
    const store: ChunkStore = new InMemoryChunkStore();
    await store.put("a", Uint8Array.from([1, 2, 3]).buffer);
    await store.put("a", Uint8Array.from([9, 9]).buffer);
    const got = await store.get("a");
    expect(Array.from(new Uint8Array(got!))).toEqual([9, 9]);
  });

  it("delete() removes the key; a second delete() is a no-op, not an error", async () => {
    const store: ChunkStore = new InMemoryChunkStore();
    await store.put("a", Uint8Array.from([1]).buffer);
    await store.delete("a");
    await expect(store.get("a")).resolves.toBeUndefined();
    await expect(store.delete("a")).resolves.toBeUndefined();
  });

  it("list() returns every stored key and nothing else, in no particular guaranteed order", async () => {
    const store: ChunkStore = new InMemoryChunkStore();
    await store.put("a", Uint8Array.from([1]).buffer);
    await store.put("b", Uint8Array.from([2]).buffer);
    await store.put("c", Uint8Array.from([3]).buffer);
    await store.delete("b");
    const keys = await store.list();
    expect(new Set(keys)).toEqual(new Set(["a", "c"]));
  });

  it("put() copies the input buffer: mutating the caller's buffer afterward does not change what get() returns", async () => {
    // A real IndexedDB `put()` structured-clones its value — the store never
    // aliases the caller's ArrayBuffer. A double that aliased instead would
    // pass every test above and still be a lie the moment a caller (like
    // `weight-cache.ts`, which slices one big fetched buffer into chunks
    // sharing no memory) reused a view into the same allocation.
    const store: ChunkStore = new InMemoryChunkStore();
    const source = Uint8Array.from([1, 2, 3]);
    await store.put("a", source.buffer);
    source[0] = 99;
    const got = new Uint8Array((await store.get("a"))!);
    expect(Array.from(got)).toEqual([1, 2, 3]);
  });

  it("get() returns a copy: mutating the returned buffer does not corrupt the store", async () => {
    const store: ChunkStore = new InMemoryChunkStore();
    await store.put("a", Uint8Array.from([1, 2, 3]).buffer);
    const first = new Uint8Array((await store.get("a"))!);
    first[0] = 99;
    const second = new Uint8Array((await store.get("a"))!);
    expect(Array.from(second)).toEqual([1, 2, 3]);
  });
});
