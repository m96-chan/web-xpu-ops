/**
 * Storage abstraction for the persistent weight cache (issue #121, parent
 * #96 / alibi-ai's "re-fetches 1.4GB of weights every visit"). `weight-cache.ts`
 * (chunk splitting, manifest-hash versioning, stale-version eviction) is
 * written entirely against this interface, never against `indexedDB`
 * directly — that is what makes it unit-testable in Node, which has no
 * IndexedDB, with `InMemoryChunkStore` below standing in for
 * `idb-chunk-store.ts#createIndexedDbChunkStore`'s real one.
 *
 * Four operations, no more: a flat key/value store of `ArrayBuffer`s. No
 * `list()`-with-prefix, no transactions, no cursors — everything this
 * cache needs beyond a plain key/value store (grouping chunks under one
 * file, one version, one namespace) is encoded into the *keys*
 * (`weight-cache.ts#chunkKey`/`currentVersionKey`), not pushed onto this
 * interface, so a real backend only has to implement four small methods to
 * be usable here.
 */
export interface ChunkStore {
  /** `undefined` for a key that was never `put()`, or was `delete()`d. */
  get(key: string): Promise<ArrayBuffer | undefined>;
  /** Overwrites any existing value at `key`. Implementations must not alias
   * `value` — the caller is free to reuse or mutate the buffer it passed in
   * once this resolves (real IndexedDB structured-clones on `put()`, so this
   * is the store behaving like the real one, not an extra guarantee bolted
   * onto the interface). */
  put(key: string, value: ArrayBuffer): Promise<void>;
  /** A second `delete()` of an already-missing key is a no-op, not an error —
   * callers (`weight-cache.ts#sweepOrphanedChunks`) delete opportunistically
   * without checking existence first. */
  delete(key: string): Promise<void>;
  /** Every key currently stored, in no particular order. */
  list(): Promise<string[]>;
}

/** A copy of `buffer`'s bytes as a freestanding `ArrayBuffer` — never the
 * same allocation, so storing it or handing it back can never alias what the
 * caller holds. */
function cloneArrayBuffer(buffer: ArrayBuffer): ArrayBuffer {
  return buffer.slice(0);
}

/**
 * The Node-testable stand-in for `idb-chunk-store.ts`'s real IndexedDB
 * backend — a `Map`, with the same copy-on-put/copy-on-get semantics a real
 * IndexedDB object store has (structured clone), so a test written against
 * this double cannot pass by accident on an aliasing bug the real backend
 * would never exhibit.
 */
export class InMemoryChunkStore implements ChunkStore {
  private readonly entries = new Map<string, ArrayBuffer>();

  async get(key: string): Promise<ArrayBuffer | undefined> {
    const found = this.entries.get(key);
    return found === undefined ? undefined : cloneArrayBuffer(found);
  }

  async put(key: string, value: ArrayBuffer): Promise<void> {
    this.entries.set(key, cloneArrayBuffer(value));
  }

  async delete(key: string): Promise<void> {
    this.entries.delete(key);
  }

  async list(): Promise<string[]> {
    return Array.from(this.entries.keys());
  }
}
