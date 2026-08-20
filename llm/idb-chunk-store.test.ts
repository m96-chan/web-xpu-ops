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

/**
 * A hand-rolled fake `indexedDB` that reproduces exactly one thing a real
 * one can do that `InMemoryChunkStore` (and any fake built only from
 * request-level `onsuccess`/`onerror`) cannot: a `put()` *request* reporting
 * `onsuccess` — meaning IndexedDB accepted and queued the write — followed
 * later by its own *transaction* aborting during commit, the textbook shape
 * of a `QuotaExceededError` on a large value (spec: commit-time quota
 * enforcement, not per-request). `createIndexedDbChunkStore`'s `put()` must
 * track the transaction's outcome, not the request's, to ever observe this.
 *
 * Every request/transaction callback fires on a fresh microtask (`queueMicrotask`,
 * never synchronously), matching real IndexedDB's own async event dispatch —
 * a fake that fired handlers synchronously could accidentally paper over a
 * bug in code that assumes (correctly, for the real API) that these events
 * never arrive before the call that requested them returns.
 */
function makeFakeIndexedDb(options: { abortWriteTransactions?: boolean } = {}) {
  const objectStoreNames = new Set<string>();
  const data = new Map<string, ArrayBuffer>();

  function makeRequest<T>(compute: () => T): { result: T | undefined; error: unknown; onsuccess: (() => void) | null; onerror: (() => void) | null } {
    const request = { result: undefined as T | undefined, error: null as unknown, onsuccess: null as (() => void) | null, onerror: null as (() => void) | null };
    queueMicrotask(() => {
      request.result = compute();
      request.onsuccess?.();
    });
    return request;
  }

  function makeObjectStore(onWriteRequested: () => void) {
    return {
      put(value: ArrayBuffer, key: string) {
        return makeRequest(() => {
          onWriteRequested();
          data.set(key, value);
          return key;
        });
      },
      get(key: string) {
        return makeRequest(() => data.get(key));
      },
      delete(key: string) {
        return makeRequest(() => {
          onWriteRequested();
          data.delete(key);
        });
      },
      getAllKeys() {
        return makeRequest(() => Array.from(data.keys()));
      },
    };
  }

  function makeTransaction(mode: "readonly" | "readwrite") {
    const transaction = {
      oncomplete: null as (() => void) | null,
      onabort: null as (() => void) | null,
      onerror: null as (() => void) | null,
      error: null as unknown,
      objectStore: (_name: string) => store,
    };
    let requested = false;
    const store = makeObjectStore(() => {
      requested = true;
    });
    // Settles once, after every synchronously-issued request this
    // transaction's caller made has had its own microtask fire — mirrors a
    // real transaction committing (or aborting) only after its whole task
    // queue drains.
    queueMicrotask(() => {
      queueMicrotask(() => {
        if (mode === "readwrite" && requested && options.abortWriteTransactions) {
          transaction.error = new DOMException("quota exceeded on commit", "QuotaExceededError");
          transaction.onabort?.();
        } else {
          transaction.oncomplete?.();
        }
      });
    });
    return transaction;
  }

  return {
    open(_name: string, _version?: number) {
      const request = {
        result: undefined as unknown,
        error: null as unknown,
        onsuccess: null as (() => void) | null,
        onerror: null as (() => void) | null,
        onupgradeneeded: null as (() => void) | null,
        onblocked: null as (() => void) | null,
      };
      queueMicrotask(() => {
        const db = {
          objectStoreNames: { contains: (n: string) => objectStoreNames.has(n) },
          createObjectStore(n: string) {
            objectStoreNames.add(n);
            return {};
          },
          transaction: (_storeNames: string, mode: "readonly" | "readwrite") => makeTransaction(mode),
        };
        request.result = db;
        request.onupgradeneeded?.();
        request.onsuccess?.();
      });
      return request;
    },
  };
}

describe("createIndexedDbChunkStore — transaction-outcome-based resolution (review item #3)", () => {
  it("put() still resolves on the ordinary path (request succeeds, transaction completes)", async () => {
    (globalThis as { indexedDB?: unknown }).indexedDB = makeFakeIndexedDb();
    const store = await createIndexedDbChunkStore("db1", "chunks");
    await expect(store.put("a", Uint8Array.from([1, 2, 3]).buffer)).resolves.toBeUndefined();
    await expect(store.get("a")).resolves.toBeDefined();
  });

  it("put() rejects when its transaction aborts on commit, even though the individual request already reported onsuccess", async () => {
    (globalThis as { indexedDB?: unknown }).indexedDB = makeFakeIndexedDb({ abortWriteTransactions: true });
    const store = await createIndexedDbChunkStore("db2", "chunks");
    await expect(store.put("a", Uint8Array.from([1, 2, 3]).buffer)).rejects.toThrow(/quota exceeded/i);
  });

  it("delete() rejects when its transaction aborts on commit, the same as put()", async () => {
    const idb = makeFakeIndexedDb();
    (globalThis as { indexedDB?: unknown }).indexedDB = idb;
    const store = await createIndexedDbChunkStore("db3", "chunks");
    await store.put("a", Uint8Array.from([1]).buffer);

    (globalThis as { indexedDB?: unknown }).indexedDB = makeFakeIndexedDb({ abortWriteTransactions: true });
    const abortingStore = await createIndexedDbChunkStore("db3", "chunks");
    await expect(abortingStore.delete("a")).rejects.toThrow(/quota exceeded/i);
  });
});
