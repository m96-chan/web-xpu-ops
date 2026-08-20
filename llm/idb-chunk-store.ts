/**
 * The real `ChunkStore` (issue #121) backend: one IndexedDB database, one
 * object store, out-of-line string keys, `ArrayBuffer` values. Browser-only
 * (there is no `indexedDB` in Node), so this module's own correctness —
 * "IndexedDB actually behaves this way" — is proven only by the real-Chrome
 * verification in this issue's PR, never by `npm test`; what `npm test`
 * (`idb-chunk-store.test.ts`) *can* and does prove without a browser is
 * narrower: that IndexedDB's absence or failure is detected and reported
 * cleanly rather than silently mishandled — see that file's own doc.
 *
 * Minimal local ambient types, not the `"DOM"` lib, for the same reason
 * `storage-quota.ts` gives: this repository's `tsconfig.json` deliberately
 * excludes `"DOM"` (`tsconfig.build.json`'s note on `ops/`/`harness/`), and
 * `@types/node` ships no IndexedDB typing at all. Every type below is named
 * `Minimal*` rather than `IDB*` so it can never collide with the real
 * `lib.dom.d.ts` names a consumer with the `"DOM"` lib in scope (e.g.
 * `examples/llm-demo`'s own `tsconfig.json`, which bundles this file) already
 * has — these are structurally compatible with, but nominally unrelated to,
 * the real ones, and the only place that matters is the
 * `globalThis as unknown as {…}` cast below, not any `declare global`.
 */
import type { ChunkStore } from "./chunk-store.js";

interface MinimalIDBRequest<T = unknown> {
  readonly result: T;
  readonly error: unknown;
  onsuccess: (() => void) | null;
  onerror: (() => void) | null;
}

interface MinimalIDBOpenDBRequest extends MinimalIDBRequest<MinimalIDBDatabase> {
  onupgradeneeded: (() => void) | null;
  onblocked: (() => void) | null;
}

interface MinimalIDBObjectStore {
  put(value: unknown, key: string): MinimalIDBRequest;
  get(key: string): MinimalIDBRequest<ArrayBuffer | undefined>;
  delete(key: string): MinimalIDBRequest;
  getAllKeys(): MinimalIDBRequest<string[]>;
}

interface MinimalIDBTransaction {
  objectStore(name: string): MinimalIDBObjectStore;
  oncomplete: (() => void) | null;
  onabort: (() => void) | null;
  onerror: (() => void) | null;
  readonly error: unknown;
}

interface MinimalIDBDatabase {
  createObjectStore(name: string): unknown;
  transaction(storeNames: string, mode: "readonly" | "readwrite"): MinimalIDBTransaction;
  readonly objectStoreNames: { contains(name: string): boolean };
}

interface MinimalIDBFactory {
  open(name: string, version?: number): MinimalIDBOpenDBRequest;
}

function getIndexedDbFactory(): MinimalIDBFactory | undefined {
  return (globalThis as unknown as { indexedDB?: MinimalIDBFactory }).indexedDB;
}

/** Whether `globalThis.indexedDB` exists at all — not whether it will
 * actually work (Safari private mode has one that throws on `open()`; that
 * failure mode is a rejected `createIndexedDbChunkStore()` promise instead,
 * caught by `browser-weights.ts`'s own fallback, not by this check). */
export function isIndexedDbSupported(): boolean {
  return getIndexedDbFactory() !== undefined;
}

export const DEFAULT_IDB_DATABASE_NAME = "web-xpu-ops.weight-cache";
export const DEFAULT_IDB_STORE_NAME = "chunks";
const DB_VERSION = 1;

function errorOf(err: unknown): Error {
  return err instanceof Error ? err : new Error(String(err));
}

/**
 * No explicit `try`/`catch` around `idb.open()`: Safari private-mode has
 * historically thrown *synchronously* there instead of failing the request
 * asynchronously, but a `Promise` executor that throws is already caught and
 * turned into a rejection by the language itself (ECMA-262
 * `NewPromiseCapability` / `PromiseConstructor`), so wrapping it here would
 * be dead code — `idb-chunk-store.test.ts`'s "propagates a synchronous
 * throw…" case was written to prove exactly this behavior, and it caught
 * that redundancy: deleting an earlier hand-rolled `try`/`catch` here left
 * every test green (rule 1 — the code being deletable and still passing
 * meant the assertion wasn't observing what it claimed to).
 */
function openDatabase(dbName: string, storeName: string): Promise<MinimalIDBDatabase> {
  const idb = getIndexedDbFactory();
  if (!idb) {
    return Promise.reject(new Error("idb-chunk-store: indexedDB is not available in this environment"));
  }
  return new Promise((resolve, reject) => {
    const request: MinimalIDBOpenDBRequest = idb.open(dbName, DB_VERSION);
    request.onupgradeneeded = () => {
      const db = request.result;
      if (!db.objectStoreNames.contains(storeName)) db.createObjectStore(storeName);
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(errorOf(request.error));
    request.onblocked = () => reject(new Error("idb-chunk-store: open() blocked by another open connection"));
  });
}

function wrapRequest<T>(request: MinimalIDBRequest<T>): Promise<T> {
  return new Promise((resolve, reject) => {
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(errorOf(request.error));
  });
}

/**
 * Resolves once `transaction` *commits* (`oncomplete`), rejects if it
 * `onabort`s or `onerror`s — never on the individual request's own
 * `onsuccess`/`onerror`. A write request can report `onsuccess` (IndexedDB
 * accepted and queued it) and still have its transaction abort later during
 * commit — the textbook shape of a `QuotaExceededError` on a large value,
 * per spec enforced at commit time, not per request. Resolving on the
 * request instead (as an earlier revision of this module did — review item
 * #3, reproduced deterministically in `idb-chunk-store.test.ts` with a fake
 * `indexedDB` that fires exactly this sequence) means `put()` can report
 * success for a chunk that was never actually written, and the caller three
 * lines later — `browser-weights.ts`'s `writeCurrentVersion` — writes a
 * `current` record pointing at it as if it existed. The bytes are corrupt or
 * missing on the next read, silently, past the version this comment sits at
 * next to `browser-weights.ts`'s own "never happens" promise about exactly
 * that outcome.
 *
 * Only used for `readwrite` transactions (`put`/`delete` below); `get`/`list`
 * stay request-based (`wrapRequest`) — a read has nothing to roll back on
 * abort, so there is no durability gap for it to close.
 */
function runReadWriteTransaction(transaction: MinimalIDBTransaction): Promise<void> {
  return new Promise((resolve, reject) => {
    transaction.oncomplete = () => resolve();
    transaction.onabort = () => reject(errorOf(transaction.error ?? new Error("idb-chunk-store: transaction aborted")));
    transaction.onerror = () => reject(errorOf(transaction.error ?? new Error("idb-chunk-store: transaction error")));
  });
}

/**
 * Opens (creating on first use) `dbName`/`storeName` and returns a
 * `ChunkStore` over it. Each operation runs in its own transaction rather
 * than sharing one across `await`s — IndexedDB auto-commits a transaction
 * once its task queue drains, so a transaction held open across an `await`
 * boundary is exactly the bug that would produce, and every method below
 * only ever awaits a request it started in the same synchronous tick as the
 * transaction that owns it.
 */
export async function createIndexedDbChunkStore(
  dbName: string = DEFAULT_IDB_DATABASE_NAME,
  storeName: string = DEFAULT_IDB_STORE_NAME,
): Promise<ChunkStore> {
  const db = await openDatabase(dbName, storeName);
  return {
    async get(key: string): Promise<ArrayBuffer | undefined> {
      const store = db.transaction(storeName, "readonly").objectStore(storeName);
      return wrapRequest(store.get(key));
    },
    async put(key: string, value: ArrayBuffer): Promise<void> {
      const transaction = db.transaction(storeName, "readwrite");
      transaction.objectStore(storeName).put(value, key);
      await runReadWriteTransaction(transaction);
    },
    async delete(key: string): Promise<void> {
      const transaction = db.transaction(storeName, "readwrite");
      transaction.objectStore(storeName).delete(key);
      await runReadWriteTransaction(transaction);
    },
    async list(): Promise<string[]> {
      const store = db.transaction(storeName, "readonly").objectStore(storeName);
      return wrapRequest(store.getAllKeys());
    },
  };
}
