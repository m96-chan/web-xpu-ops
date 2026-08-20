/**
 * `navigator.storage` (the Storage Manager API) wrappers for the persistent
 * weight cache (issue #121). Two calls only: `estimate()` for the upfront
 * quota check `weight-cache.ts#decideCacheStrategy` reads, and `persist()`
 * so a granted cache is not silently evicted under storage pressure the way
 * "best-effort" IndexedDB data can be.
 *
 * `@types/node` (see `tsconfig.json`'s deliberate lack of the `"DOM"` lib —
 * `tsconfig.build.json`'s own note on why `ops/`/`harness/` are checked
 * without it) ships no `navigator.storage` typing at all — its
 * `web-globals/navigator.d.ts` only adds `hardwareConcurrency` /
 * `language` / `platform` / `userAgent`, the Worker-safe subset, not the
 * Storage Manager API. The minimal local types below cover exactly the two
 * calls this file makes, the same approach `idb-chunk-store.ts` takes for
 * IndexedDB — read through `globalThis`, not through the global
 * `navigator`/`Navigator` symbols `@types/node` already declares, so there
 * is no ambient type to conflict with in a consumer that *does* have the
 * `"DOM"` lib (`examples/llm-demo`'s own `tsconfig.json`).
 */

export interface QuotaEstimate {
  usageBytes: number;
  quotaBytes: number;
}

interface MinimalStorageManager {
  estimate?: () => Promise<{ usage?: number; quota?: number }>;
  persist?: () => Promise<boolean>;
}

function getStorageManager(): MinimalStorageManager | undefined {
  const nav = (globalThis as unknown as { navigator?: { storage?: MinimalStorageManager } }).navigator;
  return nav?.storage;
}

/**
 * `null` whenever the result cannot be trusted — no `navigator`, no
 * `navigator.storage.estimate`, a rejected call, or a resolved value missing
 * either number (rather than treating a partial/garbage result as "zero
 * usage" or "zero quota", either of which would make
 * `decideCacheStrategy` reach a wrong-but-confident answer instead of the
 * honest "unknown" it should fall back to).
 */
export async function estimateStorageQuota(): Promise<QuotaEstimate | null> {
  const storage = getStorageManager();
  if (!storage?.estimate) return null;
  try {
    const { usage, quota } = await storage.estimate();
    if (typeof usage !== "number" || typeof quota !== "number") return null;
    return { usageBytes: usage, quotaBytes: quota };
  } catch {
    return null;
  }
}

/**
 * `false` for every failure mode (no `navigator.storage.persist`, a
 * rejected call) as well as for a genuine "denied" result — this function
 * only ever tells a caller whether persistence is now in effect, which a
 * caller (`browser-weights.ts`) treats as best-effort and never as a
 * precondition for caching at all (a non-persistent IndexedDB entry is
 * still a valid, just more evictable, cache hit).
 */
export async function requestPersistentStorage(): Promise<boolean> {
  const storage = getStorageManager();
  if (!storage?.persist) return false;
  try {
    return await storage.persist();
  } catch {
    return false;
  }
}
