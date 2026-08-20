import { afterEach, describe, expect, it } from "vitest";
import { estimateStorageQuota, requestPersistentStorage } from "./storage-quota.js";

/**
 * `navigator.storage.estimate()` / `.persist()` wrappers (issue #121's
 * "quota不足・IDB不可…はHTTP直読みへ透過フォールバック" and
 * "`navigator.storage.persist()`要求と`estimate()`による容量事前チェック").
 *
 * Node has no `navigator` global at all, so the "unsupported" path below
 * runs for real every time this file executes under `npm test` — not a
 * mock standing in for that case, the actual absence. The "supported"
 * paths inject a fake `navigator.storage` onto `globalThis` (restored in
 * `afterEach`) since there is no real one to test against outside a
 * browser; `browser-weights.cache.test.ts` and the real-Chrome
 * verification in the PR are what exercise a genuine `StorageManager`.
 */

const originalNavigator = (globalThis as { navigator?: unknown }).navigator;

afterEach(() => {
  (globalThis as { navigator?: unknown }).navigator = originalNavigator;
});

function installFakeNavigatorStorage(storage: Record<string, unknown>): void {
  (globalThis as { navigator?: unknown }).navigator = { storage };
}

describe("estimateStorageQuota", () => {
  it("returns null when there is no navigator at all (Node)", async () => {
    delete (globalThis as { navigator?: unknown }).navigator;
    await expect(estimateStorageQuota()).resolves.toBeNull();
  });

  it("returns null when navigator.storage has no estimate()", async () => {
    installFakeNavigatorStorage({});
    await expect(estimateStorageQuota()).resolves.toBeNull();
  });

  it("returns usageBytes/quotaBytes from a real-shaped estimate()", async () => {
    installFakeNavigatorStorage({
      estimate: async () => ({ usage: 123, quota: 456 }),
    });
    await expect(estimateStorageQuota()).resolves.toEqual({ usageBytes: 123, quotaBytes: 456 });
  });

  it("returns null when estimate() resolves without numeric usage/quota", async () => {
    installFakeNavigatorStorage({
      estimate: async () => ({ usage: undefined, quota: undefined }),
    });
    await expect(estimateStorageQuota()).resolves.toBeNull();
  });

  it("returns null (not a rejection) when estimate() itself throws", async () => {
    installFakeNavigatorStorage({
      estimate: async () => {
        throw new Error("boom");
      },
    });
    await expect(estimateStorageQuota()).resolves.toBeNull();
  });
});

describe("requestPersistentStorage", () => {
  it("returns false when there is no navigator at all (Node)", async () => {
    delete (globalThis as { navigator?: unknown }).navigator;
    await expect(requestPersistentStorage()).resolves.toBe(false);
  });

  it("returns false when navigator.storage has no persist()", async () => {
    installFakeNavigatorStorage({});
    await expect(requestPersistentStorage()).resolves.toBe(false);
  });

  it("returns persist()'s own boolean result", async () => {
    installFakeNavigatorStorage({ persist: async () => true });
    await expect(requestPersistentStorage()).resolves.toBe(true);
    installFakeNavigatorStorage({ persist: async () => false });
    await expect(requestPersistentStorage()).resolves.toBe(false);
  });

  it("returns false (not a rejection) when persist() itself throws", async () => {
    installFakeNavigatorStorage({
      persist: async () => {
        throw new Error("boom");
      },
    });
    await expect(requestPersistentStorage()).resolves.toBe(false);
  });
});
