/**
 * Remembering the folder the user picked, so they pick it once.
 *
 * Issue #180. A `FileSystemDirectoryHandle` survives a reload only if it is
 * stored — IndexedDB can hold one, `localStorage` cannot, because a handle is
 * structured-cloneable and not a string. Permission does not survive with it:
 * on return the handle is intact and `queryPermission` says `"prompt"`, so the
 * page has to ask again, and asking requires a user gesture. That is one click
 * instead of a 5 GB download, and it is the reason this is worth doing at all.
 *
 * Everything here degrades to "no bound folder" rather than throwing. A browser
 * without the File System Access API, a user who declines, a handle whose
 * folder was deleted — all of them mean the same thing to the caller, which is
 * that it should fall back to fetching.
 */

const DB = "anima-web";
const STORE = "handles";
const KEY = "weights-folder";

async function open(): Promise<IDBDatabase | null> {
  if (typeof indexedDB === "undefined") return null;
  return new Promise((resolve) => {
    const request = indexedDB.open(DB, 1);
    request.onupgradeneeded = () => {
      if (!request.result.objectStoreNames.contains(STORE)) request.result.createObjectStore(STORE);
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => resolve(null);
  });
}

async function transact<T>(mode: IDBTransactionMode, body: (store: IDBObjectStore) => IDBRequest<T>): Promise<T | null> {
  const db = await open();
  if (!db) return null;
  return new Promise((resolve) => {
    let request: IDBRequest<T>;
    try {
      request = body(db.transaction(STORE, mode).objectStore(STORE));
    } catch {
      resolve(null);
      return;
    }
    request.onsuccess = () => resolve(request.result ?? null);
    request.onerror = () => resolve(null);
  });
}

/** The handle from a previous visit, or null. Says nothing about permission. */
export async function rememberedFolder(): Promise<FileSystemDirectoryHandle | null> {
  return transact<FileSystemDirectoryHandle>("readonly", (store) => store.get(KEY) as IDBRequest<FileSystemDirectoryHandle>);
}

export async function rememberFolder(handle: FileSystemDirectoryHandle): Promise<void> {
  await transact("readwrite", (store) => store.put(handle, KEY) as IDBRequest<unknown>);
}

export async function forgetFolder(): Promise<void> {
  await transact("readwrite", (store) => store.delete(KEY) as IDBRequest<unknown>);
}

type Permission = { queryPermission?: (d: { mode: string }) => Promise<PermissionState>; requestPermission?: (d: { mode: string }) => Promise<PermissionState> };

/**
 * Whether `handle` is usable for `mode` right now, **without prompting**.
 *
 * Separate from `request` below because prompting needs a user gesture: a page
 * that asks on load gets refused by the browser, silently, and then reports
 * that the folder is unavailable. Query on load, request on click.
 */
export async function hasPermission(handle: FileSystemDirectoryHandle, mode: "read" | "readwrite"): Promise<boolean> {
  const query = (handle as unknown as Permission).queryPermission;
  if (!query) return false;
  return (await query.call(handle, { mode })) === "granted";
}

/** Asks. Must be called from a user gesture; returns false if declined. */
export async function requestPermission(handle: FileSystemDirectoryHandle, mode: "read" | "readwrite"): Promise<boolean> {
  const request = (handle as unknown as Permission).requestPermission;
  if (!request) return false;
  return (await request.call(handle, { mode })) === "granted";
}

/** Shows the picker. Null if unsupported or declined. */
export async function pickFolder(): Promise<FileSystemDirectoryHandle | null> {
  const picker = (globalThis as { showDirectoryPicker?: (o: { mode: string; id?: string }) => Promise<FileSystemDirectoryHandle> })
    .showDirectoryPicker;
  if (!picker) return null;
  try {
    // `readwrite` because the folder is filled here, not only read: #180's flow
    // is "point at an empty folder and the page fills it".
    return await picker({ mode: "readwrite", id: "anima-weights" });
  } catch {
    // AbortError when the user closes the picker. Not an error to report.
    return null;
  }
}
