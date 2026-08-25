/**
 * Where a weight's bytes come from, as one method.
 *
 * Issue #180. Everything this demo reads, it reads as a byte range: the
 * manifest names each tensor's offset and length, and nothing ever wants a
 * whole file. That is true of an HTTP `Range` request and it is true of
 * `File.slice()`, which is what makes swapping one for the other a change of
 * transport rather than a change of loader.
 *
 *     fetch(url, { headers: { Range: `bytes=${a}-${b}` } })
 *     file.slice(a, b + 1).arrayBuffer()
 *
 * The same operation against the same bytes, one of them without a socket.
 *
 * Modelled on `llm/chunk-store.ts`'s `ChunkStore`, and for the same reason: the
 * logic above it is written against the interface and never against `fetch` or
 * `showDirectoryPicker` directly, so it can be exercised in Node, which has
 * neither.
 */

/** One method, because one method is all the loader above this needs. */
export interface ByteSource {
  /**
   * `byteLength` bytes of `file`, starting at `byteOffset`.
   *
   * Implementations return exactly that many bytes or throw. A short read is
   * the failure this interface exists to make impossible to ignore: it would
   * otherwise arrive as a tensor of the right shape holding the wrong numbers,
   * which is a wrong image rather than an error.
   */
  read(file: string, byteOffset: number, byteLength: number): Promise<ArrayBuffer>;
  /** For error messages — where this source is reading from. */
  readonly describe: string;
}

/**
 * The default base for the converted weights.
 *
 * **Measured, not assumed** (#180): against this host a `Range` request answers
 * `206` with `content-range: bytes 0-15/3751673856` and
 * `access-control-allow-origin: *`, and the redirect that precedes it carries
 * the header too. So the existing reader works against it unchanged.
 *
 * The licence is **not** this repository's — see that repository's `LICENSE.md`
 * and `NOTICE`, and this repository's README. Converting a model did not
 * relicense it.
 */
export const DEFAULT_WEIGHTS_BASE =
  "https://huggingface.co/m96-chan/Anima-3.8B-q8-web-xpu-ops/resolve/main";

/**
 * Reads over HTTP `Range`.
 *
 * No caching of its own — a caller that wants the bytes kept wraps this in
 * something that keeps them. Mixing the two was how the previous loader ended
 * up re-reading a cache that could not hold a forward's worth (#186).
 */
export class HttpByteSource implements ByteSource {
  readonly #base: string;

  constructor(base: string = DEFAULT_WEIGHTS_BASE) {
    this.#base = base.replace(/\/+$/, "");
  }

  get describe(): string {
    return this.#base;
  }

  async read(file: string, byteOffset: number, byteLength: number): Promise<ArrayBuffer> {
    if (byteLength === 0) return new ArrayBuffer(0);
    const url = `${this.#base}/${file}`;
    const response = await fetch(url, {
      headers: { Range: `bytes=${byteOffset}-${byteOffset + byteLength - 1}` },
    });
    if (response.status !== 206) {
      throw new Error(
        `byte-source: ${url} answered ${response.status} for a Range request. ` +
          "A 200 here means the whole file was sent, which for a 3.5 GB weight is not a slow path but a broken one.",
      );
    }
    const buffer = await response.arrayBuffer();
    if (buffer.byteLength !== byteLength) {
      throw new Error(`byte-source: ${url} returned ${buffer.byteLength} bytes, expected ${byteLength}.`);
    }
    return buffer;
  }
}

/**
 * Reads from a directory the user bound with the File System Access API.
 *
 * `getFile()` is re-taken per read rather than held: a `File` is a snapshot of
 * the entry at the moment it was taken, and a handle kept across a write —
 * which is exactly what the provisioning path does — can go stale.
 */
export class DirectoryByteSource implements ByteSource {
  readonly #dir: FileSystemDirectoryHandle;

  constructor(dir: FileSystemDirectoryHandle) {
    this.#dir = dir;
  }

  get describe(): string {
    return `the folder "${this.#dir.name}"`;
  }

  async read(file: string, byteOffset: number, byteLength: number): Promise<ArrayBuffer> {
    if (byteLength === 0) return new ArrayBuffer(0);
    let handle: FileSystemFileHandle;
    try {
      handle = await this.#dir.getFileHandle(file);
    } catch {
      throw new Error(
        `byte-source: ${this.describe} has no "${file}". ` +
          "Either it holds a different model or the download did not finish.",
      );
    }
    const blob = await handle.getFile();
    if (byteOffset + byteLength > blob.size) {
      throw new Error(
        `byte-source: ${file} in ${this.describe} is ${blob.size} bytes; ` +
          `something asked for ${byteOffset}..${byteOffset + byteLength}. The file is truncated.`,
      );
    }
    return blob.slice(byteOffset, byteOffset + byteLength).arrayBuffer();
  }
}

/** Whether this browser can bind a folder at all. */
export function directoryBindingSupported(): boolean {
  return typeof (globalThis as { showDirectoryPicker?: unknown }).showDirectoryPicker === "function";
}
