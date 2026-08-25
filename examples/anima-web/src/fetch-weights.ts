/**
 * Anima's weights, over HTTP `Range` requests.
 *
 * The browser counterpart to `examples/anima/src/weights-node.ts`. Same rule:
 * **one tensor at a time**. The DiT is 3.76 GB packed, the encoder 1.19 GB and
 * the VAE 0.25 GB; none of them is something to hold on the heap.
 *
 * The mechanics are `examples/zimage-web/src/fetch-weights.ts`'s, and the two
 * hard-won parts are carried over verbatim rather than re-derived:
 *
 *   - the cache key is a **query**, not a fragment. `new Request(url)` strips
 *     fragments, so every byte range of a file would collapse onto whichever
 *     was stored first — bytes of the right length for the wrong tensor, and a
 *     black picture.
 *   - the returned length is **checked**, because that first failure was silent
 *     precisely because nothing compared what came back against what was asked.
 *
 * What is Anima's own is the manifest: `convert_dit.py` writes `kind: "q8"`
 * with per-row scales against Z-Image's q4-g128 groups, so `FetchedDitWeights`
 * cannot be reused — it refuses a manifest without a `groupSize`.
 */
import { bf16ToF32 } from "../../zimage/src/bf16.js";
import { dequantizeQ8 } from "../../zimage/src/weights.js";
import type { AnimaManifest, AnimaTensor } from "../../anima/src/manifest.js";
import type { ByteSource } from "./byte-source.js";

let store: Cache | null | undefined;
async function cacheStore(): Promise<Cache | null> {
  if (store !== undefined) return store;
  try {
    store = typeof caches === "undefined" ? null : await caches.open("anima-weights-v1");
  } catch {
    store = null;
  }
  return store;
}

/**
 * Reads a range through a `ByteSource`, keeping a copy in the Cache API.
 *
 * Only for sources that are worth caching. A `DirectoryByteSource` already
 * reads from the user's own disk, so putting a second copy in the origin's
 * quota would be the 5 GB duplication #180 exists to avoid — `cachedRange`
 * asks the source whether it wants that.
 */
async function readRange(url: string, byteOffset: number, byteLength: number): Promise<ArrayBuffer> {
  if (byteLength === 0) return new ArrayBuffer(0);
  const response = await fetch(url, {
    headers: { Range: `bytes=${byteOffset}-${byteOffset + byteLength - 1}` },
  });
  if (response.status !== 206) {
    throw new Error(
      `fetch-weights: ${url} answered ${response.status} for a Range request. ` +
        `This demo needs a server that honours Range; a 200 here means the whole file was sent.`,
    );
  }
  const buffer = await response.arrayBuffer();
  if (buffer.byteLength !== byteLength) {
    throw new Error(`fetch-weights: ${url} returned ${buffer.byteLength} bytes, expected ${byteLength}.`);
  }
  return buffer;
}

/** A byte range, from the browser's disk cache when it has been seen before. */
async function cachedRange(
  source: ByteSource | null,
  url: string,
  file: string,
  byteOffset: number,
  byteLength: number,
): Promise<ArrayBuffer> {
  // A bound folder is already on the user's disk, outside the origin's quota
  // and under their control. Caching it again would put 5 GB back into the
  // quota this exists to leave.
  if (source) return source.read(file, byteOffset, byteLength);

  const cache = await cacheStore();
  if (!cache) return readRange(url, byteOffset, byteLength);

  const key = `${url}?o=${byteOffset}&n=${byteLength}`;
  const hit = await cache.match(key);
  if (hit) {
    const cached = await hit.arrayBuffer();
    if (cached.byteLength === byteLength) return cached;
    await cache.delete(key);
  }

  const buffer = await readRange(url, byteOffset, byteLength);
  await cache.put(key, new Response(buffer.slice(0)));
  return buffer;
}

/**
 * The converted DiT: on disk after the first visit, a layer at a time in memory.
 *
 * `maxCachedTensors` bounds what the heap holds. The bytes themselves live in
 * the browser's Cache API, which is disk-backed, so a second generation does no
 * network and the heap never holds more than the layer in flight.
 */
export class FetchedAnimaWeights {
  readonly config: Record<string, number | boolean | string>;
  readonly #base: string;
  readonly #byName = new Map<string, AnimaTensor>();
  readonly #cache = new Map<string, Float32Array>();
  readonly #packed = new Map<string, { codes: Uint32Array; scale: Float32Array; N: number; K: number }>();
  readonly #limit: number;
  /**
   * Where the bytes come from, or null for "over HTTP, through the Cache API".
   *
   * Null is the old behaviour and stays the default, so a browser without the
   * File System Access API keeps working exactly as before (#180).
   */
  readonly #source: ByteSource | null;

  private constructor(
    base: string,
    manifest: AnimaManifest & { config?: Record<string, number | boolean | string> },
    limit: number,
    source: ByteSource | null,
  ) {
    this.#base = base;
    this.#limit = limit;
    this.#source = source;
    if (!manifest.config) {
      throw new Error(
        `fetch-weights: ${base}/dit.manifest.json carries no "config" — it predates ` +
          "convert_dit.py recording one. Re-run the conversion.",
      );
    }
    this.config = manifest.config;
    for (const tensor of manifest.tensors) this.#byName.set(tensor.name, tensor);
  }

  static async open(
    base: string,
    maxCachedTensors = 48,
    source: ByteSource | null = null,
  ): Promise<FetchedAnimaWeights> {
    // Through the source when there is one: a bound folder has the manifest in
    // it, and reaching past the source to `fetch` would put the network back in
    // a path that is supposed to have none.
    const response = source
      ? new Response(await source.read("dit.manifest.json", 0, await source.size("dit.manifest.json")))
      : await fetch(`${base}/dit.manifest.json`);
    if (!response.ok) {
      throw new Error(
        `fetch-weights: no manifest at ${base}/dit.manifest.json (${response.status}). ` +
          "Point the server at a directory produced by examples/anima/tools/convert_dit.py.",
      );
    }
    const manifest = (await response.json()) as AnimaManifest & { format?: { quant?: string } };
    if (manifest.format?.quant !== "q8-per-row") {
      throw new Error(
        `fetch-weights: manifest declares quantization ${JSON.stringify(manifest.format?.quant)}, ` +
          "not q8-per-row. This is the Anima loader; Z-Image's is in examples/zimage-web.",
      );
    }
    return new FetchedAnimaWeights(base, manifest, maxCachedTensors, source);
  }

  has(name: string): boolean {
    return this.#byName.has(name);
  }

  shapeOf(name: string): number[] | undefined {
    return this.#byName.get(name)?.shape;
  }

  /** Every tensor in the manifest, for `preloadAll`'s progress. */
  get tensorCount(): number {
    return this.#byName.size;
  }

  /** Preload a set of names, in parallel, before a forward touches them. */
  async preload(names: string[]): Promise<void> {
    await Promise.all(names.filter((n) => this.#byName.has(n)).map((n) => this.#fetch(n)));
  }

  /** Everything whose name starts with `prefix`, into the heap. */
  async preloadPrefix(prefix: string): Promise<void> {
    await this.preload([...this.#byName.keys()].filter((n) => n.startsWith(prefix)));
  }

  /**
   * Every tensor, into the browser's **disk** cache — not the heap.
   *
   * Fetching per block instead put the whole model on the wire in every
   * denoising step: the same bytes, forty times over. This pays the download
   * once, and `preloadPrefix` afterwards reads from disk and touches the
   * network not at all.
   *
   * The bytes are deliberately discarded here. Holding them would be 3.76 GB
   * on the heap, and the next allocation to fail is an activation.
   */
  async preloadAll(onProgress?: (done: number, total: number, bytes: number) => void): Promise<void> {
    const names = [...this.#byName.keys()];
    let bytes = 0;
    const CONCURRENCY = 8;
    for (let at = 0; at < names.length; at += CONCURRENCY) {
      const batch = names.slice(at, at + CONCURRENCY);
      await Promise.all(batch.map(async (name) => {
        bytes += await this.#cacheOnly(name);
      }));
      onProgress?.(Math.min(at + CONCURRENCY, names.length), names.length, bytes);
    }
  }

  /** Puts a tensor's bytes in the disk cache and keeps nothing. */
  async #cacheOnly(name: string): Promise<number> {
    const tensor = this.#byName.get(name)!;
    if (tensor.kind === "q8") {
      const [N, K] = tensor.shape as [number, number];
      const words = Math.ceil(K / 4) * N;
      await Promise.all([
        cachedRange(this.#source, `${this.#base}/dit.q8.bin`, "dit.q8.bin", tensor.codesOffset! * 4, words * 4),
        cachedRange(this.#source, `${this.#base}/dit.q8scales.bin`, "dit.q8scales.bin", tensor.scaleOffset! * 4, N * 4),
      ]);
      return words * 4 + N * 4;
    }
    const count = tensor.shape.reduce((a, b) => a * b, 1);
    await cachedRange(this.#source, `${this.#base}/dit.f32.bin`, "dit.f32.bin", tensor.offset! * 4, count * 4);
    return count * 4;
  }

  /**
   * The packed codes for a quantized tensor, or `null`.
   *
   * `matmulQ8` reads them directly, so a quantized weight never dequantizes to
   * four times its size on the way to the device.
   */
  packedQ8(name: string): { codes: Uint32Array; scale: Float32Array; N: number; K: number } | null {
    const tensor = this.#byName.get(name);
    if (!tensor || tensor.kind !== "q8") return null;
    const packed = this.#packed.get(name);
    if (!packed) {
      throw new Error(`fetch-weights: "${name}" was read before it was preloaded.`);
    }
    return packed;
  }

  get(name: string): Float32Array {
    const cached = this.#cache.get(name);
    if (cached) return cached;
    const packed = this.#packed.get(name);
    if (packed) {
      const dense = dequantizeQ8({ packed: packed.codes, scale: packed.scale, N: packed.N, K: packed.K });
      this.#remember(name, dense);
      return dense;
    }
    throw new Error(`fetch-weights: "${name}" was read before it was preloaded.`);
  }

  #remember(name: string, value: Float32Array): void {
    this.#cache.set(name, value);
    // Least-recently-*inserted*, and re-inserting on a hit is what makes it
    // least-recently-used. The first version did not, so `preload` evicted the
    // tensors it had just fetched.
    while (this.#cache.size > this.#limit) {
      const oldest = this.#cache.keys().next().value as string;
      this.#cache.delete(oldest);
    }
  }

  async #fetch(name: string): Promise<void> {
    const tensor = this.#byName.get(name);
    if (!tensor) throw new Error(`fetch-weights: no tensor named "${name}"`);
    if (tensor.kind === "q8") {
      if (this.#packed.has(name)) return;
      const [N, K] = tensor.shape as [number, number];
      const words = Math.ceil(K / 4) * N;
      const [codes, scale] = await Promise.all([
        cachedRange(this.#source, `${this.#base}/dit.q8.bin`, "dit.q8.bin", tensor.codesOffset! * 4, words * 4),
        cachedRange(this.#source, `${this.#base}/dit.q8scales.bin`, "dit.q8scales.bin", tensor.scaleOffset! * 4, N * 4),
      ]);
      this.#packed.set(name, {
        codes: new Uint32Array(codes), scale: new Float32Array(scale), N, K,
      });
      // Bounded the same way the dense cache is. Packed tensors are a quarter
      // the size but there are 898 of them.
      while (this.#packed.size > this.#limit * 4) {
        const oldest = this.#packed.keys().next().value as string;
        this.#packed.delete(oldest);
      }
      return;
    }
    if (this.#cache.has(name)) return;
    const count = tensor.shape.reduce((a, b) => a * b, 1);
    const bytes = await cachedRange(this.#source, `${this.#base}/dit.f32.bin`, "dit.f32.bin", tensor.offset! * 4, count * 4);
    this.#remember(name, new Float32Array(bytes));
  }
}

/** One `.safetensors` file, read a tensor at a time over `Range`. */
export class FetchedSafetensors {
  readonly #url: string;
  readonly #file: string;
  readonly #dataStart: number;
  readonly #entries: Map<string, { dtype: string; shape: number[]; data_offsets: [number, number] }>;
  readonly #source: ByteSource | null;

  private constructor(
    url: string,
    file: string,
    dataStart: number,
    entries: Map<string, { dtype: string; shape: number[]; data_offsets: [number, number] }>,
    source: ByteSource | null,
  ) {
    this.#url = url;
    this.#file = file;
    this.#dataStart = dataStart;
    this.#entries = entries;
    this.#source = source;
  }

  /** `file` is the name within `source`; `url` is used only when there is none. */
  static async open(url: string, source: ByteSource | null = null, file = url.split("/").pop()!): Promise<FetchedSafetensors> {
    const head = await cachedRange(source, url, file, 0, 8);
    const headerLength = Number(new DataView(head).getBigUint64(0, true));
    const headerBytes = await cachedRange(source, url, file, 8, headerLength);
    const header = JSON.parse(new TextDecoder().decode(headerBytes)) as Record<string, unknown>;
    const entries = new Map<string, { dtype: string; shape: number[]; data_offsets: [number, number] }>();
    for (const [name, value] of Object.entries(header)) {
      if (name === "__metadata__") continue;
      entries.set(name, value as { dtype: string; shape: number[]; data_offsets: [number, number] });
    }
    return new FetchedSafetensors(url, file, 8 + headerLength, entries, source);
  }

  has(name: string): boolean {
    return this.#entries.has(name);
  }

  shapeOf(name: string): number[] | undefined {
    return this.#entries.get(name)?.shape;
  }

  async read(name: string): Promise<Float32Array> {
    const entry = this.#entries.get(name);
    if (!entry) throw new Error(`fetch-weights: ${this.#url} has no tensor "${name}"`);
    const [from, to] = entry.data_offsets;
    const bytes = await cachedRange(this.#source, this.#url, this.#file, this.#dataStart + from, to - from);
    if (entry.dtype === "F32") return new Float32Array(bytes);
    if (entry.dtype === "BF16") return bf16ToF32(new Uint16Array(bytes));
    throw new Error(`fetch-weights: "${name}" is ${entry.dtype}; only F32 and BF16 are read.`);
  }

  /**
   * Just the rows an embedding lookup wants.
   *
   * Qwen3-0.6B's table is 151,936 x 1024 — 0.31 GB in bf16, of which a prompt
   * reads a couple of dozen rows. Fetching them individually is a `Range` each
   * and is what makes the encoder start in a second rather than a minute.
   */
  async readRows(name: string, rows: Int32Array, width: number): Promise<Float32Array> {
    const entry = this.#entries.get(name);
    if (!entry) throw new Error(`fetch-weights: ${this.#url} has no tensor "${name}"`);
    const elementBytes = entry.dtype === "BF16" ? 2 : 4;
    const [from] = entry.data_offsets;
    const out = new Float32Array(rows.length * width);
    await Promise.all(
      Array.from(rows, async (row, i) => {
        const bytes = await cachedRange(this.#source, this.#url, this.#file, this.#dataStart + from + row * width * elementBytes, width * elementBytes);
        out.set(entry.dtype === "BF16" ? bf16ToF32(new Uint16Array(bytes)) : new Float32Array(bytes), i * width);
      }),
    );
    return out;
  }
}
