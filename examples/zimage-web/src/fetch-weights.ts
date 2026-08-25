/**
 * The weights, over HTTP `Range` requests.
 *
 * The browser counterpart to `examples/zimage/src/weights-node.ts` and
 * `safetensors.ts`. Same rule as those: **one tensor at a time**. The DiT is
 * 6.17 GB packed and the text encoder 8 GB; neither is something to hold, and
 * `Float32Array` gives out well before either anyway.
 *
 * Every read is a `Range` request, so the server has to answer them — see
 * `server.mjs`. A server that ignores `Range` and returns 200 with the whole
 * file would "work" while downloading gigabytes per tensor, so the status is
 * checked: 206 is required, and a 200 is an error rather than a slow success.
 */
import type { ByteSource } from "../../web-common/src/byte-source.js";
import {
  GROUP_SIZE,
  dequantizeQ4G128,
  dequantizeQ8,
  type DitManifest,
  type DitTensor,
} from "../../zimage/src/weights.js";
import { bf16ToF32 } from "../../zimage/src/bf16.js";

/**
 * The browser's own on-disk cache, opened once.
 *
 * `null` when the page is not in a secure context (the Cache API is
 * unavailable over plain `http://` on some origins) — in which case everything
 * below still works and simply costs the network again.
 */
let store: Cache | null | undefined;
async function cacheStore(): Promise<Cache | null> {
  if (store !== undefined) return store;
  try {
    store = typeof caches === "undefined" ? null : await caches.open("zimage-weights-v2");
  } catch {
    store = null;
  }
  return store;
}

/**
 * A byte range, from the browser's disk cache when it has been seen before.
 *
 * The first version of this held every tensor in JavaScript memory so that a
 * generation did no network. That worked and moved the problem: 6.17 GB on the
 * heap left too little for the activations, and 512x512's 129 MB of attention
 * scores threw `Array buffer allocation failed` from inside a dispatch.
 *
 * The Cache API is the right place for bytes that are large, immutable, and
 * wanted again — it is backed by disk rather than the heap, so the model can be
 * fetched once and re-read every step without the heap ever holding more than
 * the layer in flight.
 */
/**
 * A byte range, through a `ByteSource` when there is one.
 *
 * Issue #194. A bound folder is already on the user's disk, outside the
 * origin's quota and under their control; caching it again would put the whole
 * model back into the quota the folder exists to leave.
 */
async function cachedRange(
  source: ByteSource | null,
  url: string,
  file: string,
  byteOffset: number,
  byteLength: number,
): Promise<ArrayBuffer> {
  if (source) return source.read(file, byteOffset, byteLength);
  return cachedRangeHttp(url, byteOffset, byteLength);
}

async function cachedRangeHttp(url: string, byteOffset: number, byteLength: number): Promise<ArrayBuffer> {
  const cache = await cacheStore();
  if (!cache) return readRange(url, byteOffset, byteLength);

  // A **query**, not a fragment. `new Request(url)` strips the fragment, so
  // `url#0+1024` and `url#4096+2048` are one cache entry — every range of a
  // file collapsing onto whichever was stored first. That returns bytes rather
  // than an error, of the right length for the wrong tensor, and the picture
  // comes out black.
  const key = `${url}?o=${byteOffset}&n=${byteLength}`;
  const hit = await cache.match(key);
  if (hit) {
    const cached = await hit.arrayBuffer();
    // The length is the one thing a wrong entry cannot fake, so it is checked
    // rather than trusted — the failure above was silent precisely because
    // nothing compared what came back against what was asked for.
    if (cached.byteLength === byteLength) return cached;
    await cache.delete(key);
  }

  const buffer = await readRange(url, byteOffset, byteLength);
  // `put` reads the body, so the stored copy is separate from what is returned.
  await cache.put(key, new Response(buffer.slice(0)));
  return buffer;
}

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

/**
 * The converted DiT: on disk after the first visit, a layer at a time in memory.
 *
 * Two wrong answers came before this one, and both are worth keeping written
 * down because they are the same mistake pointed in opposite directions.
 *
 * The first mirrored the Node loader and held a couple of dozen tensors. That
 * is right behind a file, where the OS page cache does the remembering, and
 * wrong over HTTP where nothing does: a forward touches every tensor once, so
 * any bound below the model size re-fetches all of it every step — 6.17 GB per
 * step, 49 GB for eight.
 *
 * The second held everything, which quieted the network and put 6.17 GB on the
 * heap. The next allocation to fail was an activation: 512x512's readbacks
 * threw `Array buffer allocation failed` from inside a dispatch.
 *
 * The bytes are large, immutable and wanted again, which describes a disk
 * cache rather than a heap. `cachedRange` puts them in the browser's, so a
 * generation does no network **and** memory holds only the layer in flight.
 * `maxCachedTensors` bounds that layer's worth.
 *
 * Only the packed form is held for a quantized weight. `matmulQ8` reads the
 * codes directly, so nothing dequantises to four times the size on the way in.
 */
export class FetchedDitWeights {
  readonly config: Record<string, unknown>;
  readonly #base: string;
  readonly #byName = new Map<string, DitTensor>();
  readonly #cache = new Map<string, Float32Array>();
  readonly #packedCache = new Map<string, { codes: Uint32Array; scale: Float32Array; N: number; K: number }>();
  readonly #limit: number;
  /** Where the bytes come from, or null for "over HTTP, through the Cache API". */
  readonly #source: ByteSource | null;

  private constructor(base: string, manifest: DitManifest, limit: number, source: ByteSource | null) {
    this.#base = base;
    this.#source = source;
    this.#limit = limit;
    this.config = manifest.config;
    for (const tensor of manifest.tensors) this.#byName.set(tensor.name, tensor);
  }

  static async open(
    base: string,
    maxCachedTensors = 48,
    source: ByteSource | null = null,
  ): Promise<FetchedDitWeights> {
    const response = await fetch(`${base}/dit.manifest.json`);
    if (!response.ok) {
      throw new Error(
        `fetch-weights: no manifest at ${base}/dit.manifest.json (${response.status}). ` +
          `Point the server at a directory produced by tools/convert_dit.py.`,
      );
    }
    const manifest = (await response.json()) as DitManifest;
    if (manifest.format.groupSize !== GROUP_SIZE) {
      throw new Error(`fetch-weights: manifest declares group size ${manifest.format.groupSize}, not ${GROUP_SIZE}.`);
    }
    return new FetchedDitWeights(base, manifest, maxCachedTensors, source);
  }

  /** Total bytes fetched, for the progress read-out — the number the user is waiting on. */
  bytesHeld = 0;

  /** Every tensor in the manifest, for `preloadAll`'s progress. */
  get tensorCount(): number {
    return this.#byName.size;
  }

  #remember<T>(map: Map<string, T>, name: string, value: T): T {
    // Delete first, so re-setting an existing key moves it to the newest end.
    // `Map.set` alone keeps the original position, which is what let eviction
    // pick the wrong entries.
    map.delete(name);
    map.set(name, value);
    while (map.size > this.#limit) {
      const oldest = map.keys().next().value as string;
      map.delete(oldest);
    }
    return value;
  }

  /**
   * Counts a cache hit as a use.
   *
   * Skipping this was a real bug. `preload` fetches a prefix's tensors
   * together; one of them already being resident meant an early return that
   * left it at its old position — old enough that the other three, inserted in
   * the same call, evicted it. The preload then reported success for a tensor
   * it had just discarded, and the failure surfaced as "read before it was
   * preloaded" somewhere else entirely.
   *
   * An LRU whose reads do not count as use is not an LRU.
   */
  #touch<T>(map: Map<string, T>, name: string, value: T): T {
    map.delete(name);
    map.set(name, value);
    return value;
  }

  /**
   * Everything whose name starts with `prefix`, into the cache.
   *
   * Awaited by `ditForwardGpu`'s `prefetch` hook before each block, which is
   * what lets `get`/`packedQ8` below stay synchronous and identical in shape to
   * the Node loader's.
   */
  async preload(prefix: string): Promise<void> {
    const wanted = [...this.#byName.keys()].filter((name) => name.startsWith(prefix));
    if (wanted.length === 0) {
      throw new Error(`fetch-weights: preload(${JSON.stringify(prefix)}) matched no tensor in the manifest.`);
    }
    await Promise.all(
      wanted.map((name) => (this.#byName.get(name)!.kind === "q8" ? this.#fetchPacked(name) : this.#fetchDense(name))),
    );

    // Checked here rather than discovered later. A tensor that is not resident
    // after its own preload becomes "read before it was preloaded" several
    // frames downstream, which names the symptom and not the cause — and the
    // cause has been a different thing each time: the hook placed below its own
    // use, then a bound too small to hold what one call asks for.
    const missing = wanted.filter((name) => !this.#cache.has(name) && !this.#packedCache.has(name));
    if (missing.length > 0) {
      throw new Error(
        `fetch-weights: preload(${JSON.stringify(prefix)}) fetched ${wanted.length} tensors but ` +
          `${missing.length} are not resident (${missing.slice(0, 3).join(", ")}). ` +
          `Cache limit is ${this.#limit}; dense ${this.#cache.size}, packed ${this.#packedCache.size}.`,
      );
    }
  }

  /**
   * The whole model, once.
   *
   * Called before the first generation so that the wait is one visible download
   * rather than a stall inside every denoising step. After it, `preload` finds
   * everything already cached and the network goes quiet.
   */
  async preloadAll(onProgress?: (done: number, total: number, bytes: number) => void): Promise<void> {
    const names = [...this.#byName.keys()];
    let done = 0;
    // Eight at a time: enough to keep the connection busy, few enough that a
    // browser's per-host request limit is not what decides the pace.
    const WIDTH = 8;
    for (let i = 0; i < names.length; i += WIDTH) {
      await Promise.all(
        names.slice(i, i + WIDTH).map(async (name) => {
          const tensor = this.#byName.get(name)!;
          if (tensor.kind === "q8") await this.#fetchPacked(name);
          else await this.#fetchDense(name);
          done += 1;
        }),
      );
      onProgress?.(done, names.length, this.bytesHeld);
    }
  }

  /** A tensor's shape without fetching it — the manifest already has every one. */
  shapeOf(name: string): number[] | undefined {
    return this.#byName.get(name)?.shape;
  }

  /** Cached only. Throws rather than returning a stale or empty tensor. */
  get(name: string): Float32Array {
    const cached = this.#cache.get(name);
    if (cached) return this.#touch(this.#cache, name, cached);
    const packed = this.#packedCache.get(name);
    if (packed) {
      return this.#remember(
        this.#cache,
        name,
        dequantizeQ8({ packed: packed.codes, scale: packed.scale, N: packed.N, K: packed.K }),
      );
    }
    throw new Error(`fetch-weights: ${JSON.stringify(name)} was read before it was preloaded.`);
  }

  packedQ8(name: string): { codes: Uint32Array; scale: Float32Array; N: number; K: number } | null {
    return this.#packedCache.get(name) ?? null;
  }

  async #fetchPacked(name: string): Promise<{ codes: Uint32Array; scale: Float32Array; N: number; K: number } | null> {
    const tensor = this.#byName.get(name);
    if (!tensor || tensor.kind !== "q8") return null;
    const cached = this.#packedCache.get(name);
    if (cached) return this.#touch(this.#packedCache, name, cached);
    const [N, K] = tensor.shape as [number, number];
    const words = N * Math.ceil(K / 4);
    const [codes, scale] = await Promise.all([
      cachedRange(this.#source, `${this.#base}/dit.q8.bin`, "dit.q8.bin", tensor.codesOffset * 4, words * 4),
      cachedRange(this.#source, `${this.#base}/dit.q8scales.bin`, "dit.q8scales.bin", tensor.scaleOffset * 4, N * 4),
    ]);
    this.bytesHeld += codes.byteLength + scale.byteLength;
    return this.#remember(this.#packedCache, name, {
      codes: new Uint32Array(codes),
      scale: new Float32Array(scale),
      N,
      K,
    });
  }

  async #fetchDense(name: string): Promise<Float32Array> {
    const cached = this.#cache.get(name);
    if (cached) return this.#touch(this.#cache, name, cached);
    const tensor = this.#byName.get(name);
    if (!tensor) throw new Error(`fetch-weights: no tensor named ${JSON.stringify(name)} in the manifest.`);

    const [N, K] = tensor.shape.length === 2 ? tensor.shape : [1, tensor.shape.reduce((a, b) => a * b, 1)];
    let out: Float32Array;
    if (tensor.kind === "q4") {
      const words = N! * Math.ceil(K! / 8);
      const groups = N! * Math.ceil(K! / GROUP_SIZE);
      const [codes, scales] = await Promise.all([
        cachedRange(this.#source, `${this.#base}/dit.codes.bin`, "dit.codes.bin", tensor.codesOffset * 4, words * 4),
        cachedRange(this.#source, `${this.#base}/dit.scales.bin`, "dit.scales.bin", tensor.scaleOffset * 4, groups * 4),
      ]);
      out = dequantizeQ4G128({
        packed: new Uint32Array(codes),
        scales: new Float32Array(scales),
        N: N!,
        K: K!,
      });
    } else if (tensor.kind === "q8") {
      const words = N! * Math.ceil(K! / 4);
      const [codes, scale] = await Promise.all([
        cachedRange(this.#source, `${this.#base}/dit.q8.bin`, "dit.q8.bin", tensor.codesOffset * 4, words * 4),
        cachedRange(this.#source, `${this.#base}/dit.q8scales.bin`, "dit.q8scales.bin", tensor.scaleOffset * 4, N! * 4),
      ]);
      out = dequantizeQ8({ packed: new Uint32Array(codes), scale: new Float32Array(scale), N: N!, K: K! });
    } else {
      const buffer = await cachedRange(this.#source, `${this.#base}/dit.f32.bin`, "dit.f32.bin", tensor.offset * 4, N! * K! * 4);
      out = new Float32Array(buffer);
    }
    this.bytesHeld += out.byteLength;
    return this.#remember(this.#cache, name, out);
  }
}

interface SafetensorsEntry {
  dtype: string;
  shape: number[];
  data_offsets: [number, number];
}

/** A `.safetensors` shard, read over HTTP. The header is fetched once. */
export class FetchedSafetensors {
  readonly #url: string;
  readonly #file: string;
  readonly #dataStart: number;
  readonly #entries: Map<string, SafetensorsEntry>;
  readonly #source: ByteSource | null;

  private constructor(
    url: string,
    file: string,
    dataStart: number,
    entries: Map<string, SafetensorsEntry>,
    source: ByteSource | null,
  ) {
    this.#file = file;
    this.#source = source;
    this.#url = url;
    this.#dataStart = dataStart;
    this.#entries = entries;
  }

  /** `file` is the name within `source`; `url` is used only when there is none. */
  static async open(
    url: string,
    source: ByteSource | null = null,
    file = url.split("/").pop()!,
  ): Promise<FetchedSafetensors> {
    const lengthBytes = await cachedRange(source, url, file, 0, 8);
    const headerLength = Number(new DataView(lengthBytes).getBigUint64(0, true));
    const headerBytes = await cachedRange(source, url, file, 8, headerLength);
    const header = JSON.parse(new TextDecoder().decode(headerBytes)) as Record<string, SafetensorsEntry>;
    const entries = new Map<string, SafetensorsEntry>();
    for (const [name, value] of Object.entries(header)) {
      if (name === "__metadata__") continue;
      entries.set(name, value);
    }
    return new FetchedSafetensors(url, file, 8 + headerLength, entries, source);
  }

  has(name: string): boolean {
    return this.#entries.has(name);
  }

  async read(name: string): Promise<Float32Array> {
    const entry = this.#entries.get(name);
    if (!entry) throw new Error(`fetch-weights: ${this.#url} has no tensor ${JSON.stringify(name)}.`);
    const [from, to] = entry.data_offsets;
    const elements = entry.shape.reduce((a, b) => a * b, 1);
    const perElement = { F32: 4, BF16: 2 }[entry.dtype];
    if (perElement === undefined) {
      throw new Error(`fetch-weights: ${name} has dtype ${entry.dtype}, and only F32 and BF16 are implemented.`);
    }
    const buffer = await readRange(this.#url, this.#dataStart + from, to - from);
    if (entry.dtype === "F32") return new Float32Array(buffer, 0, elements);
    return bf16ToF32(new Uint16Array(buffer, 0, elements));
  }

  /**
   * A few rows of a table, without reading the table.
   *
   * The embedding is 151936 x 2560 — 778 MB of bf16 — and a prompt reads a
   * couple of dozen rows of it. One `Range` request per row is what keeps that
   * from being the whole demo's download.
   */
  async readRows(name: string, rows: Int32Array, rowLength: number): Promise<Float32Array> {
    const entry = this.#entries.get(name);
    if (!entry) throw new Error(`fetch-weights: ${this.#url} has no tensor ${JSON.stringify(name)}.`);
    const perElement = { F32: 4, BF16: 2 }[entry.dtype];
    if (perElement === undefined) {
      throw new Error(`fetch-weights: ${name} has dtype ${entry.dtype}.`);
    }
    const out = new Float32Array(rows.length * rowLength);
    for (let i = 0; i < rows.length; i += 1) {
      const at = this.#dataStart + entry.data_offsets[0] + rows[i]! * rowLength * perElement;
      const buffer = await cachedRange(this.#source, this.#url, this.#file, at, rowLength * perElement);
      const row =
        entry.dtype === "F32" ? new Float32Array(buffer) : bf16ToF32(new Uint16Array(buffer));
      out.set(row, i * rowLength);
    }
    return out;
  }
}

/**
 * A sharded checkpoint over HTTP, addressed as if it were one file — and cached.
 *
 * Nothing is held in memory here. An earlier version cached every tensor it
 * read, which for 35 layers of bf16 expanded to f32 is **14 GB** — and that,
 * not the activations, is what left no room for a 42 MB readback and threw
 * `Array buffer allocation failed` from inside an elementwise dispatch.
 *
 * It does not need to be: `cachedRange` already puts the bytes in the browser's
 * disk cache, so a second read costs a disk hit and a bf16 expansion rather
 * than a download. The heap holds one layer at a time, which is the only amount
 * that has to fit.
 */
export class FetchedShards {
  readonly #files = new Map<string, FetchedSafetensors>();

  constructor(
    readonly base: string,
    readonly weightMap: Record<string, string>,
    /** Passed to each shard, so a bound folder is read rather than the network. */
    readonly source: ByteSource | null = null,
  ) {}

  async #file(shard: string): Promise<FetchedSafetensors> {
    let file = this.#files.get(shard);
    if (!file) {
      file = await FetchedSafetensors.open(`${this.base}/${shard}`, this.source, shard);
      this.#files.set(shard, file);
    }
    return file;
  }

  async read(name: string): Promise<Float32Array> {
    const shard = this.weightMap[name];
    if (!shard) throw new Error(`fetch-weights: no shard listed for ${JSON.stringify(name)}.`);
    return (await this.#file(shard)).read(name);
  }

  async readRows(name: string, rows: Int32Array, rowLength: number): Promise<Float32Array> {
    const shard = this.weightMap[name];
    if (!shard) throw new Error(`fetch-weights: no shard listed for ${JSON.stringify(name)}.`);
    return (await this.#file(shard)).readRows(name, rows, rowLength);
  }
}
