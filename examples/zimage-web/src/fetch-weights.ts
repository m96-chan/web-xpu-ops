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
import {
  GROUP_SIZE,
  dequantizeQ4G128,
  dequantizeQ8,
  type DitManifest,
  type DitTensor,
} from "../../zimage/src/weights.js";
import { bf16ToF32 } from "../../zimage/src/bf16.js";

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
 * The converted DiT, read over HTTP and then **kept**.
 *
 * The first version bounded the cache at a couple of dozen tensors, mirroring
 * the Node loader — which is right when the backing store is a file and the
 * page cache is doing the remembering, and badly wrong over HTTP. A forward
 * touches every tensor once, so a bound smaller than the model means every
 * layer is re-fetched on every step: 6.17 GB per step, 49 GB for an
 * eight-step generation, all of it the same bytes.
 *
 * So the default is to hold everything. That is 6.17 GB of packed codes in
 * JavaScript memory — real, and worth stating — but it makes every step after
 * the first fetch cost **nothing** on the wire. `maxCachedTensors` is still
 * there for a machine that cannot spare the memory, where the trade goes the
 * other way.
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

  private constructor(base: string, manifest: DitManifest, limit: number) {
    this.#base = base;
    this.#limit = limit;
    this.config = manifest.config;
    for (const tensor of manifest.tensors) this.#byName.set(tensor.name, tensor);
  }

  static async open(base: string, maxCachedTensors = Infinity): Promise<FetchedDitWeights> {
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
    return new FetchedDitWeights(base, manifest, maxCachedTensors);
  }

  /** Total bytes held, for the progress read-out — the number the user is waiting on. */
  bytesHeld = 0;

  /** Every tensor in the manifest, for `preloadAll`'s progress. */
  get tensorCount(): number {
    return this.#byName.size;
  }

  #remember<T>(map: Map<string, T>, name: string, value: T): T {
    map.set(name, value);
    while (map.size > this.#limit) {
      const oldest = map.keys().next().value as string;
      map.delete(oldest);
    }
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
    await Promise.all(wanted.map((name) => (this.#byName.get(name)!.kind === "q8" ? this.#fetchPacked(name) : this.#fetchDense(name))));
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
    if (cached) return cached;
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
    if (cached) return cached;
    const [N, K] = tensor.shape as [number, number];
    const words = N * Math.ceil(K / 4);
    const [codes, scale] = await Promise.all([
      readRange(`${this.#base}/dit.q8.bin`, tensor.codesOffset * 4, words * 4),
      readRange(`${this.#base}/dit.q8scales.bin`, tensor.scaleOffset * 4, N * 4),
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
    if (cached) return cached;
    const tensor = this.#byName.get(name);
    if (!tensor) throw new Error(`fetch-weights: no tensor named ${JSON.stringify(name)} in the manifest.`);

    const [N, K] = tensor.shape.length === 2 ? tensor.shape : [1, tensor.shape.reduce((a, b) => a * b, 1)];
    let out: Float32Array;
    if (tensor.kind === "q4") {
      const words = N! * Math.ceil(K! / 8);
      const groups = N! * Math.ceil(K! / GROUP_SIZE);
      const [codes, scales] = await Promise.all([
        readRange(`${this.#base}/dit.codes.bin`, tensor.codesOffset * 4, words * 4),
        readRange(`${this.#base}/dit.scales.bin`, tensor.scaleOffset * 4, groups * 4),
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
        readRange(`${this.#base}/dit.q8.bin`, tensor.codesOffset * 4, words * 4),
        readRange(`${this.#base}/dit.q8scales.bin`, tensor.scaleOffset * 4, N! * 4),
      ]);
      out = dequantizeQ8({ packed: new Uint32Array(codes), scale: new Float32Array(scale), N: N!, K: K! });
    } else {
      const buffer = await readRange(`${this.#base}/dit.f32.bin`, tensor.offset * 4, N! * K! * 4);
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
  readonly #dataStart: number;
  readonly #entries: Map<string, SafetensorsEntry>;

  private constructor(url: string, dataStart: number, entries: Map<string, SafetensorsEntry>) {
    this.#url = url;
    this.#dataStart = dataStart;
    this.#entries = entries;
  }

  static async open(url: string): Promise<FetchedSafetensors> {
    const lengthBytes = await readRange(url, 0, 8);
    const headerLength = Number(new DataView(lengthBytes).getBigUint64(0, true));
    const headerBytes = await readRange(url, 8, headerLength);
    const header = JSON.parse(new TextDecoder().decode(headerBytes)) as Record<string, SafetensorsEntry>;
    const entries = new Map<string, SafetensorsEntry>();
    for (const [name, value] of Object.entries(header)) {
      if (name === "__metadata__") continue;
      entries.set(name, value);
    }
    return new FetchedSafetensors(url, 8 + headerLength, entries);
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
      const buffer = await readRange(this.#url, at, rowLength * perElement);
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
 * The encoder runs once per generation rather than once per step, so it is not
 * the 49 GB the DiT was; it is still 7 GB re-read on every prompt, which is 7 GB
 * more than the second prompt needs. Held for the same reason and with the same
 * trade.
 */
export class FetchedShards {
  readonly #files = new Map<string, FetchedSafetensors>();
  readonly #cache = new Map<string, Float32Array>();

  constructor(
    readonly base: string,
    readonly weightMap: Record<string, string>,
  ) {}

  async #file(shard: string): Promise<FetchedSafetensors> {
    let file = this.#files.get(shard);
    if (!file) {
      file = await FetchedSafetensors.open(`${this.base}/${shard}`);
      this.#files.set(shard, file);
    }
    return file;
  }

  async read(name: string): Promise<Float32Array> {
    const cached = this.#cache.get(name);
    if (cached) return cached;
    const shard = this.weightMap[name];
    if (!shard) throw new Error(`fetch-weights: no shard listed for ${JSON.stringify(name)}.`);
    const out = await (await this.#file(shard)).read(name);
    this.#cache.set(name, out);
    return out;
  }

  async readRows(name: string, rows: Int32Array, rowLength: number): Promise<Float32Array> {
    const shard = this.weightMap[name];
    if (!shard) throw new Error(`fetch-weights: no shard listed for ${JSON.stringify(name)}.`);
    return (await this.#file(shard)).readRows(name, rows, rowLength);
  }
}
