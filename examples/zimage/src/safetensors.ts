/**
 * Reads a `.safetensors` file one tensor at a time.
 *
 * Qwen3-4B is 8 GB of bf16, which is 16 GB as f32 and not something to hold to
 * run 35 layers over 30 tokens. The unit here is therefore a tensor, read from
 * its byte range on demand — the same shape `weights-node.ts` settled on for
 * the DiT, and for the same reason.
 *
 * The format is small enough to implement rather than depend on: a `u64`
 * little-endian header length, that many bytes of JSON naming each tensor's
 * dtype, shape and byte range, then the payload. The one thing worth stating
 * is that **`data_offsets` are relative to the end of the header**, not the
 * start of the file — a reader that measures from the start gets the first
 * tensor right and every later one wrong, which is the most convincing way to
 * be broken.
 */
/** Re-exported so callers that already import this module keep working. */
export { bf16ToF32 } from "./bf16.js";
import { closeSync, openSync, readSync, statSync } from "node:fs";
import { bf16ToF32 } from "./bf16.js";

interface Entry {
  dtype: string;
  shape: number[];
  data_offsets: [number, number];
}

export class SafetensorsFile {
  readonly #fd: number;
  readonly #dataStart: number;
  readonly #entries: Map<string, Entry>;
  readonly #size: number;

  constructor(readonly path: string) {
    this.#fd = openSync(path, "r");
    this.#size = statSync(path).size;

    const lengthBytes = new Uint8Array(8);
    readSync(this.#fd, lengthBytes, 0, 8, 0);
    const headerLength = Number(new DataView(lengthBytes.buffer).getBigUint64(0, true));

    const headerBytes = new Uint8Array(headerLength);
    readSync(this.#fd, headerBytes, 0, headerLength, 8);
    const header = JSON.parse(new TextDecoder().decode(headerBytes)) as Record<string, Entry | unknown>;

    this.#dataStart = 8 + headerLength;
    this.#entries = new Map();
    for (const [name, value] of Object.entries(header)) {
      // `__metadata__` is the format's one reserved key and is not a tensor.
      if (name === "__metadata__") continue;
      this.#entries.set(name, value as Entry);
    }
  }

  names(): string[] {
    return [...this.#entries.keys()];
  }

  has(name: string): boolean {
    return this.#entries.has(name);
  }

  shape(name: string): number[] {
    return this.#entry(name).shape;
  }

  #entry(name: string): Entry {
    const entry = this.#entries.get(name);
    if (!entry) {
      throw new Error(`SafetensorsFile: ${this.path} has no tensor named ${JSON.stringify(name)}.`);
    }
    return entry;
  }

  /** One tensor, flattened, as f32. */
  read(name: string): Float32Array {
    const entry = this.#entry(name);
    const [from, to] = entry.data_offsets;
    const byteLength = to - from;
    const elements = entry.shape.reduce((a, b) => a * b, 1);

    const perElement = { F32: 4, BF16: 2 }[entry.dtype];
    if (perElement === undefined) {
      // Refused rather than guessed: F16 and BF16 are the same width and
      // different numbers, so a wrong guess reads cleanly and means nothing.
      throw new Error(
        `SafetensorsFile: ${name} has dtype ${entry.dtype}, and only F32 and BF16 are implemented.`,
      );
    }
    if (byteLength !== elements * perElement) {
      throw new Error(
        `SafetensorsFile: ${name} declares shape [${entry.shape}] (${elements * perElement} bytes) ` +
          `but its range covers ${byteLength} bytes.`,
      );
    }
    if (this.#dataStart + to > this.#size) {
      throw new Error(
        `SafetensorsFile: ${name} ends at ${this.#dataStart + to} in a file of ${this.#size} bytes — truncated.`,
      );
    }

    const buffer = new Uint8Array(byteLength);
    let read = 0;
    while (read < byteLength) {
      const n = readSync(this.#fd, buffer, read, byteLength - read, this.#dataStart + from + read);
      if (n === 0) throw new Error(`SafetensorsFile: short read on ${name} — ${read} of ${byteLength} bytes.`);
      read += n;
    }

    if (entry.dtype === "F32") return new Float32Array(buffer.buffer, 0, elements);
    return bf16ToF32(new Uint16Array(buffer.buffer, 0, elements));
  }

  close(): void {
    closeSync(this.#fd);
  }
}

/**
 * A sharded checkpoint, addressed as if it were one file.
 *
 * `model.safetensors.index.json` maps every tensor to the shard holding it.
 * Shards are opened lazily and kept open, because a 36-layer read touches each
 * of them many times and reopening per tensor is the kind of cost that only
 * shows up as "this is slow" much later.
 */
export class ShardedSafetensors {
  readonly #files = new Map<string, SafetensorsFile>();
  readonly #shardOf: Record<string, string>;

  constructor(
    readonly dir: string,
    weightMap: Record<string, string>,
  ) {
    this.#shardOf = weightMap;
  }

  has(name: string): boolean {
    return name in this.#shardOf;
  }

  read(name: string): Float32Array {
    const shard = this.#shardOf[name];
    if (!shard) throw new Error(`ShardedSafetensors: no shard listed for ${JSON.stringify(name)}.`);
    let file = this.#files.get(shard);
    if (!file) {
      file = new SafetensorsFile(`${this.dir}/${shard}`);
      this.#files.set(shard, file);
    }
    return file.read(name);
  }

  close(): void {
    for (const file of this.#files.values()) file.close();
    this.#files.clear();
  }
}
