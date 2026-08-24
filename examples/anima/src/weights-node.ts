/**
 * Reads a subset of the converted Anima DiT off disk, in Node.
 *
 * Its own loader rather than `examples/zimage`'s, for the same reason the
 * converter is its own file: that one carries a q4 format with a group size
 * this checkpoint has no concept of, and reusing it means a `groupSize`
 * check failing on a manifest that was never going to have one. The **format**
 * is shared — `ops/matvec`'s q8, four codes per `u32`, one f32 scale per row —
 * and `dequantizeQ8` is imported rather than rewritten.
 *
 * The unit is a *selection*: give it a predicate over tensor names and it reads
 * only those byte ranges. 3.76 GB is not something to hold to run one block,
 * and the browser will want the same shape a layer at a time.
 */
import { closeSync, openSync, readFileSync, readSync } from "node:fs";
import { join } from "node:path";
import { dequantizeQ8 } from "../../zimage/src/weights.js";

export interface AnimaTensor {
  name: string;
  kind: "q8" | "f32";
  shape: number[];
  /** q8 only. */
  codesOffset?: number;
  /** q8 only. */
  scaleOffset?: number;
  /** f32 only. */
  offset?: number;
}

export interface AnimaManifest {
  format: { quant: string };
  blocks: number;
  tensors: AnimaTensor[];
}

/** Reads `count` elements at `offset`, in elements not bytes. */
function readRange(path: string, offset: number, count: number): ArrayBuffer {
  const buffer = new Uint8Array(count * 4);
  const fd = openSync(path, "r");
  try {
    let read = 0;
    while (read < buffer.byteLength) {
      // A short read that goes unnoticed is a tensor of zeros in the tail —
      // numbers, not an error, which is the failure worth spending a loop on.
      const n = readSync(fd, buffer, read, buffer.byteLength - read, offset * 4 + read);
      if (n === 0) throw new Error(`${path}: wanted ${buffer.byteLength} bytes at ${offset * 4}, got ${read}`);
      read += n;
    }
  } finally {
    closeSync(fd);
  }
  return buffer.buffer;
}

/** The converted DiT, addressed by the checkpoint's own tensor names. */
export class AnimaWeights {
  readonly blocks: number;
  readonly #dir: string;
  readonly #byName = new Map<string, AnimaTensor>();
  readonly #cache = new Map<string, Float32Array>();
  readonly #limit: number;

  constructor(dir: string, manifest: AnimaManifest, maxCached = 64) {
    this.#dir = dir;
    this.#limit = maxCached;
    this.blocks = manifest.blocks;
    if (manifest.format.quant !== "q8-per-row") {
      throw new Error(
        `AnimaWeights: manifest declares quant "${manifest.format.quant}", and this loader reads q8-per-row.`,
      );
    }
    for (const tensor of manifest.tensors) this.#byName.set(tensor.name, tensor);
  }

  /** A tensor's shape without reading it — the manifest already carries it. */
  shapeOf(name: string): number[] | undefined {
    return this.#byName.get(name)?.shape;
  }

  has(name: string): boolean {
    return this.#byName.has(name);
  }

  /** The packed form, for a GPU path that dispatches `matmulQ8` directly. */
  packedQ8(name: string): { codes: Uint32Array; scale: Float32Array; N: number; K: number } | null {
    const tensor = this.#byName.get(name);
    if (!tensor || tensor.kind !== "q8") return null;
    const [N, K] = tensor.shape as [number, number];
    return {
      codes: new Uint32Array(readRange(join(this.#dir, "dit.q8.bin"), tensor.codesOffset!, N * Math.ceil(K / 4))),
      scale: new Float32Array(readRange(join(this.#dir, "dit.q8scales.bin"), tensor.scaleOffset!, N)),
      N,
      K,
    };
  }

  /** One tensor as dense f32. Dequantised once, then cached. */
  get(name: string): Float32Array {
    const cached = this.#cache.get(name);
    if (cached) {
      // Delete first, so this counts as a use — a cache whose reads do not
      // refresh position evicts the entries it was asked for. That bug already
      // happened once in `examples/zimage`.
      this.#cache.delete(name);
      this.#cache.set(name, cached);
      return cached;
    }

    const tensor = this.#byName.get(name);
    if (!tensor) {
      // Named, because an undefined here becomes NaN several ops downstream,
      // where the name that was mistyped is no longer in scope.
      throw new Error(`AnimaWeights: no tensor named ${JSON.stringify(name)} in the manifest.`);
    }

    const [N, K] = tensor.shape.length === 2 ? tensor.shape : [1, tensor.shape.reduce((a, b) => a * b, 1)];
    let out: Float32Array;
    if (tensor.kind === "q8") {
      out = dequantizeQ8({
        packed: new Uint32Array(readRange(join(this.#dir, "dit.q8.bin"), tensor.codesOffset!, N! * Math.ceil(K! / 4))),
        scale: new Float32Array(readRange(join(this.#dir, "dit.q8scales.bin"), tensor.scaleOffset!, N!)),
        N: N!,
        K: K!,
      });
    } else {
      out = new Float32Array(readRange(join(this.#dir, "dit.f32.bin"), tensor.offset!, N! * K!));
    }

    this.#cache.set(name, out);
    while (this.#cache.size > this.#limit) {
      this.#cache.delete(this.#cache.keys().next().value as string);
    }
    return out;
  }
}

/** Opens a `convert_dit.py` output directory. */
export function loadAnimaSubset(dir: string, _wanted?: (name: string) => boolean, maxCached = 64): AnimaWeights {
  const manifest = JSON.parse(readFileSync(join(dir, "dit.manifest.json"), "utf8")) as AnimaManifest;
  return new AnimaWeights(dir, manifest, maxCached);
}

/** Every tensor of one block — `(name) => name.startsWith(blockPrefix(0))`. */
export function blockPrefix(index: number): string {
  return `net.blocks.${index}.`;
}
