/**
 * Reads a subset of the converted DiT off disk, in Node.
 *
 * Kept apart from `weights.ts` because that file has to work in a browser and
 * this one opens file descriptors. The split is also what makes the 3.3 GB
 * usable at all: `Float32Array` and `Buffer` both stop well short of it on
 * some builds, and even where they do not, reading three gigabytes to run one
 * layer is a slow way to answer a small question.
 *
 * So the unit here is a *selection*. Give it a predicate over tensor names, and
 * it reads only those byte ranges, rebasing the offsets so the result is an
 * ordinary `DitWeights` that knows nothing about files. The browser will want
 * the same shape later — a layer at a time, as it streams — which is the other
 * reason this is a selection rather than a `readFileSync`.
 */
import { closeSync, openSync, readSync, readFileSync } from "node:fs";
import { join } from "node:path";
import {
  DitWeights,
  GROUP_SIZE,
  dequantizeQ4G128,
  dequantizeQ8,
  type DitManifest,
  type DitTensor,
} from "./weights.js";

/** Reads `count` elements starting at `offset`, in elements not bytes. */
function readRange(path: string, offset: number, count: number, bytesPer: number): ArrayBuffer {
  const buffer = new Uint8Array(count * bytesPer);
  const fd = openSync(path, "r");
  try {
    let read = 0;
    while (read < buffer.byteLength) {
      // `readSync` is allowed to return short, and a partial read that goes
      // unnoticed is a tensor of zeros in the tail — the exact failure the
      // truncation check in `DitWeights` exists to catch, arriving by a route
      // that check cannot see.
      const n = readSync(fd, buffer, read, buffer.byteLength - read, offset * bytesPer + read);
      if (n === 0) throw new Error(`${path}: wanted ${buffer.byteLength} bytes at ${offset * bytesPer}, got ${read}`);
      read += n;
    }
  } finally {
    closeSync(fd);
  }
  return buffer.buffer;
}

/**
 * Loads the tensors whose names satisfy `wanted` from a `convert_dit.py` output
 * directory.
 *
 * Each selected tensor is read individually rather than as one span: the
 * manifest's order is the checkpoint's, so a layer's tensors are usually
 * adjacent but nothing guarantees it, and a "read from the first to the last"
 * shortcut would quietly pull in gigabytes the moment they are not.
 */
export function loadDitSubset(dir: string, wanted: (name: string) => boolean): DitWeights {
  const manifest = JSON.parse(readFileSync(join(dir, "dit.manifest.json"), "utf8")) as DitManifest;
  const selected = manifest.tensors.filter((t) => wanted(t.name));
  if (selected.length === 0) {
    throw new Error(`loadDitSubset: no tensor in ${dir}/dit.manifest.json matched the selection.`);
  }

  const codes: Uint32Array[] = [];
  const scales: Float32Array[] = [];
  const q8: Uint32Array[] = [];
  const q8Scales: Float32Array[] = [];
  const f32: Float32Array[] = [];
  let codesAt = 0;
  let scalesAt = 0;
  let q8At = 0;
  let q8ScalesAt = 0;
  let f32At = 0;
  const rebased: DitTensor[] = [];

  for (const tensor of selected) {
    const [N, K] = tensor.shape.length === 2 ? tensor.shape : [1, tensor.shape.reduce((a, b) => a * b, 1)];
    if (tensor.kind === "q4") {
      const words = N! * Math.ceil(K! / 8);
      const groups = N! * Math.ceil(K! / GROUP_SIZE);
      codes.push(new Uint32Array(readRange(join(dir, "dit.codes.bin"), tensor.codesOffset, words, 4)));
      scales.push(new Float32Array(readRange(join(dir, "dit.scales.bin"), tensor.scaleOffset, groups, 4)));
      rebased.push({ ...tensor, codesOffset: codesAt, scaleOffset: scalesAt });
      codesAt += words;
      scalesAt += groups;
    } else if (tensor.kind === "q8") {
      const words = N! * Math.ceil(K! / 4);
      q8.push(new Uint32Array(readRange(join(dir, "dit.q8.bin"), tensor.codesOffset, words, 4)));
      q8Scales.push(new Float32Array(readRange(join(dir, "dit.q8scales.bin"), tensor.scaleOffset, N!, 4)));
      rebased.push({ ...tensor, codesOffset: q8At, scaleOffset: q8ScalesAt });
      q8At += words;
      q8ScalesAt += N!;
    } else {
      const length = N! * K!;
      f32.push(new Float32Array(readRange(join(dir, "dit.f32.bin"), tensor.offset, length, 4)));
      rebased.push({ ...tensor, offset: f32At });
      f32At += length;
    }
  }

  const concat = <T extends Uint32Array | Float32Array>(parts: T[], make: (n: number) => T): T => {
    const total = parts.reduce((sum, p) => sum + p.length, 0);
    const out = make(total);
    let at = 0;
    for (const part of parts) {
      out.set(part, at);
      at += part.length;
    }
    return out;
  };

  return new DitWeights(
    { ...manifest, tensors: rebased },
    {
      codes: concat(codes, (n) => new Uint32Array(n)),
      scales: concat(scales, (n) => new Float32Array(n)),
      q8: concat(q8, (n) => new Uint32Array(n)),
      q8Scales: concat(q8Scales, (n) => new Float32Array(n)),
      f32: concat(f32, (n) => new Float32Array(n)),
    },
  );
}

/** The tensors of one transformer layer — `loadDitSubset(dir, layerSelector(0))`. */
export function layerSelector(layer: number): (name: string) => boolean {
  const prefix = `layers.${layer}.`;
  return (name) => name.startsWith(prefix);
}

/** All `get` needs to be, so a caller can be handed either loader. */
export interface WeightSource {
  get(name: string): Float32Array;
}

/**
 * The whole DiT, read one tensor at a time and forgotten again.
 *
 * `loadDitSubset(dir, () => true)` would answer the same questions and cannot
 * be used for it: 3.34 GB of q4/q8 is **12 GB dense**, and building it tensor
 * by tensor and concatenating doubles that at the seam. Measured at 20.5 GB
 * resident before it got as far as the first layer.
 *
 * So nothing is held. `get` reads that tensor's byte range off disk,
 * dequantises it, and hands it back; a bounded cache keeps the last few, which
 * is all a forward pass needs because each layer touches its own weights once
 * and never returns to them. Peak memory becomes a function of the cache, not
 * of the model.
 *
 * This is also the shape the browser needs — weights arriving per layer as they
 * stream, rather than a 12 GB allocation nobody can make — so it is the loader
 * that survives past this verification.
 */
export class LazyDitWeights implements WeightSource {
  readonly config: Record<string, unknown>;
  readonly #dir: string;
  readonly #byName = new Map<string, DitTensor>();
  readonly #cache = new Map<string, Float32Array>();
  readonly #limit: number;

  constructor(dir: string, maxCachedTensors = 24) {
    this.#dir = dir;
    this.#limit = maxCachedTensors;
    const manifest = JSON.parse(readFileSync(join(dir, "dit.manifest.json"), "utf8")) as DitManifest;
    if (manifest.format.groupSize !== GROUP_SIZE) {
      throw new Error(
        `LazyDitWeights: manifest declares group size ${manifest.format.groupSize}, ` +
          `and this loader implements ${GROUP_SIZE}.`,
      );
    }
    this.config = manifest.config;
    for (const tensor of manifest.tensors) this.#byName.set(tensor.name, tensor);
  }

  /**
   * A q8 tensor as it sits on disk: packed codes and per-row scales.
   *
   * For the GPU path, which dispatches `ops/dequant_transpose` and gets back
   * `matmul`'s `[K, N]` operand in one pass. Handing it dense f32 instead means
   * dequantising on the CPU, transposing on the CPU, and uploading four times
   * the bytes — measured as most of a GPU forward's wall time before this
   * existed.
   *
   * `null` when the tensor is not q8, so a caller can fall back to `get`
   * without catching.
   */
  packedQ8(name: string): { codes: Uint32Array; scale: Float32Array; N: number; K: number } | null {
    const tensor = this.#byName.get(name);
    if (!tensor || tensor.kind !== "q8") return null;
    const [N, K] = tensor.shape as [number, number];
    return {
      codes: new Uint32Array(readRange(join(this.#dir, "dit.q8.bin"), tensor.codesOffset, N * Math.ceil(K / 4), 4)),
      scale: new Float32Array(readRange(join(this.#dir, "dit.q8scales.bin"), tensor.scaleOffset, N, 4)),
      N,
      K,
    };
  }

  get(name: string): Float32Array {
    const cached = this.#cache.get(name);
    if (cached) {
      // Refresh its position so the eviction below is least-recently-used
      // rather than insertion-ordered — a block re-reads nothing, but the
      // embedders are touched at both ends of the forward.
      this.#cache.delete(name);
      this.#cache.set(name, cached);
      return cached;
    }

    const tensor = this.#byName.get(name);
    if (!tensor) throw new Error(`LazyDitWeights: no tensor named ${JSON.stringify(name)} in the manifest.`);

    const [N, K] = tensor.shape.length === 2 ? tensor.shape : [1, tensor.shape.reduce((a, b) => a * b, 1)];
    let out: Float32Array;
    if (tensor.kind === "q4") {
      const words = N! * Math.ceil(K! / 8);
      const groups = N! * Math.ceil(K! / GROUP_SIZE);
      out = dequantizeQ4G128({
        packed: new Uint32Array(readRange(join(this.#dir, "dit.codes.bin"), tensor.codesOffset, words, 4)),
        scales: new Float32Array(readRange(join(this.#dir, "dit.scales.bin"), tensor.scaleOffset, groups, 4)),
        N: N!,
        K: K!,
      });
    } else if (tensor.kind === "q8") {
      const words = N! * Math.ceil(K! / 4);
      out = dequantizeQ8({
        packed: new Uint32Array(readRange(join(this.#dir, "dit.q8.bin"), tensor.codesOffset, words, 4)),
        scale: new Float32Array(readRange(join(this.#dir, "dit.q8scales.bin"), tensor.scaleOffset, N!, 4)),
        N: N!,
        K: K!,
      });
    } else {
      out = new Float32Array(readRange(join(this.#dir, "dit.f32.bin"), tensor.offset, N! * K!, 4));
    }

    this.#cache.set(name, out);
    while (this.#cache.size > this.#limit) {
      const oldest = this.#cache.keys().next().value as string;
      this.#cache.delete(oldest);
    }
    return out;
  }
}
