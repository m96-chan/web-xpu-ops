/**
 * Reads the DiT that `tools/convert_dit.py` wrote.
 *
 * The checkpoint is 12.31 GB of bf16 and the browser is not going to hold it,
 * so the converter stores Linear weights as q4-g128 (issue #137's format) and
 * everything else as f32. This is the other half of that: given the manifest
 * and the three blobs, hand back a `Float32Array` per tensor.
 *
 * **Dequantising to f32 is deliberate, and temporary.** Issue #166's first
 * stage asks whether real weights run the block that #163 verified, and that
 * block takes dense f32. Feeding the packed codes straight into `matvecQ4G128`
 * would answer a different question at the same time — how much the format
 * costs on a real model, which #137 reopened precisely because nobody has
 * measured it — and answering two questions with one number is how a wrong
 * kernel gets attributed to quantization, or the reverse. So: dense first,
 * matching the model; packed second, measured against dense.
 *
 * Correctness here is `ops/matvec`'s own quantizer, not this file's reasoning
 * about the format — `weights.test.ts` round-trips through `quantizeQ4G128`
 * and `packQ4` rather than against hand-written nibbles.
 */

/** Issue #137's group size. The manifest is checked against it rather than trusted. */
export const GROUP_SIZE = 128;

/** What the converter writes, in the fields this file reads. */
export interface DitManifest {
  format: { quant: string; groupSize: number };
  config: Record<string, unknown>;
  tensors: DitTensor[];
}

export type DitTensor =
  | { name: string; kind: "q4"; shape: number[]; codesOffset: number; scaleOffset: number }
  | { name: string; kind: "q8"; shape: number[]; codesOffset: number; scaleOffset: number }
  | { name: string; kind: "f32"; shape: number[]; offset: number };

export interface DitBlobs {
  /** Packed nibbles, `[N, K/8]` u32 per q4 tensor, concatenated. */
  codes: Uint32Array;
  /** One f32 per group, `[N, K/128]` per q4 tensor, concatenated. */
  scales: Float32Array;
  /** Packed bytes, `[N, ceil(K/4)]` u32 per q8 tensor, concatenated. */
  q8: Uint32Array;
  /** One f32 per row, `[N]` per q8 tensor, concatenated. */
  q8Scales: Float32Array;
  /** Everything not quantized, flattened and concatenated. */
  f32: Float32Array;
}

/**
 * Packed q4 back to dense f32.
 *
 * The nibble is **sign-extended**, not masked: codes are two's complement over
 * four bits, so `-1` is stored as `0xf`. Masking would turn it into `+15`,
 * which is both out of the format's range and the wrong sign — the failure
 * `packQ4`'s doc calls out one width up.
 */
export function dequantizeQ4G128({
  packed,
  scales,
  N,
  K,
}: {
  packed: Uint32Array;
  scales: Float32Array;
  N: number;
  K: number;
}): Float32Array {
  const wordsPerRow = Math.ceil(K / 8);
  const groupsPerRow = Math.ceil(K / GROUP_SIZE);
  const out = new Float32Array(N * K);

  for (let row = 0; row < N; row += 1) {
    for (let col = 0; col < K; col += 1) {
      const word = packed[row * wordsPerRow + (col >> 3)]!;
      const nibble = (word >>> ((col & 7) * 4)) & 0xf;
      const code = nibble >= 8 ? nibble - 16 : nibble;
      out[row * K + col] = code * scales[row * groupsPerRow + Math.floor(col / GROUP_SIZE)]!;
    }
  }
  return out;
}

/**
 * Packed q8 back to dense f32 — `matvecQ8`'s format, one scale per row.
 *
 * The byte is sign-extended for the same reason the nibble is: `0xff` is `-1`.
 * Four codes per word here rather than eight, so the stride differs from
 * `dequantizeQ4G128` and reading one with the other's stride yields a full
 * tensor of wrong numbers and no other symptom.
 */
export function dequantizeQ8({
  packed,
  scale,
  N,
  K,
}: {
  packed: Uint32Array;
  scale: Float32Array;
  N: number;
  K: number;
}): Float32Array {
  const wordsPerRow = Math.ceil(K / 4);
  const out = new Float32Array(N * K);
  for (let row = 0; row < N; row += 1) {
    const rowScale = scale[row]!;
    for (let col = 0; col < K; col += 1) {
      const word = packed[row * wordsPerRow + (col >> 2)]!;
      const byte = (word >>> ((col & 3) * 8)) & 0xff;
      out[row * K + col] = (byte >= 128 ? byte - 256 : byte) * rowScale;
    }
  }
  return out;
}

/** The converted DiT, addressed by the checkpoint's own tensor names. */
export class DitWeights {
  readonly config: Record<string, unknown>;
  readonly #byName = new Map<string, DitTensor>();
  readonly #blobs: DitBlobs;
  readonly #cache = new Map<string, Float32Array>();

  constructor(manifest: DitManifest, blobs: DitBlobs) {
    if (manifest.format.groupSize !== GROUP_SIZE) {
      // Striding a 64-group blob as if it were 128 produces a full tensor of
      // wrong numbers and no other symptom, so this is refused rather than
      // adapted to.
      throw new Error(
        `DitWeights: manifest declares group size ${manifest.format.groupSize}, ` +
          `and this loader implements ${GROUP_SIZE}. Re-run tools/convert_dit.py.`,
      );
    }

    this.config = manifest.config;
    this.#blobs = blobs;
    let codesEnd = 0;
    let scalesEnd = 0;
    let q8End = 0;
    let q8ScalesEnd = 0;
    let f32End = 0;
    for (const tensor of manifest.tensors) {
      this.#byName.set(tensor.name, tensor);
      const [N, K] = tensor.shape.length === 2 ? tensor.shape : [1, tensor.shape.reduce((a, b) => a * b, 1)];
      if (tensor.kind === "q4") {
        codesEnd = Math.max(codesEnd, tensor.codesOffset + N! * Math.ceil(K! / 8));
        scalesEnd = Math.max(scalesEnd, tensor.scaleOffset + N! * Math.ceil(K! / GROUP_SIZE));
      } else if (tensor.kind === "q8") {
        q8End = Math.max(q8End, tensor.codesOffset + N! * Math.ceil(K! / 4));
        q8ScalesEnd = Math.max(q8ScalesEnd, tensor.scaleOffset + N!);
      } else {
        f32End = Math.max(f32End, tensor.offset + N! * K!);
      }
    }

    // A truncated blob is otherwise undetectable: every offset but the last
    // still lands inside it, and the last tensor reads zeros — a model that
    // runs and produces noise.
    const short = (what: string, have: number, need: number) =>
      `DitWeights: ${what} blob is truncated — ${have} elements, manifest needs ${need}.`;
    if (blobs.codes.length < codesEnd) throw new Error(short("codes", blobs.codes.length, codesEnd));
    if (blobs.scales.length < scalesEnd) throw new Error(short("scales", blobs.scales.length, scalesEnd));
    if (blobs.q8.length < q8End) throw new Error(short("q8", blobs.q8.length, q8End));
    if (blobs.q8Scales.length < q8ScalesEnd) throw new Error(short("q8 scales", blobs.q8Scales.length, q8ScalesEnd));
    if (blobs.f32.length < f32End) throw new Error(short("f32", blobs.f32.length, f32End));
  }

  /** One tensor as dense f32, by its checkpoint name. Dequantised once, then cached. */
  get(name: string): Float32Array {
    const cached = this.#cache.get(name);
    if (cached) return cached;

    const tensor = this.#byName.get(name);
    if (!tensor) {
      // Named, because an undefined here becomes NaN several ops downstream,
      // where the name that was mistyped is no longer anywhere in scope.
      throw new Error(`DitWeights: no tensor named ${JSON.stringify(name)} in the manifest.`);
    }

    const [N, K] = tensor.shape.length === 2 ? tensor.shape : [1, tensor.shape.reduce((a, b) => a * b, 1)];
    let out: Float32Array;
    if (tensor.kind === "q4") {
      const words = N! * Math.ceil(K! / 8);
      const groups = N! * Math.ceil(K! / GROUP_SIZE);
      out = dequantizeQ4G128({
        packed: this.#blobs.codes.subarray(tensor.codesOffset, tensor.codesOffset + words),
        scales: this.#blobs.scales.subarray(tensor.scaleOffset, tensor.scaleOffset + groups),
        N: N!,
        K: K!,
      });
    } else if (tensor.kind === "q8") {
      out = dequantizeQ8({
        packed: this.#blobs.q8.subarray(tensor.codesOffset, tensor.codesOffset + N! * Math.ceil(K! / 4)),
        scale: this.#blobs.q8Scales.subarray(tensor.scaleOffset, tensor.scaleOffset + N!),
        N: N!,
        K: K!,
      });
    } else {
      out = this.#blobs.f32.slice(tensor.offset, tensor.offset + N! * K!);
    }
    this.#cache.set(name, out);
    return out;
  }

  /** True if the manifest carries this tensor — for probing an optional one without catching. */
  has(name: string): boolean {
    return this.#byName.has(name);
  }

  /**
   * One block's weights, keyed the way `block.ts` names them.
   *
   * The checkpoint writes `layers.0.attention.to_q.weight`; `BlockWeights`
   * spells the same thing `attention_to_q_weight`. The rule is "drop the
   * `layers.N.` prefix, dots to underscores", and it lives here so that the
   * translation exists once instead of at every call site.
   */
  block<K extends string>(layer: number, names: readonly K[]): Record<string, Float32Array> {
    const out: Record<string, Float32Array> = {};
    for (const name of names) out[name.replace(/\./g, "_")] = this.get(`layers.${layer}.${name}`);
    return out;
  }
}
