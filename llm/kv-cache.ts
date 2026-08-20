/**
 * The KV cache: one pre-allocated f32 buffer pair per layer, big enough for
 * `maxSeqLen` positions of `kvHeads` heads.
 *
 * Issue #98's scope is explicit that this is enough for now — "KVキャッシュは
 * f32事前確保でよい(最適化は後続)" — so this class does exactly one thing:
 * hold storage that does not move between forward calls, so a decode step
 * never re-allocates. It knows nothing about tokens, positions relative to a
 * generation, or which layer is "current" — `LlamaEngine` owns the position
 * counter and calls `write`/`read` with an explicit offset each time, so the
 * cache itself has no state to get out of sync.
 *
 * ## Layout, and why `read` copies
 *
 * Storage is `[kvHeads, maxSeqLen, headDim]` per layer — head-major, matching
 * what `ops/gqa`'s kernels read (`[B, kvHeads, S, D]`, see `llm/reshape.ts`).
 * Within one head, positions are contiguous; across heads they are not — head
 * `h`'s block starts at `h * maxSeqLen * headDim` regardless of how many
 * positions are actually filled. So the first `S` positions of every head
 * cannot be expressed as one `subarray` when `S < maxSeqLen`, which is every
 * step before the cache is full. `read` therefore copies a compact
 * `[kvHeads, S, headDim]` view out — cheap at the sizes a single forward pass
 * touches (a layer's worth of KV, not the model's weights), and the price of
 * `maxSeqLen` pre-allocation not reshuffling on every append.
 */
export class KVCache {
  private readonly layers: { k: Float32Array; v: Float32Array }[];

  constructor(
    public readonly numLayers: number,
    public readonly kvHeads: number,
    public readonly headDim: number,
    public readonly maxSeqLen: number,
  ) {
    this.layers = Array.from({ length: numLayers }, () => ({
      k: new Float32Array(kvHeads * maxSeqLen * headDim),
      v: new Float32Array(kvHeads * maxSeqLen * headDim),
    }));
  }

  /**
   * Writes `count` new positions starting at `at`, from head-major
   * `[kvHeads, count, headDim]` arrays — exactly what `splitHeadsMajor`
   * produces from a freshly projected (and, for K, RoPE'd) `[count, kvHeads, headDim]`
   * tensor.
   */
  write(layer: number, at: number, kHeadMajor: Float32Array, vHeadMajor: Float32Array, count: number): void {
    if (at < 0 || at + count > this.maxSeqLen) {
      throw new Error(
        `KVCache.write: [${at}, ${at + count}) does not fit maxSeqLen=${this.maxSeqLen}`,
      );
    }
    const { k, v } = this.layers[layer]!;
    const { kvHeads, headDim, maxSeqLen } = this;
    for (let h = 0; h < kvHeads; h += 1) {
      const srcOffset = h * count * headDim;
      const dstOffset = (h * maxSeqLen + at) * headDim;
      k.set(kHeadMajor.subarray(srcOffset, srcOffset + count * headDim), dstOffset);
      v.set(vHeadMajor.subarray(srcOffset, srcOffset + count * headDim), dstOffset);
    }
  }

  /** Compact `[kvHeads, length, headDim]` copies covering positions `[0, length)`. */
  read(layer: number, length: number): { k: Float32Array; v: Float32Array } {
    if (length < 0 || length > this.maxSeqLen) {
      throw new Error(`KVCache.read: length=${length} does not fit maxSeqLen=${this.maxSeqLen}`);
    }
    const { k, v } = this.layers[layer]!;
    const { kvHeads, headDim, maxSeqLen } = this;
    const outK = new Float32Array(kvHeads * length * headDim);
    const outV = new Float32Array(kvHeads * length * headDim);
    for (let h = 0; h < kvHeads; h += 1) {
      const srcOffset = h * maxSeqLen * headDim;
      const dstOffset = h * length * headDim;
      outK.set(k.subarray(srcOffset, srcOffset + length * headDim), dstOffset);
      outV.set(v.subarray(srcOffset, srcOffset + length * headDim), dstOffset);
    }
    return { k: outK, v: outV };
  }
}
