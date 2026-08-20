import { describe, expect, it } from "vitest";
import { KVCache } from "./kv-cache.js";

describe("KVCache", () => {
  it("round-trips a single write covering the whole length", () => {
    const cache = new KVCache(1, 2, 3, 8);
    const kvHeads = 2;
    const count = 4;
    const headDim = 3;
    // [kvHeads, count, headDim], values encode (head, position, channel).
    const k = Float32Array.from({ length: kvHeads * count * headDim }, (_, i) => {
      const h = Math.floor(i / (count * headDim));
      const rest = i % (count * headDim);
      const p = Math.floor(rest / headDim);
      const c = rest % headDim;
      return h * 100 + p * 10 + c;
    });
    const v = k.map((x) => x + 1000);
    cache.write(0, 0, k, v, count);
    const read = cache.read(0, count);
    expect(Array.from(read.k)).toEqual(Array.from(k));
    expect(Array.from(read.v)).toEqual(Array.from(v));
  });

  it("appends a second write after the first without disturbing it (decode continuation)", () => {
    const cache = new KVCache(1, 2, 2, 8);
    const prefill = Float32Array.from({ length: 2 * 3 * 2 }, (_, i) => i + 1); // [kvHeads=2, count=3, dim=2]
    cache.write(0, 0, prefill, prefill.map((x) => -x), 3);
    const decodeStep = Float32Array.from({ length: 2 * 1 * 2 }, (_, i) => 900 + i); // [kvHeads=2, count=1, dim=2]
    cache.write(0, 3, decodeStep, decodeStep.map((x) => -x), 1);

    const read = cache.read(0, 4);
    // Head 0: positions 0,1,2 from prefill, position 3 from the decode step.
    expect(Array.from(read.k.subarray(0, 8))).toEqual([1, 2, 3, 4, 5, 6, 900, 901]);
    // Head 1 starts at index 4*2=8 in the compact read.
    expect(Array.from(read.k.subarray(8, 16))).toEqual([7, 8, 9, 10, 11, 12, 902, 903]);
  });

  it("keeps layers independent", () => {
    const cache = new KVCache(2, 1, 1, 4);
    cache.write(0, 0, Float32Array.of(1), Float32Array.of(-1), 1);
    cache.write(1, 0, Float32Array.of(2), Float32Array.of(-2), 1);
    expect(Array.from(cache.read(0, 1).k)).toEqual([1]);
    expect(Array.from(cache.read(1, 1).k)).toEqual([2]);
  });

  it("rejects a write that would overflow maxSeqLen", () => {
    const cache = new KVCache(1, 1, 1, 2);
    expect(() => cache.write(0, 1, Float32Array.of(1), Float32Array.of(1), 2)).toThrow(/maxSeqLen=2/);
  });
});
