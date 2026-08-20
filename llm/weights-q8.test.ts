import { describe, expect, it } from "vitest";
import { packQ8 } from "../ops/matvec/index.js";
import { TINY_FIXTURE_CONFIG } from "./config.js";
import {
  assertWeightShapesQ8,
  cloneQuantizedLinear,
  concatScales,
  dequantizePackedQ8,
  gatherDequantRows,
  packInt8Rows,
  type LlamaLayerWeightsQ8,
  type LlamaWeightsQ8,
  type QuantizedLinear,
} from "./weights-q8.js";

describe("packInt8Rows", () => {
  it("packs Int8Array codes byte-for-byte the same as packQ8 given an equal-valued Int32Array", () => {
    // packInt8Rows exists so the engine never has to materialize a 4x-larger
    // Int32Array copy of a resident weight just to call packQ8 (whose own
    // signature takes Int32Array codes) — it hands packQ8 the Int8Array
    // directly. That only works if JS's `& 0xff` on an Int8Array element
    // reads back the same byte packQ8 would produce from the equal-valued
    // Int32Array — asserted here rather than assumed (rule 2).
    const N = 3;
    const K = 7;
    const values = [1, -1, 127, -127, -128, 0, 64, -64, 5, -5, 100, -100, 3, -3, 2, -2, 9, -9, 12, -12, 1];
    expect(values).toHaveLength(N * K);
    const int8 = Int8Array.from(values);
    const int32 = Int32Array.from(values);
    const viaInt8 = packInt8Rows(int8, N, K);
    const viaInt32 = packQ8({ codes: int32, N, K });
    expect(Array.from(viaInt8)).toEqual(Array.from(viaInt32));
  });

  it("packs a K not a multiple of 4 the same way packQ8 does (padding lanes included)", () => {
    const N = 2;
    const K = 5;
    const values = [1, 2, 3, 4, 5, -1, -2, -3, -4, -5];
    const int8 = Int8Array.from(values);
    const int32 = Int32Array.from(values);
    expect(Array.from(packInt8Rows(int8, N, K))).toEqual(Array.from(packQ8({ codes: int32, N, K })));
  });
});

describe("dequantizePackedQ8", () => {
  it("round-trips packInt8Rows: dequantizing the packed weight recovers codes * scale exactly", () => {
    const N = 4;
    const K = 9; // not a multiple of 4, exercises the tail lane
    // Sweeps through -128 explicitly (`((i*37)%255)-127` alone never lands on
    // it — its range is [-127, 127] — and -128 is the one code whose packed
    // byte (0x80) a sign-extension boundary error (>= 128 vs. >= 129) mis-signs
    // while every other byte in this sweep still happens to round-trip).
    const codes = Int8Array.from({ length: N * K }, (_, i) => (i === 0 ? -128 : ((i * 37) % 255) - 127));
    const scale = Float32Array.from({ length: N }, (_, r) => 0.01 * (r + 1) + 0.003);
    const packed = packInt8Rows(codes, N, K);

    const dequant = dequantizePackedQ8(packed, scale, N, K);

    const want = new Float32Array(N * K);
    for (let row = 0; row < N; row += 1) {
      for (let col = 0; col < K; col += 1) {
        want[row * K + col] = codes[row * K + col]! * scale[row]!;
      }
    }
    expect(Array.from(dequant)).toEqual(Array.from(want));
  });

  it("does not read the padding lanes of the last word when K % 4 != 0", () => {
    // K=5: row 0 has one packed word fully live (cols 0-3) plus a second word
    // with only lane 0 live (col 4) and lanes 1-3 padding. Codes at what would
    // be those padding lanes, in a neighbouring row's data, must not leak in.
    const N = 2;
    const K = 5;
    const codes = Int8Array.from([1, 2, 3, 4, 5, 10, 20, 30, 40, 50]);
    const scale = Float32Array.from([1, 1]);
    const packed = packInt8Rows(codes, N, K);
    const dequant = dequantizePackedQ8(packed, scale, N, K);
    expect(Array.from(dequant)).toEqual([1, 2, 3, 4, 5, 10, 20, 30, 40, 50]);
  });
});

describe("concatScales", () => {
  it("concatenates per-row scale arrays in order", () => {
    const a = Float32Array.from([1, 2]);
    const b = Float32Array.from([3]);
    const c = Float32Array.from([4, 5, 6]);
    expect(Array.from(concatScales([a, b, c]))).toEqual([1, 2, 3, 4, 5, 6]);
  });
});

describe("gatherDequantRows", () => {
  it("dequantizes only the requested rows, in request order, duplicates included", () => {
    // [4 rows, D=2]
    const codes = Int8Array.from([1, 2, 10, 20, -1, -2, 5, 6]);
    const scale = Float32Array.from([1, 0.5, 2, 10]);
    const table: QuantizedLinear = { codes, scale };
    const out = gatherDequantRows(table, [2, 0, 2], 2);
    // row 2: [-1,-2]*2 = [-2,-4]; row 0: [1,2]*1 = [1,2]; row 2 again
    expect(Array.from(out)).toEqual([-2, -4, 1, 2, -2, -4]);
  });
});

describe("cloneQuantizedLinear", () => {
  it("copies codes and scale into fresh buffers, independent of the source", () => {
    // Issue #105's real-model memory concern: `LlamaEngineQ8` reads
    // `embedTokens` from a manifest loader whose `codes`/`scale` are views
    // into one big shared buffer covering *every* weight (`weights-q8-io.ts`) —
    // retaining that view (not a copy) would keep the whole ~1.4 GiB buffer
    // resident just for a ~183 MiB embedding table. `cloneQuantizedLinear`
    // is what decouples `embedTokens` from that buffer once, at construction.
    const source: QuantizedLinear = { codes: Int8Array.from([1, -2, 3]), scale: Float32Array.from([0.5, 1.5]) };
    const cloned = cloneQuantizedLinear(source);

    expect(Array.from(cloned.codes)).toEqual([1, -2, 3]);
    expect(Array.from(cloned.scale)).toEqual([0.5, 1.5]);
    expect(cloned.codes.buffer).not.toBe(source.codes.buffer);
    expect(cloned.scale.buffer).not.toBe(source.scale.buffer);

    // Mutating the source after cloning must not be observable through the clone.
    source.codes[0] = 99;
    source.scale[0] = 99;
    expect(cloned.codes[0]).toBe(1);
    expect(cloned.scale[0]).toBe(0.5);
  });
});

describe("assertWeightShapesQ8", () => {
  function tinyQ8Weights(): LlamaWeightsQ8 {
    const { hiddenSize, numHeads, numKvHeads, headDim, ffnHidden, vocabSize, numLayers } = TINY_FIXTURE_CONFIG;
    const linear = (n: number, k: number): QuantizedLinear => ({
      codes: new Int8Array(n * k),
      scale: new Float32Array(n),
    });
    const layer = (): LlamaLayerWeightsQ8 => ({
      attnNorm: new Float32Array(hiddenSize),
      wq: linear(numHeads * headDim, hiddenSize),
      wk: linear(numKvHeads * headDim, hiddenSize),
      wv: linear(numKvHeads * headDim, hiddenSize),
      wo: linear(hiddenSize, numHeads * headDim),
      ffnNorm: new Float32Array(hiddenSize),
      wGate: linear(ffnHidden, hiddenSize),
      wUp: linear(ffnHidden, hiddenSize),
      wDown: linear(hiddenSize, ffnHidden),
    });
    return {
      embedTokens: linear(vocabSize, hiddenSize),
      layers: Array.from({ length: numLayers }, layer),
      finalNorm: new Float32Array(hiddenSize),
      lmHead: linear(vocabSize, hiddenSize),
    };
  }

  it("accepts correctly-shaped weights", () => {
    expect(() => assertWeightShapesQ8(TINY_FIXTURE_CONFIG, tinyQ8Weights())).not.toThrow();
  });

  it("rejects a wrong-length codes array with a message naming the field", () => {
    const weights = tinyQ8Weights();
    weights.layers[0]!.wq.codes = new Int8Array(3);
    expect(() => assertWeightShapesQ8(TINY_FIXTURE_CONFIG, weights)).toThrow(/layers\[0\]\.wq\.codes/);
  });

  it("rejects a wrong-length scale array with a message naming the field", () => {
    const weights = tinyQ8Weights();
    weights.lmHead.scale = new Float32Array(1);
    expect(() => assertWeightShapesQ8(TINY_FIXTURE_CONFIG, weights)).toThrow(/lmHead\.scale/);
  });

  it("rejects the wrong number of layers", () => {
    const weights = tinyQ8Weights();
    weights.layers.pop();
    expect(() => assertWeightShapesQ8(TINY_FIXTURE_CONFIG, weights)).toThrow(/layers/);
  });
});
