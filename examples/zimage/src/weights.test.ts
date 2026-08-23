import { describe, expect, it } from "vitest";
import { GROUP_SIZE, dequantizeQ4G128, dequantizeQ8, DitWeights, type DitManifest } from "./weights.js";
import { packQ4, packQ8, quantizeQ4G128 } from "../../../ops/matvec/reference.js";

/**
 * The loader, against `ops/matvec`'s own quantizer.
 *
 * The converter (`tools/convert_dit.py`) is Python and cannot run here, so the
 * pairing tested is the one that has to hold for the loader to mean anything:
 * whatever `quantizeQ4G128`/`packQ4` produce, `dequantizeQ4G128` reads back.
 * That the *Python* side produces the same bytes is checked separately, by
 * quantizing one real tensor in both languages and comparing them word for
 * word — it needs the 12 GB checkpoint, so it is not part of this suite.
 */

function synth(N: number, K: number, fill: (row: number, col: number) => number): Float32Array {
  const out = new Float32Array(N * K);
  for (let row = 0; row < N; row += 1) {
    for (let col = 0; col < K; col += 1) out[row * K + col] = fill(row, col);
  }
  return out;
}

describe("dequantizeQ4G128", () => {
  it("recovers a group's largest magnitude exactly", () => {
    // Absmax quantization is exact at the extreme by construction: the value
    // that set the scale maps to code 7 and back to itself. Anything else has
    // rounding error, so this is the one entry where an exact assertion is
    // available, and it pins scale and sign together.
    const N = 2;
    const K = GROUP_SIZE;
    const input = synth(N, K, (row, col) => (col === 5 ? (row === 0 ? 3.5 : -3.5) : 0.01));
    const { codes, scales } = quantizeQ4G128({ input, N, K });
    const out = dequantizeQ4G128({ packed: packQ4({ codes, N, K }), scales, N, K });
    expect(out[5]).toBeCloseTo(3.5, 6);
    expect(out[K + 5]).toBeCloseTo(-3.5, 6);
  });

  it("sign-extends negative nibbles", () => {
    // A nibble masked instead of sign-extended turns -1 into +15, i.e. every
    // negative weight becomes a large positive one. The mean would still look
    // finite, so the check is on the sign of each entry, not on a norm.
    const N = 1;
    const K = GROUP_SIZE;
    const input = synth(N, K, (_row, col) => -(col + 1) / K);
    const { codes, scales } = quantizeQ4G128({ input, N, K });
    const out = dequantizeQ4G128({ packed: packQ4({ codes, N, K }), scales, N, K });
    expect([...out].every((v) => v <= 0)).toBe(true);
    expect(out[K - 1]).toBeCloseTo(-1, 2);
  });

  it("keeps each group on its own scale", () => {
    // Two groups differing by 1000x in magnitude. A loader that applied one
    // scale to the whole row would still return plausible numbers for the
    // large group and destroy the small one — which is precisely what group
    // scaling exists to prevent.
    const N = 1;
    const K = GROUP_SIZE * 2;
    const input = synth(N, K, (_row, col) => (col < GROUP_SIZE ? 1000 : 0.001));
    const { codes, scales } = quantizeQ4G128({ input, N, K });
    const out = dequantizeQ4G128({ packed: packQ4({ codes, N, K }), scales, N, K });
    expect(out[0]).toBeCloseTo(1000, 1);
    expect(out[GROUP_SIZE]).toBeCloseTo(0.001, 9);
  });

  it("returns zeros for an all-zero group rather than NaN", () => {
    // The case the Python converter and this file initially disagreed on: an
    // all-zero group has no absmax to divide by. Both store scale 1; a
    // converter storing 0 would still give zeros here, but any consumer
    // dividing by the scale would get an infinity out of a weight that is
    // simply absent.
    const N = 1;
    const K = GROUP_SIZE;
    const input = new Float32Array(K);
    const { codes, scales } = quantizeQ4G128({ input, N, K });
    expect(scales[0]).toBe(1);
    const out = dequantizeQ4G128({ packed: packQ4({ codes, N, K }), scales, N, K });
    expect([...out].every((v) => v === 0)).toBe(true);
  });

  it("stays within the error the format allows", () => {
    // 4 bits symmetric over [-absmax, absmax] puts the step at absmax/7, so no
    // entry can be off by more than half of that. Asserting the bound rather
    // than a measured average means a loader that is subtly wrong in one group
    // fails, instead of being averaged away across 128 of them.
    const N = 4;
    const K = GROUP_SIZE * 3;
    const input = synth(N, K, (row, col) => Math.sin(row * 7 + col * 0.13) * (1 + row));
    const { codes, scales } = quantizeQ4G128({ input, N, K });
    const out = dequantizeQ4G128({ packed: packQ4({ codes, N, K }), scales, N, K });
    const groupsPerRow = K / GROUP_SIZE;
    for (let row = 0; row < N; row += 1) {
      for (let col = 0; col < K; col += 1) {
        const scale = scales[row * groupsPerRow + Math.floor(col / GROUP_SIZE)]!;
        expect(Math.abs(out[row * K + col]! - input[row * K + col]!)).toBeLessThanOrEqual(scale / 2 + 1e-6);
      }
    }
  });
});

/** `quantize`'s convention, per row: absmax/127. Kept here because the q8 path
 *  in `ops/matvec` exposes the packer and the kernel but not a quantizer. */
function quantizeQ8(input: Float32Array, N: number, K: number): { codes: Int32Array; scale: Float32Array } {
  const codes = new Int32Array(N * K);
  const scale = new Float32Array(N);
  for (let row = 0; row < N; row += 1) {
    let absmax = 0;
    for (let col = 0; col < K; col += 1) absmax = Math.max(absmax, Math.abs(input[row * K + col]!));
    scale[row] = absmax === 0 ? 1 : absmax / 127;
    const inverse = absmax === 0 ? 0 : 127 / absmax;
    for (let col = 0; col < K; col += 1) {
      codes[row * K + col] = Math.max(-127, Math.min(127, Math.round(input[row * K + col]! * inverse)));
    }
  }
  return { codes, scale };
}

describe("dequantizeQ8", () => {
  it("recovers a row's largest magnitude exactly", () => {
    const N = 2;
    const K = 12;
    const input = synth(N, K, (row, col) => (col === 3 ? (row === 0 ? 2.25 : -2.25) : 0.02));
    const { codes, scale } = quantizeQ8(input, N, K);
    const out = dequantizeQ8({ packed: packQ8({ codes, N, K }), scale, N, K });
    expect(out[3]).toBeCloseTo(2.25, 6);
    expect(out[K + 3]).toBeCloseTo(-2.25, 6);
  });

  it("sign-extends negative bytes", () => {
    // `0xff` is -1, not 255. Masking flips the sign of every negative weight,
    // and the magnitudes stay plausible, so only the signs can catch it.
    const N = 1;
    const K = 16;
    const input = synth(N, K, (_row, col) => -(col + 1) / K);
    const { codes, scale } = quantizeQ8(input, N, K);
    const out = dequantizeQ8({ packed: packQ8({ codes, N, K }), scale, N, K });
    expect([...out].every((v) => v <= 0)).toBe(true);
  });

  it("is markedly closer than q4 on the same data", () => {
    // The measurement this format exists for: on `layers.0`, adaLN at q4 costs
    // 4.8% relative RMS on the block's output and at q8 costs almost nothing.
    // Here it is only asserted that the weight error itself is far smaller —
    // the block-level number lives in the golden's manifest, where it was
    // measured rather than derived.
    const N = 4;
    const K = GROUP_SIZE * 2;
    const input = synth(N, K, (row, col) => Math.sin(row * 3 + col * 0.07) * (1 + row));
    const q4 = quantizeQ4G128({ input, N, K });
    const dq4 = dequantizeQ4G128({ packed: packQ4({ codes: q4.codes, N, K }), scales: q4.scales, N, K });
    const q8 = quantizeQ8(input, N, K);
    const dq8 = dequantizeQ8({ packed: packQ8({ codes: q8.codes, N, K }), scale: q8.scale, N, K });
    const rms = (a: Float32Array) =>
      Math.sqrt([...a].reduce((s, v, i) => s + (v - input[i]!) ** 2, 0) / a.length);
    expect(rms(dq8)).toBeLessThan(rms(dq4) / 4);
  });
});

describe("DitWeights", () => {
  const N = 2;
  const K = GROUP_SIZE;
  const dense = synth(N, K, (row, col) => Math.cos(row + col * 0.05));
  const { codes, scales } = quantizeQ4G128({ input: dense, N, K });
  const packed = packQ4({ codes, N, K });
  const plain = new Float32Array([1, 2, 3, 4]);
  const ada = synth(N, K, (row, col) => Math.sin(row + col * 0.02));
  const adaQ8 = quantizeQ8(ada, N, K);
  const adaPacked = packQ8({ codes: adaQ8.codes, N, K });

  const manifest: DitManifest = {
    format: { quant: "q4-g128", groupSize: GROUP_SIZE },
    config: { dim: 3840 },
    tensors: [
      { name: "layers.0.attention.to_q.weight", kind: "q4", shape: [N, K], codesOffset: 0, scaleOffset: 0 },
      { name: "layers.0.attention_norm1.weight", kind: "f32", shape: [4], offset: 0 },
      { name: "layers.0.adaLN_modulation.0.weight", kind: "q8", shape: [N, K], codesOffset: 0, scaleOffset: 0 },
    ],
  };
  const make = () =>
    new DitWeights(manifest, { codes: packed, scales, f32: plain, q8: adaPacked, q8Scales: adaQ8.scale });

  it("dequantizes a q4 tensor on demand", () => {
    const got = make().get("layers.0.attention.to_q.weight");
    expect(got).toHaveLength(N * K);
    expect(got[0]).toBeCloseTo(dense[0]!, 2);
  });

  it("dequantizes a q8 tensor on demand", () => {
    // adaLN is the one weight the converter keeps at q8 — it is 2.6% more
    // bytes per layer and three times less error, measured on `layers.0`. A
    // loader that read it with the q4 stride would return numbers, all wrong.
    const got = make().get("layers.0.adaLN_modulation.0.weight");
    expect(got).toHaveLength(N * K);
    for (let i = 0; i < got.length; i += 1) expect(got[i]!).toBeCloseTo(ada[i]!, 2);
  });

  it("returns an f32 tensor as stored", () => {
    expect([...make().get("layers.0.attention_norm1.weight")]).toEqual([1, 2, 3, 4]);
  });

  it("names the tensor it cannot find", () => {
    // A loader returning undefined for a typo produces NaN several ops later,
    // where the name is no longer in scope. The error carries it.
    expect(() => make().get("layers.0.attention.to_Q.weight")).toThrow(/layers\.0\.attention\.to_Q\.weight/);
  });

  it("refuses a manifest whose group size is not the one it implements", () => {
    // Reading a 64-group blob with a 128-group stride yields numbers, all
    // wrong, with nothing to signal it.
    const other = { ...manifest, format: { ...manifest.format, groupSize: 64 } };
    expect(() => new DitWeights(other, { codes: packed, scales, f32: plain, q8: adaPacked, q8Scales: adaQ8.scale })).toThrow(/group size/i);
  });

  it("refuses a blob shorter than the manifest claims", () => {
    // Truncated downloads are the failure this format cannot otherwise detect:
    // offsets stay in range for every tensor but the last, and the last reads
    // zeros.
    expect(() => new DitWeights(manifest, { codes: packed.slice(0, 4), scales, f32: plain, q8: adaPacked, q8Scales: adaQ8.scale })).toThrow(/truncat/i);
  });

  it("collects one block's weights under the names the block expects", () => {
    // `block.ts` takes `attention_to_q_weight`; the checkpoint says
    // `layers.0.attention.to_q.weight`. The translation is here rather than at
    // each call site so there is one place for it to be wrong.
    const w = make().block(0, ["attention.to_q.weight", "attention_norm1.weight"]);
    expect(Object.keys(w).sort()).toEqual(["attention_norm1_weight", "attention_to_q_weight"]);
  });
});
