import { describe, expect, it } from "vitest";
import { expectAgrees, gpuTest, kernel, params, useGpu } from "../../harness/index.js";
import { packInt8Rows } from "../../llm/weights-q8.js";
import { dequantTranspose } from "./reference.js";

const code = kernel(import.meta.url);

describe("dequant_transpose / wgsl", () => {
  useGpu();

  const SHAPES = [
    { outFeatures: 1, inFeatures: 1, why: "one of everything" },
    { outFeatures: 5, inFeatures: 3, why: "inFeatures not a multiple of 4 — packing pads the last word" },
    { outFeatures: 3, inFeatures: 8, why: "inFeatures exactly two packed words" },
    { outFeatures: 16, inFeatures: 112, why: "Sarashina2.2-1B's Q projection shape" },
    // Sized to be well past one workgroup in both directions without the
    // full wGate/wUp shape's own byte count (6272 x 1792, a 44.8 MiB "out"
    // buffer): a repeated large-allocate-then-free was measured elsewhere in
    // this harness to destabilise this device's Node/Dawn binding on its own
    // (issue #51, `harness/wgsl.ts`'s own doc), and this shape already
    // exercises "many workgroups, several packed words per row" without
    // paying that risk on every run.
    { outFeatures: 512, inFeatures: 1792, why: "past one workgroup many times over, Sarashina2.2-1B's own inFeatures" },
  ];

  for (const { outFeatures, inFeatures, why } of SHAPES) {
    gpuTest(`agrees with the reference at outFeatures=${outFeatures} inFeatures=${inFeatures} — ${why}`, async (run) => {
      const codes = Int8Array.from({ length: outFeatures * inFeatures }, (_, i) => ((i * 37) % 200) - 100);
      const scale = Float32Array.from({ length: outFeatures }, (_, i) => 0.01 + i * 1e-4);
      const weight = packInt8Rows(codes, outFeatures, inFeatures);
      const want = dequantTranspose({ weight, scale, outFeatures, inFeatures });

      await expectAgrees(
        run,
        {
          code,
          bindings: [
            { kind: "storage", data: weight },
            { kind: "storage", data: scale },
            { kind: "out", type: "f32", length: outFeatures * inFeatures },
            { kind: "uniform", data: params([["u32", outFeatures], ["u32", inFeatures]]) },
          ],
          workgroups: [Math.ceil((outFeatures * inFeatures) / 256)],
        },
        [want],
        { rel: 0, abs: 0 },
      );
    });
  }

  it("rejects a weight/scale sized for the wrong outFeatures/inFeatures", () => {
    const codes = new Int8Array(3 * 5);
    const weight = packInt8Rows(codes, 3, 5);
    const scale = new Float32Array(3);
    expect(() => dequantTranspose({ weight, scale, outFeatures: 3, inFeatures: 5 })).not.toThrow();
    expect(() => dequantTranspose({ weight, scale, outFeatures: 4, inFeatures: 5 })).toThrow(/expected 8 packed words, got 6/);
    expect(() => dequantTranspose({ weight, scale: new Float32Array(2), outFeatures: 3, inFeatures: 5 })).toThrow(/expected 3 scales, got 2/);
  });
});
