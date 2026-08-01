import { describe } from "vitest";
import { expectAgrees, gpuTest, kernel, params, useGpu } from "../../harness/index.js";
import { dequantize } from "./reference.js";

const code = kernel(import.meta.url);

describe("dequantize / wgsl", () => {
  useGpu();

  gpuTest("agrees with the reference across the int8 range", async (run) => {
    const input = Int32Array.from({ length: 300 }, (_, i) => ((i * 37) % 255) - 127);
    const [weightScale, inputScale] = [0.0125, 0.031];
    await expectAgrees(
      run,
      {
        code,
        bindings: [
          { kind: "storage", data: input },
          { kind: "out", type: "f32", length: input.length },
          { kind: "uniform", data: params([["f32", weightScale]]) },
          { kind: "uniform", data: params([["f32", inputScale]]) },
          { kind: "uniform", data: params([["u32", input.length]]) },
        ],
        workgroups: [Math.ceil(input.length / 256)],
      },
      [dequantize({ input, weightScale, inputScale })],
    );
  });
});
