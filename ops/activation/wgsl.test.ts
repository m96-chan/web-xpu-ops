import { describe } from "vitest";
import { expectAgrees, gpuTest, kernel, params, useGpu } from "../../harness/index.js";
import { ACTIVATION, activation } from "./reference.js";

const code = kernel(import.meta.url);
const wave = (n: number, k = 0.37) => Float32Array.from({ length: n }, (_, i) => Math.sin(i * k) * 2);

describe("activation / wgsl", () => {
  useGpu();

  for (const [label, kind] of [["relu2", ACTIVATION.relu2], ["silu", ACTIVATION.silu]] as const) {
    gpuTest(`agrees with the reference for ${label}`, async (run) => {
      // Saturating magnitudes on both sides: silu underflows for very negative
      // x and relu2 squares, so large values are where they diverge if at all.
      const input = Float32Array.from([-20, -4, -2, -0.5, 0, 0.5, 2, 4, 20, ...wave(64)]);
      await expectAgrees(
        run,
        {
          code,
          bindings: [
            { kind: "storage", data: input },
            { kind: "out", type: "f32", length: input.length },
            { kind: "uniform", data: params([["u32", input.length], ["u32", kind]]) },
          ],
          workgroups: [Math.ceil(input.length / 256)],
        },
        [activation({ input, kind })],
      );
    });
  }
});
