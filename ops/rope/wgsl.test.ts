import { describe } from "vitest";
import { expectAgrees, gpuTest, kernel, params, useGpu } from "../../harness/index.js";
import { rope } from "./reference.js";

const code = kernel(import.meta.url);
const wave = (n: number, k = 0.37) => Float32Array.from({ length: n }, (_, i) => Math.sin(i * k) * 2);

describe("rope / wgsl", () => {
  useGpu();

  // Loosened on measurement, not on principle. GPU sin and cos carry up to
  // 1.86e-4 of absolute error here — three orders of magnitude worse than f32
  // epsilon — and this op calls both per element. `pow` was measured separately
  // and agrees to 2.8e-7, so the transcendentals are the whole difference; the
  // kernel and the reference compute the same expression.
  //
  // Nothing tighter is reachable, and a test demanding it would be reporting
  // the hardware rather than the shader.
  const transcendental = { abs: 1e-3 };

  for (const posOffset of [0, 7]) {
    gpuTest(`agrees with the reference at posOffset=${posOffset}`, async (run) => {
      const [N, numHeads, headDim, thetaBase] = [3, 4, 16, 10000];
      const input = wave(N * numHeads * headDim, 0.29);
      await expectAgrees(
        run,
        {
          code,
          bindings: [
            { kind: "storage", data: input },
            { kind: "out", type: "f32", length: input.length },
            {
              kind: "uniform",
              data: params([
                ["u32", N], ["u32", numHeads], ["u32", headDim],
                ["u32", posOffset], ["f32", thetaBase],
              ]),
            },
          ],
          workgroups: [Math.ceil(input.length / 256)],
        },
        [rope({ input, N, numHeads, headDim, posOffset, thetaBase })],
        transcendental,
      );
    });
  }
});
