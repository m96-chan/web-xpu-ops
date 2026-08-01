import { describe } from "vitest";
import { expectAgrees, gpuTest, kernel, params, useGpu } from "../../harness/index.js";
import { quantize } from "./reference.js";

const code = kernel(import.meta.url);
const wave = (n: number, k = 0.37) => Float32Array.from({ length: n }, (_, i) => Math.sin(i * k) * 2);

describe("quantize / wgsl", () => {
  useGpu();

  for (const [N, D] of [[1, 64], [2, 300]] as const) {
    gpuTest(`agrees with the reference at N=${N} D=${D}`, async (run) => {
      const input = wave(N * D, 0.41);
      const want = quantize({ input, N, D });
      await expectAgrees(
        run,
        {
          code,
          bindings: [
            { kind: "storage", data: input },
            { kind: "out", type: "i32", length: N * D },
            { kind: "out", type: "f32", length: N },
            { kind: "uniform", data: params([["u32", N], ["u32", D]]) },
          ],
          workgroups: [N],
        },
        [want.output, want.scales],
        // Codes are integers and must match exactly; the scales beside them are
        // f32 and differ by an ulp, so they get the ordinary tolerance.
        [{ rel: 0, abs: 0 }, undefined],
      );
    });
  }

  gpuTest("does not divide by zero on an all-zero row", async (run) => {
    // absmax is 0, so the reciprocal has to be forced rather than computed.
    const D = 16;
    const input = new Float32Array(D);
    const want = quantize({ input, N: 1, D });
    await expectAgrees(
      run,
      {
        code,
        bindings: [
          { kind: "storage", data: input },
          { kind: "out", type: "i32", length: D },
          { kind: "out", type: "f32", length: 1 },
          { kind: "uniform", data: params([["u32", 1], ["u32", D]]) },
        ],
        workgroups: [1],
      },
      [want.output, want.scales],
      [{ rel: 0, abs: 0 }, undefined],
    );
  });
});
