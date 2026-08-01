import { describe } from "vitest";
import { expectAgrees, gpuTest, kernel, params, useGpu } from "../../harness/index.js";
import { softmax } from "./reference.js";

const code = kernel(import.meta.url);
const wave = (n: number, k = 0.37) => Float32Array.from({ length: n }, (_, i) => Math.sin(i * k) * 2);

describe("softmax / wgsl", () => {
  useGpu();

  for (const [N, D] of [[1, 16], [2, 256], [3, 500]] as const) {
    gpuTest(`agrees with the reference at N=${N} D=${D}`, async (run) => {
      const input = wave(N * D, 0.23);
      await expectAgrees(
        run,
        {
          code,
          bindings: [
            { kind: "storage", data: input },
            { kind: "out", type: "f32", length: N * D },
            { kind: "uniform", data: params([["u32", N], ["u32", D]]) },
          ],
          workgroups: [N],
        },
        [softmax({ input, N, D })],
      );
    });
  }

  gpuTest("survives logits that would overflow exp", async (run) => {
    // exp(800) is Infinity in f32 and takes the whole row to NaN. Subtracting
    // the row maximum is the only thing preventing it, so this is the case that
    // actually tests it — ordinary inputs never come close.
    const D = 8;
    const input = Float32Array.from([100, 200, 300, 400, 500, 600, 700, 800]);
    await expectAgrees(
      run,
      {
        code,
        bindings: [
          { kind: "storage", data: input },
          { kind: "out", type: "f32", length: D },
          { kind: "uniform", data: params([["u32", 1], ["u32", D]]) },
        ],
        workgroups: [1],
      },
      [softmax({ input, N: 1, D })],
    );
  });
});
