import { describe } from "vitest";
import { expectAgrees, gpuTest, kernel, params, useGpu } from "../../harness/index.js";
import { rmsnorm } from "./reference.js";

const code = kernel(import.meta.url);
const wave = (n: number, k = 0.37) => Float32Array.from({ length: n }, (_, i) => Math.sin(i * k) * 2);

describe("rmsnorm / wgsl", () => {
  useGpu();

  // D spans the 256-wide workgroup deliberately: below it, exactly on it, and
  // not a multiple of it. The strided loop and the tree reduction each behave
  // differently depending on which, and only one of those is the common case.
  for (const [N, D] of [[2, 8], [1, 256], [3, 300], [2, 2560]] as const) {
    gpuTest(`agrees with the reference at N=${N} D=${D}`, async (run) => {
      const input = wave(N * D);
      const weight = Float32Array.from({ length: D }, (_, i) => 0.5 + Math.cos(i * 0.11) * 0.4);
      const eps = 1e-5;
      await expectAgrees(
        run,
        {
          code,
          bindings: [
            { kind: "storage", data: input },
            { kind: "storage", data: weight },
            { kind: "out", type: "f32", length: N * D },
            { kind: "uniform", data: params([["u32", N], ["u32", D], ["f32", eps]]) },
          ],
          workgroups: [N],
        },
        [rmsnorm({ input, weight, N, D, eps })],
      );
    });
  }

  gpuTest("keeps eps meaningful on an all-but-zero row", async (run) => {
    // Without this case, removing eps from the shader entirely still passes:
    // on ordinary input sumSquares/D dwarfs it. Here it is the only thing
    // between the reciprocal square root and a division by zero.
    const D = 8;
    const input = new Float32Array(D);
    const weight = new Float32Array(D).fill(1);
    const eps = 1e-5;
    await expectAgrees(
      run,
      {
        code,
        bindings: [
          { kind: "storage", data: input },
          { kind: "storage", data: weight },
          { kind: "out", type: "f32", length: D },
          { kind: "uniform", data: params([["u32", 1], ["u32", D], ["f32", eps]]) },
        ],
        workgroups: [1],
      },
      [rmsnorm({ input, weight, N: 1, D, eps })],
    );
  });
});
