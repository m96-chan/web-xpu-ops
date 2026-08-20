import { describe } from "vitest";
import { expectAgrees, gpuTest, kernel, params, useGpu } from "../../harness/index.js";
import { matvecQ8Residual, packQ8 } from "./reference.js";

/** Own file, same reasoning as `q8.wgsl.test.ts` and `q8_ffn.wgsl.test.ts` — issue #38's per-process GPU dispatch ceiling. */
const code = kernel(import.meta.url, "q8_residual");

describe("matvecQ8Residual / wgsl", () => {
  useGpu();

  const codeWave = (n: number, k: number, phase = 0) =>
    Int32Array.from({ length: n }, (_, i) => Math.round(Math.sin(i * k + phase) * 100));
  const wave = (n: number, k: number, phase = 0) => Float32Array.from({ length: n }, (_, i) => Math.sin(i * k + phase) * 1.5);
  const scaleWave = (n: number) => Float32Array.from({ length: n }, (_, i) => 0.02 * (i + 1) + 0.007);
  /** Distinct from `vector`/`weight`'s own wave phase — a residual dropped on the floor (output === projection alone) would still coincidentally "agree" if residual and vector produced the same numbers. */
  const residualWave = (n: number) => Float32Array.from({ length: n }, (_, i) => Math.cos(i * 0.29 + 2.1) * 5);

  const TOLERANCE = { rel: 1e-4, abs: 5e-5 };

  for (const [N, K] of [[1, 3], [1, 8], [5, 1024], [3, 1030], [7, 1027], [2, 2600]] as const) {
    gpuTest(`agrees with the reference at N=${N} K=${K}`, async (run) => {
      const codes = codeWave(N * K, 0.37);
      const weight = packQ8({ codes, N, K });
      const scale = scaleWave(N);
      const vector = wave(K, 0.11, 0.8);
      const residual = residualWave(N);
      await expectAgrees(
        run,
        {
          code,
          bindings: [
            { kind: "storage", data: weight },
            { kind: "storage", data: scale },
            { kind: "storage", data: vector },
            { kind: "storage", data: residual },
            { kind: "out", type: "f32", length: N },
            { kind: "uniform", data: params([["u32", N], ["u32", K]]) },
          ],
          workgroups: [N],
        },
        [matvecQ8Residual({ weight, scale, vector, residual, N, K })],
        [TOLERANCE],
      );
    });
  }

  gpuTest("adds each row's own residual, not row 0's", async (run) => {
    // Every row's projection is 0 (all-zero codes), so the output is exactly
    // the residual — a kernel that broadcasts residual[0] to every row, or
    // drops the residual add on the floor, both fail this even though every
    // "agrees at N=... K=..." case above already has a non-zero residual
    // added in.
    const N = 3;
    const K = 8;
    const codes = new Int32Array(N * K);
    const weight = packQ8({ codes, N, K });
    const scale = scaleWave(N);
    const vector = wave(K, 0.11, 0.8);
    const residual = Float32Array.from([1, 20, 300]);
    await expectAgrees(
      run,
      {
        code,
        bindings: [
          { kind: "storage", data: weight },
          { kind: "storage", data: scale },
          { kind: "storage", data: vector },
          { kind: "storage", data: residual },
          { kind: "out", type: "f32", length: N },
          { kind: "uniform", data: params([["u32", N], ["u32", K]]) },
        ],
        workgroups: [N],
      },
      [residual],
    );
  });

  gpuTest("reads the words past the first workgroup pass", async (run) => {
    const N = 3;
    const K = 1030;
    const codes = new Int32Array(N * K);
    const vector = new Float32Array(K);
    vector[1027] = 2;
    vector[K - 1] = -3;
    for (let row = 0; row < N; row += 1) {
      codes[row * K + 1027] = 1 + row;
      codes[row * K + K - 1] = 2 * (row + 1);
    }
    const weight = packQ8({ codes, N, K });
    const scale = scaleWave(N);
    const residual = residualWave(N);
    await expectAgrees(
      run,
      {
        code,
        bindings: [
          { kind: "storage", data: weight },
          { kind: "storage", data: scale },
          { kind: "storage", data: vector },
          { kind: "storage", data: residual },
          { kind: "out", type: "f32", length: N },
          { kind: "uniform", data: params([["u32", N], ["u32", K]]) },
        ],
        workgroups: [N],
      },
      [matvecQ8Residual({ weight, scale, vector, residual, N, K })],
    );
  });
});
