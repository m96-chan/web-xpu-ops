import { describe } from "vitest";
import { expectAgrees, gpuTest, kernel, params, useGpu } from "../../harness/index.js";
import { matvecQ8Ffn, packQ8 } from "./reference.js";

/**
 * `matvecQ8Ffn`'s own file, same reasoning as `q8.wgsl.test.ts`'s own doc
 * (issue #38's per-process GPU dispatch ceiling) — this op's dispatches on
 * top of `matvec`'s and `matvecQ8`'s own would only get closer to whatever
 * that ceiling is, not further from it, so a fresh file keeps this op's own
 * eight-ish dispatches isolated from every other op's.
 */
const code = kernel(import.meta.url, "q8_ffn");

describe("matvecQ8Ffn / wgsl", () => {
  useGpu();

  /** Same wave generators as q8.wgsl.test.ts, one more phase for the up weight so gate and up are not the same numbers wearing different scales. */
  const codeWave = (n: number, phase: number) =>
    Int32Array.from({ length: n }, (_, i) => Math.round(Math.sin(i * 0.37 + phase) * 100));
  const wave = (n: number, phase: number) => Float32Array.from({ length: n }, (_, i) => Math.sin(i * 0.11 + phase) * 1.5);
  const scaleWave = (n: number, phase: number) => Float32Array.from({ length: n }, (_, i) => 0.02 * (i + 1) + phase);

  // Same K spans as q8.wgsl.test.ts: below a word, below one pass, exactly one
  // pass, twice not a multiple of either — the fused kernel walks the same
  // packed-word loop as q8.wgsl, just twice per pass, so the same boundaries
  // matter here.
  const TOLERANCE = { rel: 1e-4, abs: 5e-5 };

  for (const [N, K] of [[1, 3], [1, 8], [5, 1024], [3, 1030], [7, 1027], [2, 2600]] as const) {
    gpuTest(`agrees with the reference at N=${N} K=${K}`, async (run) => {
      const codesGate = codeWave(N * K, 0.37);
      const codesUp = codeWave(N * K, 1.9); // distinct phase — a gate/up mix-up would still "agree" if the two weights were identical
      const weightGate = packQ8({ codes: codesGate, N, K });
      const weightUp = packQ8({ codes: codesUp, N, K });
      const scaleGate = scaleWave(N, 0.007);
      const scaleUp = scaleWave(N, 0.013); // distinct from scaleGate — a scale mix-up would still "agree" if the two matched
      const vector = wave(K, 0.8);
      await expectAgrees(
        run,
        {
          code,
          bindings: [
            { kind: "storage", data: weightGate },
            { kind: "storage", data: scaleGate },
            { kind: "storage", data: weightUp },
            { kind: "storage", data: scaleUp },
            { kind: "storage", data: vector },
            { kind: "out", type: "f32", length: N },
            { kind: "uniform", data: params([["u32", N], ["u32", K]]) },
          ],
          workgroups: [N],
        },
        [matvecQ8Ffn({ weightGate, scaleGate, weightUp, scaleUp, vector, N, K })],
        [TOLERANCE],
      );
    });
  }

  gpuTest("reads the words past the first workgroup pass, for both weights", async (run) => {
    // Same shape as q8.wgsl.test.ts's own "reads past first pass" case,
    // doubled: gate and up each get their own non-zero column, past column
    // 1023 (the first 256-lane pass), and distinct from each other's column —
    // a kernel that only advances one of the two loop variables past the
    // first pass would return 0 for the other.
    const N = 2;
    const K = 1030;
    const vector = new Float32Array(K);
    vector[1027] = 2;
    vector[K - 1] = -3;
    const codesGate = new Int32Array(N * K);
    const codesUp = new Int32Array(N * K);
    for (let row = 0; row < N; row += 1) {
      codesGate[row * K + 1027] = 1 + row;
      codesGate[row * K + K - 1] = 2 * (row + 1);
      codesUp[row * K + 1027] = 3 + row;
      codesUp[row * K + K - 1] = -1 * (row + 1);
    }
    const weightGate = packQ8({ codes: codesGate, N, K });
    const weightUp = packQ8({ codes: codesUp, N, K });
    const scaleGate = scaleWave(N, 0.007);
    const scaleUp = scaleWave(N, 0.013);
    await expectAgrees(
      run,
      {
        code,
        bindings: [
          { kind: "storage", data: weightGate },
          { kind: "storage", data: scaleGate },
          { kind: "storage", data: weightUp },
          { kind: "storage", data: scaleUp },
          { kind: "storage", data: vector },
          { kind: "out", type: "f32", length: N },
          { kind: "uniform", data: params([["u32", N], ["u32", K]]) },
        ],
        workgroups: [N],
      },
      [matvecQ8Ffn({ weightGate, scaleGate, weightUp, scaleUp, vector, N, K })],
    );
  });
});
