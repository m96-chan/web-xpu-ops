import { describe } from "vitest";
import { expectAgrees, gpuTest, kernel, params, useGpu } from "../../harness/index.js";
import { matvec } from "./reference.js";

const code = kernel(import.meta.url);

/** Deterministic, and mixed in sign so a dropped term cannot hide in a monotone sum. */
const wave = (n: number, k: number, phase = 0) =>
  Float32Array.from({ length: n }, (_, i) => Math.sin(i * k + phase) * 1.5);

describe("matvec / wgsl", () => {
  useGpu();

  // K spans the 256-wide workgroup deliberately: below it, exactly on it, and
  // twice not a multiple of it. The strided read loop and the tree reduction
  // each behave differently depending on which, and the tail is the one that
  // gets written wrong. M is kept distinct from K throughout, so indexing the
  // vector with the row offset — the classic GEMV slip — cannot pass by
  // coincidence.
  //
  // The default tolerance is left alone. A dot product accumulated in f32 and in
  // a different order than the f64 reference is where a kernel usually needs
  // room, so it was measured rather than assumed: worst observed relative
  // difference across these cases is 1.44e-6, at K=2560, against a default of
  // 1e-5. The tree reduction is what keeps it there — the error grows with the
  // depth of the summation, not with K.
  for (const [M, K] of [[1, 8], [5, 256], [3, 300], [7, 1027], [2, 2560]] as const) {
    gpuTest(`agrees with the reference at M=${M} K=${K}`, async (run) => {
      const matrix = wave(M * K, 0.37);
      const vector = wave(K, 0.11, 0.8);
      await expectAgrees(
        run,
        {
          code,
          bindings: [
            { kind: "storage", data: matrix },
            { kind: "storage", data: vector },
            { kind: "out", type: "f32", length: M },
            { kind: "uniform", data: params([["u32", M], ["u32", K]]) },
          ],
          workgroups: [M],
        },
        [matvec({ matrix, vector, M, K })],
      );
    });
  }

  gpuTest("reads the columns past the workgroup width", async (run) => {
    // Everything the 256 lanes see on their first pass is zero. The only
    // non-zero terms sit at column 260 and at the very last column, so a kernel
    // that stops after one pass, or that loses the tail of the strided loop,
    // returns 0 instead of a number of order one. The general cases above would
    // only shift such an error into the tolerance's noise; here it is the whole
    // answer.
    const M = 3;
    const K = 301;
    const matrix = new Float32Array(M * K);
    const vector = new Float32Array(K);
    vector[260] = 2;
    vector[K - 1] = -3;
    for (let row = 0; row < M; row += 1) {
      matrix[row * K + 260] = 1 + row;
      matrix[row * K + K - 1] = 0.5 * (row + 1);
    }
    await expectAgrees(
      run,
      {
        code,
        bindings: [
          { kind: "storage", data: matrix },
          { kind: "storage", data: vector },
          { kind: "out", type: "f32", length: M },
          { kind: "uniform", data: params([["u32", M], ["u32", K]]) },
        ],
        workgroups: [M],
      },
      [matvec({ matrix, vector, M, K })],
    );
  });

  gpuTest("does not read past the end of a row", async (run) => {
    // Both buffers are bound longer than the kernel is told to read, and the
    // slack is filled with a value nothing else here could produce. That padding
    // is the whole point of the case: with buffers sized exactly M*K and K, an
    // off-by-one column bound reads `vector[K]`, which is out of bounds, and
    // WGSL's robust buffer access hands back 0.0 — the wrong term multiplies to
    // nothing and the bug passes. Measured: `col < K` changed to `col <= K`
    // leaves the whole suite green without this padding, and fails here with it.
    //
    // K is not a multiple of the workgroup width, so a row's weights run out
    // mid-pass, which is where such a bound goes wrong in the first place. Rows
    // are distinct constants so that borrowing from the neighbouring row also
    // shows, rather than cancelling into the tolerance.
    const M = 4;
    const K = 300;
    const SENTINEL = 1000;
    const matrix = Float32Array.from({ length: M * K + 256 }, (_, i) =>
      i < M * K ? Math.floor(i / K) + 1 : SENTINEL,
    );
    const vector = Float32Array.from({ length: K + 256 }, (_, i) => (i < K ? 1 : SENTINEL));
    await expectAgrees(
      run,
      {
        code,
        bindings: [
          { kind: "storage", data: matrix },
          { kind: "storage", data: vector },
          { kind: "out", type: "f32", length: M },
          { kind: "uniform", data: params([["u32", M], ["u32", K]]) },
        ],
        workgroups: [M],
      },
      [matvec({ matrix, vector, M, K })],
    );
  });
});
