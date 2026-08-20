import { describe } from "vitest";
import { expectAgrees, gpuTest, kernel, params, useGpu } from "../../harness/index.js";
import { matvecQ8, packQ8 } from "./reference.js";

/**
 * `matvecQ8`'s own file rather than a second describe block in
 * `wgsl.test.ts`. Not required by the resolution grammar — `q8` is a distinct
 * entry point beside `kernel`, and `kernel()` already reads either by name —
 * but `scripts/test.mjs` gives each **test file** its own process (issue
 * #38), and this repository's own suite crashes the GPU worker with a glibc
 * assertion (`pthread_mutex_lock`) when this op's fifteen dispatches — seven
 * from `matvec`, eight from here — run back to back in one process. Split
 * across two files, each set of dispatches stays comfortably under whatever
 * this binding's per-process ceiling is; measured repeatedly, both files pass
 * on their own and only the combined file crashes.
 */
const code = kernel(import.meta.url, "q8");

describe("matvecQ8 / wgsl", () => {
  useGpu();

  /** Deterministic int8 codes, mixed sign, within [-100, 100] so no case brushes the [-127, 127] clamp. */
  const codeWave = (n: number, k: number, phase = 0) =>
    Int32Array.from({ length: n }, (_, i) => Math.round(Math.sin(i * k + phase) * 100));
  const wave = (n: number, k: number, phase = 0) =>
    Float32Array.from({ length: n }, (_, i) => Math.sin(i * k + phase) * 1.5);
  /** Strictly increasing and never 1 — a row picking up its neighbour's scale changes the answer, not just its magnitude. */
  const scaleWave = (n: number) => Float32Array.from({ length: n }, (_, i) => 0.02 * (i + 1) + 0.007);

  // K spans the kernel's actual unit of work, a packed *word* (4 columns), and
  // its unit of parallelism, the 256-wide workgroup (1024 columns per pass):
  // below a word, below one pass, exactly one pass, and twice not a multiple
  // of either. N is kept distinct from K, as in `matvec`'s own cases, so an
  // index built from the wrong dimension cannot pass by coincidence.
  //
  // Tolerance is widened past the harness default and measured, not assumed:
  // codes here run to ±100 (`codeWave`) against `matvec`'s own case values of
  // order 1, so a term in this dot product is roughly two orders of magnitude
  // larger, and f32 accumulation error scales with the magnitude of what is
  // being accumulated. Instrumented with the tolerance forced to zero, the
  // worst observed relative difference across these six cases is 3.84e-5 (at
  // N=3, K=1030) against the harness default of 1e-5, and the worst absolute
  // difference is 2.60e-5 (at N=5, K=1024) against a default of 1e-6. `rel:
  // 1e-4` and `abs: 5e-5` both keep roughly 2x headroom over those measured
  // worsts rather than being opened until green.
  const TOLERANCE = { rel: 1e-4, abs: 5e-5 };

  for (const [N, K] of [[1, 3], [1, 8], [5, 1024], [3, 1030], [7, 1027], [2, 2600]] as const) {
    gpuTest(`agrees with the reference at N=${N} K=${K}`, async (run) => {
      const codes = codeWave(N * K, 0.37);
      const weight = packQ8({ codes, N, K });
      const scale = scaleWave(N);
      const vector = wave(K, 0.11, 0.8);
      await expectAgrees(
        run,
        {
          code,
          bindings: [
            { kind: "storage", data: weight },
            { kind: "storage", data: scale },
            { kind: "storage", data: vector },
            { kind: "out", type: "f32", length: N },
            { kind: "uniform", data: params([["u32", N], ["u32", K]]) },
          ],
          workgroups: [N],
        },
        [matvecQ8({ weight, scale, vector, N, K })],
        [TOLERANCE],
      );
    });
  }

  gpuTest("reads the words past the first workgroup pass", async (run) => {
    // Every lane's first pass (words 0..255, columns 0..1023) sees only zero
    // codes. The only non-zero terms sit at column 1027 — just past that first
    // pass — and at the very last column, so a kernel that stops after one
    // pass over the packed words, or that loses the tail of the strided loop,
    // returns 0 instead of a number of order one.
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
    await expectAgrees(
      run,
      {
        code,
        bindings: [
          { kind: "storage", data: weight },
          { kind: "storage", data: scale },
          { kind: "storage", data: vector },
          { kind: "out", type: "f32", length: N },
          { kind: "uniform", data: params([["u32", N], ["u32", K]]) },
        ],
        workgroups: [N],
      },
      [matvecQ8({ weight, scale, vector, N, K })],
    );
  });

  gpuTest("does not read past the end of a row's packed words", async (run) => {
    // Both the packed-weight and vector buffers are bound longer than the
    // kernel is told to read, and the slack is filled with a value nothing
    // else here could produce. With buffers sized exactly to
    // N * ceil(K/4) words and K floats, an off-by-one word bound reads
    // `weight[row_word_offset + words_per_row]` — one word past this row,
    // which for the last row is out of bounds entirely — and WGSL's robust
    // buffer access hands back 0u instead. Padding both buffers is what makes
    // that wrong term visible rather than silently zero either way.
    //
    // K is not a multiple of 4, so a row's packed words run out mid-lane,
    // which is where a bound like this goes wrong in the first place. Rows
    // carry distinct codes so that borrowing from a neighbouring row's words
    // also shows, rather than cancelling into the tolerance.
    const N = 4;
    const K = 1023;
    const wordsPerRow = Math.ceil(K / 4);
    const SENTINEL_WORD = 0x7f7f7f7f;
    const SENTINEL_VEC = 1000;
    const codes = Int32Array.from({ length: N * K }, (_, i) => (Math.floor(i / K) % 5) + 1);
    const packedRows = packQ8({ codes, N, K });
    const weight = Uint32Array.from({ length: N * wordsPerRow + 256 }, (_, i) =>
      i < N * wordsPerRow ? packedRows[i]! : SENTINEL_WORD,
    );
    const vector = Float32Array.from({ length: K + 256 }, (_, i) => (i < K ? 1 : SENTINEL_VEC));
    const scale = scaleWave(N);
    await expectAgrees(
      run,
      {
        code,
        bindings: [
          { kind: "storage", data: weight },
          { kind: "storage", data: scale },
          { kind: "storage", data: vector },
          { kind: "out", type: "f32", length: N },
          { kind: "uniform", data: params([["u32", N], ["u32", K]]) },
        ],
        workgroups: [N],
      },
      [matvecQ8({ weight: packedRows, scale, vector: vector.slice(0, K), N, K })],
    );
  });
});
