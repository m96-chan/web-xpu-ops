import { describe } from "vitest";
import { expectAgrees, gpuTest, kernel, params, useGpu } from "../../harness/index.js";
import { Q4_GROUP_SIZE, matvecQ4G128, packQ4, quantizeQ4G128 } from "./reference.js";

/**
 * `matvecQ4G128`'s own file rather than a second `describe` block in
 * `wgsl.test.ts` or `q8.wgsl.test.ts` — the same reason those two are already
 * apart (`ops/matvec/q8.wgsl.test.ts`'s own header): `scripts/test.mjs` gives
 * each **test file** its own process (issue #38), and this repository's GPU
 * worker has been measured crashing when one process runs too many of this
 * op's dispatches back to back. This file holds eight, matching what `q8`
 * already runs cleanly on its own.
 */
const code = kernel(import.meta.url, "q4_g128");

describe("matvecQ4G128 / wgsl", () => {
  useGpu();

  const wave = (n: number, k: number, phase = 0) =>
    Float32Array.from({ length: n }, (_, i) => Math.sin(i * k + phase) * 1.5);
  /**
   * A weight matrix whose per-group absmax genuinely varies — the amplitude
   * ramps with the group index, so group 0 of a row and group 3 of the same
   * row have scales two orders of magnitude apart. A kernel that read one
   * scale per row (or the wrong group's) would still produce plausible
   * numbers on a flat matrix; here it cannot.
   *
   * The frequency (2.1 rad per column, so the sign changes every second or
   * third column) is not cosmetic. An earlier version used 0.37, which is
   * fine for the deep shapes but leaves the two smallest cases (K=3, K=8)
   * entirely inside the first positive half-period — every code non-negative,
   * so **nothing in either case depended on sign extension**. That showed up
   * as a mutation that should have reddened all six shapes reddening only
   * four (rule 1: look at the ones that stayed green). At 2.1 every case here
   * carries negative codes.
   */
  const rampedMatrix = (N: number, K: number) =>
    Float32Array.from({ length: N * K }, (_, i) => {
      const row = Math.floor(i / K);
      const group = Math.floor((i % K) / Q4_GROUP_SIZE);
      return Math.sin(i * 2.1 + row) * Math.pow(10, group % 3) * (1 + row);
    });

  /**
   * Widened past the harness default and measured, not assumed (rules 2/9).
   * The dot products here run over group scales spanning two orders of
   * magnitude (`rampedMatrix`), so a row's sum is dominated by its largest
   * group while the smallest group's terms still have to land — f32
   * accumulation error scales with the magnitude of what is accumulated, and
   * the kernel accumulates in a different order (256 lanes, tree reduction)
   * from the reference's left-to-right f64.
   *
   * Instrumented with the tolerance forced to zero (`rel: 0, abs: 0`) on this
   * machine's RTX 5090 / Dawn (webgpu 0.4.x), the six shapes below disagree by
   * at worst **3.08e-5 relative** (N=7, K=1027, an output element of 5.25 left
   * over from cancelling terms three orders of magnitude larger — relative
   * error is the wrong measure for that element and this is what it looks
   * like) and **3.20e-4 absolute** (N=5, K=2048, where the sums run to ~1e2).
   * They are different elements in different cases; neither fails both
   * measures at once. `rel: 1e-4` and `abs: 1e-3` keep ~3x headroom over each
   * measured worst rather than being opened until green.
   */
  const TOLERANCE = { rel: 1e-4, abs: 1e-3 };

  // K spans every unit this kernel has: a packed word (8 columns), a scale
  // group (128), and one pass of the 256-lane workgroup (2048 columns). Below
  // a word; exactly a word; two groups with the second one short; exactly one
  // workgroup pass; ragged in both word and group; and past one pass. N is
  // kept distinct from K throughout so an index built from the wrong dimension
  // cannot pass by coincidence.
  for (const [N, K, why] of [
    [1, 3, "shorter than one packed word"],
    [1, 8, "exactly one packed word"],
    [3, 130, "two groups, the second one two columns long"],
    [5, 2048, "exactly one workgroup pass, 16 whole groups"],
    [7, 1027, "ragged in both word and group"],
    [2, 2600, "past one workgroup pass, ragged group tail"],
  ] as const) {
    gpuTest(`agrees with the reference at N=${N} K=${K} — ${why}`, async (run) => {
      const matrix = rampedMatrix(N, K);
      const { codes, scales } = quantizeQ4G128({ input: matrix, N, K });
      const weight = packQ4({ codes, N, K });
      const vector = wave(K, 0.11, 0.8);
      await expectAgrees(
        run,
        {
          code,
          bindings: [
            { kind: "storage", data: weight },
            { kind: "storage", data: scales },
            { kind: "storage", data: vector },
            { kind: "out", type: "f32", length: N },
            { kind: "uniform", data: params([["u32", N], ["u32", K]]) },
          ],
          workgroups: [N],
        },
        [matvecQ4G128({ weight, scale: scales, vector, N, K })],
        [TOLERANCE],
      );
    });
  }

  gpuTest("reads the words past the first workgroup pass", async (run) => {
    // Every lane's first pass covers words 0..255, columns 0..2047, and every
    // code there is zero. The only non-zero terms sit at column 2051 — just
    // past that first pass — and at the last column, so a kernel that stops
    // after one pass over the packed words, or that loses the tail of the
    // strided loop, returns 0 instead of a number of order one.
    const N = 3;
    const K = 2600;
    const codes = new Int32Array(N * K);
    const vector = new Float32Array(K);
    vector[2051] = 2;
    vector[K - 1] = -3;
    for (let row = 0; row < N; row += 1) {
      codes[row * K + 2051] = 1 + row;
      // Negative on purpose: the only two live codes in this case, and one of
      // them has to need sign extension or this case is blind to it.
      codes[row * K + K - 1] = -2 * (row + 1);
    }
    const weight = packQ4({ codes, N, K });
    const groupsPerRow = Math.ceil(K / Q4_GROUP_SIZE);
    // Distinct per group and per row, so a scale read from the wrong group of
    // the right row is as visible as one read from the wrong row.
    const scale = Float32Array.from(
      { length: N * groupsPerRow },
      (_, i) => 0.5 + 0.25 * (i % groupsPerRow) + i * 0.01,
    );
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
      [matvecQ4G128({ weight, scale, vector, N, K })],
    );
  });

  gpuTest("does not read past the end of a row's packed words", async (run) => {
    // Both the packed-weight and vector buffers are bound longer than the
    // kernel is told to read, and the slack carries a value nothing else here
    // could produce. With buffers sized exactly to N * ceil(K/8) words and K
    // floats, an off-by-one word bound reads one word past this row — out of
    // bounds entirely for the last row — and WGSL's robust buffer access hands
    // back 0u instead, hiding the mistake. Padding both buffers is what makes
    // the wrong term visible rather than silently zero.
    //
    // K is a multiple of neither 8 nor 128, so a row's packed words run out
    // mid-nibble and its last scale group is short — the two places a bound
    // like this goes wrong. Rows carry distinct codes so that borrowing from a
    // neighbouring row's words also shows rather than cancelling.
    const N = 4;
    const K = 1021;
    const wordsPerRow = Math.ceil(K / 8);
    const groupsPerRow = Math.ceil(K / Q4_GROUP_SIZE);
    const SENTINEL_WORD = 0x77777777;
    const SENTINEL_VEC = 1000;
    // Distinct per row (so a borrowed neighbouring word shows) and alternating
    // in sign (so this case, like the shape cases, exercises sign extension).
    const codes = Int32Array.from(
      { length: N * K },
      (_, i) => ((Math.floor(i / K) % 5) + 1) * (i % 2 === 0 ? 1 : -1),
    );
    const packedRows = packQ4({ codes, N, K });
    const weight = Uint32Array.from({ length: N * wordsPerRow + 256 }, (_, i) =>
      i < N * wordsPerRow ? packedRows[i]! : SENTINEL_WORD,
    );
    const vector = Float32Array.from({ length: K + 256 }, (_, i) => (i < K ? 1 : SENTINEL_VEC));
    const scale = Float32Array.from({ length: N * groupsPerRow }, (_, i) => 0.02 * (i + 1) + 0.007);
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
      [matvecQ4G128({ weight: packedRows, scale, vector: vector.slice(0, K), N, K })],
    );
  });
});
