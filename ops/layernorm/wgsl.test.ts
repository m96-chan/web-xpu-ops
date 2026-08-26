import { describe } from "vitest";
import { expectAgrees, gpuTest, kernel, params, useGpu } from "../../harness/index.js";
import { layernorm } from "./reference.js";

const code = kernel(import.meta.url);
const wave = (n: number, k = 0.37) => Float32Array.from({ length: n }, (_, i) => Math.sin(i * k) * 2);
const weights = (d: number) => Float32Array.from({ length: d }, (_, i) => 0.5 + Math.cos(i * 0.11) * 0.4);
const biases = (d: number) => Float32Array.from({ length: d }, (_, i) => Math.sin(i * 0.29) * 0.3);
const EPS = 1e-5;

describe("layernorm / wgsl", () => {
  useGpu();

  /**
   * A dispatch too wide for one row of workgroups.
   *
   * 65,535 is the ceiling on every backend measured (#211); the fold reads
   * `num_workgroups` rather than a uniform, so every existing one-dimensional
   * caller keeps working. Narrow on purpose, so most of the work lands on rows
   * the unfolded kernel never reaches.
   */
  gpuTest("folds a two-dimensional dispatch back into one row index", async (run) => {
    // The file's own generator and a D it already covers, so the arithmetic is
    // the arithmetic every other case here measures and the only new thing is
    // where the work lands.
    const N = 700;
    const D = 256;
    const input = wave(N * D);
    const weight = wave(D, 0.11);
    const bias = wave(D, 0.23);
    const x = 8;
    const y = Math.ceil(N / x);
    await expectAgrees(
      run,
      {
        code,
        bindings: [
          { kind: "storage", data: input },
          { kind: "storage", data: weight },
          { kind: "storage", data: bias },
          { kind: "out", type: "f32", length: N * D },
          { kind: "uniform", data: params([["u32", N], ["u32", D], ["f32", 1e-5]]) },
        ],
        workgroups: [x, y],
      },
      // The file's default tolerance, as every other case here uses: this is
      // about *where* the work lands, not about the arithmetic.
      [layernorm({ input, weight, bias, N, D, eps: 1e-5 })],
    );
  });

  // D spans the 256-wide workgroup deliberately: below it, exactly on it, and
  // not a multiple of it. The strided loop and the two tree reductions each
  // behave differently depending on which, and only one of those is the common
  // case. Every case carries a non-zero bias, so dropping the bias term is a
  // failure everywhere rather than nowhere.
  for (const [N, D] of [[2, 8], [1, 256], [3, 300], [2, 2560]] as const) {
    gpuTest(`agrees with the reference at N=${N} D=${D}`, async (run) => {
      const input = wave(N * D);
      const weight = weights(D);
      const bias = biases(D);
      await expectAgrees(
        run,
        {
          code,
          bindings: [
            { kind: "storage", data: input },
            { kind: "storage", data: weight },
            { kind: "storage", data: bias },
            { kind: "out", type: "f32", length: N * D },
            { kind: "uniform", data: params([["u32", N], ["u32", D], ["f32", EPS]]) },
          ],
          workgroups: [N],
        },
        [layernorm({ input, weight, bias, N, D, eps: EPS })],
      );
    });
  }

  gpuTest("holds on rows whose mean dwarfs their spread", async (run) => {
    // This is the case the op exists to get right, and the reason the reference
    // takes two passes instead of using `var = E[x²] - E[x]²`.
    //
    // The rows are integers, so the f32 arithmetic up to the variance is exact
    // and nothing here is a tolerance argument: every element is `M + d` with
    // `d` a whole number in [-4, 4], each partial sum stays under 2^24 where f32
    // counts exactly, and `M` is a power of two so dividing the sum by D is
    // exact too. The true variance is 6.678162 for both rows.
    //
    // What the two formulations actually returned on this device, recovered
    // from the shader's own output rather than reasoned about:
    //   two-pass, M = 8192  ->  variance 6.67816, max relative error 2.9e-6
    //   two-pass, M = 1024  ->  variance 6.67816, max relative error 2.9e-6
    //   one-pass, M = 8192  ->  NaN. E[x²] lands past 2^26 where representable
    //                           values are 8 apart, every square rounds, and the
    //                           subtraction comes out *negative*; inverseSqrt of
    //                           that is NaN and the whole row is destroyed.
    //   one-pass, M = 1024  ->  variance 6.62347. Less spectacular and more
    //                           dangerous: 0.8% low, a number that looks
    //                           entirely reasonable and is wrong.
    // The 2.9e-6 is why the default tolerance stands rather than being widened.
    const D = 256;
    const deviation = (i: number) => ((i * 37) % 9) - 4;
    const input = Float32Array.from({ length: 2 * D }, (_, i) =>
      (i < D ? 8192 : 1024) + deviation(i % D),
    );
    const weight = weights(D);
    const bias = biases(D);
    await expectAgrees(
      run,
      {
        code,
        bindings: [
          { kind: "storage", data: input },
          { kind: "storage", data: weight },
          { kind: "storage", data: bias },
          { kind: "out", type: "f32", length: 2 * D },
          { kind: "uniform", data: params([["u32", 2], ["u32", D], ["f32", EPS]]) },
        ],
        workgroups: [2],
      },
      [layernorm({ input, weight, bias, N: 2, D, eps: EPS })],
    );
  });

  gpuTest("keeps eps meaningful on a constant row", async (run) => {
    // A row with no spread at all: the variance is exactly zero and eps is the
    // only thing between the reciprocal square root and a division by zero.
    // Without it the shader returns NaN where the answer is just the bias.
    const D = 8;
    const input = new Float32Array(D).fill(3.5);
    const weight = weights(D);
    const bias = biases(D);
    await expectAgrees(
      run,
      {
        code,
        bindings: [
          { kind: "storage", data: input },
          { kind: "storage", data: weight },
          { kind: "storage", data: bias },
          { kind: "out", type: "f32", length: D },
          { kind: "uniform", data: params([["u32", 1], ["u32", D], ["f32", EPS]]) },
        ],
        workgroups: [1],
      },
      [layernorm({ input, weight, bias, N: 1, D, eps: EPS })],
    );
  });

  gpuTest("reads no further than the last column of a row", async (run) => {
    // One row, deliberately. With two or more, an over-run on the first row
    // reads the second row's first element and every multi-row shape above
    // catches it; the last row is the one that cannot be caught that way,
    // because past it there is nothing. Measured on this device, a storage read
    // past the end of a buffer returns 0, and a stray zero is invisible here —
    // it changes neither the sum nor the sum of squares, and D comes from the
    // params rather than from the loop. So the buffer is longer than the
    // operand and the tail holds a value big enough that one leaked term moves
    // the mean well past the tolerance. Confirmed by mutation: widening the
    // reduction bound to `col <= params.D` fails this test with the padding and
    // passes it with the padding set to zero.
    //
    // D = 300 leaves the strided loop ragged, which is where an off-by-one in
    // the bound lives.
    const N = 1;
    const D = 300;
    const input = new Float32Array(N * D + 512).fill(1e3);
    input.set(wave(N * D));
    const weight = weights(D);
    const bias = biases(D);
    await expectAgrees(
      run,
      {
        code,
        bindings: [
          { kind: "storage", data: input },
          { kind: "storage", data: weight },
          { kind: "storage", data: bias },
          { kind: "out", type: "f32", length: N * D },
          { kind: "uniform", data: params([["u32", N], ["u32", D], ["f32", EPS]]) },
        ],
        workgroups: [N],
      },
      [layernorm({ input: input.subarray(0, N * D), weight, bias, N, D, eps: EPS })],
    );
  });

  gpuTest("leaves rows past N untouched", async (run) => {
    // The row guard is otherwise unreachable: dispatching exactly N workgroups
    // means no invocation ever sees a row past the end, and deleting the guard
    // changes nothing. Here one extra workgroup is dispatched, with the output
    // buffer sized for it, so an unguarded kernel writes a row of biases where
    // the buffer should have been left alone.
    const N = 2;
    const D = 64;
    const input = wave(N * D);
    const weight = weights(D);
    const bias = biases(D);
    const expected = new Float32Array((N + 1) * D);
    expected.set(layernorm({ input, weight, bias, N, D, eps: EPS }));
    await expectAgrees(
      run,
      {
        code,
        bindings: [
          { kind: "storage", data: input },
          { kind: "storage", data: weight },
          { kind: "storage", data: bias },
          { kind: "out", type: "f32", length: (N + 1) * D },
          { kind: "uniform", data: params([["u32", N], ["u32", D], ["f32", EPS]]) },
        ],
        workgroups: [N + 1],
      },
      [expected],
    );
  });
});
