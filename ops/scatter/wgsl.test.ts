import { describe } from "vitest";
import { expectAgrees, gpuTest, kernel, params, useGpu } from "../../harness/index.js";
import { scatter } from "./reference.js";

const code = kernel(import.meta.url);

/** Positive values only, so a collided sum measures rounding and not cancellation. */
const values = (n: number) => Float32Array.from({ length: n }, (_, i) => 1 + Math.sin(i * 0.37) * 0.5);

/** `slot -> (slot * 7 + 3) mod S`, a permutation whenever 7 does not divide S. */
const permutation = (N: number, S: number) =>
  Int32Array.from({ length: N * S }, (_, i) => ((i % S) * 7 + 3) % S);

const exact = { rel: 0, abs: 0 };

const dispatch = (src: Float32Array, index: Int32Array, N: number, S: number, D: number) => ({
  code,
  bindings: [
    { kind: "storage" as const, data: src },
    { kind: "storage" as const, data: index },
    { kind: "out" as const, type: "f32" as const, length: N * D },
    { kind: "uniform" as const, data: params([["u32", N], ["u32", S], ["u32", D]]) },
  ],
  workgroups: [N] as [number],
});

describe("scatter / wgsl", () => {
  useGpu();

  // S spans the 256-wide workgroup deliberately: below it, exactly on it, and
  // not a multiple of it. Each thread walks its row with a stride of 256, so
  // only the middle case leaves no thread idle on the final step.
  for (const [N, S] of [[2, 8], [1, 256], [3, 300]] as const) {
    gpuTest(`agrees with the reference at N=${N} S=${S} with no collisions`, async (run) => {
      // A permutation, so every value lands in its own slot and the result is
      // a pure permutation of the input — bit-exact, no summation involved.
      const D = S;
      const src = values(N * S);
      const index = permutation(N, S);
      await expectAgrees(run, dispatch(src, index, N, S, D), [scatter({ src, index, N, S, D })], exact);
    });
  }

  // The decision this op exists to pin down: collisions accumulate. A kernel
  // that stored instead of adding would leave one of the 75 values in each
  // slot rather than their sum, and every one of these cases would fail.
  for (const [N, S, D] of [[1, 8, 1], [2, 256, 64], [2, 300, 4]] as const) {
    gpuTest(`accumulates ${S / D} collisions per slot at N=${N} S=${S} D=${D}`, async (run) => {
      const src = values(N * S);
      const index = Int32Array.from({ length: N * S }, (_, i) => (i % S) % D);
      await expectAgrees(
        run,
        dispatch(src, index, N, S, D),
        [scatter({ src, index, N, S, D })],
        // Not exact, and the reason is the semantics rather than the hardware:
        // the reference sums a slot in slot order, the GPU in whatever order
        // its threads win the atomic, and f32 addition is not associative.
        //
        // Measured, by running these same three cases at { rel: 0, abs: 0 }
        // four times on this GPU: 8 collisions agreed bit-for-bit every run,
        // 4 collisions differed by at most 1.25e-7 (stable across runs), and
        // 75 collisions by 2.0e-7, 3.0e-7, 4.0e-7 and 4.1e-7 — it varies
        // because the winning order does. That is 1 to 4 f32 epsilons
        // (1.19e-7) against a worst-case bound of 74 of them, the errors
        // being as likely to cancel as to compound. 2e-6 is five times the
        // worst seen: enough that another device's ordering does not turn
        // this red, far too little to hide a wrong sum.
        { rel: 2e-6, abs: 0 },
      );
    });
  }

  gpuTest("drops out-of-range indices instead of writing into another row", async (run) => {
    // Unchecked, `row * D + u32(column)` stays inside the flat buffer and lands
    // in a neighbouring row, which is why dropping is a correctness matter and
    // not tidiness. Which row it lands in decides where each kind of bad index
    // has to sit to be observable at all:
    //
    // - Too large, in row 0: column D lands on row 1 column 0. Visible.
    // - Negative, in row 1: u32(-1) wraps, and 8 + 0xFFFFFFFF is 7 — row 0's
    //   last column. Visible. A negative in row 0 would address past the end
    //   of the buffer instead, and out-of-bounds writes are the implementation's
    //   business, so that placement tests nothing. This one was measured: with
    //   the negatives in row 0, deleting `column < 0` from the kernel left the
    //   test green.
    //
    // Negatives being out of range at all is PyTorch's rule — `scatter_add_`
    // does not take Python-style negative indexing.
    const [N, S, D] = [2, 8, 8];
    const src = values(N * S);
    const index = Int32Array.from([
      0, 1, 2, 8, 4, 100, 6, 7, // row 0: two too large, and they aim at row 1
      0, -1, 2, 3, 4, 5, 6, -7, // row 1: two negative, and they aim at row 0
    ]);
    await expectAgrees(run, dispatch(src, index, N, S, D), [scatter({ src, index, N, S, D })], exact);
  });

  gpuTest("leaves untouched columns at zero", async (run) => {
    // The op starts from an implicitly zero `self`, so a column no index names
    // must read back as 0 rather than as whatever the buffer held.
    const [N, S, D] = [2, 8, 32];
    const src = values(N * S);
    const index = Int32Array.from({ length: N * S }, (_, i) => (i % S) * 2);
    await expectAgrees(run, dispatch(src, index, N, S, D), [scatter({ src, index, N, S, D })], exact);
  });
});
