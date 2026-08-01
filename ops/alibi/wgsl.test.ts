import { describe } from "vitest";
import { expectAgrees, gpuTest, kernel, params, useGpu } from "../../harness/index.js";
import { alibi, alibiSlopes } from "./reference.js";

const biasCode = kernel(import.meta.url);

const WG = 256;
const wave = (n: number, k: number, phase = 0) =>
  Float32Array.from({ length: n }, (_, i) => Math.sin(i * k + phase) * 3);

/** Rounded up to the dispatch, so a write past the end of the result lands somewhere visible. */
const dispatchLength = (total: number) => Math.ceil(total / WG) * WG;

/** The reference, followed by the zeros the kernel must not have written over. */
const withUntouchedTail = (expected: Float32Array | Float64Array, length: number) => {
  const padded = new Float32Array(length);
  padded.set(expected);
  return padded;
};

/**
 * The bias application, on its own.
 *
 * The slopes are an *input* here rather than something this kernel derives, and
 * they are tested in `slopes.test.ts` against `reference.test.ts`'s pinned
 * numbers. That split is the point: a fault in the slopes must not be able to
 * hide inside a correct bias, or the other way round. It is also what keeps
 * each file's GPU test count low enough to report reliably — see the note in
 * `slopes.test.ts`.
 */
describe("alibi bias / wgsl", () => {
  useGpu();

  // [numHeads, M, N, posOffset, why]
  const SHAPES: [number, number, number, number, string][] = [
    [1, 1, 1, 0, "one element"],
    [1, 4, 4, 0, "one head, square"],
    [3, 5, 7, 0, "shorter than one workgroup"],
    [4, 8, 8, 0, "exactly one workgroup"],
    [2, 16, 16, 0, "exactly two workgroups"],
    [5, 7, 11, 0, "not a multiple of the workgroup"],
    [12, 3, 40, 0, "a head count that is not a power of two"],
    [8, 1, 37, 36, "decoding: one query row, its position past the keys"],
    [12, 4, 33, 29, "a KV-cache window, ragged and non-power-of-two heads"],
    [7, 9, 9, 3, "posOffset with a square block"],
  ];

  for (const [numHeads, M, N, posOffset, why] of SHAPES) {
    gpuTest(
      `adds the bias at heads=${numHeads} M=${M} N=${N} posOffset=${posOffset} — ${why}`,
      async (run) => {
        const scores = wave(numHeads * M * N, 0.31, 0.2);
        const slopes = Float32Array.from(alibiSlopes(numHeads));
        await expectAgrees(
          run,
          {
            code: biasCode,
            bindings: [
              { kind: "storage", data: scores },
              { kind: "storage", data: slopes },
              { kind: "out", type: "f32", length: scores.length },
              {
                kind: "uniform",
                data: params([
                  ["u32", numHeads], ["u32", M], ["u32", N], ["u32", posOffset],
                ]),
              },
            ],
            workgroups: [Math.ceil(scores.length / WG)],
          },
          [alibi({ scores, numHeads, M, N, posOffset, slopes })],
        );
      },
    );
  }

  gpuTest("does not write past the end of the score matrix", async (run) => {
    // Without this the bounds check cannot be observed at all. The result
    // buffer is exactly as long as the scores in every shape above, so a lane
    // past the end writes off the end of the buffer and this driver discards
    // it; deleting the guard changes nothing.
    //
    // Here the buffer is a whole dispatch long, so the stray write lands. The
    // inputs are padded too, and with a sentinel rather than left short: reads
    // past the end of a buffer return 0 on this device, and 0 + 0 * anything is
    // the zero the tail is being checked for. The sentinel makes the stray
    // write a number that cannot be mistaken for an untouched slot.
    const [numHeads, M, N, posOffset] = [3, 5, 9, 2];
    const total = numHeads * M * N; // 135, so the dispatch overhangs by 121
    const length = dispatchLength(total);
    const scores = new Float32Array(length).fill(1e4);
    scores.set(wave(total, 0.43, 1.1));
    const slopes = new Float32Array(length).fill(1e4);
    slopes.set(alibiSlopes(numHeads));

    await expectAgrees(
      run,
      {
        code: biasCode,
        bindings: [
          { kind: "storage", data: scores },
          { kind: "storage", data: slopes },
          { kind: "out", type: "f32", length },
          {
            kind: "uniform",
            data: params([["u32", numHeads], ["u32", M], ["u32", N], ["u32", posOffset]]),
          },
        ],
        workgroups: [Math.ceil(total / WG)],
      },
      [
        withUntouchedTail(
          alibi({
            scores: scores.subarray(0, total),
            numHeads,
            M,
            N,
            posOffset,
            slopes: slopes.subarray(0, numHeads),
          }),
          length,
        ),
      ],
    );
  });

  gpuTest("reads the slope of the head it is on, not the first one", async (run) => {
    // Every shape above uses the real slopes, which fall off geometrically, so
    // a kernel that indexed the wrong head would be wrong by a small amount on
    // a small bias. Slopes an order of magnitude apart per head, with zero
    // scores, make the head index the only thing the result depends on.
    const [numHeads, M, N, posOffset] = [4, 3, 5, 1];
    const scores = new Float32Array(numHeads * M * N);
    const slopes = Float32Array.from([1, 10, 100, 1000]);
    await expectAgrees(
      run,
      {
        code: biasCode,
        bindings: [
          { kind: "storage", data: scores },
          { kind: "storage", data: slopes },
          { kind: "out", type: "f32", length: scores.length },
          {
            kind: "uniform",
            data: params([["u32", numHeads], ["u32", M], ["u32", N], ["u32", posOffset]]),
          },
        ],
        workgroups: [Math.ceil(scores.length / WG)],
      },
      [alibi({ scores, numHeads, M, N, posOffset, slopes })],
    );
  });
});
