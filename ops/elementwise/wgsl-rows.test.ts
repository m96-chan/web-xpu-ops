import { describe } from "vitest";
import { expectAgrees, gpuTest, kernel, params, useGpu } from "../../harness/index.js";
import { ELEMENTWISE, elementwiseRows } from "./reference.js";

/**
 * Issue #150's row-broadcast entry point, `wgsl/rows.wgsl`.
 *
 * Kept apart from `wgsl.test.ts` for two reasons. The dispatch budget is one —
 * #38/#68: this setup kills the vitest worker often enough above about five
 * dispatches in a file that `ops/rope` already had to split for it, and
 * `wgsl.test.ts` has two of its own. The other is that `wgsl.test.ts` covers
 * the same-shape kernel the LLM engines already bind, and issue #150's main
 * requirement is that that path did not move; leaving its file untouched is the
 * cheapest way to show it.
 */

const code = kernel(import.meta.url, "rows");

/**
 * Anything but zero, and not producible by any of the arithmetic below.
 *
 * This GPU returns 0 for reads past the end of a storage buffer and drops
 * writes past it, so buffers sized exactly to the data hide a missing bound
 * twice over. Padding the inputs with a live value and sizing the output for
 * the *whole* dispatch breaks both halves: a thread past `S*D` that was never
 * stopped reads a sentinel instead of a zero and writes it where the
 * expectation is zero.
 */
const SENTINEL = 1000;

/** Deterministic and mixed in sign, so a dropped or duplicated term cannot hide in a monotone result. */
const wave = (n: number, k: number, phase = 0) =>
  Float32Array.from({ length: n }, (_, i) => Math.sin(i * k + phase) * 2);

/** `wave`, but with 0 and 1 planted in it — the two values that make a broadcast invisible if they are all `b` has. */
function bias(D: number): Float32Array {
  const b = wave(D, 0.19, 0.8);
  b[0] = 0;
  b[1] = 1;
  b[D - 1] = -1;
  return b;
}

const padded = (data: Float32Array, extra = 256) =>
  Float32Array.from({ length: data.length + extra }, (_, i) => (i < data.length ? data[i]! : SENTINEL));

/** The dispatch covers whole workgroups, so the expectation has to cover them too — the pad is where an unguarded thread lands. */
function withTail(expected: Float32Array, outLength: number): Float32Array {
  const full = new Float32Array(outLength);
  full.set(expected);
  return full;
}

describe("elementwise rows / wgsl", () => {
  useGpu();

  // S*D = 301, deliberately not a multiple of the 256-wide workgroup, so the
  // tail guard runs; and D = 43 is not a multiple of anything the kernel could
  // mask with instead of the modulo it uses.
  const S = 7;
  const D = 43;

  for (const [label, kind] of [["add", ELEMENTWISE.add], ["multiply", ELEMENTWISE.multiply]] as const) {
    gpuTest(`agrees with the reference for ${label}`, async (run) => {
      const a = wave(S * D, 0.37);
      const b = bias(D);
      const outLength = Math.ceil((S * D) / 256) * 256;
      await expectAgrees(
        run,
        {
          code,
          bindings: [
            { kind: "storage", data: padded(a) },
            // `b` is padded too: a kernel that read it at the flat index rather
            // than at the column would run off the end for every row past the
            // first, and robust buffer access would quietly hand it 0.0 —
            // which for `add` is indistinguishable from a correct answer at the
            // columns where `b` happens to be 0.
            { kind: "storage", data: padded(b) },
            { kind: "out", type: "f32", length: outLength },
            { kind: "uniform", data: params([["u32", S], ["u32", D], ["u32", kind]]) },
          ],
          workgroups: [Math.ceil((S * D) / 256)],
        },
        [withTail(elementwiseRows({ a, b, S, D, kind }), outLength)],
      );
    });
  }

  /**
   * The case the others cannot cover, and the one issue #150 names: with
   * `S == D`, indexing `b` by the row instead of by the column is still in
   * bounds and still returns finite numbers everywhere. So is using `b[0]`
   * for everything. Both give a well-formed wrong answer that only a square
   * input with an `a` varying down the rows and a `b` varying across the
   * columns can separate — which is what this is.
   *
   * 17 x 17 = 289 also straddles one workgroup, so the tail guard is live here
   * as well.
   */
  gpuTest("indexes b by the column, not by the row", async (run) => {
    const N = 17;
    const a = wave(N * N, 0.41);
    const b = bias(N);
    const outLength = Math.ceil((N * N) / 256) * 256;
    await expectAgrees(
      run,
      {
        code,
        bindings: [
          { kind: "storage", data: padded(a) },
          { kind: "storage", data: padded(b) },
          { kind: "out", type: "f32", length: outLength },
          { kind: "uniform", data: params([["u32", N], ["u32", N], ["u32", ELEMENTWISE.add]]) },
        ],
        workgroups: [Math.ceil((N * N) / 256)],
      },
      [withTail(elementwiseRows({ a, b, S: N, D: N, kind: ELEMENTWISE.add }), outLength)],
    );
  });

  /**
   * The two degenerate shapes, one dispatch each, each carrying the `b` the
   * issue asks for.
   *
   * `S = 1` with an all-one `b` under multiply, and `D = 1` with an all-zero
   * `b` under add, are both the identity — and that is the point of pairing
   * them this way rather than testing an all-zero `b` under *multiply*: an
   * all-zero product expects an all-zero output, which a kernel that never ran
   * also produces, since the output buffer starts zeroed. Here the expectation
   * is `a` itself, a wave of order one, so a kernel that skipped these shapes
   * returns zeros and fails.
   */
  gpuTest("handles S=1 with an all-one b", async (run) => {
    const D1 = 301;
    const a = wave(D1, 0.29);
    const b = new Float32Array(D1).fill(1);
    const outLength = Math.ceil(D1 / 256) * 256;
    await expectAgrees(
      run,
      {
        code,
        bindings: [
          { kind: "storage", data: padded(a) },
          { kind: "storage", data: padded(b) },
          { kind: "out", type: "f32", length: outLength },
          { kind: "uniform", data: params([["u32", 1], ["u32", D1], ["u32", ELEMENTWISE.multiply]]) },
        ],
        workgroups: [Math.ceil(D1 / 256)],
      },
      [withTail(elementwiseRows({ a, b, S: 1, D: D1, kind: ELEMENTWISE.multiply }), outLength)],
    );
  });

  gpuTest("handles D=1 with an all-zero b", async (run) => {
    const S1 = 301;
    const a = wave(S1, 0.23);
    const b = new Float32Array([0]);
    const outLength = Math.ceil(S1 / 256) * 256;
    await expectAgrees(
      run,
      {
        code,
        bindings: [
          { kind: "storage", data: padded(a) },
          { kind: "storage", data: padded(b) },
          { kind: "out", type: "f32", length: outLength },
          { kind: "uniform", data: params([["u32", S1], ["u32", 1], ["u32", ELEMENTWISE.add]]) },
        ],
        workgroups: [Math.ceil(S1 / 256)],
      },
      [withTail(elementwiseRows({ a, b, S: S1, D: 1, kind: ELEMENTWISE.add }), outLength)],
    );
  });
});
