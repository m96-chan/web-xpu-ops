import { describe } from "vitest";
import { expectAgrees, gpuTest, kernel, params, useGpu } from "../../harness/index.js";
import { pope } from "./reference.js";

const code = kernel(import.meta.url);

const WG = 256;
const dispatchLength = (total: number) => Math.ceil(total / WG) * WG;

describe("pope / wgsl", () => {
  useGpu();

  // Measured, not widened until green. The kernel and the reference evaluate
  // the same recurrence, so the whole difference is f32 against f64 over
  // `order` steps of a multiply-and-subtract. The error therefore grows with
  // the polynomial order and with nothing else — not with dModel, not with the
  // dispatch shape.
  //
  // Measured on this device by running the shapes below and reading back the
  // worst absolute disagreement:
  //
  //   max order   4   (N=5,   dModel=17, off=0):  8.9e-8
  //   max order   4   (N=3,   dModel=512, off=2): 2.4e-7   (width changes nothing)
  //   max order  16   (N=16,  dModel=32, off=1):  1.5e-7
  //   max order  30   (N=24,  dModel=48, off=7):  1.4e-6
  //   max order  70   (N=64,  dModel=16, off=7):  4.1e-6
  //   max order 128   (not tested below)          1.6e-5
  //
  // 2e-5 is ~5x the worst observed over the orders these tests actually reach.
  // It is a ceiling for this order range, not a general one: at order 128 the
  // f32 recurrence is already at 1.6e-5 on its own, so a test that went deeper
  // would have to raise this and say so.
  //
  // It is not slack either. Values here are bounded by 1 (|P_n(x)| <= 1 on the
  // domain), so an indexing or recurrence fault is wrong by O(1) — five orders
  // of magnitude clear of this — and that is checked by mutation, not assumed.
  //
  // `rel` is left at the harness default and carries nothing here: the
  // polynomials cross zero repeatedly and relative error at a crossing means
  // nothing. `abs` is the measure doing the work.
  const RECURRENCE = { abs: 2e-5 };

  // [N, dModel, posOffset, why]
  const SHAPES: [number, number, number, string][] = [
    [1, 1, 0, "one element, the constant polynomial"],
    [1, 1, 1, "one element, the linear polynomial"],
    [4, 4, 0, "the first four orders, sampled four ways"],
    [8, 32, 0, "exactly one workgroup"],
    [16, 32, 1, "exactly two workgroups, paper's 1-based positions"],
    [5, 17, 0, "not a multiple of the workgroup"],
    [24, 48, 7, "a KV-cache continuation, ragged"],
    [64, 16, 7, "orders past 70, where the f32 recurrence is furthest from f64"],
    [3, 512, 2, "a row wider than the workgroup"],
  ];

  for (const [N, dModel, posOffset, why] of SHAPES) {
    gpuTest(`agrees with the reference at N=${N} dModel=${dModel} posOffset=${posOffset} — ${why}`, async (run) => {
      const total = N * dModel;
      await expectAgrees(
        run,
        {
          code,
          bindings: [
            { kind: "out", type: "f32", length: total },
            {
              kind: "uniform",
              data: params([["u32", N], ["u32", dModel], ["u32", posOffset]]),
            },
          ],
          workgroups: [Math.ceil(total / WG)],
        },
        [pope({ N, dModel, posOffset })],
        RECURRENCE,
      );
    });
  }

  gpuTest("does not write past the end of the table", async (run) => {
    // The shapes above allocate the result at exactly N*dModel, so a lane past
    // the end writes off the end of the buffer and this driver drops it —
    // deleting the guard changes nothing they can see.
    //
    // A whole dispatch of room makes the stray write land. It cannot be
    // mistaken for an untouched slot: the first lane past the end evaluates
    // P_(N+posOffset) at x = -1, which is exactly +/-1.
    const [N, dModel, posOffset] = [5, 9, 1];
    const total = N * dModel; // 45, so the dispatch overhangs by 211
    const length = dispatchLength(total);
    const expected = new Float32Array(length);
    expected.set(pope({ N, dModel, posOffset }));

    await expectAgrees(
      run,
      {
        code,
        bindings: [
          { kind: "out", type: "f32", length },
          {
            kind: "uniform",
            data: params([["u32", N], ["u32", dModel], ["u32", posOffset]]),
          },
        ],
        workgroups: [Math.ceil(total / WG)],
      },
      [expected],
      RECURRENCE,
    );
  });
});
