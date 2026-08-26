import { describe } from "vitest";
import { expectAgrees, gpuTest, kernel, params, useGpu } from "../../harness/index.js";
import { ELEMENTWISE, elementwise } from "./reference.js";

const code = kernel(import.meta.url);
const wave = (n: number, k = 0.37) => Float32Array.from({ length: n }, (_, i) => Math.sin(i * k) * 2);

describe("elementwise / wgsl", () => {
  useGpu();

  /**
   * A dispatch too wide for one row of workgroups.
   *
   * 65,535 workgroups is the ceiling on every backend measured (#211), which at
   * 256 threads is 16.7 M elements — `examples/h3-encoder`'s first level passes
   * it on a 256x256 reference with five frames. The fold uses `num_workgroups`
   * rather than a uniform, so **every existing one-dimensional caller keeps
   * working unchanged**: at `[n]` the y extent is 1 and `gid.y` is 0. Narrow on
   * purpose, so most of the work lands on rows the unfolded kernel never
   * reaches.
   */
  gpuTest("folds a two-dimensional dispatch back into one index", async (run) => {
    const n = 5000;
    const a = wave(n);
    const b = wave(n, 0.19);
    const x = 4;
    const y = Math.ceil(Math.ceil(n / 256) / x);
    await expectAgrees(
      run,
      {
        code,
        bindings: [
          { kind: "storage", data: a },
          { kind: "storage", data: b },
          { kind: "out", type: "f32", length: n },
          { kind: "uniform", data: params([["u32", n], ["u32", ELEMENTWISE.add]]) },
        ],
        workgroups: [x, y],
      },
      [elementwise({ a, b, kind: ELEMENTWISE.add })],
      { rel: 0, abs: 0 },
    );
  });

  for (const [label, kind] of [["add", ELEMENTWISE.add], ["multiply", ELEMENTWISE.multiply]] as const) {
    gpuTest(`agrees with the reference for ${label}`, async (run) => {
      // 300 is not a multiple of the 256-wide workgroup, so the tail guard runs.
      const a = wave(300);
      const b = wave(300, 0.19);
      await expectAgrees(
        run,
        {
          code,
          bindings: [
            { kind: "storage", data: a },
            { kind: "storage", data: b },
            { kind: "out", type: "f32", length: a.length },
            { kind: "uniform", data: params([["u32", a.length], ["u32", kind]]) },
          ],
          workgroups: [Math.ceil(a.length / 256)],
        },
        [elementwise({ a, b, kind })],
      );
    });
  }
});
