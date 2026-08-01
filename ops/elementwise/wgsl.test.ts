import { describe } from "vitest";
import { expectAgrees, gpuTest, kernel, params, useGpu } from "../../harness/index.js";
import { ELEMENTWISE, elementwise } from "./reference.js";

const code = kernel(import.meta.url);
const wave = (n: number, k = 0.37) => Float32Array.from({ length: n }, (_, i) => Math.sin(i * k) * 2);

describe("elementwise / wgsl", () => {
  useGpu();

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
