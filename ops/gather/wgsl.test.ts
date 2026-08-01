import { describe } from "vitest";
import { expectAgrees, gpuTest, kernel, params, useGpu } from "../../harness/index.js";
import { gather } from "./reference.js";

const code = kernel(import.meta.url);
const wave = (n: number, k = 0.37) => Float32Array.from({ length: n }, (_, i) => Math.sin(i * k) * 2);

/**
 * Scattered, repeating, and never starting at 0 — an index list that happens to
 * be the identity would let a kernel that ignores `indices` entirely pass.
 */
const pick = (n: number, rows: number) => Int32Array.from({ length: n }, (_, i) => (i * 7 + 3) % rows);

const dispatch = (total: number): [number] => [Math.ceil(total / 256)];

describe("gather / wgsl", () => {
  useGpu();

  // The flat thread mapping means the workgroup boundary falls on N*D, so the
  // cases span it: below it, exactly on it, an exact multiple of it, and not a
  // multiple of it. D crosses 256 separately, because the row width is what
  // decides whether one workgroup covers one output row or many.
  for (const [N, D, rows] of [
    [3, 8, 5], // total 24 — below one workgroup
    [1, 256, 4], // total 256 — exactly one
    [2, 256, 64], // total 512 — an exact multiple
    [5, 300, 7], // total 1500 — not a multiple; D wider than a workgroup
    [257, 1, 33], // total 257 — one thread past the boundary
    [300, 7, 128], // total 2100 — not a multiple, many rows per workgroup
  ] as const) {
    gpuTest(`agrees with the reference at N=${N} D=${D} rows=${rows}`, async (run) => {
      const table = wave(rows * D);
      const indices = pick(N, rows);
      await expectAgrees(
        run,
        {
          code,
          bindings: [
            { kind: "storage", data: table },
            { kind: "storage", data: indices },
            { kind: "out", type: "f32", length: N * D },
            { kind: "uniform", data: params([["u32", N], ["u32", D], ["u32", rows]]) },
          ],
          workgroups: dispatch(N * D),
        },
        [gather({ table, indices, rows, D })],
      );
    });
  }

  gpuTest("gathers one row for every index", async (run) => {
    // The worst case the access pattern can produce: every lane reads the same
    // table row. Nobody controls the index data, so this shape is not a corner
    // case, it is what a repeated token looks like.
    const [N, D, rows] = [512, 3, 100];
    const table = wave(rows * D);
    const indices = new Int32Array(N).fill(57);
    await expectAgrees(
      run,
      {
        code,
        bindings: [
          { kind: "storage", data: table },
          { kind: "storage", data: indices },
          { kind: "out", type: "f32", length: N * D },
          { kind: "uniform", data: params([["u32", N], ["u32", D], ["u32", rows]]) },
        ],
        workgroups: dispatch(N * D),
      },
      [gather({ table, indices, rows, D })],
    );
  });

  gpuTest("gathers zeros for indices outside the table", async (run) => {
    // PyTorch raises here and a kernel cannot, so the reference picks zeros and
    // this is what holds the kernel to that choice. Without it, clamping and
    // wrapping both pass, and both hand back a plausible embedding for a token
    // that does not exist.
    //
    // The table buffer deliberately holds more rows than `rows` claims. With a
    // buffer sized exactly to `rows`, an index just past the end reads out of
    // bounds, WebGPU hands back zero, and dropping the upper bound from the
    // kernel passes anyway — the guard would be untestable. Spare rows put real
    // values where a missing bound would reach.
    const [D, rows, allocated] = [4, 6, 9];
    const table = wave(allocated * D);
    const indices = Int32Array.from([0, -1, 6, 7, 3, -100, 2, 1000]);
    const N = indices.length;
    await expectAgrees(
      run,
      {
        code,
        bindings: [
          { kind: "storage", data: table },
          { kind: "storage", data: indices },
          { kind: "out", type: "f32", length: N * D },
          { kind: "uniform", data: params([["u32", N], ["u32", D], ["u32", rows]]) },
        ],
        workgroups: dispatch(N * D),
      },
      [gather({ table, indices, rows, D })],
    );
  });

  gpuTest("leaves the tail of the output buffer alone", async (run) => {
    // N*D is not a multiple of the workgroup size, so the last workgroup has
    // threads with nothing to gather. Asking for an output buffer as long as the
    // dispatch — rather than as long as the result — is what makes their writes
    // observable: WebGPU zeroes a fresh buffer, and discards or clamps a write
    // past the end, so a bounds check tested against an exactly sized buffer is
    // a bounds check nothing can fail.
    const [N, D, rows] = [3, 100, 8];
    const total = N * D;
    const padded = dispatch(total)[0] * 256;
    const table = wave(rows * D);
    const indices = pick(N, rows);
    const expected = new Float32Array(padded);
    expected.set(gather({ table, indices, rows, D }));
    await expectAgrees(
      run,
      {
        code,
        bindings: [
          { kind: "storage", data: table },
          { kind: "storage", data: indices },
          { kind: "out", type: "f32", length: padded },
          { kind: "uniform", data: params([["u32", N], ["u32", D], ["u32", rows]]) },
        ],
        workgroups: dispatch(total),
      },
      [expected],
    );
  });
});
