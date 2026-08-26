import { describe } from "vitest";
import { expectAgrees, gpuTest, kernel, params, useGpu, type Dispatch } from "../../harness/index.js";
import { pad, padOutputLength, type PadMode } from "./reference.js";

const code = kernel(import.meta.url, "kernel");
const WORKGROUP = 256;

/**
 * A value that is not zero, parked past the end of the input buffer.
 *
 * `constant` mode writes zeros into the padded positions, and a read past the
 * end of a storage buffer returns zero on this device — so a kernel that
 * forgot the mode entirely and just read `input[source]` would produce the
 * right answer for `constant` by accident. The sentinel makes that read land on
 * a number instead.
 */
const SENTINEL = 7.25;
const SLACK = 64;

const wave = (n: number, k: number) => Float32Array.from({ length: n }, (_, i) => Math.sin(i * k) * 1.5 + 0.25);

function withTail(data: Float32Array): Float32Array {
  const buffer = new Float32Array(data.length + SLACK).fill(SENTINEL);
  buffer.set(data);
  return buffer;
}

interface Case {
  name: string;
  outer: number;
  L: number;
  inner: number;
  before: number;
  after: number;
  mode: PadMode;
  value?: number;
}

const CASES: Case[] = [
  // The 1D shapes, one per mode. `inner = 1` is the case the flat dispatch
  // exists for: a mapping with x over `inner` would run one useful thread in
  // every 256 here.
  { name: "1D constant", outer: 3, L: 700, inner: 1, before: 5, after: 9, mode: "constant" },
  { name: "1D constant, non-zero value", outer: 3, L: 700, inner: 1, before: 5, after: 9, mode: "constant", value: -2.5 },
  { name: "1D replicate", outer: 3, L: 700, inner: 1, before: 5, after: 9, mode: "replicate" },
  { name: "1D reflect", outer: 3, L: 700, inner: 1, before: 5, after: 9, mode: "reflect" },
  // One end only — the causal shape, and the one where a kernel that used
  // `before` for both ends still gets the right length and the wrong data.
  { name: "1D replicate, before only", outer: 4, L: 513, inner: 1, before: 6, after: 0, mode: "replicate" },
  { name: "1D constant, after only", outer: 4, L: 513, inner: 1, before: 0, after: 6, mode: "constant" },
  // Reflect at the widest torch allows, so the mirrored index reaches the far
  // edge of the axis exactly.
  { name: "1D reflect at the maximum width", outer: 2, L: 64, inner: 1, before: 63, after: 63, mode: "reflect" },
  // `inner > 1`: each padded position carries a whole row, which is the H and D
  // axes of a video tensor. A kernel that dropped `inner` from either index
  // would still fill the buffer.
  { name: "inner = 64 rows, reflect", outer: 5, L: 9, inner: 64, before: 2, after: 3, mode: "reflect" },
  { name: "inner = 65 rows, replicate (inner not a multiple of the workgroup)", outer: 5, L: 9, inner: 65, before: 2, after: 3, mode: "replicate" },
  // Exactly one workgroup of output, so no thread is surplus at all.
  { name: "output exactly a workgroup", outer: 1, L: 250, inner: 1, before: 3, after: 3, mode: "reflect" },
  // Nothing to pad: every element passes through, in every mode.
  { name: "no padding at all", outer: 4, L: 300, inner: 3, before: 0, after: 0, mode: "reflect" },
  // Large enough that the flat range needs more than one row of the (x, y)
  // tile — this is what `stride_y` is for, and a wrong `stride_y` reads and
  // writes the wrong elements everywhere past the first row.
  { name: "flat range spans several tile rows", outer: 40, L: 17, inner: 512, before: 4, after: 4, mode: "replicate" },
];

/** The x extent in threads; `gid.y` then advances the flat index by that much. */
const TILE_X = 256 * 256;

function prepare(shape: Case): { name: string; dispatch: Dispatch; expected: Float32Array } {
  const { name, outer, L, inner, before, after, mode, value = 0 } = shape;
  const input = wave(outer * L * inner, 0.37);
  const result = pad({ input, outer, L, inner, before, after, mode, value });

  const total = outer * padOutputLength({ L, before, after }) * inner;
  const rows = Math.ceil(total / TILE_X);
  const MODE = { constant: 0, reflect: 1, replicate: 2 }[mode];

  // The output buffer is the *dispatch* extent, not the result length, with the
  // surplus expected to stay zero.
  //
  // This is not neatness. With an exactly-sized buffer the `flat >= total`
  // guard could be deleted and every test still passed: a write past the end of
  // a storage buffer is dropped by the driver, so the surplus threads had
  // nowhere to land and nothing to disturb. Measured — that mutation survived
  // the first version of this file. Giving them somewhere to land is what makes
  // the guard observable.
  const groupsX = Math.min(TILE_X, Math.ceil(total / WORKGROUP) * WORKGROUP) / WORKGROUP;
  const dispatchThreads = groupsX * WORKGROUP * rows;
  const expected = new Float32Array(dispatchThreads);
  expected.set(result);

  return {
    name,
    expected,
    dispatch: {
      code,
      bindings: [
        { kind: "storage", data: withTail(input) },
        { kind: "out", type: "f32", length: expected.length },
        {
          kind: "uniform",
          data: params([
            ["u32", outer],
            ["u32", L],
            ["u32", inner],
            ["u32", before],
            ["u32", padOutputLength({ L, before, after })],
            ["u32", MODE],
            ["f32", value],
            ["u32", TILE_X],
          ]),
        },
      ],
      workgroups: [groupsX, rows, 1],
    },
  };
}

const PREPARED = CASES.map(prepare);

/**
 * Exact.
 *
 * A pad performs no arithmetic on the data — every output element is either a
 * copy of an input element or the fill value. There is nothing for f32 to round,
 * so any difference at all is a wrong index or a wrong mode, and a tolerance
 * would only hide it.
 */
const TOLERANCE = { abs: 0, rel: 0 };

describe("pad / wgsl", () => {
  useGpu();

  for (const { name, dispatch, expected } of PREPARED) {
    gpuTest(`agrees with the reference: ${name}`, async (run) => {
      await expectAgrees(run, dispatch, [expected], TOLERANCE);
    });
  }
});
