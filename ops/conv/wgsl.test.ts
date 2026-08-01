import { describe } from "vitest";
import { expectAgrees, gpuTest, kernel, params, useGpu, type Dispatch } from "../../harness/index.js";
import { conv1d, conv1dOutputLength } from "./reference.js";

const code = kernel(import.meta.url);
const WORKGROUP = 256;

/**
 * A value that is not zero, parked past the end of every input buffer.
 *
 * This device returns 0 for a read past the end of a storage buffer, which
 * makes a missing padding guard invisible: the guard's whole job is to
 * contribute zero, and the out-of-range read already does. Filling the slack
 * with something else means an unguarded read lands on a number that changes
 * the answer.
 */
const SENTINEL = 7.25;
const SLACK = 64;

const wave = (n: number, k: number, phase = 0) =>
  Float32Array.from({ length: n }, (_, i) => Math.sin(i * k + phase) * 1.5 + 0.25);

/** Real data followed by sentinel slack, so an over-read is observable. */
function withTail(data: Float32Array): Float32Array {
  const buffer = new Float32Array(data.length + SLACK).fill(SENTINEL);
  buffer.set(data);
  return buffer;
}

interface Case {
  name: string;
  N: number;
  Cin: number;
  Cout: number;
  L: number;
  K: number;
  stride: number;
  padding: number;
  dilation: number;
  groups: number;
}

// Lout spans the 256-wide workgroup deliberately: well below it, one under it,
// exactly on it, and two sizes that are not multiples. One thread per output
// element means the guard on the ragged tail only has anything to guard when
// Lout is not a multiple of 256.
//
//   Lout = floor((L + 2*padding - dilation*(K-1) - 1) / stride) + 1
const CASES: Case[] = [
  { name: "plain, Lout=8", N: 1, Cin: 1, Cout: 1, L: 10, K: 3, stride: 1, padding: 0, dilation: 1, groups: 1 },
  { name: "padded, Lout=255", N: 3, Cin: 2, Cout: 2, L: 763, K: 7, stride: 3, padding: 3, dilation: 1, groups: 1 },
  { name: "padded, Lout=256", N: 2, Cin: 3, Cout: 4, L: 256, K: 5, stride: 1, padding: 2, dilation: 1, groups: 1 },
  {
    name: "strided + dilated + grouped, Lout=300",
    N: 2, Cin: 4, Cout: 6, L: 599, K: 3, stride: 2, padding: 2, dilation: 2, groups: 2,
  },
  {
    name: "depthwise, Lout=777",
    N: 1, Cin: 3, Cout: 3, L: 778, K: 4, stride: 1, padding: 1, dilation: 1, groups: 3,
  },
  // L=4, K=3, padding=2: the first two and last two windows are mostly pad, so
  // nearly every output element depends on the guard rather than on the data.
  // Cin=2 and N=2 matter — with more than one channel row, a window that runs
  // off the left of its own row lands on the *previous* row's tail, which is
  // real data rather than the zero an out-of-range read returns. With a single
  // row the left-edge guard has nothing to disagree with.
  {
    name: "pad dominates the window, both edges",
    N: 2, Cin: 2, Cout: 3, L: 4, K: 3, stride: 1, padding: 2, dilation: 1, groups: 1,
  },
  // dilation=4 with K=4 reaches 12 past the window start, so the last outputs
  // read well beyond L. Past the final channel that is the sentinel slack, not
  // zero, so a missing right-edge guard shows up as a wrong number rather than
  // as the right one.
  {
    name: "dilated window overruns the right edge",
    N: 1, Cin: 2, Cout: 2, L: 20, K: 4, stride: 1, padding: 6, dilation: 4, groups: 1,
  },
];

/**
 * Everything expensive is built here, at module scope; the tests below only
 * dispatch and compare.
 *
 * Not a style choice. Measured on this machine (#49): once `useGpu`'s
 * `beforeAll` has created the device, a test that takes even ~10ms before its
 * first dispatch kills the vitest worker with `std::system_error` and no result
 * reported at all. The same work at module scope, before the device exists, is
 * fine at a full second. This reference is four nested loops over
 * `N*Cout*Lout*(Cin/groups)*K` terms, which is exactly the shape that trips it.
 *
 * Two further things here are load-bearing rather than convenience, and both
 * exist because this device is quiet about the edges:
 *
 * - Every input buffer carries sentinel slack; see SENTINEL above.
 * - The output buffer is the *dispatch* length, not the result length, with the
 *   surplus expected to stay zero. A write past the end of an exactly-sized
 *   buffer is dropped by the driver, so the `ol >= Lout` guard could otherwise
 *   be deleted and the suite would stay green. With the slack allocated, the
 *   last row's surplus threads have somewhere real to land.
 */
function prepare(shape: Case): { name: string; dispatch: Dispatch; expected: Float32Array } {
  const { name, N, Cin, Cout, L, K, stride, padding, dilation, groups } = shape;
  const Lout = conv1dOutputLength({ L, K, stride, padding, dilation });
  const input = wave(N * Cin * L, 0.37);
  const weight = wave(Cout * (Cin / groups) * K, 0.61, 1.1);
  // Non-zero everywhere on purpose: bias is the one term a surplus thread adds
  // no matter what it reads, so it is what makes the tail guard observable.
  const bias = Float32Array.from({ length: Cout }, (_, i) => 0.5 + i * 0.25);

  const dispatchLength = Math.ceil(Lout / WORKGROUP) * WORKGROUP;
  const expected = new Float32Array(N * Cout * Lout + (dispatchLength - Lout));
  expected.set(conv1d({ input, weight, bias, N, Cin, Cout, L, K, stride, padding, dilation, groups }));

  return {
    name,
    expected,
    dispatch: {
      code,
      bindings: [
        { kind: "storage", data: withTail(input) },
        { kind: "storage", data: withTail(weight) },
        { kind: "storage", data: withTail(bias) },
        { kind: "out", type: "f32", length: expected.length },
        {
          kind: "uniform",
          data: params([
            ["u32", Cin],
            ["u32", Cout],
            ["u32", L],
            ["u32", K],
            ["u32", Lout],
            ["u32", stride],
            ["u32", padding],
            ["u32", dilation],
            ["u32", Cin / groups],
            ["u32", Cout / groups],
            ["u32", 0],
            ["u32", 0],
          ]),
        },
      ],
      workgroups: [dispatchLength / WORKGROUP, Cout, N],
    },
  };
}

const PREPARED = CASES.map(prepare);

describe("conv1d / wgsl", () => {
  useGpu();

  for (const { name, dispatch, expected } of PREPARED) {
    gpuTest(`agrees with the reference: ${name}`, async (run) => {
      await expectAgrees(run, dispatch, [expected]);
    });
  }
});
