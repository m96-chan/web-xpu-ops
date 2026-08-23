import { describe } from "vitest";
import { expectAgrees, gpuTest, kernel, params, useGpu, type Dispatch } from "../../harness/index.js";
import { conv2d, conv2dOutputSize } from "./reference.js";

const code = kernel(import.meta.url, "conv2d");
const WORKGROUP = 256;

/**
 * A value that is not zero, parked past the end of every input buffer.
 *
 * Same reason as `wgsl.test.ts`: this device returns 0 for a read past the end
 * of a storage buffer, which makes a missing padding guard invisible — the
 * guard's whole job is to contribute zero, and the out-of-range read already
 * does. Filling the slack with something else means an unguarded read lands on
 * a number that changes the answer.
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
  H: number;
  W: number;
  KH: number;
  KW: number;
  stride: readonly [number, number];
  padding: readonly [number, number];
  dilation: readonly [number, number];
  groups: number;
}

// `Wout` spans the 256-wide workgroup deliberately — one thread per output
// element means the ragged-tail guard has nothing to guard when `Wout` is a
// multiple of 256. Every case that is not about the tail keeps `Wout` small, so
// the reference (six nested loops) stays cheap enough to build at module scope.
//
//   Hout = floor((H + 2*padH - dilationH*(KH-1) - 1) / strideH) + 1, and the
//   same on W. Written out per case below.
const CASES: Case[] = [
  // Wout = 5, Hout = 3. The plain shape, and the one an image decoder actually
  // runs everywhere: 3x3, stride 1, padding 1 would be square — this one is
  // padding 0 and a non-square input, so a kernel that swapped H and W would
  // read a different element rather than the same one.
  {
    name: "plain 3x3, Hout=3 Wout=5",
    N: 1, Cin: 1, Cout: 1, H: 5, W: 7, KH: 3, KW: 3,
    stride: [1, 1], padding: [0, 0], dilation: [1, 1], groups: 1,
  },
  // Wout = 255: one thread short of a full workgroup, so the tail guard has to
  // stop exactly one surplus thread.
  {
    name: "one under a workgroup, Wout=255",
    N: 1, Cin: 2, Cout: 2, H: 4, W: 257, KH: 3, KW: 3,
    stride: [1, 1], padding: [0, 0], dilation: [1, 1], groups: 1,
  },
  // Wout = 256: exactly a workgroup, so no thread is surplus at all.
  {
    name: "exactly a workgroup, Wout=256",
    N: 2, Cin: 3, Cout: 4, H: 3, W: 258, KH: 3, KW: 3,
    stride: [1, 1], padding: [0, 0], dilation: [1, 1], groups: 1,
  },
  // Every knob off its default and different on the two axes, N > 1 and
  // Cout > 1 together (the dispatch packs both into z, so a kernel that split
  // z the other way round would still be in range and still be wrong), and
  // Wout = 301 leaves a ragged tail of 211 threads.
  {
    name: "strided + dilated + grouped, non-square kernel, Hout=4 Wout=301",
    N: 2, Cin: 4, Cout: 6, H: 9, W: 300, KH: 3, KW: 2,
    stride: [2, 1], padding: [1, 2], dilation: [2, 3], groups: 2,
  },
  // Depthwise, groups = Cin = Cout: each output channel sees exactly one input
  // channel, so a kernel that mixed them would read the neighbouring plane.
  {
    name: "depthwise, Hout=6 Wout=776",
    N: 1, Cin: 3, Cout: 3, H: 6, W: 777, KH: 2, KW: 4,
    stride: [1, 1], padding: [1, 1], dilation: [1, 1], groups: 3,
  },
  // 4x4 input, 3x3 kernel, padding 2 on both axes: nearly every output element
  // depends on a guard rather than on data. Cin = 2 and N = 2 matter — with
  // more than one plane, a window that runs off the top of its own plane lands
  // on the *previous* plane's rows, which is real data rather than the zero an
  // out-of-range read returns, and a window that runs off the left of a row
  // lands on the previous row. With a single plane neither guard has anything
  // to disagree with.
  {
    name: "pad dominates the window, all four edges",
    N: 2, Cin: 2, Cout: 3, H: 4, W: 4, KH: 3, KW: 3,
    stride: [1, 1], padding: [2, 2], dilation: [1, 1], groups: 1,
  },
  // dilation (2, 4) with a 4x4 kernel reaches 6 rows and 12 columns past the
  // window start, so the last outputs read well beyond H and W. Past the final
  // plane that is the sentinel slack, not zero, so a missing bottom/right guard
  // shows up as a wrong number rather than as the right one.
  {
    name: "dilated window overruns the bottom and right edges",
    N: 1, Cin: 2, Cout: 2, H: 6, W: 20, KH: 4, KW: 4,
    stride: [1, 1], padding: [3, 6], dilation: [2, 4], groups: 1,
  },
];

/**
 * Everything expensive is built here, at module scope; the tests below only
 * dispatch and compare — for the reason `wgsl.test.ts` writes out in full
 * (#49: once `useGpu`'s `beforeAll` has created the device, a test that takes
 * even ~10ms before its first dispatch kills the vitest worker with no result
 * reported; the same work before the device exists is fine at a full second).
 * Six nested loops over `N*Cout*Hout*Wout*(Cin/groups)*KH*KW` terms is exactly
 * that shape, more so than the 1D reference beside it.
 *
 * Two things here are load-bearing rather than convenient:
 *
 * - Every input buffer carries sentinel slack; see SENTINEL above.
 * - The output buffer is the *dispatch* width, not the result width, with the
 *   surplus expected to stay zero. A write past the end of an exactly-sized
 *   buffer is dropped by the driver, so the `ow >= Wout` guard could otherwise
 *   be deleted and the last row's surplus threads would have nowhere to land —
 *   the interior rows would still catch it (a surplus thread there overwrites
 *   the *next* row's data), but only the allocated slack makes the last row's
 *   overrun observable too.
 */
function prepare(shape: Case): { name: string; dispatch: Dispatch; expected: Float32Array } {
  const { name, N, Cin, Cout, H, W, KH, KW, stride, padding, dilation, groups } = shape;
  const { Hout, Wout } = conv2dOutputSize({ H, W, KH, KW, stride, padding, dilation });
  const input = wave(N * Cin * H * W, 0.37);
  const weight = wave(Cout * (Cin / groups) * KH * KW, 0.61, 1.1);
  // Non-zero everywhere on purpose: bias is the one term a surplus thread adds
  // no matter what it reads, so it is what makes the tail guard observable.
  const bias = Float32Array.from({ length: Cout }, (_, i) => 0.5 + i * 0.25);

  const dispatchWidth = Math.ceil(Wout / WORKGROUP) * WORKGROUP;
  const expected = new Float32Array(N * Cout * Hout * Wout + (dispatchWidth - Wout));
  expected.set(conv2d({ input, weight, bias, N, Cin, Cout, H, W, KH, KW, stride, padding, dilation, groups }));

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
            ["u32", H],
            ["u32", W],
            ["u32", KH],
            ["u32", KW],
            ["u32", Hout],
            ["u32", Wout],
            ["u32", stride[0]],
            ["u32", stride[1]],
            ["u32", padding[0]],
            ["u32", padding[1]],
            ["u32", dilation[0]],
            ["u32", dilation[1]],
            ["u32", Cin / groups],
            ["u32", Cout / groups],
          ]),
        },
      ],
      workgroups: [dispatchWidth / WORKGROUP, Hout, N * Cout],
    },
  };
}

const PREPARED = CASES.map(prepare);

/**
 * Absolute tolerance, measured rather than widened until green.
 *
 * The default `abs` of 1e-6 fails two of the cases below, and the reason is
 * cancellation, not a wrong sum: a 2D window is up to `(Cin/groups) * KH * KW`
 * terms — 27 in the `Wout=256` case against `conv1d`'s 3 to 7 — of magnitude up
 * to ~3, so an output near zero has passed through an accumulator near 80,
 * where one f32 ulp is already ~7.6e-6.
 *
 * Measured on this machine (RTX 5090, Dawn/Vulkan, f32), worst element per
 * case, kernel against the f64 reference:
 *
 *   plain 3x3        9.5e-7      pad-heavy      1.9e-6
 *   Wout=255         2.9e-6      overrun        1.6e-6
 *   Wout=256         3.3e-6      depthwise      1.9e-6
 *   knobs            1.4e-6
 *
 * What says this is rounding and not arithmetic: the same comparison against a
 * JavaScript reference that rounds **every** multiply and add to f32, exactly
 * as the shader does, lands in the same place (9.5e-7 / 2.6e-6 / 3.8e-6 /
 * 1.7e-6 / 1.9e-6 / 1.9e-6 / 1.7e-6) rather than collapsing to zero — the two
 * differ only in summation order and in the FMA the driver is free to use, and
 * that difference is what is left when the maths agrees.
 *
 * 6e-6 is under one ulp of that accumulator and under twice the worst measured
 * element, so it is still far tighter than any of the mutations tried against
 * this kernel, each of which moves elements by 1e-1 or more. `rel` stays at the
 * harness default of 1e-5, which every element large enough to have a
 * meaningful relative error already passes.
 */
const TOLERANCE = { abs: 6e-6 };

describe("conv2d / wgsl", () => {
  useGpu();

  for (const { name, dispatch, expected } of PREPARED) {
    gpuTest(`agrees with the reference: ${name}`, async (run) => {
      await expectAgrees(run, dispatch, [expected], TOLERANCE);
    });
  }
});
