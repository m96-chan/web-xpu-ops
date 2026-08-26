import { describe } from "vitest";
import { expectAgrees, gpuTest, kernel, params, useGpu, type Dispatch } from "../../harness/index.js";
import { conv3d, conv3dOutputSize } from "./reference.js";

const code = kernel(import.meta.url, "conv3d");
const WORKGROUP = 256;

/**
 * A value that is not zero, parked past the end of every input buffer.
 *
 * Same reason as `conv2d.wgsl.test.ts`, and it matters more here: this device
 * returns 0 for a read past the end of a storage buffer, which makes a missing
 * padding guard invisible — the guard's whole job is to contribute zero, and
 * the out-of-range read already does. 3D adds a third pair of edges to get
 * wrong, and the temporal one is the pair a video model leans on.
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
  D: number;
  H: number;
  W: number;
  KD: number;
  KH: number;
  KW: number;
  stride: readonly [number, number, number];
  padding: readonly [number, number, number];
  dilation: readonly [number, number, number];
  groups: number;
}

// The reference is eight nested loops, so every case keeps the total term count
// modest — but `Wout` still has to span the 256-wide workgroup in some of them,
// because a ragged tail is the only thing that exercises the x guard.
const CASES: Case[] = [
  // The plain shape, and the one a video VAE runs everywhere: a 3x3x3 window.
  // D, H and W all differ, so a kernel that read the triple in any other order
  // would produce a differently-shaped tensor.
  {
    name: "plain 3x3x3, D/H/W all different",
    N: 1, Cin: 1, Cout: 1, D: 4, H: 5, W: 7, KD: 3, KH: 3, KW: 3,
    stride: [1, 1, 1], padding: [0, 0, 0], dilation: [1, 1, 1], groups: 1,
  },
  // Wout = 255: one thread short of a full workgroup, so the tail guard has to
  // stop exactly one surplus thread.
  {
    name: "one under a workgroup, Wout=255",
    N: 1, Cin: 2, Cout: 2, D: 3, H: 3, W: 257, KD: 2, KH: 3, KW: 3,
    stride: [1, 1, 1], padding: [0, 0, 0], dilation: [1, 1, 1], groups: 1,
  },
  // Wout = 256: exactly a workgroup, so no thread is surplus at all — and
  // N, Cout and Dout are all greater than one together, which is what makes the
  // three-way split of z checkable. A kernel that unpacked z in a different
  // order stays in range and is still wrong.
  {
    name: "exactly a workgroup, N/Cout/Dout all > 1",
    N: 2, Cin: 2, Cout: 3, D: 4, H: 3, W: 258, KD: 2, KH: 3, KW: 3,
    stride: [1, 1, 1], padding: [0, 0, 0], dilation: [1, 1, 1], groups: 1,
  },
  // Every knob off its default and different on all three axes, with a ragged
  // tail of 211 threads.
  {
    name: "strided + dilated + grouped, non-cubic kernel, ragged tail",
    N: 2, Cin: 4, Cout: 6, D: 6, H: 9, W: 300, KD: 2, KH: 3, KW: 2,
    stride: [2, 2, 1], padding: [1, 1, 2], dilation: [1, 2, 3], groups: 2,
  },
  // Depthwise, groups = Cin = Cout: each output channel sees exactly one input
  // channel, so a kernel that mixed them would read the neighbouring volume.
  {
    name: "depthwise",
    N: 1, Cin: 3, Cout: 3, D: 4, H: 6, W: 777, KD: 2, KH: 2, KW: 4,
    stride: [1, 1, 1], padding: [1, 1, 1], dilation: [1, 1, 1], groups: 3,
  },
  // Pad dominates the window on all six edges. Cin = 2 and N = 2 matter: with
  // more than one volume, a window that runs off the front of its own volume
  // lands on the *previous channel's last frames*, which is real data rather
  // than the zero an out-of-range read returns.
  {
    name: "pad dominates the window, all six edges",
    N: 2, Cin: 2, Cout: 3, D: 3, H: 4, W: 4, KD: 3, KH: 3, KW: 3,
    stride: [1, 1, 1], padding: [2, 2, 2], dilation: [1, 1, 1], groups: 1,
  },
  // Temporal padding alone, which is the shape H3's causal convolutions reduce
  // to once their asymmetric pad has been applied by `ops/pad`: the spatial
  // axes are already exact and only D is padded.
  {
    name: "temporal padding only",
    N: 1, Cin: 2, Cout: 2, D: 5, H: 4, W: 300, KD: 3, KH: 1, KW: 1,
    stride: [1, 1, 1], padding: [1, 0, 0], dilation: [1, 1, 1], groups: 1,
  },
  // A dilated window that overruns the back, bottom and right edges at once.
  // Past the final volume that is the sentinel slack, not zero, so a missing
  // upper guard shows up as a wrong number rather than as the right one.
  {
    name: "dilated window overruns the far edges of all three axes",
    N: 1, Cin: 2, Cout: 2, D: 4, H: 6, W: 20, KD: 3, KH: 3, KW: 4,
    stride: [1, 1, 1], padding: [2, 2, 6], dilation: [2, 2, 4], groups: 1,
  },
];

/**
 * Everything expensive is built at module scope; the tests below only dispatch
 * and compare — see `conv2d.wgsl.test.ts` for why (#49: work between `useGpu`'s
 * `beforeAll` and the first dispatch kills the vitest worker, the same work
 * before the device exists is fine).
 *
 * The output buffer is the *dispatch* width, not the result width, with the
 * surplus expected to stay zero: a write past the end of an exactly-sized
 * buffer is dropped by the driver, so without the slack the `ow >= Wout` guard
 * could be deleted and the final row's surplus threads would have nowhere to
 * land.
 */
function prepare(shape: Case): { name: string; dispatch: Dispatch; expected: Float32Array } {
  const { name, N, Cin, Cout, D, H, W, KD, KH, KW, stride, padding, dilation, groups } = shape;
  const { Dout, Hout, Wout } = conv3dOutputSize({ D, H, W, KD, KH, KW, stride, padding, dilation });
  const input = wave(N * Cin * D * H * W, 0.37);
  const weight = wave(Cout * (Cin / groups) * KD * KH * KW, 0.61, 1.1);
  // Non-zero everywhere on purpose: bias is the one term a surplus thread adds
  // no matter what it reads, so it is what makes the tail guard observable.
  const bias = Float32Array.from({ length: Cout }, (_, i) => 0.5 + i * 0.25);

  const dispatchWidth = Math.ceil(Wout / WORKGROUP) * WORKGROUP;
  const expected = new Float32Array(N * Cout * Dout * Hout * Wout + (dispatchWidth - Wout));
  expected.set(
    conv3d({ input, weight, bias, N, Cin, Cout, D, H, W, KD, KH, KW, stride, padding, dilation, groups }),
  );

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
            ["u32", D],
            ["u32", H],
            ["u32", W],
            ["u32", KD],
            ["u32", KH],
            ["u32", KW],
            ["u32", Dout],
            ["u32", Hout],
            ["u32", Wout],
            ["u32", stride[0]],
            ["u32", stride[1]],
            ["u32", stride[2]],
            ["u32", padding[0]],
            ["u32", padding[1]],
            ["u32", padding[2]],
            ["u32", dilation[0]],
            ["u32", dilation[1]],
            ["u32", dilation[2]],
            ["u32", Cin / groups],
            ["u32", Cout / groups],
            ["u32", 0],
            ["u32", 0],
          ]),
        },
      ],
      workgroups: [dispatchWidth / WORKGROUP, Hout, N * Cout * Dout],
    },
  };
}

const PREPARED = CASES.map(prepare);

/**
 * Absolute tolerance, measured rather than widened until green.
 *
 * A 3D window is up to `(Cin/groups) * KD * KH * KW` terms — 54 in the
 * `Wout=256` case against `conv2d`'s 27 — of magnitude up to ~3, so an output
 * near zero has passed through an accumulator near 160, where one f32 ulp is
 * already ~1.5e-5.
 *
 * Worst element per case, kernel against the reference, measured on this
 * machine (RTX 5090, driver 610.57.04, Dawn/Vulkan, f32):
 *
 *   plain 3x3x3              2.86e-6     depthwise                2.38e-6
 *   Wout=255                 3.34e-6     pad dominates            7.63e-6
 *   Wout=256, N/Cout/Dout>1  4.41e-6     temporal padding only    4.77e-7
 *   strided+dilated+grouped  2.86e-6     dilated overrun          7.63e-6
 *
 * 1.5e-5 is under two ulps of that accumulator and under twice the worst
 * measured element. It is far tighter than any mutation tried against this
 * kernel — deleting a padding guard, transposing an axis or unpacking `z` in
 * the wrong order each move elements by 1e-1 or more. `rel` stays at the
 * harness default.
 *
 * The first version of this file said 4e-5 and claimed it was measured. It was
 * not; the numbers above are, and they are five times tighter.
 */
const ABS = 1.5e-5;

describe("conv3d / wgsl", () => {
  useGpu();

  for (const { name, dispatch, expected } of PREPARED) {
    gpuTest(`agrees with the reference: ${name}`, async (run) => {
      await expectAgrees(run, dispatch, [expected], { abs: ABS });
    });
  }
});
