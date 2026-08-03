import { expectAgrees, kernel, params, type Dispatch, type Runner } from "../../harness/index.js";
import { convTranspose1d, convTranspose1dOutputLength } from "./reference.js";

/**
 * Shared setup for the two WGSL test files.
 *
 * The cases are split across `wgsl.test.ts` and `wgsl-shapes.test.ts` rather
 * than run from one file. Measured on this machine: nine dispatches of these
 * sizes in a single vitest process take the worker down partway through more
 * often than not — 1 clean run in 6, against 5 in 6 for `ops/conv`'s file at
 * the same moment. It is not this kernel: a shader with no loop and no integer
 * division, at the same bindings and the same dispatch sizes, dies just the
 * same, and every case passes on its own. It is the #38 / #62 boundary, and the
 * only lever this side of it is how much one process is asked to do. Two files
 * means two processes (`scripts/test.mjs` runs one file per process), which is
 * that lever.
 *
 * Not a test file itself, since `vitest.config.ts` globs only names ending in
 * `.test.ts`, and not public either: `package.json` exports each op directory's
 * `index.ts` and nothing beside it.
 */

/** The kernel both files dispatch. */
export const code = kernel(import.meta.url);

export const WORKGROUP = 256;

/**
 * A value that is not zero, parked past the end of every input buffer.
 *
 * This device returns 0 for a read past the end of a storage buffer, which
 * makes a missing range guard invisible: the guard's whole job is to contribute
 * nothing, and the out-of-range read already contributes nothing. Filling the
 * slack with something else means an unguarded read lands on a number that
 * changes the answer.
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

export interface Case {
  name: string;
  N: number;
  Cin: number;
  Cout: number;
  L: number;
  K: number;
  stride: number;
  padding: number;
  outputPadding: number;
  dilation: number;
  groups: number;
}

export interface Prepared {
  name: string;
  dispatch: Dispatch;
  expected: Float32Array;
}

/**
 * Builds one case: inputs, the reference answer, and the dispatch that should
 * reproduce it.
 *
 * Callers run this at **module scope**, before any test body. Not a style
 * choice. Measured on this machine (#49): once `useGpu`'s `beforeAll` has
 * created the device, a test that takes even ~10ms before its first dispatch
 * kills the vitest worker with `std::system_error` and no result reported at
 * all. The same work at module scope, before the device exists, is fine at a
 * full second. This reference is five nested loops over
 * `N*Cin*(Cout/groups)*L*K` terms, which is exactly the shape that trips it.
 *
 * Two things here are load-bearing rather than convenience, and both exist
 * because this device is quiet about the edges:
 *
 * - Every input buffer carries sentinel slack; see SENTINEL above.
 * - The output buffer is the *dispatch* length, not the result length, with the
 *   surplus expected to stay zero. A write past the end of an exactly-sized
 *   buffer is dropped by the driver, so the `ol >= Lout` guard could otherwise
 *   be deleted and the suite would stay green. With the slack allocated, the
 *   last row's surplus threads have somewhere real to land.
 */
export function prepare(shape: Case): Prepared {
  const { name, N, Cin, Cout, L, K, stride, padding, outputPadding, dilation, groups } = shape;
  const Lout = convTranspose1dOutputLength({ L, K, stride, padding, outputPadding, dilation });
  const input = wave(N * Cin * L, 0.37);
  const weight = wave(Cin * (Cout / groups) * K, 0.61, 1.1);
  // Non-zero everywhere on purpose: bias is the one term a surplus thread adds
  // no matter what it reads, so it is what makes the tail guard observable.
  const bias = Float32Array.from({ length: Cout }, (_, i) => 0.5 + i * 0.25);

  const dispatchLength = Math.ceil(Lout / WORKGROUP) * WORKGROUP;
  const expected = new Float32Array(N * Cout * Lout + (dispatchLength - Lout));
  expected.set(
    convTranspose1d({ input, weight, bias, N, Cin, Cout, L, K, stride, padding, outputPadding, dilation, groups }),
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
          // output_padding is deliberately not here: it only widens Lout, and
          // Lout is passed. See the kernel header.
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

/** Dispatches one prepared case and asserts it agrees with the reference. */
export async function check(run: Runner["run"], { dispatch, expected }: Prepared): Promise<void> {
  await expectAgrees(run, dispatch, [expected]);
}
