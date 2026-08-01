import { params } from "../../harness/index.js";
import { moeDispatch } from "./reference.js";

/**
 * Shared by the six GPU test files of this op.
 *
 * Six files, and every fixture built at module scope, because of what this
 * harness turned out to tolerate. Measured on this machine while writing the
 * op, not guessed:
 *
 * - A test file that runs more than roughly half a dozen dispatches of any
 *   size, or any dispatch with a binding past ~64 KiB, kills its vitest worker
 *   — silently, with no summary, so `scripts/test.mjs` reports it as "produced
 *   no summary" and retries. Reduced to one shader and a `setTimeout` between
 *   dispatches, with nothing of this op involved, it fails the same way, so it
 *   is issue #38's family rather than anything about MoE. All ten router tests
 *   in one file failed five runs out of six; split six and four, both halves
 *   passed three out of three.
 * - Fixtures built inside `gpuTest` fail where the same fixtures built at
 *   module scope pass: `useGpu` creates the device in `beforeAll`, so module
 *   scope is the one place work is free of it. `gather.test.ts` failed six runs
 *   out of six one way and passed four out of four the other, unchanged
 *   otherwise.
 *
 * None of this is how the op wants to be tested. It is how it can be tested
 * here today, which is why the shapes below stay small and why the round trip
 * asks the reference how much capacity it needs instead of over-allocating.
 */

export const wave = (n: number, k = 0.37): Float32Array =>
  Float32Array.from({ length: n }, (_, i) => Math.sin(i * k) * 3);

/**
 * A buffer longer than the kernel is supposed to read, with non-zero values
 * past the end of the real data.
 *
 * This device returns zero for a read past the end of a *buffer*, which means a
 * kernel that runs off the end of its *data* still agrees with a reference that
 * stopped in the right place — the extra terms are all zero. Real values out
 * there are what makes the difference observable.
 */
export function withJunk(data: Float32Array, extra: number): Float32Array {
  const out = new Float32Array(data.length + extra);
  out.set(data);
  for (let i = data.length; i < out.length; i += 1) out[i] = 1 + (i % 5);
  return out;
}

export const routerParams = (T: number, E: number, k: number, normalize: boolean): ArrayBuffer =>
  params([["u32", T], ["u32", E], ["u32", k], ["u32", normalize ? 1 : 0]]);

export const moeParams = (T: number, k: number, E: number, C: number, D: number): ArrayBuffer =>
  params([["u32", T], ["u32", k], ["u32", E], ["u32", C], ["u32", D]]);

/** The router runs one thread per token. */
export const tokenThreads = (T: number): [number] => [Math.ceil(T / 256)];

/**
 * The largest number of assignments any expert receives — the smallest capacity
 * at which nothing is dropped.
 *
 * Used by the round-trip test, which needs "nothing dropped" to be true without
 * asking for `T*k` seats per expert: that buffer is `E*T*k*D` and this harness
 * is not reliable at that size.
 */
export function maxLoad(expert: Int32Array, E: number): number {
  const count = new Int32Array(E);
  for (const e of expert) if (e >= 0 && e < E) count[e] = count[e]! + 1;
  return count.reduce((a, b) => Math.max(a, b), 0);
}

/**
 * A dispatch fixture: the reference result, and an output buffer padded by
 * `pad` experts' worth.
 *
 * The padding is not decoration. A missing capacity guard writes at
 * `(e*C + slot)*D` with `slot >= C`, which for any expert but the last lands
 * inside the *next* expert's rows and is caught by the comparison; for the last
 * expert it would run off the end, where this driver discards it, so the spare
 * rows are the only place such a write can be seen.
 */
export function dispatched(
  T: number,
  k: number,
  E: number,
  C: number,
  D: number,
  expert: Int32Array,
  pad: number,
) {
  const shape = { T, k, E, C, D };
  const x = wave(T * D, 0.11);
  const { buffer, pos } = moeDispatch({ x, expert, ...shape });
  const expected = new Float32Array((E + pad) * C * D);
  expected.set(buffer);
  return { shape, x, expert, pos, expected, length: (E + pad) * C * D };
}
