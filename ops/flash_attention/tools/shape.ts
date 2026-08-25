/**
 * What every flash-attention generation here has in common: a tile shape, this
 * device's limits, and the check that the two are compatible.
 *
 * Issue #177. The generations differ in how they *schedule* the tile loop, not
 * in what a tile is, so the shape and the rejection rules are shared and the
 * loop is not. Anything that reads on both sides of a `if (generation === ...)`
 * belongs here; anything that does not belongs in `fa2.wgsl.ts` or
 * `fa3.wgsl.ts`.
 */

/** Which generation's schedule a kernel uses. See `GENERATIONS` for what each means. */
export type Generation = "fa2" | "fa3";

export interface FlashShape {
  /** Query rows per workgroup — the whole point. */
  bq: number;
  /** Keys staged per pass. */
  tileS: number;
  /** Threads per workgroup. */
  threads: number;
  /**
   * How the accumulate loop is nested.
   *
   * `"row"` walks rows outermost, which re-reads each staged value of `v` once
   * per row: `BQ * TILE_S` reads from workgroup memory per thread per tile.
   * `"key"` walks keys outermost and holds the value in a register while every
   * row uses it — `TILE_S` reads for the same `BQ * TILE_S` multiply-adds.
   *
   * Measured, not assumed. `key` wins by a factor of about 1.5 on both of
   * Anima's shapes — see `bench.ts`'s output in the issue — but it was not
   * obvious beforehand that it would: workgroup memory is not global memory and
   * a compiler is free to hoist the read itself.
   */
  accumulate: "row" | "key";
  /**
   * How the next tile's global loads are issued, on generations that overlap
   * them with compute. Ignored by `fa2`, which does not overlap them at all.
   *
   * `"direct"` writes the next tile straight into the idle half of the staging
   * buffer before the scores are computed, and relies on the compiler to keep
   * the score arithmetic from waiting on the store.
   * `"registers"` loads the next tile into registers first, computes the
   * scores, and only then stores — so the wait for global memory lands after
   * the arithmetic whatever the compiler does.
   */
  prefetch: "direct" | "registers";
  /**
   * How the score dot product reads `q` and `k` out of workgroup memory.
   *
   * `"scalar"` walks `D` one channel at a time: **two workgroup loads per fused
   * multiply-add**, which is the ratio `ops/matmul` had before its register
   * tile. `"vec4"` stages both as `vec4<f32>` and uses the builtin `dot`, so
   * four multiply-adds cost two loads — a quarter of the traffic for the same
   * arithmetic.
   *
   * This is the loop that matters. Deleting it entirely takes the self shape
   * from 16.45 ms to 5.28 ms, against 15.23 ms for deleting the accumulate —
   * see `tools/where.ts`, which measures that by removing each in turn.
   */
  scoreReads: "scalar" | "vec4";
  /**
   * Whether each staged row of `q` and `k` gets one unused element on the end.
   *
   * Workgroup memory is banked. In the score phase adjacent threads differ by
   * `slot`, so they read addresses `slot * D` apart; with `D = 128` f32 that is
   * a multiple of the bank count and every one of them lands on the same bank,
   * serialising a read that should have been one transaction. One element of
   * padding makes the stride coprime with the banks.
   *
   * **A guess until measured.** WebGPU does not say how many banks there are,
   * or that there are banks; the sweep is the only way to find out whether this
   * device behaves like one that has them.
   */
  padRows: boolean;
  /**
   * How the staging loops turn a flat index into a global address.
   *
   * `"divide"` writes what the arithmetic says: row `base + e / Dv`, channel
   * `e % Dv`, address `v_head + j * Dv + d`. Readable, and it costs an integer
   * division and a modulo per element — **a GPU has no integer divide
   * instruction**, so each is a short sequence the compiler emits inline, and
   * `Dv` is a uniform so it cannot fold them.
   *
   * `"linear"` uses the fact that they cancel:
   *
   *     (base + e / Dv) * Dv + e % Dv  ==  base * Dv + e
   *
   * which is exact for unsigned integers, not an approximation. The bound check
   * `j < S` becomes `e < (S - base) * Dv`, so nothing divides at all.
   *
   * Measured, because "fewer instructions" is not "faster": `tools/where.ts`
   * priced the `v` staging at **16%** of the self shape against **6%** for `k`,
   * which stages the same 4 KB per tile with the same coalescing but runs its
   * loop a quarter as many times. The division count is the difference between
   * them, and this is the field that tests whether that is the reason.
   */
  stageAddressing: "divide" | "linear";
}

/**
 * The limits this device reports.
 *
 * Read from the adapter rather than assumed — `harness/device-limits.test.ts`
 * exists because three separate dispatches here were written against limits
 * nobody had requested, and Dawn rejected all three.
 */
export const WORKGROUP_STORAGE_LIMIT = 49152;
export const INVOCATION_LIMIT = 1024;

/** How many halves of the k/v staging buffer a generation keeps live. */
export function stagingBuffers(generation: Generation): number {
  return generation === "fa3" ? 2 : 1;
}

/** Workgroup storage in bytes for a given head dimension. */
export function flashStorageBytes({ bq, tileS }: FlashShape, D: number, generation: Generation): number {
  // q rows, the staged k and v tiles, one score per (row, key), and the
  // per-row softmax state — a running maximum, a running sum, and the
  // correction the tile's new maximum implies. fa3 double-buffers k and v so
  // the next tile can land while the current one is still being read.
  // vec4 staging rounds each row up to a multiple of four channels.
  const padded = Math.ceil(D / 4) * 4 + 4;
  const staged = tileS * padded * stagingBuffers(generation) + tileS * D * stagingBuffers(generation);
  return (bq * padded + staged + bq * tileS + bq * 3) * 4;
}

/** Why a shape cannot be dispatched, or null. */
export function rejectReason(shape: FlashShape, D: number, generation: Generation): string | null {
  if (shape.threads > INVOCATION_LIMIT) return `${shape.threads} invocations exceeds ${INVOCATION_LIMIT}`;
  const bytes = flashStorageBytes(shape, D, generation);
  if (bytes > WORKGROUP_STORAGE_LIMIT) return `${bytes} B exceeds ${WORKGROUP_STORAGE_LIMIT}`;
  if ((shape.tileS * D) % shape.threads !== 0) return "k/v tile is not a multiple of the thread count";
  if ((shape.bq * shape.tileS) % shape.threads !== 0) return "score tile is not a multiple of the thread count";
  return null;
}

/** How many channels of one row a thread carries in registers. `Dv` can exceed the workgroup. */
export function sweepsFor(threads: number, maxDv: number): number {
  return Math.max(1, Math.ceil(maxDv / threads));
}
