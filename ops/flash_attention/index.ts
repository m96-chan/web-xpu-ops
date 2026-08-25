export {
  flashAttention,
  DEFAULT_TILE,
  type FlashAttentionArgs,
  type FlashAttentionResult,
} from "./reference.js";
// The mask is `ops/attention`'s — `FlashAttentionArgs` already extends
// `AttentionArgs`, so re-exporting the helpers keeps one mask type across the
// three attention ops (#77).
export { keyPaddingBias, type MaskShape } from "../attention/reference.js";

/**
 * The shipped kernel's shape, and the grid a caller dispatches for it.
 *
 * The kernel takes `BQ` query rows per workgroup rather than one — see
 * `tools/fa2.wgsl.ts` for why that is the only thing that ever moved this op —
 * so the grid is not one workgroup per query.
 *
 * **Measured, on an RTX 5090 (Dawn, f32, D = Dv = 128), by `tools/bench.ts`.**
 * Ratios are to the previous shipped shape, timed in the same rounds so a clock
 * that moved moved both:
 *
 *     BQ=32 TILE_S=8 128t key vec4 +pad   0.35 cross,  0.35 self   <- this
 *     BQ=16 TILE_S=8 128t key vec4 +pad   0.40 cross,  0.42 self
 *     BQ=32 TILE_S=8 128t key vec4        0.78 cross,  0.81 self
 *     BQ=32 TILE_S=8 256t key scalar +pad 0.79 cross,  0.78 self
 *     BQ=32 TILE_S=8 256t key scalar      1.00 cross,  1.00 self
 *     BQ=64 TILE_S=8 256t key scalar      1.49 cross
 *
 * **Three generations of FlashAttention measured within a few percent of each
 * other, and none of them was the problem.** `tools/where.ts` says why: it
 * deletes each inner loop in turn and times what is left. On the self shape,
 * full is 16.45 ms, deleting the accumulate leaves 15.23, deleting the score
 * dot leaves 5.28, deleting both leaves 2.76. **The score dot product is 76% of
 * this kernel.** FA1, FA2 and FA3 differ in how the tile loop is *scheduled*
 * and all three run the same dot product, so none of them could move it.
 *
 * Two things moved it, and they compose better than they multiply (0.81 x 0.78
 * would be 0.63; together they measure 0.35):
 *
 *   - **`vec4` reads.** The scalar loop is two workgroup loads per fused
 *     multiply-add, the ratio `ops/matmul` had before its register tile. Four
 *     multiply-adds per two loads instead.
 *   - **One element of padding per staged row.** Workgroup memory is banked and
 *     adjacent threads in the score phase differ by `slot`, so they read
 *     addresses `slot * D` apart; at `D = 128` f32 that is a multiple of the
 *     bank count and every one of them serialises on the same bank. This was a
 *     guess — WebGPU does not say there are banks, or how many — and it is the
 *     single largest change in this op's history.
 *
 * Fixing `params.D` to a compile-time 128 so the loop could unroll was also
 * tried, and measured **1.06: slower**.
 *
 * `key-outer` is the accumulate nesting: walking keys outermost reads each
 * staged value of `v` once into a register and lets all `BQ` rows use it, where
 * walking rows outermost re-read it `BQ` times.
 *
 * The two shapes agreed on every ranking above, so nothing here is a trade
 * between them. **`BQ = 64` is slower and that bounds the query tile** — sixty
 * -four accumulators a thread costs more occupancy than the traffic it saves.
 *
 * **Where it stands, against a measured ceiling (rule 9).** 10.6 TFLOP/s, or
 * **21% of this device's measured 50.4 TFLOP/s**, from 7.7% before. The
 * compute roofline is still not the right ceiling: at 16 FLOP per byte of k/v
 * against a measured crossover of 29.6, bandwidth caps this kernel at
 * 27.2 TFLOP/s, which is 54% of compute peak. So it now runs at **39% of what
 * its arithmetic intensity allows**, and the way past that is more intensity,
 * not more tuning.
 */
export const FLASH_TILE = {
  bq: 32,
  tileS: 8,
  threads: 128,
  accumulate: "key",
  prefetch: "direct",
  scoreReads: "vec4",
  padRows: true,
  maxHeadDim: 128,
  maxValueDim: 128,
} as const;

/**
 * Which generation `flashAttention` dispatches.
 *
 * **FA3 is shipped and not selected, because it measured a tie.** Its portable
 * idea is software pipelining — stage tile `t+1` while tile `t` is scored, so
 * the memory latency is spent on arithmetic — and on this device it bought
 * nothing: 16.54 ms against FA2's 16.50 on the self shape, spread 0%, over a
 * sequence long enough for 494 tiles. The likely reason is that the latency was
 * already hidden by *occupancy*: at 256 threads and 26 KB of workgroup storage
 * many workgroups are resident per multiprocessor, and one computes while
 * another waits. FA3's own gains on an H100 come from tiles large enough that
 * only one or two blocks fit, which is not this shape.
 *
 * It stays selectable rather than deleted because that argument is about *this*
 * device's occupancy, and a device with less of it could rank them the other
 * way. Re-run `tools/bench.ts` rather than assuming either way.
 */
export const FLASH_GENERATION: "fa2" | "fa3" = "fa2";

/** `[x, y, z]` for the shipped kernel over `L` queries, `H` heads, `B` batches. */
export function flashGrid(L: number, H: number, B: number): [number, number, number] {
  return [Math.ceil(L / FLASH_TILE.bq), H, B];
}

/**
 * Whether the shipped kernels can serve these head dimensions.
 *
 * The staged tiles are workgroup arrays, so their size is fixed at compile
 * time — but the loops run to `params.D`, so one program serves everything up
 * to the bounds it was generated for. Past them a dispatch would read beyond
 * the staged tile, which is a wrong answer rather than an error, so the caller
 * is asked instead.
 *
 * **`Dv` is bounded tighter than it needs to be, and that is a measured
 * trade.** A thread carries `ceil(Dv / threads)` channels in registers, and
 * generating for `Dv` up to 320 — enough for `wgsl.test.ts`'s widest case —
 * makes that three per thread instead of one for every caller, including the
 * ones with `Dv = 128`. Measured on Anima: 5.00 s a forward at one, 5.92 s at
 * three. So the shipped program is generated for 128, and anything wider goes
 * through `ops/attention`'s two dispatches, which have no such bound.
 */
export function flashSupports(D: number, Dv: number): boolean {
  return D <= FLASH_TILE.maxHeadDim && Dv <= FLASH_TILE.maxValueDim;
}
