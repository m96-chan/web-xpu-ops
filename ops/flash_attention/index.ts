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
 * **Measured, on an RTX 5090 (Dawn, f32, D = Dv = 128), by
 * `tools/bench.ts`.** Against the previous shipped shape, taken as 1.00 in the
 * same rounds so a clock that moved moved both:
 *
 *     BQ=32 TILE_S=8 256t key-outer   0.51 cross,  0.51 self   <- this
 *     BQ=16 TILE_S=8 128t key-outer   0.51 cross,  0.57 self
 *     BQ=32 TILE_S=4 128t key-outer   0.47 cross,  0.62 self
 *     BQ=32 TILE_S=16 256t key-outer  0.74 cross,  0.73 self
 *     BQ=64 TILE_S=8 256t key-outer   0.76 cross,   -    self
 *
 * `key-outer` is the accumulate nesting and is worth about 1.5x on its own:
 * walking keys outermost reads each staged value of `v` once into a register
 * and lets all `BQ` rows use it, where walking rows outermost re-read it `BQ`
 * times from workgroup memory.
 *
 * The two shapes disagree and `self` wins the argument — it is roughly four
 * times the cost of `cross` in an Anima block — which is why `TILE_S = 4` is
 * not taken despite being the faster of the two on `cross`.
 *
 * **`BQ = 64` is slower, and that bounds the query tile.** More rows per
 * workgroup is what made this op tractable at all, so the obvious next step was
 * more of it; measured it costs 1.5x. Sixty-four accumulators a thread is past
 * what the register file will hold at this occupancy, and the traffic saved is
 * worth less than the occupancy lost.
 *
 * **Where the remaining headroom is, stated against a measured ceiling
 * (rule 9).** The shipped shape reaches 3.88 TFLOP/s, which is 7.7% of this
 * device's measured 50.4 TFLOP/s. That is not the ceiling to compare against:
 * at `BQ = 32` the kernel does 16 FLOP per byte of k/v, against a measured
 * crossover of 29.6 (50.4 TFLOP/s over 1.70 TB/s), so bandwidth caps it at
 * 27.2 TFLOP/s — 54% of compute peak — however well it is written.
 *
 * It is not near that either. The logical k/v traffic is 4.01 GB in 16.50 ms on
 * `self` and 1.04 GB in 4.34 ms on `cross`: **243 GB/s and 240 GB/s, both 14%
 * of the 1.70 TB/s roofline.** Two shapes that differ by 4x in size and 2x in
 * head count landing on the same figure says the limit is neither compute nor
 * DRAM but something inside the workgroup that both hit equally. The candidate
 * is the score loop, which reads two workgroup-memory values per fused
 * multiply-add — the same ratio `ops/matmul` had before its register tile took
 * it from 1.8% to 72%. **Untested.** `BQ * TILE_S` is 256 against 256 threads,
 * so there is no work to give a thread without taking threads away, and which
 * of those two costs more is a measurement nobody here has made.
 */
export const FLASH_TILE = {
  bq: 32,
  tileS: 8,
  threads: 256,
  accumulate: "key",
  prefetch: "direct",
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
