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
 * `kernel.wgsl`'s query tile, and the grid a caller dispatches for it.
 *
 * The kernel takes `BQ` query rows per workgroup rather than one — see its
 * header for why that is the only thing that moves this op — so the grid is not
 * one workgroup per query.
 */
export const FLASH_TILE = { bq: 16, threads: 128, maxHeadDim: 128, maxValueDim: 128 } as const;

/** `[x, y, z]` for `kernel.wgsl` over `L` queries, `H` heads, `B` batches. */
export function flashGrid(L: number, H: number, B: number): [number, number, number] {
  return [Math.ceil(L / FLASH_TILE.bq), H, B];
}

/**
 * Whether `kernel.wgsl` can serve these head dimensions.
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
