/**
 * The free half of a buffer pool, and how much of it to give back.
 *
 * Issue #223. `DitGpu` and `VideoDecoderGpu` (`model-gpu.ts`, `decoder-gpu.ts`)
 * both keep a `Map<size, GPUBuffer[]>` of buffers that are free but never
 * destroyed, so a buffer allocated for the widest intermediate a run ever saw
 * stays resident even once that stretch of the block is long over. At 19,027
 * rows the attention stretch (7 x 549 MB) and the feed-forward stretch
 * (3 x 1095 MB, a 4th refused) of a block never run at the same time, but both
 * sizes' peaks stayed in the pool at once — resident memory is the SUM of
 * every size class's own peak where what a step actually needs is the MAX.
 *
 * `evictionPlan` is the pure decision; the two `release()`s call it with a
 * budget and destroy what it names.
 */

/** One size class of the free pool: how large each buffer is, and how many sit free. */
export interface FreeClass {
  size: number;
  count: number;
}

/**
 * Which free buffers to destroy so the pool's free bytes fit `budgetBytes`.
 *
 * Deterministic and largest-first: walks size classes from largest to
 * smallest, evicting one buffer at a time until the total free bytes are
 * `<= budgetBytes`. Largest-first because one 1095 MB buffer frees as much as
 * fifty small ones and costs one recreation, and because the small classes
 * (uniforms, positions) are reused every dispatch while the big ones are
 * reused once per block — see `examples/h3-dit/src/model-gpu.ts`'s
 * `maxFreePoolBytes` for the measured argument.
 *
 * Returns a plan, not an action: pure, so it is tested without a GPU.
 */
export function evictionPlan(free: FreeClass[], budgetBytes: number): Map<number, number> {
  const plan = new Map<number, number>();
  let total = free.reduce((sum, { size, count }) => sum + size * count, 0);
  if (total <= budgetBytes) return plan;

  const sorted = [...free].sort((a, b) => b.size - a.size);
  for (const { size, count } of sorted) {
    let evicted = 0;
    while (total > budgetBytes && evicted < count) {
      total -= size;
      evicted++;
    }
    if (evicted > 0) plan.set(size, evicted);
    if (total <= budgetBytes) break;
  }
  return plan;
}
