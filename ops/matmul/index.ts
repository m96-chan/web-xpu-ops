export {
  matmul,
  matmulQ4G128,
  matmulQ8,
  type MatmulArgs,
  type MatmulQ4G128Args,
  type MatmulQ8Args,
} from "./reference.js";

/**
 * `kernel.wgsl`'s workgroup tile, and the grid a caller dispatches for it.
 *
 * Exported because the alternative is every caller writing 64 and 128 down
 * again: `llm/kernels.ts` already carries a `MATMUL_TILE = 16` with a comment
 * saying it must match `kernel.wgsl`, and a constant kept in step by a comment
 * is a constant that eventually is not.
 *
 * The numbers come from `tools/bench.ts` sweeping 345 shapes at the dimensions
 * a real model uses — see `wgsl/kernel.wgsl`'s own header for the measurement and its
 * conditions. They are this device's answer, not a universal one; rerun the
 * sweep on different hardware.
 */
export const MATMUL_TILE = { bm: 64, bn: 128, threads: 512 } as const;

/** `[x, y, z]` a caller dispatches for `kernel.wgsl` computing `[M, N]`. */
export function matmulGrid(M: number, N: number): [number, number, number] {
  return [Math.ceil(N / MATMUL_TILE.bn), Math.ceil(M / MATMUL_TILE.bm), 1];
}

/**
 * `q8.wgsl`'s workgroup tile. Not the same shape as `MATMUL_TILE`: the sweep
 * that chose it ran against the packed kernel, and it won with a taller,
 * narrower thread tile — 8x2 against the dense kernel's 4x4.
 */
export const MATMUL_Q8_TILE = { bm: 128, bn: 64, threads: 512 } as const;

/** `[x, y, z]` a caller dispatches for `q8.wgsl` computing `[N, M]`. */
export function matmulQ8Grid(N: number, M: number): [number, number, number] {
  return [Math.ceil(M / MATMUL_Q8_TILE.bn), Math.ceil(N / MATMUL_Q8_TILE.bm), 1];
}
