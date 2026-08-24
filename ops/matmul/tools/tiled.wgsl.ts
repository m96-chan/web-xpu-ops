/**
 * A tiled `matmul` kernel with its tile shape as a parameter, for sweeping.
 *
 * Issue #177 / #134. `ops/matmul/wgsl/kernel.wgsl` says of itself that
 * `TILE = 16 is not a measured optimum — nothing here is tuned yet`, and the
 * measurement says it reaches 1.5-1.8% of this device's roofline. This is the
 * shape to try instead, and the reason it is generated rather than written is
 * that the answer is a *number* nobody knows yet: WGSL has no specialisation
 * constants for workgroup-array sizes, so a tile shape is a compile-time
 * constant and a sweep needs one program per candidate.
 *
 * The structure is `mul_mat_reg_tile.wgsl`'s from llama.cpp (MIT, from the
 * `ggml-webgpu` backend), which is CUTLASS's two-level decomposition:
 *
 *   - a **workgroup tile** of `BM x BN` outputs, walked over `K` in `TILE_K`
 *     chunks staged into workgroup memory;
 *   - a **thread tile** of `TILE_M x TILE_N` held in registers, so each value
 *     loaded from workgroup memory is used `TILE_N` (or `TILE_M`) times rather
 *     than once.
 *
 * The second level is what the current kernel has none of, and what its 1.8%
 * is a symptom of.
 *
 * **The numbers are not copied.** llama.cpp stages in `f16` — this device has
 * no `shader-f16`, so the same tile needs twice the workgroup storage — and
 * their model's dimensions are not Anima's. What carries over is the shape of
 * the loop; the constants come from `bench.ts` measuring them here.
 */

export interface TileShape {
  /** Threads down the M axis. */
  wgM: number;
  /** Threads across the N axis. */
  wgN: number;
  /** Outputs per thread down M. */
  tileM: number;
  /** Outputs per thread across N. */
  tileN: number;
  /** How much of K one staging round covers. */
  tileK: number;
}

export const WORKGROUP_STORAGE_LIMIT = 49152;
export const INVOCATION_LIMIT = 1024;

/** Workgroup storage in bytes, f32 and both operands. */
export function storageBytes({ wgM, wgN, tileM, tileN, tileK }: TileShape): number {
  return (wgM * tileM * tileK + tileK * wgN * tileN) * 4;
}

/**
 * Why a shape cannot be dispatched here, or null.
 *
 * Checked rather than discovered: an over-large workgroup array is a shader
 * compilation failure, and a sweep that silently skips half its candidates
 * reports the best of the half it ran as though it were the best.
 */
export function rejectReason(shape: TileShape): string | null {
  const threads = shape.wgM * shape.wgN;
  if (threads > INVOCATION_LIMIT) return `${threads} invocations exceeds ${INVOCATION_LIMIT}`;
  const bytes = storageBytes(shape);
  if (bytes > WORKGROUP_STORAGE_LIMIT) return `${bytes} B of workgroup storage exceeds ${WORKGROUP_STORAGE_LIMIT}`;
  // One staging round has to be loadable by the threads there are. Not a
  // hardware limit — a limit of the loop below, which strides by the thread
  // count and would otherwise need a second level of looping.
  if ((shape.wgM * shape.tileM * shape.tileK) % threads !== 0) return "A tile is not a multiple of the thread count";
  if ((shape.tileK * shape.wgN * shape.tileN) % threads !== 0) return "B tile is not a multiple of the thread count";
  return null;
}

/**
 * `a: [M, K]` by `b: [K, N]` to `c: [M, N]`, all row-major — `ops/matmul`'s own
 * layout, so the reference is what this is checked against.
 *
 * Every load and store is guarded. M, N and K are the model's, not multiples of
 * anything convenient: Anima's `M` is 3,952 and no tile divides it. A kernel
 * that is only correct on aligned shapes would be measured on shapes it cannot
 * run, which is worse than being slower.
 */
export function tiledMatmul(shape: TileShape): string {
  const { wgM, wgN, tileM, tileN, tileK } = shape;
  const bm = wgM * tileM;
  const bn = wgN * tileN;
  const threads = wgM * wgN;
  return `
struct Params { M: u32, N: u32, K: u32 }

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

const WG_M: u32 = ${wgM}u;
const WG_N: u32 = ${wgN}u;
const TILE_M: u32 = ${tileM}u;
const TILE_N: u32 = ${tileN}u;
const TILE_K: u32 = ${tileK}u;
const BM: u32 = ${bm}u;
const BN: u32 = ${bn}u;
const THREADS: u32 = ${threads}u;

// A as [BM, TILE_K] and B as [TILE_K, BN], both row-major in one array so the
// two staging loops are the same shape.
var<workgroup> sa: array<f32, ${bm * tileK}>;
var<workgroup> sb: array<f32, ${tileK * bn}>;

@compute @workgroup_size(${threads})
fn main(
  @builtin(workgroup_id) wg: vec3<u32>,
  @builtin(local_invocation_index) tid: u32,
) {
  let m0 = wg.y * BM;
  let n0 = wg.x * BN;
  // Thread (ty, tx) owns C[m0 + ty*TILE_M .. +TILE_M, n0 + tx*TILE_N .. +TILE_N].
  let ty = tid / WG_N;
  let tx = tid % WG_N;

  var acc: array<array<f32, TILE_N>, TILE_M>;
  for (var i: u32 = 0u; i < TILE_M; i = i + 1u) {
    for (var j: u32 = 0u; j < TILE_N; j = j + 1u) { acc[i][j] = 0.0; }
  }

  for (var k0: u32 = 0u; k0 < params.K; k0 = k0 + TILE_K) {
    // Staging. Each thread takes every THREADS-th element of the tile, so the
    // reads are contiguous across a workgroup rather than strided per thread.
    for (var t: u32 = tid; t < BM * TILE_K; t = t + THREADS) {
      let r = t / TILE_K;
      let kk = t % TILE_K;
      let m = m0 + r;
      let k = k0 + kk;
      sa[t] = select(0.0, a[m * params.K + k], m < params.M && k < params.K);
    }
    for (var t: u32 = tid; t < TILE_K * BN; t = t + THREADS) {
      let kk = t / BN;
      let cc = t % BN;
      let k = k0 + kk;
      let n = n0 + cc;
      sb[t] = select(0.0, b[k * params.N + n], k < params.K && n < params.N);
    }
    workgroupBarrier();

    for (var kk: u32 = 0u; kk < TILE_K; kk = kk + 1u) {
      // The whole point: each of these is read once and used TILE_N (or
      // TILE_M) times out of registers.
      var va: array<f32, TILE_M>;
      for (var i: u32 = 0u; i < TILE_M; i = i + 1u) { va[i] = sa[(ty * TILE_M + i) * TILE_K + kk]; }
      var vb: array<f32, TILE_N>;
      for (var j: u32 = 0u; j < TILE_N; j = j + 1u) { vb[j] = sb[kk * BN + tx * TILE_N + j]; }
      for (var i: u32 = 0u; i < TILE_M; i = i + 1u) {
        for (var j: u32 = 0u; j < TILE_N; j = j + 1u) {
          acc[i][j] = fma(va[i], vb[j], acc[i][j]);
        }
      }
    }
    workgroupBarrier();
  }

  for (var i: u32 = 0u; i < TILE_M; i = i + 1u) {
    let m = m0 + ty * TILE_M + i;
    if (m >= params.M) { continue; }
    for (var j: u32 = 0u; j < TILE_N; j = j + 1u) {
      let n = n0 + tx * TILE_N + j;
      if (n < params.N) { c[m * params.N + n] = acc[i][j]; }
    }
  }
}
`;
}

/** How many workgroups a shape needs for `[M, N]`. */
export function gridFor(shape: TileShape, M: number, N: number): [number, number, number] {
  return [Math.ceil(N / (shape.wgN * shape.tileN)), Math.ceil(M / (shape.wgM * shape.tileM)), 1];
}
