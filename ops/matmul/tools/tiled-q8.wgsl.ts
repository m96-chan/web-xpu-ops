/**
 * A tiled `matmulQ8` with its tile shape as a parameter, for sweeping.
 *
 * Issue #177. The same two-level decomposition `wgsl/kernel.wgsl` now uses —
 * see its header for where the structure comes from and why the constants are
 * measured rather than copied — applied to the packed path.
 *
 * **The layout is not the dense kernel's, and the difference is in this op's
 * favour.** `q8.wgsl` computes `C[N, M] = A[N, K] @ W[M, K]^T`: the weight is
 * stored as `[out, in]` and used transposed, so *both* operands run along `K`
 * contiguously. The dense kernel stages a `[K, N]` operand across the grain;
 * here both tiles stage the same way and the inner product walks the contiguous
 * axis of both.
 *
 * Dequantisation happens **once per staged element**, not once per use: a value
 * pulled into workgroup memory is unpacked and scaled there, and then read
 * `TILE_M` times out of it. The shipped kernel multiplies by `scale[col]` on
 * every staging too, but with one output per thread "every staging" and "every
 * use" are the same thing.
 */

import type { TileShape } from "./tiled.wgsl.js";

export { rejectReason as rejectQ8Reason } from "./tiled.wgsl.js";

/** Workgroup storage in bytes. Both tiles are f32 once dequantised. */
export function q8StorageBytes({ wgM, wgN, tileM, tileN, tileK }: TileShape): number {
  return (wgM * tileM * tileK + wgN * tileN * tileK) * 4;
}

/**
 * `a: [N, K]` f32 by packed `weight: [M, K]` i8 with `scale: [M]`, to
 * `output: [N, M]`.
 *
 * The parameter names are `q8.wgsl`'s own — `N` is the row count of `a` and `M`
 * the output-channel count, which is the opposite of the dense kernel's naming
 * and is kept rather than fixed so the two files can be read against each other.
 */
export function tiledMatmulQ8(shape: TileShape): string {
  const { wgM, wgN, tileM, tileN, tileK } = shape;
  const bm = wgM * tileM;
  const bn = wgN * tileN;
  const threads = wgM * wgN;
  return `
struct Params { N: u32, M: u32, K: u32 }

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<u32>;
@group(0) @binding(2) var<storage, read> scale: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

const WG_M: u32 = ${wgM}u;
const WG_N: u32 = ${wgN}u;
const TILE_M: u32 = ${tileM}u;
const TILE_N: u32 = ${tileN}u;
const TILE_K: u32 = ${tileK}u;
const BM: u32 = ${bm}u;
const BN: u32 = ${bn}u;
const THREADS: u32 = ${threads}u;

// Both [rows, TILE_K]: sa over rows of a, sb over output channels.
var<workgroup> sa: array<f32, ${bm * tileK}>;
var<workgroup> sb: array<f32, ${bn * tileK}>;

fn unpack_i8(word: u32, lane: u32) -> f32 {
  return f32(extractBits(bitcast<i32>(word), lane * 8u, 8u));
}

@compute @workgroup_size(${threads})
fn main(
  @builtin(workgroup_id) wg: vec3<u32>,
  @builtin(local_invocation_index) tid: u32,
) {
  let r0 = wg.y * BM;
  let c0 = wg.x * BN;
  let ty = tid / WG_N;
  let tx = tid % WG_N;
  let words_per_row = (params.K + 3u) / 4u;

  var acc: array<array<f32, TILE_N>, TILE_M>;
  for (var i: u32 = 0u; i < TILE_M; i = i + 1u) {
    for (var j: u32 = 0u; j < TILE_N; j = j + 1u) { acc[i][j] = 0.0; }
  }

  for (var k0: u32 = 0u; k0 < params.K; k0 = k0 + TILE_K) {
    for (var t: u32 = tid; t < BM * TILE_K; t = t + THREADS) {
      let r = t / TILE_K;
      let kk = t % TILE_K;
      let row = r0 + r;
      let k = k0 + kk;
      sa[t] = select(0.0, a[row * params.K + k], row < params.N && k < params.K);
    }
    for (var t: u32 = tid; t < BN * TILE_K; t = t + THREADS) {
      let c = t / TILE_K;
      let kk = t % TILE_K;
      let col = c0 + c;
      let k = k0 + kk;
      // Unpacked and scaled here, once, and read TILE_M times below.
      var value = 0.0;
      if (col < params.M && k < params.K) {
        let word = weight[col * words_per_row + (k >> 2u)];
        value = unpack_i8(word, k & 3u) * scale[col];
      }
      sb[t] = value;
    }
    workgroupBarrier();

    for (var kk: u32 = 0u; kk < TILE_K; kk = kk + 1u) {
      var va: array<f32, TILE_M>;
      for (var i: u32 = 0u; i < TILE_M; i = i + 1u) { va[i] = sa[(ty * TILE_M + i) * TILE_K + kk]; }
      var vb: array<f32, TILE_N>;
      for (var j: u32 = 0u; j < TILE_N; j = j + 1u) { vb[j] = sb[(tx * TILE_N + j) * TILE_K + kk]; }
      for (var i: u32 = 0u; i < TILE_M; i = i + 1u) {
        for (var j: u32 = 0u; j < TILE_N; j = j + 1u) {
          acc[i][j] = fma(va[i], vb[j], acc[i][j]);
        }
      }
    }
    workgroupBarrier();
  }

  for (var i: u32 = 0u; i < TILE_M; i = i + 1u) {
    let row = r0 + ty * TILE_M + i;
    if (row >= params.N) { continue; }
    for (var j: u32 = 0u; j < TILE_N; j = j + 1u) {
      let col = c0 + tx * TILE_N + j;
      if (col < params.M) { output[row * params.M + col] = acc[i][j]; }
    }
  }
}
`;
}
