// Tiled matmul: `a: [M, K]` by `b: [K, N]` to `c: [M, N]`, all row-major.
//
// The same function as `kernel.wgsl` next door, computed with a two-level
// decomposition instead of one output per thread. That one says of itself that
// `TILE = 16 is not a measured optimum`, and it reaches 1.4-3.0% of an RTX
// 5090's measured roofline. This reaches 70-72%.
//
// | shape (M x K x N)      | kernel.wgsl | this   |
// | ---------------------- | ----------- | ------ |
// | 3952 x 2048 x 2048     | 1.4%        | 71.7%  |
// | 3952 x 2048 x 8192     | 1.9%        | 72.4%  |
// | 3952 x 8192 x 2048     | 3.0%        | 70.1%  |
//
// Measured, not chosen: `tools/bench.ts` sweeps 345 shapes that fit this
// device's limits, checks every one against `reference.ts` on a ragged
// 37x43x29 first, and this shape won all three. Conditions and the sweep are in
// that file; rerun it on different hardware rather than trusting these numbers
// there.
//
// The structure is llama.cpp's `mul_mat_reg_tile.wgsl` (MIT, `ggml-webgpu`),
// which is CUTLASS's decomposition: a workgroup tile walked over K in chunks
// staged into workgroup memory, and a **thread tile** held in registers so each
// staged value is used TILE_N (or TILE_M) times rather than once. The constants
// are not theirs — they stage in f16, this device has no `shader-f16`, and
// their model's dimensions are not this one's.
//
// **The grid is not `kernel.wgsl`'s.** One workgroup covers BM x BN outputs,
// so a caller dispatches `[ceil(N / BN), ceil(M / BM), 1]`. `matmulTiledGrid`
// in `index.ts` is that arithmetic, exported so no caller writes 64 and 128
// down a second time.
//
// Every load and store is guarded. Anima's M is 3,952 and no tile divides it;
// a kernel correct only on aligned shapes would be one the sweep could not
// have measured on the shapes that matter.

struct Params { M: u32, N: u32, K: u32 }

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

const WG_M: u32 = 16u;
const WG_N: u32 = 32u;
const TILE_M: u32 = 4u;
const TILE_N: u32 = 4u;
const TILE_K: u32 = 16u;
const BM: u32 = 64u;
const BN: u32 = 128u;
const THREADS: u32 = 512u;

// A as [BM, TILE_K] and B as [TILE_K, BN], both row-major in one array so the
// two staging loops are the same shape.
var<workgroup> sa: array<f32, 1024>;
var<workgroup> sb: array<f32, 2048>;

@compute @workgroup_size(512)
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
