// Matmul (GEMM): C = A @ B, two-level tiled.
//
// Layout:
//   a:      [M, K] f32, row-major
//   b:      [K, N] f32, row-major
//   output: [M, N] f32, row-major
//
// One workgroup owns a BM x BN block of C. It walks K in TILE_K chunks staged
// into workgroup memory, and each thread holds a TILE_M x TILE_N block of the
// output **in registers**, so every staged value is used TILE_N (or TILE_M)
// times rather than once.
//
// That second level is what this kernel used to lack. It said of itself that
// `TILE = 16 is not a measured optimum — nothing here is tuned yet`, and it was
// right: one output per invocation reached 1.4-3.0% of an RTX 5090's measured
// roofline. This reaches 70-72%.
//
// | shape (M x K x N)      | one-per-thread | this   |
// | ---------------------- | -------------- | ------ |
// | 3952 x 2048 x 2048     | 1.4%           | 71.7%  |
// | 3952 x 2048 x 8192     | 1.9%           | 72.4%  |
// | 3952 x 8192 x 2048     | 3.0%           | 70.1%  |
//
// **Measured, not chosen.** `tools/bench.ts` generates the 345 tile shapes this
// device can dispatch, checks every one against `reference.ts` on a ragged
// 37x43x29 *before* timing it, and times the survivors at the dimensions a real
// model uses. All 345 were correct; this shape won all three sizes. Rerun it on
// other hardware rather than trusting these numbers there — the constants are
// this device's answer, not a universal one.
//
// The structure is llama.cpp's `mul_mat_reg_tile.wgsl` (MIT, from the
// `ggml-webgpu` backend), which is CUTLASS's decomposition. The constants are
// not theirs: they stage in f16, and a device without `shader-f16` pays twice
// the workgroup storage for the same tile.
//
// Two device limits have to be asked for or most of this cannot be dispatched —
// `maxComputeWorkgroupStorageSize` (default 16384, this adapter 49152) and
// `maxComputeWorkgroupSizeX` (default 256, this adapter 1024). All three
// runtimes in this repository request them; a device that cannot grant them
// cannot run this kernel.
//
// **The grid is not one workgroup per 16x16 output.** A caller dispatches
// `[ceil(N / BN), ceil(M / BM), 1]`, and `matmulGrid` in `index.ts` is that
// arithmetic — exported so no caller writes 64 and 128 down a second time, the
// way `TILE = 16` used to be copied into `llm/kernels.ts` with a comment asking
// for it to be kept in step.
//
// Every load and store is guarded. Real M values are not multiples of anything
// convenient — Anima's is 3,952 — and a kernel correct only on aligned shapes
// is one the sweep could not have measured where it matters.

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
