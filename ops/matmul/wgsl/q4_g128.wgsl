// matmulQ4G128 (W4A32 GEMM): output[n, m] = sum_k a[n, k] * unpack(weight[m, k]) * scale[m, k/128]
//
// Layout:
//   a:      [N, K] f32, row-major — matmul's own `a` operand, unchanged.
//   weight: [M, ceil(K/8)] u32, row-major, 8 int4 codes packed per word,
//           least-significant nibble first — matvecQ4G128's own wire format
//           (ops/matvec/wgsl/q4_g128.wgsl), read here without transposing.
//   scale:  [M, ceil(K/128)] f32, row-major — one absmax-derived factor per
//           group of GROUP_SIZE contracted positions. Not [M]: at four bits a
//           single scale per row annihilates every column two orders of
//           magnitude below the row's peak (measured — README, "The q4
//           format"), which is why this op exists rather than a 4-bit
//           matmulQ8.
//   output: [N, M] f32, row-major
//
// Built on ops/matmul/wgsl/q8.wgsl's tiled structure, which is itself
// ops/matmul/wgsl/kernel.wgsl's: one workgroup owns one TILE x TILE block of
// output and walks K a tile at a time, so every value staged into workgroup
// memory is read TILE times instead of once. The only change from q8.wgsl is
// what feeds `tile_b` — a nibble instead of a byte, and a scale indexed by
// (weight row, group of k) instead of by weight row alone.
//
// The scale lookup is per element here, not hoisted the way
// ops/matvec/wgsl/q4_g128.wgsl hoists it to one lookup per packed word. It
// could be hoisted to one per tile — TILE = 16 divides GROUP_SIZE = 128, so a
// K-tile lies entirely within one group — but that would be an optimization
// with nothing measured behind it (rule 8: the plain form has to agree with
// the reference first), and the load is already once per TILE reuses rather
// than once per multiply-add.
//
// TILE = 16 — must match ops/matmul/wgsl/kernel.wgsl's own TILE,
// ops/matmul/wgsl/q8.wgsl's, and llm/engine-q8-resident.ts#MATMUL_TILE (rule
// 2: copied, not re-derived). Changing it means changing q4_g128.wgsl.test.ts
// too.
//
// Dispatch limits are the caller's, as they are for every other kernel in this
// directory: `maxComputeWorkgroupsPerDimension` (65535) is exceeded by
// ceil(M/16) at a real vocabulary size, and WebGPU answers an over-large
// dispatch with silence and zeros rather than an error (#112). llm/kernels.ts
// carries the guard for the engine's own dispatches; a new caller needs its
// own.

struct Params {
  N: u32,
  M: u32,
  K: u32,
}

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<u32>;
@group(0) @binding(2) var<storage, read> scale: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

const TILE: u32 = 16u;
const GROUP_SIZE: u32 = 128u;
/** Codes per packed word. GROUP_SIZE must stay a multiple of this. */
const CODES_PER_WORD: u32 = 8u;

var<workgroup> tile_a: array<array<f32, TILE>, TILE>;
var<workgroup> tile_b: array<array<f32, TILE>, TILE>;

// `ops/matvec/wgsl/q4_g128.wgsl#unpack_i4`, copied rather than imported — WGSL
// has no cross-file include, and every kernel here that needs an unpack makes
// the same choice (`ops/matmul/wgsl/q8.wgsl`'s own doc).
fn unpack_i4(word: u32, lane: u32) -> f32 {
  return f32(extractBits(bitcast<i32>(word), lane * 4u, 4u));
}

@compute @workgroup_size(TILE, TILE)
fn main(
  @builtin(workgroup_id) wg_id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>,
) {
  let lx = local_id.x;
  let ly = local_id.y;
  // `row` walks N (a's rows / output's rows); `col` walks M (weight's rows /
  // output's columns) — ops/matmul/wgsl/q8.wgsl's own naming, kept so the two
  // quantized GEMMs read the same way.
  let row = wg_id.y * TILE + ly;
  let col = wg_id.x * TILE + lx;
  let words_per_row = (params.K + CODES_PER_WORD - 1u) / CODES_PER_WORD;
  let groups_per_row = (params.K + GROUP_SIZE - 1u) / GROUP_SIZE;

  // No early return for invocations off the edge of output — same reasoning as
  // ops/matmul/wgsl/kernel.wgsl: every lane still has to reach every barrier so
  // the ones that stayed do not hang.
  var acc: f32 = 0.0;
  let k_tiles = (params.K + TILE - 1u) / TILE;
  for (var t: u32 = 0u; t < k_tiles; t += 1u) {
    let k_base = t * TILE;
    // Ragged K tail: `k_len` is how much of this tile is real K, and lanes past
    // it are neither written nor read.
    let k_len = min(TILE, params.K - k_base);

    if (row < params.N && lx < k_len) {
      tile_a[ly][lx] = a[row * params.K + k_base + lx];
    }
    // tile_b[ly][lx] holds weight[col, k_base + ly] * scale[col, (k_base+ly)/128]
    // — `ly` is the local K index, `lx` selects which output feature (`col`)
    // this thread column is responsible for.
    //
    // Both halves of the guard protect the *reads* above all else, the same way
    // ops/matmul/wgsl/q8.wgsl's own equivalent does (its doc works through why
    // neither half is load-bearing for `output` itself): without `ly < k_len`,
    // `k = k_base + ly` runs past `K` in the ragged tile, so `k / 8` can reach
    // `words_per_row` — the next weight row's first word, or past the buffer
    // for the last row — and `k / 128` can reach `groups_per_row`, which is the
    // next row's first scale. Without `col < params.M`, both offsets walk past
    // the last row entirely. WGSL does not promise an out-of-bounds storage
    // read is a no-op, only that it will not reach another resource.
    if (col < params.M && ly < k_len) {
      let k = k_base + ly;
      let word = weight[col * words_per_row + k / CODES_PER_WORD];
      let group_scale = scale[col * groups_per_row + k / GROUP_SIZE];
      tile_b[ly][lx] = unpack_i4(word, k % CODES_PER_WORD) * group_scale;
    }
    workgroupBarrier();

    for (var k: u32 = 0u; k < k_len; k += 1u) {
      acc += tile_a[ly][k] * tile_b[k][lx];
    }
    workgroupBarrier();
  }

  // `col < M` matters: past the last output feature, row * M + col is
  // output[row + 1][col - M] — a live element of the next row, silently
  // overwritten. `row < N` is hygiene against writing off the end of the
  // buffer (same reasoning as plain matmul's own store guard).
  if (row < params.N && col < params.M) {
    output[row * params.M + col] = acc;
  }
}
