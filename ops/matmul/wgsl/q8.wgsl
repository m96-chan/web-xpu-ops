// matmulQ8 (W8A32 GEMM): output[n, m] = (sum_k a[n, k] * unpack(weight[m, k])) * scale[m]
//
// Layout:
//   a:      [N, K] f32, row-major — matmul's own `a` operand, unchanged.
//   weight: [M, ceil(K/4)] u32, row-major, 4 int8 codes packed per word,
//           least-significant byte first — matvecQ8's own wire format
//           (ops/matvec/wgsl/q8.wgsl), read here without transposing.
//   scale:  [M]    f32, one per weight row (`quantize`'s per-row absmax convention)
//   output: [N, M] f32, row-major
//
// Built on ops/matmul/wgsl/kernel.wgsl's tiled structure — one workgroup owns
// one TILE x TILE block of output, walking K a tile at a time so every value
// staged into workgroup memory is read TILE times instead of once. The only
// change from that kernel is what feeds `tile_b`: instead of a plain f32 load
// of `b[k, col]`, each lane unpacks its own byte from the packed weight word
// and multiplies in that column's scale before it ever lands in shared
// memory — `matvecQ8`'s own "scale applied once, not per k-term" trick
// (ops/matvec/wgsl/q8.wgsl's doc), except here the value is computed once per
// *tile* load and then reused TILE times by every row sharing this
// workgroup, not once per row.
//
// TILE = 16 — must match ops/matmul/wgsl/kernel.wgsl's own TILE and
// llm/engine-q8-resident.ts#MATMUL_TILE (rule 2: copied, not re-derived).
// Changing it here means changing it in q8.wgsl.test.ts too.

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

var<workgroup> tile_a: array<array<f32, TILE>, TILE>;
var<workgroup> tile_b: array<array<f32, TILE>, TILE>;

// `ops/matvec/wgsl/q8.wgsl#unpack_i8`, copied rather than imported — WGSL has
// no cross-file include, and every WGSL kernel in this repository that needs
// this makes the same choice (`ops/dequant_transpose/wgsl/kernel.wgsl`'s own
// doc).
fn unpack_i8(word: u32, lane: u32) -> f32 {
  return f32(extractBits(bitcast<i32>(word), lane * 8u, 8u));
}

@compute @workgroup_size(TILE, TILE)
fn main(
  @builtin(workgroup_id) wg_id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>,
) {
  let lx = local_id.x;
  let ly = local_id.y;
  // `row` walks N (a's rows / output's rows); `col` walks M (weight's rows /
  // output's columns) — matmul's own `row`/`col` naming, with N and M swapped
  // from that kernel's M/N because this op's uniform names the dimensions the
  // way `MatmulQ8Args` (reference.ts) does, not the way plain `matmul` does.
  let row = wg_id.y * TILE + ly;
  let col = wg_id.x * TILE + lx;
  let words_per_row = (params.K + 3u) / 4u;

  // No early return for invocations off the edge of output — same reasoning
  // as ops/matmul/wgsl/kernel.wgsl: every lane still has to reach every
  // barrier so the ones that stayed do not hang.
  var acc: f32 = 0.0;
  let k_tiles = (params.K + TILE - 1u) / TILE;
  for (var t: u32 = 0u; t < k_tiles; t += 1u) {
    let k_base = t * TILE;

    // Ragged K tail, same mechanism as plain matmul: `k_len` is how much of
    // this tile is real K, and lanes past it are neither written nor read —
    // no zero-padding to double up with the loop bound (ops/matmul/wgsl/kernel.wgsl's
    // own doc explains why one mechanism, not two, matters here).
    let k_len = min(TILE, params.K - k_base);

    if (row < params.N && lx < k_len) {
      tile_a[ly][lx] = a[row * params.K + k_base + lx];
    }
    // tile_b[ly][lx] holds weight[col, k_base + ly] * scale[col] — `ly` is
    // the local K index here (matching plain matmul's `tile_b[ly][lx] =
    // b[(k_base+ly)*N+col]` load), `lx` selects which output feature (`col`)
    // this lane's own thread column is responsible for.
    //
    // Neither half of this guard is load-bearing for the *output* — checked
    // by mutation, the same way plain matmul's own `row < M` tile_a guard is
    // labelled hygiene rather than assumed to be: removing either half alone
    // leaves every test green, because what each one guards is a *write* into
    // `tile_b`, and that write's value never reaches `acc` regardless — the
    // reduction loop below reads `tile_b[k][lx]` only for `k < k_len` (so a
    // slot written at `ly >= k_len` is never read), and an invocation with
    // `col >= params.M` has its own `acc` discarded by the store guard at the
    // bottom (no *other* lane ever reads `tile_b[*][lx]` at a different `lx`).
    //
    // Both halves are still real guards on the `weight` *read* just above
    // this comment, though, not just on the `tile_b` write — the read and the
    // write share one `if`, so losing either half changes what gets read, not
    // only what gets written. `ly < k_len`'s absence lets `k = k_base + ly`
    // run to `K` or past it in the tile covering the ragged end of `K`, so
    // `k >> 2u` can reach `words_per_row` or beyond — reading the *next*
    // weight row's first word or two (or past the whole buffer, for the last
    // row). `col < params.M`'s absence lets `col * words_per_row` itself walk
    // past the last row entirely. WGSL does not promise an out-of-bounds
    // storage read is a no-op, only that it will not reach another resource,
    // so both stay — the same reason plain matmul keeps its own hygiene guard
    // — even though neither read's result ever surfaces in `output`.
    if (col < params.M && ly < k_len) {
      let k = k_base + ly;
      let word = weight[col * words_per_row + (k >> 2u)];
      tile_b[ly][lx] = unpack_i8(word, k & 3u) * scale[col];
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
  // buffer — WGSL does not promise that is a no-op, only that it will not
  // reach another resource (same reasoning as plain matmul's own store guard).
  if (row < params.N && col < params.M) {
    output[row * params.M + col] = acc;
  }
}
