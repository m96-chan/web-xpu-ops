// Matmul (GEMM): C = A @ B, shared-memory tiled.
//
// Layout:
//   a:      [M, K] f32, row-major
//   b:      [K, N] f32, row-major
//   output: [M, N] f32, row-major
//
// One workgroup owns one TILE x TILE block of C. It walks K a tile at a time,
// staging A's block and B's block in workgroup memory, so each loaded value is
// used TILE times instead of once. That reuse is the whole reason this op is
// separate from GEMV, which has none to find.
//
// TILE = 16 is not a measured optimum — nothing here is tuned yet (see #3/#4
// for the roofline harness). It is the plain choice that fits the limits with
// room to grow:
//   * 16 x 16 = 256 invocations per workgroup, the same width the other ops in
//     this repo use, and well under maxComputeInvocationsPerWorkgroup (1024).
//   * two f32 tiles = 2 * 16 * 16 * 4 = 2048 bytes of workgroup storage, against
//     maxComputeWorkgroupStorageSize (49152), so a later register-blocked or
//     double-buffered variant has somewhere to go.
//   * one output per invocation, which keeps the indexing readable. Correctness
//     first (rule 8); the register blocking that makes this fast comes after
//     there is a number to improve on.
// Changing TILE here means changing it in wgsl.test.ts too — the ragged shapes
// are chosen around it.

struct Params {
  M: u32,
  N: u32,
  K: u32,
}

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

const TILE: u32 = 16u;

var<workgroup> tile_a: array<array<f32, TILE>, TILE>;
var<workgroup> tile_b: array<array<f32, TILE>, TILE>;

@compute @workgroup_size(TILE, TILE)
fn main(
  @builtin(workgroup_id) wg_id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>,
) {
  let lx = local_id.x;
  let ly = local_id.y;
  let row = wg_id.y * TILE + ly;
  let col = wg_id.x * TILE + lx;

  // No early return for invocations off the edge of C. They have no output to
  // write, but they still have to load their share of the tiles and reach every
  // barrier — leaving early would hang the ones that stayed and would leave the
  // tile half filled.
  var acc: f32 = 0.0;
  let k_tiles = (params.K + TILE - 1u) / TILE;
  for (var t: u32 = 0u; t < k_tiles; t += 1u) {
    let k_base = t * TILE;

    // The ragged K tail lives here and nowhere else: `k_len` is how much of this
    // tile is real, and lanes past it are neither written nor read. The obvious
    // alternative — pad the tiles with zeros and always run the full TILE — is
    // not used, because then the padding and the loop bound each mask the other:
    // zeroing either factor makes the product vanish, so removing one of them
    // leaves the tests green and the guard untested. One mechanism, one thing to
    // break. (This is uniform across the workgroup, so the barriers below stay
    // in uniform control flow.)
    let k_len = min(TILE, params.K - k_base);

    // A lane whose row is past M, or whose column is past N, leaves its slot
    // holding whatever the previous tile left there. That is safe and deliberate:
    // tile_a[ly][*] is only ever read by lanes with this same `ly` — the same
    // row — and tile_b[*][lx] only by lanes with this same `lx`, so a stale slot
    // can only reach an accumulator that is thrown away at the store below.
    if (row < params.M && lx < k_len) {
      tile_a[ly][lx] = a[row * params.K + k_base + lx];
    }
    if (col < params.N && ly < k_len) {
      tile_b[ly][lx] = b[(k_base + ly) * params.N + col];
    }
    workgroupBarrier();

    for (var k: u32 = 0u; k < k_len; k += 1u) {
      acc += tile_a[ly][k] * tile_b[k][lx];
    }
    // Before overwriting the tiles on the next pass, everyone must be done
    // reading them.
    workgroupBarrier();
  }

  // The two halves are not equally load-bearing, and the difference is worth
  // knowing rather than assuming. `col < N` is the one that matters: past the
  // last column, row * N + col is C[row + 1][col - N] — a live element of the
  // next row, silently overwritten with this invocation's accumulator. Dropping
  // it turns the N-tail tests red immediately.
  //
  // `row < M` is hygiene. Past the last row the index is off the end of the
  // buffer, and this implementation discards the write, so dropping it leaves
  // every test green (checked, by mutation). It stays because WGSL does not
  // promise that an out-of-bounds write is a no-op — only that it will not
  // reach another resource.
  if (row < params.M && col < params.N) {
    output[row * params.N + col] = acc;
  }
}
