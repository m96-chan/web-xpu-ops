// Gather: output[n, :] = table[indices[n], :]
//
// Row selection, matching torch.index_select(table, 0, indices) — see
// reference.ts for why that rather than torch.gather, and for why an index
// outside [0, rows) yields a row of zeros.
//
// One thread per output element, flat over N*D. Not one workgroup per output
// row: embedding tables are read with a row width that has nothing to do with
// the workgroup size, and a row-per-workgroup mapping idles 255 of 256 lanes on
// the narrow ones. Flat keeps the write side contiguous whatever the indices do,
// which is the only part of the access pattern this op gets to choose — the read
// side is decided by data nobody controls.
//
// The caller dispatches ceil(N*D / 256) workgroups in x alone, so N*D is capped
// by maxComputeWorkgroupsPerDimension — 65535 on this adapter, about 16.7M
// elements. Large enough for the shapes tested here, not large enough for every
// embedding lookup; splitting across y is a change to make when something needs
// it, not before.
//
// Layout:
//   table:   [rows, D] f32
//   indices: [N]       i32
//   output:  [N, D]    f32

struct Params {
  N: u32,
  D: u32,
  rows: u32,
}

@group(0) @binding(0) var<storage, read> table: array<f32>;
@group(0) @binding(1) var<storage, read> indices: array<i32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let total = params.N * params.D;
  if (gid.x >= total) {
    return;
  }

  let n = gid.x / params.D;
  let d = gid.x % params.D;

  let row = indices[n];
  // Out of range gathers zeros rather than clamping or wrapping: both of those
  // return a real embedding for a token that was never in the vocabulary.
  //
  // One comparison covers both ends. WGSL's i32-to-u32 conversion is modular, so
  // a negative index lands at row + 2^32, far above any `rows` a table can have,
  // and an explicit `row < 0` beside this would be a branch no test could ever
  // take. The reference states the same rule the other way round, because JS
  // numbers do not wrap and there it has to be spelled out.
  if (u32(row) >= params.rows) {
    output[gid.x] = 0.0;
    return;
  }

  output[gid.x] = table[u32(row) * params.D + d];
}
