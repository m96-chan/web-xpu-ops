// Elementwise add/multiply with `b` broadcast along the last dimension:
//
//   output[s, d] = a[s, d] (+ or *) b[d]
//
// Issue #150. `op == 0` is biasAdd (a Linear's bias over every row of its
// [S, D] output); `op == 1` is rowwiseAffine (AdaLN's per-channel scale).
//
// A separate entry point from `kernel.wgsl` rather than a flag on it. That
// kernel is bound, today, by `llm/engine-q8-resident.ts` (residual add, SwiGLU
// multiply) and by `harness/resident.test.ts` with a two-field Params; adding a
// field would have changed the struct every one of those call sites uploads,
// so "the same-shape path is unchanged" would have needed an argument instead
// of a diff that does not touch it. It also costs those callers nothing: they
// do not pay the `%` this kernel needs.
//
// Layout:
//   a:      [S, D] f32, row-major
//   b:      [D]    f32
//   output: [S, D] f32
//
// One thread per output element, flat over S*D, so the dispatch is sized the
// same way `kernel.wgsl`'s is (ceil(S*D / 256)) and the row is recovered from
// the index rather than from a second workgroup dimension. D is not assumed to
// be a multiple of anything, so the column comes from a modulo, not a mask.

struct Params {
  S: u32,
  D: u32,
  op: u32,  // 0 = add, 1 = multiply
}

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
) {
  let idx = gid.x;
  if (idx >= params.S * params.D) {
    return;
  }

  // The column within the row — the only index `b` is ever read at. Reading it
  // at `idx / params.D` (the row) would still be in bounds whenever S <= D and
  // would still produce finite numbers, which is why the tests include a square
  // case where the two differ.
  let col = idx % params.D;
  let scalar = b[col];

  if (params.op == 0u) {
    output[idx] = a[idx] + scalar;
  } else {
    output[idx] = a[idx] * scalar;
  }
}
