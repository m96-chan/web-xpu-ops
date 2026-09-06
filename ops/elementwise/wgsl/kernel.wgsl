// Elementwise operations: add, multiply
//
// Used for residual connections and gating.
//
// Layout:
//   a:      [N] f32
//   b:      [N] f32
//   output: [N] f32

// A dispatch wider than one row of workgroups is folded back into one index
// here rather than by a uniform: `num_workgroups.x` is what the host asked for,
// so **every existing one-dimensional caller keeps working unchanged** — at
// `[n]` the y extent is 1 and `gid.y` is 0. The ceiling is 65,535 workgroups on
// every backend measured (#211), which at 256 threads is 16.7 M elements, and
// `examples/h3-encoder`'s first level passes that on a 256x256 reference with
// five frames. `ops/pad` solves the same problem with a `stride_y` uniform;
// this way needs no caller to change.

struct Params {
  N: u32,
  op: u32,  // 0 = add, 1 = multiply
}

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(num_workgroups) workgroups: vec3<u32>,
) {
  let idx = gid.x + gid.y * workgroups.x * 256u;
  if (idx >= params.N) {
    return;
  }

  if (params.op == 0u) {
    output[idx] = a[idx] + b[idx];
  } else {
    output[idx] = a[idx] * b[idx];
  }
}
