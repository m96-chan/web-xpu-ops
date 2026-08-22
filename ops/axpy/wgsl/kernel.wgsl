// axpy: out[i] = y[i] + a * x[i], with `a` a scalar
//
// One dispatch for a rectified-flow scheduler's `latent += dt * velocity`,
// which `ops/elementwise` cannot express: it takes two equally sized arrays,
// so a scalar coefficient has to be materialised as a full-length buffer and
// run through multiply-then-add. See `ops/axpy/reference.ts`.
//
// Layout:
//   x:      [N] f32   the vector scaled by `a`  (the velocity field)
//   y:      [N] f32   the vector added to       (the latent)
//   output: [N] f32
//
// `y` and `output` must be **different buffers**. Binding one buffer to both
// is not a compile error and not a bind-group error — measured on this device
// (Dawn / Vulkan / RTX 5090): `createBindGroup` succeeds, and the command
// buffer is then invalidated at `finish()` with
//
//   [Buffer] usage (Storage(read-write)|Storage(read-only)) includes writable
//   usage and another usage in the same synchronization scope
//
// so the whole submit is dropped and the readback comes back **all zeros**,
// with nothing thrown on the JS side. That is #46's silent-zeros failure mode
// wearing a different hat. In-place is a real requirement, not a mistake, and
// it has its own entry point — `inplace.wgsl` — where `y` is a single
// read_write binding rather than an aliased pair.
//
// The arithmetic is written as one expression on purpose. This device's WGSL
// compiler contracts `y + a * x` into an FMA — measured, not assumed: at
// a = f32(0.1), x = 3, y = -f32(0.3) it returns -2^-27, the single-rounded
// answer, where multiply-then-add cancels to exactly 0. That is the answer
// `torch.add(y, x, alpha=a)` gives on CPU and CUDA alike, so the fused kernel
// is closer to PyTorch than the two dispatches it replaces, not merely faster.

struct Params {
  N: u32,
  // The scalar. In the uniform rather than in a storage buffer because it
  // changes every diffusion step and nothing else in this block does: 16 bytes
  // rewritten per step, next to a dispatch that moves 3N floats. See
  // `ops/axpy/inplace.wgsl.test.ts` for the resident-loop shape this implies.
  a: f32,
}

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> y: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
) {
  let idx = gid.x;
  if (idx >= params.N) {
    return;
  }

  output[idx] = y[idx] + params.a * x[idx];
}
