// axpy, in place: y[i] += a * x[i] — BLAS `saxpy`'s own signature
//
// The second entry point of `ops/axpy`, and the one a sampler loop wants.
// `kernel.wgsl` writes a third buffer, so a caller stepping a latent has to
// ping-pong two latent-sized allocations and swap them every step. This one
// updates the latent where it already lives: one fewer allocation of the same
// size, one fewer thing for the caller to get wrong, and the same 3N floats of
// traffic either way (read x, read y, write y).
//
// Layout:
//   x: [N] f32        read only  — the vector scaled by `a` (the velocity field)
//   y: [N] f32        read_write — read and written in place (the latent)
//
// ## Why one read_write binding rather than `kernel.wgsl` with `y` bound twice
//
// Because the aliased spelling fails silently. Measured on this device (Dawn /
// Vulkan / RTX 5090): binding one buffer to `kernel.wgsl`'s read-only `y` and
// its read_write `output` passes `createBindGroup` without an error, then
// invalidates the command buffer at `finish()` —
//
//   [Buffer] usage (Storage(read-write)|Storage(read-only)) includes writable
//   usage and another usage in the same synchronization scope
//
// — so the submit is dropped and the readback is all zeros, with nothing
// thrown on the JS side to say so. One `read_write` binding read and written
// by the same invocation is a different thing entirely, and is well defined:
// invocation `i` touches element `i` and no other, so there is no
// cross-invocation ordering to depend on and no barrier to need.
//
// ## Where the scalar lives, and why it is still a uniform
//
// `a` changes every step; `x` and `y` do not move. Keeping `a` in the uniform
// means the per-step cost is one 16-byte `queue.writeBuffer` — the same shape
// `harness/resident.ts` already sanctions for position counters — and nothing
// else is re-uploaded, which is the trap #144 fixed (a whole gigabyte of
// unchanged weight bytes re-packed and re-sent per call). Recording several
// steps with different `a` into **one** submit needs one 16-byte uniform
// buffer per step, since writes to a single uniform collapse to the last one
// before the submit that reads it; that is a buffer per step, not a binding
// per dispatch, so the layout below does not change to allow it.

struct Params {
  N: u32,
  a: f32,
}

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> y: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
) {
  let idx = gid.x;
  if (idx >= params.N) {
    return;
  }

  y[idx] = y[idx] + params.a * x[idx];
}
