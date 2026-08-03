// Snake: x + (1/(α_c + 1e-9)) · sin²(α_c · x), α learned per channel.
//
// The periodic activation from arXiv:2006.08195, in the form DACVAE and
// BigVGAN use (`Snake1d`). Its α is a trained tensor with one entry per
// channel, which is why this is its own op and not a kind of `activation` —
// see `reference.ts` for that decision and its reason.
//
// Two things are upstream's rather than choices, and both are easy to
// "improve" into something subtly different:
//
//   - The epsilon is INSIDE the reciprocal and it is 1e-9. Not `1/α` guarded
//     afterwards, and not `α/(α² + ε)`. At α = 0 this is 1e9 · sin²(0) = 0,
//     where `1/α` is inf · 0 = NaN.
//   - sin²(αx) is the SQUARE OF THE SINE, not the sine of a square.
//
// Layout:
//   input:  [N, C, L] f32, contiguous
//   alpha:  [C]       f32
//   output: [N, C, L] f32

struct Params {
  N: u32,
  C: u32,
  L: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> alpha: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
) {
  let idx = gid.x;
  if (idx >= params.N * params.C * params.L) {
    return;
  }

  // The channel this element belongs to: divide out the length, then drop the
  // batch. Both halves matter — without the division α would be indexed by
  // position, and without the modulo it would run off the end of α on the
  // second batch item.
  let c = (idx / params.L) % params.C;
  let a = alpha[c];
  let x = input[idx];

  let s = sin(a * x);
  output[idx] = x + (1.0 / (a + 1e-9)) * (s * s);
}
