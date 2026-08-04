// SnakeBeta: x + (1/(β_c + 1e-9)) · sin²(α_c · x), α and β learned per channel.
//
// The second entry point of this op, and the form BigVGAN calls `SnakeBeta` —
// MioCodec's decoder is built from it. `kernel.wgsl` is the one-parameter
// DACVAE form; this one separates the sine's frequency (α) from the amplitude
// it is added at (β). Setting β = α reproduces `kernel.wgsl` exactly.
//
// Two entry points rather than one kernel with an optional binding: since #46
// an unreferenced binding throws rather than quietly reading zeros, so the
// α-only path would have to bind a β buffer it does not use on every call.
// See `reference.ts` for the whole decision.
//
// What is upstream's rather than a choice here:
//
//   - The epsilon guards the DIVISOR, which in this form is β, not α. Same
//     value and same placement as `kernel.wgsl`, different parameter.
//     `candle-miotts` writes `(beta + 1e-9).recip()`.
//   - α and β are ACTUAL VALUES. A BigVGAN-family checkpoint stores their
//     logarithms and its loader exponentiates; doing that here would break
//     every checkpoint that stores them plainly. See `reference.ts` for how
//     that error fails — silently, as the identity function.
//   - sin²(αx) is the SQUARE OF THE SINE, not the sine of a square.
//
// Layout:
//   input:  [N, C, L] f32, contiguous
//   alpha:  [C]       f32
//   beta:   [C]       f32
//   output: [N, C, L] f32

struct Params {
  N: u32,
  C: u32,
  L: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> alpha: array<f32>;
@group(0) @binding(2) var<storage, read> beta: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
) {
  let idx = gid.x;
  if (idx >= params.N * params.C * params.L) {
    return;
  }

  // The channel this element belongs to: divide out the length, then drop the
  // batch. Both parameters are indexed by it, and by the same expression — a
  // kernel that got this right for α and wrong for β would still produce
  // plausible audio.
  let c = (idx / params.L) % params.C;
  let a = alpha[c];
  let b = beta[c];
  let x = input[idx];

  let s = sin(a * x);
  output[idx] = x + (1.0 / (b + 1e-9)) * (s * s);
}
