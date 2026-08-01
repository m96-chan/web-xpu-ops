// PoPE — Legendre orthogonal polynomial position encoding.
// Aggarwal, arXiv:2405.04585v1, Equation (14):
//
//   PE(pos, i) = P_pos(x_i),  x_i = -1 + 2i/d_model
//
// The polynomial order is the token position and the argument is the feature
// index swept across [-1, 1) — the opposite assignment from the sinusoidal
// scheme, where the feature index picks a frequency. `pos_offset` decides
// whether positions start at 0 or 1, which the paper does not state; see
// `reference.ts` for why that is the caller's decision here.
//
// Evaluated by the three-term recurrence the paper quotes as Equation (15):
//
//   P_{n+1}(x) = ((2n + 1) x P_n(x) - n P_{n-1}(x)) / (n + 1)
//
// Rodrigues' formula (Equation 12) is the compact definition, not a method: the
// explicit coefficient sum overflows and cancels well before the orders a real
// sequence length needs. The recurrence stays inside [-1, 1] at every step
// because |P_n(x)| <= 1 on the domain, which is the only reason an f32 kernel
// can walk it at all.
//
// Cost is O(pos) per lane, so the last row is the slowest and the lanes within
// a workgroup diverge in trip count. Left alone: rule 8 — this agrees with the
// reference first, and a prefix-shared version can come after there is a
// roofline to judge it against.
//
// Layout:
//   encoding: [N, d_model] f32
//   Dispatched one lane per element.

struct Params {
  N: u32,          // tokens
  d_model: u32,    // model width, and the number of samples of [-1, 1)
  pos_offset: u32, // position of the first row
}

@group(0) @binding(0) var<storage, read_write> encoding: array<f32>;
@group(0) @binding(1) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
) {
  let total = params.N * params.d_model;
  let index = gid.x;
  if (index >= total) {
    return;
  }

  let i = index % params.d_model;
  let token = index / params.d_model;
  let order = token + params.pos_offset;

  let x = -1.0 + 2.0 * f32(i) / f32(params.d_model);

  var previous = 1.0; // P_0
  var current = x;    // P_1
  for (var k = 1u; k < order; k = k + 1u) {
    let kf = f32(k);
    let next = ((2.0 * kf + 1.0) * x * current - kf * previous) / (kf + 1.0);
    previous = current;
    current = next;
  }

  // Order 0 is the constant polynomial, and `current` was seeded with P_1. This
  // is reachable whenever pos_offset is 0, and the two answers differ by 1 - x
  // at every column but the one where x happens to be 1.
  encoding[index] = select(current, 1.0, order == 0u);
}
