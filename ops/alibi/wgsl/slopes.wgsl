// ALiBi per-head slopes (Press et al., arXiv:2108.12409).
//
// One f32 per head, in closed form. The paper's own implementation is a
// recursion over Python lists; that shape has nothing for a GPU to do, and the
// recursion only ever descends one level anyway, because a head count is always
// less than twice the closest power of two below it.
//
// The derivation, so the two branches below can be read against the paper:
//
//   start    = 2^(-2^-(log2(n) - 3)) = 2^(-8/n)
//   slope[h] = start * start^h = start^(h+1) = 2^(-8(h+1)/n)
//
// For a power-of-two head count that is the whole answer, and it always ends at
// 2^-8 for the last head whatever n is. Otherwise `n` is the closest power of
// two *below* the head count, the run above covers heads [0, n), and the
// remainder takes every other slope of the run for 2n:
//
//   slope[n + t] = 2^(-8(2t + 1)/(2n)) = 2^(-4(2t + 1)/n)
//
// Note what this does not do: it does not interpolate, sort, or shorten a
// longer run. The appended slopes are *larger* than the ones before them, so
// the sequence is not monotonic. `reference.ts` keeps the paper's recursion and
// `reference.test.ts` pins the numbers, so this derivation is checked rather
// than trusted.
//
// Layout:
//   slopes: [num_heads] f32
//   Dispatched one lane per head.

struct Params {
  num_heads: u32,
}

@group(0) @binding(0) var<storage, read_write> slopes: array<f32>;
@group(0) @binding(1) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
) {
  let head = gid.x;
  if (head >= params.num_heads) {
    return;
  }

  // Largest power of two <= num_heads, by doubling rather than by
  // exp2(floor(log2(n))). log2 of an exact power of two is allowed to come back
  // a hair under the integer on hardware, and the floor would then land a
  // factor of two low and silently change every slope. Doubling is exact.
  var closest = 1u;
  while (closest * 2u <= params.num_heads) {
    closest = closest * 2u;
  }
  let n = f32(closest);

  if (head < closest) {
    slopes[head] = exp2(-8.0 * f32(head + 1u) / n);
  } else {
    let t = head - closest;
    slopes[head] = exp2(-4.0 * f32(2u * t + 1u) / n);
  }
}
