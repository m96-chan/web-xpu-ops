// Rotary Position Embeddings (RoPE), with NTK and YaRN context scaling.
//
// For each pair (x[2i], x[2i+1]) at position `pos`:
//   theta = pos * inv_freq(i)
//   out[2i]   = m * (x[2i] * cos(theta) - x[2i+1] * sin(theta))
//   out[2i+1] = m * (x[2i] * sin(theta) + x[2i+1] * cos(theta))
//
// Layout:
//   input:  [N, num_heads, head_dim] f32
//   output: [N, num_heads, head_dim] f32
//   Dispatched per (token, head, pair)
//
// ## One kernel, no variants, no mode switch
//
// NTK and YaRN differ from plain RoPE, and from each other, only in how
// inv_freq(i) is built — and every ingredient of that beyond `i` itself is
// constant across the whole tensor. `ops/rope/reference.ts` reduces all three
// schemes to the five scalars in `Params` below, on the host, once per model.
//
// What is left here is one expression that is *every* scheme:
//
//   inv_freq = extrapolation + (interpolation - extrapolation) * ramp
//
//   - Plain RoPE and NTK arrive with interpolation_factor = 1, which makes
//     `interpolation - extrapolation` an exact IEEE zero. The ramp then
//     multiplies zero and inv_freq is the unscaled frequency **bit for bit** —
//     not merely to within a tolerance. attention_factor = 1 is likewise an
//     exact multiply. Asking for no scaling really does change nothing.
//   - NTK is then only a larger `effective_base`. It needed no shader change
//     at all: the pre-scaling kernel computes NTK correctly if handed the
//     adjusted base, which is the strongest evidence available that a separate
//     NTK variant would have been duplication.
//   - YaRN arrives with interpolation_factor = s and a live ramp.
//
// The alternatives cost more and buy nothing. A `mode` uniform with a branch
// per scheme would move a log, a floor, a ceiling, two clamps and a
// divide-by-zero guard out of host code that runs once per model and into
// shader code that runs per element, and each arm would be reachable by
// exactly one test. Three .wgsl files would keep three copies of the rotation
// in step for one line of difference each.
//
// ## The rotary cache, and what it does when it runs out
//
// The angles depend on position and pair alone, so decoding recomputes the same
// `pow`, `sin` and `cos` for every head, every step. `ropeCache` in the
// reference precomputes them into `cache` and this kernel reads them instead.
//
// A table has a length and decoding runs past it. Of the three things that
// could happen there, two are unavailable and one is right:
//
//   - growing the table needs a host, and there is no host inside a dispatch
//   - wrapping — `pos % cache_positions` — is a real angle for the wrong
//     position, so it returns a plausible tensor instead of an error, which is
//     worse than returning nothing
//   - falling back to computing the angle here is correct, and costs only the
//     saving
//
// So: `pos < cache_positions` reads the table, and everything else computes.
// The fallback is not a special case — it is the uncached kernel, unchanged,
// which is why `cache_positions = 0` is exactly the op this file was before the
// cache existed. Same degenerate-case argument as the scaling above.
//
// `cache` is bound either way, since a binding a shader ignores is dropped by
// `layout: "auto"`. Callers who want no cache bind two floats and pass
// `cache_positions = 0`; nothing reads them.

struct Params {
  N: u32,          // sequence length
  num_heads: u32,
  head_dim: u32,
  pos_offset: u32, // starting position (for KV-cache continuation)
  // Positions [0, cache_positions) are in `cache`. 0 means no cache at all.
  cache_positions: u32,

  // From `ropeFrequencyParams` in the reference, which documents these against
  // the papers' notation. The short version:
  effective_base: f32,       // the base actually raised to -2i/D; NTK scales it
  interpolation_factor: f32, // YaRN's s, dividing the interpolated branch; 1 otherwise
  ramp_low: f32,             // pair index where YaRN stops extrapolating
  ramp_high: f32,            // pair index where YaRN is fully interpolated
  attention_factor: f32,     // YaRN's sqrt(1/t), a gain on cos and sin; 1 otherwise
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
// [cache_positions, head_dim/2, 2] — cos then sin, with attention_factor
// already folded in, from `ropeCache`. Built from the same five scalars below,
// so the table and the fallback are the same rotation by construction.
@group(0) @binding(1) var<storage, read> cache: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
) {
  let half_dim = params.head_dim / 2u;
  let total_pairs = params.N * params.num_heads * half_dim;

  let pair_idx = gid.x;
  if (pair_idx >= total_pairs) {
    return;
  }

  // Decompose linear index into (token, head, dim_pair)
  let dim_pair = pair_idx % half_dim;
  let remainder = pair_idx / half_dim;
  let head = remainder % params.num_heads;
  let token = remainder / params.num_heads;

  let pos = token + params.pos_offset;

  var cos_theta: f32;
  var sin_theta: f32;
  if (pos < params.cache_positions) {
    let at = (pos * half_dim + dim_pair) * 2u;
    cos_theta = cache[at];
    sin_theta = cache[at + 1u];
  } else {
    let freq_exp = -2.0 * f32(dim_pair) / f32(params.head_dim);

    // The reference implementations write these as 1/pos_freqs and
    // 1/(s*pos_freqs) with pos_freqs = b^(2i/D). Written as a negative exponent
    // instead, so that the unscaled path is the expression this kernel has
    // always evaluated.
    let extrapolation = pow(params.effective_base, freq_exp);
    let interpolation = extrapolation / params.interpolation_factor;

    // YaRN's gamma(i): 0 below ramp_low — this pair turns fast enough to have
    // been seen at every phase during training, so extrapolate — and 1 above
    // ramp_high, where the pair has not completed a single rotation and must be
    // interpolated. Linear between.
    let ramp = clamp((f32(dim_pair) - params.ramp_low) / (params.ramp_high - params.ramp_low), 0.0, 1.0);
    let inv_freq = extrapolation + (interpolation - extrapolation) * ramp;

    let theta = f32(pos) * inv_freq;

    // The attention temperature folds into cos/sin, as it does in both
    // reference implementations. Scaling q and k before the dot product is the
    // same thing one step later. The table above has it folded in already, for
    // the same reason and at the same place.
    cos_theta = cos(theta) * params.attention_factor;
    sin_theta = sin(theta) * params.attention_factor;
  }

  let base_idx = (token * params.num_heads + head) * params.head_dim + dim_pair * 2u;
  let x0 = input[base_idx];
  let x1 = input[base_idx + 1u];

  output[base_idx]      = x0 * cos_theta - x1 * sin_theta;
  output[base_idx + 1u] = x0 * sin_theta + x1 * cos_theta;
}
