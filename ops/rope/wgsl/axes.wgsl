// Multi-axis RoPE: the head split into blocks of channels, each turned by its
// own position. Z-Image's `[32, 48, 48]` at theta 256 is the case it exists for.
//
// A second entry point beside `kernel.wgsl` rather than a mode on it. The two
// share the rotation and nothing else: this one has no cache, no scaling and no
// head range, and `kernel.wgsl` has no axis table and one position per token,
// so a merged kernel would carry both sets of uniforms and branch per element
// on which half is live. `ops/rope/reference.ts` says why none of the three are
// here yet, and `harness/resolve.ts` says why a second entry point is a first
// class thing rather than a variant.
//
// For each pair (x[2i], x[2i+1]) of the block belonging to axis `a`, at that
// token's position `p_a` on that axis:
//   theta     = p_a * theta_base^(-2 * i_a / d_a)     i_a counted within the axis
//   out[2i]   = x[2i] * cos(theta) - x[2i+1] * sin(theta)
//   out[2i+1] = x[2i] * sin(theta) + x[2i+1] * cos(theta)
//
// Layout:
//   input:     [N, num_heads, head_dim] f32, head_dim = sum(axis_dims)
//   axis_dims: [num_axes] u32 — channels per axis, in order
//   positions: [N, num_axes] i32 — token-major, upstream's `ids` flattened
//   output:    [N, num_heads, head_dim] f32
//   Dispatched per (token, head, pair), exactly as `kernel.wgsl` is.
//
// ## The denominator is the axis's own channel count
//
// `d_a`, not `head_dim`. Z-Image's `RopeEmbedder.precompute_freqs_cis` builds
// each axis as `1 / theta ** (arange(0, d, 2) / d)` with `d` that axis's dim, so
// every axis sweeps the same frequency range however many channels it holds.
// Dividing by `head_dim` here is the plausible wrong answer; it moves Z-Image's
// own geometry by 2.6 in absolute terms, measured by mutating the reference.
//
// ## Finding the axis, and why it is a loop rather than a table
//
// A thread knows its pair index within the head and has to find which axis owns
// it. That could arrive precomputed — one entry per pair, built on the host —
// but `num_axes` is 3 and the loop runs at most that many times against a
// buffer that would otherwise cost `head_dim/2` reads to build and bind. The
// loop also keeps the kernel's inputs the same shape the reference takes, so
// there is one description of the split rather than two that have to agree.
//
// ## Pairing
//
// Adjacent channels, `2i`/`2i+1`, which is `torch.view_as_complex` upstream and
// the same convention `kernel.wgsl` uses — not HF Llama's `rotate_half`. A
// checkpoint written for `rotate_half` needs `llm/weights.ts#permuteRopeChannels`
// first, exactly as it does for 1-D rope.

struct Params {
  N: u32,          // tokens
  num_heads: u32,
  head_dim: u32,   // must equal the sum of axis_dims; the host checks it
  num_axes: u32,
  theta_base: f32, // shared by every axis, as upstream's single `theta` is
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
// Channels per axis, in order. Axis `a`'s block starts after every earlier
// axis's, which is what upstream's `torch.cat(result, dim=-1)` builds.
@group(0) @binding(1) var<storage, read> axis_dims: array<u32>;
// [N, num_axes], token-major. i32 rather than u32 for the reason `gather`'s
// indices are: a negative position stays negative — it turns the rotation the
// other way — instead of becoming a huge positive one nothing can tell from a
// real position.
@group(0) @binding(2) var<storage, read> positions: array<i32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

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

  let base_idx = (token * params.num_heads + head) * params.head_dim + dim_pair * 2u;
  let x0 = input[base_idx];
  let x1 = input[base_idx + 1u];

  // Which axis owns this pair, and which pair of that axis it is.
  var axis = params.num_axes;
  var first_pair = 0u;
  for (var a = 0u; a < params.num_axes; a = a + 1u) {
    let pairs = axis_dims[a] / 2u;
    if (dim_pair < first_pair + pairs) {
      axis = a;
      break;
    }
    first_pair = first_pair + pairs;
  }

  // Unreachable when the axis dims sum to head_dim, which the host refuses to
  // dispatch without (`ropeAxes` throws). A shader cannot raise, so the pair is
  // copied rather than left alone: `output` arrives zeroed, and a channel
  // nobody wrote reads back as a plausible zero that goes on into attention.
  if (axis == params.num_axes) {
    output[base_idx]      = x0;
    output[base_idx + 1u] = x1;
    return;
  }

  let pair_in_axis = dim_pair - first_pair;
  let pos = positions[token * params.num_axes + axis];

  // f32(axis_dims[axis]), not f32(params.head_dim) — see the note above.
  let inv_freq = pow(params.theta_base, -2.0 * f32(pair_in_axis) / f32(axis_dims[axis]));
  let theta = f32(pos) * inv_freq;
  let cos_theta = cos(theta);
  let sin_theta = sin(theta);

  output[base_idx]      = x0 * cos_theta - x1 * sin_theta;
  output[base_idx + 1u] = x0 * sin_theta + x1 * cos_theta;
}
