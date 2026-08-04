// GroupNorm: (x - mean_g) * w_c / sqrt(var_g + eps) + b_c
//
// One workgroup per (batch, group). Two reductions in one dispatch, the same
// tree shape layernorm and rmsnorm use:
//   1. Sum the group            -> mean
//   2. Sum the squared deviations from that mean -> variance
//   3. Normalize, then scale and shift PER CHANNEL
//
// What makes this its own kernel rather than layernorm with a different length:
// the statistics are pooled over `C/G` channels while the affine transform is
// not. Step 3 has to recover the channel from the offset; steps 1 and 2 must
// not.
//
// The layout earns the simple loop. `[N, C, L]` contiguous with contiguous
// channel groups means a group's `(C/G) * L` elements are one unbroken run, so
// this reduces over a span rather than a stride.
//
// The second reduction is done against the mean rather than by accumulating x²
// alongside x. The one-pass identity var = E[x²] - E[x]² would fold both into
// the first and cancel away the spread whenever the mean is large: at x ≈ 8192
// the squares land past 2^26, where f32 steps by 8. This op reduces a longer
// run than layernorm does — `(C/G) * L` rather than `L` — so there is more of
// that to accumulate, not less.
//
// Layout:
//   input:  [N, C, L] f32
//   weight: [C]       f32 (learnable scale, per channel)
//   bias:   [C]       f32 (learnable shift, per channel)
//   output: [N, C, L] f32

struct Params {
  N: u32,
  C: u32,
  L: u32,
  G: u32,
  eps: f32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

const WORKGROUP_SIZE: u32 = 256u;

var<workgroup> shared_sum: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
  @builtin(workgroup_id) wg_id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>,
) {
  let slot = wg_id.x;
  // Uniform across the workgroup, so the barriers below stay in uniform control
  // flow. Callers may dispatch more workgroups than there are groups.
  //
  // Stated as a contract, not as something the tests check: removing this guard
  // leaves the suite green (measured). It cannot be caught here, because a
  // surplus workgroup computes a `base` past the end of the buffer and WebGPU
  // discards the out-of-range writes — the output is identical either way. Kept
  // because it is right and free, not because a red test demanded it. What the
  // over-dispatch test does establish is that rounding the dispatch up is safe,
  // which is the property a caller actually relies on.
  if (slot >= params.N * params.G) {
    return;
  }

  let tid = local_id.x;
  let channels_per_group = params.C / params.G;
  let count = channels_per_group * params.L;

  // The batch item and the group within it. Any bijection from `slot` onto
  // (n, g) computes the same answer — swapping these two lines for
  // `slot % N` / `slot / N` is still a bijection and the suite stays green
  // (measured), so this is a choice about memory order rather than
  // correctness. `slot` walking groups fastest keeps consecutive workgroups on
  // consecutive spans of the input.
  let n = slot / params.G;
  let g = slot % params.G;
  let first_channel = g * channels_per_group;
  let base = (n * params.C + first_channel) * params.L;

  // Pass 1: mean
  var local_sum: f32 = 0.0;
  for (var i = tid; i < count; i += WORKGROUP_SIZE) {
    local_sum += input[base + i];
  }

  shared_sum[tid] = local_sum;
  workgroupBarrier();

  for (var stride = WORKGROUP_SIZE / 2u; stride > 0u; stride >>= 1u) {
    if (tid < stride) {
      shared_sum[tid] += shared_sum[tid + stride];
    }
    workgroupBarrier();
  }

  let mean = shared_sum[0] / f32(count);
  // Everyone has read slot 0 before pass 2 overwrites the same scratch.
  workgroupBarrier();

  // Pass 2: variance, as the mean of the squared deviations. Biased (1/count),
  // matching torch.nn.functional.group_norm — measured, not assumed: the
  // unbiased form is out by 1.6e-1 on the fixture in reference.test.ts.
  var local_sq: f32 = 0.0;
  for (var i = tid; i < count; i += WORKGROUP_SIZE) {
    let deviation = input[base + i] - mean;
    local_sq += deviation * deviation;
  }

  shared_sum[tid] = local_sq;
  workgroupBarrier();

  for (var stride = WORKGROUP_SIZE / 2u; stride > 0u; stride >>= 1u) {
    if (tid < stride) {
      shared_sum[tid] += shared_sum[tid + stride];
    }
    workgroupBarrier();
  }

  let variance = shared_sum[0] / f32(count);
  let inv_std = inverseSqrt(variance + params.eps);

  // Pass 3: normalize, then scale and shift by the element's OWN channel. The
  // group decided the statistics and does not decide these: `weight[g]` would
  // be well-formed, wrong, and invisible to any test whose weight is constant
  // inside a group.
  for (var i = tid; i < count; i += WORKGROUP_SIZE) {
    let channel = first_channel + i / params.L;
    output[base + i] = (input[base + i] - mean) * inv_std * weight[channel] + bias[channel];
  }
}
