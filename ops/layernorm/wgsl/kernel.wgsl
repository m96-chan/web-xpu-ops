// LayerNorm: (x_i - mean) * w_i / sqrt(var + eps) + b_i
//
// Two reductions in one dispatch, both the tree shape rmsnorm uses:
//   1. Sum the row            -> mean
//   2. Sum the squared deviations from that mean -> variance
//   3. Normalize, scale, shift
//
// The second reduction is what separates this from rmsnorm, and it is done
// against the mean rather than by accumulating x² alongside x. The one-pass
// identity var = E[x²] - E[x]² would fold both reductions into the first, but
// it subtracts two large near-equal numbers: at x ≈ 8192 the squares land past
// 2^26, where f32 steps by 8, and a true variance of 6.678 comes back as 0.
// One extra pass over a row already in cache is the cheaper side of that trade.
//
// Layout:
//   input:  [N, D] f32
//   weight: [D]    f32 (learnable scale)
//   bias:   [D]    f32 (learnable shift)
//   output: [N, D] f32

// A dispatch wider than one row of workgroups is folded back into one index
// here rather than by a uniform: `num_workgroups.x` is what the host asked for,
// so **every existing one-dimensional caller keeps working unchanged** — at
// `[n]` the y extent is 1 and the second term is 0. The ceiling is 65,535 on
// every backend measured (#211), and `examples/h3-video`'s decoder passes it at
// 42 latent frames — a seven-second clip.

struct Params {
  N: u32,
  D: u32,
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
  @builtin(num_workgroups) workgroups: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>,
) {
  let row = wg_id.x + wg_id.y * workgroups.x;
  // Uniform across the workgroup, so the barriers below stay in uniform control
  // flow. Callers may dispatch more workgroups than there are rows.
  if (row >= params.N) {
    return;
  }

  let tid = local_id.x;
  let row_offset = row * params.D;

  // Pass 1: mean
  var local_sum: f32 = 0.0;
  for (var col = tid; col < params.D; col += WORKGROUP_SIZE) {
    local_sum += input[row_offset + col];
  }

  shared_sum[tid] = local_sum;
  workgroupBarrier();

  for (var stride = WORKGROUP_SIZE / 2u; stride > 0u; stride >>= 1u) {
    if (tid < stride) {
      shared_sum[tid] += shared_sum[tid + stride];
    }
    workgroupBarrier();
  }

  let mean = shared_sum[0] / f32(params.D);
  // Everyone has read slot 0 before pass 2 overwrites the same scratch.
  workgroupBarrier();

  // Pass 2: variance, as the mean of the squared deviations. Biased (1/D),
  // matching torch.nn.functional.layer_norm.
  var local_sq: f32 = 0.0;
  for (var col = tid; col < params.D; col += WORKGROUP_SIZE) {
    let deviation = input[row_offset + col] - mean;
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

  let variance = shared_sum[0] / f32(params.D);
  let inv_std = inverseSqrt(variance + params.eps);

  // Pass 3: normalize, scale, shift
  for (var col = tid; col < params.D; col += WORKGROUP_SIZE) {
    output[row_offset + col] = (input[row_offset + col] - mean) * inv_std * weight[col] + bias[col];
  }
}
