// RMSNorm: x_i * w_i / sqrt(mean(x²) + eps)
//
// Two-pass within one dispatch:
//   1. Compute sum of squares (workgroup reduction)
//   2. Normalize: x_i * w_i * rsqrt(mean_sq + eps)
//
// Layout:
//   input:  [N, D] f32
//   weight: [G, D] f32 (learnable scale; row n uses group n % G)
//   output: [N, D] f32
//
// G is the extent of the axis immediately left of D — the fastest varying of
// the axes flattened into N. For QK-norm's [B, S, H, Dh] that is H, which is
// why `row % G` is the head index. reference.ts states the layout and why the
// [B, H, S, Dh] one is refused rather than served wrongly.

struct Params {
  N: u32,
  D: u32,
  eps: f32,
  // Group count. Zero means one group: a caller packing the three-word params
  // this op took before #78 leaves this word zero, and must keep its old
  // behaviour rather than take an indeterminate `% 0`.
  //
  // The `max(params.G, 1u)` below is stated as a contract because **no test can
  // kill it here**, and that is worth writing down rather than discovering. Rule
  // 1 asks for a mutation that goes red; this one does not. Measured: dropping
  // the `max` left every pre-#78 case green, so Tint lowers `x % 0` to `0` on
  // this device — the same answer the guard gives. WGSL only promises an
  // *indeterminate* value, so the guard is what makes the compatibility path a
  // property of this op rather than of one driver. Deleting it would be green
  // here and wrong somewhere else.
  G: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

const WORKGROUP_SIZE: u32 = 256u;

var<workgroup> shared_sum: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
  @builtin(workgroup_id) wg_id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>,
) {
  let row = wg_id.x;
  if (row >= params.N) {
    return;
  }

  let tid = local_id.x;
  let row_offset = row * params.D;

  // Pass 1: Sum of squares
  var local_sum: f32 = 0.0;
  for (var col = tid; col < params.D; col += WORKGROUP_SIZE) {
    let val = input[row_offset + col];
    local_sum += val * val;
  }

  shared_sum[tid] = local_sum;
  workgroupBarrier();

  for (var stride = WORKGROUP_SIZE / 2u; stride > 0u; stride >>= 1u) {
    if (tid < stride) {
      shared_sum[tid] += shared_sum[tid + stride];
    }
    workgroupBarrier();
  }

  let rms = inverseSqrt(shared_sum[0] / f32(params.D) + params.eps);

  // Pass 2: Normalize
  let weight_offset = (row % max(params.G, 1u)) * params.D;
  for (var col = tid; col < params.D; col += WORKGROUP_SIZE) {
    output[row_offset + col] = input[row_offset + col] * rms * weight[weight_offset + col];
  }
}
