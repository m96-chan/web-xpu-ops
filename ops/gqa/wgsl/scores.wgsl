// GQA / MQA, dispatch 1 of 2: probs = softmax(mask(scale * Q @ K_g^T)).
//
// Layout:
//   q:      [B, H,       L, D]  f32, row-major
//   k:      [B, kv_heads, S, D] f32, row-major   <- fewer heads than q
//   probs:  [B, H,       L, S]  f32, row-major
//
// Structurally this is ops/attention's scores kernel. The only difference is
// which head of K a query head reads, and that difference is two lines, which is
// the point: sharing KV heads is an addressing change, not an arithmetic one.
// Everything about masking, the scale and the softmax is unchanged, and is
// documented in ops/attention rather than restated here.
//
// The dispatch is exactly [L, H, B] — over the *query* heads, since every query
// head still gets its own probability row. As in attention, every invocation has
// real work and no bounds guard is written, because a guard nothing can trip is
// a guard no test can keep honest (rule 1). Contract: H % kv_heads == 0, checked
// by the caller (reference.ts throws), so the integer division below is exact.

struct Params {
  H: u32,
  // Key / value heads. H is plain MHA, 1 is MQA, a divisor of H is GQA.
  kv_heads: u32,
  L: u32,
  S: u32,
  D: u32,
  scale: f32,
  causal: u32,
  query_offset: i32,
}

@group(0) @binding(0) var<storage, read> q: array<f32>;
@group(0) @binding(1) var<storage, read> k: array<f32>;
@group(0) @binding(2) var<storage, read_write> probs: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

const WORKGROUP_SIZE: u32 = 256u;

// Stands in for the -infinity PyTorch adds as `attn_bias`; WGSL cannot write an
// infinity literal. See ops/attention/wgsl/scores.wgsl for the measurements that
// show -FLT_MAX behaves identically for every input this op accepts.
const MASKED: f32 = -3.402823e+38;

var<workgroup> shared_val: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
  @builtin(workgroup_id) wg: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>,
) {
  let i = wg.x;                                  // query position
  let batch = wg.z;
  let head = wg.y;                               // query head
  let tid = local_id.x;

  let q_head = batch * params.H + head;
  // Contiguous groups: query heads 0..g-1 share KV head 0, g..2g-1 share KV head
  // 1, and so on. Strided groups (`head % params.kv_heads`) are the other
  // reading and are wrong; both agree at kv_heads == H and at kv_heads == 1, so
  // only the middle cases can tell them apart. The batch stride is kv_heads, not
  // H — K has fewer heads per batch, and reusing H here walks past its end.
  let kv_head = batch * params.kv_heads + head / (params.H / params.kv_heads);

  let q_row = (q_head * params.L + i) * params.D;
  let k_head = kv_head * params.S * params.D;
  let p_row = (q_head * params.L + i) * params.S;

  // Pass 1: scaled dot products, straight into `probs`.
  var local_max: f32 = MASKED;
  for (var j = tid; j < params.S; j += WORKGROUP_SIZE) {
    var value: f32 = MASKED;
    if (params.causal == 0u || i32(j) <= i32(i) + params.query_offset) {
      var dot: f32 = 0.0;
      for (var d: u32 = 0u; d < params.D; d += 1u) {
        dot += q[q_row + d] * k[k_head + j * params.D + d];
      }
      value = dot * params.scale;
    }
    probs[p_row + j] = value;
    local_max = max(local_max, value);
  }

  shared_val[tid] = local_max;
  workgroupBarrier();
  for (var stride = WORKGROUP_SIZE / 2u; stride > 0u; stride >>= 1u) {
    if (tid < stride) {
      shared_val[tid] = max(shared_val[tid], shared_val[tid + stride]);
    }
    workgroupBarrier();
  }
  let row_max = shared_val[0];
  workgroupBarrier();

  // Pass 2: exponentiate in place, and sum. The row max comes off first because
  // exp() overflows f32 at 89 and attention logits reach the hundreds.
  var local_sum: f32 = 0.0;
  for (var j = tid; j < params.S; j += WORKGROUP_SIZE) {
    let e = exp(probs[p_row + j] - row_max);
    probs[p_row + j] = e;
    local_sum += e;
  }

  shared_val[tid] = local_sum;
  workgroupBarrier();
  for (var stride = WORKGROUP_SIZE / 2u; stride > 0u; stride >>= 1u) {
    if (tid < stride) {
      shared_val[tid] += shared_val[tid + stride];
    }
    workgroupBarrier();
  }
  let inv_sum = 1.0 / shared_val[0];
  workgroupBarrier();

  // Pass 3: normalise. Masked columns are already 0 and stay 0.
  for (var j = tid; j < params.S; j += WORKGROUP_SIZE) {
    probs[p_row + j] = probs[p_row + j] * inv_sum;
  }
}
