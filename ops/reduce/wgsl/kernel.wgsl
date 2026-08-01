// Axis reduction: sum / max / min / mean over one axis of a tensor.
//
// The tensor is viewed as [outer, axis, inner], so any axis of any rank fits:
// `outer` is the product of the dimensions before it, `inner` the product of
// those after. Output is [outer, inner].
//
// One workgroup per output element. It walks its slice of the axis in
// 256-wide strides, then collapses the 256 partials with a tree reduction —
// the same shape rmsnorm and softmax already use for their row statistics,
// with the combiner and the identity lifted out.
//
// Layout:
//   input:  [outer, axis, inner] f32
//   output: [outer, inner]       f32

struct Params {
  outer: u32,
  axis: u32,
  inner: u32,
  kind: u32,  // 0 = sum, 1 = max, 2 = min, 3 = mean
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

const WORKGROUP_SIZE: u32 = 256u;

const KIND_SUM: u32 = 0u;
const KIND_MAX: u32 = 1u;
const KIND_MIN: u32 = 2u;
const KIND_MEAN: u32 = 3u;

// The extreme finite f32, not the infinities. WGSL has no infinity literal and
// the spec lets an implementation assume none occur, so ±FLT_MAX is what can
// actually be relied on. It is enough: no finite f32 lies outside it, so
// combining it with any real element yields that element, which is the whole
// job of an identity. It matters for a short axis, where most of the 256 lanes
// load nothing and carry the identity into the tree.
const NEG_FLT_MAX: f32 = -3.402823e+38;
const POS_FLT_MAX: f32 = 3.402823e+38;

var<workgroup> shared_val: array<f32, 256>;

fn identity(kind: u32) -> f32 {
  if (kind == KIND_MAX) {
    return NEG_FLT_MAX;
  }
  if (kind == KIND_MIN) {
    return POS_FLT_MAX;
  }
  return 0.0;
}

fn combine(kind: u32, a: f32, b: f32) -> f32 {
  if (kind == KIND_MAX) {
    return max(a, b);
  }
  if (kind == KIND_MIN) {
    return min(a, b);
  }
  return a + b;
}

// NaN, measured on this adapter rather than read off the spec: `max(nan, 1.0)`
// and `max(1.0, nan)` both return 1.0, and `min` likewise, so NaN is swallowed
// by max and min here. The reference propagates it, as PyTorch does. The gap is
// left rather than patched: WGSL permits an implementation to assume operands
// are NaN-free, so a hand-rolled `select(a, b, a != a)` guard would be paying
// for a promise the language does not make. `sum` and `mean` do propagate —
// `0.0 + nan` is nan and `0.0 / 0.0` is nan on this adapter.

@compute @workgroup_size(256)
fn main(
  @builtin(workgroup_id) wg_id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>,
) {
  let out_index = wg_id.x;
  if (out_index >= params.outer * params.inner) {
    return;
  }

  let tid = local_id.x;
  // Consecutive elements along the axis sit `inner` apart. The axis is
  // contiguous only when nothing follows it.
  let base = (out_index / params.inner) * params.axis * params.inner + (out_index % params.inner);

  var acc = identity(params.kind);
  for (var a = tid; a < params.axis; a += WORKGROUP_SIZE) {
    acc = combine(params.kind, acc, input[base + a * params.inner]);
  }

  shared_val[tid] = acc;
  workgroupBarrier();

  for (var stride = WORKGROUP_SIZE / 2u; stride > 0u; stride >>= 1u) {
    if (tid < stride) {
      shared_val[tid] = combine(params.kind, shared_val[tid], shared_val[tid + stride]);
    }
    workgroupBarrier();
  }

  if (tid == 0u) {
    var result = shared_val[0];
    if (params.kind == KIND_MEAN) {
      // The axis length, matching PyTorch — not the number of lanes that
      // loaded something, and not the workgroup width. An empty axis is
      // therefore 0/0 = NaN, which is also what torch.mean gives.
      result = result / f32(params.axis);
    }
    output[out_index] = result;
  }
}
