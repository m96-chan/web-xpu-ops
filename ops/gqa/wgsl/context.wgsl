// GQA / MQA, dispatch 2 of 2: output = probs @ V_g.
//
// Layout:
//   probs:  [B, H,        L, S]  f32, row-major — what scores.wgsl produced
//   v:      [B, kv_heads, S, Dv] f32, row-major   <- fewer heads than probs
//   output: [B, H,        L, Dv] f32, row-major
//
// ops/attention's context kernel with the same one change scores.wgsl makes:
// which head of V a query head reads. Nothing here knows the mask exists —
// masked columns arrive as exactly 0 from the softmax — which is why sharing KV
// heads and choosing a causal convention stay independent of each other.
//
// Dispatch is exactly [L, H, B], over the query heads, so no bounds guard is
// reachable and none is written (rule 1).

struct Params {
  H: u32,
  kv_heads: u32,
  L: u32,
  S: u32,
  Dv: u32,
  // Issue #117, same field and same contract as scores.wgsl's: bounds the
  // `probs @ V` sum, `S` stays `v`'s per-head stride. Must be given the same
  // value `scores.wgsl` was — `probs` columns past it were never written by
  // that dispatch (see scores.wgsl's doc), so reading further here would read
  // whatever this buffer held before. Default `S` (appended, not inserted —
  // no existing field's offset moves).
  s_eff: u32,
}

@group(0) @binding(0) var<storage, read> probs: array<f32>;
@group(0) @binding(1) var<storage, read> v: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

const WORKGROUP_SIZE: u32 = 256u;

@compute @workgroup_size(256)
fn main(
  @builtin(workgroup_id) wg: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>,
) {
  let i = wg.x;                                  // query position
  let batch = wg.z;
  let head = wg.y;                               // query head

  let q_head = batch * params.H + head;
  // Same mapping as scores.wgsl, and it has to stay the same: K and V are
  // indexed by the same KV head, so a disagreement between the two dispatches
  // would attend to one head's keys and read another head's values.
  let kv_head = batch * params.kv_heads + head / (params.H / params.kv_heads);

  let p_row = (q_head * params.L + i) * params.S;
  let v_head = kv_head * params.S * params.Dv;
  let o_row = (q_head * params.L + i) * params.Dv;

  // One output channel per invocation, striding when Dv exceeds the workgroup.
  for (var c = local_id.x; c < params.Dv; c += WORKGROUP_SIZE) {
    var acc: f32 = 0.0;
    for (var j: u32 = 0u; j < params.s_eff; j += 1u) {
      acc += probs[p_row + j] * v[v_head + j * params.Dv + c];
    }
    output[o_row + c] = acc;
  }
}
