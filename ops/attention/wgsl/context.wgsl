// Attention, dispatch 2 of 2: output = probs @ V.
//
// Layout:
//   probs:  [B, H, L, S]  f32, row-major — what scores.wgsl produced
//   v:      [B, H, S, Dv] f32, row-major
//   output: [B, H, L, Dv] f32, row-major
//
// A batched matmul, and deliberately not the tiled one in ops/matmul: this is
// the readable half of an op whose job is to be obviously correct. It shares
// scores.wgsl's decomposition — one workgroup per query row, dispatch exactly
// [L, H, B], so no bounds guard is reachable and none is written (rule 1).
//
// Nothing here knows the mask exists. Masked columns arrive as exactly 0 from
// the softmax, so they contribute nothing to the sum on their own. Keeping the
// mask in one dispatch is what makes the causal convention a single place to
// look, which matters because #24 (GQA/MQA) and #26 (paged KV cache) both
// change how K and V are addressed without changing what is masked.
//
// `Dv` is its own parameter and not reused from `D`. PyTorch allows V to have a
// different head dim from Q and K, and this dispatch never sees `D` at all —
// which is the structural reason the two are impossible to confuse here.

struct Params {
  H: u32,
  L: u32,
  S: u32,
  Dv: u32,
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
  let head = wg.z * params.H + wg.y;             // flattened (batch, head)

  let p_row = (head * params.L + i) * params.S;
  let v_head = head * params.S * params.Dv;
  let o_row = (head * params.L + i) * params.Dv;

  // One output channel per invocation, striding when Dv exceeds the workgroup.
  // Each invocation re-reads the whole probability row; staging that row in
  // workgroup memory is the obvious first optimisation and is deliberately not
  // done here, because S is a runtime KV cache length and a workgroup array
  // would have to be sized at compile time. That trade is #19's to make.
  for (var c = local_id.x; c < params.Dv; c += WORKGROUP_SIZE) {
    var acc: f32 = 0.0;
    for (var j: u32 = 0u; j < params.S; j += 1u) {
      acc += probs[p_row + j] * v[v_head + j * params.Dv + c];
    }
    output[o_row + c] = acc;
  }
}
