// Attention, dispatch 1 of 2: probs = softmax(mask(scale * Q @ K^T) + bias).
//
// Layout:
//   q:      [B, H, L, D]  f32, row-major
//   k:      [B, H, S, D]  f32, row-major
//   mask:   [MB, MH, MR, S] f32, row-major — the additive attention bias
//   probs:  [B, H, L, S]  f32, row-major — the materialised attention matrix
//
// One workgroup owns one query row: all of `probs[b][h][i][*]`. That is the
// unit the softmax needs anyway, so the row max and the row sum are a workgroup
// reduction and never leave the group. Nothing crosses workgroups, so the two
// halves of attention split cleanly into two dispatches with a buffer between
// them — which is the whole point of this op existing (#18). `flash_attention`
// (#19) is what avoids writing that buffer; this one is what proves it right.
//
// The dispatch is exactly [L, H, B], so every invocation has real work and
// there is no "am I off the end" guard anywhere below. That is a contract with
// the caller, not an accident: a guard nothing can trip is a guard no test can
// keep honest (rule 1), so it is stated here instead of written as dead code.
//
// Not tiled, not fused, no shared staging of K. Rule 8: this is the reference
// implementation the fast one gets measured against, so it is written to be
// read. #3/#4 add the roofline harness; tuning comes after there is a number.

struct Params {
  H: u32,
  L: u32,
  S: u32,
  D: u32,
  // PyTorch's `scale`, resolved by the caller. Defaults to 1/sqrt(D) — from the
  // *query's* head dim, not V's. Passed in rather than computed here so there
  // is one definition of it (`defaultScale` in reference.ts) instead of two.
  scale: f32,
  causal: u32,
  // Absolute position of query row 0 in the key sequence, so row i sits at
  // i + query_offset. 0 reproduces torch's `is_causal=True` (upper-left
  // aligned); S - L reproduces `causal_lower_right`, which is what a KV cache
  // decode step wants. See reference.ts for why this is one signed int and not
  // a second flag.
  query_offset: i32,
  // The broadcast shape of `mask`, each either 1 or the full dimension. The
  // last axis is always S. [B, 1, 1] is a key-padding mask, [B, H, L] a full
  // per-element bias, [1, H, L] the shape ops/alibi hands back. See #77.
  mask_batch: u32,
  mask_heads: u32,
  mask_rows: u32,
}

@group(0) @binding(0) var<storage, read> q: array<f32>;
@group(0) @binding(1) var<storage, read> k: array<f32>;
@group(0) @binding(2) var<storage, read> mask: array<f32>;
@group(0) @binding(3) var<storage, read_write> probs: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

const WORKGROUP_SIZE: u32 = 256u;

// Stands in for the -infinity PyTorch adds as `attn_bias`. WGSL has no way to
// write an infinity: `bitcast<f32>(0xff800000u)` is rejected at compile time
// with "value -inf cannot be represented as 'f32'" (measured, not assumed).
//
// -FLT_MAX behaves identically for every input this op accepts. Measured on
// this device: `exp(MASKED - x)` is exactly 0 for finite x, `MASKED - 5.0` does
// not overflow, and `max(MASKED, -2.0)` is -2.0, so a masked column loses the
// row max and contributes nothing to the row sum.
//
// It is deliberately *not* the fully-masked sentinel. A caller's mask may hold
// a real -inf and torch tells the two apart: measured on 2.10, a mask row of
// -FLT_MAX returns the mean of V while a row of -inf returns zeros. See the
// softmax denominator below for how this kernel separates them.
const MASKED: f32 = -3.402823e+38;

var<workgroup> shared_val: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
  @builtin(workgroup_id) wg: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>,
) {
  let i = wg.x;                                  // query position
  let batch = wg.z;
  let h = wg.y;                                  // attention head
  let head = batch * params.H + h;               // flattened (batch, head)
  let tid = local_id.x;

  let q_row = (head * params.L + i) * params.D;
  let k_head = head * params.S * params.D;
  let p_row = (head * params.L + i) * params.S;

  // Where this query row's slice of the bias starts. Each axis is collapsed to
  // 0 when the caller broadcast it, which is why the same buffer serves a
  // key-padding mask and a full [B, H, L, S] bias with no branch per element.
  let mb = select(0u, batch, params.mask_batch > 1u);
  let mh = select(0u, h, params.mask_heads > 1u);
  let mr = select(0u, i, params.mask_rows > 1u);
  let m_row = ((mb * params.mask_heads + mh) * params.mask_rows + mr) * params.S;

  // Pass 1: the scaled dot products, straight into `probs`. Keeping them there
  // rather than in workgroup memory is what lets S be arbitrary — a workgroup
  // array would have to be sized at compile time, and S is a KV cache length.
  var local_max: f32 = MASKED;
  for (var j = tid; j < params.S; j += WORKGROUP_SIZE) {
    var value: f32 = MASKED;
    // The mask, evaluated exactly once per element and nowhere else. `<=`, and
    // `j` against `i`, is torch's `tril` taken from the top-left corner.
    if (params.causal == 0u || i32(j) <= i32(i) + params.query_offset) {
      var dot: f32 = 0.0;
      for (var d: u32 = 0u; d < params.D; d += 1u) {
        dot += q[q_row + d] * k[k_head + j * params.D + d];
      }
      // The additive bias, PyTorch's float `attn_mask`: added to the score
      // rather than selecting on it, so -inf masks and 0 does not, and an ALiBi
      // bias composes with a key mask by ordinary addition.
      value = dot * params.scale + mask[m_row + j];
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

  // Pass 2: exponentiate in place, and sum. Subtracting the row max is not an
  // optimisation — attention logits reach the hundreds and exp() overflows f32
  // at 89, taking the row to Infinity/Infinity = NaN.
  //
  // `row_max` is at least MASKED whatever the mask holds, because that is where
  // the reduction starts, so `probs - row_max` is never `-inf - -inf` and this
  // pass needs no guard of its own. A row of caller-supplied -inf comes through
  // here as exp(-inf - -FLT_MAX) = 0, exactly (measured on this device).
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
  // Every key masked. Unreachable before #77 — `query_offset >= 0` meant column
  // 0 was always visible — and reachable now, by accident: an empty sequence in
  // a padded batch is one. Guarded rather than declared out of contract,
  // because 1/0 here is +inf and 0 * inf is NaN, and a NaN row is a wrong
  // answer that looks like a result. Zeros instead, which is what SDPA gives:
  // it routes through `aten::_safe_softmax`, not `torch.softmax`. Measured on
  // torch 2.10 — on the same all-(-inf) row the two give [0,0,0,0] and
  // [nan,nan,nan,nan].
  //
  // The condition is the *sum* and not the maximum, which is not interchangeable
  // here: `local_max` starts at MASKED, so `row_max` never reports -inf even
  // when every score is one (measured — `max(MASKED, -inf)` is MASKED). The sum
  // is exact: whichever key holds the maximum contributes exp(0) = 1, so the sum
  // is 0 if and only if no score was finite.
  //
  // That also keeps -FLT_MAX out of it, which matters because the causal branch
  // above writes exactly -FLT_MAX and a caller's mask may too. A row of -FLT_MAX
  // sums to S and comes out uniform, which is what torch does with it (measured:
  // the mean of V, not zeros).
  let inv_sum = select(1.0 / shared_val[0], 0.0, shared_val[0] == 0.0);
  workgroupBarrier();

  // Pass 3: normalise. Masked columns are already 0 and stay 0, so the second
  // dispatch never has to know the mask exists — including a fully masked row,
  // which arrives at the second dispatch as zeros and produces a zero output.
  for (var j = tid; j < params.S; j += WORKGROUP_SIZE) {
    probs[p_row + j] = probs[p_row + j] * inv_sum;
  }
}
