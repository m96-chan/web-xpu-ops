// Flash attention: softmax(mask(scale * Q @ K^T)) @ V, in one dispatch, without
// the score matrix ever existing.
//
// Layout:
//   q:      [B, H, L, D]  f32, row-major
//   k:      [B, H, S, D]  f32, row-major
//   v:      [B, H, S, Dv] f32, row-major
//   mask:   [MB, MH, MR, S] f32, row-major — the additive attention bias
//   output: [B, H, L, Dv] f32, row-major
//
// The difference from ops/attention is not arithmetic — it computes the same
// function, checked against that op's reference — it is what gets allocated.
// The unfused version writes [B, H, L, S] between its two dispatches, and at any
// sequence length worth fusing for, that buffer is the largest thing in the
// computation. Here the only thing that ever holds scores is one tile of 64 in
// workgroup memory, which does not depend on S at all. That is the claim, and
// wgsl.test.ts counts the bytes rather than taking it on trust.
//
// One workgroup owns one query row, dispatch exactly [L, H, B], so every
// invocation has real work and no "am I off the end" guard appears below. That
// is a contract with the caller, not an oversight: a guard nothing can trip is a
// guard no test can keep honest (rule 1).
//
// Not tiled over Q, no shared staging of K, no subgroup reductions. Rule 8: get
// it agreeing with the reference first. #3 / #4 add the roofline harness, and
// tuning belongs after there is a number to move.

struct Params {
  H: u32,
  L: u32,
  S: u32,
  D: u32,
  Dv: u32,
  // PyTorch's `scale`, resolved by the caller. Defaults to 1/sqrt(D) — from the
  // *query's* head dim, not V's. Passed in rather than computed here so there is
  // one definition of it (`defaultScale` in ops/attention/reference.ts).
  scale: f32,
  causal: u32,
  // Absolute position of query row 0 in the key sequence, so row i sits at
  // i + query_offset. 0 reproduces torch's `is_causal=True` (upper-left
  // aligned); S - L reproduces `causal_lower_right`, which is what a KV cache
  // decode step wants. Inherited from ops/attention (#18) unchanged.
  query_offset: i32,
  // The broadcast shape of `mask`, each either 1 or the full dimension; the
  // last axis is always S. Inherited from ops/attention (#77) unchanged, which
  // is the point — one mask serves all three attention ops.
  mask_batch: u32,
  mask_heads: u32,
  mask_rows: u32,
}

@group(0) @binding(0) var<storage, read> q: array<f32>;
@group(0) @binding(1) var<storage, read> k: array<f32>;
@group(0) @binding(2) var<storage, read> v: array<f32>;
@group(0) @binding(3) var<storage, read> mask: array<f32>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;
@group(0) @binding(5) var<uniform> params: Params;

const WORKGROUP_SIZE: u32 = 256u;

// Keys folded in per step, and the entire memory footprint of the score matrix.
// 64 f32 is 256 bytes of workgroup memory against a 49152-byte budget, so the
// number is not chosen to fit — it is chosen small because the invocations that
// compute a tile are the ones that then idle while it is consumed, and 64 keeps
// that tail short. The answer does not depend on it: ops/flash_attention's
// reference takes the tile as a parameter and the tests sweep it from 1 to past
// the sequence length, which is what makes this a free constant rather than a
// load-bearing one.
const TILE: u32 = 64u;

// Stands in for the -infinity PyTorch adds as `attn_bias`. WGSL has no way to
// write an infinity: `bitcast<f32>(0xff800000u)` is rejected at compile time
// with "value -inf cannot be represented as 'f32'" (measured, not assumed).
//
// -FLT_MAX behaves identically for every input this op accepts: `exp(MASKED - x)`
// is exactly 0 for finite x, and `MASKED - x` does not overflow. The one input
// where it would differ is a row with every column masked, where
// `exp(MASKED - MASKED)` is 1 and this would return a uniform row. That row is
// unreachable — `query_offset >= 0` is part of the op's contract, so column 0 is
// always visible — so it is stated here rather than written as a guard no test
// can trip.
const MASKED: f32 = -3.402823e+38;

// The one tile of scores that is ever alive. Nothing else in this kernel holds
// more than a scalar per invocation, which is the whole point of the op.
var<workgroup> tile_scores: array<f32, TILE>;

@compute @workgroup_size(WORKGROUP_SIZE)
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
  let v_head = head * params.S * params.Dv;
  let o_row = (head * params.L + i) * params.Dv;

  // Where this query row's slice of the bias starts; each axis collapses to 0
  // when the caller broadcast it. Identical to ops/attention's scores kernel.
  let mb = select(0u, batch, params.mask_batch > 1u);
  let mh = select(0u, h, params.mask_heads > 1u);
  let mr = select(0u, i, params.mask_rows > 1u);
  let m_row = ((mb * params.mask_heads + mh) * params.mask_rows + mr) * params.S;

  let tiles = (params.S + TILE - 1u) / TILE;
  // One invocation per output channel. When Dv exceeds the workgroup the keys
  // are walked again for the next block of channels — recomputing the scores
  // rather than storing them, which is the trade this op is named for. Both
  // bounds are uniform across the workgroup, which is what lets the barriers
  // below sit inside these loops at all.
  let passes = (params.Dv + WORKGROUP_SIZE - 1u) / WORKGROUP_SIZE;

  for (var sweep = 0u; sweep < passes; sweep += 1u) {
    let c = sweep * WORKGROUP_SIZE + tid;
    // Invocations past Dv still build tiles and still reach every barrier; what
    // they skip is the accumulation, which has no channel to accumulate into and
    // would index V out of bounds looking for one.
    let owns_channel = c < params.Dv;

    var m: f32 = MASKED;   // running maximum over every score seen so far
    var l: f32 = 0.0;      // running sum of exp(score - m), rescaled as m moves
    var acc: f32 = 0.0;    // running sum of exp(score - m) * v[:, c]

    for (var t = 0u; t < tiles; t += 1u) {
      let base = t * TILE;
      // The last tile is short whenever S is not a multiple of TILE, and every
      // loop below stops at `limit` rather than padding the tail with MASKED.
      // Same answer either way; this way there is no index past S to get wrong.
      let limit = min(TILE, params.S - base);

      // Nobody may still be reading the previous tile when this one lands.
      workgroupBarrier();
      if (tid < limit) {
        let j = base + tid;
        var value: f32 = MASKED;
        // The mask, evaluated exactly once per element and nowhere else. `<=`,
        // and `j` against `i`, is torch's `tril` taken from the top-left corner.
        if (params.causal == 0u || i32(j) <= i32(i) + params.query_offset) {
          var dot: f32 = 0.0;
          for (var d: u32 = 0u; d < params.D; d += 1u) {
            dot += q[q_row + d] * k[k_head + j * params.D + d];
          }
          // PyTorch's float `attn_mask`, added rather than selected on, so a
          // key-padding mask and an ALiBi bias compose. See ops/attention.
          value = dot * params.scale + mask[m_row + j];
        }
        tile_scores[tid] = value;
      }
      workgroupBarrier();

      if (owns_channel) {
        // The online update. Every invocation runs it on the same tile and
        // reaches the same m and l, which is why neither needs a workgroup
        // reduction: they are replicated, not shared. Each invocation needs
        // every probability in the tile for its own dot product against V
        // anyway, so the only thing duplicated is the exp.
        var tile_max: f32 = MASKED;
        for (var s: u32 = 0u; s < limit; s += 1u) {
          tile_max = max(tile_max, tile_scores[s]);
        }

        // `corr` is what everything accumulated so far is worth now that the
        // maximum has moved. It is exactly 1 while the maximum stands, so a
        // kernel missing it is right on any input whose largest score is in the
        // first tile — which is why there is a test whose largest score is not.
        let m_new = max(m, tile_max);
        let corr = exp(m - m_new);
        l = l * corr;
        acc = acc * corr;

        for (var s: u32 = 0u; s < limit; s += 1u) {
          let p = exp(tile_scores[s] - m_new);
          l = l + p;
          acc = acc + p * v[v_head + (base + s) * params.Dv + c];
        }
        m = m_new;
      }
    }

    // The single normalisation, after every key has been seen. `l >= 1` whenever
    // any score was finite: the key holding the final maximum contributed
    // exp(0). `l == 0` is the row where every key is masked — unreachable before
    // #77, since `query_offset >= 0` kept column 0 visible, and reachable now
    // through `mask`. It gets zeros rather than 0/0 = NaN, which is what SDPA
    // returns (`aten::_safe_softmax`, measured on torch 2.10).
    //
    // The test is on `l` and not on `m`, and the two are not interchangeable:
    // `m` starts at MASKED, so `max(m, -inf)` is MASKED (measured on this
    // device) and the running maximum never reports -inf however masked the row
    // is. `l` is exact — the key holding the maximum contributes exp(0) = 1.
    if (owns_channel) {
      output[o_row + c] = select(acc / l, 0.0, l == 0.0);
    }
  }
}
