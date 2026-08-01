// MoE router: top-k over expert logits, plus the gate weights.
//
//   expert[t, r] = the r-th best expert for token t
//   gate[t, r]   = its weight
//
// The conventions — softmax before top-k, renormalisation as a flag rather than
// a default, ties to the lower expert index — are argued in ops/moe/reference.ts
// and are not re-argued here. What follows is only how they are computed.
//
// One thread per token, flat over T. Not one workgroup per token: E is the
// expert count, which is 8 to 64 in every model that ships, so a workgroup per
// token would idle 200-odd lanes and buy a tree reduction over a row shorter
// than a cache line. The cost of the choice is that the top-k scan is serial in
// each thread, O(k*E), which at those sizes is tens of iterations.
//
// The caller dispatches ceil(T / 256) workgroups in x alone.
//
// Layout:
//   logits: [T, E] f32
//   expert: [T, k] i32, -1 where no expert was left to pick (only when k > E)
//   gate:   [T, k] f32, 0 alongside a -1

struct Params {
  T: u32,
  E: u32,
  k: u32,
  normalize: u32, // 1 = gates sum to 1 over the k (Mixtral), 0 = full softmax
}

@group(0) @binding(0) var<storage, read> logits: array<f32>;
@group(0) @binding(1) var<storage, read_write> expert: array<i32>;
@group(0) @binding(2) var<storage, read_write> gate: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let t = gid.x;
  if (t >= params.T) {
    return;
  }

  let row = t * params.E;
  let out_row = t * params.k;

  // Softmax denominator, max-subtracted. exp(800) is Infinity in f32 and takes
  // the row to NaN, at which point every comparison in the scan below is false
  // and the top-k returns nothing. Logits at that scale are not hypothetical.
  var row_max = logits[row];
  for (var e = 1u; e < params.E; e += 1u) {
    row_max = max(row_max, logits[row + e]);
  }
  var denominator = 0.0;
  for (var e = 0u; e < params.E; e += 1u) {
    denominator += exp(logits[row + e] - row_max);
  }

  // Top-k by selection, one pass over E per rank, with the previous winner as
  // the ceiling. The ceiling is a (value, index) pair rather than a value
  // because equal logits have to be walked through in index order — see the
  // tie rule in reference.ts.
  //
  // Selection over the logits, not over the probabilities: exp() in f32 can
  // round two distinct logits onto the same probability, which would invent a
  // tie and hand a well-defined order to the index rule for no reason.
  var have_ceiling = false;
  var ceiling_value = 0.0;
  var ceiling_index = -1;
  var gate_sum = 0.0;

  for (var r = 0u; r < params.k; r += 1u) {
    var best_index = -1;
    var best_value = 0.0;
    for (var e = 0u; e < params.E; e += 1u) {
      let value = logits[row + e];
      let index = i32(e);
      // Strictly below the previous winner in the (value desc, index asc) order.
      if (have_ceiling && !(value < ceiling_value || (value == ceiling_value && index > ceiling_index))) {
        continue;
      }
      // `>` and not `>=`, scanning e upwards: on a tie the first one seen wins,
      // and that is the lowest index.
      if (best_index < 0 || value > best_value) {
        best_index = index;
        best_value = value;
      }
    }

    // Reached only when k > E: every expert has already been taken. Written
    // rather than skipped because the output buffer belongs to the caller and
    // nothing else fills it in.
    if (best_index < 0) {
      expert[out_row + r] = -1;
      gate[out_row + r] = 0.0;
      continue;
    }

    let weight = exp(best_value - row_max) / denominator;
    expert[out_row + r] = best_index;
    gate[out_row + r] = weight;
    gate_sum += weight;
    have_ceiling = true;
    ceiling_value = best_value;
    ceiling_index = best_index;
  }

  if (params.normalize != 0u) {
    // gate_sum contains the largest probability in the row, which is at least
    // 1/E, so this is never a division by zero. Nothing here divides by a token
    // count — that is the division that makes an empty expert produce NaN.
    for (var r = 0u; r < params.k; r += 1u) {
      gate[out_row + r] = gate[out_row + r] / gate_sum;
    }
  }
}
