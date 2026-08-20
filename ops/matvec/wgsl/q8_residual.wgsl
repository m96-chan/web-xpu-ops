// matvecQ8Residual: out[i] = residual[i] + (sum_k unpack(weight[i,k]) * vector[k]) * scale[i]
//
// Issue #111: fuses matvecQ8 + a following elementwise(add) — two dispatches
// in llm/engine-q8-resident.ts's decode step, once for o_proj's post-attention
// residual and once for down_proj's post-FFN residual — into one. Same
// reasoning as q8_ffn.wgsl: each dispatch pays a fixed per-submit cost this
// repository's own measurement shows dominates over bytes moved, so cutting
// dispatch count is the lever.
//
// Identical wire format and streaming shape to q8.wgsl (one workgroup per
// output row, 256 lanes striding across packed words — see that file's own
// doc), plus one more per-row input: `residual`, added after the row's scale
// the same way `scale` itself is applied once per row rather than once per
// term (q8.wgsl's own note on why: the row scale/residual do not depend on
// the column, so folding either into the per-term loop would be extra work
// for the same answer).
//
// Layout:
//   weight:   [N, ceil(K/4)] u32, row-major, matvecQ8's packed format
//   scale:    [N]            f32
//   vector:   [K]            f32, shared by every row
//   residual: [N]            f32, added once per row after the scale
//   output:   [N]            f32

struct Params {
  N: u32,
  K: u32,
}

@group(0) @binding(0) var<storage, read> weight: array<u32>;
@group(0) @binding(1) var<storage, read> scale: array<f32>;
@group(0) @binding(2) var<storage, read> vector: array<f32>;
@group(0) @binding(3) var<storage, read> residual: array<f32>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;
@group(0) @binding(5) var<uniform> params: Params;

const WORKGROUP_SIZE: u32 = 256u;

var<workgroup> shared_sum: array<f32, 256>;

// Sign-extends the byte at `lane` (0..3) of a packed word — see q8.wgsl's
// own `unpack_i8` doc.
fn unpack_i8(word: u32, lane: u32) -> f32 {
  return f32(extractBits(bitcast<i32>(word), lane * 8u, 8u));
}

@compute @workgroup_size(256)
fn main(
  @builtin(workgroup_id) wg_id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>,
) {
  let row = wg_id.x;
  let tid = local_id.x;
  let words_per_row = (params.K + 3u) / 4u;
  let row_word_offset = row * words_per_row;

  var partial: f32 = 0.0;
  for (var word_index = tid; word_index < words_per_row; word_index += WORKGROUP_SIZE) {
    let word = weight[row_word_offset + word_index];
    let base_col = word_index * 4u;
    for (var lane = 0u; lane < 4u; lane += 1u) {
      let col = base_col + lane;
      if (col >= params.K) {
        break;
      }
      partial += unpack_i8(word, lane) * vector[col];
    }
  }

  shared_sum[tid] = partial;
  workgroupBarrier();

  for (var stride = WORKGROUP_SIZE / 2u; stride > 0u; stride >>= 1u) {
    if (tid < stride) {
      shared_sum[tid] += shared_sum[tid + stride];
    }
    workgroupBarrier();
  }

  if (tid == 0u) {
    output[row] = shared_sum[0] * scale[row] + residual[row];
  }
}
