// matvecQ8Ffn: out[i] = silu(sum_k unpack(weight_gate[i,k])*vector[k]*scale_gate[i])
//                          * (sum_k unpack(weight_up[i,k])*vector[k]*scale_up[i])
//
// Issue #111: fuses matvecQ8(gate) + matvecQ8(up) + activation(silu) +
// elementwise(multiply) — four dispatches in llm/engine-q8-resident.ts's
// decode step — into one. Each dispatch there pays a fixed per-submit cost
// independent of the bytes it moves, so cutting dispatch count is the actual
// lever (see q8.wgsl's own doc for the streaming shape this keeps: one
// workgroup per output row, 256 lanes striding across packed words).
//
// Same wire format as q8.wgsl for *both* weights (see that file's own doc:
// [N, ceil(K/4)] u32, four codes per word, least-significant byte first,
// two's-complement). The fusion reads one word from each of the two packed
// weights per pass over `vector` — `vector` itself is read once per pass,
// shared by both dot products, rather than twice (once per un-fused
// dispatch) the way running matvecQ8(gate) and matvecQ8(up) apart would.
//
// Layout:
//   weight_gate, weight_up: [N, ceil(K/4)] u32, row-major, matvecQ8's packed format
//   scale_gate, scale_up:   [N]            f32
//   vector:                 [K]            f32, shared by every row and both weights
//   output:                 [N]            f32

struct Params {
  N: u32,
  K: u32,
}

@group(0) @binding(0) var<storage, read> weight_gate: array<u32>;
@group(0) @binding(1) var<storage, read> scale_gate: array<f32>;
@group(0) @binding(2) var<storage, read> weight_up: array<u32>;
@group(0) @binding(3) var<storage, read> scale_up: array<f32>;
@group(0) @binding(4) var<storage, read> vector: array<f32>;
@group(0) @binding(5) var<storage, read_write> output: array<f32>;
@group(0) @binding(6) var<uniform> params: Params;

const WORKGROUP_SIZE: u32 = 256u;

// x and y hold the gate and up partial sums together, so one reduction pass
// (below) finishes both instead of running the tree-reduce twice.
var<workgroup> shared_sum: array<vec2<f32>, 256>;

// Sign-extends the byte at `lane` (0..3) of a packed word — see q8.wgsl's
// own `unpack_i8` doc for why `extractBits` on a bitcast signed base is
// exactly two's-complement unpacking.
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

  var partial_gate: f32 = 0.0;
  var partial_up: f32 = 0.0;
  for (var word_index = tid; word_index < words_per_row; word_index += WORKGROUP_SIZE) {
    let wg = weight_gate[row_word_offset + word_index];
    let wu = weight_up[row_word_offset + word_index];
    let base_col = word_index * 4u;
    for (var lane = 0u; lane < 4u; lane += 1u) {
      let col = base_col + lane;
      if (col >= params.K) {
        break;
      }
      let v = vector[col];
      partial_gate += unpack_i8(wg, lane) * v;
      partial_up += unpack_i8(wu, lane) * v;
    }
  }

  shared_sum[tid] = vec2<f32>(partial_gate, partial_up);
  workgroupBarrier();

  for (var stride = WORKGROUP_SIZE / 2u; stride > 0u; stride >>= 1u) {
    if (tid < stride) {
      shared_sum[tid] += shared_sum[tid + stride];
    }
    workgroupBarrier();
  }

  if (tid == 0u) {
    let g = shared_sum[0].x * scale_gate[row];
    let u = shared_sum[0].y * scale_up[row];
    let silu = g / (1.0 + exp(-g));
    output[row] = silu * u;
  }
}
