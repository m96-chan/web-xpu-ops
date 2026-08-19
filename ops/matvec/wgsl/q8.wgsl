// matvecQ8 (W8A32 GEMV): out[i] = (sum_k unpack(weight[i, k]) * vector[k]) * scale[i]
//
// Layout:
//   weight: [N, ceil(K/4)] u32, row-major, 4 int8 codes packed per word,
//           least-significant byte first (word bits 0..7 hold column 4*w+0,
//           8..15 hold 4*w+1, and so on). A code is stored as its
//           two's-complement byte, so unpacking sign-extends the low byte
//           rather than just masking it.
//   scale:  [N]    f32, one per row (`quantize`'s per-row absmax convention)
//   vector: [K]    f32, shared by every row
//   output: [N]    f32
//
// Same streaming shape as ops/matvec/wgsl/kernel.wgsl: one workgroup per
// output row, 256 lanes walking that row with a stride of the workgroup
// width. The unit of stride here is a *word*, not a column — lane t reads
// packed word base + t on each pass, which is 256 consecutive u32 (1024
// int8 codes) the memory system can serve as a burst, then unpacks its own
// four codes from the word it already holds instead of issuing four
// separate byte reads. That is the whole reason to pack in the first place:
// four columns for the price of one 32-bit transaction.
//
// The row scale multiplies the fully-reduced partial sum once, not each of
// its terms — algebraically identical since the scale does not depend on
// the column, and it is `K - 1` fewer multiplies per row.

struct Params {
  N: u32,
  K: u32,
}

@group(0) @binding(0) var<storage, read> weight: array<u32>;
@group(0) @binding(1) var<storage, read> scale: array<f32>;
@group(0) @binding(2) var<storage, read> vector: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

const WORKGROUP_SIZE: u32 = 256u;

var<workgroup> shared_sum: array<f32, 256>;

// Sign-extends the byte at `lane` (0..3) of a packed word: bits 0..7 for lane
// 0, up through 24..31 for lane 3. `extractBits` on a *signed* base carries
// the sign of the extracted field, which is exactly two's-complement
// unpacking; `word` only reaches here as a `u32` because storage buffers
// cannot hold mixed-sign words, so it is bitcast first.
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

  // Partial dot product, one packed word — up to four columns — per lane per
  // pass. `col < params.K` is the tail guard: when K is not a multiple of 4
  // the last word's high lane(s) hold padding that must never reach the sum.
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
    output[row] = shared_sum[0] * scale[row];
  }
}
