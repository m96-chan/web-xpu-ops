// matvecQ4G128 (W4A32 GEMV): out[i] = sum_k unpack(weight[i, k]) * scale[i, k/128] * vector[k]
//
// Layout:
//   weight: [N, ceil(K/8)] u32, row-major, 8 int4 codes packed per word,
//           least-significant nibble first (word bits 0..3 hold column 8*w+0,
//           4..7 hold 8*w+1, and so on). A code is stored as its
//           two's-complement nibble, so unpacking sign-extends four bits
//           rather than masking them — `0xf` is -1, not 15.
//   scale:  [N, ceil(K/128)] f32, row-major — one absmax-derived factor per
//           group of GROUP_SIZE contiguous columns (ops/matvec/reference.ts's
//           `quantizeQ4G128`). Not [N]: four bits cannot hold a whole row's
//           dynamic range, which is the measurement that made this op exist
//           (issue #137).
//   vector: [K]    f32, shared by every row
//   output: [N]    f32
//
// Same streaming shape as ops/matvec/wgsl/q8.wgsl: one workgroup per output
// row, 256 lanes walking that row with a stride of the workgroup width, the
// unit of stride being a *word* rather than a column. A word here carries
// eight columns instead of four, so one pass of the workgroup covers 2048
// columns.
//
// GROUP_SIZE = 128 is a multiple of the eight codes in a word, which is the
// property this kernel is built on: a packed word never straddles two groups,
// so the scale can be read **once per word** and applied to that word's eight
// terms together, instead of once per column. Without that alignment every
// lane would need its own per-column scale lookup and the format would cost
// more to read than it saves in bytes. ops/matvec/reference.ts#Q4_GROUP_SIZE
// carries the same constant on the host side; changing one means changing
// both (rule 2 — copied, not re-derived).
//
// Unlike q8.wgsl the row scale cannot be hoisted all the way out of the
// reduction, because it does depend on the column — only on the *group* of it.
// So the hoisting stops at the word, and the workgroup reduction below sums
// terms that are already scaled.

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
const GROUP_SIZE: u32 = 128u;
/** Codes per packed word. GROUP_SIZE must stay a multiple of this. */
const CODES_PER_WORD: u32 = 8u;

var<workgroup> shared_sum: array<f32, 256>;

// Sign-extends the nibble at `lane` (0..7) of a packed word: bits 0..3 for
// lane 0, up through 28..31 for lane 7. `extractBits` on a *signed* base
// carries the sign of the extracted field, which is exactly two's-complement
// unpacking; `word` only reaches here as a `u32` because storage buffers
// cannot hold mixed-sign words, so it is bitcast first. Same shape as
// ops/matvec/wgsl/q8.wgsl#unpack_i8 at half the field width.
fn unpack_i4(word: u32, lane: u32) -> f32 {
  return f32(extractBits(bitcast<i32>(word), lane * 4u, 4u));
}

@compute @workgroup_size(256)
fn main(
  @builtin(workgroup_id) wg_id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>,
) {
  let row = wg_id.x;
  let tid = local_id.x;
  let words_per_row = (params.K + CODES_PER_WORD - 1u) / CODES_PER_WORD;
  let groups_per_row = (params.K + GROUP_SIZE - 1u) / GROUP_SIZE;
  let row_word_offset = row * words_per_row;
  let row_group_offset = row * groups_per_row;

  // Partial dot product, one packed word — up to eight columns — per lane per
  // pass. `col < params.K` is the tail guard: when K is not a multiple of 8
  // the last word's high nibble(s) hold padding that must never reach the sum.
  var partial: f32 = 0.0;
  for (var word_index = tid; word_index < words_per_row; word_index += WORKGROUP_SIZE) {
    let word = weight[row_word_offset + word_index];
    let base_col = word_index * CODES_PER_WORD;
    // One lookup per word, legal only because a word cannot straddle a group.
    let group_scale = scale[row_group_offset + base_col / GROUP_SIZE];
    var word_sum: f32 = 0.0;
    for (var lane = 0u; lane < CODES_PER_WORD; lane += 1u) {
      let col = base_col + lane;
      if (col >= params.K) {
        break;
      }
      word_sum += unpack_i4(word, lane) * vector[col];
    }
    partial += word_sum * group_scale;
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
    output[row] = shared_sum[0];
  }
}
