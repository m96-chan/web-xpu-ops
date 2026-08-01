// ALiBi bias application (Press et al., arXiv:2108.12409).
//
//   out[h, i, j] = scores[h, i, j] + m_h * (j - (i + pos_offset))
//
// The bias is the *relative* distance from query to key, which is the form the
// paper defines: zero on the diagonal, one slope more negative per step back.
// BLOOM's variant uses the key's absolute index instead; the two differ by a
// per-row constant that the following softmax removes, but this op hands back
// the tensor and not the softmax, so the difference is observable. See
// `reference.ts` for the full note.
//
// Nothing is masked here. Entries with j > i get the plain linear extension and
// the caller's causal mask is expected to remove them; writing -inf would fold
// a masking policy into a bias op.
//
// The slopes arrive as a buffer rather than being derived here. They are the
// half of ALiBi that implementations quietly disagree on, so they get their own
// kernel (`slopes.wgsl`) and their own test — a fault in one must not be able
// to hide inside the other.
//
// Layout:
//   scores: [num_heads, M, N] f32
//   slopes: [num_heads] f32
//   output: [num_heads, M, N] f32
//   Dispatched one lane per score.

struct Params {
  num_heads: u32,
  M: u32,          // query rows
  N: u32,          // key columns
  pos_offset: u32, // position of the first query row, for KV-cache continuation
}

@group(0) @binding(0) var<storage, read> scores: array<f32>;
@group(0) @binding(1) var<storage, read> slopes: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
) {
  let total = params.num_heads * params.M * params.N;
  let index = gid.x;
  if (index >= total) {
    return;
  }

  let key = index % params.N;
  let rest = index / params.N;
  let query = rest % params.M;
  let head = rest / params.M;

  // Both sides widened before subtracting: `key - (query + pos_offset)` in u32
  // wraps to a huge positive number for every entry below the diagonal, which
  // is most of a causal block.
  let distance = f32(key) - (f32(query) + f32(params.pos_offset));

  output[index] = scores[index] + slopes[head] * distance;
}
