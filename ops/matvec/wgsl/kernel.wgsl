// matvec (GEMV): out[i] = sum_k matrix[i, k] * vector[k]
//
// Layout:
//   matrix: [M, K] f32, row-major
//   vector: [K]    f32, shared by every row
//   output: [M]    f32
//
// Written for streaming, not for tiling. Every weight is read exactly once and
// reused by nobody, so there is no reuse for a tile to capture — staging the
// matrix through workgroup memory would only add a barrier and a copy in front
// of the same number of bytes. The one thing that matters is the order the
// weights leave memory in.
//
// So: one workgroup owns one output row, and its 256 lanes walk that row with a
// stride of the workgroup width. At every step lane t reads element base + t,
// which is 256 consecutive f32 — one contiguous run the memory system can serve
// as a burst. The obvious alternative, giving each lane a contiguous chunk of
// K/256 elements, issues 256 reads that are K/256 elements apart and pays for a
// separate transaction per lane.
//
// `vector` is the exception to "read once": all M workgroups read all of it. It
// is K elements against M*K, so it is the small side of the traffic, and it is
// read in the same coalesced order as the matrix.
//
// The dispatch is exactly one workgroup per row, so there is no row-bounds
// branch here. A guard for `wg_id.x >= M` could never be reached, and an
// unreachable branch is a branch no test can hold to account.

struct Params {
  M: u32,
  K: u32,
}

@group(0) @binding(0) var<storage, read> matrix: array<f32>;
@group(0) @binding(1) var<storage, read> vector: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

const WORKGROUP_SIZE: u32 = 256u;

var<workgroup> shared_sum: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
  @builtin(workgroup_id) wg_id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>,
) {
  let row = wg_id.x;
  let tid = local_id.x;
  let row_offset = row * params.K;

  // Partial dot product. `col < params.K` is the whole of the tail handling:
  // when K is not a multiple of 256 the last pass simply has fewer lanes in it.
  // Lanes that fall off the end contribute the 0.0 they started with, which is
  // also what happens for every K below the workgroup width.
  var partial: f32 = 0.0;
  for (var col = tid; col < params.K; col += WORKGROUP_SIZE) {
    partial += matrix[row_offset + col] * vector[col];
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
