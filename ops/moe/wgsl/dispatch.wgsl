// MoE dispatch: reorder tokens into per-expert contiguous runs.
//
//   buffer[e, pos, :] = x[t, :]   where expert[t, r] == e
//   pos[t, r]         = that position, or -1 if the token was dropped
//
// The buffer is what an expert's weights are applied to, which is the reason
// this op exists: one matmul per expert over a contiguous block, instead of a
// gather inside the FFN.
//
// The capacity rule — rank first, then token index, everything past C dropped —
// is GShard's, and is argued in ops/moe/reference.ts. It is a policy and not an
// implementation detail: it decides which tokens the model does not see.
//
// It is also why there is no atomic counter here. A counter would give each
// arriving token the next free seat, which is one instruction instead of the
// scan below — and which set of tokens gets dropped would then depend on the
// order the scheduler happened to run the workgroups in, differing between two
// runs of the same input on the same device. ops/scatter refuses "last write
// wins" for the same reason. So each pair counts, from the data, how many pairs
// rank ahead of it; the answer depends on nothing but the input.
//
// One workgroup per (token, rank) pair, flat in row-major (t, r) order, so the
// caller dispatches exactly T*k workgroups and there is no partial one. Lane 0
// does the counting scan, then the whole workgroup copies the row. The scan is
// O(T*k) per pair and is the obvious thing to make parallel later; it is left
// serial because a wrong capacity rule is worse than a slow one, and nothing
// here has been measured yet.
//
// Layout:
//   x:      [T, D]    f32
//   expert: [T, k]    i32, anything outside [0, E) is dropped
//   buffer: [E, C, D] f32, seats nobody reached are left untouched — the caller
//                          supplies the buffer and gather never reads them
//   pos:    [T, k]    i32

struct Params {
  T: u32,
  k: u32,
  E: u32,
  C: u32,
  D: u32,
}

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> expert: array<i32>;
@group(0) @binding(2) var<storage, read_write> buffer: array<f32>;
@group(0) @binding(3) var<storage, read_write> pos: array<i32>;
@group(0) @binding(4) var<uniform> params: Params;

const WORKGROUP_SIZE: u32 = 256u;

var<workgroup> shared_slot: i32;

@compute @workgroup_size(256)
fn main(
  @builtin(workgroup_id) wg_id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>,
) {
  let pair = wg_id.x;
  let token = pair / params.k;
  let rank = pair % params.k;
  let chosen = expert[pair];

  if (local_id.x == 0u) {
    var slot = -1;
    // Out of range is dropped, not clamped: a clamped id routes the token to an
    // expert it never chose, and every value downstream still looks like a
    // number. -1 is what the router itself writes when k exceeds E.
    if (chosen >= 0 && chosen < i32(params.E)) {
      var earlier = 0;
      let pairs = params.T * params.k;
      for (var p = 0u; p < pairs; p += 1u) {
        if (expert[p] != chosen) {
          continue;
        }
        let other_token = p / params.k;
        let other_rank = p % params.k;
        // Rank first, then token: a token's first choice outranks another
        // token's second.
        if (other_rank < rank || (other_rank == rank && other_token < token)) {
          earlier += 1;
        }
      }
      // Past capacity. `earlier` counts the dropped ones too, so a drop does
      // not free the seat for whoever comes next — the rule is "the first C in
      // this order", not "the first C that happened to survive".
      if (earlier < i32(params.C)) {
        slot = earlier;
      }
    }
    shared_slot = slot;
    pos[pair] = slot;
  }
  workgroupBarrier();

  let slot = shared_slot;
  if (slot < 0) {
    return;
  }

  let destination = (u32(chosen) * params.C + u32(slot)) * params.D;
  let source = token * params.D;
  for (var d = local_id.x; d < params.D; d += WORKGROUP_SIZE) {
    // Verbatim. The gate is applied in gather, once — weighting here would put
    // it inside the expert FFN, and applying it at both ends squares it.
    buffer[destination + d] = x[source + d];
  }
}
