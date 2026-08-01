// MoE gather (GShard calls it combine): put the expert outputs back in token
// order and weight them.
//
//   out[t, :] = Σ_r gate[t, r] * y[expert[t, r], pos[t, r], :]
//
// This is where the gate is applied, and the only place. Dispatch copies token
// rows verbatim; multiplying at both ends squares the gate, which for
// normalised weights and a small k gives an answer that is a few per cent small
// and otherwise entirely plausible.
//
// Not ops/gather. That one is index_select over an embedding table.
//
// One workgroup per token, so the caller dispatches exactly T workgroups and
// there is no partial one; lanes walk the row at the workgroup stride, which
// keeps the write contiguous. Every lane re-reads the k routing entries — they
// are the same for the whole workgroup and could be staged in shared memory,
// which is the first thing to try when this is measured. It is k values against
// D floats of traffic.
//
// Layout:
//   y:      [E, C, D] f32, expert outputs in the layout dispatch produced
//   expert: [T, k]    i32
//   pos:    [T, k]    i32, -1 for a token the capacity rule dropped
//   gate:   [T, k]    f32
//   out:    [T, D]    f32, a zero row for a token whose every choice was dropped

struct Params {
  T: u32,
  k: u32,
  E: u32,
  C: u32,
  D: u32,
}

@group(0) @binding(0) var<storage, read> y: array<f32>;
@group(0) @binding(1) var<storage, read> expert: array<i32>;
@group(0) @binding(2) var<storage, read> pos: array<i32>;
@group(0) @binding(3) var<storage, read> gate: array<f32>;
@group(0) @binding(4) var<storage, read_write> combined: array<f32>;
@group(0) @binding(5) var<uniform> params: Params;

const WORKGROUP_SIZE: u32 = 256u;

@compute @workgroup_size(256)
fn main(
  @builtin(workgroup_id) wg_id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>,
) {
  let token = wg_id.x;
  let out_row = token * params.D;
  let route_row = token * params.k;

  for (var d = local_id.x; d < params.D; d += WORKGROUP_SIZE) {
    var accumulator = 0.0;
    for (var r = 0u; r < params.k; r += 1u) {
      let slot = pos[route_row + r];
      let chosen = expert[route_row + r];
      // A dropped slot contributes nothing. The other two are inconsistent
      // inputs the kernel is not entitled to trust: both an expert past E and a
      // position past C address inside the flat buffer, so an unchecked read
      // returns a real number belonging to a different expert.
      if (slot < 0 || slot >= i32(params.C) || chosen < 0 || chosen >= i32(params.E)) {
        continue;
      }
      accumulator += gate[route_row + r] * y[(u32(chosen) * params.C + u32(slot)) * params.D + d];
    }
    // Written unconditionally: a token that lost every seat gets zeros, which
    // in a transformer block is the residual passing through untouched.
    combined[out_row + d] = accumulator;
  }
}
