// Scatter: output[row][indices[row][slot]] += src[row][slot]
//
// Colliding indices accumulate — see ops/scatter/reference.ts for why that is
// the rule and what it corresponds to in PyTorch (`scatter_add_` along dim=1,
// onto an implicitly zero `self`). The short version: on a GPU nothing else is
// implementable. "Last write wins" has no meaning when the order threads reach
// a slot is unspecified.
//
// The output is declared atomic<u32> and bitcast, because WGSL has atomics on
// integers only. The compare-exchange loop below is what an f32 atomic add
// costs; it is the price of the semantics, not an optimisation left undone.
//
// Layout:
//   src:     [N, S] f32
//   indices: [N, S] i32, each in [0, D); anything else is dropped
//   output:  [N, D] f32, zero where no index named it
//
// One workgroup per row, each thread walking the row with a stride of 256.

struct Params {
  N: u32,
  S: u32,
  D: u32,
}

@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read> indices: array<i32>;
@group(0) @binding(2) var<storage, read_write> output: array<atomic<u32>>;
@group(0) @binding(3) var<uniform> params: Params;

const WORKGROUP_SIZE: u32 = 256u;

// f32 atomic add by compare-exchange. Retries on contention, which is exactly
// the case this op is for: the sum is only wrong if a losing thread keeps the
// stale value it read, so the loop reloads from `old_value` rather than
// starting over with an atomicLoad.
fn add_f32(slot: u32, value: f32) {
  var old = atomicLoad(&output[slot]);
  loop {
    let attempt = atomicCompareExchangeWeak(&output[slot], old, bitcast<u32>(bitcast<f32>(old) + value));
    if (attempt.exchanged) {
      break;
    }
    old = attempt.old_value;
  }
}

@compute @workgroup_size(256)
fn main(
  @builtin(workgroup_id) wg_id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>,
) {
  let row = wg_id.x;
  if (row >= params.N) {
    return;
  }

  let row_src = row * params.S;
  let row_out = row * params.D;

  for (var slot = local_id.x; slot < params.S; slot += WORKGROUP_SIZE) {
    let column = indices[row_src + slot];
    // Not a formality: `row_out + u32(column)` for an out-of-range column is
    // still inside the flat output buffer, so an unchecked write corrupts a
    // different row rather than being caught. Negative is out of range too.
    if (column < 0 || column >= i32(params.D)) {
      continue;
    }
    add_f32(row_out + u32(column), src[row_src + slot]);
  }
}
