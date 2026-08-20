// `codes[outFeatures, inFeatures]` int8 (packed, matvecQ8's wire format) ->
// `output[inFeatures, outFeatures]` f32: dequantize (row scale) and
// transpose in one dispatch. See reference.ts for why this exists (issue
// #117's GPU-resident LLM prefill) and rule 8's own record of the CPU-side
// three-pass cost this replaces.
//
// One invocation per output element (`outFeatures * inFeatures` of them),
// each reading its own packed word and unpacking one byte from it — the
// same redundant-word-read matvecQ8's own kernel structure avoids by
// striding a whole workgroup across one row's words at once, not taken
// here because this dispatch is one-shot per weight per generation (never
// per token, `reference.ts`'s doc), so read amplification here costs
// nothing this class's own decode-time `matvecQ8` dispatches would notice.

struct Params {
  outFeatures: u32,
  inFeatures: u32,
}

@group(0) @binding(0) var<storage, read> weight: array<u32>;
@group(0) @binding(1) var<storage, read> scale: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

const WORKGROUP_SIZE: u32 = 256u;

// `ops/matvec/wgsl/q8.wgsl#unpack_i8`, copied rather than imported — WGSL
// has no cross-file include, and this is the one place this op needs it.
fn unpack_i8(word: u32, lane: u32) -> f32 {
  return f32(extractBits(bitcast<i32>(word), lane * 8u, 8u));
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let total = params.outFeatures * params.inFeatures;
  if (idx >= total) {
    return;
  }
  let row = idx / params.inFeatures;
  let col = idx % params.inFeatures;
  let words_per_row = (params.inFeatures + 3u) / 4u;
  let word = weight[row * words_per_row + (col >> 2u)];
  let value = unpack_i8(word, col & 3u) * scale[row];
  output[col * params.outFeatures + row] = value;
}
