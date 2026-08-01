// STFT: one windowed DFT per frame, one-sided, frame-major output.
//
// A naive DFT, not an FFT. n_fft = 1920 is 2^7 * 15, so radix-2 does not apply
// and the alternatives (Bluestein, mixed radix) are a different op's worth of
// code. This runs beside a transformer over a few hundred frames, where it is
// not the bottleneck, and the conventions being right matters more than the
// asymptotics — see reference.ts. Optimise after something measures it.
//
// Layout:
//   signal:  [signalLength] f32, plus whatever the caller left past the end
//   window:  [nFft] f32
//   outReal: [frames, bins] f32   frame-major, bins = floor(nFft/2) + 1
//   outImag: [frames, bins] f32
//
// One workgroup per frame; the 256 lanes walk the bins at a stride. Dispatched
// with exactly `frames` workgroups, so the frame index needs no bound check —
// there is no frame to guard against.

struct Params {
  nFft: u32,
  hop: u32,
  bins: u32,
  signalLength: u32,
  // floor(nFft/2) when centred, 0 when not. Resolved by the caller so the
  // shader has no convention of its own.
  pad: u32,
}

@group(0) @binding(0) var<storage, read> signal: array<f32>;
@group(0) @binding(1) var<storage, read> win: array<f32>;
@group(0) @binding(2) var<storage, read_write> outReal: array<f32>;
@group(0) @binding(3) var<storage, read_write> outImag: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

const WORKGROUP_SIZE: u32 = 256u;
const TWO_PI: f32 = 6.28318530717958647692;

/// Reflect padding: [-1] -> [1], [L] -> [L-2]. The edge sample is not repeated,
/// which is what pad_mode="reflect" means in torch. One reflection is enough
/// because the reference refuses a signal shorter than the padding.
fn reflected(index: i32) -> f32 {
  let last = i32(params.signalLength) - 1;
  var i = index - i32(params.pad);
  if (i < 0) { i = -i; }
  if (i > last) { i = 2 * last - i; }
  return signal[u32(i)];
}

@compute @workgroup_size(256)
fn main(
  @builtin(workgroup_id) wg_id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>,
) {
  let frame = wg_id.x;
  let start = frame * params.hop;

  for (var k = local_id.x; k < params.bins; k += WORKGROUP_SIZE) {
    var sumReal: f32 = 0.0;
    var sumImag: f32 = 0.0;
    for (var n = 0u; n < params.nFft; n += 1u) {
      let value = reflected(i32(start + n)) * win[n];
      // Folded into one turn while it is still an integer. k*n reaches 1.8M at
      // nFft=1920, and f32 holds that many radians to about a tenth of one —
      // the twiddle would be noise. u32 carries the product exactly.
      let angle = -TWO_PI * (f32((k * n) % params.nFft) / f32(params.nFft));
      sumReal += value * cos(angle);
      sumImag += value * sin(angle);
    }
    let out = frame * params.bins + k;
    outReal[out] = sumReal;
    outImag[out] = sumImag;
  }
}
