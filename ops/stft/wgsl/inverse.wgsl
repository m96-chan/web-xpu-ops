// ISTFT: inverse one-sided DFT per frame, windowed, overlap-added, and divided
// by the overlap-added w² envelope. The thing ONNX cannot express.
//
// Layout:
//   real, imag: [frames, bins] f32, frame-major
//   window:     [nFft] f32
//   out:        [outLength] f32
//
// One thread per **output sample**, gathering rather than scattering. Written
// this way for a reason: the natural overlap-add scatters frames into a shared
// buffer, where two frames land on the same sample and the sum needs atomics or
// a second pass. Reading instead of writing, each output sample is computed by
// exactly one thread and there is nothing to order. It costs re-deriving the
// inverse transform once per overlapping frame — two of them, at 2x overlap.
//
// The envelope division is not optional and not an ordinary normalisation: see
// reference.ts. A periodic Hann at 50% overlap is COLA in w but not in w², so
// skipping it is wrong by up to 2x in a way that still sounds like audio.

struct Params {
  nFft: u32,
  hop: u32,
  bins: u32,
  frames: u32,
  outLength: u32,
  /// floor(nFft/2) when centred, 0 when not. The caller resolves the convention.
  pad: u32,
}

@group(0) @binding(0) var<storage, read> real: array<f32>;
@group(0) @binding(1) var<storage, read> imag: array<f32>;
@group(0) @binding(2) var<storage, read> win: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

const TWO_PI: f32 = 6.28318530717958647692;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let t = global_id.x;
  if (t >= params.outLength) {
    return;
  }
  let position = i32(t + params.pad);
  // Even nFft has a Nyquist bin, which stands for itself rather than for a
  // conjugate pair and so is counted once. Odd nFft has none, and every bin
  // above DC doubles. torch.fft.irfft splits the same way.
  let hasNyquist = (params.nFft % 2u) == 0u;

  var numerator: f32 = 0.0;
  var envelope: f32 = 0.0;
  for (var frame = 0u; frame < params.frames; frame += 1u) {
    let n = position - i32(frame * params.hop);
    // The `n < 0` half of this cannot be caught by a test on this device, and
    // is written down rather than left to be discovered. Dropping it makes
    // `u32(n)` wrap to about 4e9; this GPU reads that far past a buffer as
    // zero, the window value comes back 0, and the frame contributes nothing —
    // which is accidentally the right answer. WGSL allows an implementation to
    // clamp the index instead, and a device that clamps would read a real
    // window value and add a whole frame that does not belong to this sample.
    if (n < 0 || n >= i32(params.nFft)) {
      continue;
    }
    let base = frame * params.bins;
    // DC counts once, and its imaginary part is dropped along with Nyquist's.
    var acc: f32 = real[base];
    for (var k = 1u; k < params.bins; k += 1u) {
      let weight = select(2.0, 1.0, hasNyquist && k == params.bins - 1u);
      // Folded into one turn as an integer, as in the forward kernel.
      let angle = TWO_PI * (f32((k * u32(n)) % params.nFft) / f32(params.nFft));
      acc += weight * (real[base + k] * cos(angle) - imag[base + k] * sin(angle));
    }
    let w = win[u32(n)];
    numerator += w * (acc / f32(params.nFft));
    envelope += w * w;
  }
  // No NOLA guard here. A shader cannot raise, and a guard that silently
  // substituted a value would hand back a waveform for a window that cannot
  // reconstruct one. The reference refuses those inputs before they get here.
  out[t] = numerator / envelope;
}
