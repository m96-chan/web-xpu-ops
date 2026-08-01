// Mel spectrogram: a filterbank applied to an STFT, optionally in decibels.
//
// Spectrum, then the matrix, then the log — in that order, which is what makes
// `power` more than a scale factor: the filterbank sums |X|² for power=2 and
// |X| for power=1, and a sum of squares is not the square of a sum. See
// reference.ts for that and for every other convention here.
//
// The log is fused rather than left to a following elementwise dispatch because
// log-mel is the feature, not mel; splitting it would write [frames, nMels]
// twice to save a branch that resolves the same way for every lane in the
// dispatch. What is *not* fused is `top_db`, which torchaudio offers: it clamps
// against the maximum over the whole spectrogram, so it cannot be computed
// before the last frame exists. That is `reduce` and then `elementwise`.
//
// Layout:
//   inReal, inImag: [frames, bins] f32, frame-major, as ops/stft emits
//   bank:           [nMels, bins] f32, row-major, from filterbank.wgsl
//   outMel:         [frames, nMels] f32, frame-major
//
// One workgroup per frame; the 256 lanes walk the mel bands at a stride, and
// each lane walks every bin. Dispatched with exactly `frames` workgroups, so
// the frame index needs no bound check — there is no frame to guard against.

struct Params {
  bins: u32,
  nMels: u32,
  // 1 = magnitude spectrum, 2 = power spectrum.
  power: u32,
  // 0 = linear, 1 = decibels.
  db: u32,
  // Floor for the dB argument. 1e-10, as torchaudio's amplitude_to_DB.
  amin: f32,
}

@group(0) @binding(0) var<storage, read> inReal: array<f32>;
@group(0) @binding(1) var<storage, read> inImag: array<f32>;
@group(0) @binding(2) var<storage, read> bank: array<f32>;
@group(0) @binding(3) var<storage, read_write> outMel: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

const WORKGROUP_SIZE: u32 = 256u;
const POWER: u32 = 2u;
const INV_LN10: f32 = 0.4342944819032518;

@compute @workgroup_size(256)
fn main(
  @builtin(workgroup_id) wg_id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>,
) {
  let frame = wg_id.x;
  let base = frame * params.bins;

  for (var m = local_id.x; m < params.nMels; m += WORKGROUP_SIZE) {
    var acc: f32 = 0.0;
    for (var k = 0u; k < params.bins; k += 1u) {
      let re = inReal[base + k];
      let im = inImag[base + k];
      let energy = re * re + im * im;
      // The spectrum, not the spectrogram: |X|² or |X|, chosen before the
      // filterbank sees it.
      let spectrum = select(sqrt(energy), energy, params.power == POWER);
      acc += bank[m * params.bins + k] * spectrum;
    }

    var value = acc;
    if (params.db != 0u) {
      // 10 for a power spectrum, 20 for a magnitude one, which is the same
      // decibel for the same underlying signal — torchaudio spells the choice
      // `stype`. The floor clamps the argument rather than being added to it,
      // so everything above amin comes through untouched and a silent band
      // lands exactly on multiplier * log10(amin).
      let multiplier = 20.0 / f32(params.power);
      value = multiplier * log(max(acc, params.amin)) * INV_LN10;
    }
    outMel[frame * params.nMels + m] = value;
  }
}
