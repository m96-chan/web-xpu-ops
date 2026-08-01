// Mel filterbank construction: the [nMels, bins] triangle matrix.
//
// Separate from the kernel that applies it, and dispatched separately, because
// the matrix depends only on scalars — sample rate, nFft, band count, the two
// convention flags — so it is built once and reused for every frame of every
// utterance at that configuration. Fusing it into the application would rebuild
// 77k triangles per dispatch to save one buffer.
//
// The conventions are `reference.ts`'s and are named there: the default is
// `torchaudio.transforms.MelSpectrogram`, and scale=1 with norm=1 is librosa's.
// Nothing here decides anything; it computes what that file says.
//
// Layout:
//   outBank: [nMels, bins] f32, row-major. Row m is a triangle rising from
//            edge[m] to 1 at edge[m+1] and back to 0 at edge[m+2].
//
// One thread per entry, flat. The last workgroup runs past the end of the
// matrix, so the guard below is load-bearing — the test allocates the output at
// the dispatch length rather than the matrix length so that removing it writes
// somewhere the assertions can see.

struct Params {
  nMels: u32,
  bins: u32,
  nFft: u32,
  sampleRate: f32,
  fMin: f32,
  fMax: f32,
  // 0 = htk, 1 = slaney.
  scale: u32,
  // 0 = none, 1 = slaney.
  norm: u32,
}

@group(0) @binding(0) var<storage, read_write> outBank: array<f32>;
@group(0) @binding(1) var<uniform> params: Params;

const HTK: u32 = 0u;
const SLANEY_NORM: u32 = 1u;

// WGSL has a natural log and a log2 but no log10; the reference is written in
// log10, so this constant is what keeps the two the same expression.
const INV_LN10: f32 = 0.4342944819032518;

// Slaney's constants: 200/3 Hz per mel below the 1 kHz breakpoint, and above it
// a factor of 6.4 in frequency every 27 mels. `logstep` is log(6.4)/27
// evaluated in f64 and written out rather than computed here, so the shader and
// the reference start from the same number.
const SLANEY_F_SP: f32 = 66.66666666666667;
const SLANEY_MIN_LOG_HZ: f32 = 1000.0;
const SLANEY_MIN_LOG_MEL: f32 = 15.0;
const SLANEY_LOGSTEP: f32 = 0.06875177742094912;

fn hzToMel(hz: f32) -> f32 {
  if (params.scale == HTK) {
    return 2595.0 * log(1.0 + hz / 700.0) * INV_LN10;
  }
  if (hz < SLANEY_MIN_LOG_HZ) {
    return hz / SLANEY_F_SP;
  }
  return SLANEY_MIN_LOG_MEL + log(hz / SLANEY_MIN_LOG_HZ) / SLANEY_LOGSTEP;
}

fn melToHz(mel: f32) -> f32 {
  if (params.scale == HTK) {
    return 700.0 * (pow(10.0, mel / 2595.0) - 1.0);
  }
  if (mel < SLANEY_MIN_LOG_MEL) {
    return SLANEY_F_SP * mel;
  }
  return SLANEY_MIN_LOG_HZ * exp(SLANEY_LOGSTEP * (mel - SLANEY_MIN_LOG_MEL));
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let index = gid.x;
  if (index >= params.nMels * params.bins) { return; }
  let m = index / params.bins;
  let k = index % params.bins;

  // The DFT bin this weight multiplies. k * sampleRate / nFft rather than a
  // linspace up to sampleRate/2 — see reference.ts for why, and for when the
  // two differ (they do not for any even nFft).
  let frequency = f32(k) * params.sampleRate / f32(params.nFft);

  // The band's three corners, evenly spaced in mel. Recomputed per thread
  // rather than staged in workgroup memory: it is three transcendentals against
  // a buffer this kernel would otherwise not need, and the whole dispatch runs
  // once per configuration.
  let melMin = hzToMel(params.fMin);
  let step = (hzToMel(params.fMax) - melMin) / f32(params.nMels + 1u);
  let left = melToHz(melMin + f32(m) * step);
  let centre = melToHz(melMin + f32(m + 1u) * step);
  let right = melToHz(melMin + f32(m + 2u) * step);

  // Two slopes and a min, which is how torchaudio and librosa both write it.
  // Clipping the triangle would be the same arithmetic and not the same
  // rounding, and matching them is the point.
  let rising = (frequency - left) / (centre - left);
  let falling = (right - frequency) / (right - centre);
  var weight = max(0.0, min(rising, falling));

  // Slaney normalisation makes the triangle's area rather than its peak the
  // constant, so a wide high band does not swamp a narrow low one.
  if (params.norm == SLANEY_NORM) { weight *= 2.0 / (right - left); }

  outBank[index] = weight;
}
