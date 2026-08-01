/**
 * Mel filterbank: its construction, and its application to an STFT.
 *
 * The definition of correct for the op. Deliberately the slowest, plainest
 * expression of it — the filterbank is built one triangle at a time and applied
 * as a dense matrix, with no sparsity and no fusion — because its job is to be
 * obviously right rather than fast.
 *
 * ## Conventions, and what they match
 *
 * Mel is the op where "reasonable" is not one thing. Every line below has at
 * least two defensible answers in wide use, and they produce different numbers
 * from the same audio. The defaults here are **`torchaudio.transforms.MelSpectrogram`
 * (torchaudio 2.10)**; the other convention in each pair is reachable by an
 * argument, and the combination `{ scale: "slaney", norm: "slaney" }` is
 * **`librosa.filters.mel` / `librosa.feature.melspectrogram`** defaults.
 * Every one was checked against the library numerically rather than read off
 * the documentation — see `reference.test.ts`.
 *
 * | thing | default here | matches | the other answer |
 * | --- | --- | --- | --- |
 * | mel scale | `scale: "htk"`, `2595 log10(1 + f/700)` | `torchaudio(mel_scale="htk")`, `librosa(htk=True)` | `"slaney"`: linear below 1 kHz, log above — `librosa`'s default |
 * | filter normalisation | `norm: "none"`, triangles peak at 1 | `torchaudio(norm=None)` | `"slaney"`: scaled by `2/(fRight-fLeft)`, roughly constant energy per band — `librosa`'s default |
 * | spectrum | `power: 2`, `re² + im²` | `torchaudio(power=2.0)`, `librosa(power=2.0)` | `power: 1`, magnitude `sqrt(re² + im²)` |
 * | log | `db: false` — none | `MelSpectrogram` alone | `db: true`, below |
 * | log base | base 10, scaled to decibels by `20/power` (10 for power, 20 for magnitude) | `torchaudio.transforms.AmplitudeToDB(stype="power" \| "magnitude")` | natural log, as many TTS front ends |
 * | log floor | clamp the **argument** at `amin = 1e-10` | `amplitude_to_DB`, `librosa.power_to_db` | `log(x + eps)`, which shifts every value, not just the small ones |
 * | band edges | `nMels + 2` points evenly spaced **in mel** between `fMin` and `fMax` | both | — |
 * | `fMax` default | `sampleRate / 2` | `torchaudio(f_max=None)` | — |
 *
 * Two things torchaudio does that this does not:
 *
 * - **`top_db` is not here.** `amplitude_to_DB(top_db=...)` clamps at
 *   `max(x) - top_db`, which needs a reduction over the whole spectrogram
 *   before any element can be written. That is `reduce` followed by
 *   `elementwise`, not a mel kernel, and folding it in would make the op's
 *   output depend on how the caller chunked their audio — which torchaudio
 *   itself warns about. `top_db=None` is what the numbers here match, and it is
 *   also `librosa.power_to_db(top_db=None)`; librosa's *default* is `top_db=80`.
 * - **Bin frequencies are `k * sampleRate / nFft`,** the actual DFT bin centre
 *   and what `librosa.fft_frequencies` returns. torchaudio uses
 *   `linspace(0, sample_rate // 2, bins)`, which is the same number whenever
 *   `nFft` and `sampleRate` are both even — every real configuration — and
 *   diverges otherwise: at `nFft = 7` torchaudio puts the top bin at
 *   `sampleRate / 2` where the transform actually resolves `3 * sampleRate / 7`.
 *   Following the transform matters more here than following the rounding.
 *
 * ## Layout
 *
 * The filterbank is `[nMels, bins]` row-major and the spectrogram is
 * `[frames, nMels]`, both frame- and filter-major, following `ops/stft`'s
 * `[frames, bins]`. torchaudio is transposed from this in both.
 */

/** `htk`: `2595 log10(1 + f/700)`. `slaney`: linear to 1 kHz, log above. */
export type MelScale = "htk" | "slaney";

/** `slaney` divides each triangle by its width in Hz; `none` leaves it peaking at 1. */
export type MelNorm = "none" | "slaney";

/** Power spectrum (`re² + im²`) or magnitude (`sqrt` of it). */
export type MelPower = 1 | 2;

export interface MelFrequenciesArgs {
  /** Number of mel bands. The result is `nMels + 2` long: every band's three corners. */
  nMels: number;
  /** Low edge of the lowest band, Hz. Default 0. */
  fMin?: number;
  /** High edge of the highest band, Hz. */
  fMax: number;
  scale?: MelScale;
}

export interface MelFilterbankArgs {
  nMels: number;
  nFft: number;
  sampleRate: number;
  /** Default 0. */
  fMin?: number;
  /** Default `sampleRate / 2`. */
  fMax?: number;
  /** Default `"htk"`, as torchaudio. */
  scale?: MelScale;
  /** Default `"none"`, as torchaudio. */
  norm?: MelNorm;
}

export interface MelSpectrogramArgs {
  /** `[frames, bins]` real part, as `ops/stft` returns. */
  real: ArrayLike<number>;
  imag: ArrayLike<number>;
  frames: number;
  bins: number;
  /** `[nMels, bins]` row-major, from `melFilterbank`. */
  filterbank: ArrayLike<number>;
  nMels: number;
  /** Default 2 (power), as torchaudio. */
  power?: MelPower;
  /** Convert to decibels. Default false. */
  db?: boolean;
  /** Floor the dB argument here. Default 1e-10, as `amplitude_to_DB`. Ignored unless `db`. */
  amin?: number;
}

/** One-sided bin count, the same count `ops/stft` produces. */
export function melBins(nFft: number): number {
  return Math.floor(nFft / 2) + 1;
}

/**
 * Hz to mel.
 *
 * `htk` is the closed form from the HTK book, which torchaudio defaults to.
 * `slaney` is the Auditory Toolbox's: linear at 200/3 Hz per mel up to 1 kHz,
 * then logarithmic, which librosa defaults to. They are not a reparametrisation
 * of one another — at 4 kHz one says 2146 and the other 35.2, and the band
 * edges they place differ by hundreds of Hz.
 */
export function hzToMel(hz: number, scale: MelScale = "htk"): number {
  if (scale === "htk") return 2595 * Math.log10(1 + hz / 700);
  const mels = hz / SLANEY_F_SP;
  if (hz < SLANEY_MIN_LOG_HZ) return mels;
  return SLANEY_MIN_LOG_MEL + Math.log(hz / SLANEY_MIN_LOG_HZ) / SLANEY_LOGSTEP;
}

/** Inverse of `hzToMel` on the same scale. */
export function melToHz(mel: number, scale: MelScale = "htk"): number {
  if (scale === "htk") return 700 * (10 ** (mel / 2595) - 1);
  if (mel < SLANEY_MIN_LOG_MEL) return SLANEY_F_SP * mel;
  return SLANEY_MIN_LOG_HZ * Math.exp(SLANEY_LOGSTEP * (mel - SLANEY_MIN_LOG_MEL));
}

// Slaney's constants, as torchaudio's `_hz_to_mel` and librosa's `hz_to_mel`
// spell them: 200/3 Hz per mel below the 1 kHz breakpoint, and above it a
// factor of 6.4 in frequency every 27 mels.
const SLANEY_F_SP = 200 / 3;
const SLANEY_MIN_LOG_HZ = 1000;
const SLANEY_MIN_LOG_MEL = SLANEY_MIN_LOG_HZ / SLANEY_F_SP; // exactly 15
const SLANEY_LOGSTEP = Math.log(6.4) / 27;

/**
 * The `nMels + 2` band edges, evenly spaced in mel between `fMin` and `fMax`.
 *
 * Band `m` runs from `edges[m]` up through `edges[m + 1]` to `edges[m + 2]`, so
 * neighbouring bands overlap by half. Equal to
 * `librosa.mel_frequencies(n_mels=nMels + 2, fmin=fMin, fmax=fMax, htk=...)`.
 */
export function melFrequencies({ nMels, fMin = 0, fMax, scale = "htk" }: MelFrequenciesArgs): Float64Array {
  if (nMels < 1) throw new Error(`nMels must be positive, got ${nMels}`);
  if (!(fMax > fMin)) throw new Error(`fMax must exceed fMin, got ${fMin}..${fMax}`);
  const melMin = hzToMel(fMin, scale);
  const melMax = hzToMel(fMax, scale);
  const step = (melMax - melMin) / (nMels + 1);
  return Float64Array.from({ length: nMels + 2 }, (_, j) => melToHz(melMin + j * step, scale));
}

/**
 * The `[nMels, bins]` matrix, row-major.
 *
 * Row `m` is a triangle rising from `edges[m]` to 1 at `edges[m + 1]` and back
 * to 0 at `edges[m + 2]`, sampled at each DFT bin centre. Written as the two
 * slopes and a `min`, which is how torchaudio and librosa both write it — the
 * arithmetic is the same as clipping the triangle, but the rounding is not, and
 * matching them is the point.
 *
 * A row can come out all zero when `nMels` is large enough that a band falls
 * between two bin centres. torchaudio warns; this does not, because the caller
 * that would act on the warning is the one choosing `nMels`, and a shader
 * cannot warn. `melFilterbankHasEmptyBand` is here for a caller that wants to
 * ask.
 */
export function melFilterbank({
  nMels,
  nFft,
  sampleRate,
  fMin = 0,
  fMax = sampleRate / 2,
  scale = "htk",
  norm = "none",
}: MelFilterbankArgs): Float32Array {
  if (nFft < 1) throw new Error(`nFft must be positive, got ${nFft}`);
  if (sampleRate <= 0) throw new Error(`sampleRate must be positive, got ${sampleRate}`);
  const bins = melBins(nFft);
  const edges = melFrequencies({ nMels, fMin, fMax, scale });
  const weights = new Float32Array(nMels * bins);

  for (let m = 0; m < nMels; m += 1) {
    const left = edges[m]!;
    const centre = edges[m + 1]!;
    const right = edges[m + 2]!;
    // Slaney normalisation makes the triangle's area rather than its peak the
    // constant, so a wide high band does not swamp a narrow low one.
    const enorm = norm === "slaney" ? 2 / (right - left) : 1;
    for (let k = 0; k < bins; k += 1) {
      const frequency = (k * sampleRate) / nFft;
      const rising = (frequency - left) / (centre - left);
      const falling = (right - frequency) / (right - centre);
      weights[m * bins + k] = Math.max(0, Math.min(rising, falling)) * enorm;
    }
  }
  return weights;
}

/** True when some band caught no bin at all — `nMels` too high for `nFft`. */
export function melFilterbankHasEmptyBand(filterbank: ArrayLike<number>, nMels: number, bins: number): boolean {
  for (let m = 0; m < nMels; m += 1) {
    let peak = 0;
    for (let k = 0; k < bins; k += 1) peak = Math.max(peak, filterbank[m * bins + k]!);
    if (peak === 0) return true;
  }
  return false;
}

/**
 * `[frames, nMels]` from a `[frames, bins]` complex spectrogram.
 *
 * Spectrum first, then the matrix, then the log — in that order, which is what
 * makes `power` more than a scale factor: the filterbank sums `|X|²` for
 * `power: 2` and `|X|` for `power: 1`, and a sum of squares is not the square
 * of a sum. `MelSpectrogram(power=1)` and `MelSpectrogram(power=2)` are two
 * different features, not one feature in two units.
 */
export function melSpectrogram({
  real,
  imag,
  frames,
  bins,
  filterbank,
  nMels,
  power = 2,
  db = false,
  amin = 1e-10,
}: MelSpectrogramArgs): Float32Array {
  for (const [name, side] of [["real", real], ["imag", imag]] as const) {
    if (side.length !== frames * bins) {
      throw new Error(`${name} must be ${frames} x ${bins} = ${frames * bins}, got ${side.length}`);
    }
  }
  if (filterbank.length !== nMels * bins) {
    throw new Error(`filterbank must be ${nMels} x ${bins} = ${nMels * bins}, got ${filterbank.length}`);
  }
  if (power !== 1 && power !== 2) throw new Error(`power must be 1 or 2, got ${power}`);

  // 10 for a power spectrum, 20 for a magnitude one: the same decibel for the
  // same underlying signal. `torchaudio.transforms.AmplitudeToDB` picks between
  // them with `stype`, which callers pair with `power` by hand.
  const multiplier = 20 / power;
  const out = new Float32Array(frames * nMels);

  for (let frame = 0; frame < frames; frame += 1) {
    for (let m = 0; m < nMels; m += 1) {
      let acc = 0;
      for (let k = 0; k < bins; k += 1) {
        const re = real[frame * bins + k]!;
        const im = imag[frame * bins + k]!;
        const energy = re * re + im * im;
        acc += filterbank[m * bins + k]! * (power === 2 ? energy : Math.sqrt(energy));
      }
      // The floor clamps the argument, not the result: everything above `amin`
      // is untouched, where `log(x + eps)` would shift every value down. A
      // silent band lands on `multiplier * log10(amin)` exactly — -100 dB for
      // power, -200 dB for magnitude — which is what torchaudio returns.
      out[frame * nMels + m] = db ? multiplier * Math.log10(Math.max(acc, amin)) : acc;
    }
  }
  return out;
}
