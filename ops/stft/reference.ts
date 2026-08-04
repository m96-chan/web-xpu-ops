/**
 * Short-time Fourier transform and its inverse.
 *
 * The definition of correct for both ops. Deliberately the slowest, plainest
 * expression of the maths — a naive DFT per frame, no FFT, no factorisation —
 * because its job is to be obviously right rather than fast.
 *
 * ## Conventions, and what they match
 *
 * Every one of these has more than one defensible answer, and picking silently
 * means half the callers get a subtly wrong waveform. All of them follow
 * **`torch.stft` / `torch.istft` (PyTorch 2.10)**, and each was checked against
 * it numerically rather than read off the documentation:
 *
 * | thing | here | matches |
 * | --- | --- | --- |
 * | centring | `center = true` by default: frame `f` is centred on sample `f * hop` | `torch.stft(center=True)` |
 * | padding | reflect, without repeating the edge sample | `pad_mode="reflect"` |
 * | frame count | `1 + floor(L / hop)` centred, `1 + floor((L - nFft) / hop)` not | torch |
 * | sidedness | one-sided, `floor(nFft / 2) + 1` bins; the input is real | `onesided=True` |
 * | scaling | none on the forward transform, `1 / nFft` on the inverse | `normalized=False` |
 * | window | required here; torch defaults to a rectangular window and warns | — |
 * | `hannWindow` | periodic by default | `torch.hann_window(periodic=True)`, `scipy.signal.get_window("hann")`, librosa — **not** `np.hanning`, which is symmetric |
 * | overlap-add | divided by the overlap-added `w²` envelope | `torch.istft` |
 * | Nyquist bin | counted once and its imaginary part dropped, for even `nFft` | `torch.fft.irfft` |
 *
 * Two departures, both deliberate:
 *
 * - **Layout is frame-major, `[frames, bins]`.** torch returns `[bins, frames]`.
 *   A vocoder head emits one row per frame — MioCodec's `istft_head` is a
 *   `Linear[.., 1922]` where `1922 = 2 * (1920 / 2 + 1)` — so frame-major is
 *   what arrives, and it is also what lets a kernel write consecutively.
 * - **`length` beyond what the frames cover raises.** torch pads the tail with
 *   zeros and warns. A silent zero tail is indistinguishable from silence in a
 *   vocoder output, and neither a warning nor a raise survives into a shader,
 *   so the reference refuses instead.
 *
 * ## COLA / NOLA
 *
 * `istft` does not assume the window is COLA (constant overlap-add). It divides
 * by the overlap-added `w²` envelope, which inverts any window satisfying NOLA
 * (that envelope nowhere zero) — the same least-squares inverse `torch.istft`
 * computes. This matters for the common case: a periodic Hann at 50% overlap is
 * COLA in `w` but **not** in `w²` — `sin⁴θ + cos⁴θ` runs between 0.5 and 1 — so
 * a reconstruction that skipped the division would be wrong by up to 2x, and
 * would still look plausible. The envelope threshold below which this refuses,
 * `1e-11`, is torch's, bracketed by bisection against `torch.istft`.
 */

/** `[frames, bins]`, frame-major. `imag` is negated-exponent convention, as torch. */
export interface Spectrogram {
  real: Float32Array;
  imag: Float32Array;
  frames: number;
  bins: number;
}

export interface StftArgs {
  /** Real signal. */
  input: ArrayLike<number>;
  nFft: number;
  /** Samples between frame starts. */
  hop: number;
  /** Analysis window, exactly `nFft` long. */
  window: ArrayLike<number>;
  /** Pad by `floor(nFft / 2)` either side so frame `f` centres on sample `f * hop`. Default true. */
  center?: boolean;
}

/**
 * How the synthesised signal is trimmed at its edges.
 *
 * - `"center"` — drop `floor(nFft / 2)` from each end, undoing `stft`'s centring
 *   pad. `torch.istft(center=True)`.
 * - `"none"` — return everything the frames cover. `torch.istft(center=False)`.
 * - `"same"` — drop `(nFft - hop) / 2` from each end, so `T` frames give exactly
 *   `T * hop` samples. **Not a torch mode**, and it cannot be built from one:
 *   see `istft` for why `"none"` plus a slice fails before it can be sliced.
 *   It is the Vocos / X-Codec-2 / MioCodec vocoder convention, chosen because a
 *   vocoder wants as many samples out as frames times hop.
 */
export type IstftPadding = "center" | "none" | "same";

export interface IstftArgs {
  real: ArrayLike<number>;
  imag: ArrayLike<number>;
  frames: number;
  nFft: number;
  hop: number;
  window: ArrayLike<number>;
  /** Kept for callers written before `padding` existed; `true` is `"center"`. */
  center?: boolean;
  /** Edge trimming. Defaults to `"center"`. Giving both this and `center` raises. */
  padding?: IstftPadding;
  /** Output samples. Defaults to `istftLength(...)`; may not exceed `istftMaxLength(...)`. */
  length?: number;
}

/** One-sided bin count: DC, the positive frequencies, and Nyquist when `nFft` is even. */
export function stftBins(nFft: number): number {
  return Math.floor(nFft / 2) + 1;
}

/** Frames `stft` will produce for a signal of `length` samples. */
export function stftFrames(nFft: number, hop: number, length: number, center = true): number {
  return center ? 1 + Math.floor(length / hop) : 1 + Math.floor((length - nFft) / hop);
}

/**
 * Samples `istft` returns when `length` is not given — torch's default.
 *
 * Centred, this is `hop * (frames - 1)`: the trailing half-window is dropped
 * because it came from padding. See `istftMaxLength` for what is recoverable.
 */
export function istftLength(
  nFft: number,
  hop: number,
  frames: number,
  mode: IstftPadding | boolean = true,
): number {
  return nFft + hop * (frames - 1) - 2 * padding(nFft, hop, mode);
}

/**
 * The last sample any frame reaches — the most `length` may ask for.
 *
 * Larger than `istftLength` by the leading pad when centred, and asking for it
 * is how a signal whose length is not a multiple of `hop` gets its tail back.
 */
export function istftMaxLength(
  nFft: number,
  hop: number,
  frames: number,
  mode: IstftPadding | boolean = true,
): number {
  const trim = padding(nFft, hop, mode);
  // `"same"` crops both ends, so nothing is left over to ask back for: every
  // frame's contribution is either kept or cropped. The other two crop only the
  // head, and the extra reach is how a signal whose length is not a multiple of
  // hop gets its tail back.
  return resolve(mode) === "same"
    ? nFft + hop * (frames - 1) - 2 * trim
    : nFft + hop * (frames - 1) - trim;
}

function resolve(mode: IstftPadding | boolean): IstftPadding {
  if (mode === true) return "center";
  if (mode === false) return "none";
  return mode;
}

/**
 * Samples dropped from the head.
 *
 * `"same"` uses `(nFft - hop) / 2`, which is upstream's `(win_length -
 * hop_length) // 2` — the window here is always `nFft` long. At MioCodec's
 * `nFft = 1920, hop = 480` that is 720, where `"center"` would take 960 and
 * `"none"` none.
 */
function padding(nFft: number, hop: number, mode: IstftPadding | boolean): number {
  const resolved = resolve(mode);
  if (resolved === "none") return 0;
  if (resolved === "same") return Math.floor((nFft - hop) / 2);
  return Math.floor(nFft / 2);
}

/**
 * Periodic Hann by default: `0.5 - 0.5 cos(2πi/n)`.
 *
 * Periodic is the STFT convention — it is what `torch.hann_window`,
 * `scipy.signal.get_window("hann", n)` and librosa give. `np.hanning` is the
 * symmetric one, `0.5 - 0.5 cos(2πi/(n-1))`, which is for filter design and
 * does not overlap-add cleanly. The difference is small enough to look like a
 * rounding error and large enough to hear.
 */
export function hannWindow(n: number, periodic = true): Float32Array {
  const denominator = periodic ? n : n - 1;
  return Float32Array.from({ length: n }, (_, i) => 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / denominator));
}

export function stft({ input, nFft, hop, window, center = true }: StftArgs): Spectrogram {
  if (nFft < 1) throw new Error(`nFft must be positive, got ${nFft}`);
  if (hop < 1) throw new Error(`hop must be positive, got ${hop}`);
  if (window.length !== nFft) throw new Error(`window must be ${nFft} long, got ${window.length}`);
  const signalLength = input.length;
  // The forward has only torch's two modes; `"same"` is a synthesis-side
  // convention and has no analysis counterpart to invert.
  const pad = padding(nFft, hop, center);
  // torch raises the same way: reflection may not reach past the far edge.
  if (center && pad >= signalLength) {
    throw new Error(`reflect padding of ${pad} needs more than ${signalLength} samples`);
  }
  if (!center && signalLength < nFft) {
    throw new Error(`an uncentred transform needs at least nFft=${nFft} samples, got ${signalLength}`);
  }

  const bins = stftBins(nFft);
  const frames = stftFrames(nFft, hop, signalLength, center);
  const real = new Float32Array(frames * bins);
  const imag = new Float32Array(frames * bins);

  for (let frame = 0; frame < frames; frame += 1) {
    for (let k = 0; k < bins; k += 1) {
      let sumReal = 0;
      let sumImag = 0;
      for (let n = 0; n < nFft; n += 1) {
        const value = reflected(input, frame * hop + n - pad) * window[n]!;
        // Folded into one turn before it becomes an angle. k*n reaches 1.8M at
        // nFft=1920, and that many radians has no precision left in f32 — the
        // kernel has to do this, so the reference does it the same way.
        const angle = (-2 * Math.PI * ((k * n) % nFft)) / nFft;
        sumReal += value * Math.cos(angle);
        sumImag += value * Math.sin(angle);
      }
      real[frame * bins + k] = sumReal;
      imag[frame * bins + k] = sumImag;
    }
  }
  return { real, imag, frames, bins };
}

/** Reflect padding: `[-1] -> [1]`, `[L] -> [L-2]`. The edge sample is not repeated. */
function reflected(input: ArrayLike<number>, index: number): number {
  const last = input.length - 1;
  let i = index;
  if (i < 0) i = -i;
  if (i > last) i = 2 * last - i;
  return input[i]!;
}

export function istft({
  real,
  imag,
  frames,
  nFft,
  hop,
  window,
  center,
  padding: mode,
  length,
}: IstftArgs): Float32Array {
  if (center !== undefined && mode !== undefined) {
    // Ranking them would make one of two explicit requests silently lose. torch
    // takes the same line with `is_causal` and `attn_mask`.
    throw new Error("give either center or padding, not both");
  }
  const resolved = resolve(mode ?? center ?? true);
  if (nFft < 1) throw new Error(`nFft must be positive, got ${nFft}`);
  if (hop < 1) throw new Error(`hop must be positive, got ${hop}`);
  if (window.length !== nFft) throw new Error(`window must be ${nFft} long, got ${window.length}`);
  const bins = stftBins(nFft);
  for (const [name, side] of [["real", real], ["imag", imag]] as const) {
    if (side.length !== frames * bins) {
      throw new Error(`${name} must be ${frames} x ${bins} = ${frames * bins}, got ${side.length}`);
    }
  }
  const samples = length ?? istftLength(nFft, hop, frames, resolved);
  const reach = istftMaxLength(nFft, hop, frames, resolved);
  if (samples > reach) {
    // torch zero-pads here and warns. See the header for why this refuses.
    throw new Error(`${frames} frames reach ${reach} samples, cannot produce ${samples}`);
  }

  const pad = padding(nFft, hop, resolved);
  const even = nFft % 2 === 0;
  const output = new Float32Array(samples);

  for (let t = 0; t < samples; t += 1) {
    const position = t + pad;
    let numerator = 0;
    let envelope = 0;
    for (let frame = 0; frame < frames; frame += 1) {
      const n = position - frame * hop;
      if (n < 0 || n >= nFft) continue;
      // Inverse one-sided DFT of this frame at offset n. Every bin between DC
      // and Nyquist stands for a conjugate pair and so counts twice; DC counts
      // once, and so does Nyquist when it exists, its imaginary part dropped —
      // which is what torch.fft.irfft does with whatever is in those slots.
      const base = frame * bins;
      let acc = real[base]!;
      for (let k = 1; k < bins; k += 1) {
        const weight = even && k === bins - 1 ? 1 : 2;
        const angle = (2 * Math.PI * ((k * n) % nFft)) / nFft;
        acc += weight * (real[base + k]! * Math.cos(angle) - imag[base + k]! * Math.sin(angle));
      }
      const w = window[n]!;
      numerator += w * (acc / nFft);
      envelope += w * w;
    }
    // The floor is checked over the samples actually returned, which is the
    // only place it can be. This loop never visits a cropped sample, so nothing
    // is relaxed for `"same"` — the check simply lives where the answer is.
    // That distinction matters: upstream folds the whole signal and asserts only
    // over the retained region for exactly this reason, and a floor applied to
    // cropped samples would make `"same"` throw for the very reason it exists.
    // The edge samples `"none"` returns really are unreconstructable; the ones
    // `"same"` crops are simply not asked for.
    if (envelope < NOLA_FLOOR) {
      throw new Error(
        `window fails NOLA at sample ${t}: the w² envelope is ${envelope}, below ${NOLA_FLOOR}`,
      );
    }
    output[t] = numerator / envelope;
  }
  return output;
}

/** torch's own threshold, bracketed by bisection against `torch.istft` 2.10. */
const NOLA_FLOOR = 1e-11;
