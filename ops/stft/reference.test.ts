import { describe, expect, it } from "vitest";
import { agree } from "../../harness/index.js";
import { hannWindow, istft, istftLength, istftMaxLength, stft, stftFrames } from "./reference.js";

/**
 * The conventions, pinned to the library they claim to match.
 *
 * Everything here was produced by **torch 2.10.0+cu128** in float64 and pasted
 * in. It is the only test in this op that can catch a convention being wrong
 * rather than an arithmetic slip: the WGSL tests compare a kernel against this
 * reference, so if the reference picked symmetric Hann, or dropped the `w²`
 * envelope, or transposed the output, they would agree with each other all the
 * way to a wrong waveform.
 *
 * Touches no GPU.
 *
 * Tolerance is 1e-6: the reference stores its result as f32 and torch's is f64,
 * which is the whole of the difference here (values reach 8.5, so an f32 ulp is
 * ~5e-7). Every convention this is guarding against is wrong by 1e-2 or more.
 */
const TOLERANCE = { rel: 1e-6, abs: 1e-6 };

const N_FFT = 8;
const HOP = 4;
const SIGNAL = Array.from({ length: 16 }, (_, i) => Math.sin(0.37 * i) * 2 + 0.1 * i);

// torch.hann_window(8) — periodic. Note w[0] === 0 and no repeat at the end;
// the symmetric window numpy gives is zero at *both* ends.
const HANN8 = [
  0.0, 0.1464466094067262, 0.49999999999999994, 0.8535533905932737,
  1.0, 0.8535533905932738, 0.5000000000000001, 0.14644660940672632,
];

// torch.stft(x, n_fft=8, hop_length=4, window=hann, center=True,
//            return_complex=True), transposed to [frames, bins].
const STFT_REAL = [
  3.5664749214687754, -0.5605849805674752, -1.5485758232562903, 0.5605849805674752,
  -0.46932327495619486, 8.492071532628007, -4.849901317609982, 0.5209083052110484,
  0.0663779394594215, 0.03315861325101821, 4.449830643575211, -2.3344222627756905,
  0.09446320446087286, 0.012037191198758901, 0.006013090656906694, -1.8654231831115304,
  1.5167187983927297, -0.5037780408099046, -0.06419507609946429, -0.03206817985519117,
  -1.5671060337841447, 0.5811342782873913, 0.17048325114508178, 0.18945643942554702,
  -0.31504190393189546,
];
const STFT_IMAG = [
  0.0, 5.551115123125783e-16, 0.0, -1.1102230246251565e-16,
  0.0, 0.0, 0.6179182604604858, -0.14721450363901223,
  -0.026636860755398173, 0.0, 0.0, -2.167131170787887,
  0.6154280676828381, 0.0856720555012731, 0.0, 0.0,
  -0.31465338418519784, 0.10815577059544967, 0.010969692854284618, 0.0,
  0.0, -0.6651239832524852, 0.7549272381505261, -0.32415748096232144,
  0.0,
];

// An arbitrary complex spectrogram — what a vocoder head emits, which is *not*
// the STFT of anything. Nothing about it is Hermitian-consistent, so the
// inverse's own conventions are what decide the answer.
const FRAMES = 5;
const BINS = 5;
const ARBITRARY_REAL = Array.from({ length: FRAMES * BINS }, (_, i) => Math.cos(0.7 * i));
const ARBITRARY_IMAG = Array.from({ length: FRAMES * BINS }, (_, i) => Math.sin(0.3 * i) - 0.2);

// torch.istft(that, n_fft=8, hop_length=4, window=hann, center=True)
const ISTFT_OUT = [
  -0.015285027529679841, -0.1127699897975585, -0.2705202153510059, 0.049024203158730406,
  -0.016772928300166287, 0.195329033233566, 0.011144808911503153, 0.07040024687528103,
  0.046699269273959454, 0.022915544954493498, 0.13299576444233474, 0.10771855582479432,
  -0.07069075770621946, -0.11324549647897113, 0.11918595028055268, -0.1881816052340515,
];

describe("stft / reference", () => {
  it("gives the periodic Hann window, as torch does", () => {
    expect(agree(hannWindow(8), HANN8, TOLERANCE)).toBeNull();
    // The symmetric one is a different window, not a rounding difference: at
    // i=1 it is 0.188 against 0.146. np.hanning(8) is this.
    expect(hannWindow(8, false)[1]).toBeCloseTo(0.18825509907063326, 7);
  });

  it("matches torch.stft, frame for frame and bin for bin", () => {
    const spectrogram = stft({ input: SIGNAL, nFft: N_FFT, hop: HOP, window: hannWindow(N_FFT) });
    expect([spectrogram.frames, spectrogram.bins]).toEqual([5, 5]);
    expect(agree(spectrogram.real, STFT_REAL, TOLERANCE)).toBeNull();
    expect(agree(spectrogram.imag, STFT_IMAG, TOLERANCE)).toBeNull();
  });

  it("matches torch.istft on a spectrogram that is not anyone's transform", () => {
    const out = istft({
      real: ARBITRARY_REAL,
      imag: ARBITRARY_IMAG,
      frames: FRAMES,
      nFft: N_FFT,
      hop: HOP,
      window: hannWindow(N_FFT),
    });
    expect(out).toHaveLength(16);
    expect(agree(out, ISTFT_OUT, TOLERANCE)).toBeNull();
  });

  it("counts frames and output samples the way torch does", () => {
    // torch.stft frame counts, read off tensors: L=16 and L=17 both give 5.
    expect(stftFrames(8, 4, 16)).toBe(5);
    expect(stftFrames(8, 4, 17)).toBe(5);
    expect(stftFrames(8, 4, 20)).toBe(6);
    expect(stftFrames(8, 4, 64, false)).toBe(15);
    // torch.istft default lengths, likewise. Odd nFft is not hop*(frames-1).
    expect(istftLength(8, 4, 17)).toBe(64);
    expect(istftLength(7, 3, 11)).toBe(31);
    expect(istftLength(8, 4, 17, false)).toBe(72);
  });

  it("round-trips a signal whose length is not a multiple of hop", () => {
    // The default output length drops the last 16 - 5 = 1 sample, because the
    // frames were counted by floor. torch has the same gap and the same escape:
    // ask for the length back, up to what the frames reach.
    const input = Array.from({ length: 17 }, (_, i) => Math.cos(0.21 * i) - 0.03 * i);
    const window = hannWindow(N_FFT);
    const { real, imag, frames } = stft({ input, nFft: N_FFT, hop: HOP, window });
    expect(istftLength(N_FFT, HOP, frames)).toBe(16);
    expect(istftMaxLength(N_FFT, HOP, frames)).toBe(20);

    const out = istft({ real, imag, frames, nFft: N_FFT, hop: HOP, window, length: 17 });
    // 1e-5 because the spectrogram was stored as f32 in between; torch does the
    // same round trip in f64 and lands at 4e-16.
    expect(agree(out, input, { rel: 1e-5, abs: 1e-5 })).toBeNull();
  });

  it("refuses a length the frames do not reach", () => {
    // torch pads the tail with zeros and warns. Silence is a plausible-looking
    // answer, which is exactly why this raises instead.
    const window = hannWindow(N_FFT);
    expect(() =>
      istft({
        real: ARBITRARY_REAL, imag: ARBITRARY_IMAG, frames: FRAMES,
        nFft: N_FFT, hop: HOP, window, length: 21,
      }),
    ).toThrow(/reach 20 samples/);
  });

  it("refuses a window that fails NOLA, where torch refuses", () => {
    // Nonzero over the first two taps only, so with hop=4 every fifth sample is
    // covered by nothing. torch: "window overlap add min: 1".
    const gappy = Float32Array.from({ length: N_FFT }, (_, i) => (i < 2 ? 1 : 0));
    expect(() =>
      istft({
        real: ARBITRARY_REAL, imag: ARBITRARY_IMAG, frames: FRAMES,
        nFft: N_FFT, hop: HOP, window: gappy,
      }),
    ).toThrow(/NOLA/);
  });

  it("refuses a spectrogram whose length does not match its frame count", () => {
    const window = hannWindow(N_FFT);
    expect(() =>
      istft({ real: ARBITRARY_REAL, imag: ARBITRARY_IMAG, frames: 4, nFft: N_FFT, hop: HOP, window }),
    ).toThrow(/must be 4 x 5 = 20, got 25/);
  });
});
