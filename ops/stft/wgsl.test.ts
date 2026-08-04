import { describe, expect } from "vitest";
import { agree, expectAgrees, gpuTest, kernel, params, useGpu, type Dispatch } from "../../harness/index.js";
import { hannWindow, istft, istftLength, stft, stftBins, stftFrames } from "./reference.js";

const forward = kernel(import.meta.url);
const inverse = kernel(import.meta.url, "inverse");

/**
 * Everything expensive is computed here, at module scope, and the tests below
 * only dispatch and compare.
 *
 * Not a style choice. Measured on this machine: once `useGpu`'s `beforeAll` has
 * created the device, a test that takes even ~10ms before its first dispatch
 * kills the vitest worker — `terminate called after throwing an instance of
 * 'std::system_error'`, or "The futex facility returned an unexpected error
 * code", with no test result reported. A blocking loop does it and so does an
 * idle `await setTimeout`, so it is elapsed time rather than a starved event
 * loop. The same work at module scope, before the device exists, is fine at a
 * full second. This op is the first with a reference costing more than a
 * millisecond — 1920 points times 961 bins times 6 frames is 11M terms with a
 * `sin` and a `cos` each — which is why it is the first to hit it. Same family
 * as #38.
 */

/**
 * Slack past the end of every buffer, because this device is quiet about the
 * edges: it reads past the end of a storage buffer as zero and drops writes
 * past it. An input sized exactly to its contents hides a bad index behind a
 * convenient zero, and an output sized exactly to its results hides a missing
 * bound check behind a dropped write. So inputs carry a tail of SENTINEL —
 * which is nothing this arithmetic could produce — and outputs are longer than
 * the answer, with zeros expected in the slack.
 */
const SLACK = 64;
const SENTINEL = 7.77;

function withTail(data: Float32Array, fill: number): Float32Array {
  const padded = new Float32Array(data.length + SLACK).fill(fill);
  padded.set(data);
  return padded;
}

/** The answer, then zeros out to `total` — the slack a stray write would land in. */
function expandTo(data: ArrayLike<number>, total: number): Float32Array {
  const full = new Float32Array(total);
  full.set(data as ArrayLike<number> & Iterable<number>);
  return full;
}

const wave = (n: number, k = 0.37) => Float32Array.from({ length: n }, (_, i) => Math.sin(i * k) * 2);

/** Speech-ish: two partials and a slow envelope, amplitude about 1, at 24 kHz. */
const voice = (n: number) =>
  Float32Array.from({ length: n }, (_, i) => {
    const t = i / 24000;
    return (Math.sin(2 * Math.PI * 220 * t) * 0.6 + Math.sin(2 * Math.PI * 517 * t) * 0.3) *
      (0.6 + 0.4 * Math.sin(2 * Math.PI * 3 * t));
  });

/**
 * Hann lifted clear of zero.
 *
 * Uncentred, the first output sample is covered by frame 0 alone, and Hann is
 * exactly zero there — the envelope vanishes and NOLA fails. `torch.istft`
 * refuses the same call. Lifting the window keeps the uncentred case about the
 * arithmetic rather than about that edge.
 */
const liftedHann = (n: number) => Float32Array.from(hannWindow(n), (w) => w + 0.3);

/** An arbitrary complex spectrogram: a vocoder head's output, not anyone's transform. */
function arbitrary(frames: number, bins: number) {
  const real = Float32Array.from({ length: frames * bins }, (_, i) => Math.cos(0.7 * i) * (1 + (i % 5)));
  const imag = Float32Array.from({ length: frames * bins }, (_, i) => Math.sin(0.31 * i) - 0.2);
  return { real, imag };
}

/**
 * Tolerances are measured, not chosen. Each `abs` is about five times the worst
 * disagreement actually observed at that shape; `rel` stays at the harness
 * default, since the elements that miss on relative are the ones near zero and
 * they clear the absolute bound instead.
 *
 * Measured on this GPU with a throwaway shader, over exactly the angles these
 * kernels ask for — `2π(m mod N)/N`, folded into one turn: `cos` 6.4e-7 and
 * `sin` 7.2e-7 absolute. That is three orders better than the 1.86e-4 the
 * README records for `rope`, and the difference is the folding: `rope`'s angles
 * run into the thousands of radians, where f32 has nothing left. Here `k*n`
 * reaches 1.8M and is reduced modulo `nFft` as an integer before it becomes an
 * angle at all, so the transcendentals are asked only about one turn.
 *
 * What is left is f32 accumulation. Forward, one accumulator walks `nFft` terms:
 * at nFft=1920 over bins peaking at 2.2e2 the worst gap from the f64 reference
 * was 2.0e-4, and at the small shapes 9.5e-7. Inverse, over `bins` terms, 5.2e-7.
 */
const SMALL_FORWARD = { rel: 1e-5, abs: 5e-6 };
const SMALL_INVERSE = { rel: 1e-5, abs: 3e-6 };
const VOCODER_FORWARD = { rel: 1e-5, abs: 1e-3 };
/**
 * The inverse is 2500 times tighter than the forward at the same shape, and is
 * set from its own measurement rather than from the forward's. Not pedantry:
 * at the forward's bound, dropping the modular fold from the inverse's twiddle
 * passes — the mutation is invisible until the tolerance says what was actually
 * measured.
 */
const VOCODER_INVERSE = { rel: 1e-5, abs: 2e-6 };
/** Both directions end to end, worst observed 1.4e-6 on a signal of amplitude 1. */
const ROUND_TRIP = { rel: 1e-5, abs: 1e-5 };

interface Shape {
  name: string;
  nFft: number;
  hop: number;
  length: number;
  center: boolean;
  forwardTolerance: { rel: number; abs: number };
  inverseTolerance: { rel: number; abs: number };
}

// Shapes chosen to cross what breaks a windowed transform: bins well under the
// 256-wide workgroup, bins spanning it and not a multiple of it, a hop that
// does not divide nFft, an odd nFft (no Nyquist bin at all), and the uncentred
// variant where no padding is invented.
const small = { forwardTolerance: SMALL_FORWARD, inverseTolerance: SMALL_INVERSE };
const SHAPES: Shape[] = [
  { name: "nFft=8 hop=4 centred", nFft: 8, hop: 4, length: 16, center: true, ...small },
  { name: "nFft=16 hop=5 centred", nFft: 16, hop: 5, length: 100, center: true, ...small },
  { name: "nFft=7 hop=3 centred (odd, no Nyquist bin)", nFft: 7, hop: 3, length: 40, center: true, ...small },
  { name: "nFft=8 hop=4 uncentred", nFft: 8, hop: 4, length: 25, center: false, ...small },
  // MioCodec's istft_head: Linear[.., 1922] where 1922 = 2 * (1920/2 + 1),
  // 24 kHz at a 25 Hz frame rate, so hop 960 and exactly 2x overlap. 961 bins
  // is three and a bit workgroups wide.
  {
    name: "nFft=1920 hop=960 centred (vocoder)", nFft: 1920, hop: 960, length: 4800, center: true,
    forwardTolerance: VOCODER_FORWARD, inverseTolerance: VOCODER_INVERSE,
  },
];

function forwardDispatch(
  signal: Float32Array, length: number, nFft: number, hop: number, center: boolean, window: Float32Array,
): Dispatch {
  const bins = stftBins(nFft);
  const frames = stftFrames(nFft, hop, length, center);
  return {
    code: forward,
    bindings: [
      { kind: "storage", data: withTail(signal, SENTINEL) },
      { kind: "storage", data: window },
      { kind: "out", type: "f32", length: frames * bins + SLACK },
      { kind: "out", type: "f32", length: frames * bins + SLACK },
      {
        kind: "uniform",
        data: params([
          ["u32", nFft], ["u32", hop], ["u32", bins], ["u32", length], ["u32", padding(nFft, center)],
        ]),
      },
    ],
    workgroups: [frames],
  };
}

function inverseDispatch(
  real: Float32Array, imag: Float32Array, frames: number,
  nFft: number, hop: number, center: boolean, window: Float32Array, samples: number,
): Dispatch {
  const groups = Math.ceil(samples / 256);
  return {
    code: inverse,
    bindings: [
      { kind: "storage", data: withTail(real, SENTINEL) },
      { kind: "storage", data: withTail(imag, SENTINEL) },
      { kind: "storage", data: window },
      // Sized by the dispatch rather than by the answer, so that dropping the
      // bound check writes somewhere visible instead of somewhere ignored.
      { kind: "out", type: "f32", length: groups * 256 },
      {
        kind: "uniform",
        data: params([
          ["u32", nFft], ["u32", hop], ["u32", stftBins(nFft)], ["u32", frames],
          ["u32", samples], ["u32", padding(nFft, center)],
        ]),
      },
    ],
    workgroups: [groups],
  };
}

const padding = (nFft: number, center: boolean) => (center ? Math.floor(nFft / 2) : 0);

/**
 * `"same"` on the GPU, with no kernel change at all.
 *
 * The kernel already takes the leading crop as a uniform and says so: "the
 * caller resolves the convention". That claim is only worth anything if a
 * convention it has never seen works without touching it, which is what this
 * asserts — the mode is arithmetic on the host, not a branch in the shader.
 */
const SAME_N_FFT = 16;
const SAME_HOP = 4;
const SAME_FRAMES = 6;
const SAME_BINS = SAME_N_FFT / 2 + 1;
const SAME_WINDOW = hannWindow(SAME_N_FFT);
const SAME_REAL = Float32Array.from(
  { length: SAME_FRAMES * SAME_BINS },
  (_, i) => Math.cos(0.41 * i) * 1.7,
);
const SAME_IMAG = Float32Array.from(
  { length: SAME_FRAMES * SAME_BINS },
  (_, i) => Math.sin(0.29 * i) - 0.3,
);
const SAME_EXPECTED = istft({
  real: SAME_REAL,
  imag: SAME_IMAG,
  frames: SAME_FRAMES,
  nFft: SAME_N_FFT,
  hop: SAME_HOP,
  window: SAME_WINDOW,
  padding: "same",
});
const SAME_PAD = (SAME_N_FFT - SAME_HOP) / 2;

/** Every dispatch and every expected array, built before any device exists. */
gpuTest("synthesises same padding, with the crop resolved on the host", async (run) => {
  // Length is hop * frames = 24, not the 20 centred gives or the 36 uncentred
  // does — a wrong crop here is a waveform that is merely the wrong length.
  expect(SAME_EXPECTED).toHaveLength(SAME_HOP * SAME_FRAMES);
  await expectAgrees(
    run,
    {
      code: inverse,
      bindings: [
        { kind: "storage", data: SAME_REAL },
        { kind: "storage", data: SAME_IMAG },
        { kind: "storage", data: SAME_WINDOW },
        { kind: "out", type: "f32", length: SAME_EXPECTED.length },
        {
          kind: "uniform",
          data: params([
            ["u32", SAME_N_FFT], ["u32", SAME_HOP], ["u32", SAME_BINS],
            ["u32", SAME_FRAMES], ["u32", SAME_EXPECTED.length], ["u32", SAME_PAD],
          ]),
        },
      ],
      workgroups: [Math.ceil(SAME_EXPECTED.length / 256)],
    },
    [SAME_EXPECTED],
    SMALL_INVERSE,
  );
});

const FORWARD_CASES = SHAPES.map(({ name, nFft, hop, length, center, forwardTolerance: tolerance }) => {
  const window = center ? hannWindow(nFft) : liftedHann(nFft);
  const signal = nFft > 64 ? voice(length) : wave(length);
  const { real, imag, frames, bins } = stft({ input: signal, nFft, hop, window, center });
  return {
    name,
    tolerance,
    dispatch: forwardDispatch(signal, length, nFft, hop, center, window),
    expected: [
      expandTo(real, frames * bins + SLACK),
      expandTo(imag, frames * bins + SLACK),
    ],
  };
});

const INVERSE_CASES = SHAPES.map(({ name, nFft, hop, length, center, inverseTolerance: tolerance }) => {
  const window = center ? hannWindow(nFft) : liftedHann(nFft);
  const frames = stftFrames(nFft, hop, length, center);
  const { real, imag } = arbitrary(frames, stftBins(nFft));
  const samples = istftLength(nFft, hop, frames, center);
  const expected = istft({ real, imag, frames, nFft, hop, window, center });
  return {
    name,
    tolerance,
    dispatch: inverseDispatch(real, imag, frames, nFft, hop, center, window, samples),
    expected: [expandTo(expected, Math.ceil(samples / 256) * 256)],
  };
});

const ROUND_TRIPS = [
  { name: "nFft=8 hop=4", nFft: 8, hop: 4, length: 64 },
  { name: "nFft=1920 hop=960 (vocoder)", nFft: 1920, hop: 960, length: 4800 },
].map(({ name, nFft, hop, length }) => {
  const window = hannWindow(nFft);
  const signal = nFft > 64 ? voice(length) : wave(length);
  const frames = stftFrames(nFft, hop, length, true);
  return {
    name, nFft, hop, length, window, signal, frames,
    bins: stftBins(nFft),
    forward: forwardDispatch(signal, length, nFft, hop, true, window),
  };
});

describe("stft / wgsl", () => {
  useGpu();

  for (const { name, dispatch, expected, tolerance } of FORWARD_CASES) {
    gpuTest(`stft agrees with the reference at ${name}`, async (run) => {
      await expectAgrees(run, dispatch, expected, tolerance);
    });
  }

  for (const { name, dispatch, expected, tolerance } of INVERSE_CASES) {
    gpuTest(`istft agrees with the reference at ${name}`, async (run) => {
      await expectAgrees(run, dispatch, expected, tolerance);
    });
  }

  // The point of the whole op: two dispatches, nothing in between, and the
  // signal comes back. Several frames long, and at the shape a real vocoder
  // head emits.
  for (const { name, nFft, hop, length, window, signal, frames, bins, forward: dispatch } of ROUND_TRIPS) {
    gpuTest(`istft(stft(x)) returns x at ${name}`, async (run) => {
      // A length that is a multiple of hop is recoverable in full;
      // reference.test.ts covers the tail that flooring otherwise drops.
      expect(istftLength(nFft, hop, frames, true)).toBe(length);

      const [real, imag] = await run(dispatch);
      const [recovered] = await run(
        inverseDispatch(
          (real as Float32Array).slice(0, frames * bins),
          (imag as Float32Array).slice(0, frames * bins),
          frames, nFft, hop, true, window, length,
        ),
      );
      expect(agree((recovered as Float32Array).subarray(0, length), signal, ROUND_TRIP)).toBeNull();
    });
  }
});
