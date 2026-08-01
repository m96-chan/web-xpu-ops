import { describe, expect } from "vitest";
import { agree, expectAgrees, gpuTest, kernel, params, useGpu, type Dispatch } from "../../harness/index.js";
import { hannWindow, stft, stftBins, stftFrames } from "../stft/reference.js";
import {
  melBins,
  melFilterbank,
  melSpectrogram,
  type MelNorm,
  type MelPower,
  type MelScale,
} from "./reference.js";

const build = kernel(import.meta.url, "filterbank");
const apply = kernel(import.meta.url);
const forwardStft = kernel(new URL("../stft/reference.ts", import.meta.url));

/**
 * Everything expensive is computed here, at module scope, and the tests below
 * only dispatch and compare.
 *
 * Not a style choice, and not this file's discovery — `ops/stft/wgsl.test.ts`
 * records the measurement. Once `useGpu`'s `beforeAll` has created the device, a
 * test that spends even ~10ms before its first dispatch kills the vitest worker
 * with no result reported, and it is elapsed time rather than a starved event
 * loop. Mel is the reference this was predicted to bite next: an 80-band bank
 * over 961 bins is 77k triangle evaluations, and the STFT feeding the
 * end-to-end case below is another 165k terms with a `sin` and a `cos` each.
 * Before the device exists, the same work is free.
 */

/**
 * Slack past the end of every buffer, because this device is quiet about the
 * edges: it reads past the end of a storage buffer as zero and drops writes
 * past it. An input sized exactly to its contents hides a bad index behind a
 * convenient zero, and an output sized exactly to its results hides a missing
 * bound check behind a dropped write. Inputs carry a tail of SENTINEL — nothing
 * this arithmetic could produce — and outputs are longer than the answer, with
 * zeros expected in the slack.
 */
const SLACK = 64;
const SENTINEL = 7.77;

function withTail(data: Float32Array, fill = SENTINEL): Float32Array {
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

const SCALE_CODE: Record<MelScale, number> = { htk: 0, slaney: 1 };
const NORM_CODE: Record<MelNorm, number> = { none: 0, slaney: 1 };

/**
 * Tolerances are measured, not chosen. Each case below carries the worst
 * disagreement actually observed at that shape and is checked at five times it,
 * which is the same rule `ops/stft/wgsl.test.ts` uses.
 *
 * What the filterbank's numbers say is worth reading rather than skipping,
 * because it is a property of the op rather than of this kernel. The error is
 * dominated by the f32 rounding of the *mel coordinate*, not by `pow`: the band
 * edges are `melMin + m * step`, and one f32 ulp of `step` accumulated over
 * eighty bands is ~2e-4 mel, which near the bottom of the scale is ~1e-4 Hz.
 * That is harmless against a band hundreds of Hz wide and is not harmless
 * against a narrow one, because the triangle's slope is `1 / bandwidth`. So the
 * measured error tracks band density and nothing else:
 *
 *   4 bands over 17 bins    5.4e-7   bands ~2000 Hz wide
 *   80 bands over 961 bins  8.6e-6
 *   300 bands over 129 bins 4.3e-5   bands narrower than the bin spacing
 *
 * Two orders of magnitude, from the same kernel, on the same device. A caller
 * choosing `nMels` is choosing this, which is why the last row is in the suite
 * rather than left out for being degenerate.
 */

interface BankCase {
  name: string;
  nMels: number;
  nFft: number;
  sampleRate: number;
  fMin?: number;
  fMax?: number;
  scale: MelScale;
  norm: MelNorm;
  /** Worst absolute disagreement observed at this shape. Checked at 5x. */
  worst: number;
}

// Sized to cross the 256-wide dispatch three ways — under it, exactly on it,
// and one thread over — because the guard on the last workgroup is the only
// thing this kernel can get wrong that is not arithmetic. Then the two
// realistic banks, and all four corners of scale x norm, because those are the
// conventions the op exists to name.
const BANKS: BankCase[] = [
  { name: "htk/none, 4 x 17 = 68 threads (under one workgroup)", nMels: 4, nFft: 32, sampleRate: 16000, scale: "htk", norm: "none", worst: 5.4e-7 },
  { name: "htk/none, 4 x 64 = 256 threads (exactly one workgroup)", nMels: 4, nFft: 126, sampleRate: 16000, scale: "htk", norm: "none", worst: 4.1e-7 },
  { name: "slaney/slaney, 4 x 65 = 260 threads (one thread over)", nMels: 4, nFft: 128, sampleRate: 16000, scale: "slaney", norm: "slaney", worst: 3.5e-10 },
  { name: "slaney/none, 4 x 17", nMels: 4, nFft: 32, sampleRate: 16000, scale: "slaney", norm: "none", worst: 5.3e-7 },
  { name: "htk/slaney, 4 x 17", nMels: 4, nFft: 32, sampleRate: 16000, scale: "htk", norm: "slaney", worst: 3.1e-10 },
  // whisper's bank, on librosa's conventions and band-limited the way speech
  // front ends limit it.
  { name: "slaney/slaney, 80 x 201 band-limited to 50..7600 Hz", nMels: 80, nFft: 400, sampleRate: 16000, fMin: 50, fMax: 7600, scale: "slaney", norm: "slaney", worst: 6.4e-8 },
  // MioCodec's transform: 24 kHz, nFft 1920, 961 bins. 76880 threads.
  { name: "htk/none, 80 x 961 (vocoder)", nMels: 80, nFft: 1920, sampleRate: 24000, scale: "htk", norm: "none", worst: 8.7e-6 },
  // Bands narrower than the bin spacing. torchaudio warns here and this does
  // not; the tolerance is what the warning would have been about.
  { name: "htk/none, 300 x 129 (bands narrower than the bins)", nMels: 300, nFft: 256, sampleRate: 48000, scale: "htk", norm: "none", worst: 4.3e-5 },
];

/** The bank the end-to-end case builds, 20 x 257: worst observed 2.4e-6. */
const CHAIN_BANK = { rel: 1e-5, abs: 1.2e-5 };

function bankDispatch({ nMels, nFft, sampleRate, fMin = 0, fMax, scale, norm }: BankCase): Dispatch {
  const bins = melBins(nFft);
  const groups = Math.ceil((nMels * bins) / 256);
  return {
    code: build,
    bindings: [
      // Sized by the dispatch rather than by the answer, so that dropping the
      // bound check writes somewhere visible instead of somewhere ignored.
      { kind: "out", type: "f32", length: groups * 256 },
      {
        kind: "uniform",
        data: params([
          ["u32", nMels], ["u32", bins], ["u32", nFft], ["f32", sampleRate],
          ["f32", fMin], ["f32", fMax ?? sampleRate / 2],
          ["u32", SCALE_CODE[scale]], ["u32", NORM_CODE[norm]],
        ]),
      },
    ],
    workgroups: [groups],
  };
}

const BANK_CASES = BANKS.map((bank) => {
  const bins = melBins(bank.nFft);
  const groups = Math.ceil((bank.nMels * bins) / 256);
  return {
    name: bank.name,
    tolerance: { rel: 1e-5, abs: 5 * bank.worst },
    dispatch: bankDispatch(bank),
    expected: [expandTo(melFilterbank(bank), groups * 256)],
  };
});

// --------------------------------------------------------------- application

interface ApplyCase {
  name: string;
  frames: number;
  nFft: number;
  nMels: number;
  sampleRate: number;
  scale: MelScale;
  norm: MelNorm;
  power: MelPower;
  db: boolean;
  /** Feed zeros, so every band lands on the dB floor. */
  silent?: boolean;
  tolerance: { rel: number; abs: number };
}

/**
 * The application kernel is handed the reference's own filterbank, so none of
 * the construction error above reaches it and what is left is one f32 dot
 * product and one `log`. Measured over every case below: **2.7e-7 relative** on
 * the linear scale, worst at 961 bins, and **3.9e-6 relative / 3.8e-6
 * absolute** in decibels. Both bounds are about five times that. The dB bound
 * needs its `abs` leg because decibels cross zero and relative error there
 * means nothing; the linear bound does not, so its `abs` is set below anything
 * it could rescue.
 */
const LINEAR = { rel: 1.5e-6, abs: 1e-9 };
const DECIBEL = { rel: 2e-5, abs: 2e-5 };

const APPLIES: ApplyCase[] = [
  { name: "3 frames, 4 mels, 17 bins, power, linear", frames: 3, nFft: 32, nMels: 4, sampleRate: 16000, scale: "htk", norm: "none", power: 2, db: false, tolerance: LINEAR },
  { name: "5 frames, 16 mels, 129 bins, power, dB", frames: 5, nFft: 256, nMels: 16, sampleRate: 48000, scale: "htk", norm: "none", power: 2, db: true, tolerance: DECIBEL },
  { name: "4 frames, 16 mels, 129 bins, magnitude, linear", frames: 4, nFft: 256, nMels: 16, sampleRate: 48000, scale: "htk", norm: "none", power: 1, db: false, tolerance: LINEAR },
  { name: "4 frames, 16 mels, 129 bins, magnitude, dB", frames: 4, nFft: 256, nMels: 16, sampleRate: 48000, scale: "htk", norm: "none", power: 1, db: true, tolerance: DECIBEL },
  { name: "3 frames, 80 mels, 201 bins, slaney/slaney, power, linear", frames: 3, nFft: 400, nMels: 80, sampleRate: 16000, scale: "slaney", norm: "slaney", power: 2, db: false, tolerance: LINEAR },
  // 961 bins is three and a bit workgroups of accumulation per output element.
  { name: "2 frames, 80 mels, 961 bins (vocoder), power, dB", frames: 2, nFft: 1920, nMels: 80, sampleRate: 24000, scale: "htk", norm: "none", power: 2, db: true, tolerance: DECIBEL },
  { name: "2 frames, 80 mels, 961 bins (vocoder), power, linear", frames: 2, nFft: 1920, nMels: 80, sampleRate: 24000, scale: "htk", norm: "none", power: 2, db: false, tolerance: LINEAR },
  // 300 mels over 129 bins: the lane loop wraps past 256 and does not land on
  // it, and enough bands fall between two bin centres that the dB floor is
  // reached from a real dispatch rather than from a silent one.
  { name: "2 frames, 300 mels, 129 bins, dB (bands narrower than the bins)", frames: 2, nFft: 256, nMels: 300, sampleRate: 48000, scale: "htk", norm: "none", power: 2, db: true, tolerance: DECIBEL },
  { name: "2 frames, 300 mels, 129 bins, linear", frames: 2, nFft: 256, nMels: 300, sampleRate: 48000, scale: "htk", norm: "none", power: 2, db: false, tolerance: LINEAR },
  // Silence in dB is -100 for a power spectrum and -200 for a magnitude one,
  // both of which are the floor and neither of which is zero — so this case
  // still fails if the dispatch never runs and the buffer reads back empty.
  { name: "silence floors at -100 dB (power)", frames: 2, nFft: 256, nMels: 16, sampleRate: 48000, scale: "htk", norm: "none", power: 2, db: true, silent: true, tolerance: DECIBEL },
  { name: "silence floors at -200 dB (magnitude)", frames: 2, nFft: 256, nMels: 16, sampleRate: 48000, scale: "htk", norm: "none", power: 1, db: true, silent: true, tolerance: DECIBEL },
];

/** An arbitrary complex spectrogram: a vocoder head's output, not anyone's transform. */
function arbitrary(frames: number, bins: number) {
  const real = Float32Array.from({ length: frames * bins }, (_, i) => Math.cos(0.7 * i) * (1 + (i % 5)));
  const imag = Float32Array.from({ length: frames * bins }, (_, i) => Math.sin(0.31 * i) - 0.2);
  return { real, imag };
}

const AMIN = 1e-10;

function applyDispatch(
  real: Float32Array, imag: Float32Array, filterbank: Float32Array,
  frames: number, bins: number, nMels: number, power: MelPower, db: boolean,
): Dispatch {
  return {
    code: apply,
    bindings: [
      { kind: "storage", data: withTail(real) },
      { kind: "storage", data: withTail(imag) },
      { kind: "storage", data: withTail(filterbank) },
      { kind: "out", type: "f32", length: frames * nMels + SLACK },
      {
        kind: "uniform",
        data: params([
          ["u32", bins], ["u32", nMels], ["u32", power], ["u32", db ? 1 : 0], ["f32", AMIN],
        ]),
      },
    ],
    workgroups: [frames],
  };
}

const APPLY_CASES = APPLIES.map((c) => {
  const bins = melBins(c.nFft);
  const { real, imag } = c.silent
    ? { real: new Float32Array(c.frames * bins), imag: new Float32Array(c.frames * bins) }
    : arbitrary(c.frames, bins);
  const filterbank = melFilterbank({
    nMels: c.nMels, nFft: c.nFft, sampleRate: c.sampleRate, scale: c.scale, norm: c.norm,
  });
  const expected = melSpectrogram({
    real, imag, frames: c.frames, bins, filterbank, nMels: c.nMels,
    power: c.power, db: c.db, amin: AMIN,
  });
  return {
    name: c.name,
    tolerance: c.tolerance,
    dispatch: applyDispatch(real, imag, filterbank, c.frames, bins, c.nMels, c.power, c.db),
    expected: [expandTo(expected, c.frames * c.nMels + SLACK)],
  };
});

// ------------------------------------------------------------- the whole chain

/** A deterministic hash, so the noise floor below is the same in every run. */
const hiss = (i: number) => {
  let x = (Math.imul(i, 1103515245) + 12345) >>> 0;
  x ^= x >>> 15;
  x = Math.imul(x, 2246822519) >>> 0;
  x ^= x >>> 13;
  return (x / 4294967296) * 2 - 1;
};

/**
 * Speech-ish: two partials, a slow envelope, and a noise floor 60 dB down.
 *
 * The hiss is not decoration. Without it this is two pure tones, and a mel band
 * at 5 kHz then holds nothing but the Hann window's leakage — a number around
 * 1e-13 assembled by near-total cancellation inside the DFT, which f32 cannot
 * represent to any useful precision. Measured: the same end-to-end case runs at
 * **0.92 dB** of disagreement on the pure tones and **4.2e-4 dB** once a -60 dB
 * floor is under them, from the same two kernels. Real audio always has that
 * floor; a synthetic input that omits it is not a harder test, it is a test of
 * whether f32 can hold a number that is not there.
 */
const voice = (n: number) =>
  Float32Array.from({ length: n }, (_, i) => {
    const t = i / 24000;
    return (Math.sin(2 * Math.PI * 220 * t) * 0.6 + Math.sin(2 * Math.PI * 517 * t) * 0.3) *
      (0.6 + 0.4 * Math.sin(2 * Math.PI * 3 * t)) + hiss(i) * 1e-3;
  });

const E2E = { nFft: 512, hop: 128, nMels: 20, sampleRate: 24000, length: 1024 };
const CHAIN = (() => {
  const { nFft, hop, nMels, sampleRate, length } = E2E;
  const window = hannWindow(nFft);
  const signal = voice(length);
  const bins = stftBins(nFft);
  const frames = stftFrames(nFft, hop, length);
  const spectrogram = stft({ input: signal, nFft, hop, window });
  const filterbank = melFilterbank({ nMels, nFft, sampleRate });
  return {
    ...E2E, bins, frames, window, signal, filterbank,
    expected: melSpectrogram({
      real: spectrogram.real, imag: spectrogram.imag, frames, bins, filterbank, nMels,
      power: 2, db: true, amin: AMIN,
    }),
    stft: {
      code: forwardStft,
      bindings: [
        { kind: "storage", data: withTail(signal) },
        { kind: "storage", data: window },
        { kind: "out", type: "f32", length: frames * bins + SLACK },
        { kind: "out", type: "f32", length: frames * bins + SLACK },
        {
          kind: "uniform",
          data: params([
            ["u32", nFft], ["u32", hop], ["u32", bins], ["u32", length], ["u32", Math.floor(nFft / 2)],
          ]),
        },
      ],
      workgroups: [frames],
    } satisfies Dispatch,
  };
})();

/** Both kernels end to end, worst observed 4.2e-4 dB over 180 bands. */
const CHAIN_TOLERANCE = { rel: 1e-5, abs: 2.5e-3 };

describe("mel / wgsl", () => {
  useGpu();

  for (const { name, dispatch, expected, tolerance } of BANK_CASES) {
    gpuTest(`the filterbank agrees with the reference: ${name}`, async (run) => {
      await expectAgrees(run, dispatch, expected, tolerance);
    });
  }

  for (const { name, dispatch, expected, tolerance } of APPLY_CASES) {
    gpuTest(`applying it agrees with the reference: ${name}`, async (run) => {
      await expectAgrees(run, dispatch, expected, tolerance);
    });
  }

  // The point of the op: an STFT and a log-mel, two dispatches, nothing on the
  // CPU in between. Compared against the reference chain, never against torch —
  // reference.test.ts is where the conventions are pinned to a library.
  gpuTest("log-mel of an STFT, both kernels, nothing on the CPU in between", async (run) => {
    const { bins, frames, nMels, nFft, sampleRate, filterbank, expected } = CHAIN;
    const [real, imag] = await run(CHAIN.stft);

    const [bank] = await run(bankDispatch({
      nMels, nFft, sampleRate, scale: "htk", norm: "none", name: "", worst: 0,
    }));
    expect(agree((bank as Float32Array).subarray(0, nMels * bins), filterbank, CHAIN_BANK)).toBeNull();

    const [mel] = await run(applyDispatch(
      (real as Float32Array).slice(0, frames * bins),
      (imag as Float32Array).slice(0, frames * bins),
      (bank as Float32Array).slice(0, nMels * bins),
      frames, bins, nMels, 2, true,
    ));
    expect(agree((mel as Float32Array).subarray(0, frames * nMels), expected, CHAIN_TOLERANCE)).toBeNull();
  });
});
