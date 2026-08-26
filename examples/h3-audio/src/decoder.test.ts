/**
 * The audio decoder against MiniMax-H3's own Python.
 *
 * Issue #200. `fixtures/waveform.bin` was produced by running the bundle's
 * `DacAudioVAE.decode` — the publisher's code, unmodified — on
 * `fixtures/latent.bin`, and the port has to reproduce it. Rule 7: this
 * library does not decide what a BigVGAN decoder means, the model does.
 *
 * **The weights are not in this repository.** They are 260 MB, and they are
 * under a licence that is not this code's. `H3_AUDIO_DIR` points at a directory
 * `tools/convert_audio_vae.py` wrote; without it the tests skip with a message
 * rather than passing, because a suite that goes green on a missing model is
 * the failure mode `gpu-tests-pass-vacuously` names.
 */
import { existsSync, readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { AudioVaeWeights, amplePadding, decodeAudio, type AudioVaeManifest } from "./decoder.js";

const fixture = (name: string): string => fileURLToPath(new URL(`../fixtures/${name}`, import.meta.url));
const f32 = (path: string): Float32Array => {
  const buffer = readFileSync(path);
  return new Float32Array(buffer.buffer, buffer.byteOffset, buffer.byteLength / 4);
};

const GOLDEN = JSON.parse(readFileSync(fixture("golden.json"), "utf8")) as {
  T: number;
  latentChannels: number;
  samples: number;
};

const weightsDir = process.env["H3_AUDIO_DIR"];
const haveWeights =
  weightsDir !== undefined &&
  existsSync(`${weightsDir}/decoder.manifest.json`) &&
  existsSync(`${weightsDir}/decoder.bin`);

describe("h3 audio decoder", () => {
  it("pads a dilated kernel the way BigVGAN does", () => {
    // `dac_utils.py`: get_padding(k, d) = int((k * d - d) / 2). Written out
    // because it is the one arithmetic in the port that is not a library call,
    // and an off-by-one here changes the length rather than the values — which
    // is the failure that looks like a shape bug three layers away.
    expect(amplePadding(3, 1)).toBe(1);
    expect(amplePadding(3, 3)).toBe(3);
    expect(amplePadding(3, 5)).toBe(5);
    expect(amplePadding(7, 1)).toBe(3);
    expect(amplePadding(7, 3)).toBe(9);
    expect(amplePadding(11, 5)).toBe(25);
  });

  it("has a golden whose length is the hop times the latent frames", () => {
    // 800 samples per latent frame — the upsample rates multiply to it, and
    // 32000 / 800 is the 40 Hz the model card states. This runs with or without
    // the weights: it is a property of the fixture, and a fixture that drifted
    // would take every comparison below with it.
    expect(GOLDEN.samples).toBe(GOLDEN.T * 800);
    expect(f32(fixture("latent.bin")).length).toBe(GOLDEN.latentChannels * GOLDEN.T);
    expect(f32(fixture("waveform.bin")).length).toBe(GOLDEN.samples);
  });

  const maybe = haveWeights ? it : it.skip;

  maybe("reproduces the waveform the model's own Python produced", () => {
    const manifest = JSON.parse(readFileSync(`${weightsDir}/decoder.manifest.json`, "utf8")) as AudioVaeManifest;
    const weights = new AudioVaeWeights(manifest, f32(`${weightsDir}/decoder.bin`));
    const latent = f32(fixture("latent.bin"));
    const want = f32(fixture("waveform.bin"));

    const got = decodeAudio(latent, GOLDEN.T, manifest, weights);

    expect(got.length).toBe(want.length);
    let worst = 0;
    for (let i = 0; i < want.length; i += 1) worst = Math.max(worst, Math.abs(got[i]! - want[i]!));
    // Measured on this fixture, not widened until green: worst element
    // **1.788e-6** at sample 1015 (0.820884 against 0.820886), RMS 2.248e-7,
    // against a signal peaking at 0.896. The decoder is 127 anti-aliased
    // activations deep and each is a convolution pair, so f32
    // order-of-summation differences accumulate; that is what this is.
    //
    // 1e-5 is about five times the worst measured, which is margin for a
    // different latent rather than for a different answer: every wrong
    // convention tried here — the snake logarithms left unexponentiated,
    // `tanh` instead of the clamp, `get_padding` off by one — moves samples by
    // 1e-1 or more, four decades above this.
    expect(worst).toBeLessThan(1e-5);
  });

  maybe("stays inside [-1, 1], because this configuration clamps rather than squashes", () => {
    const manifest = JSON.parse(readFileSync(`${weightsDir}/decoder.manifest.json`, "utf8")) as AudioVaeManifest;
    const weights = new AudioVaeWeights(manifest, f32(`${weightsDir}/decoder.bin`));
    const got = decodeAudio(f32(fixture("latent.bin")), GOLDEN.T, manifest, weights);
    for (const v of got) expect(Math.abs(v)).toBeLessThanOrEqual(1);
  });

  maybe("names a missing weight rather than returning an empty view", () => {
    const manifest = JSON.parse(readFileSync(`${weightsDir}/decoder.manifest.json`, "utf8")) as AudioVaeManifest;
    const weights = new AudioVaeWeights(manifest, f32(`${weightsDir}/decoder.bin`));
    expect(() => weights.get("ups.99.weight")).toThrow(/ups\.99\.weight/);
  });

  if (!haveWeights) {
    it("says why the comparison did not run", () => {
      // Not a passing test dressed as a skip: this asserts the reason is the
      // one stated, so a broken path cannot read as "no weights available".
      expect(weightsDir === undefined || !existsSync(`${weightsDir}/decoder.bin`)).toBe(true);
      console.log(
        "h3 audio: set H3_AUDIO_DIR to a directory written by tools/convert_audio_vae.py " +
          "to compare against the model's own output",
      );
    });
  }
});
