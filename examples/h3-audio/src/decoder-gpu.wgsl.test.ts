/**
 * The GPU decoder against the same waveform the model's own Python produced.
 *
 * Issue #200. Not against `decoder.ts`: agreeing with the CPU port only says
 * the two share a mistake. The golden is the model's, and both are held to it.
 *
 * Skipped, loudly, without `H3_AUDIO_DIR` — the weights are 260 MB and are not
 * this repository's to ship. A suite that goes green on a missing model is the
 * failure `gpu-tests-pass-vacuously` names.
 */
import { existsSync, readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { describe, expect } from "vitest";
import { gpuTest, useGpu } from "../../../harness/index.js";
import { createResidentDevice } from "../../../harness/resident.js";
import { AudioVaeWeights, type AudioVaeManifest } from "./decoder.js";
import { AudioVaeGpu } from "./decoder-gpu.js";
import { audioKernels } from "./kernels-node.js";

const fixture = (name: string): string => fileURLToPath(new URL(`../fixtures/${name}`, import.meta.url));
const f32 = (path: string): Float32Array => {
  const buffer = readFileSync(path);
  return new Float32Array(buffer.buffer, buffer.byteOffset, buffer.byteLength / 4);
};

const dir = process.env["H3_AUDIO_DIR"];
const haveWeights = dir !== undefined && existsSync(`${dir}/decoder.bin`);
const GOLDEN = JSON.parse(readFileSync(fixture("golden.json"), "utf8")) as { T: number; samples: number };

describe("h3 audio decoder / wgsl", () => {
  useGpu();

  gpuTest("reproduces the model's own waveform on the device", async () => {
    if (!haveWeights) {
      console.log("h3 audio (gpu): set H3_AUDIO_DIR to compare against the model's own output");
      return;
    }
    const device = await createResidentDevice();
    if (!device) return;
    try {
      const manifest = JSON.parse(readFileSync(`${dir}/decoder.manifest.json`, "utf8")) as AudioVaeManifest;
      const weights = new AudioVaeWeights(manifest, f32(`${dir}/decoder.bin`));
      const decoder = new AudioVaeGpu(device, audioKernels(), manifest, weights);
      const got = await decoder.decode(f32(fixture("latent.bin")), GOLDEN.T);
      decoder.destroy();

      const want = f32(fixture("waveform.bin"));
      expect(got.length).toBe(want.length);
      let worst = 0;
      for (let i = 0; i < want.length; i += 1) worst = Math.max(worst, Math.abs(got[i]! - want[i]!));
      console.log(`h3 audio (gpu): worst element ${worst.toExponential(3)}`);
      // Measured: **5.007e-6** worst element on this fixture, RTX 5090 / driver
      // 610.57.04 / Dawn `webgpu@0.4.0`, f32. Looser than the CPU port's 1.788e-6
      // because a GPU sums a convolution window in a different order and the
      // compiler is free to contract into an FMA, and the decoder is 127
      // anti-aliased activations deep.
      //
      // 2e-5 is four times that — margin for a different latent, not for a
      // different answer. Every wrong convention moves samples by 1e-1.
      expect(worst).toBeLessThan(2e-5);
    } finally {
      device.destroy();
    }
  });
});
