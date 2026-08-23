import { existsSync, readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { beforeAll, describe, expect } from "vitest";
import { gpuTest, useGpu } from "../../../harness/index.js";
import { type DecoderConfig } from "./decoder.js";
import { decodeGpu } from "./decoder-gpu.js";

/**
 * The GPU decoder against the model's own decode of the same latent.
 *
 * Not against `decoder.ts`. Both paths are checked against the same external
 * answer instead, because two of this repository's implementations agreeing
 * only says they share an assumption — and they do share one, since the GPU
 * path was written by reading the CPU path. The model is the third party.
 *
 * Skipped visibly when the 190 MB weights are absent (CI has no checkpoint),
 * rather than passing with nothing run.
 */

const fixtures = new URL("../fixtures-small/", import.meta.url);
const present = existsSync(fileURLToPath(new URL("decoder.bin", fixtures)));

interface Entry { name: string; shape: number[]; offset: number; length: number }
interface Manifest {
  config: {
    block_out_channels: number[]; layers_per_block: number; norm_num_groups: number;
    latent_channels: number; out_channels: number; scaling_factor: number; shift_factor: number;
  };
  latent: Entry[];
  decoder: Entry[];
}

describe.skipIf(!present)("Z-Image VAE decoder on the GPU", () => {
  useGpu();

  let manifest: Manifest;
  let cfg: DecoderConfig;
  let weights: (name: string) => Float32Array;
  let latent: Float32Array;
  let reference: Float32Array;
  let latentH = 0;
  let latentW = 0;

  beforeAll(() => {
    manifest = JSON.parse(readFileSync(fileURLToPath(new URL("manifest.json", fixtures)), "utf8")) as Manifest;
    const read = (file: string): Float32Array => {
      const raw = readFileSync(fileURLToPath(new URL(file, fixtures)));
      return new Float32Array(raw.buffer, raw.byteOffset, raw.byteLength / 4);
    };
    const latentBlob = read("latent.bin");
    const weightBlob = read("decoder.bin");
    const pick = (blob: Float32Array, table: Entry[], name: string): Float32Array => {
      const e = table.find((t) => t.name === name);
      if (!e) throw new Error(`fixture has no "${name}" — regenerate with tools/gen_latent.py`);
      return blob.subarray(e.offset, e.offset + e.length);
    };
    const c = manifest.config;
    cfg = {
      blockOutChannels: c.block_out_channels, layersPerBlock: c.layers_per_block,
      normNumGroups: c.norm_num_groups, latentChannels: c.latent_channels,
      outChannels: c.out_channels, scalingFactor: c.scaling_factor, shiftFactor: c.shift_factor,
    };
    weights = (n) => pick(weightBlob, manifest.decoder, n);
    latent = pick(latentBlob, manifest.latent, "latent");
    reference = pick(latentBlob, manifest.latent, "reference");
    const shape = manifest.latent.find((e) => e.name === "latent")!.shape;
    latentH = shape[2]!;
    latentW = shape[3]!;
  });

  gpuTest("reproduces the model's own decode", async (run) => {
    const started = performance.now();
    const got = await decodeGpu(run, cfg, weights, latent, latentH, latentW);
    const elapsed = performance.now() - started;

    expect(got.data.length).toBe(reference.length);

    let worst = 0;
    for (let i = 0; i < reference.length; i += 1) {
      worst = Math.max(worst, Math.abs(got.data[i]! - reference[i]!));
    }

    // Reported rather than only asserted: the CPU path's own figure is
    // 1.878e-6 on this fixture, so a reader can see whether moving to the GPU
    // moved the answer, and by how much.
    console.log(`GPU decode ${got.H}x${got.W} in ${(elapsed / 1000).toFixed(2)}s, worst |ours - model| = ${worst.toExponential(3)}`);

    // Pixels are in [-1, 1], so this is an absolute bound on the same scale as
    // the output. Set from the CPU path's measured 1.878e-6 with room for the
    // GPU summing in a different order across 49.5M parameters; a wrong
    // dispatch is orders of magnitude away, not a few ulps.
    expect(worst).toBeLessThan(5e-3);
  });
});
