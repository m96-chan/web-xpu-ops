import { existsSync, readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { beforeAll, describe, expect, it } from "vitest";
import { type DecoderConfig, decode } from "./decoder.js";

/**
 * Decodes Z-Image's own latent with this repository's ops and checks the
 * result against the model's own decode of the same latent.
 *
 * The weights are 190 MB and gitignored — `tools/gen_latent.py` regenerates
 * them from `Tongyi-MAI/Z-Image`. Skipped, visibly, when they are absent:
 * CI has no checkpoint, and a silent pass there would make this look like
 * coverage it is not (the trap this repository has hit before, `gpuTest`'s
 * early return).
 */

// The 32px golden. The CPU reference is six nested loops by design — the
// reference is meant to be the slowest, plainest statement of the maths — and
// it costs 64.9s at 64px, over this runner's 60s per-file budget. The GPU test
// beside this one uses 64px, and the demo does 256px; correctness does not
// depend on the size, since every block, channel count and group is identical
// between them and only H and W shrink.
const fixtures = new URL("../fixtures-tiny/", import.meta.url);
const weightsPath = fileURLToPath(new URL("decoder.bin", fixtures));
const present = existsSync(weightsPath);

interface Manifest {
  config: {
    block_out_channels: number[];
    layers_per_block: number;
    norm_num_groups: number;
    latent_channels: number;
    out_channels: number;
    scaling_factor: number;
    shift_factor: number;
  };
  latent: { name: string; shape: number[]; offset: number; length: number }[];
  decoder: { name: string; shape: number[]; offset: number; length: number }[];
}

describe.skipIf(!present)("Z-Image VAE decoder, composed from ops", () => {
  // Everything is read in `beforeAll`, not in the describe body. `skipIf` skips
  // the *tests*; the body around them still runs, so an eager read here would
  // throw ENOENT on any machine without the gitignored weights — which is every
  // CI runner. It did, once.
  let cfg: DecoderConfig;
  let weights: (name: string) => Float32Array;
  let latent: Float32Array;
  let want: Float32Array;
  let latentH = 0;
  let latentW = 0;

  beforeAll(() => {
    const manifest = JSON.parse(readFileSync(fileURLToPath(new URL("manifest.json", fixtures)), "utf8")) as Manifest;
    const read = (file: string): Float32Array => {
      const raw = readFileSync(fileURLToPath(new URL(file, fixtures)));
      return new Float32Array(raw.buffer, raw.byteOffset, raw.byteLength / 4);
    };
    const latentBlob = read("latent.bin");
    const weightBlob = read("decoder.bin");
    const slice = (blob: Float32Array, table: Manifest["latent"], name: string): Float32Array => {
      const t = table.find((e) => e.name === name);
      if (!t) throw new Error(`fixture has no "${name}" — regenerate with tools/gen_latent.py`);
      return blob.subarray(t.offset, t.offset + t.length);
    };
    cfg = {
      blockOutChannels: manifest.config.block_out_channels,
      layersPerBlock: manifest.config.layers_per_block,
      normNumGroups: manifest.config.norm_num_groups,
      latentChannels: manifest.config.latent_channels,
      outChannels: manifest.config.out_channels,
      scalingFactor: manifest.config.scaling_factor,
      shiftFactor: manifest.config.shift_factor,
    };
    weights = (name) => slice(weightBlob, manifest.decoder, name);
    latent = slice(latentBlob, manifest.latent, "latent");
    want = slice(latentBlob, manifest.latent, "reference");
    const shape = manifest.latent.find((e) => e.name === "latent")!.shape;
    latentH = shape[2]!;
    latentW = shape[3]!;
  });

  it("reproduces the model's own decode", () => {
    const got = decode(cfg, weights, latent, latentH, latentW);

    expect(got.data.length).toBe(want.length);

    let worst = 0;
    for (let i = 0; i < want.length; i += 1) worst = Math.max(worst, Math.abs(got.data[i]! - want[i]!));

    // Pixels live in [-1, 1], so this bound is on the same scale as the output
    // rather than relative. The measured worst is printed so the number in this
    // comment can be checked against a run rather than trusted.
    //
    // 49.5M parameters and ~1.4 GFLOP of f32 accumulate visibly more than a
    // single block does, and none of it is a difference in arithmetic: the
    // reference sums in a different order, in a different library, on a GPU.
    console.log(`worst |ours - model| = ${worst.toExponential(3)} over ${want.length} pixels`);
    expect(worst).toBeLessThan(2e-3);
  });
});
