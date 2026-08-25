/**
 * The encoder resnet block against the model's own `ResnetBlock3D`.
 *
 * Issue #200. This is where `ops/conv`'s 3D entry and `ops/pad` are used — the
 * decoder is a ViT and needs neither, so without this they are ops with no
 * caller.
 *
 * `tools/gen_resnet_golden.py` imports the block from the bundle the checkpoint
 * ships. `H3_ENCODER_DIR` points at its output; without it the comparison skips
 * with a message rather than passing.
 */
import { existsSync, readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { h3EncoderResnetBlock, padForCausalConv, type ResnetBlockConfig, type ResnetBlockWeights } from "./encoder-block.js";

const dir = process.env["H3_ENCODER_DIR"];
const have = dir !== undefined && existsSync(`${dir}/resnet.bin`) && existsSync(`${dir}/resnet.manifest.json`);

const f32 = (path: string): Float32Array => {
  const buffer = readFileSync(path);
  return new Float32Array(buffer.buffer, buffer.byteOffset, buffer.byteLength / 4);
};

describe("h3 video vae / encoder resnet block", () => {
  it("pads space symmetrically and time only in front", () => {
    // The property that makes the convolution causal, checked on a tensor small
    // enough to read: one channel, three frames of 2x2 holding 1..12.
    const x = Float32Array.from({ length: 12 }, (_, i) => i + 1);
    const out = padForCausalConv(x, 1, 3, 2, 2, 1, 2);
    expect([out.D, out.H, out.W]).toEqual([5, 4, 4]);
    // The two prepended frames are zero, and they are at the **front**.
    expect([...out.data.slice(0, 32)].every((v) => v === 0)).toBe(true);
    // The last frame is the real one, not a pad.
    expect([...out.data.slice(-16)].some((v) => v !== 0)).toBe(true);
  });

  const maybe = have ? it : it.skip;

  maybe("reproduces the block the model's own code produced", () => {
    const manifest = JSON.parse(readFileSync(`${dir}/resnet.manifest.json`, "utf8")) as {
      inChannels: number;
      outChannels: number;
      dims: [number, number, number];
      config: { groups: number; norm_eps: number };
      tensors: { name: string; offset: number; count: number }[];
      elements: number;
    };
    const flat = f32(`${dir}/resnet.bin`);
    expect(flat.length).toBe(manifest.elements);
    const view = (name: string): Float32Array => {
      const entry = manifest.tensors.find((t) => t.name === name);
      if (!entry) throw new Error(`resnet.bin has no tensor named "${name}"`);
      return flat.subarray(entry.offset, entry.offset + entry.count);
    };

    const [D, H, W] = manifest.dims;
    const cfg: ResnetBlockConfig = {
      inChannels: manifest.inChannels,
      outChannels: manifest.outChannels,
      D, H, W,
      groups: manifest.config.groups,
      normEps: manifest.config.norm_eps,
    };
    const weights: ResnetBlockWeights = {
      norm1Weight: view("norm1.weight"),
      norm1Bias: view("norm1.bias"),
      conv1Weight: view("conv1.weight"),
      conv1Bias: view("conv1.bias"),
      norm2Weight: view("norm2.weight"),
      norm2Bias: view("norm2.bias"),
      conv2Weight: view("conv2.weight"),
      conv2Bias: view("conv2.bias"),
      ...(manifest.inChannels === manifest.outChannels
        ? {}
        : { shortcutWeight: view("nin_shortcut.weight"), shortcutBias: view("nin_shortcut.bias") }),
    };

    const got = h3EncoderResnetBlock(cfg, weights, f32(`${dir}/resnet-input.bin`));
    const want = f32(`${dir}/resnet-want.bin`);

    expect(got.length).toBe(want.length);
    let worst = 0;
    let sum = 0;
    for (let i = 0; i < want.length; i += 1) {
      const d = Math.abs(got[i]! - want[i]!);
      sum += d * d;
      if (d > worst) worst = d;
    }
    console.log(`h3 encoder resnet: worst ${worst.toExponential(3)}  rms ${Math.sqrt(sum / want.length).toExponential(3)}`);
    // Set from the measurement in the PR, not widened until green.
    expect(worst).toBeLessThan(1e-4);
  });

  if (!have) {
    it("says why the comparison did not run", () => {
      expect(dir === undefined || !existsSync(`${dir}/resnet.bin`)).toBe(true);
      console.log("h3 encoder: set H3_ENCODER_DIR to a directory written by tools/gen_resnet_golden.py");
    });
  }
});
