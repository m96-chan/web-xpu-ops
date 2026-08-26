/**
 * The block against MiniMax-H3's own output.
 *
 * Issue #200. `tools/gen_block_golden.py` **imports** `TransformerBlock` and
 * `RotaryEmbeddingND` from the bundle the checkpoint ships and runs block 0 on
 * a fixed input, so what this is held to is the publisher's arithmetic rather
 * than a reading of it (rule 7).
 *
 * **The weights are not in this repository.** Block 0 alone is 268 MB, and the
 * model is under a licence that is not this code's. `H3_VIDEO_DIR` points at a
 * directory the generator wrote; without it the comparison skips with a message
 * rather than passing.
 */
import { existsSync, readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { h3VideoBlock, permuteQkvForRope, type H3BlockConfig, type H3BlockWeights } from "./block.js";

const dir = process.env["H3_VIDEO_DIR"];
const have = dir !== undefined && existsSync(`${dir}/block.bin`) && existsSync(`${dir}/block.manifest.json`);

const f32 = (path: string): Float32Array => {
  const buffer = readFileSync(path);
  return new Float32Array(buffer.buffer, buffer.byteOffset, buffer.byteLength / 4);
};

interface Manifest {
  config: Record<string, number | string | boolean>;
  dim: number;
  ffnHidden: number;
  ropeApplyDim: number;
  numPatches: number;
  numSuffix: number;
  tensors: { name: string; shape: number[]; offset: number; count: number }[];
  elements: number;
}

describe("h3 video vae / block", () => {
  const maybe = have ? it : it.skip;

  maybe("reproduces the block the model's own code produced", () => {
    const manifest = JSON.parse(readFileSync(`${dir}/block.manifest.json`, "utf8")) as Manifest;
    const flat = f32(`${dir}/block.bin`);
    expect(flat.length).toBe(manifest.elements);

    const view = (name: string): Float32Array => {
      const entry = manifest.tensors.find((t) => t.name === name);
      // Named rather than undefined: a missing weight otherwise becomes a
      // zero-length multiply three layers down.
      if (!entry) throw new Error(`block.bin has no tensor named "${name}"`);
      return flat.subarray(entry.offset, entry.offset + entry.count);
    };

    const dim = manifest.dim;
    const nHeads = manifest.config["heads"] as number;
    const headDim = manifest.config["dim_head"] as number;
    const seq = manifest.numPatches + manifest.numSuffix;

    // The rope permutation goes into the weights, once — the same thing
    // `permuteForRope` does for Anima, and the reason the run-time path has no
    // shuffle in it.
    const permuted = permuteQkvForRope(view("attn.to_qkv.weight"), view("attn.to_qkv.bias"), nHeads, headDim, dim);

    const cfg: H3BlockConfig = {
      dim,
      nHeads,
      headDim,
      seq,
      ffnHidden: manifest.ffnHidden,
      normEps: manifest.config["eps"] as number,
      // Three real axes plus one pinned at position zero, which is the identity
      // and is how `rope_dim_ratio: 0.75` is covered without a second path.
      ropeAxesDims: [16, 16, 16, 16],
      ropeTheta: manifest.config["rope_theta"] as number,
    };
    const weights: H3BlockWeights = {
      norm1Weight: view("norm1.weight"),
      toQkvWeight: permuted.weight,
      toQkvBias: permuted.bias,
      toOutWeight: view("attn.to_out.weight"),
      toOutBias: view("attn.to_out.bias"),
      scale1: view("scale1"),
      norm2Weight: view("norm2.weight"),
      w1Weight: view("ff.w1.weight"),
      w1Bias: view("ff.w1.bias"),
      w2Weight: view("ff.w2.weight"),
      w2Bias: view("ff.w2.bias"),
      scale2: view("scale2"),
    };

    // `[N, 3]` angles from the generator, widened to the four axes `ropeAxes`
    // wants with the fourth left at zero.
    const three = f32(`${dir}/block-positions.bin`);
    const positions = new Float32Array(seq * 4);
    for (let token = 0; token < seq; token += 1) {
      for (let axis = 0; axis < 3; axis += 1) positions[token * 4 + axis] = three[token * 3 + axis]!;
    }

    const got = h3VideoBlock(cfg, weights, f32(`${dir}/block-input.bin`).slice(), positions);
    const want = f32(`${dir}/block-want.bin`);

    expect(got.length).toBe(want.length);
    let worst = 0;
    let sum = 0;
    for (let i = 0; i < want.length; i += 1) {
      const d = Math.abs(got[i]! - want[i]!);
      sum += d * d;
      if (d > worst) worst = d;
    }
    console.log(`h3 video block: worst ${worst.toExponential(3)}  rms ${Math.sqrt(sum / want.length).toExponential(3)}`);
    // Measured: worst element **1.192e-7**, RMS **1.773e-9**, over 29 tokens of
    // 2048 channels — f32 rounding on a block whose residual stream is order 1.
    // 1e-6 is eight times that. Every wrong convention tried against it moves
    // values by 1e-1 or more: the QKV heads read as three blocks instead of
    // interleaved, the SwiGLU halves swapped, LayerScale dropped, the rope
    // permutation left out.
    expect(worst).toBeLessThan(1e-6);
  });

  if (!have) {
    it("says why the comparison did not run", () => {
      expect(dir === undefined || !existsSync(`${dir}/block.bin`)).toBe(true);
      console.log(
        "h3 video: set H3_VIDEO_DIR to a directory written by tools/gen_block_golden.py to compare " +
          "against the model's own block",
      );
    });
  }
});
