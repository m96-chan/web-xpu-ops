/**
 * The DiT block against diffusers' own `MiniMaxH3TransformerBlock`.
 *
 * Issue #200. The checkpoint ships no code for the DiT — it names a diffusers
 * class — so `tools/gen_block_golden.py` **imports** that class from diffusers'
 * main branch and runs block 0 on a fixed input. What this is held to is
 * upstream's arithmetic, not a reading of it (rule 7).
 *
 * **The weights are not in this repository.** Block 0 alone is ~1.7 GB at f32,
 * and the model is under a licence that is not this code's. `H3_DIT_DIR` points
 * at a directory the generator wrote; without it the comparison skips with a
 * message rather than passing.
 */
import { closeSync, existsSync, openSync, readFileSync, readSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { h3DitBlock, permuteChannelWeight, permuteProjectionForRope, type DitBlockConfig, type DitBlockWeights } from "./block.js";

const dir = process.env["H3_DIT_DIR"];
const have = dir !== undefined && existsSync(`${dir}/block.bin`) && existsSync(`${dir}/block.manifest.json`);

const f32 = (path: string): Float32Array => {
  const buffer = readFileSync(path);
  return new Float32Array(buffer.buffer, buffer.byteOffset, buffer.byteLength / 4);
};

interface Manifest {
  config: Record<string, number | number[]>;
  seq: number;
  modalityNum: number;
  adalnIndex: number;
  tensors: { name: string; offset: number; count: number }[];
  elements: number;
}

describe("h3 dit / block", () => {
  const maybe = have ? it : it.skip;

  maybe("reproduces the block diffusers' own implementation produced", () => {
    const manifest = JSON.parse(readFileSync(`${dir}/block.manifest.json`, "utf8")) as Manifest;
    // Read one tensor at a time. `block.bin` is 2.58 GB -- `adaln_proj` alone
    // projects into `6 * hidden * 3` and is 1.04 GB of it -- and `readFileSync`
    // refuses anything past 2 GiB.
    const fd = openSync(`${dir}/block.bin`, "r");
    const view = (name: string): Float32Array => {
      const entry = manifest.tensors.find((t) => t.name === name);
      if (!entry) throw new Error(`block.bin has no tensor named "${name}"`);
      const bytes = Buffer.allocUnsafe(entry.count * 4);
      const got = readSync(fd, bytes, 0, entry.count * 4, entry.offset * 4);
      if (got !== entry.count * 4) throw new Error(`${name}: read ${got} bytes where ${entry.count * 4} were wanted`);
      return new Float32Array(bytes.buffer, bytes.byteOffset, entry.count);
    };

    const c = manifest.config as unknown as {
      hidden_size: number; num_attention_heads: number; attention_head_dim: number;
      ffn_dim: number; time_embed_dim: number; norm_eps: number; qk_norm_eps: number;
      rope_freq_dim: number; rope_theta: number;
    };
    const cfg: DitBlockConfig = {
      hiddenSize: c.hidden_size,
      numHeads: c.num_attention_heads,
      headDim: c.attention_head_dim,
      ffnDim: c.ffn_dim,
      timeEmbedDim: c.time_embed_dim,
      seq: manifest.seq,
      normEps: c.norm_eps,
      qkNormEps: c.qk_norm_eps,
      ropeFreqDim: c.rope_freq_dim,
      ropeTheta: c.rope_theta,
      modalityNum: manifest.modalityNum,
    };

    // The permutation belongs in the weights — see `h3RopePermutation`. Q and K
    // only; V is never rotated and permuting it would reorder channels the
    // output projection reads in the original order.
    const rotDim = 2 * 3 * cfg.ropeFreqDim;
    const permute = (name: string): Float32Array =>
      permuteProjectionForRope(view(name), cfg.numHeads, cfg.headDim, rotDim, cfg.hiddenSize);

    const weights: DitBlockWeights = {
      norm1Weight: view("norm1.weight"),
      toQWeight: permute("attn.to_q.weight"),
      toKWeight: permute("attn.to_k.weight"),
      toVWeight: view("attn.to_v.weight"),
      // Permuted too: they are per-channel and index the channels `to_q` and
      // `to_k` produce, which the permutation above reordered.
      normQWeight: permuteChannelWeight(view("attn.norm_q.weight"), cfg.headDim, rotDim),
      normKWeight: permuteChannelWeight(view("attn.norm_k.weight"), cfg.headDim, rotDim),
      toOutWeight: view("attn.to_out.0.weight"),
      norm2Weight: view("norm2.weight"),
      ffProjWeight: view("ff.net.0.proj.weight"),
      ffOutWeight: view("ff.net.2.weight"),
      adalnWeight: view("adaln_proj.linear.weight"),
      adalnBias: view("adaln_proj.linear.bias"),
    };

    // `(t, h, w)` per token, whole numbers here — the DiT indexes a grid where
    // the visual VAE normalises to (-1, 1). The fourth axis stays at zero.
    const three = f32(`${dir}/positions.bin`);
    const positions = new Float32Array(manifest.seq * 4);
    for (let token = 0; token < manifest.seq; token += 1) {
      for (let axis = 0; axis < 3; axis += 1) positions[token * 4 + axis] = three[token * 3 + axis]!;
    }

    const got = h3DitBlock(cfg, weights, f32(`${dir}/hidden.bin`).slice(), f32(`${dir}/temb.bin`), manifest.adalnIndex, positions);
    closeSync(fd);
    const want = f32(`${dir}/want.bin`);

    expect(got.length).toBe(want.length);
    let worst = 0;
    let sum = 0;
    for (let i = 0; i < want.length; i += 1) {
      const d = Math.abs(got[i]! - want[i]!);
      sum += d * d;
      if (d > worst) worst = d;
    }
    console.log(`h3 dit block: worst ${worst.toExponential(3)}  rms ${Math.sqrt(sum / want.length).toExponential(3)}`);
    // Measured, and the number needs its two halves stated.
    //
    // Against the **f32** golden: worst **5.859e-3**, RMS 3.065e-4, on a block
    // output of order 350. Against an **f64** one — the same generator with
    // `--f64` — worst **1.953e-3** and RMS 4.353e-5, seven times tighter on the
    // RMS. So a third of the gap is torch's own f32 rounding over 5,376- and
    // 14,336-term dot products, and the rest is this port storing its
    // intermediates in `Float32Array` between ops, which is what a f32 port is:
    // the feed-forward turns inputs of order one into outputs of order a
    // thousand, where one f32 ulp is already 1e-4.
    //
    // 2e-2 is three times the worst measured. Every wrong convention tried
    // against it — the SwiGLU halves swapped, the modulation row misindexed,
    // the QK-norm weights left unpermuted — moves values by tens, four decades
    // above this.
    expect(worst).toBeLessThan(2e-2);
  });

  if (!have) {
    it("says why the comparison did not run", () => {
      expect(dir === undefined || !existsSync(`${dir}/block.bin`)).toBe(true);
      console.log("h3 dit: set H3_DIT_DIR to a directory written by tools/gen_block_golden.py");
    });
  }
});
