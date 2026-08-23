import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { agree } from "../../../harness/agree.js";
import { type BlockConfig, type BlockWeights, zimageBlock } from "./block.js";

/**
 * Issue #163: does this repository's op set actually compose into Z-Image's
 * transformer block?
 *
 * The golden comes from Z-Image's own module — `tools/gen_block_golden.py`
 * imports `ZImageTransformerBlock` from musubi-tuner's copy of the Tongyi-MAI
 * implementation and runs it, rather than reimplementing it in the generator.
 * A transcription mistake in a hand-written generator would produce a golden
 * that agrees with a wrong port, which is the failure this arrangement exists
 * to rule out.
 *
 * What a pass means: every op named in `block.ts` lines up — RMSNorm's eps
 * placement, QK-Norm's axis, three-axis RoPE's channel split and adjacent-pair
 * convention, MHA's scale, SwiGLU's operand order, the `1.0 +` on the scales,
 * `tanh` on the gates, and both residuals. What it does not mean: that the
 * whole model runs, that any of it runs on a GPU, or that it is fast. Those
 * are #148's job.
 */

const fixtures = new URL("../fixtures/", import.meta.url);

interface Manifest {
  config: {
    dim: number;
    nHeads: number;
    headDim: number;
    seq: number;
    normEps: number;
    ffnHidden: number;
    ropeAxesDims: number[];
    ropeTheta: number;
  };
  ids: number[][];
  tensors: { name: string; shape: number[]; offset: number; length: number }[];
}

function load(): { manifest: Manifest; get: (name: string) => Float32Array } {
  const manifest = JSON.parse(readFileSync(fileURLToPath(new URL("block.manifest.json", fixtures)), "utf8")) as Manifest;
  const raw = readFileSync(fileURLToPath(new URL("block.bin", fixtures)));
  const all = new Float32Array(raw.buffer, raw.byteOffset, raw.byteLength / 4);
  const byName = new Map(manifest.tensors.map((t) => [t.name, t]));
  return {
    manifest,
    get(name) {
      const t = byName.get(name);
      if (!t) throw new Error(`golden has no tensor "${name}" — regenerate with tools/gen_block_golden.py`);
      return all.subarray(t.offset, t.offset + t.length);
    },
  };
}

describe("Z-Image transformer block, composed from ops (#163)", () => {
  const { manifest, get } = load();
  const cfg: BlockConfig = { ...manifest.config };

  const weights = {} as BlockWeights;
  for (const key of [
    "attention_to_q_weight", "attention_to_k_weight", "attention_to_v_weight", "attention_to_out_0_weight",
    "attention_norm_q_weight", "attention_norm_k_weight",
    "feed_forward_w1_weight", "feed_forward_w2_weight", "feed_forward_w3_weight",
    "attention_norm1_weight", "ffn_norm1_weight", "attention_norm2_weight", "ffn_norm2_weight",
    "adaLN_modulation_0_weight", "adaLN_modulation_0_bias",
  ] as const) {
    weights[key] = get(key);
  }

  const positions = Int32Array.from(manifest.ids.flat());

  it("matches Z-Image's own block on the same weights and input", () => {
    const got = zimageBlock(cfg, weights, get("x"), get("adalnInput"), positions);
    const want = get("output");

    expect(got.length).toBe(want.length);

    // Tolerance measured, not widened until green (rule 7). Instrumented with
    // the tolerance forced to zero, the worst of the 384 outputs differs by
    // **5.96e-8 absolute / 1.05e-6 relative** — one f32 ulp at this magnitude,
    // from summation order across the 170-wide FFN and 64-wide projections,
    // not from different arithmetic.
    //
    // `abs: 1e-6` clears that by ~17x, `rel: 1e-4` by ~95x. Both sit far below
    // what a real divergence produces: dropping one `1.0 +` on the scales was
    // measured at **3.43e-2 absolute / 4.41 relative** — six orders of
    // magnitude above the bound, so the size of the margin is not what decides
    // whether a mistake is caught.
    expect(agree(got, want, { rel: 1e-4, abs: 1e-6 })).toBeNull();
  });

  it("is sensitive to the modulation it applies", () => {
    // A block whose gates are ignored still produces a plausible tensor of the
    // right shape, which is exactly the failure mode a shape-only check misses.
    // Zeroing the adaLN input changes the answer, so the path is observed.
    const neutral = zimageBlock(cfg, weights, get("x"), new Float32Array(get("adalnInput").length), positions);
    expect(agree(neutral, get("output"), { rel: 1e-5, abs: 1e-5 })).not.toBeNull();
  });
});
