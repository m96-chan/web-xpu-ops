/**
 * A prompt in, a PNG out — Z-Image end to end, on this repository's ops.
 *
 * Issue #166 stages 4 and 5. Everything it runs has already been checked
 * against the model on its own:
 *
 *   | piece | file | against the model |
 *   | --- | --- | --- |
 *   | tokenizer | `llm/tokenizer-bpe.ts` | exact ids (#153) |
 *   | text encoder | `text-encoder.ts` | 6.197e-7 |
 *   | DiT | `dit-gpu.ts` | 4.386e-6 |
 *   | sampler | `sampler.ts` | exact schedule |
 *   | VAE decoder | `examples/zimage-vae` | 1.085e-5 |
 *
 * So a bad picture here is a wiring mistake, not an unknown: every part has a
 * number, and the parts are what the numbers are about.
 *
 *     npx tsx examples/zimage/src/generate.ts \
 *         --dit ~/zimage-q8 --prompt "a red apple on a wooden table" --size 256
 *
 * Writes `generated.png` and prints where the time went. It is not fast — see
 * the README — but it is a picture rather than a promise.
 */
import { existsSync, readFileSync, readdirSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { createRunner } from "../../../harness/wgsl.js";
import { type BpeVocab, ByteLevelBpeTokenizer } from "../../../llm/tokenizer-bpe.js";
import { type DecoderConfig } from "../../zimage-vae/src/decoder.js";
import { decodeGpu } from "../../zimage-vae/src/decoder-gpu.js";
import { encodePng } from "../../zimage-vae/src/render.js";
import { type DitConfig } from "./dit.js";
import { ditForwardGpu } from "./dit-gpu.js";
import { ditKernels } from "./kernels-node.js";
import {
  DEFAULT_FLOW_SHIFT,
  DEFAULT_STEPS,
  NUM_TRAIN_TIMESTEPS,
  eulerStep,
  flowSchedule,
  modelTimestep,
} from "./sampler.js";
import { ShardedSafetensors, SafetensorsFile } from "./safetensors.js";
import {
  type Qwen3Config,
  type Qwen3LayerWeights,
  permuteLayerForRope,
  qwen3Encode,
} from "./text-encoder.js";
import { LazyDitWeights } from "./weights-node.js";

function arg(flag: string, fallback: string): string {
  const at = process.argv.indexOf(flag);
  return at >= 0 && process.argv[at + 1] !== undefined ? process.argv[at + 1]! : fallback;
}

/** The same resolution `tools/models.py` does. */
function modelRoot(): string {
  const explicit = process.env.ZIMAGE_MODEL_DIR;
  if (explicit) return explicit;
  const hub = join(process.env.HF_HOME ?? join(homedir(), ".cache", "huggingface"), "hub");
  const repo = join(hub, "models--Tongyi-MAI--Z-Image", "snapshots");
  if (!existsSync(repo)) {
    throw new Error("No Z-Image checkpoint found. Set ZIMAGE_MODEL_DIR, or fetch it with tools/models.py.");
  }
  return join(repo, readdirSync(repo)[0]!);
}

/**
 * A normal-ish sample from a seeded generator.
 *
 * Not `Math.random()`: the same seed has to give the same picture, which is the
 * only way a change to any of the five parts above can be told from noise.
 * Box-Muller over a small LCG — the distribution is what matters here, not the
 * generator's statistical pedigree.
 */
function seededNoise(length: number, seed: number): Float32Array {
  let state = (seed >>> 0) || 1;
  const next = (): number => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return (state >>> 8) / 16777216;
  };
  const out = new Float32Array(length);
  for (let i = 0; i < length; i += 2) {
    const u = Math.max(next(), 1e-12);
    const v = next();
    const r = Math.sqrt(-2 * Math.log(u));
    out[i] = r * Math.cos(2 * Math.PI * v);
    if (i + 1 < length) out[i + 1] = r * Math.sin(2 * Math.PI * v);
  }
  return out;
}

async function main(): Promise<void> {
  const ditDir = arg("--dit", process.env.ZIMAGE_DIT_DIR ?? "");
  if (!ditDir) {
    console.error("generate: pass --dit <dir>, the output of tools/convert_dit.py.");
    process.exit(2);
  }
  const prompt = arg("--prompt", "a red apple on a wooden table");
  const size = Number(arg("--size", "256"));
  const steps = Number(arg("--steps", String(DEFAULT_STEPS)));
  const shift = Number(arg("--shift", String(DEFAULT_FLOW_SHIFT)));
  const seed = Number(arg("--seed", "0"));
  const outPath = arg("--out", "generated.png");

  const root = modelRoot();
  const runner = await createRunner();
  if (!runner) {
    console.error("generate: no WebGPU adapter available.");
    process.exit(2);
  }
  const run = runner.run;
  const K = ditKernels();
  const started = Date.now();
  const mark = (label: string, from: number): number => {
    console.log(`  ${label.padEnd(18)} ${((Date.now() - from) / 1000).toFixed(1)}s`);
    return Date.now();
  };
  let at = started;

  // --- 1. tokenize, with the chat template upstream uses ---
  const vocab = JSON.parse(
    readFileSync(fileURLToPath(new URL("../../../llm/data/qwen-qwen3-4b.bpe-vocab.json", import.meta.url)), "utf8"),
  ) as BpeVocab;
  const tokenizer = new ByteLevelBpeTokenizer(vocab);
  // `apply_chat_template(..., add_generation_prompt=True, enable_thinking=True)`
  // expands to exactly this for a single user turn — checked against the
  // formatted string in `fixtures/text-encoder.manifest.json`.
  const formatted = `<|im_start|>user\n${prompt}<|im_end|>\n<|im_start|>assistant\n`;
  const ids = Int32Array.from(tokenizer.encode(formatted));
  console.log(`prompt: ${JSON.stringify(prompt)} -> ${ids.length} tokens`);
  at = mark("tokenize", at);

  // --- 2. text encoder ---
  const encoderDir = join(root, "text_encoder");
  const indexPath = join(encoderDir, "model.safetensors.index.json");
  let readEncoder: (name: string) => Float32Array;
  let closeEncoder: () => void;
  if (existsSync(indexPath)) {
    const map = (JSON.parse(readFileSync(indexPath, "utf8")) as { weight_map: Record<string, string> }).weight_map;
    const shards = new ShardedSafetensors(encoderDir, map);
    readEncoder = (name) => shards.read(name);
    closeEncoder = () => shards.close();
  } else {
    const file = new SafetensorsFile(join(encoderDir, "model.safetensors"));
    readEncoder = (name) => file.read(name);
    closeEncoder = () => file.close();
  }
  const encoderConfig = JSON.parse(readFileSync(join(encoderDir, "config.json"), "utf8")) as Record<string, number>;
  const qwenCfg: Qwen3Config = {
    hiddenSize: encoderConfig.hidden_size!,
    numLayers: encoderConfig.num_hidden_layers!,
    numHeads: encoderConfig.num_attention_heads!,
    numKvHeads: encoderConfig.num_key_value_heads!,
    headDim: encoderConfig.head_dim!,
    ffnHidden: encoderConfig.intermediate_size!,
    rmsNormEps: encoderConfig.rms_norm_eps!,
    ropeTheta: encoderConfig.rope_theta!,
    vocabSize: encoderConfig.vocab_size!,
    // `hidden_states[-2]`, measured — see `fixtures/text-encoder.manifest.json`.
    stopAfterLayer: encoderConfig.num_hidden_layers! - 2,
  };
  const capFeats = qwen3Encode(qwenCfg, {
    numLayers: qwenCfg.numLayers,
    embed: (wanted) => {
      const table = readEncoder("model.embed_tokens.weight");
      const out = new Float32Array(wanted.length * qwenCfg.hiddenSize);
      for (let i = 0; i < wanted.length; i += 1) {
        out.set(
          table.subarray(wanted[i]! * qwenCfg.hiddenSize, (wanted[i]! + 1) * qwenCfg.hiddenSize),
          i * qwenCfg.hiddenSize,
        );
      }
      return out;
    },
    layer: (index) => {
      const p = `model.layers.${index}.`;
      const raw: Qwen3LayerWeights = {
        input_layernorm: readEncoder(`${p}input_layernorm.weight`),
        q_proj: readEncoder(`${p}self_attn.q_proj.weight`),
        k_proj: readEncoder(`${p}self_attn.k_proj.weight`),
        v_proj: readEncoder(`${p}self_attn.v_proj.weight`),
        o_proj: readEncoder(`${p}self_attn.o_proj.weight`),
        q_norm: readEncoder(`${p}self_attn.q_norm.weight`),
        k_norm: readEncoder(`${p}self_attn.k_norm.weight`),
        post_attention_layernorm: readEncoder(`${p}post_attention_layernorm.weight`),
        gate_proj: readEncoder(`${p}mlp.gate_proj.weight`),
        up_proj: readEncoder(`${p}mlp.up_proj.weight`),
        down_proj: readEncoder(`${p}mlp.down_proj.weight`),
      };
      return permuteLayerForRope(raw, qwenCfg);
    },
  }, ids);
  closeEncoder();
  at = mark("text encoder", at);

  // --- 3. denoise ---
  const weights = new LazyDitWeights(ditDir);
  const ditConfigRaw = weights.config as Record<string, number | number[]>;
  const patchSize = (ditConfigRaw.all_patch_size as number[])[0]!;
  const ditCfg: DitConfig = {
    dim: ditConfigRaw.dim as number,
    nHeads: ditConfigRaw.n_heads as number,
    nLayers: ditConfigRaw.n_layers as number,
    nRefinerLayers: ditConfigRaw.n_refiner_layers as number,
    inChannels: ditConfigRaw.in_channels as number,
    patchSize,
    capFeatDim: ditConfigRaw.cap_feat_dim as number,
    normEps: ditConfigRaw.norm_eps as number,
    ropeAxesDims: ditConfigRaw.axes_dims as number[],
    ropeTheta: ditConfigRaw.rope_theta as number,
    tScale: ditConfigRaw.t_scale as number,
    adalnEmbedDim: 256,
    frequencyEmbeddingSize: 256,
    maxPeriod: 10000,
  };

  // `vae_scale = ZIMAGE_VAE_SCALE_FACTOR * 2`, then `2 * (size // vae_scale)` —
  // upstream's own arithmetic, which lands on size/8 rounded so the latent is
  // divisible by 16.
  const latentSide = 2 * Math.floor(size / (8 * 2));
  const latentLen = ditCfg.inChannels * latentSide * latentSide;
  let latents = seededNoise(latentLen, seed);
  const { timesteps, sigmas } = flowSchedule(steps, shift, NUM_TRAIN_TIMESTEPS);
  const capMask = new Float32Array(ids.length).fill(1);

  console.log(
    `denoising ${steps} steps at ${size}x${size} ` +
      `(latent ${latentSide}x${latentSide}, ${(latentSide / patchSize) ** 2} image tokens)`,
  );
  for (let step = 0; step < steps; step += 1) {
    const stepStart = Date.now();
    const velocity = await ditForwardGpu(run, K, ditCfg, weights, {
      latent: latents,
      F: 1,
      H: latentSide,
      W: latentSide,
      t: modelTimestep(timesteps[step]!, NUM_TRAIN_TIMESTEPS),
      capFeats,
      capMask,
    });
    // Negated: Z-Image predicts negative noise (`zimage_generate_image.py:604`).
    const negated = new Float32Array(velocity.length);
    for (let i = 0; i < velocity.length; i += 1) negated[i] = -velocity[i]!;
    latents = eulerStep(latents, negated, sigmas, step);
    console.log(`  step ${step + 1}/${steps}  ${((Date.now() - stepStart) / 1000).toFixed(1)}s`);
  }
  at = mark("denoise", at);

  // --- 4. VAE decode ---
  const vaeDir = new URL("../../zimage-vae/fixtures-small/", import.meta.url);
  const vaeManifest = JSON.parse(
    readFileSync(fileURLToPath(new URL("manifest.json", vaeDir)), "utf8"),
  ) as { config: Record<string, number | number[]>; decoder: { name: string; offset: number; length: number }[] };
  const vaeRaw = readFileSync(fileURLToPath(new URL("decoder.bin", vaeDir)));
  const vaeBlob = new Float32Array(vaeRaw.buffer, vaeRaw.byteOffset, vaeRaw.byteLength / 4);
  const vaeWeight = (name: string): Float32Array => {
    const e = vaeManifest.decoder.find((t) => t.name === name);
    if (!e) throw new Error(`VAE fixture has no "${name}" — regenerate with tools/gen_latent.py`);
    return vaeBlob.subarray(e.offset, e.offset + e.length);
  };
  const vc = vaeManifest.config as unknown as {
    block_out_channels: number[];
    layers_per_block: number;
    norm_num_groups: number;
    latent_channels: number;
    out_channels: number;
    scaling_factor: number;
    shift_factor: number;
  };
  const decoderCfg: DecoderConfig = {
    blockOutChannels: vc.block_out_channels,
    layersPerBlock: vc.layers_per_block,
    normNumGroups: vc.norm_num_groups,
    latentChannels: vc.latent_channels,
    outChannels: vc.out_channels,
    scalingFactor: vc.scaling_factor,
    shiftFactor: vc.shift_factor,
  };
  // `decodeGpu` applies `latent / scalingFactor + shiftFactor` itself, which is
  // `shift_scale_latents_for_decode` — so the latent goes in as the sampler
  // left it, not pre-scaled. Doing both would double the shift and give an
  // image with the right structure and the wrong colours.
  const image = await decodeGpu(run, decoderCfg, vaeWeight, latents, latentSide, latentSide);
  at = mark("vae decode", at);

  writeFileSync(outPath, encodePng(image.data, image.H, image.W));
  console.log(`\nwrote ${outPath} (${image.W}x${image.H}) in ${((Date.now() - started) / 1000).toFixed(1)}s total`);
}

await main();
process.exit(0);
