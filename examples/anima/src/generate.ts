/**
 * Anima-3.8B, prompt to latent, on the GPU.
 *
 * Issue #170 stage 5. Everything this joins is checked against ComfyUI's own
 * model independently first — a pipeline whose parts are each verified can
 * still be wired wrongly, but a pipeline whose parts are not verified cannot be
 * debugged at all:
 *
 * | stage | checked by | agreement |
 * | --- | --- | --- |
 * | tokenizers | `tokenize.test.ts` | exact ids, 23 cases each |
 * | encoder + adapter | `verify-encoder.ts` | 9.775e-7 |
 * | DiT, resident | `verify-forward-gpu.ts` | 1.182e-5 |
 * | schedule and stepper | `sampler.test.ts` | 2e-6 on the trajectory |
 *
 * **There is no VAE yet — issue #174.** Anima decodes with Wan 2.1's 3D causal
 * VAE, which shares nothing with `examples/zimage-vae`'s 2D one:
 * `CausalConv3d`, `RMS_norm` over channels, temporal upsampling. Until it
 * exists this writes the latent out, and renders the 16-to-3 projection
 * `latent_formats.Wan21.latent_rgb_factors` provides — what ComfyUI itself
 * shows as a live preview while sampling.
 *
 * **The preview is not the image the model makes.** It is a linear map at one
 * eighth the resolution, against a decoder with 194 tensors. It shows that a
 * latent has structure, composition and colour, and nothing finer. Judging the
 * model by it would be judging the projection.
 *
 *     npx tsx examples/anima/src/generate.ts \
 *       --encoder ~/anima-src/qwen_3_06b_base.safetensors --dit ~/anima-q8 \
 *       --prompt "1girl, silver hair" --steps 40 --cfg 8
 */
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { createResidentDevice } from "../../../harness/resident.js";
import { encodePng } from "../../zimage-vae/src/render.js";
import type { BpeVocab } from "../../../llm/tokenizer-bpe.js";
import { qwen3Encode, type Qwen3LayerWeights, type Qwen3Weights, permuteLayerForRope } from "../../zimage/src/text-encoder.js";
import { SafetensorsFile } from "../../zimage/src/safetensors.js";
import type { AnimaConfig, AnimaInput } from "./dit.js";
import { animaForwardResident, releaseAnimaWeights, type AnimaWeightSource } from "./dit-resident.js";
import { ditKernels } from "../../zimage/src/kernels-node.js";
import {
  ADAPTER,
  type AdapterBlockWeights,
  type AdapterWeights,
  QWEN3_06B,
  llmAdapterForward,
  padContext,
  permuteAdapterBlock,
} from "./text-encoder.js";
import {
  DEFAULTS,
  LATENT,
  applyCfg,
  betaSchedule,
  calculateDenoised,
  cfgEnabled,
  flowSigmas,
  latentToVae,
  noiseScaling,
  resMultistep,
  timestepOf,
} from "./sampler.js";
import { type T5Vocab, animaTokenizers, tokenizePrompt } from "./tokenize.js";
import { loadAnimaSubset, withRopePermutation } from "./weights-node.js";

const arg = (name: string): string | undefined => {
  const at = process.argv.indexOf(name);
  return at >= 0 ? process.argv[at + 1] : undefined;
};
const number = (name: string, fallback: number): number => {
  const raw = arg(name);
  if (raw === undefined) return fallback;
  const value = Number(raw);
  if (!Number.isFinite(value)) throw new Error(`${name}: expected a number, got ${JSON.stringify(raw)}`);
  return value;
};

const encoderPath = arg("--encoder") ?? process.env.ANIMA_ENCODER;
const ditDir = arg("--dit") ?? process.env.ANIMA_DIT_DIR;
if (!encoderPath || !ditDir) {
  console.error(
    "generate: pass --encoder <qwen_3_06b_base.safetensors> --dit <convert_dit.py output dir>\n" +
      "  [--prompt <text>] [--negative <text>] [--steps N] [--cfg N] [--width N] [--height N] [--seed N] [--out path]",
  );
  process.exit(2);
}

const prompt = arg("--prompt") ?? "1girl, silver hair, red eyes, looking at viewer, detailed background";
// The released workflow's negative is empty, and CFG against an empty prompt is
// not the same as CFG against zeros — the adapter turns "" into a real context
// (one `</s>`), which is what the model was sampled with.
const negative = arg("--negative") ?? "";
const steps = number("--steps", DEFAULTS.steps);
const guidance = number("--cfg", DEFAULTS.guidance);
const width = number("--width", DEFAULTS.width);
const height = number("--height", DEFAULTS.height);
const seed = number("--seed", 0);
const outPath = arg("--out") ?? "anima-out";

/**
 * The VAE's spatial stride. Wan 2.1 downsamples by 8 in H and W, so a 832x1216
 * image is a 104x152 latent — and both have to be multiples of 8 for the DiT's
 * patch size of 2 to divide the result.
 */
const VAE_STRIDE = 8;
const latentH = Math.round(height / VAE_STRIDE);
const latentW = Math.round(width / VAE_STRIDE);
if (latentH * VAE_STRIDE !== height || latentW * VAE_STRIDE !== width) {
  throw new Error(`generate: ${width}x${height} is not a multiple of ${VAE_STRIDE}`);
}
if (latentH % 2 !== 0 || latentW % 2 !== 0) {
  throw new Error(
    `generate: ${width}x${height} gives a ${latentW}x${latentH} latent, which the DiT's patch size of 2 does not divide. ` +
      `Use a multiple of ${VAE_STRIDE * 2}.`,
  );
}

/** xorshift128+, so a seed reproduces a run without pulling in a dependency. */
function gaussianNoise(count: number, seedValue: number): Float32Array {
  let s0 = (seedValue ^ 0x9e3779b9) >>> 0 || 1;
  let s1 = (seedValue * 0x85ebca6b + 0xc2b2ae35) >>> 0 || 2;
  const next = (): number => {
    let x = s0;
    const y = s1;
    s0 = y;
    x ^= x << 23;
    x ^= x >>> 17;
    x ^= y ^ (y >>> 26);
    s1 = x >>> 0;
    return ((s0 + s1) >>> 0) / 4294967296;
  };
  const out = new Float32Array(count);
  // Box-Muller. Not torch's Philox, so this does **not** reproduce ComfyUI's
  // image for a given seed — only this port's own, run to run. Matching torch's
  // generator is a separate piece of work and would be the only way to compare
  // images rather than tensors.
  for (let i = 0; i < count; i += 2) {
    const u = Math.max(next(), Number.MIN_VALUE);
    const v = next();
    const r = Math.sqrt(-2 * Math.log(u));
    out[i] = r * Math.cos(2 * Math.PI * v);
    if (i + 1 < count) out[i + 1] = r * Math.sin(2 * Math.PI * v);
  }
  return out;
}

// --- weights ---
console.log(`reading the encoder from ${encoderPath}`);
const encoderFile = new SafetensorsFile(encoderPath);
const layerCache = new Map<number, Qwen3LayerWeights>();
const embedTable = encoderFile.read("model.embed_tokens.weight");
const encoderWeights: Qwen3Weights = {
  numLayers: QWEN3_06B.numLayers,
  embed(ids) {
    const rows = new Float32Array(ids.length * QWEN3_06B.hiddenSize);
    for (let i = 0; i < ids.length; i += 1) {
      rows.set(embedTable.subarray(ids[i]! * QWEN3_06B.hiddenSize, (ids[i]! + 1) * QWEN3_06B.hiddenSize), i * QWEN3_06B.hiddenSize);
    }
    return rows;
  },
  layer(index) {
    const cached = layerCache.get(index);
    if (cached) return cached;
    const p = `model.layers.${index}.`;
    const built = permuteLayerForRope({
      input_layernorm: encoderFile.read(`${p}input_layernorm.weight`),
      q_proj: encoderFile.read(`${p}self_attn.q_proj.weight`),
      k_proj: encoderFile.read(`${p}self_attn.k_proj.weight`),
      v_proj: encoderFile.read(`${p}self_attn.v_proj.weight`),
      o_proj: encoderFile.read(`${p}self_attn.o_proj.weight`),
      q_norm: encoderFile.read(`${p}self_attn.q_norm.weight`),
      k_norm: encoderFile.read(`${p}self_attn.k_norm.weight`),
      post_attention_layernorm: encoderFile.read(`${p}post_attention_layernorm.weight`),
      gate_proj: encoderFile.read(`${p}mlp.gate_proj.weight`),
      up_proj: encoderFile.read(`${p}mlp.up_proj.weight`),
      down_proj: encoderFile.read(`${p}mlp.down_proj.weight`),
    }, QWEN3_06B);
    layerCache.set(index, built);
    return built;
  },
  finalNorm: encoderFile.read("model.norm.weight"),
};

console.log(`reading the DiT from ${ditDir}`);
const dit = loadAnimaSubset(ditDir);
const blockCache = new Map<number, AdapterBlockWeights>();
const adapterWeights: AdapterWeights = {
  embed(ids) {
    const table = dit.get("net.llm_adapter.embed.weight");
    const rows = new Float32Array(ids.length * ADAPTER.targetDim);
    for (let i = 0; i < ids.length; i += 1) {
      rows.set(table.subarray(ids[i]! * ADAPTER.targetDim, (ids[i]! + 1) * ADAPTER.targetDim), i * ADAPTER.targetDim);
    }
    return rows;
  },
  block(index) {
    const cached = blockCache.get(index);
    if (cached) return cached;
    const p = `net.llm_adapter.blocks.${index}.`;
    const built = permuteAdapterBlock({
      norm_self_attn: dit.get(`${p}norm_self_attn.weight`),
      self_attn_q_proj: dit.get(`${p}self_attn.q_proj.weight`),
      self_attn_k_proj: dit.get(`${p}self_attn.k_proj.weight`),
      self_attn_v_proj: dit.get(`${p}self_attn.v_proj.weight`),
      self_attn_o_proj: dit.get(`${p}self_attn.o_proj.weight`),
      self_attn_q_norm: dit.get(`${p}self_attn.q_norm.weight`),
      self_attn_k_norm: dit.get(`${p}self_attn.k_norm.weight`),
      norm_cross_attn: dit.get(`${p}norm_cross_attn.weight`),
      cross_attn_q_proj: dit.get(`${p}cross_attn.q_proj.weight`),
      cross_attn_k_proj: dit.get(`${p}cross_attn.k_proj.weight`),
      cross_attn_v_proj: dit.get(`${p}cross_attn.v_proj.weight`),
      cross_attn_o_proj: dit.get(`${p}cross_attn.o_proj.weight`),
      cross_attn_q_norm: dit.get(`${p}cross_attn.q_norm.weight`),
      cross_attn_k_norm: dit.get(`${p}cross_attn.k_norm.weight`),
      norm_mlp: dit.get(`${p}norm_mlp.weight`),
      mlp_0_weight: dit.get(`${p}mlp.0.weight`),
      mlp_0_bias: dit.get(`${p}mlp.0.bias`),
      mlp_2_weight: dit.get(`${p}mlp.2.weight`),
      mlp_2_bias: dit.get(`${p}mlp.2.bias`),
    });
    blockCache.set(index, built);
    return built;
  },
  out_proj: dit.get("net.llm_adapter.out_proj.weight"),
  out_proj_bias: dit.get("net.llm_adapter.out_proj.bias"),
  norm: dit.get("net.llm_adapter.norm.weight"),
};

const fixtures = new URL("../fixtures/", import.meta.url);
const readJson = (url: URL): unknown => JSON.parse(readFileSync(fileURLToPath(url), "utf8"));
const tokenizers = animaTokenizers(
  readJson(new URL("../../../llm/data/qwen-qwen3-4b.bpe-vocab.json", import.meta.url)) as BpeVocab,
  readJson(new URL("t5.unigram-vocab.json", fixtures)) as T5Vocab,
);

// --- conditioning ---
/** Prompt to the `[512, 1024]` the DiT cross-attends to. */
function conditioning(text: string): Float32Array {
  const { qwenIds, t5Ids, t5Weights } = tokenizePrompt(tokenizers, text);
  // An empty prompt still has a `</s>`, so the encoder never sees zero tokens.
  const source = qwen3Encode(QWEN3_06B, encoderWeights, qwenIds);
  const context = llmAdapterForward(adapterWeights, source, qwenIds.length, t5Ids);
  return padContext(context, t5Ids.length, t5Weights);
}

const started = Date.now();
console.log(`\nprompt: ${JSON.stringify(prompt)}`);
const positive = conditioning(prompt);
const useCfg = cfgEnabled(guidance);
const unconditional = useCfg ? conditioning(negative) : null;
console.log(
  `  conditioned in ${((Date.now() - started) / 1000).toFixed(1)}s` +
    `${useCfg ? `, plus an unconditional pass for CFG ${guidance}` : `, no CFG (${guidance} <= 1)`}`,
);

// --- the DiT, resident ---
/**
 * The model's shape, out of the manifest `convert_dit.py` writes — which reads
 * it with ComfyUI's `detect_unet_config` rather than transcribing it.
 *
 * Not out of a golden. `verify-forward-gpu.ts` takes it from the fixture it
 * compares against, which is right for a checker and wrong here: a generator
 * that reads its configuration from a test fixture is one edit away from
 * running a different model than the one it loaded.
 */
const manifest = JSON.parse(readFileSync(`${ditDir}/dit.manifest.json`, "utf8")) as {
  config?: Record<string, number | boolean | string>;
};
if (!manifest.config) {
  throw new Error(
    `${ditDir}/dit.manifest.json carries no "config" — it predates convert_dit.py recording one. Re-run the conversion.`,
  );
}
const c = manifest.config as unknown as {
  num_blocks: number; model_channels: number; num_heads: number; adaln_lora_dim: number;
  in_channels: number; out_channels: number; patch_spatial: number; patch_temporal: number;
  crossattn_emb_channels: number; concat_padding_mask: boolean;
  rope_t_extrapolation_ratio: number; rope_h_extrapolation_ratio: number; rope_w_extrapolation_ratio: number;
};
const cfg: AnimaConfig = {
  numBlocks: c.num_blocks,
  modelChannels: c.model_channels,
  numHeads: c.num_heads,
  adalnLoraDim: c.adaln_lora_dim,
  inChannels: c.in_channels,
  outChannels: c.out_channels,
  patchSpatial: c.patch_spatial,
  patchTemporal: c.patch_temporal,
  crossattnEmbChannels: c.crossattn_emb_channels,
  concatPaddingMask: c.concat_padding_mask,
  maxPeriod: 10000,
  normEps: 1e-6,
  ropeExtrapolation: {
    t: c.rope_t_extrapolation_ratio,
    h: c.rope_h_extrapolation_ratio,
    w: c.rope_w_extrapolation_ratio,
  },
};
const device = await createResidentDevice();
const held = new Map<string, GPUBuffer>();
// The self-attention rope permutation, which the DiT expects and which the
// adapter's weights above must *not* get. See `withRopePermutation`.
const gpuWeights: AnimaWeightSource = withRopePermutation(
  dit,
  cfg.numHeads,
  cfg.modelChannels / cfg.numHeads,
  cfg.modelChannels,
);

const sigmas = betaSchedule(flowSigmas(), steps);
console.log(`\n${sigmas.length - 1} steps, ${latentW}x${latentH} latent for a ${width}x${height} image`);

const shape = { T: 1, H: latentH, W: latentW };
const count = LATENT.channels * shape.T * shape.H * shape.W;
const x0 = noiseScaling(sigmas[0]!, gaussianNoise(count, seed), null);

let modelCalls = 0;
const sampleStarted = Date.now();

/**
 * `res_multistep` with a denoiser that has to be awaited.
 *
 * The sampler in `sampler.ts` is synchronous because ComfyUI's is, and it is
 * pinned step by step against ComfyUI's on a toy denoiser. A 52-block forward
 * is not synchronous. Making the sampler `async` would mean the thing under
 * test and the thing that runs are different functions, which is how a
 * carefully verified stepper quietly stops being the one in the pipeline.
 *
 * So the verified stepper is used unchanged, and re-run over its own growing
 * prefix: after step `k` the `k + 1` predictions collected so far are replayed
 * through it to get the latent. Its whole state is `old_denoised` and
 * `old_sigma_down`, both functions of that sequence, so the replay reproduces
 * exactly what a streaming implementation would — and it is arithmetic on one
 * latent, sitting next to a forward that takes a second.
 */
async function sampleAsync(): Promise<Float32Array> {
  const predictions: Float32Array[] = [];
  let out = x0;

  for (let step = 0; step < sigmas.length - 1; step += 1) {
    const sigma = sigmas[step]!;
    const t = timestepOf(sigma);
    const stepStarted = Date.now();

    const forward = async (context: Float32Array): Promise<Float32Array> => {
      modelCalls += 1;
      return animaForwardResident(
        device, ditKernels(), cfg, gpuWeights,
        { latent: out, ...shape, t, context }, undefined, held,
      );
    };

    const cond = await forward(positive);
    const prediction = unconditional ? applyCfg(cond, await forward(unconditional), guidance) : cond;
    predictions.push(calculateDenoised(sigma, prediction, out));

    let cursor = 0;
    out = resMultistep(() => predictions[cursor++]!, x0, sigmas.slice(0, step + 2));

    // Per-channel spatial standard deviation, not the tensor's overall one.
    // A latent that is flat in space but offset channel to channel has a large
    // overall std and no picture in it — which is exactly the failure this line
    // exists to make visible while it is still running.
    const per = out.length / LATENT.channels;
    const spread: number[] = [];
    for (let ch = 0; ch < LATENT.channels; ch += 1) {
      let sum = 0, sq = 0;
      for (let i = 0; i < per; i += 1) { const v = out[ch * per + i]!; sum += v; sq += v * v; }
      spread.push(Math.sqrt(Math.max(0, sq / per - (sum / per) ** 2)));
    }
    spread.sort((x, y) => x - y);
    process.stdout.write(
      `\r  step ${step + 1}/${sigmas.length - 1}  sigma ${sigma.toFixed(4)}  ` +
        `spread ${spread[spread.length >> 1]!.toFixed(3)}  ` +
        `${((Date.now() - stepStarted) / 1000).toFixed(2)}s   ` + (process.env.ANIMA_TRACE ? "\n" : ""),
    );
  }
  process.stdout.write("\n");
  return out;
}

const sampled = await sampleAsync();
const elapsed = (Date.now() - sampleStarted) / 1000;
console.log(
  `sampled in ${elapsed.toFixed(1)}s — ${modelCalls} model calls, ` +
    `${(elapsed / modelCalls).toFixed(2)}s each`,
);

releaseAnimaWeights(held);
await device.destroy();

// --- output ---
const forVae = latentToVae(sampled);
mkdirSync(dirname(`${outPath}.bin`) || ".", { recursive: true });
writeFileSync(`${outPath}.bin`, Buffer.from(forVae.buffer, forVae.byteOffset, forVae.byteLength));
writeFileSync(
  `${outPath}.json`,
  JSON.stringify({ prompt, negative, steps: sigmas.length - 1, guidance, width, height, seed, shape, channels: LATENT.channels }, null, 1),
);
console.log(`\nwrote ${outPath}.bin (${(forVae.byteLength / 1e6).toFixed(1)} MB, ${LATENT.channels}x${latentH}x${latentW}, VAE input)`);
console.log(`      ${outPath}.json`);

// The 16-to-3 projection ComfyUI shows while sampling — `latent_rgb_factors`
// applied to the raw latent, before `process_out`. It is one eighth of the
// resolution and a linear map, so it shows composition and colour and nothing
// finer. It is **not** the model's image, and is written under a name that says
// so rather than under `${outPath}.png`.
const preview = new Float32Array(3 * latentH * latentW);
for (let i = 0; i < latentH * latentW; i += 1) {
  for (let c = 0; c < 3; c += 1) {
    let sum = LATENT.rgbBias[c]!;
    for (let ch = 0; ch < LATENT.channels; ch += 1) {
      sum += sampled[ch * latentH * latentW + i]! * LATENT.rgbFactors[ch * 3 + c]!;
    }
    preview[c * latentH * latentW + i] = sum * 2;
  }
}
writeFileSync(`${outPath}.preview.png`, encodePng(preview, latentH, latentW));
console.log(`      ${outPath}.preview.png (${latentW}x${latentH}, the latent's RGB projection — not the image)`);
console.log(
  "\nNo decoded image: Anima uses Wan 2.1's 3D causal VAE, which this repository\n" +
    "does not have yet. The .bin above is exactly what that decoder takes.",
);
