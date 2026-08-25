/**
 * Anima-3.8B in a browser: a prompt in, an image out, on WebGPU.
 *
 * Issue #170 stage 6. Nothing here is new arithmetic — every stage is the one
 * already checked against ComfyUI's own model in `examples/anima`, and this is
 * the wiring plus the two things a page needs that a script does not: weights
 * over HTTP instead of off a disk, and a device that survives between steps.
 *
 * | stage | checked by | agreement |
 * | --- | --- | --- |
 * | tokenizers | `tokenize.test.ts` | exact ids, 24 cases each |
 * | encoder + adapter | `verify-encoder.ts` | 9.775e-7 |
 * | DiT, resident | `verify-forward-gpu.ts` | 1.182e-5 |
 * | the sampling loop | `verify-trajectory.ts` | 3.274e-3 over 8 steps |
 * | VAE decoder | `verify-vae.ts` | 6.917e-7 |
 *
 * **Two devices.** The DiT runs on a resident one, where its 3.76 GB of weights
 * stay uploaded between steps — the difference between 0.2 s and 20 s a step,
 * measured on the Z-Image port before this one existed. The encoder and the VAE
 * run on the per-dispatch `Runner`, which is simpler and which they only touch
 * once each.
 *
 * The adapter's six blocks stay on the CPU. They are 512 tokens at dim 1024
 * against a DiT that is 3,952 tokens at 2048, and moving them would be work
 * with nothing measurable behind it.
 */
import { createBrowserRunner } from "../../llm-demo/src/browser-runtime.js";
import { createBrowserResidentDevice } from "./browser-resident.js";
import { FetchedAnimaWeights, FetchedSafetensors } from "./fetch-weights.js";
import { ditKernels, encoderKernels, vaeKernels } from "./kernels-web.js";
import { type BpeVocab } from "../../../llm/tokenizer-bpe.js";
import { permuteLayerForRope, type Qwen3LayerWeights } from "../../zimage/src/text-encoder.js";
import { qwen3EncodeGpu, type Qwen3GpuWeights } from "../../zimage/src/text-encoder-gpu.js";
import type { AnimaConfig, AnimaInput } from "../../anima/src/dit.js";
import {
  animaForwardResident, releaseAnimaWeights,
  type AnimaProfile, type AnimaWeightSource, accountForForward } from "../../anima/src/dit-resident.js";
import { permuteForRope } from "../../anima/src/block.js";
import {
  ADAPTER,
  type AdapterBlockWeights,
  type AdapterWeights,
  QWEN3_06B,
  llmAdapterForward,
  padContext,
  permuteAdapterBlock,
} from "../../anima/src/text-encoder.js";
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
} from "../../anima/src/sampler.js";
import { type T5Vocab, animaTokenizers, tokenizePrompt } from "../../anima/src/tokenize.js";
import { type VaeWeights, type WanVaeConfig } from "../../anima/src/vae.js";
import { wanVaeDecodeGpu } from "../../anima/src/vae-gpu.js";

declare const BUILD_STAMP: string;

const DIT_BASE = "/weights/dit";
const ENCODER_URL = "/weights/encoder/qwen_3_06b_base.safetensors";
const VAE_URL = "/weights/vae/qwen_image_vae.safetensors";

const $ = <T extends HTMLElement>(id: string): T => {
  const element = document.getElementById(id);
  if (!element) throw new Error(`no element #${id}`);
  return element as T;
};

const status = $<HTMLParagraphElement>("status");
const detail = $<HTMLParagraphElement>("detail");
const canvas = $<HTMLCanvasElement>("out");
const goButton = $<HTMLButtonElement>("go");
const promptInput = $<HTMLTextAreaElement>("prompt");
const negativeInput = $<HTMLTextAreaElement>("negative");
const sizeSelect = $<HTMLSelectElement>("size");
const stepsInput = $<HTMLInputElement>("steps");
const cfgInput = $<HTMLInputElement>("cfg");
const seedInput = $<HTMLInputElement>("seed");
const profileBox = $<HTMLInputElement>("profile");
const profileOut = $<HTMLDivElement>("profile-out");

function say(message: string, extra = ""): void {
  status.textContent = message;
  detail.textContent = extra;
}

$<HTMLElement>("build").textContent = BUILD_STAMP;

/** xorshift128+ and Box-Muller, matching `examples/anima/src/generate.ts`. */
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
  // Not torch's Philox, so a seed reproduces this port's own runs and **not**
  // ComfyUI's image for the same number.
  for (let i = 0; i < count; i += 2) {
    const u = Math.max(next(), Number.MIN_VALUE);
    const v = next();
    const r = Math.sqrt(-2 * Math.log(u));
    out[i] = r * Math.cos(2 * Math.PI * v);
    if (i + 1 < count) out[i + 1] = r * Math.sin(2 * Math.PI * v);
  }
  return out;
}

/** `[3, H, W]` in roughly `[-1, 1]` onto the canvas. */
function draw(image: Float32Array, H: number, W: number): void {
  canvas.width = W;
  canvas.height = H;
  const context = canvas.getContext("2d");
  if (!context) throw new Error("no 2d context");
  const data = context.createImageData(W, H);
  const hw = H * W;
  for (let i = 0; i < hw; i += 1) {
    for (let c = 0; c < 3; c += 1) {
      const v = (image[c * hw + i]! + 1) * 0.5 * 255;
      data.data[i * 4 + c] = Math.max(0, Math.min(255, Math.round(v)));
    }
    data.data[i * 4 + 3] = 255;
  }
  context.putImageData(data, 0, 0);
}

/**
 * The per-kernel breakdown, as a table under the controls.
 *
 * Reported as a share of the profiled forward rather than as seconds alone: the
 * profiling itself costs passes, so the absolute numbers are larger than the
 * shipping ones and the proportions are what carry over.
 */
function reportProfile(
  profile: AnimaProfile,
  wallSeconds: number,
  dispatches: number,
  preloadMs: number,
  yieldMs: number,
): void {
  if (!profile.supported) {
    profileOut.innerHTML =
      '<p class="note">This device has no <code>timestamp-query</code>, so there is no GPU breakdown — ' +
      "not a breakdown of zeros.</p>";
    return;
  }
  if (profile.byKernel.size === 0) {
    profileOut.innerHTML =
      '<p class="note">The device has <code>timestamp-query</code> but every query came back zero, ' +
      "which means the driver declined them. No breakdown, rather than a breakdown of zeros.</p>";
    return;
  }
  const rows = [...profile.byKernel.entries()].sort((a, b) => b[1].seconds - a[1].seconds);
  const total = rows.reduce((sum, [, v]) => sum + v.seconds, 0);
  const counted = rows.reduce((n, [, v]) => n + v.dispatches, 0);
  const acct = accountForForward(profile, wallSeconds);
  const ms = (v: number): string => `${v.toFixed(0)} ms`;
  const pct = (v: number): string => `${((v / acct.wallMs) * 100).toFixed(0)}%`;
  // Coverage is what caught the last wrong answer: a profile that timed 78% of
  // the forward reported shares of a partial total as though they were shares
  // of the whole. It is now a share of the *named* wall clock rather than of
  // GPU alone, so a browser forward that spends half its time outside every
  // dispatch says so instead of reading as a 52% mystery.
  // Two different failures, and the second one used to read as success: a
  // counter summing more forwards than it measured named 2853% of the wall and
  // the cap turned that into "100% accounted for".
  const warn = acct.overAccountedMs > 0
    ? ` <strong>The phases name ${ms(acct.overAccountedMs)} more than this forward lasted — ` +
      "a counter is measuring something other than this forward, so nothing below is trustworthy.</strong>"
    : acct.coverage < 90
      ? ' <strong>Under 90% — some of this forward is still unaccounted for.</strong>'
      : "";
  profileOut.innerHTML =
    `<p class="note">One forward, ${ms(acct.wallMs)} wall — ${acct.coverage.toFixed(0)}% accounted for.${warn}</p>` +
    "<table>" +
    `<tr><td>inside compute passes</td><td>${ms(acct.inPassesMs)}</td><td>${pct(acct.inPassesMs)}</td>` +
    `<td>${counted} of ${dispatches}</td></tr>` +
    `<tr><td>queue and driver, outside the passes</td><td>${ms(acct.aroundPassesMs)}</td>` +
    `<td>${pct(acct.aroundPassesMs)}</td><td></td></tr>` +
    `<tr><td>recording the commands</td><td>${ms(acct.encodeMs)}</td><td>${pct(acct.encodeMs)}</td><td></td></tr>` +
    `<tr><td>reading results back</td><td>${ms(acct.readbackMs)}</td><td>${pct(acct.readbackMs)}</td><td></td></tr>` +
    `<tr><td>building bind groups</td><td>${ms(acct.bindGroupMs)}</td><td>${pct(acct.bindGroupMs)}</td>` +
    `<td>${profile.bindGroups}</td></tr>` +
    `<tr><td>the caller's callbacks</td><td>${ms(acct.hostCallbackMs)}</td><td>${pct(acct.hostCallbackMs)}</td>` +
    "<td></td></tr>" +
    `<tr><td>&nbsp;&nbsp;— re-reading weights (preloadPrefix)</td><td>${ms(preloadMs)}</td><td>${pct(preloadMs)}</td>` +
    "<td></td></tr>" +
    `<tr><td>&nbsp;&nbsp;— yielding to the event loop</td><td>${ms(yieldMs)}</td><td>${pct(yieldMs)}</td>` +
    "<td>55</td></tr>" +
    `<tr><td>unattributed</td><td>${ms(acct.unattributedMs)}</td><td>${pct(acct.unattributedMs)}</td><td></td></tr>` +
    "</table>" +
    `<p class="note">Of the ${ms(acct.inPassesMs)} inside compute passes, by kernel:</p><table>` +
    rows.map(([name, v]) =>
      `<tr><td>${name}</td><td>${(v.seconds * 1000).toFixed(1)} ms</td>` +
      `<td>${((v.seconds / total) * 100).toFixed(1)}%</td><td>${v.dispatches}</td></tr>`).join("") +
    "</table>";
}

async function main(): Promise<void> {
  if (!navigator.gpu) {
    say("This browser has no WebGPU.", "Chrome 113+ or Edge 113+ on a machine with a GPU.");
    return;
  }

  say("opening the model …");
  const runner = await createBrowserRunner();
  const residentDevice = await createBrowserResidentDevice();

  const [dit, encoderFile, vaeFile, qwenVocab, t5Vocab] = await Promise.all([
    FetchedAnimaWeights.open(DIT_BASE),
    FetchedSafetensors.open(ENCODER_URL),
    FetchedSafetensors.open(VAE_URL),
    fetch("/weights/tokenizer/qwen-qwen3-4b.bpe-vocab.json").then((r) => r.json() as Promise<BpeVocab>),
    fetch("/weights/tokenizer/t5.unigram-vocab.json").then((r) => r.json() as Promise<T5Vocab>),
  ]);
  const tokenizers = animaTokenizers(qwenVocab, t5Vocab);

  // The DiT's shape, out of the manifest `convert_dit.py` writes — which reads
  // it with `detect_unet_config` rather than transcribing it.
  const c = dit.config as unknown as {
    num_blocks: number; model_channels: number; num_heads: number; adaln_lora_dim: number;
    in_channels: number; out_channels: number; patch_spatial: number; patch_temporal: number;
    crossattn_emb_channels: number; concat_padding_mask: boolean;
    rope_t_extrapolation_ratio: number; rope_h_extrapolation_ratio: number; rope_w_extrapolation_ratio: number;
  };
  const cfg: AnimaConfig = {
    numBlocks: c.num_blocks, modelChannels: c.model_channels, numHeads: c.num_heads,
    adalnLoraDim: c.adaln_lora_dim, inChannels: c.in_channels, outChannels: c.out_channels,
    patchSpatial: c.patch_spatial, patchTemporal: c.patch_temporal,
    crossattnEmbChannels: c.crossattn_emb_channels, concatPaddingMask: c.concat_padding_mask,
    maxPeriod: 10000, normEps: 1e-6,
    ropeExtrapolation: {
      t: c.rope_t_extrapolation_ratio, h: c.rope_h_extrapolation_ratio, w: c.rope_w_extrapolation_ratio,
    },
  };
  const headDim = cfg.modelChannels / cfg.numHeads;

  /**
   * The DiT's weights, with the self-attention rope permutation applied.
   *
   * `packedQ8` returns `null` for anything permuted, so the resident path's
   * fast route cannot go around the relabelling. That is the same rule
   * `withRopePermutation` enforces for Node, restated here because this loader
   * is a different class — and skipping it is measured at 1.068e-1.
   */
  const permuted = new Map<string, Float32Array>();
  const isPermuted = (name: string): "projection" | "norm" | null => {
    if (/\.self_attn\.(q|k)_proj\.weight$/.test(name)) return "projection";
    if (/\.self_attn\.(q|k)_norm\.weight$/.test(name)) return "norm";
    return null;
  };
  const ditWeights: AnimaWeightSource = {
    has: (name) => dit.has(name),
    shapeOf: (name) => dit.shapeOf(name),
    get: (name) => {
      const cached = permuted.get(name);
      if (cached) return cached;
      const kind = isPermuted(name);
      const raw = dit.get(name);
      if (!kind) return raw;
      const out = kind === "projection"
        ? permuteForRope(raw, cfg.numHeads, headDim, cfg.modelChannels)
        : permuteForRope(raw, 1, headDim, 1);
      permuted.set(name, out);
      return out;
    },
    packedQ8: (name) => (isPermuted(name) ? null : dit.packedQ8(name)),
  };

  const encoderLayers = new Map<number, Qwen3LayerWeights>();
  const encoderWeights: Qwen3GpuWeights = {
    numLayers: QWEN3_06B.numLayers,
    embed: (ids) => encoderFile.readRows("model.embed_tokens.weight", ids, QWEN3_06B.hiddenSize),
    async layer(index) {
      const cached = encoderLayers.get(index);
      if (cached) return cached;
      const p = `model.layers.${index}.`;
      const [
        input_layernorm, q_proj, k_proj, v_proj, o_proj, q_norm, k_norm,
        post_attention_layernorm, gate_proj, up_proj, down_proj,
      ] = await Promise.all([
        encoderFile.read(`${p}input_layernorm.weight`),
        encoderFile.read(`${p}self_attn.q_proj.weight`),
        encoderFile.read(`${p}self_attn.k_proj.weight`),
        encoderFile.read(`${p}self_attn.v_proj.weight`),
        encoderFile.read(`${p}self_attn.o_proj.weight`),
        encoderFile.read(`${p}self_attn.q_norm.weight`),
        encoderFile.read(`${p}self_attn.k_norm.weight`),
        encoderFile.read(`${p}post_attention_layernorm.weight`),
        encoderFile.read(`${p}mlp.gate_proj.weight`),
        encoderFile.read(`${p}mlp.up_proj.weight`),
        encoderFile.read(`${p}mlp.down_proj.weight`),
      ]);
      const built = permuteLayerForRope({
        input_layernorm, q_proj, k_proj, v_proj, o_proj, q_norm, k_norm,
        post_attention_layernorm, gate_proj, up_proj, down_proj,
      }, QWEN3_06B);
      // Two layers at a time: 28 of them at 1024 wide is a gigabyte held, and
      // the Cache API already remembers the bytes.
      if (encoderLayers.size > 2) encoderLayers.delete(encoderLayers.keys().next().value as number);
      encoderLayers.set(index, built);
      return built;
    },
    // `layer="last"` is after `model.norm` — Z-Image's encoder stops before it.
    finalNorm: encoderFile.read("model.norm.weight"),
  };

  /**
   * The adapter, materialized once and then held.
   *
   * It lives inside the DiT file, so its tensors come through the same loader —
   * but that loader is an LRU of 48 and the adapter is 118 tensors, so
   * preloading them all and reading them afterwards evicts most of them before
   * the first one is used. That was the first thing this page did, and it
   * failed on `net.llm_adapter.out_proj.bias`.
   *
   * Built a block at a time instead: fetch a block's nineteen tensors, permute
   * them for rope, keep the result, and let the raw ones fall out of the LRU.
   * What stays is the six permuted blocks and the four outer tensors — about
   * 350 MB with the embedding table, which is held because a 32,128-row table
   * cannot survive an LRU that is counting tensors rather than bytes.
   */
  const adapterBlocks = new Map<number, AdapterBlockWeights>();
  let adapterEmbed: Float32Array | null = null;
  let adapterOuter: { out_proj: Float32Array; out_proj_bias: Float32Array; norm: Float32Array } | null = null;

  const blockTensorNames = (index: number): string[] => {
    const p = `net.llm_adapter.blocks.${index}.`;
    return [
      "norm_self_attn.weight", "self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight",
      "self_attn.o_proj.weight", "self_attn.q_norm.weight", "self_attn.k_norm.weight",
      "norm_cross_attn.weight", "cross_attn.q_proj.weight", "cross_attn.k_proj.weight", "cross_attn.v_proj.weight",
      "cross_attn.o_proj.weight", "cross_attn.q_norm.weight", "cross_attn.k_norm.weight",
      "norm_mlp.weight", "mlp.0.weight", "mlp.0.bias", "mlp.2.weight", "mlp.2.bias",
    ].map((n) => p + n);
  };

  async function buildAdapter(report: (message: string) => void): Promise<void> {
    if (adapterOuter && adapterEmbed) return;
    const outerNames = [
      "net.llm_adapter.embed.weight", "net.llm_adapter.out_proj.weight",
      "net.llm_adapter.out_proj.bias", "net.llm_adapter.norm.weight",
    ];
    await dit.preload(outerNames);
    adapterEmbed = dit.get("net.llm_adapter.embed.weight");
    adapterOuter = {
      out_proj: dit.get("net.llm_adapter.out_proj.weight"),
      out_proj_bias: dit.get("net.llm_adapter.out_proj.bias"),
      norm: dit.get("net.llm_adapter.norm.weight"),
    };
    for (let index = 0; index < ADAPTER.numLayers; index += 1) {
      report(`adapter block ${index + 1}/${ADAPTER.numLayers}`);
      await dit.preload(blockTensorNames(index));
      const p = `net.llm_adapter.blocks.${index}.`;
      adapterBlocks.set(index, permuteAdapterBlock({
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
      }));
    }
  }

  const adapterWeights: AdapterWeights = {
    embed(ids) {
      if (!adapterEmbed) throw new Error("adapter: embed read before buildAdapter");
      const rows = new Float32Array(ids.length * ADAPTER.targetDim);
      for (let i = 0; i < ids.length; i += 1) {
        rows.set(adapterEmbed.subarray(ids[i]! * ADAPTER.targetDim, (ids[i]! + 1) * ADAPTER.targetDim), i * ADAPTER.targetDim);
      }
      return rows;
    },
    block(index) {
      const built = adapterBlocks.get(index);
      if (!built) throw new Error(`adapter: block ${index} read before buildAdapter`);
      return built;
    },
    get out_proj() {
      if (!adapterOuter) throw new Error("adapter: out_proj read before buildAdapter");
      return adapterOuter.out_proj;
    },
    get out_proj_bias() {
      if (!adapterOuter) throw new Error("adapter: out_proj_bias read before buildAdapter");
      return adapterOuter.out_proj_bias;
    },
    get norm() {
      if (!adapterOuter) throw new Error("adapter: norm read before buildAdapter");
      return adapterOuter.norm;
    },
  };

  const vaeCache = new Map<string, Float32Array>();
  const vaeWeights: VaeWeights = {
    get(name) {
      const value = vaeCache.get(name);
      if (!value) throw new Error(`vae: "${name}" was read before it was fetched`);
      return value;
    },
    has: (name) => vaeFile.has(name),
  };
  const vaeCfg: WanVaeConfig = { dim: 96, zDim: LATENT.channels, dimMult: [1, 2, 4, 4], numResBlocks: 2 };

  /**
   * The whole DiT into the browser's **disk** cache, once, before anything
   * generates.
   *
   * Fetching per block during the forward instead would put 3.76 GB on the wire
   * in every denoising step — the same bytes forty times over. This pays the
   * download once; every step after it, and every generation after this one,
   * reads from disk and touches the network not at all.
   */
  const gb = (bytes: number): string => `${(bytes / 1e9).toFixed(2)} GB`;
  await dit.preloadAll((done, total, bytes) => {
    say("caching the DiT …", `${done}/${total} tensors, ${gb(bytes)}`);
  });

  say("ready.", `${cfg.numBlocks} blocks at ${cfg.modelChannels} wide, plus a ${ADAPTER.numLayers}-block adapter.`);
  goButton.disabled = false;

  goButton.addEventListener("click", () => {
    void generate().catch((error: unknown) => {
      say("failed.", error instanceof Error ? error.message : String(error));
      goButton.disabled = false;
      console.error(error);
    });
  });

  async function generate(): Promise<void> {
    goButton.disabled = true;
    const [width, height] = sizeSelect.value.split("x").map(Number) as [number, number];
    const steps = Number(stepsInput.value) || DEFAULTS.steps;
    const guidance = Number(cfgInput.value);
    const seed = Number(seedInput.value) || 0;
    const latentH = height / 8;
    const latentW = width / 8;

    /**
     * Where one forward's GPU time goes, by kernel.
     *
     * **Not free**: `batch()` needs a compute pass per timestamp pair, so a
     * profiled forward runs one pass per dispatch against 55 submits' worth
     * unprofiled. Only the first forward is profiled, and the numbers are
     * reported as proportions rather than as the shipping cost.
     */
    const wantProfile = profileBox.checked;
    let profile: AnimaProfile | null = null;
    let profiledWallSeconds = 0;
    let profiledDispatches = 0;
    let profiledPreloadMs = 0;
    let profiledYieldMs = 0;
    profileOut.innerHTML = "";
    // The profiled forward is the **second** step: the first uploads 3.63 GB
    // and hydrates the heap, so its timings are the loading cost. One step
    // therefore has nothing to profile — said here rather than left as a table
    // that never appears, which is how the first version failed.
    if (wantProfile && steps < 2) {
      profileOut.innerHTML =
        '<p class="note">Profiling needs at least 2 steps — the first forward is the upload, ' +
        "so the second is the one measured.</p>";
    }

    /**
     * Cumulative across every forward; a forward's share is a difference.
     *
     * Split, because the callback does two things and they have different
     * fixes. `preloadPrefix` re-reads from the Cache API — the heap holds at
     * most `48 * 4 = 192` packed tensors against 898 in the model, so the LRU
     * thrashes and every forward re-reads what the last one evicted, even
     * though the weights have been resident on the device since the first.
     * The `setTimeout(0)` yield is 55 turns of the event loop a forward
     * against a browser's clamp. Which of the two costs 1370 ms is a
     * measurement, and reading the code is how the last four guesses were
     * made.
     */
    let hostCallbackMsTotal = 0;
    let preloadMsTotal = 0;
    let yieldMsTotal = 0;

    const started = performance.now();

    // --- conditioning ---
    say("conditioning …", "reading the encoder");
    const conditioning = async (text: string): Promise<Float32Array> => {
      const { qwenIds, t5Ids, t5Weights } = tokenizePrompt(tokenizers, text);
      const source = await qwen3EncodeGpu(runner.run, encoderKernels, QWEN3_06B, encoderWeights, qwenIds, (index) => {
        say("conditioning …", `encoder layer ${index + 1}/${QWEN3_06B.numLayers}`);
      });
      await buildAdapter((message) => say("conditioning …", message));
      const context = llmAdapterForward(adapterWeights, source, qwenIds.length, t5Ids);
      return padContext(context, t5Ids.length, t5Weights);
    };

    const positive = await conditioning(promptInput.value);
    const useCfg = cfgEnabled(guidance);
    const unconditional = useCfg ? await conditioning(negativeInput.value) : null;
    const conditioningSeconds = (performance.now() - started) / 1000;

    // --- sampling ---
    const sigmas = betaSchedule(flowSigmas(), steps);
    const shape = { T: 1, H: latentH, W: latentW };
    const x0 = noiseScaling(sigmas[0]!, gaussianNoise(LATENT.channels * latentH * latentW, seed), null);
    const held = new Map<string, GPUBuffer>();
    /**
     * Set once a forward has run to completion with `held`.
     *
     * Not `held.size > 0`: that is true partway through the first forward,
     * while the prefixes it has not reached yet still need preloading.
     */
    let weightsResident = false;
    const predictions: Float32Array[] = [];
    let out = x0;
    const samplingStart = performance.now();
    // The first forward uploads 3.63 GB of weights and hydrates the heap from
    // the disk cache; every one after it does neither. Reported apart from the
    // rest, because folding a one-off into a per-step average is how a page
    // ends up disagreeing with a command line for no visible reason.
    let firstForwardSeconds = 0;

    for (let step = 0; step < sigmas.length - 1; step += 1) {
      const sigma = sigmas[step]!;
      const t = timestepOf(sigma);
      const stepStarted = performance.now();

      const forward = async (context: Float32Array, first = false): Promise<Float32Array> => {
        const input: AnimaInput = { latent: out, ...shape, t, context };
        // The *second* step, not the first: the first uploads 3.63 GB and
        // hydrates the heap, so its timings are the loading cost rather than
        // the steady-state one.
        const wanted = wantProfile && first && step === 1;
        if (wanted && !profile) {
          profile = { byKernel: new Map(), supported: residentDevice.timestampsSupported, encodeMs: 0, submitToDoneMs: 0, readbackMs: 0, bindGroupMs: 0, bindGroups: 0, hostCallbackMs: 0 };
        }
        const stats = wanted
          ? { dispatches: 0, submits: 0, poolSlots: 0, poolBytes: 0, weightBuffers: 0, uploadedBytes: 0 }
          : undefined;
        // Device counters are cumulative; a forward's share is the difference.
        const bindGroupsBefore = residentDevice.stats.bindGroupMs;
        const hostCallbackBefore = hostCallbackMsTotal;
        const preloadBefore = preloadMsTotal;
        const yieldBefore = yieldMsTotal;
        const bindGroupCountBefore = residentDevice.stats.bindGroups;
        const forwardStart = wanted ? performance.now() : 0;
        const result = await animaForwardResident(
          residentDevice, ditKernels, cfg, ditWeights, input, stats, held, undefined, undefined,
          // Hydrates the heap a block at a time, from the disk cache, for the
          // first forward only — the guard below is what makes that true. It
          // used to be an unchecked claim in this comment, and it was false.
          async (prefix) => {
            // **Only while the device does not already hold the weights.**
            // `weightBuffer()` and `project()` both check `held` before asking
            // the source for anything — "Already on the device: dispatch
            // without asking the source for anything" — so from the second
            // forward on, this pulled tensors off disk that nothing would
            // read. Measured at 1346 ms of a 3769 ms forward, ~36%, and every
            // millisecond of it wasted; the heap holds 192 packed tensors
            // against 898 in the model, so the LRU cannot even make the
            // re-read cheap.
            //
            // **`held.size > 0` is the wrong test, and running it is how that
            // was found.** `held` fills *during* the first forward, so it is
            // non-empty from the first prefix onward and every later prefix
            // was skipped on the very forward that needed them:
            // `"net.t_embedder.1.linear_1.weight" was read before it was
            // preloaded`. The condition is "a forward has finished", which
            // nothing about `held` expresses partway through one.
            if (weightsResident) return;
            const t0 = performance.now();
            await dit.preloadPrefix(prefix);
            const t1 = performance.now();
            // Yield, so a 52-block first forward does not freeze the tab.
            await new Promise((resolve) => setTimeout(resolve, 0));
            preloadMsTotal += t1 - t0;
            yieldMsTotal += performance.now() - t1;
            // Into a counter that runs for every forward, never straight into
            // `profile`: this callback fires on all eighty and `profile`
            // outlives the one being measured, so `+= into profile` summed the
            // whole generation and reported 2853% of one forward.
            hostCallbackMsTotal += performance.now() - t0;
          },
          wanted ? profile ?? undefined : undefined,
        );
        // Every weight the DiT touches is on the device now, so nothing after
        // this needs the source.
        weightsResident = true;
        if (wanted && stats) {
          profiledWallSeconds = (performance.now() - forwardStart) / 1000;
          profiledDispatches = stats.dispatches;
          if (profile) {
            profile.bindGroupMs = residentDevice.stats.bindGroupMs - bindGroupsBefore;
            profile.bindGroups = residentDevice.stats.bindGroups - bindGroupCountBefore;
            profile.hostCallbackMs = hostCallbackMsTotal - hostCallbackBefore;
            profiledPreloadMs = preloadMsTotal - preloadBefore;
            profiledYieldMs = yieldMsTotal - yieldBefore;
          }
        }
        return result;
      };
      const cond = await forward(positive, true);
      const prediction = unconditional ? applyCfg(cond, await forward(unconditional), guidance) : cond;
      predictions.push(calculateDenoised(sigma, prediction, out));

      // The verified stepper, replayed over its own growing prefix — see
      // `generate.ts` for why the sampler is not made async instead.
      let cursor = 0;
      out = resMultistep(() => predictions[cursor++]!, x0, sigmas.slice(0, step + 2));

      const stepSeconds = (performance.now() - stepStarted) / 1000;
      if (step === 0) firstForwardSeconds = stepSeconds;
      say(
        `step ${step + 1}/${sigmas.length - 1}`,
        `sigma ${sigma.toFixed(4)}, ${stepSeconds.toFixed(2)}s`,
      );
      // Yield, so the status text above actually paints.
      await new Promise((resolve) => setTimeout(resolve, 0));
    }
    const samplingSeconds = (performance.now() - samplingStart) / 1000;
    releaseAnimaWeights(held);

    // --- decode ---
    const decodeStart = performance.now();
    say("decoding …", "reading the VAE");
    if (vaeCache.size === 0) {
      const names = ["conv2.weight", "conv2.bias", "decoder.conv1.weight", "decoder.conv1.bias",
        "decoder.head.0.gamma", "decoder.head.2.weight", "decoder.head.2.bias"];
      for (const prefix of ["decoder.middle.0.", "decoder.middle.2."]) {
        names.push(`${prefix}residual.0.gamma`, `${prefix}residual.2.weight`, `${prefix}residual.2.bias`,
          `${prefix}residual.3.gamma`, `${prefix}residual.6.weight`, `${prefix}residual.6.bias`);
      }
      names.push("decoder.middle.1.norm.gamma", "decoder.middle.1.to_qkv.weight", "decoder.middle.1.to_qkv.bias",
        "decoder.middle.1.proj.weight", "decoder.middle.1.proj.bias");
      for (let i = 0; i < 15; i += 1) {
        const p = `decoder.upsamples.${i}.`;
        for (const suffix of ["residual.0.gamma", "residual.2.weight", "residual.2.bias", "residual.3.gamma",
          "residual.6.weight", "residual.6.bias", "shortcut.weight", "shortcut.bias",
          "resample.1.weight", "resample.1.bias"]) {
          if (vaeFile.has(p + suffix)) names.push(p + suffix);
        }
      }
      await Promise.all(names.map(async (n) => vaeCache.set(n, await vaeFile.read(n))));
    }
    const image = await wanVaeDecodeGpu(
      runner.run, vaeKernels, vaeCfg, vaeWeights, latentToVae(out), latentH, latentW, undefined,
      (label, done, total) => say("decoding …", `${label} ${done}/${total}`),
    );
    draw(image, height, width);

    const decodeSeconds = (performance.now() - decodeStart) / 1000;
    const elapsed = (performance.now() - started) / 1000;
    const taken = sigmas.length - 1;
    // Split, because a total is not comparable to anything. The command-line
    // path reports the same three separately, and a page that reports only
    // their sum cannot be checked against it.
    const perStep = (samplingSeconds - firstForwardSeconds) / Math.max(1, taken - 1);
    say(
      "done.",
      `${width}x${height}, ${taken} steps, ${elapsed.toFixed(1)}s — ` +
        `conditioning ${conditioningSeconds.toFixed(1)}s, ` +
        `sampling ${samplingSeconds.toFixed(1)}s (first step ${firstForwardSeconds.toFixed(1)}s, ` +
        `then ${perStep.toFixed(2)}s each), ` +
        `decode ${decodeSeconds.toFixed(1)}s`,
    );
    if (profile) {
      reportProfile(profile, profiledWallSeconds, profiledDispatches, profiledPreloadMs, profiledYieldMs);
    }
    else if (wantProfile && steps >= 2) {
      profileOut.innerHTML =
        '<p class="note">Profiling was requested and produced nothing, which should not happen — ' +
        "the second forward did not run.</p>";
    }
    goButton.disabled = false;
  }
}

void main().catch((error: unknown) => {
  say("failed to start.", error instanceof Error ? error.message : String(error));
  console.error(error);
});
