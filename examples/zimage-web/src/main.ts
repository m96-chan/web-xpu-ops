/**
 * The browser demo's entry point: a prompt in, a picture on a canvas.
 *
 * Every stage below is the same code the Node verifiers run — `qwen3EncodeGpu`,
 * `ditForwardGpu`, `decodeGpu`, `flowSchedule`/`eulerStep` — with two things
 * swapped: the kernels come from esbuild's `text` loader instead of the
 * filesystem, and the weights come from HTTP `Range` requests instead of file
 * descriptors. There is deliberately no second copy of the forward pass, since
 * the copy is what would drift away from the numbers those verifiers earned.
 *
 * What each part was measured at, against the model itself:
 *
 *   | tokenizer     | exact ids   |
 *   | text encoder  | 7.993e-7    |
 *   | DiT           | 4.386e-6    |
 *   | sampler       | exact       |
 *   | VAE decoder   | 1.085e-5    |
 */
import { createBrowserRunner } from "../../llm-demo/src/browser-runtime.js";
import { type BpeVocab, ByteLevelBpeTokenizer } from "../../../llm/tokenizer-bpe.js";
import { type DecoderConfig } from "../../zimage-vae/src/decoder.js";
import { decodeGpu } from "../../zimage-vae/src/decoder-gpu.js";
import { type DitConfig } from "../../zimage/src/dit.js";
import { type PackedWeightSource } from "../../zimage/src/dit-gpu.js";
import { ditForwardResident, releaseDitWeights, type ResidentDitStats } from "../../zimage/src/dit-resident.js";
import { createBrowserResidentDevice } from "./browser-resident.js";
import {
  DEFAULT_FLOW_SHIFT,
  NUM_TRAIN_TIMESTEPS,
  VARIANTS,
  type Variant,
  applyCfg,
  cfgEnabled,
  eulerStep,
  flowSchedule,
  modelTimestep,
} from "../../zimage/src/sampler.js";
import { type Qwen3Config, type Qwen3LayerWeights, permuteLayerForRope } from "../../zimage/src/text-encoder.js";
import { qwen3EncodeGpu } from "../../zimage/src/text-encoder-gpu.js";
import { FetchedDitWeights, FetchedShards, FetchedSafetensors } from "./fetch-weights.js";
import { decoderKernels, ditKernels, encoderKernels } from "./kernels-web.js";

const $ = <T extends HTMLElement>(id: string): T => document.getElementById(id) as T;
const canvas = $<HTMLCanvasElement>("canvas");
const statusLine = $<HTMLDivElement>("status");
const bar = $<HTMLDivElement>("bar").firstElementChild as HTMLElement;
const logBox = $<HTMLDivElement>("log");
const form = $<HTMLFormElement>("form");
const go = $<HTMLButtonElement>("go");

// On the page, not only in the log. Four rounds of this session were spent
// telling a fixed build from a cached one by counting line numbers in a pasted
// stack; the answer belongs somewhere it can be read at a glance.
$<HTMLElement>("build").textContent = `build ${BUILD_STAMP}`;

function log(message: string): void {
  logBox.textContent = `${logBox.textContent}${logBox.textContent ? "\n" : ""}${message}`;
  logBox.scrollTop = logBox.scrollHeight;
}
function status(message: string, fraction?: number): void {
  statusLine.textContent = message;
  if (fraction !== undefined) bar.style.width = `${Math.max(0, Math.min(1, fraction)) * 100}%`;
}
/** Lets the browser paint between dispatches — without it the tab is frozen for minutes. */
const breathe = (): Promise<void> => new Promise((resolve) => setTimeout(resolve, 0));

/** The same seeded noise `generate.ts` uses, so a seed means the same picture in both. */
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

/** `[3, H, W]` in [0, 1]-ish to the canvas. Clamped, because a VAE overshoots. */
function draw(planes: Float32Array, H: number, W: number): void {
  canvas.width = W;
  canvas.height = H;
  const ctx = canvas.getContext("2d")!;
  const image = ctx.createImageData(W, H);
  const hw = H * W;
  for (let p = 0; p < hw; p += 1) {
    for (let c = 0; c < 3; c += 1) {
      const v = (planes[c * hw + p]! + 1) / 2;
      image.data[p * 4 + c] = Math.max(0, Math.min(255, Math.round(v * 255)));
    }
    image.data[p * 4 + 3] = 255;
  }
  ctx.putImageData(image, 0, 0);
}

/** Replaced at bundle time — see `build.mjs`. */
declare const BUILD_STAMP: string;

const DIT_BASE = "/weights/dit";
const ENCODER_BASE = "/weights/text_encoder";
const VAE_BASE = "/weights/vae";

/**
 * The DiT's weights, on the GPU, for as long as the page lives.
 *
 * Held outside `generate` so a second prompt does not re-upload 6.17 GB.
 * `releaseDitWeights` is how they go, and nothing calls it yet — the page owns
 * them until it is closed, which is the whole point.
 */
const heldWeights = new Map<string, GPUBuffer>();

async function main(): Promise<void> {
  let runner: Awaited<ReturnType<typeof createBrowserRunner>>;
  let resident: Awaited<ReturnType<typeof createBrowserResidentDevice>>;
  try {
    runner = await createBrowserRunner();
    // A second device for the DiT. The per-dispatch `Runner` above still drives
    // the text encoder and the VAE; the DiT is the one that moves 48 GB a step
    // through it, and the one this replaces.
    resident = await createBrowserResidentDevice();
  } catch (error) {
    status("WebGPU is unavailable in this browser.");
    log(String(error));
    return;
  }
  const run = runner.run;
  const K = ditKernels;
  const EK = encoderKernels;

  // Printed so that "which bundle is this page running" is a question with an
  // answer. It cost an hour once.
  status("Reading manifests…");
  const [dit, encoderIndex, encoderConfig, vaeManifest, vocab] = await Promise.all([
    FetchedDitWeights.open(DIT_BASE, 48),
    fetch(`${ENCODER_BASE}/model.safetensors.index.json`).then((r) => (r.ok ? r.json() : null)),
    fetch(`${ENCODER_BASE}/config.json`).then((r) => r.json()),
    fetch(`${VAE_BASE}/manifest.json`).then((r) => r.json()),
    fetch("/weights/tokenizer/qwen-qwen3-4b.bpe-vocab.json").then((r) => r.json()),
  ]);

  const tokenizer = new ByteLevelBpeTokenizer(vocab as BpeVocab);
  const encoder = encoderIndex
    ? new FetchedShards(ENCODER_BASE, (encoderIndex as { weight_map: Record<string, string> }).weight_map)
    : await FetchedSafetensors.open(`${ENCODER_BASE}/model.safetensors`);

  // The VAE decoder is 198 MB and every layer of it is used, so it is fetched
  // whole rather than by tensor — the only weight here that is.
  status("Fetching the VAE decoder (198 MB)…");
  const vaeBlob = new Float32Array(await (await fetch(`${VAE_BASE}/decoder.bin`)).arrayBuffer());
  const vaeTable = (vaeManifest as { decoder: { name: string; offset: number; length: number }[] }).decoder;
  const vaeWeight = (name: string): Float32Array => {
    const e = vaeTable.find((t) => t.name === name);
    if (!e) throw new Error(`the VAE fixture has no "${name}"`);
    return vaeBlob.subarray(e.offset, e.offset + e.length);
  };
  const vc = (vaeManifest as { config: Record<string, never> }).config as unknown as {
    block_out_channels: number[]; layers_per_block: number; norm_num_groups: number;
    latent_channels: number; out_channels: number; scaling_factor: number; shift_factor: number;
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

  const ec = encoderConfig as Record<string, number>;
  const qwenCfg: Qwen3Config = {
    hiddenSize: ec.hidden_size!,
    numLayers: ec.num_hidden_layers!,
    numHeads: ec.num_attention_heads!,
    numKvHeads: ec.num_key_value_heads!,
    headDim: ec.head_dim!,
    ffnHidden: ec.intermediate_size!,
    rmsNormEps: ec.rms_norm_eps!,
    ropeTheta: ec.rope_theta!,
    vocabSize: ec.vocab_size!,
    // `hidden_states[-2]` — measured, see `examples/zimage/fixtures`.
    stopAfterLayer: ec.num_hidden_layers! - 2,
  };

  const ditRaw = dit.config as Record<string, number | number[]>;
  const patchSize = (ditRaw.all_patch_size as number[])[0]!;
  const ditCfg: DitConfig = {
    dim: ditRaw.dim as number,
    nHeads: ditRaw.n_heads as number,
    nLayers: ditRaw.n_layers as number,
    nRefinerLayers: ditRaw.n_refiner_layers as number,
    inChannels: ditRaw.in_channels as number,
    patchSize,
    capFeatDim: ditRaw.cap_feat_dim as number,
    normEps: ditRaw.norm_eps as number,
    ropeAxesDims: ditRaw.axes_dims as number[],
    ropeTheta: ditRaw.rope_theta as number,
    tScale: ditRaw.t_scale as number,
    adalnEmbedDim: 256,
    frequencyEmbeddingSize: 256,
    maxPeriod: 10000,
  };

  // The whole DiT, once, with a bar on it. Fetching per layer instead put 6.17
  // GB on the wire in every denoising step — the same bytes, eight times over.
  const gb = (bytes: number): string => `${(bytes / 1e9).toFixed(2)} GB`;
  // Into the browser's disk cache, not the heap. The first run pays the
  // download; every run after it — and every step within one — reads from disk
  // and touches the network not at all.
  await dit.preloadAll((done, total, bytes) => {
    status(`Caching the DiT — ${done}/${total} tensors, ${gb(bytes)}`, (done / total) * 0.98);
  });

  // What is left for everything else. Holding the model costs the heap it
  // costs, and the next allocation to fail is an activation — at 512x512 the
  // attention scores are 129 MB, and `Array buffer allocation failed` is what
  // that looks like from inside a dispatch. Reported rather than left for the
  // failure to explain, since it is the number that decides which sizes work.
  const budget = (performance as Performance & { memory?: { jsHeapSizeLimit: number } }).memory;
  if (budget) {
    log(`heap limit ${gb(budget.jsHeapSizeLimit)}, model ${gb(dit.bytesHeld)}`);
  }
  log(`DiT cached on disk: ${gb(dit.bytesHeld)} across ${dit.tensorCount} tensors`);

  status(`Ready — ${gb(dit.bytesHeld)} cached, nothing more to download.`, 1);
  go.disabled = false;

  // Picking a checkpoint fills in its own numbers. They stay editable — the
  // point is that the defaults are never the other model's.
  const variantSelect = $<HTMLSelectElement>("variant");
  const applyVariantDefaults = (): void => {
    const chosen = VARIANTS[variantSelect.value as Variant];
    $<HTMLInputElement>("steps").value = String(chosen.steps);
    $<HTMLInputElement>("guidance").value = String(chosen.guidance);
  };
  variantSelect.addEventListener("change", applyVariantDefaults);
  applyVariantDefaults();

  form.addEventListener("submit", (event) => {
    event.preventDefault();
    void generate();
  });

  async function generate(): Promise<void> {
    go.disabled = true;
    logBox.textContent = "";
    const started = performance.now();
    try {
      const prompt = $<HTMLInputElement>("prompt").value.trim() || "a photograph";
      const size = Number($<HTMLSelectElement>("size").value);
      const steps = Number($<HTMLInputElement>("steps").value);
      const seed = Number($<HTMLInputElement>("seed").value);
      const guidance = Number($<HTMLInputElement>("guidance").value);
      const shift = DEFAULT_FLOW_SHIFT;

      // The chat template upstream applies, expanded for one user turn.
      const formatted = `<|im_start|>user\n${prompt}<|im_end|>\n<|im_start|>assistant\n`;
      const ids = Int32Array.from(tokenizer.encode(formatted));
      log(`prompt -> ${ids.length} tokens`);
      log(`${variantSelect.value}: ${steps} steps, CFG ${guidance}`);

      // --- text encoder ---
      // Cached after the first prompt, so this is a one-time cost per page
      // rather than per generation.
      const encodeStart = performance.now();
      const capFeats = await qwen3EncodeGpu(
        run,
        EK,
        qwenCfg,
        {
          numLayers: qwenCfg.numLayers,
          // A couple of dozen rows of a 151936-row table, one Range request
          // each — see `FetchedSafetensors.readRows`.
          embed: (wanted) => encoder.readRows("model.embed_tokens.weight", wanted, qwenCfg.hiddenSize),
          layer: async (index) => {
            status(`Text encoder — layer ${index + 1}/${qwenCfg.stopAfterLayer + 1}`,
              (index / (qwenCfg.stopAfterLayer + 1)) * 0.35);
            const p = `model.layers.${index}.`;
            const [
              inputLayernorm, qProj, kProj, vProj, oProj, qNorm, kNorm, postNorm, gate, up, down,
            ] = await Promise.all([
              encoder.read(`${p}input_layernorm.weight`),
              encoder.read(`${p}self_attn.q_proj.weight`),
              encoder.read(`${p}self_attn.k_proj.weight`),
              encoder.read(`${p}self_attn.v_proj.weight`),
              encoder.read(`${p}self_attn.o_proj.weight`),
              encoder.read(`${p}self_attn.q_norm.weight`),
              encoder.read(`${p}self_attn.k_norm.weight`),
              encoder.read(`${p}post_attention_layernorm.weight`),
              encoder.read(`${p}mlp.gate_proj.weight`),
              encoder.read(`${p}mlp.up_proj.weight`),
              encoder.read(`${p}mlp.down_proj.weight`),
            ]);
            return permuteLayerForRope(
              {
                input_layernorm: inputLayernorm,
                q_proj: qProj,
                k_proj: kProj,
                v_proj: vProj,
                o_proj: oProj,
                q_norm: qNorm,
                k_norm: kNorm,
                post_attention_layernorm: postNorm,
                gate_proj: gate,
                up_proj: up,
                down_proj: down,
              },
              qwenCfg,
            );
          },
        },
        ids,
        breathe,
      );
      log(`text encoder ${((performance.now() - encodeStart) / 1000).toFixed(1)}s`);

      // The unconditional branch. Upstream uses **zeros** when no negative
      // prompt is given (`zimage_generate_image.py:579`) — not an empty string,
      // which would be a vector the model has seen.
      const negativeFeats = cfgEnabled(guidance) ? new Float32Array(capFeats.length) : null;

      // --- denoise ---
      // Upstream's own arithmetic: `vae_scale = 8 * 2`, then
      // `2 * (size // vae_scale)`, which lands on size/8 rounded so the latent
      // is divisible by 16.
      const latentSide = 2 * Math.floor(size / 16);
      let latents = seededNoise(ditCfg.inChannels * latentSide * latentSide, seed);
      const { timesteps, sigmas } = flowSchedule(steps, shift, NUM_TRAIN_TIMESTEPS);
      const capMask = new Float32Array(ids.length).fill(1);
      const imageTokens = (latentSide / patchSize) ** 2;
      log(`latent ${latentSide}x${latentSide}, ${imageTokens} image tokens`);

      const source: PackedWeightSource = {
        get: (name) => dit.get(name),
        packedQ8: (name) => dit.packedQ8(name),
        shapeOf: (name) => dit.shapeOf(name),
      };
      // The weights stay on the GPU across every step. Uploading them per
      // forward would be 6.17 GB eight times over, for bytes that never change
      // — which is what "resident" is supposed to mean and, in the first
      // version of `dit-resident.ts`, did not.
      const stepStats: ResidentDitStats = {
        dispatches: 0, submits: 0, buffersCreated: 0, poolSlots: 0, poolBytes: 0, weightBuffers: 0,
        uploadedBytes: 0, readBackBytes: 0,
      };

      for (let step = 0; step < steps; step += 1) {
        const stepStart = performance.now();
        const base = 0.35 + (step / steps) * 0.6;
        const t = modelTimestep(timesteps[step]!, NUM_TRAIN_TIMESTEPS);
        const forward = (feats: Float32Array): Promise<Float32Array> =>
          ditForwardResident(
          resident,
          K,
          ditCfg,
          source,
          {
            latent: latents,
            F: 1,
            H: latentSide,
            W: latentSide,
            t,
            capFeats: feats,
            capMask,
          },
          stepStats,
          // Only called while a layer's weights are still on disk — the first
          // generation, and nothing after it.
          async (prefix) => {
            await dit.preload(prefix);
            await breathe();
          },
          heldWeights,
          // Called every layer, always. The bar is about the forward, not
          // about whether anything had to be fetched for it.
          (label, done, total) => {
            status(`Step ${step + 1}/${steps} — ${label}`, base + (done / total) * (0.6 / steps));
          },
        );

        const conditional = await forward(capFeats);
        // `cond + scale * (cond - uncond)` — two forwards a step, which is what
        // CFG costs and the reason Turbo exists.
        const velocity = negativeFeats
          ? applyCfg(conditional, await forward(negativeFeats), guidance)
          : conditional;
        // Negated: Z-Image predicts negative noise.
        const negated = new Float32Array(velocity.length);
        for (let i = 0; i < velocity.length; i += 1) negated[i] = -velocity[i]!;
        latents = eulerStep(latents, negated, sigmas, step);
        log(
          `step ${step + 1}/${steps}  ${((performance.now() - stepStart) / 1000).toFixed(1)}s` +
            `  uploaded ${(stepStats.uploadedBytes / 1e9).toFixed(2)} GB`,
        );
      }

      // --- decode ---
      status("VAE decode…", 0.96);
      await breathe();
      // `decodeGpu` applies `latent / scalingFactor + shiftFactor` itself, so
      // the latent goes in as the sampler left it. Scaling here too would
      // double the shift and give an image with the right structure and the
      // wrong colours.
      const image = await decodeGpu(run, decoderKernels, decoderCfg, vaeWeight, latents, latentSide, latentSide);
      draw(image.data, image.H, image.W);

      const total = (performance.now() - started) / 1000;
      status(`Done — ${total.toFixed(1)}s`, 1);
      log(`total ${total.toFixed(1)}s`);
    } catch (error) {
      status("Failed — see the log.");
      // The stamp goes on the error, not only in the log above it. A pasted
      // stack is otherwise indistinguishable from the same stack out of a
      // cached build that has already been fixed — which cost four rounds of
      // "please reload" before anyone could tell the two apart.
      log(`[build ${BUILD_STAMP}] ${String(error)}`);
      console.error(`[build ${BUILD_STAMP}]`, error);
    } finally {
      go.disabled = false;
    }
  }
}

void main();
