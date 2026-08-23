/**
 * The GPU forward, against the same golden the CPU one is held to.
 *
 * Issue #166 stage 5. The bar and the golden are `verify-forward.ts`'s — the
 * point is that moving to the GPU changed nothing, so it is compared against
 * the model rather than against the CPU port. (Comparing the two ports to each
 * other would pass if both drifted the same way, which is the failure mode
 * `CLAUDE.local.md`'s "kernels are compared to the reference, never to each
 * other" is written against.)
 *
 * Issue #166 stage 2's notes on `verify-real-block.ts` covers one layer; this covers the
 * 34 of them plus everything around them — patchify, the timestep embedding,
 * the caption pad token, the position ids, the final layer.
 *
 * The intermediates are compared **in order, and it stops at the first one that
 * disagrees**. A 34-layer stack that is wrong somewhere and only reports its
 * last tensor tells you that something is wrong and nothing about where; the
 * golden carries hooks at four points precisely so that this does not happen.
 *
 * The bar is set against the model run with **identically quantized** weights,
 * and the full-precision run is printed beside it. Comparing against full
 * precision alone was tried first and is the wrong measurement: it reported
 * 3.8e-3 at the noise refiner, which is what q4 costs after two layers, not a
 * mistake in the port. One number cannot separate the two, so there are two.
 *
 *     npx tsx examples/zimage/src/verify-forward.ts --dit ~/zimage-q4
 */
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { type DitConfig, type DitTrace } from "./dit.js";
import { ditForwardGpu } from "./dit-gpu.js";
import { ditKernels } from "./kernels-node.js";
import { createRunner } from "../../../harness/wgsl.js";
import { LazyDitWeights } from "./weights-node.js";

interface ForwardManifest {
  config: {
    dim: number;
    n_heads: number;
    n_layers: number;
    n_refiner_layers: number;
    in_channels: number;
    cap_feat_dim: number;
    norm_eps: number;
    rope_theta: number;
    patchSize: number;
    latent: number;
    capLen: number;
    tScale: number;
    frequencyEmbeddingSize: number;
    maxPeriod: number;
    adalnEmbedDim: number;
    ropeAxesDims: number[];
  };
  quantizationCost: { relativeRms: number };
  tensors: { name: string; shape: number[]; offset: number; length: number }[];
}

function stats(got: Float32Array, want: Float32Array): { absMax: number; relRms: number } {
  let absMax = 0;
  let sqErr = 0;
  let sqWant = 0;
  for (let i = 0; i < want.length; i += 1) {
    const d = got[i]! - want[i]!;
    absMax = Math.max(absMax, Math.abs(d));
    sqErr += d * d;
    sqWant += want[i]! * want[i]!;
  }
  return { absMax, relRms: Math.sqrt(sqErr) / Math.sqrt(sqWant) };
}

async function main(): Promise<void> {
  const flag = process.argv.indexOf("--dit");
  const dit = flag >= 0 ? process.argv[flag + 1] : process.env.ZIMAGE_DIT_DIR;
  if (!dit) {
    console.error("verify-forward-gpu: pass --dit <dir>, the output of tools/convert_dit.py.");
    process.exit(2);
  }

  const fixtures = new URL("../fixtures/", import.meta.url);
  const golden = JSON.parse(
    readFileSync(fileURLToPath(new URL("forward.manifest.json", fixtures)), "utf8"),
  ) as ForwardManifest;
  const raw = readFileSync(fileURLToPath(new URL("forward.bin", fixtures)));
  const blob = new Float32Array(raw.buffer, raw.byteOffset, raw.byteLength / 4);
  const get = (name: string): Float32Array => {
    const entry = golden.tensors.find((t) => t.name === name);
    if (!entry) throw new Error(`golden has no tensor ${name}`);
    return blob.subarray(entry.offset, entry.offset + entry.length);
  };

  const c = golden.config;
  const cfg: DitConfig = {
    dim: c.dim,
    nHeads: c.n_heads,
    nLayers: c.n_layers,
    nRefinerLayers: c.n_refiner_layers,
    inChannels: c.in_channels,
    patchSize: c.patchSize,
    capFeatDim: c.cap_feat_dim,
    normEps: c.norm_eps,
    ropeAxesDims: c.ropeAxesDims,
    ropeTheta: c.rope_theta,
    tScale: c.tScale,
    adalnEmbedDim: c.adalnEmbedDim,
    frequencyEmbeddingSize: c.frequencyEmbeddingSize,
    maxPeriod: c.maxPeriod,
  };

  // Lazily, one tensor at a time. Reading the whole DiT dense was measured at
  // 20.5 GB resident and had not reached the first layer; 3.34 GB of q4/q8 is
  // 12 GB of f32, and building it in pieces doubles that at the concatenation.
  console.log(`reading the DiT from ${dit} lazily ...`);
  const weights = new LazyDitWeights(dit);

  const runner = await createRunner();
  if (!runner) {
    console.error("verify-forward-gpu: no WebGPU adapter available.");
    process.exit(2);
  }

  const trace: DitTrace = {};
  const ran = Date.now();
  const out = await ditForwardGpu(
    runner.run,
    ditKernels(),
    cfg,
    weights,
    {
      latent: get("x"),
      F: 1,
      H: c.latent,
      W: c.latent,
      t: get("t")[0]!,
      capFeats: get("capFeats"),
      capMask: get("capMask"),
    },
    trace,
  );
  console.log(`  forward in ${((Date.now() - ran) / 1000).toFixed(1)}s (GPU)`);

  // In order. The first disagreement is the informative one — everything after
  // it is downstream of the same mistake.
  const checkpoints: [string, Float32Array | undefined][] = [
    ["adalnInput", trace.adalnInput],
    ["afterNoiseRefiner", trace.afterNoiseRefiner],
    ["afterContextRefiner", trace.afterContextRefiner],
    ["afterLayer0", trace.afterLayer0],
    ["afterLayers", trace.afterLayers],
    ["output", out],
  ];

  // The port's own error on one layer with these weights is 8.7e-8 (stage 1).
  // Over 34 layers f32 accumulation drifts further, so the bar is 1e-4 — four
  // orders above stage 1's number and two below what quantization costs, which
  // is the gap that lets a failure here mean "the port", not "the format".
  const bar = 1e-4;
  let failed = false;
  console.log("");
  for (const [name, got] of checkpoints) {
    if (!got) {
      console.error(`  ${name}: not captured — the trace is not being filled`);
      failed = true;
      break;
    }
    const s = stats(got, get(name));
    // The same checkpoint from the full-precision run, when the golden carries
    // it — informational, never the bar.
    const denseName = `${name}Dense`;
    const dense = golden.tensors.some((t) => t.name === denseName) ? stats(got, get(denseName)) : null;
    const verdict = s.relRms < bar ? "ok" : "MISMATCH";
    console.log(
      `  ${name.padEnd(20)} vs quantized ${s.relRms.toExponential(3)}  ` +
        `vs dense ${dense ? dense.relRms.toExponential(3) : "   n/a   "}  ${verdict}`,
    );
    if (s.relRms >= bar) {
      console.error(`\nFirst mismatch at ${name}. Everything after it is downstream of the same cause.`);
      failed = true;
      break;
    }
  }

  if (failed) process.exit(1);
  console.log(
    `\nOK: the full forward matches the model, from the timestep embedding to the unpatchified latent.\n` +
      `The "vs dense" column is what q4/q8 costs over 34 layers, measured independently in torch as ` +
      `${golden.quantizationCost.relativeRms.toExponential(3)}.`,
  );
}

await main();
process.exit(0);
