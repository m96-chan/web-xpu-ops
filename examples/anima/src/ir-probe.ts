/**
 * What an IR would have to carry, measured by running two shapes and diffing.
 *
 * Issue #185. `dit-resident.ts` is 1,035 lines of TypeScript, and the question
 * is which of it is structure — the same for every input — and which is
 * arithmetic on the shape. Designing an IR without knowing that ratio means
 * guessing at how much of a language it needs.
 *
 * So: record every dispatch at two resolutions and compare them position by
 * position.
 *
 *   - the same **kernel** in the same position, at both  -> structure
 *   - workgroups or uniforms that **differ**             -> shape-derived
 *   - a **different length**, or kernels out of step     -> control flow
 *
 * The third is the expensive one. An IR that only has to hold constants and
 * shape expressions is a data format; one that has to hold branches is a
 * language, and the difference is most of the work.
 *
 *     npx tsx examples/anima/src/ir-probe.ts --dit ~/anima-q8
 */
import { createResidentDevice } from "../../../harness/resident.js";
import { ditKernels } from "../../zimage/src/kernels-node.js";
import type { AnimaConfig } from "./dit.js";
import { animaForwardResident, releaseAnimaWeights } from "./dit-resident.js";
import { LATENT } from "./sampler.js";
import { loadAnimaSubset, withRopePermutation } from "./weights-node.js";
import { startIrTrace, stopIrTrace, type IrDispatch, type IrTrace } from "./ir-trace.js";
import { readFileSync } from "node:fs";

const flag = process.argv.indexOf("--dit");
const ditDir = flag >= 0 ? process.argv[flag + 1]! : process.env.ANIMA_DIT_DIR!;
if (!ditDir) {
  console.error("ir-probe: pass --dit <dir>");
  process.exit(2);
}

const manifest = JSON.parse(readFileSync(`${ditDir}/dit.manifest.json`, "utf8")) as {
  config: Record<string, number | boolean>;
};
const c = manifest.config as unknown as {
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

const device = await createResidentDevice();
if (!device) {
  console.error("ir-probe: no WebGPU adapter available.");
  process.exit(2);
}
const weights = withRopePermutation(
  loadAnimaSubset(ditDir), cfg.numHeads, cfg.modelChannels / cfg.numHeads, cfg.modelChannels,
);
const held = new Map<string, GPUBuffer>();

/** One forward at `latent` x `latent`, traced. */
async function traceAt(latent: number, w = latent): Promise<IrTrace> {
  const input = {
    latent: new Float32Array(LATENT.channels * latent * w),
    T: 1, H: latent, W: w, t: 0.7,
    context: new Float32Array(512 * cfg.crossattnEmbChannels),
  };
  const trace = startIrTrace();
  await animaForwardResident(device!, ditKernels(), cfg, weights, input, undefined, held);
  stopIrTrace();
  return trace;
}

// The first forward uploads the weights; tracing it would mix that in.
// The pair is a flag, so the #112 chunking can be provoked. 104x152 is the
// shipped size and 3,952 tokens -- far past the 65,535-workgroup limit for the
// ops that dispatch one workgroup per row.
const arg = (n: string, d: number): number => {
  const at = process.argv.indexOf(`--${n}`);
  return at >= 0 ? Number(process.argv[at + 1]) : d;
};
const [la, lb] = [arg("a", 32), arg("b", 64)];
await traceAt(la);
const a = await traceAt(la);
const b = await traceAt(lb);

const tokens = (l: number): number => (l / cfg.patchSpatial) ** 2;
let tokensA = tokens(la), tokensB = tokens(lb);
console.log(`\n${a.dispatches.length} dispatches at ${tokensA} tokens, ${b.dispatches.length} at ${tokensB}\n`);

if (a.dispatches.length !== b.dispatches.length) {
  console.log("**The two differ in length.** The op sequence itself depends on the shape,");
  console.log("which is control flow an IR has to express rather than a table it can hold.\n");
}

const n = Math.min(a.dispatches.length, b.dispatches.length);
const same = (x: number[], y: number[]): boolean => x.length === y.length && x.every((v, i) => v === y[i]);

let kernelMismatch = 0;
const constantWg: number[] = [];
const varyingWg: number[] = [];
const constantUniform: number[] = [];
const varyingUniform: number[] = [];
const edgesStable: number[] = [];
const edgesVarying: number[] = [];

for (let i = 0; i < n; i += 1) {
  const p = a.dispatches[i]!;
  const q = b.dispatches[i]!;
  if (p.kernel !== q.kernel) {
    kernelMismatch += 1;
    continue;
  }
  (same(p.workgroups, q.workgroups) ? constantWg : varyingWg).push(i);
  (same(p.uniforms, q.uniforms) ? constantUniform : varyingUniform).push(i);
  (same(p.buffers, q.buffers) ? edgesStable : edgesVarying).push(i);
}

const pct = (k: number): string => `${((k / n) * 100).toFixed(1)}%`;
console.log(`  kernels in the same position     ${n - kernelMismatch}/${n}  (${pct(n - kernelMismatch)})`);
console.log(`  workgroups identical             ${constantWg.length}/${n}  (${pct(constantWg.length)})`);
console.log(`  workgroups shape-derived         ${varyingWg.length}/${n}  (${pct(varyingWg.length)})`);
console.log(`  uniforms identical               ${constantUniform.length}/${n}  (${pct(constantUniform.length)})`);
console.log(`  uniforms shape-derived           ${varyingUniform.length}/${n}  (${pct(varyingUniform.length)})`);
console.log(`  buffer bindings identical        ${edgesStable.length}/${n}  (${pct(edgesStable.length)})`);

/** How a varying number relates to the token count, if simply. */
function relation(x: number, y: number, tx: number, ty: number): string {
  if (y === x) return "constant";
  for (const [name, f] of [
    ["tokens", (t: number) => t],
    ["ceilDiv(tokens,32)", (t: number) => Math.ceil(t / 32)],
    ["ceilDiv(tokens,64)", (t: number) => Math.ceil(t / 64)],
    ["ceilDiv(tokens,128)", (t: number) => Math.ceil(t / 128)],
    ["ceilDiv(tokens,256)", (t: number) => Math.ceil(t / 256)],
    ["tokens^2", (t: number) => t * t],
  ] as [string, (t: number) => number][]) {
    if (f(tx) === x && f(ty) === y) return name;
    for (const k of [1, 2, 3, 4, 8, 16]) if (f(tx) * k === x && f(ty) * k === y) return `${name} * ${k}`;
  }
  return "**not a simple function of the token count**";
}

console.log("\n  how the varying workgroup counts relate to the token count:");
const seen = new Map<string, number>();
for (const i of varyingWg) {
  const p = a.dispatches[i]!;
  const q = b.dispatches[i]!;
  for (let d = 0; d < p.workgroups.length; d += 1) {
    if (p.workgroups[d] === q.workgroups[d]) continue;
    const r = `${p.kernel}[${"xyz"[d]}] = ${relation(p.workgroups[d]!, q.workgroups[d]!, tokensA, tokensB)}`;
    seen.set(r, (seen.get(r) ?? 0) + 1);
  }
}
for (const [r, k] of [...seen.entries()].sort((x, y) => y[1] - x[1])) {
  console.log(`    ${String(k).padStart(4)}x  ${r}`);
}

// Where the two sequences differ, by kernel. A length that depends on the
// shape is control flow, and naming which op introduces it says whether it is
// one special case or the whole design.
const countBy = (t: IrTrace): Map<string, number> => {
  const m = new Map<string, number>();
  for (const d of t.dispatches) m.set(d.kernel, (m.get(d.kernel) ?? 0) + 1);
  return m;
};
const ca = countBy(a);
const cb = countBy(b);
console.log("\n  dispatches per kernel:");
for (const k of new Set([...ca.keys(), ...cb.keys()])) {
  const x = ca.get(k) ?? 0;
  const y = cb.get(k) ?? 0;
  console.log(`    ${k.padEnd(16)} ${String(x).padStart(5)} -> ${String(y).padStart(5)}  ${x === y ? "" : "**differs**"}`);
}

console.log("\n  distinct kernels:", new Set(a.dispatches.map((d) => d.kernel)).size);
console.log("  distinct buffers bound:", new Set(a.dispatches.flatMap((d: IrDispatch) => d.buffers)).size);

releaseAnimaWeights(held);
device.destroy();
