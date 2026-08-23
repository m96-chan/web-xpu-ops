/**
 * The same VAE decoder, on the GPU.
 *
 * `decoder.ts` is the definition of correct and is unusably slow by design —
 * six nested loops, 64.9s for a 64x64 image. This runs the identical structure
 * through `harness/wgsl.ts`'s runner, dispatching the WGSL kernels each op
 * already ships. No new kernel is written: if this needed one, that would be a
 * missing op, and saying so is more useful than hiding it here.
 *
 * The two are checked against each other rather than one being trusted. Both
 * are checked against the model.
 *
 * Every dispatch's binding order and uniform layout is copied from that op's
 * own `wgsl.test.ts` rather than re-derived (rule 2) — those files are where
 * the layouts are pinned, and a second reading of them is a second chance to
 * get one wrong.
 */
import type { Runner } from "../../../harness/wgsl.js";
import { params } from "../../../harness/wgsl.js";
import { ACTIVATION } from "../../../ops/activation/index.js";
import { conv2dOutputSize } from "../../../ops/conv/index.js";
import { ELEMENTWISE } from "../../../ops/elementwise/index.js";
import { nearestUpsampleScale } from "../../../ops/upsample/index.js";
import type { DecoderConfig, Map4, Weights } from "./decoder.js";

/**
 * The kernels this decoder dispatches, by name.
 *
 * Passed in rather than read from disk, because the browser has no `fs` and
 * this file is the same code in both places — `decoder-kernels-node.ts` reads
 * them off the filesystem, the browser build inlines them at bundle time. The
 * alternative, a second copy of the decoder for the browser, is the copy that
 * drifts.
 */
export interface DecoderKernels {
  conv2d: string;
  upsample: string;
  groupNorm: string;
  activation: string;
  elementwise: string;
  matmul: string;
  scores: string;
  context: string;
}

/** The kernel names, so both loaders fetch the same set. */
export const DECODER_KERNEL_SOURCES: { key: keyof DecoderKernels; op: string; entry: string }[] = [
  { key: "conv2d", op: "conv", entry: "conv2d" },
  { key: "upsample", op: "upsample", entry: "kernel" },
  { key: "groupNorm", op: "group_norm", entry: "kernel" },
  { key: "activation", op: "activation", entry: "kernel" },
  { key: "elementwise", op: "elementwise", entry: "kernel" },
  { key: "matmul", op: "matmul", entry: "kernel" },
  { key: "scores", op: "attention", entry: "scores" },
  { key: "context", op: "attention", entry: "context" },
];

const WG = 256;

/**
 * WebGPU caps a dispatch dimension at 65535 workgroups, and exceeding it is not
 * an error a caller sees: the command buffer is invalidated, the submit does
 * nothing, and the output buffer comes back holding whatever it held before —
 * plausible numbers, silently wrong. Issue #112 records that failure mode, and
 * this decoder walked straight into it: the second-to-last up block is
 * 256 x 256 x 256 = 16,777,216 elements, which is ceil(/256) = 65,536
 * workgroups. One over.
 *
 * So elementwise-shaped dispatches are split. Both ops here are per-element and
 * carry no state between lanes, so a split is exact rather than an
 * approximation — the only thing that changes is how many submits it takes.
 */
const MAX_WORKGROUPS = 65535;
const MAX_ELEMS_PER_DISPATCH = MAX_WORKGROUPS * WG;

async function chunked(len: number, out: Float32Array, one: (offset: number, n: number) => Promise<Float32Array>): Promise<Float32Array> {
  for (let offset = 0; offset < len; offset += MAX_ELEMS_PER_DISPATCH) {
    const n = Math.min(MAX_ELEMS_PER_DISPATCH, len - offset);
    out.set(await one(offset, n), offset);
  }
  return out;
}
type Run = Runner["run"];

async function conv(run: Run, K: DecoderKernels, x: Map4, w: Weights, prefix: string, outC: number, k: number, pad: number): Promise<Map4> {
  const { Hout, Wout } = conv2dOutputSize({ H: x.H, W: x.W, KH: k, KW: k, padding: pad });
  // The kernel writes one lane per output column and rounds the dispatch up to
  // a whole workgroup, so the buffer carries that slack — same shape as the
  // op's own test.
  const width = Math.ceil(Wout / WG) * WG;
  const [out] = await run({
    code: K.conv2d,
    bindings: [
      { kind: "storage", data: x.data },
      { kind: "storage", data: w(`${prefix}.weight`) },
      { kind: "storage", data: w(`${prefix}.bias`) },
      { kind: "out", type: "f32", length: outC * Hout * Wout + (width - Wout) },
      {
        kind: "uniform",
        data: params([
          ["u32", x.C], ["u32", outC], ["u32", x.H], ["u32", x.W], ["u32", k], ["u32", k],
          ["u32", Hout], ["u32", Wout], ["u32", 1], ["u32", 1], ["u32", pad], ["u32", pad],
          ["u32", 1], ["u32", 1], ["u32", x.C], ["u32", outC],
        ]),
      },
    ],
    workgroups: [width / WG, Hout, outC],
  });
  return { data: (out as Float32Array).subarray(0, outC * Hout * Wout), C: outC, H: Hout, W: Wout };
}

async function groupNormSilu(run: Run, K: DecoderKernels, x: Map4, w: Weights, prefix: string, G: number): Promise<Map4> {
  const len = x.C * x.H * x.W;
  const [normed] = await run({
    code: K.groupNorm,
    bindings: [
      { kind: "storage", data: x.data },
      { kind: "storage", data: w(`${prefix}.weight`) },
      { kind: "storage", data: w(`${prefix}.bias`) },
      { kind: "out", type: "f32", length: len },
      { kind: "uniform", data: params([["u32", 1], ["u32", x.C], ["u32", x.H * x.W], ["u32", G], ["f32", 1e-6]]) },
    ],
    workgroups: [G],
  });
  const src = normed as Float32Array;
  const activated = await chunked(len, new Float32Array(len), async (offset, n) => {
    const [part] = await run({
      code: K.activation,
      bindings: [
        { kind: "storage", data: src.subarray(offset, offset + n) },
        { kind: "out", type: "f32", length: n },
        { kind: "uniform", data: params([["u32", n], ["u32", ACTIVATION.silu], ["f32", 1]]) },
      ],
      workgroups: [Math.ceil(n / WG)],
    });
    return part as Float32Array;
  });
  return { data: activated, C: x.C, H: x.H, W: x.W };
}

async function add(run: Run, K: DecoderKernels, a: Float32Array, b: Float32Array): Promise<Float32Array> {
  return chunked(a.length, new Float32Array(a.length), async (offset, n) => {
    const [part] = await run({
      code: K.elementwise,
      bindings: [
        { kind: "storage", data: a.subarray(offset, offset + n) },
        { kind: "storage", data: b.subarray(offset, offset + n) },
        { kind: "out", type: "f32", length: n },
        { kind: "uniform", data: params([["u32", n], ["u32", ELEMENTWISE.add]]) },
      ],
      workgroups: [Math.ceil(n / WG)],
    });
    return part as Float32Array;
  });
}

async function resnet(run: Run, K: DecoderKernels, x: Map4, w: Weights, prefix: string, outC: number, G: number, hasShortcut: boolean): Promise<Map4> {
  let h = await groupNormSilu(run, K, x, w, `${prefix}.norm1`, G);
  h = await conv(run, K, h, w, `${prefix}.conv1`, outC, 3, 1);
  h = await groupNormSilu(run, K, h, w, `${prefix}.norm2`, G);
  h = await conv(run, K, h, w, `${prefix}.conv2`, outC, 3, 1);
  const skip = hasShortcut ? await conv(run, K, x, w, `${prefix}.conv_shortcut`, outC, 1, 0) : x;
  return { data: await add(run, K, skip.data, h.data), C: outC, H: h.H, W: h.W };
}

/** `[C, H*W]` to `[H*W, C]`; the transpose stays on the CPU (see the README). */
function toTokens(x: Map4): Float32Array {
  const hw = x.H * x.W;
  const out = new Float32Array(hw * x.C);
  for (let c = 0; c < x.C; c += 1) for (let p = 0; p < hw; p += 1) out[p * x.C + c] = x.data[c * hw + p]!;
  return out;
}

function toMap(tokens: Float32Array, C: number, hw: number): Float32Array {
  const out = new Float32Array(C * hw);
  for (let p = 0; p < hw; p += 1) for (let c = 0; c < C; c += 1) out[c * hw + p] = tokens[p * C + c]!;
  return out;
}

async function linear(run: Run, K: DecoderKernels, x: Float32Array, w: Weights, prefix: string, rows: number, inC: number, outC: number): Promise<Float32Array> {
  const weight = w(`${prefix}.weight`);
  const bias = w(`${prefix}.bias`);
  const wT = new Float32Array(inC * outC);
  for (let o = 0; o < outC; o += 1) for (let i = 0; i < inC; i += 1) wT[i * outC + o] = weight[o * inC + i]!;
  const [y] = await run({
    code: K.matmul,
    bindings: [
      { kind: "storage", data: x },
      { kind: "storage", data: wT },
      { kind: "out", type: "f32", length: rows * outC },
      { kind: "uniform", data: params([["u32", rows], ["u32", outC], ["u32", inC]]) },
    ],
    // TILE = 16, as ops/matmul/wgsl.test.ts dispatches it.
    workgroups: [Math.ceil(outC / 16), Math.ceil(rows / 16)],
  });
  const out = y as Float32Array;
  for (let r = 0; r < rows; r += 1) for (let o = 0; o < outC; o += 1) out[r * outC + o] = out[r * outC + o]! + bias[o]!;
  return out;
}

async function midAttention(run: Run, K: DecoderKernels, x: Map4, w: Weights, prefix: string, G: number): Promise<Map4> {
  const hw = x.H * x.W;
  const [normed] = await run({
    code: K.groupNorm,
    bindings: [
      { kind: "storage", data: x.data },
      { kind: "storage", data: w(`${prefix}.group_norm.weight`) },
      { kind: "storage", data: w(`${prefix}.group_norm.bias`) },
      { kind: "out", type: "f32", length: x.C * hw },
      { kind: "uniform", data: params([["u32", 1], ["u32", x.C], ["u32", hw], ["u32", G], ["f32", 1e-6]]) },
    ],
    workgroups: [G],
  });
  const tokens = toTokens({ data: normed as Float32Array, C: x.C, H: x.H, W: x.W });

  const q = await linear(run, K, tokens, w, `${prefix}.to_q`, hw, x.C, x.C);
  const k = await linear(run, K, tokens, w, `${prefix}.to_k`, hw, x.C, x.C);
  const v = await linear(run, K, tokens, w, `${prefix}.to_v`, hw, x.C, x.C);

  const [probs] = await run({
    code: K.scores,
    bindings: [
      { kind: "storage", data: q },
      { kind: "storage", data: k },
      // A zero bias, [B=1, 1, 1] shaped: the smallest buffer that says nothing
      // is masked. Layout copied from ops/attention/wgsl.test.ts, not re-derived.
      { kind: "storage", data: new Float32Array(hw) },
      { kind: "out", type: "f32", length: hw * hw },
      {
        kind: "uniform",
        data: params([
          ["u32", 1], ["u32", hw], ["u32", hw], ["u32", x.C],
          ["f32", 1 / Math.sqrt(x.C)],
          ["u32", 0], ["i32", 0],
          ["u32", 1], ["u32", 1], ["u32", 1],
        ]),
      },
    ],
    workgroups: [hw, 1, 1],
  });
  const [attended] = await run({
    code: K.context,
    bindings: [
      { kind: "storage", data: probs as Float32Array },
      { kind: "storage", data: v },
      { kind: "out", type: "f32", length: hw * x.C },
      { kind: "uniform", data: params([["u32", 1], ["u32", hw], ["u32", hw], ["u32", x.C]]) },
    ],
    workgroups: [hw, 1, 1],
  });

  const projected = await linear(run, K, attended as Float32Array, w, `${prefix}.to_out.0`, hw, x.C, x.C);
  return { data: await add(run, K, x.data, toMap(projected, x.C, hw)), C: x.C, H: x.H, W: x.W };
}

export async function decodeGpu(run: Run, K: DecoderKernels, cfg: DecoderConfig, w: Weights, latent: Float32Array, latentH: number, latentW: number): Promise<Map4> {
  const G = cfg.normNumGroups;
  const unscaled = new Float32Array(latent.length);
  for (let i = 0; i < latent.length; i += 1) unscaled[i] = latent[i]! / cfg.scalingFactor + cfg.shiftFactor;

  let x: Map4 = { data: unscaled, C: cfg.latentChannels, H: latentH, W: latentW };
  const top = cfg.blockOutChannels[cfg.blockOutChannels.length - 1]!;
  x = await conv(run, K, x, w, "conv_in", top, 3, 1);

  x = await resnet(run, K, x, w, "mid_block.resnets.0", top, G, false);
  x = await midAttention(run, K, x, w, "mid_block.attentions.0", G);
  x = await resnet(run, K, x, w, "mid_block.resnets.1", top, G, false);

  const reversed = [...cfg.blockOutChannels].reverse();
  for (const [i, outC] of reversed.entries()) {
    for (let r = 0; r < cfg.layersPerBlock + 1; r += 1) {
      const inC = r === 0 ? x.C : outC;
      x = await resnet(run, K, x, w, `up_blocks.${i}.resnets.${r}`, outC, G, inC !== outC);
    }
    if (i !== cfg.blockOutChannels.length - 1) {
      const outH = x.H * 2;
      const outW = x.W * 2;
      const width = Math.ceil(outW / WG) * WG;
      const [up] = await run({
        code: K.upsample,
        bindings: [
          { kind: "storage", data: x.data },
          { kind: "out", type: "f32", length: x.C * outH * outW },
          {
            kind: "uniform",
            data: params([
              ["u32", x.H], ["u32", x.W], ["u32", outH], ["u32", outW],
              ["f32", nearestUpsampleScale(x.H, outH)], ["f32", nearestUpsampleScale(x.W, outW)],
              ["u32", 0], ["u32", 0],
            ]),
          },
        ],
        workgroups: [width / WG, outH, x.C],
      });
      x = { data: up as Float32Array, C: x.C, H: outH, W: outW };
      x = await conv(run, K, x, w, `up_blocks.${i}.upsamplers.0.conv`, outC, 3, 1);
    }
  }

  x = await groupNormSilu(run, K, x, w, "conv_norm_out", G);
  return conv(run, K, x, w, "conv_out", cfg.outChannels, 3, 1);
}
