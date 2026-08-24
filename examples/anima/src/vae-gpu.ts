/**
 * `wanVaeDecode` on the GPU, through the per-dispatch `Runner`.
 *
 * Issue #174. The same decomposition `vae.ts` establishes and the same
 * checkpoints, driven by `harness/wgsl.ts` — which is how `examples/zimage-vae`
 * runs its decoder in the browser, so this follows a path already known to work
 * rather than inventing one. Whether it is fast enough at Anima's shipped
 * resolution is a measurement, recorded in the README.
 *
 * **`RMS_norm` normalizes over channels, and `ops/rmsnorm` normalizes over the
 * last axis.** In `[C, H, W]` the channel axis is the outer one, so each norm
 * transposes to `[HW, C]`, normalizes, and transposes back. Two extra passes
 * over the tensor per norm, using ops that are already verified, against one
 * new kernel that would not be — `ops/rmsnorm`'s own `wgsl.test.ts` is worth
 * more here than the bandwidth.
 *
 * The eps convention differs and the difference is bounded rather than ignored.
 * Upstream is `F.normalize`, which divides by `max(||x||, 1e-12)` — a clamp on
 * the norm — where `ops/rmsnorm` adds eps inside the square root. Passing
 * `1e-12` makes them agree to within f32 everywhere the norm is not itself
 * around 1e-12, and agree exactly at an all-zero pixel, where both give zero.
 * The golden is what says whether that is good enough, and it reads 4.7e-7.
 */
import { params, type Runner } from "../../../harness/wgsl.js";
import { ACTIVATION } from "../../../ops/activation/index.js";
import { ELEMENTWISE } from "../../../ops/elementwise/index.js";
import { conv2dOutputSize } from "../../../ops/conv/index.js";
import { nearestUpsampleScale } from "../../../ops/upsample/index.js";
import { defaultScale } from "../../../ops/attention/index.js";
import type { VaeTrace, VaeWeights, WanVaeConfig } from "./vae.js";

export interface VaeKernels {
  conv2d: string;
  upsample: string;
  rmsnorm: string;
  transpose: string;
  activation: string;
  elementwise: string;
  scores: string;
  context: string;
}

/** The kernel names, so the Node and browser loaders fetch the same set. */
export const VAE_KERNEL_SOURCES: { key: keyof VaeKernels; op: string; entry: string }[] = [
  { key: "conv2d", op: "conv", entry: "conv2d" },
  { key: "upsample", op: "upsample", entry: "kernel" },
  { key: "rmsnorm", op: "rmsnorm", entry: "kernel" },
  { key: "transpose", op: "transpose", entry: "kernel" },
  { key: "activation", op: "activation", entry: "kernel" },
  { key: "elementwise", op: "elementwise", entry: "kernel" },
  { key: "scores", op: "attention", entry: "scores" },
  { key: "context", op: "attention", entry: "context" },
];

const WG = 256;
/** `ops/transpose`'s `workgroup_size(16, 16)`. */
const TRANSPOSE_TILE = 16;
const MAX_WORKGROUPS = 65535;
const MAX_ELEMS = MAX_WORKGROUPS * WG;

/**
 * `F.normalize`'s clamp, written as `ops/rmsnorm`'s eps. See the module comment
 * — the two differ only where the norm is itself around this magnitude.
 */
const NORM_EPS = 1e-12;

type Run = Runner["run"];

/**
 * Refuses a dispatch that exceeds the workgroup limit, naming the op.
 *
 * Going over is **not** an error the caller sees: the command buffer is
 * invalidated, the submit does nothing, and the output buffer comes back
 * holding whatever it held before — plausible numbers, silently wrong. Issue
 * #112 records that, and this decoder hit it twice: once in `rmsnorm` at
 * 256x256 and once before that in the DiT. Throwing here turns "the picture
 * looks a bit off" into a stack trace with a name in it.
 */
function guard<T extends readonly number[]>(op: string, workgroups: T): T {
  const over = workgroups.findIndex((n) => n > MAX_WORKGROUPS);
  if (over >= 0) {
    throw new Error(
      `wanVaeDecodeGpu: ${op} wants ${workgroups[over]} workgroups on dimension ${"xyz"[over]}, ` +
        `and the limit is ${MAX_WORKGROUPS} (issue #112).`,
    );
  }
  return workgroups;
}

interface Plane { data: Float32Array; C: number; H: number; W: number }

/**
 * Splits a per-element dispatch that would exceed the workgroup limit.
 *
 * At 832x1216 the head's input is 96 x 1216 x 832 = 97,124,352 elements, which
 * is 379,392 workgroups against a limit of 65,535. Issue #112's failure mode is
 * that exceeding it invalidates the command buffer silently and the output
 * buffer keeps whatever it held, so this is not optional at this resolution.
 */
async function chunked(len: number, one: (offset: number, n: number) => Promise<Float32Array>): Promise<Float32Array> {
  const out = new Float32Array(len);
  for (let offset = 0; offset < len; offset += MAX_ELEMS) {
    const n = Math.min(MAX_ELEMS, len - offset);
    out.set(await one(offset, n), offset);
  }
  return out;
}

async function conv(
  run: Run, K: VaeKernels, x: Plane, weight: Float32Array, bias: Float32Array, outC: number, k: number,
): Promise<Plane> {
  const pad = (k - 1) / 2;
  const { Hout, Wout } = conv2dOutputSize({ H: x.H, W: x.W, KH: k, KW: k, padding: pad });
  // The kernel writes one lane per output column and rounds up to a whole
  // workgroup, so the output buffer carries that slack — as the op's own test
  // does.
  const width = Math.ceil(Wout / WG) * WG;
  const [out] = await run({
    code: K.conv2d,
    bindings: [
      { kind: "storage", data: x.data },
      { kind: "storage", data: weight },
      { kind: "storage", data: bias },
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
    workgroups: guard("conv2d", [width / WG, Hout, outC]),
  });
  return { data: (out as Float32Array).subarray(0, outC * Hout * Wout), C: outC, H: Hout, W: Wout };
}

/** `[rows, cols]` to `[cols, rows]`. */
async function transpose(run: Run, K: VaeKernels, input: Float32Array, rows: number, cols: number): Promise<Float32Array> {
  const [out] = await run({
    code: K.transpose,
    bindings: [
      { kind: "storage", data: input },
      { kind: "out", type: "f32", length: rows * cols },
      { kind: "uniform", data: params([["u32", rows], ["u32", cols]]) },
    ],
    // `ops/transpose`'s kernel is `workgroup_size(16, 16)`, and x walks the
    // input's columns while y walks its rows — the grid is the shape of the
    // *input* in tiles, not of the output. A one-dimensional dispatch here
    // silently transposes only the first 16 rows: it read 8.118e-1.
    workgroups: guard("transpose", [Math.ceil(cols / TRANSPOSE_TILE), Math.ceil(rows / TRANSPOSE_TILE)]),
  });
  return out as Float32Array;
}

/**
 * `RMS_norm` over channels — see the module comment for why it transposes.
 *
 * **Split by rows**, because `ops/rmsnorm` dispatches one workgroup per row and
 * a row here is a pixel. A 256x256 image is 65,536 of them, one over the limit;
 * a 832x1216 one is 1,011,712. Rows carry no state between them, so the split
 * is exact. This was not caught by the 8x8 golden and cost an invalid command
 * buffer at the first real resolution — issue #112's shape exactly, which is
 * why `guard` below refuses rather than letting a submit quietly do nothing.
 */
async function rmsNorm(run: Run, K: VaeKernels, x: Plane, gamma: Float32Array): Promise<Plane> {
  const hw = x.H * x.W;
  const rows = await transpose(run, K, x.data, x.C, hw);
  const normed = new Float32Array(hw * x.C);
  for (let row0 = 0; row0 < hw; row0 += MAX_WORKGROUPS) {
    const count = Math.min(MAX_WORKGROUPS, hw - row0);
    const [part] = await run({
      code: K.rmsnorm,
      bindings: [
        { kind: "storage", data: rows.subarray(row0 * x.C, (row0 + count) * x.C) },
        { kind: "storage", data: gamma },
        { kind: "out", type: "f32", length: count * x.C },
        { kind: "uniform", data: params([["u32", count], ["u32", x.C], ["f32", NORM_EPS]]) },
      ],
      workgroups: guard("rmsnorm", [count]),
    });
    normed.set(part as Float32Array, row0 * x.C);
  }
  return { data: await transpose(run, K, normed, hw, x.C), C: x.C, H: x.H, W: x.W };
}

async function silu(run: Run, K: VaeKernels, x: Plane): Promise<Plane> {
  const data = await chunked(x.data.length, async (offset, n) => {
    const [part] = await run({
      code: K.activation,
      bindings: [
        { kind: "storage", data: x.data.subarray(offset, offset + n) },
        { kind: "out", type: "f32", length: n },
        { kind: "uniform", data: params([["u32", n], ["u32", ACTIVATION.silu], ["f32", 1]]) },
      ],
      workgroups: [Math.ceil(n / WG)],
    });
    return part as Float32Array;
  });
  return { ...x, data };
}

async function add(run: Run, K: VaeKernels, a: Float32Array, b: Float32Array): Promise<Float32Array> {
  return chunked(a.length, async (offset, n) => {
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

/**
 * The last temporal tap of a `CausalConv3d` weight — see `vae.ts` for why that
 * is the whole of it at `T = 1`.
 *
 * Sliced on the host rather than on the device: it happens once per weight per
 * decode and moves a third of the bytes the dispatch would otherwise upload.
 */
function lastTap(weight: Float32Array, Cout: number, Cin: number, kernel: number): Float32Array {
  const perTap = Cout * Cin * kernel * kernel;
  const taps = weight.length / perTap;
  if (!Number.isInteger(taps) || taps < 1) {
    throw new Error(`lastTap: weight of ${weight.length} is not [${Cout}, ${Cin}, KT, ${kernel}, ${kernel}]`);
  }
  if (taps === 1) return weight;
  const out = new Float32Array(perTap);
  for (let o = 0; o < Cout; o += 1) {
    for (let i = 0; i < Cin; i += 1) {
      const from = ((o * Cin + i) * taps + (taps - 1)) * kernel * kernel;
      out.set(weight.subarray(from, from + kernel * kernel), (o * Cin + i) * kernel * kernel);
    }
  }
  return out;
}

async function causalConv(
  run: Run, K: VaeKernels, x: Plane, w: VaeWeights, prefix: string, outC: number, k: number,
): Promise<Plane> {
  return conv(run, K, x, lastTap(w.get(`${prefix}.weight`), outC, x.C, k), w.get(`${prefix}.bias`), outC, k);
}

async function residualBlock(
  run: Run, K: VaeKernels, x: Plane, w: VaeWeights, prefix: string, outDim: number,
): Promise<Plane> {
  let h = await rmsNorm(run, K, x, w.get(`${prefix}residual.0.gamma`));
  h = await silu(run, K, h);
  h = await causalConv(run, K, h, w, `${prefix}residual.2`, outDim, 3);
  h = await rmsNorm(run, K, h, w.get(`${prefix}residual.3.gamma`));
  h = await silu(run, K, h);
  h = await causalConv(run, K, h, w, `${prefix}residual.6`, outDim, 3);

  const shortcut = w.has(`${prefix}shortcut.weight`)
    ? await causalConv(run, K, x, w, `${prefix}shortcut`, outDim, 1)
    : x;
  return { data: await add(run, K, h.data, shortcut.data), C: outDim, H: x.H, W: x.W };
}

/**
 * `AttentionBlock` — one head, the sequence being every pixel and the head
 * dimension every channel.
 *
 * At Anima's shipped resolution the middle block still sees the latent's
 * 104x152, which is 15,808 pixels: the score matrix is 250 million entries, or
 * a gigabyte in f32. That is the largest single allocation in the decoder and
 * the reason this is the one place worth watching if a device runs out.
 */
async function attentionBlock(run: Run, K: VaeKernels, x: Plane, w: VaeWeights, prefix: string): Promise<Plane> {
  const { C, H, W } = x;
  const hw = H * W;
  const normed = await rmsNorm(run, K, x, w.get(`${prefix}norm.gamma`));
  const qkv = await conv(run, K, normed, w.get(`${prefix}to_qkv.weight`), w.get(`${prefix}to_qkv.bias`), C * 3, 1);

  // `[3C, HW]` to three `[HW, C]`. Read straight out of the channel-major
  // layout rather than through `transpose`: the first version transposed the
  // whole `[3C, HW]` and then indexed the result as though it were still
  // channel-major, which is the same arithmetic written twice and disagreeing
  // with itself. It read 7.919e-1.
  const slice = (part: number): Float32Array => {
    const out = new Float32Array(hw * C);
    for (let i = 0; i < hw; i += 1) {
      for (let c = 0; c < C; c += 1) out[i * C + c] = qkv.data[(part * C + c) * hw + i]!;
    }
    return out;
  };

  const q = slice(0), k = slice(1), v = slice(2);

  /**
   * Split by **query rows**, because the score matrix does not fit otherwise.
   *
   * At 832x1216 the middle block still sees the latent's 104x152, so `hw` is
   * 15,808 and the scores are 250 million floats — 1.0 GB, against a
   * `maxStorageBufferBindingSize` this adapter reports in the hundreds of MiB.
   * A query row attends to every key and to nothing else, so splitting rows is
   * exact: each chunk softmaxes over the full key axis and writes its own slice.
   * Only the dispatch count changes. Same budget as `examples/zimage-vae`,
   * which met this first.
   */
  const ROW_BUDGET = 256 * 1024 * 1024;
  const rowsPerChunk = Math.max(1, Math.min(hw, Math.floor(ROW_BUDGET / (hw * 4))));
  const attended = new Float32Array(hw * C);

  for (let row0 = 0; row0 < hw; row0 += rowsPerChunk) {
    const rows = Math.min(rowsPerChunk, hw - row0);
    const [probs] = await run({
      code: K.scores,
      bindings: [
        { kind: "storage", data: q.subarray(row0 * C, (row0 + rows) * C) },
        { kind: "storage", data: k },
        // A zero bias: the smallest buffer that says nothing is masked. Layout
        // copied from `ops/attention/wgsl.test.ts`, not re-derived.
        { kind: "storage", data: new Float32Array(hw) },
        { kind: "out", type: "f32", length: rows * hw },
        {
          kind: "uniform",
          data: params([
            ["u32", 1], ["u32", rows], ["u32", hw], ["u32", C],
            ["f32", defaultScale(C)],
            ["u32", 0], ["i32", 0],
            ["u32", 1], ["u32", 1], ["u32", 1],
          ]),
        },
      ],
      workgroups: guard("attention scores", [rows, 1, 1]),
    });
    const [chunk] = await run({
      code: K.context,
      bindings: [
        { kind: "storage", data: probs as Float32Array },
        { kind: "storage", data: v },
        { kind: "out", type: "f32", length: rows * C },
        { kind: "uniform", data: params([["u32", 1], ["u32", rows], ["u32", hw], ["u32", C]]) },
      ],
      workgroups: guard("attention context", [rows, 1, 1]),
    });
    attended.set(chunk as Float32Array, row0 * C);
  }

  const planes = await transpose(run, K, attended, hw, C);
  const projected = await conv(
    run, K, { data: planes, C, H, W }, w.get(`${prefix}proj.weight`), w.get(`${prefix}proj.bias`), C, 1,
  );
  return { data: await add(run, K, projected.data, x.data), C, H, W };
}

async function resample(run: Run, K: VaeKernels, x: Plane, w: VaeWeights, prefix: string, outDim: number): Promise<Plane> {
  const { C, H, W } = x;
  const outH = H * 2, outW = W * 2;
  // The kernel writes one lane per output column and rounds up to a workgroup.
  const width = Math.ceil(outW / WG) * WG;

  /**
   * Split by **channels**, for the reason `examples/zimage-vae` splits it: the
   * output of the last upsample is 96 x 1216 x 832 = 0.39 GB in one buffer,
   * against a binding size in the hundreds of MiB. Nearest-neighbour reads and
   * writes one channel at a time with nothing shared between them, so a
   * per-channel split is exact.
   */
  const CHANNEL_BUDGET = 256 * 1024 * 1024;
  const perChunk = Math.max(1, Math.min(C, Math.floor(CHANNEL_BUDGET / (outH * outW * 4))));
  const upsampled = new Float32Array(C * outH * outW);
  for (let c0 = 0; c0 < C; c0 += perChunk) {
    const channels = Math.min(perChunk, C - c0);
    const [part] = await run({
      code: K.upsample,
      bindings: [
        { kind: "storage", data: x.data.subarray(c0 * H * W, (c0 + channels) * H * W) },
        { kind: "out", type: "f32", length: channels * outH * outW },
        {
          kind: "uniform",
          data: params([
            ["u32", H], ["u32", W], ["u32", outH], ["u32", outW],
            ["f32", nearestUpsampleScale(H, outH)], ["f32", nearestUpsampleScale(W, outW)],
            ["u32", 0], ["u32", 0],
          ]),
        },
      ],
      workgroups: guard("upsample", [width / WG, outH, channels]),
    });
    upsampled.set(part as Float32Array, c0 * outH * outW);
  }
  return conv(
    run, K, { data: upsampled, C, H: outH, W: outW },
    w.get(`${prefix}resample.1.weight`), w.get(`${prefix}resample.1.bias`), outDim, 3,
  );
}

/** `wanVaeDecode`'s signature, on the GPU. Checkpoints are the same names. */
export async function wanVaeDecodeGpu(
  run: Run,
  K: VaeKernels,
  cfg: WanVaeConfig,
  weights: VaeWeights,
  latent: Float32Array,
  H: number,
  W: number,
  trace?: VaeTrace,
  onProgress?: (label: string, done: number, total: number) => void,
): Promise<Float32Array> {
  if (latent.length !== cfg.zDim * H * W) {
    throw new Error(`wanVaeDecodeGpu: latent is ${latent.length} values, expected ${cfg.zDim} x ${H} x ${W}`);
  }
  const dims = [cfg.dim * cfg.dimMult[cfg.dimMult.length - 1]!, ...[...cfg.dimMult].reverse().map((m) => cfg.dim * m)];
  const stages = 2 + 3 + (cfg.dimMult.length * (cfg.numResBlocks + 1) + cfg.dimMult.length - 1) + 1;
  let done = 0;
  const step = (label: string): void => {
    done += 1;
    onProgress?.(label, done, stages);
  };

  step("conv2");
  let x = await causalConv(run, K, { data: latent, C: cfg.zDim, H, W }, weights, "conv2", cfg.zDim, 1);

  step("conv1");
  x = await causalConv(run, K, x, weights, "decoder.conv1", dims[0]!, 3);
  if (trace) trace.afterConv1 = x.data.slice();

  step("middle");
  x = await residualBlock(run, K, x, weights, "decoder.middle.0.", dims[0]!);
  x = await attentionBlock(run, K, x, weights, "decoder.middle.1.");
  if (trace) trace.afterAttention = x.data.slice();
  x = await residualBlock(run, K, x, weights, "decoder.middle.2.", dims[0]!);
  if (trace) trace.afterMiddle = x.data.slice();

  let at = 0;
  for (let stage = 0; stage < cfg.dimMult.length; stage += 1) {
    const outDim = dims[stage + 1]!;
    for (let block = 0; block < cfg.numResBlocks + 1; block += 1) {
      step(`upsample block ${at}`);
      x = await residualBlock(run, K, x, weights, `decoder.upsamples.${at}.`, outDim);
      at += 1;
    }
    if (stage !== cfg.dimMult.length - 1) {
      step(`resample ${at}`);
      x = await resample(run, K, x, weights, `decoder.upsamples.${at}.`, outDim / 2);
      if (trace) trace[`afterUpsample${at}`] = x.data.slice();
      at += 1;
    }
  }

  step("head");
  x = await rmsNorm(run, K, x, weights.get("decoder.head.0.gamma"));
  x = await silu(run, K, x);
  x = await causalConv(run, K, x, weights, "decoder.head.2", 3, 3);
  return x.data;
}
