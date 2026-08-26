/**
 * Qwen3-VL's conditioner on the GPU — the third model R2V needs resident.
 *
 * Issue #212. `text-encoder.ts` and `vision.ts` are the same arithmetic on the
 * CPU, held to `transformers`' own models at 3.576e-7 and 1.192e-7; this is the
 * resident version, built the way `examples/h3-dit/src/model-gpu.ts` is so a
 * divergence between the two is readable.
 *
 * **26.25 GB of int8**, and it is a third of a generation rather than the whole
 * of one: the DiT is 20.08 and the VAE decoder 2.43, which is 48.7 together and
 * fits on no card. The three run in sequence.
 *
 * **No new kernel.** `matmulQ8`, `rmsnorm`, `layernorm`, `activation`,
 * `elementwise`, `permute`, `rope`'s axes entry and `flash_attention` cover
 * both stacks.
 *
 * ## Grouped-query attention, by expansion
 *
 * The text stack has **64 query heads over 8 key/value heads**, and `ops/gqa`
 * has kernels for exactly that. This does not use them: it **repeats** the 8
 * heads into 64 with buffer copies and dispatches the same flash attention the
 * DiT uses.
 *
 * That is a trade, made because the sequence is short. A `ref2va` presentation
 * is a few hundred tokens, so 64 heads of keys is single-digit megabytes and
 * 128 copies a layer are cheap to record. At a decoder's sequence lengths the
 * trade reverses and `ops/gqa` is the answer — its own doc has the numbers.
 *
 * ## Deepstack
 *
 * The vision tower taps three of its layers, and those outputs are **added into
 * the text stack's hidden states at the visual rows** of its first three
 * layers. Not concatenated, not prepended: added, in place, at the rows a
 * vision block occupies. A port that skipped it produces a well-formed
 * conditioning that has seen the image once instead of four times.
 */
import type { ResidentDevice, ResidentOp } from "../../../harness/resident.js";
import { params } from "../../../harness/wgsl.js";
import { ACTIVATION } from "../../../ops/activation/index.js";
import { ELEMENTWISE } from "../../../ops/elementwise/index.js";
import { FLASH_GENERATION, flashGrid } from "../../../ops/flash_attention/index.js";
import { matmulQ8Grid } from "../../../ops/matmul/index.js";
import { mropePositions, type PositionGrid } from "./text-encoder.js";
import {
  interpolatePositionEmbedding, toMergeBlockOrder, visionCoordinates, visionPositions, type Grid,
} from "./vision.js";

export const CONDITIONER_KERNEL_SOURCES: { key: keyof ConditionerKernels; op: string; entry: string }[] = [
  { key: "matmul", op: "matmul", entry: "kernel" },
  { key: "matmulQ8", op: "matmul", entry: "q8" },
  { key: "rmsnorm", op: "rmsnorm", entry: "kernel" },
  { key: "layernorm", op: "layernorm", entry: "kernel" },
  { key: "activation", op: "activation", entry: "kernel" },
  { key: "elementwise", op: "elementwise", entry: "kernel" },
  { key: "rows", op: "elementwise", entry: "rows" },
  { key: "ropeAxes", op: "rope", entry: "axes" },
  { key: "permute", op: "permute", entry: "kernel" },
  { key: "flashAttention", op: "flash_attention", entry: FLASH_GENERATION },
];

export interface ConditionerKernels {
  matmul: string;
  matmulQ8: string;
  rmsnorm: string;
  layernorm: string;
  activation: string;
  elementwise: string;
  rows: string;
  ropeAxes: string;
  permute: string;
  flashAttention: string;
}

export interface ConditionerManifest {
  textConfig: {
    hidden_size: number; intermediate_size: number; num_hidden_layers: number;
    num_attention_heads: number; num_key_value_heads: number; head_dim: number;
    rms_norm_eps: number; rope_theta: number;
    rope_scaling: { mrope_section: [number, number, number] };
  };
  visionConfig: {
    hidden_size: number; intermediate_size: number; num_heads: number; depth: number;
    in_channels: number; patch_size: number; spatial_merge_size: number;
    temporal_patch_size: number; out_hidden_size: number;
    num_position_embeddings: number; deepstack_visual_indexes: number[];
  };
  textEncoderLayer: number;
  layers: number;
  dtype: string;
  tensors: { name: string; shape: number[]; offset: number; count: number; kind?: string }[];
  visionBytes: number;
  residentBytes: number;
}

const WG = 256;
const MM_BM = 64;
const MM_BN = 128;
const UNIFORM_BYTES = 128;
const LAYERNORM_EPS = 1e-6;
/** 65,535 everywhere measured — see issue #211. */
const MAX_WORKGROUPS = 65535;

interface Mat { buffer: GPUBuffer; rows: number; cols: number }

export interface ConditionerRequest {
  /** The presentation's token ids. */
  tokenIds: Int32Array;
  /** `(t, h, w)` per token, as `Qwen3VLModel` builds it. */
  positions: PositionGrid;
  /** Patches from `./processor.ts`, concatenated in reference order. */
  patches: Float32Array;
  /** One per reference, in the same order. */
  grids: Grid[];
  /** Rows of the token stream a vision block occupies, as `[start, length]` runs. */
  visualRuns: [number, number][];
}

export class ConditionerGpu {
  private readonly weights = new Map<string, GPUBuffer>();
  private readonly pool = new Map<number, GPUBuffer[]>();
  private readonly lent: GPUBuffer[] = [];
  private readonly lentUniforms: GPUBuffer[] = [];
  private readonly freeUniforms: GPUBuffer[] = [];

  submitMs = 0;
  readbackMs = 0;
  recordMs = 0;
  dispatches = 0;

  private constructor(
    private readonly device: ResidentDevice,
    private readonly kernels: ConditionerKernels,
    readonly manifest: ConditionerManifest,
  ) {}

  static async create(
    device: ResidentDevice,
    kernels: ConditionerKernels,
    manifest: ConditionerManifest,
    read: (offsetBytes: number, byteLength: number) => Uint8Array | Promise<Uint8Array>,
    /**
     * The two tables the forward reads on the **host**.
     *
     * `embed_tokens` is a 151,936-row gather, one row per token — no kernel
     * here gathers, and the alternative is a 0.78 GB pass to fetch 78 KB.
     * `visual.pos_embed` is bilinearly resampled per image, which is host
     * arithmetic on 2.6 M values.
     *
     * Required rather than settable: a forward that ran without them would
     * fail forty layers in, and the caller reading them is one line.
     */
    hostTables: { embedTokens: Float32Array; posEmbed: Float32Array },
    onProgress?: (doneBytes: number, totalBytes: number) => void | Promise<void>,
  ): Promise<ConditionerGpu> {
    const self = new ConditionerGpu(device, kernels, manifest);
    self.hostTables.set("embed_tokens.host", hostTables.embedTokens);
    self.hostTables.set("visual.pos_embed.weight", hostTables.posEmbed);
    let done = 0;
    for (const entry of manifest.tensors) {
      // Bytes, never floats: a q8 tensor is packed int8 in u32 words.
      const data = await read(entry.offset * 4, entry.count * 4);
      if (data.byteLength !== entry.count * 4) {
        throw new Error(`conditioner.bin: ${entry.name} read ${data.byteLength} of ${entry.count * 4}`);
      }
      const buffer = device.createStorageBuffer(data.byteLength);
      device.upload(buffer, 0, data);
      self.weights.set(entry.name, buffer);
      done += entry.count * 4;
      if (onProgress) await onProgress(done, manifest.residentBytes);
    }
    self.addConstants();
    return self;
  }

  private addConstants(): void {
    const t = this.manifest.textConfig;
    const v = this.manifest.visionConfig;
    // Two axis-dim tables, both "as many two-channel axes as there are
    // frequencies" — see `text-encoder.ts` for why that shape is what lets
    // `ropeAxes` express a rotation whose frequencies it does not own.
    for (const [name, headDim] of [["text", t.head_dim], ["vision", v.hidden_size / v.num_heads]] as const) {
      const dims = Uint32Array.from(new Array<number>(headDim / 2).fill(2));
      const buffer = this.device.createStorageBuffer(dims.byteLength);
      this.device.upload(buffer, 0, dims);
      this.weights.set(`${name}.axisDims`, buffer);
    }
    this.ensureMask(4096);
  }

  private maskBytes = 0;

  private ensureMask(seq: number): void {
    const bytes = Math.max(256, seq * 4);
    if (bytes <= this.maskBytes) return;
    this.weights.get("attn.noMask")?.destroy();
    this.weights.set("attn.noMask", this.device.createStorageBuffer(bytes));
    this.maskBytes = bytes;
  }

  private w(name: string): GPUBuffer {
    const buffer = this.weights.get(name);
    if (!buffer) throw new Error(`the conditioner has no tensor named "${name}"`);
    return buffer;
  }

  private take(elements: number): GPUBuffer {
    const GRAIN = 4 << 20;
    const bytes = Math.max(GRAIN, Math.ceil((Math.max(4, elements) * 4) / GRAIN) * GRAIN);
    const buffer = this.pool.get(bytes)?.pop() ?? this.device.createStorageBuffer(bytes);
    this.lent.push(buffer);
    return buffer;
  }

  private release(keep: GPUBuffer[] = []): void {
    for (const buffer of this.lent) {
      if (keep.includes(buffer)) continue;
      const free = this.pool.get(buffer.size) ?? [];
      free.push(buffer);
      this.pool.set(buffer.size, free);
    }
    this.lent.length = 0;
    this.lent.push(...keep);
    this.freeUniforms.push(...this.lentUniforms);
    this.lentUniforms.length = 0;
  }

  private async flush(ops: ResidentOp[], keep: GPUBuffer[]): Promise<void> {
    if (ops.length === 0) return;
    const at = performance.now();
    await this.device.batch(ops, []);
    this.submitMs += performance.now() - at;
    this.dispatches += ops.length;
    ops.length = 0;
    this.release(keep);
  }

  destroy(): void {
    for (const buffer of this.weights.values()) buffer.destroy();
    for (const list of this.pool.values()) for (const buffer of list) buffer.destroy();
    for (const buffer of this.lent) buffer.destroy();
    for (const buffer of this.lentUniforms) buffer.destroy();
    for (const buffer of this.freeUniforms) buffer.destroy();
  }

  private uniform(values: Parameters<typeof params>[0]): GPUBuffer {
    const buffer = this.freeUniforms.pop() ?? this.device.createUniformBuffer(UNIFORM_BYTES);
    this.device.upload(buffer, 0, new Uint8Array(params(values)));
    this.lentUniforms.push(buffer);
    return buffer;
  }

  private async dispatch(
    ops: ResidentOp[], code: string, buffers: GPUBuffer[],
    workgroups: [number] | [number, number] | [number, number, number],
  ): Promise<void> {
    const pipeline = await this.device.pipelineFor(code);
    ops.push({ kind: "dispatch", pipeline, bindGroup: await this.device.bindGroup(pipeline, buffers), workgroups });
  }

  private async dispatchSliced(
    ops: ResidentOp[], code: string,
    slices: { buffer: GPUBuffer; offset: number; size: number }[],
    workgroups: [number] | [number, number] | [number, number, number],
  ): Promise<void> {
    const pipeline = await this.device.pipelineFor(code);
    ops.push({ kind: "dispatch", pipeline, bindGroup: await this.device.bindGroupSliced(pipeline, slices), workgroups });
  }

  /** Row ranges no flat dispatch can exceed — #211's grid ceiling, on row boundaries. */
  private rowChunks(rows: number, cols: number): { start: number; count: number }[] {
    let perChunk = Math.floor((MAX_WORKGROUPS * WG) / cols);
    if (perChunk >= rows) return [{ start: 0, count: rows }];
    const gcd = (a: number, b: number): number => (b === 0 ? a : gcd(b, a % b));
    const step = 256 / gcd(cols * 4, 256);
    perChunk = Math.floor(perChunk / step) * step;
    if (perChunk < 1) throw new Error(`rowChunks: ${cols}-wide rows cannot be split under the grid limit`);
    const chunks: { start: number; count: number }[] = [];
    for (let start = 0; start < rows; start += perChunk) {
      chunks.push({ start, count: Math.min(perChunk, rows - start) });
    }
    return chunks;
  }

  private async flat(
    ops: ResidentOp[], code: string, rows: number, cols: number,
    layout: { buffer: GPUBuffer; sliced: boolean }[],
    uniform: (count: number) => Parameters<typeof params>[0],
  ): Promise<void> {
    for (const chunk of this.rowChunks(rows, cols)) {
      const byteOffset = chunk.start * cols * 4;
      const byteLength = chunk.count * cols * 4;
      await this.dispatchSliced(ops, code, [
        ...layout.map(({ buffer, sliced }) => sliced
          ? { buffer, offset: byteOffset, size: byteLength }
          : { buffer, offset: 0, size: buffer.size }),
        { buffer: this.uniform(uniform(chunk.count)), offset: 0, size: UNIFORM_BYTES },
      ], [Math.ceil((chunk.count * cols) / WG)]);
    }
  }

  private async linear(
    ops: ResidentOp[], a: Mat, name: string, bias: GPUBuffer | null, N: number,
  ): Promise<Mat> {
    const out = this.take(a.rows * N);
    if (this.manifest.dtype === "q8") {
      await this.dispatch(ops, this.kernels.matmulQ8, [
        a.buffer, this.w(name), this.w(`${name}.scale`), out,
        this.uniform([["u32", a.rows], ["u32", N], ["u32", a.cols]]),
      ], matmulQ8Grid(a.rows, N));
    } else {
      await this.dispatch(ops, this.kernels.matmul, [a.buffer, this.w(name), out, this.uniform([
        ["u32", a.rows], ["u32", N], ["u32", a.cols],
      ])], [Math.ceil(N / MM_BN), Math.ceil(a.rows / MM_BM), 1]);
    }
    if (!bias) return { buffer: out, rows: a.rows, cols: N };
    const biased = this.take(a.rows * N);
    await this.flat(ops, this.kernels.rows, a.rows, N, [
      { buffer: out, sliced: true }, { buffer: bias, sliced: false }, { buffer: biased, sliced: true },
    ], (count) => [["u32", count], ["u32", N], ["u32", ELEMENTWISE.add]]);
    return { buffer: biased, rows: a.rows, cols: N };
  }

  private async rms(ops: ResidentOp[], x: Mat, weight: GPUBuffer, eps: number, rows = x.rows, cols = x.cols): Promise<Mat> {
    const out = this.take(rows * cols);
    await this.dispatch(ops, this.kernels.rmsnorm, [x.buffer, weight, out, this.uniform([
      ["u32", rows], ["u32", cols], ["f32", eps], ["u32", 1],
    ])], [rows]);
    return { buffer: out, rows: x.rows, cols: x.cols };
  }

  private async layer(ops: ResidentOp[], x: Mat, weight: GPUBuffer, bias: GPUBuffer, rows: number, cols: number): Promise<Mat> {
    const out = this.take(rows * cols);
    await this.dispatch(ops, this.kernels.layernorm, [x.buffer, weight, bias, out, this.uniform([
      ["u32", rows], ["u32", cols], ["f32", LAYERNORM_EPS],
    ])], [rows]);
    return { buffer: out, rows, cols };
  }

  private async pointwise(ops: ResidentOp[], a: Mat, b: Mat, kind: number): Promise<Mat> {
    const out = this.take(a.rows * a.cols);
    await this.flat(ops, this.kernels.elementwise, a.rows, a.cols, [
      { buffer: a.buffer, sliced: true }, { buffer: b.buffer, sliced: true }, { buffer: out, sliced: true },
    ], (count) => [["u32", count * a.cols], ["u32", kind]]);
    return { buffer: out, rows: a.rows, cols: a.cols };
  }

  private async act(ops: ResidentOp[], x: Mat, kind: number): Promise<Mat> {
    const out = this.take(x.rows * x.cols);
    await this.flat(ops, this.kernels.activation, x.rows, x.cols, [
      { buffer: x.buffer, sliced: true }, { buffer: out, sliced: true },
    ], (count) => [["u32", count * x.cols], ["u32", kind], ["f32", 1]]);
    return { buffer: out, rows: x.rows, cols: x.cols };
  }

  private async swapLeading(ops: ResidentOp[], x: GPUBuffer, dim0: number, dim1: number, D: number): Promise<GPUBuffer> {
    const workgroups = Math.ceil((dim0 * dim1 * D) / WG);
    if (workgroups > MAX_WORKGROUPS) {
      throw new Error(`swapLeading: ${dim0}x${dim1}x${D} needs ${workgroups} workgroups — see issue #211`);
    }
    const out = this.take(dim0 * dim1 * D);
    await this.dispatch(ops, this.kernels.permute, [x, out, this.uniform([
      ["u32", dim0], ["u32", dim1], ["u32", D],
    ])], [workgroups]);
    return out;
  }

  private async rope(
    ops: ResidentOp[], x: Mat, seq: number, heads: number, headDim: number, positions: GPUBuffer, table: string,
  ): Promise<Mat> {
    const width = heads * headDim;
    const out = this.take(seq * width);
    // `thetaBase` is 1 on purpose: every axis is two channels wide, so the
    // kernel's own frequency is `1 ** 0` and the real one arrived folded into
    // the position. `text-encoder.ts` has the reason.
    await this.dispatch(ops, this.kernels.ropeAxes, [
      x.buffer, this.w(`${table}.axisDims`), positions, out,
      this.uniform([["u32", seq], ["u32", heads], ["u32", headDim], ["u32", headDim / 2], ["f32", 1]]),
    ], [Math.ceil((seq * heads * (headDim / 2)) / WG)]);
    return { buffer: out, rows: seq, cols: width };
  }

  /**
   * Full attention over `[seq, heads * headDim]`, optionally causal.
   *
   * `kvHeads` below `heads` is served by **repeating** the key and value heads
   * rather than by `ops/gqa` — see this file's own note on the trade.
   */
  private async attend(
    ops: ResidentOp[], q: Mat, k: Mat, v: Mat, seq: number, heads: number, kvHeads: number, headDim: number,
    causal: boolean,
  ): Promise<Mat> {
    const qh = await this.swapLeading(ops, q.buffer, seq, heads, headDim);
    let kh = await this.swapLeading(ops, k.buffer, seq, kvHeads, headDim);
    let vh = await this.swapLeading(ops, v.buffer, seq, kvHeads, headDim);
    if (kvHeads !== heads) {
      const group = heads / kvHeads;
      const bytes = seq * headDim * 4;
      const wideK = this.take(heads * seq * headDim);
      const wideV = this.take(heads * seq * headDim);
      for (let h = 0; h < heads; h += 1) {
        // Contiguous groups, as `enable_gqa=True` and `ops/gqa`'s `kvHeadOf`
        // define them: head `h` reads kv head `floor(h / group)`.
        const from = Math.floor(h / group) * bytes;
        ops.push({ kind: "copy", src: kh, srcOffset: from, dst: wideK, dstOffset: h * bytes, size: bytes });
        ops.push({ kind: "copy", src: vh, srcOffset: from, dst: wideV, dstOffset: h * bytes, size: bytes });
      }
      kh = wideK;
      vh = wideV;
    }
    const attended = this.take(heads * seq * headDim);
    await this.dispatch(ops, this.kernels.flashAttention, [
      qh, kh, vh, this.w("attn.noMask"), attended,
      this.uniform([
        ["u32", heads], ["u32", seq], ["u32", seq], ["u32", headDim], ["u32", headDim],
        ["f32", 1 / Math.sqrt(headDim)],
        ["u32", causal ? 1 : 0], ["i32", 0], ["u32", 1], ["u32", 1], ["u32", 1],
      ]),
    ], flashGrid(seq, heads, 1));
    const merged = await this.swapLeading(ops, attended, heads, seq, headDim);
    return { buffer: merged, rows: seq, cols: heads * headDim };
  }

  /** `norm -> fc1 -> exact GELU -> fc2` over `merge²` consecutive tokens. */
  private async merger(
    ops: ResidentOp[], prefix: string, x: Mat, tokens: number, postShuffle: boolean,
  ): Promise<Mat> {
    const v = this.manifest.visionConfig;
    const merge = v.spatial_merge_size;
    const wide = v.hidden_size * merge * merge;
    const rows = tokens / (merge * merge);
    const normed = postShuffle
      ? await this.layer(ops, x, this.w(`${prefix}.norm.weight`), this.w(`${prefix}.norm.bias`), rows, wide)
      : await this.layer(ops, x, this.w(`${prefix}.norm.weight`), this.w(`${prefix}.norm.bias`), tokens, v.hidden_size);
    const first = await this.linear(
      ops, { buffer: normed.buffer, rows, cols: wide }, `${prefix}.linear_fc1.weight`,
      this.w(`${prefix}.linear_fc1.bias`), wide);
    // The **exact** GELU here; the blocks' MLP uses the tanh one.
    const activated = await this.act(ops, first, ACTIVATION.gelu);
    return this.linear(ops, activated, `${prefix}.linear_fc2.weight`, this.w(`${prefix}.linear_fc2.bias`), v.out_hidden_size);
  }

  /** The vision tower: patches in, pooled tokens and three deepstack features out. */
  private async visionForward(
    ops: ResidentOp[], patches: Float32Array, grids: Grid[],
  ): Promise<{ pooled: Mat; deepstack: Mat[] }> {
    const v = this.manifest.visionConfig;
    const headDim = v.hidden_size / v.num_heads;
    const merge = v.spatial_merge_size;
    const patchDim = v.in_channels * v.temporal_patch_size * v.patch_size * v.patch_size;
    const tokens = grids.reduce((sum, [t, h, w]) => sum + t * h * w, 0);
    const gridPerSide = Math.round(Math.sqrt(v.num_position_embeddings));

    const input = this.take(patches.length);
    this.device.upload(input, 0, patches);
    let x = await this.linear(
      ops, { buffer: input, rows: tokens, cols: patchDim },
      "visual.patch_embed.proj.weight", this.w("visual.patch_embed.proj.bias"), v.hidden_size);

    // The position embedding and the rotary coordinates are host arithmetic on
    // a table this port already reproduces exactly; only the sums are on the
    // device.
    const table = this.readWeightAsFloats("visual.pos_embed.weight");
    const positional = new Float32Array(tokens * v.hidden_size);
    const coords: [number, number][] = [];
    let at = 0;
    for (const [frames, height, width] of grids) {
      const raster = interpolatePositionEmbedding(table, gridPerSide, v.hidden_size, height, width);
      const repeated = new Float32Array(frames * height * width * v.hidden_size);
      for (let t = 0; t < frames; t += 1) repeated.set(raster, t * height * width * v.hidden_size);
      positional.set(toMergeBlockOrder(repeated, frames, height, width, merge, v.hidden_size), at * v.hidden_size);
      coords.push(...visionCoordinates(frames, height, width, merge));
      at += frames * height * width;
    }
    const positionalBuffer = this.take(positional.length);
    this.device.upload(positionalBuffer, 0, positional);
    x = await this.pointwise(ops, x, { buffer: positionalBuffer, rows: tokens, cols: v.hidden_size }, ELEMENTWISE.add);

    const rotary = visionPositions(coords, headDim, 10000);
    const rotaryBuffer = this.take(rotary.length);
    this.device.upload(rotaryBuffer, 0, rotary);

    const deepstack: Mat[] = [];
    for (let i = 0; i < v.depth; i += 1) {
      const p = `visual.blocks.${i}`;
      const normed = await this.layer(ops, x, this.w(`${p}.norm1.weight`), this.w(`${p}.norm1.bias`), tokens, v.hidden_size);
      // Three projections, not one fused one: the converter split it, because
      // un-interleaving a fused `qkv` costs a copy per token per block.
      let q = await this.linear(ops, normed, `${p}.attn.q.weight`, this.w(`${p}.attn.q.bias`), v.hidden_size);
      let k = await this.linear(ops, normed, `${p}.attn.k.weight`, this.w(`${p}.attn.k.bias`), v.hidden_size);
      const value = await this.linear(ops, normed, `${p}.attn.v.weight`, this.w(`${p}.attn.v.bias`), v.hidden_size);
      q = await this.rope(ops, q, tokens, v.num_heads, headDim, rotaryBuffer, "vision");
      k = await this.rope(ops, k, tokens, v.num_heads, headDim, rotaryBuffer, "vision");
      const merged = await this.attend(ops, q, k, value, tokens, v.num_heads, v.num_heads, headDim, false);
      const projected = await this.linear(ops, merged, `${p}.attn.proj.weight`, this.w(`${p}.attn.proj.bias`), v.hidden_size);
      x = await this.pointwise(ops, x, projected, ELEMENTWISE.add);

      const normed2 = await this.layer(ops, x, this.w(`${p}.norm2.weight`), this.w(`${p}.norm2.bias`), tokens, v.hidden_size);
      const wide = await this.linear(ops, normed2, `${p}.mlp.linear_fc1.weight`, this.w(`${p}.mlp.linear_fc1.bias`), v.intermediate_size);
      // **`gelu_pytorch_tanh` here**, the approximation.
      const activated = await this.act(ops, wide, ACTIVATION.gelu_tanh);
      const ff = await this.linear(ops, activated, `${p}.mlp.linear_fc2.weight`, this.w(`${p}.mlp.linear_fc2.bias`), v.hidden_size);
      x = await this.pointwise(ops, x, ff, ELEMENTWISE.add);

      const tap = v.deepstack_visual_indexes.indexOf(i);
      if (tap >= 0) deepstack.push(await this.merger(ops, `visual.deepstack_merger_list.${tap}`, x, tokens, true));
    }
    return { pooled: await this.merger(ops, "visual.merger", x, tokens, false), deepstack };
  }

  private readonly hostTables = new Map<string, Float32Array>();

  /** One of the two tables `create` was given. */
  private readWeightAsFloats(name: string): Float32Array {
    const cached = this.hostTables.get(name);
    if (!cached) throw new Error(`the conditioner has no host copy of "${name}"`);
    return cached;
  }

  /**
   * The conditioning MiniMax-H3 reads: `hidden_states[textEncoderLayer]`.
   *
   * The vision tower runs first, its pooled tokens are written into the
   * embedding at the visual rows, and its three deepstack features are **added
   * in** at the first three text layers.
   */
  async forward(request: ConditionerRequest): Promise<Float32Array> {
    const t = this.manifest.textConfig;
    const v = this.manifest.visionConfig;
    const seq = request.tokenIds.length;
    const hidden = t.hidden_size;
    const headDim = t.head_dim;
    this.ensureMask(seq);
    this.submitMs = 0;
    this.readbackMs = 0;
    this.dispatches = 0;
    const startedAt = performance.now();

    const ops: ResidentOp[] = [];
    const vision = await this.visionForward(ops, request.patches, request.grids);

    // The token embedding, gathered on the host: one row per token out of a
    // 151,936-row table, which is a gather no kernel here has and 78 KB of
    // reads rather than a 0.78 GB pass.
    const embedTable = this.readWeightAsFloats("embed_tokens.host");
    const embeds = new Float32Array(seq * hidden);
    for (let i = 0; i < seq; i += 1) {
      embeds.set(embedTable.subarray(request.tokenIds[i]! * hidden, (request.tokenIds[i]! + 1) * hidden), i * hidden);
    }
    const stream = this.take(seq * hidden);
    this.device.upload(stream, 0, embeds);
    let x: Mat = { buffer: stream, rows: seq, cols: hidden };

    // The vision tokens replace the pad rows the presentation left for them.
    const rowBytes = hidden * 4;
    let taken = 0;
    for (const [start, length] of request.visualRuns) {
      ops.push({
        kind: "copy", src: vision.pooled.buffer, srcOffset: taken * rowBytes,
        dst: x.buffer, dstOffset: start * rowBytes, size: length * rowBytes,
      });
      taken += length;
    }
    if (taken !== vision.pooled.rows) {
      throw new Error(`forward: the visual runs cover ${taken} rows and the tower produced ${vision.pooled.rows}`);
    }
    await this.flush(ops, [x.buffer, ...vision.deepstack.map((d) => d.buffer)]);

    const positions = mropePositions(request.positions, headDim, t.rope_scaling.mrope_section, t.rope_theta);
    const positionBuffer = this.take(positions.length);
    this.device.upload(positionBuffer, 0, positions);

    const qWidth = t.num_attention_heads * headDim;
    const kvWidth = t.num_key_value_heads * headDim;
    for (let i = 0; i < this.manifest.layers; i += 1) {
      const p = `layers.${i}`;
      const normed = await this.rms(ops, x, this.w(`${p}.input_layernorm.weight`), t.rms_norm_eps);
      let q = await this.linear(ops, normed, `${p}.self_attn.q_proj.weight`, null, qWidth);
      let k = await this.linear(ops, normed, `${p}.self_attn.k_proj.weight`, null, kvWidth);
      const value = await this.linear(ops, normed, `${p}.self_attn.v_proj.weight`, null, kvWidth);
      q = await this.rms(ops, q, this.w(`${p}.self_attn.q_norm.weight`), t.rms_norm_eps, seq * t.num_attention_heads, headDim);
      k = await this.rms(ops, k, this.w(`${p}.self_attn.k_norm.weight`), t.rms_norm_eps, seq * t.num_key_value_heads, headDim);
      q = await this.rope(ops, q, seq, t.num_attention_heads, headDim, positionBuffer, "text");
      k = await this.rope(ops, k, seq, t.num_key_value_heads, headDim, positionBuffer, "text");
      const merged = await this.attend(
        ops, q, k, value, seq, t.num_attention_heads, t.num_key_value_heads, headDim, true);
      const projected = await this.linear(ops, merged, `${p}.self_attn.o_proj.weight`, null, hidden);
      x = await this.pointwise(ops, x, projected, ELEMENTWISE.add);

      const normed2 = await this.rms(ops, x, this.w(`${p}.post_attention_layernorm.weight`), t.rms_norm_eps);
      // `down(silu(gate) * up)` — **gate** is the activated half here, the
      // opposite of the DiT's SwiGLU.
      const gate = await this.act(
        ops, await this.linear(ops, normed2, `${p}.mlp.gate_proj.weight`, null, t.intermediate_size), ACTIVATION.silu);
      const up = await this.linear(ops, normed2, `${p}.mlp.up_proj.weight`, null, t.intermediate_size);
      const ff = await this.linear(
        ops, await this.pointwise(ops, gate, up, ELEMENTWISE.multiply), `${p}.mlp.down_proj.weight`, null, hidden);
      x = await this.pointwise(ops, x, ff, ELEMENTWISE.add);

      // **Deepstack: added in, at the visual rows, after the layer.** Three of
      // the vision tower's layers feed the text stack's first three.
      if (i < vision.deepstack.length) {
        const feature = vision.deepstack[i]!;
        // **Into a fresh buffer.** WebGPU refuses a buffer bound as both
        // read-only and read-write inside one compute pass, so an in-place add
        // is `usage (Storage(read-write)|Storage(read-only)) includes writable
        // usage and another usage in the same synchronization scope` -- an
        // invalid command buffer, whose output is whatever was in the pool.
        const next = this.take(seq * hidden);
        ops.push({ kind: "copy", src: x.buffer, srcOffset: 0, dst: next, dstOffset: 0, size: seq * rowBytes });
        let seen = 0;
        for (const [start, length] of request.visualRuns) {
          await this.dispatchSliced(ops, this.kernels.elementwise, [
            { buffer: x.buffer, offset: start * rowBytes, size: length * rowBytes },
            { buffer: feature.buffer, offset: seen * rowBytes, size: length * rowBytes },
            { buffer: next, offset: start * rowBytes, size: length * rowBytes },
            { buffer: this.uniform([["u32", length * hidden], ["u32", ELEMENTWISE.add]]), offset: 0, size: UNIFORM_BYTES },
          ], [Math.ceil((length * hidden) / WG)]);
          seen += length;
        }
        x = { buffer: next, rows: seq, cols: hidden };
      }
      await this.flush(ops, [x.buffer, positionBuffer, ...vision.deepstack.map((d) => d.buffer)]);
    }

    const staging = this.device.createStorageBuffer(
      seq * hidden * 4, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);
    const readAt = performance.now();
    const [out] = await this.device.batch(ops, [
      { staging, source: x.buffer, sourceOffset: 0, length: seq * hidden, type: "f32" },
    ]);
    this.readbackMs += performance.now() - readAt;
    this.dispatches += ops.length;
    ops.length = 0;
    this.release();
    staging.destroy();
    this.recordMs = performance.now() - startedAt - this.submitMs - this.readbackMs;
    return out as Float32Array;
  }
}

/** `[seq, 3]` grid the text stack reads, as `Qwen3VLModel` builds it for a flat prompt. */
export function textPositionGrid(seq: number): PositionGrid {
  const row = Array.from({ length: seq }, (_, i) => i);
  return { t: row, h: [...row], w: [...row] };
}
