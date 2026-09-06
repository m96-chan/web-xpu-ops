/**
 * MiniMax-H3's 50-layer DiT on the GPU — the generator, one velocity per step.
 *
 * Issue #210. `model.ts` is the same forward on the CPU, held to diffusers'
 * own model at 1.490e-7; this is the resident version. The two files are
 * deliberately the same shape so a divergence is readable.
 *
 * **No new kernel.** `matmul`, `matmulQ8`, `rmsnorm`, `flash_attention`,
 * `rope`'s axes entry, `activation`, `elementwise` and `permute` cover it —
 * the same set `examples/h3-video`'s decoder needed.
 *
 * ## What the converter did so this file did not have to
 *
 * **`adaln_proj` is not here at all.** It is 13.01 G of the checkpoint's 33.12
 * G parameters — **39.3%** — and it exists to project `temb`, a *two-row*
 * tensor whose value depends only on the timestep. `tools/convert_dit.py`
 * evaluates the modulation tables for the step counts it is given and ships
 * those instead, which is the difference between 20.1 GB of resident int8 and
 * 33.1. `norm_out.linear` goes the same way.
 *
 * The tables also arrive with **`1 + scale` already stored**, so the
 * modulation is a multiply and an add rather than a fused expression no kernel
 * here has.
 *
 * The RoPE permutation is folded into the q and k weights, and into
 * `norm_q` / `norm_k`'s per-channel weights — permuting the projection and not
 * the norm cost 8% in #208.
 *
 * ## Why modulation is a handful of dispatches and not one
 *
 * Every row of the packed sequence picks its own row of the `(timestep,
 * modality)` table, and nothing here gathers. It does not need to: the layout
 * is `[text | audio | video]`, so **rows that share a table row are
 * contiguous**, and each run is one `elementwise`-rows dispatch over a slice.
 * The runs are derived from the indices rather than assumed, so a layout with
 * conditioning rows in it still works — it just costs more dispatches.
 */
import type { ResidentDevice, ResidentOp } from "../../../harness/resident.js";
import { params } from "../../../harness/wgsl.js";
import { ACTIVATION } from "../../../ops/activation/index.js";
import { ELEMENTWISE } from "../../../ops/elementwise/index.js";
import { FLASH_GENERATION, FLASH_TOKEN_ENTRY, flashGrid } from "../../../ops/flash_attention/index.js";
import { matmulQ8Grid } from "../../../ops/matmul/index.js";
import { evictionPlan } from "../../h3-video/src/pool.js";
import { MODALITY_NUM, type PackedLayout } from "./model.js";

export const DIT_KERNEL_SOURCES: { key: keyof DitKernels; op: string; entry: string }[] = [
  { key: "matmul", op: "matmul", entry: "kernel" },
  { key: "matmulQ8", op: "matmul", entry: "q8" },
  { key: "rmsnorm", op: "rmsnorm", entry: "kernel" },
  { key: "activation", op: "activation", entry: "kernel" },
  { key: "elementwise", op: "elementwise", entry: "kernel" },
  { key: "rows", op: "elementwise", entry: "rows" },
  { key: "ropeAxes", op: "rope", entry: "axes" },
  { key: "permute", op: "permute", entry: "kernel" },
  { key: "flashAttention", op: "flash_attention", entry: FLASH_GENERATION },
  // Token-major (#223): `q`/`k`/`v` are already `[seq, heads * headDim]` here
  // — see `attention()` — so this entry reads that layout directly instead
  // of the three `swapLeading` copies the head-major entry above needed.
  { key: "flashAttentionToken", op: "flash_attention", entry: FLASH_TOKEN_ENTRY },
];

export interface DitKernels {
  matmul: string;
  matmulQ8: string;
  rmsnorm: string;
  activation: string;
  elementwise: string;
  rows: string;
  ropeAxes: string;
  permute: string;
  flashAttention: string;
  flashAttentionToken: string;
}

export interface DitManifest {
  config: {
    num_attention_heads: number;
    attention_head_dim: number;
    hidden_size: number;
    num_layers: number;
    num_refiner_layers: number;
    ffn_dim: number;
    in_channels: number;
    audio_in_channels: number;
    patch_size: [number, number, number];
    text_dim: number;
    freq_dim: number;
    time_embed_dim: number;
    rope_freq_dim: number;
    rope_theta: number;
    norm_eps: number;
    qk_norm_eps: number;
    final_norm_eps: number;
  };
  layers: number;
  dtype: string;
  stepCounts: number[];
  workflow?: string;
  /**
   * `levels` and `maxLevels` are written by conversions from #212 on.
   *
   * An older `t2va` conversion has neither, and two is what it built —
   * the video schedule and the audio one. `ref2va` carries up to four,
   * because its conditioning rows sit at their own noise-augmentation
   * levels, and a stack that assumed two indexed past the end of the
   * table on the fifteenth step of the run that found it.
   */
  schedules: Record<string, {
    video: number[]; audio: number[]; levelsPerStep: number[];
    levels?: number[][]; maxLevels?: number;
  }>;
  tensors: { name: string; shape: number[]; offset: number; count: number; kind?: string }[];
  tables: { name: string; shape: number[]; offset: number; count: number }[];
  residentBytes: number;
}

const WG = 256;
const MM_BM = 64;
const MM_BN = 128;
const UNIFORM_BYTES = 128;

interface Mat {
  buffer: GPUBuffer;
  rows: number;
  cols: number;
}

/** A maximal run of rows that share one AdaLN table row. */
interface Run {
  start: number;
  length: number;
  tableRow: number;
}

/** The contiguous runs of `adalnIndices`, derived rather than assumed. */
export function modulationRuns(adalnIndices: Int32Array): Run[] {
  const runs: Run[] = [];
  for (let i = 0; i < adalnIndices.length; i += 1) {
    const row = adalnIndices[i]!;
    const last = runs[runs.length - 1];
    if (last && last.tableRow === row && last.start + last.length === i) last.length += 1;
    else runs.push({ start: i, length: 1, tableRow: row });
  }
  return runs;
}

export interface DitForwardInputs {
  /** `[numVideoTokens, inChannels * prod(patchSize)]`, in `videoIndices` order. */
  video: Float32Array;
  audio: Float32Array;
  text: Float32Array;
  layout: PackedLayout;
  /** Which schedule the tables were built for, and where in it this step is. */
  steps: number;
  stepIndex: number;
}

export class DitGpu {
  private readonly weights = new Map<string, GPUBuffer>();
  private readonly tables = new Map<string, GPUBuffer>();
  private readonly pool = new Map<number, GPUBuffer[]>();
  private readonly lent: GPUBuffer[] = [];
  private readonly lentUniforms: GPUBuffer[] = [];
  private readonly freeUniforms: GPUBuffer[] = [];

  /** Blocks recorded into one command buffer — see `examples/h3-video` for why 1. */
  blocksPerSubmit = 1;

  /**
   * Flush **inside** a block once the sequence is longer than this.
   *
   * A block holds around twenty intermediates alive at once: `q`, `k`, `v` and
   * their rotated and head-swapped copies at `heads * head_dim` wide, the
   * feed-forward's four at `ffn_dim`, and a handful at `hidden_size`. They can
   * only be recycled at a submit, because until then they are what the recorded
   * dispatches read. At 19,027 rows that is about 12 GB on top of 20 GB of
   * weights, and a 32 GB card refuses somewhere in the middle of it — as an
   * invalid buffer rather than an error, so it surfaces one bind group later
   * as "invalid due to a previous error".
   *
   * Splitting the block at the residual add costs one extra submit per block
   * and halves the peak: the attention half's buffers go back to the pool
   * before the feed-forward half asks for its own.
   *
   * **Measured at no cost, so it is always on.** The threshold was going to be
   * a compromise with the extra submit; against the real-content golden at 572
   * rows the split run is bit-identical (worst 1.394e+0 either way) and 2409 ms
   * against 2663 ms, which is noise in the direction of faster. A knob whose
   * off position is never better is a knob to delete, so the default is 0 and
   * what is left is a way to turn it *off* for a measurement.
   */
  splitBlockAboveRows = 0;

  /**
   * The most the FREE half of the pool (buffers in neither `lent` nor
   * `quarantine`) is allowed to hold, enforced at the end of `release()`.
   *
   * Issue #223: the pool only ever grew, so resident memory was the SUM of
   * every size class's own peak — at 19,027 rows, after the token-major
   * kernel change (5d52019):
   *
   *     12.02 GB — 156 x   77 MB   weights
   *      8.02 GB — 208 x   39 MB   weights
   *      3.85 GB —   7 x  549 MB   attention-width activations
   *      3.28 GB —   3 x 1095 MB   ffn-width activations (a 4th refused)
   *      1.64 GB —   4 x  411 MB   hidden-width
   *      1.16 GB —  50 x   23 MB
   *      ...                                          (32.14 GB held, refused)
   *
   * but what a step actually needs at once is the MAX: the attention stretch
   * and the feed-forward stretch of a block run one after the other and never
   * need their buffers at the same time. 2 GiB keeps most of the attention
   * set (4 x 549 MB) warm between blocks — churn of ~1-2 recreations of
   * ~549 MB per block, microseconds-to-ms each against an 85 s step — while
   * forcing the attention and ffn stretches to share rather than accumulate.
   *
   * Settable, so a measurement can turn eviction off with `Infinity`. Do not
   * tune this beyond making the run fit — it is a correctness-of-fit change,
   * not a performance one (rule 8).
   */
  maxFreePoolBytes = 2 * 1024 ** 3;

  submitMs = 0;
  readbackMs = 0;
  recordMs = 0;
  dispatches = 0;

  private constructor(
    private readonly device: ResidentDevice,
    private readonly kernels: DitKernels,
    readonly manifest: DitManifest,
  ) {}

  /**
   * Uploads 20 GB one tensor at a time, awaiting each.
   *
   * **Bytes, not floats**: a `q8` tensor is packed int8 in `u32` words, and a
   * `Float32Array` view would run every word through f32's NaN canonicalisation
   * on the way to the device — a corruption that only shows on the bit patterns
   * that happen to be signalling NaNs.
   */
  static async create(
    device: ResidentDevice,
    kernels: DitKernels,
    manifest: DitManifest,
    readWeights: (offsetBytes: number, byteLength: number) => Uint8Array | Promise<Uint8Array>,
    readTables: (offsetBytes: number, byteLength: number) => Uint8Array | Promise<Uint8Array>,
    onProgress?: (doneBytes: number, totalBytes: number) => void | Promise<void>,
  ): Promise<DitGpu> {
    const self = new DitGpu(device, kernels, manifest);
    const totalBytes = manifest.residentBytes
      + manifest.tables.reduce((sum, t) => sum + t.count * 4, 0);
    let done = 0;
    for (const entry of manifest.tensors) {
      const data = await readWeights(entry.offset * 4, entry.count * 4);
      if (data.byteLength !== entry.count * 4) {
        throw new Error(`dit.bin: ${entry.name} read ${data.byteLength} bytes, manifest says ${entry.count * 4}`);
      }
      const buffer = device.createStorageBuffer(data.byteLength);
      device.upload(buffer, 0, data);
      self.weights.set(entry.name, buffer);
      done += entry.count * 4;
      if (onProgress) await onProgress(done, totalBytes);
    }
    for (const entry of manifest.tables) {
      const data = await readTables(entry.offset * 4, entry.count * 4);
      const buffer = device.createStorageBuffer(data.byteLength);
      device.upload(buffer, 0, data);
      self.tables.set(entry.name, buffer);
      done += entry.count * 4;
      if (onProgress) await onProgress(done, totalBytes);
    }
    self.addConstants();
    return self;
  }

  private addConstants(): void {
    const c = this.manifest.config;
    const rotDim = 2 * 3 * c.rope_freq_dim;
    // Three real axes plus a fourth pinned at position zero, covering the
    // channels H3 leaves unrotated — a rotation by zero is the identity.
    const axisDims = Uint32Array.from([
      2 * c.rope_freq_dim, 2 * c.rope_freq_dim, 2 * c.rope_freq_dim, c.attention_head_dim - rotDim,
    ]);
    const axisBuffer = this.device.createStorageBuffer(axisDims.byteLength);
    this.device.upload(axisBuffer, 0, axisDims);
    this.weights.set("rope.axisDims", axisBuffer);
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
    if (!buffer) throw new Error(`dit.bin has no tensor named "${name}"`);
    return buffer;
  }

  private table(name: string): GPUBuffer {
    const buffer = this.tables.get(name);
    if (!buffer) throw new Error(`adaln.bin has no table named "${name}" — was it converted for these steps?`);
    return buffer;
  }

  private take(elements: number): GPUBuffer {
    // 4 MB grain, not a power of two: the widths here are 5376, 7168, 14336 and
    // 28672 times a token count, and a power of two wastes up to half of every
    // buffer. `examples/h3-video` found this the hard way at 16,384 tokens.
    const GRAIN = 4 << 20;
    const bytes = Math.max(GRAIN, Math.ceil((Math.max(4, elements) * 4) / GRAIN) * GRAIN);
    const buffer = this.pool.get(bytes)?.pop() ?? this.device.createStorageBuffer(bytes);
    this.lent.push(buffer);
    return buffer;
  }

  /**
   * Hand one buffer back **before** the submit, because it is already dead.
   *
   * Issue #223. The attention path holds eleven buffers at `heads * head_dim`
   * wide -- q, k, v, the two `qkNorm` outputs, the two rotated ones, three head
   * swaps, the attended one and the merge -- and every one of them is written
   * once, read once, and then never touched again. At 19,027 rows that is 6.0
   * GB of the 31.3 GB that filled a 32 GB card.
   *
   * **Held until the submit, not returned to the pool immediately**, and that
   * distinction was measured rather than reasoned. Handing a buffer straight
   * back lets a later dispatch in the same pass *write* what an earlier one
   * still *reads* -- a write-after-read hazard, which Dawn does not barrier
   * even though it barriers read-after-write. It cost nothing at 2 latent
   * frames and moved `examples/h3-video`'s twelve-frame golden from 1.753e-1 to
   * 3.512e+0, 10 wrong pixel levels to 201. The small golden could not see it.
   *
   * So `recycle` only takes the buffer out of `lent`; the pool gets it at the
   * next flush, where the submit is a real barrier. The win is therefore the
   * *number of flushes*, which is why the block is split at its residual add.
   * Recycling something still to be read is still a silent wrong answer, so
   * each call site names the read it comes after.
   */
  /** `recycle` the old tensor and return the new one, for `x = consume(x, f(x))`. */
  private consume(dead: Mat, made: Mat): Mat {
    if (dead.buffer !== made.buffer) this.recycle(dead.buffer);
    return made;
  }

  private readonly quarantine: GPUBuffer[] = [];

  private recycle(buffer: GPUBuffer): void {
    const at = this.lent.indexOf(buffer);
    if (at < 0) return;
    this.lent.splice(at, 1);
    this.quarantine.push(buffer);
  }

  /** Returns how many bytes it actually destroyed — 0 when the plan was empty. */
  private release(keep: GPUBuffer[] = []): number {
    for (const buffer of this.lent) {
      if (keep.includes(buffer)) continue;
      const free = this.pool.get(buffer.size) ?? [];
      free.push(buffer);
      this.pool.set(buffer.size, free);
    }
    for (const buffer of this.quarantine) {
      const free = this.pool.get(buffer.size) ?? [];
      free.push(buffer);
      this.pool.set(buffer.size, free);
    }
    this.quarantine.length = 0;
    this.lent.length = 0;
    this.lent.push(...keep);
    this.freeUniforms.push(...this.lentUniforms);
    this.lentUniforms.length = 0;

    // Issue #223: bound the free half of the pool now that `lent` and
    // `quarantine` have both been folded into it — evicting before either
    // was drained would plan against sizes that had not yet arrived.
    const free = [...this.pool.entries()].map(([size, buffers]) => ({ size, count: buffers.length }));
    const plan = evictionPlan(free, this.maxFreePoolBytes);
    let evictedBytes = 0;
    for (const [size, count] of plan) {
      const buffers = this.pool.get(size);
      if (!buffers) continue;
      for (let i = 0; i < count; i++) {
        const buffer = buffers.pop();
        if (!buffer) break;
        buffer.destroy();
        evictedBytes += size;
      }
      if (buffers.length === 0) this.pool.delete(size);
    }
    return evictedBytes;
  }

  private async flush(ops: ResidentOp[], keep: GPUBuffer[]): Promise<void> {
    if (ops.length === 0) return;
    const at = performance.now();
    await this.device.batch(ops, []);
    this.submitMs += performance.now() - at;
    this.dispatches += ops.length;
    ops.length = 0;
    const evicted = this.release(keep);
    // An eviction is bookkeeping until Dawn ticks: a destroyed buffer's VRAM
    // comes back only after RECLAIM_ROUND_TRIPS submits (issue #213, measured
    // in harness/verify-reclaim.ts). Without this the run refused a 411 MB
    // allocation while our own counter said 25 of 32 GB -- the other seven
    // were destroyed and not yet returned.
    if (evicted > 0) await this.device.reclaim();
  }

  destroy(): void {
    for (const buffer of this.weights.values()) buffer.destroy();
    for (const buffer of this.tables.values()) buffer.destroy();
    for (const list of this.pool.values()) for (const buffer of list) buffer.destroy();
    for (const buffer of this.lent) buffer.destroy();
    for (const buffer of this.lentUniforms) buffer.destroy();
    for (const buffer of this.freeUniforms) buffer.destroy();
  }

  private uniform(values: Parameters<typeof params>[0]): GPUBuffer {
    const data = params(values);
    const buffer = this.freeUniforms.pop() ?? this.device.createUniformBuffer(UNIFORM_BYTES);
    this.device.upload(buffer, 0, new Uint8Array(data));
    this.lentUniforms.push(buffer);
    return buffer;
  }

  private async dispatch(
    ops: ResidentOp[],
    code: string,
    buffers: GPUBuffer[],
    workgroups: [number] | [number, number] | [number, number, number],
  ): Promise<void> {
    const pipeline = await this.device.pipelineFor(code);
    ops.push({ kind: "dispatch", pipeline, bindGroup: await this.device.bindGroup(pipeline, buffers), workgroups });
  }

  private async dispatchSliced(
    ops: ResidentOp[],
    code: string,
    slices: { buffer: GPUBuffer; offset: number; size: number }[],
    workgroups: [number] | [number, number] | [number, number, number],
  ): Promise<void> {
    const pipeline = await this.device.pipelineFor(code);
    ops.push({ kind: "dispatch", pipeline, bindGroup: await this.device.bindGroupSliced(pipeline, slices), workgroups });
  }

  /**
   * The device's grid ceiling. **65,535 on every device measured** — the spec's
   * default, reported by Dawn Node and by Chrome alike, and not raised by
   * asking.
   *
   * A field rather than a constant so a test can lower it and exercise the
   * chunking without a 17-million-element buffer.
   */
  maxWorkgroupsPerDimension = 65535;

  /**
   * Row ranges no dispatch of `cols`-wide rows can exceed.
   *
   * One thread per element and `ceil(n / 256)` workgroups means a flat dispatch
   * runs out of grid at **16,776,960 elements** — which a 5,376-wide residual
   * stream reaches at 3,120 tokens and a 14,336-wide feed-forward reaches at
   * 1,170. A 576x320 clip is 1,350 packed rows, so this is not a corner: it is
   * the second size anybody picks.
   *
   * Split on **row** boundaries rather than element ones, because
   * `elementwise`'s rows entry recovers its column with `idx % D` and a chunk
   * that starts mid-row would read the wrong scalar. The count is rounded down
   * to keep every slice 256-byte aligned, which `bindGroupSliced` requires.
   */
  private rowChunks(rows: number, cols: number): { start: number; count: number }[] {
    const perDispatch = this.maxWorkgroupsPerDimension * WG;
    let perChunk = Math.floor(perDispatch / cols);
    if (perChunk >= rows) return [{ start: 0, count: rows }];
    if (perChunk < 1) {
      throw new Error(`rowChunks: a single ${cols}-wide row needs ${Math.ceil(cols / WG)} workgroups, past the limit`);
    }
    // `start * cols * 4` has to be a multiple of 256 for every chunk.
    const rowBytes = cols * 4;
    const gcd = (a: number, b: number): number => (b === 0 ? a : gcd(b, a % b));
    const step = 256 / gcd(rowBytes, 256);
    perChunk = Math.floor(perChunk / step) * step;
    if (perChunk < 1) {
      throw new Error(`rowChunks: ${cols}-wide rows cannot be split on a 256-byte boundary under the grid limit`);
    }
    const chunks: { start: number; count: number }[] = [];
    for (let start = 0; start < rows; start += perChunk) {
      chunks.push({ start, count: Math.min(perChunk, rows - start) });
    }
    return chunks;
  }

  /**
   * A flat, per-element kernel over `[rows, cols]`, split to fit the grid.
   *
   * `sliced` names the buffers that are indexed by element and must be offset
   * with the chunk; `whole` names the ones that are not — a bias or a
   * modulation vector, read at `idx % D`.
   */
  private async dispatchFlat(
    ops: ResidentOp[],
    code: string,
    rows: number,
    cols: number,
    layout: { buffer: GPUBuffer; sliced: boolean }[],
    uniform: (count: number) => Parameters<typeof params>[0],
  ): Promise<void> {
    for (const chunk of this.rowChunks(rows, cols)) {
      const byteOffset = chunk.start * cols * 4;
      const byteLength = chunk.count * cols * 4;
      await this.dispatchSliced(ops, code, [
        ...layout.map(({ buffer, sliced }) =>
          sliced
            ? { buffer, offset: byteOffset, size: byteLength }
            : { buffer, offset: 0, size: buffer.size }),
        { buffer: this.uniform(uniform(chunk.count)), offset: 0, size: UNIFORM_BYTES },
      ], [Math.ceil((chunk.count * cols) / WG)]);
    }
  }

  private async linear(ops: ResidentOp[], a: Mat, name: string, bias: GPUBuffer | null, N: number): Promise<Mat> {
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
    await this.dispatchFlat(ops, this.kernels.rows, a.rows, N, [
      { buffer: out, sliced: true }, { buffer: bias, sliced: false }, { buffer: biased, sliced: true },
    ], (count) => [["u32", count], ["u32", N], ["u32", ELEMENTWISE.add]]);
    return { buffer: biased, rows: a.rows, cols: N };
  }

  private async norm(ops: ResidentOp[], x: Mat, weight: GPUBuffer, eps: number): Promise<Mat> {
    const out = this.take(x.rows * x.cols);
    await this.dispatch(ops, this.kernels.rmsnorm, [x.buffer, weight, out, this.uniform([
      ["u32", x.rows], ["u32", x.cols], ["f32", eps], ["u32", 1],
    ])], [x.rows]);
    return { buffer: out, rows: x.rows, cols: x.cols };
  }

  private async pointwise(ops: ResidentOp[], a: Mat, b: Mat, kind: number): Promise<Mat> {
    const out = this.take(a.rows * a.cols);
    await this.dispatchFlat(ops, this.kernels.elementwise, a.rows, a.cols, [
      { buffer: a.buffer, sliced: true }, { buffer: b.buffer, sliced: true }, { buffer: out, sliced: true },
    ], (count) => [["u32", count * a.cols], ["u32", kind]]);
    return { buffer: out, rows: a.rows, cols: a.cols };
  }

  /**
   * `x` combined with one chunk of the modulation table, run by run.
   *
   * `chunk` indexes the six the table holds — shift, scale, gate for attention
   * then the same three for the feed-forward — and `scale` arrives as
   * `1 + scale`, so this is a plain multiply.
   */
  private async modulateChunk(
    ops: ResidentOp[],
    x: Mat,
    table: GPUBuffer,
    stepOffsetRows: number,
    runs: Run[],
    chunk: number,
    chunksPerRow: number,
    kind: number,
    into: GPUBuffer,
  ): Promise<Mat> {
    const hidden = x.cols;
    const rowBytes = hidden * 4;
    for (const run of runs) {
      const vectorOffset = ((stepOffsetRows + run.tableRow) * chunksPerRow + chunk) * rowBytes;
      // Chunked for the same reason `dispatchFlat` is: a run of a 5,376-wide
      // stream runs out of grid at 3,120 rows.
      for (const chunk of this.rowChunks(run.length, hidden)) {
        const at = (run.start + chunk.start) * rowBytes;
        await this.dispatchSliced(ops, this.kernels.rows, [
          { buffer: x.buffer, offset: at, size: chunk.count * rowBytes },
          { buffer: table, offset: vectorOffset, size: rowBytes },
          { buffer: into, offset: at, size: chunk.count * rowBytes },
          { buffer: this.uniform([["u32", chunk.count], ["u32", hidden], ["u32", kind]]), offset: 0, size: UNIFORM_BYTES },
        ], [Math.ceil((chunk.count * hidden) / WG)]);
      }
    }
    return { buffer: into, rows: x.rows, cols: hidden };
  }

  /** `[rows, heads, headDim]` <-> `[heads, rows, headDim]`. */
  private async swapLeading(ops: ResidentOp[], x: GPUBuffer, dim0: number, dim1: number, D: number): Promise<GPUBuffer> {
    // **The one flat dispatch here that cannot be chunked.** A transpose writes
    // strided, so a slice of the input has no slice of the output to land in;
    // splitting it needs a second grid dimension in the kernel, which is an
    // `ops/permute` change. At 56 heads of 128 this runs out of grid at 2,340
    // tokens -- a 576x320 clip is 1,350, so it is close. Refused with the
    // number rather than left to arrive as `batch is not valid`.
    // **Tiled, since #214 gave `ops/permute` a second grid dimension.** It
    // folds `y` back in from `num_workgroups`, so the host only chooses the
    // tiling and a one-dimensional caller is unaffected. Before that this threw
    // here, and 56 heads of 128 runs out of one row of workgroups at 2,340
    // tokens — a 25-second reference read at 2 fps is well past it.
    const workgroups = Math.ceil((dim0 * dim1 * D) / WG);
    const x0 = Math.min(workgroups, this.maxWorkgroupsPerDimension);
    const y0 = Math.ceil(workgroups / x0);
    if (y0 > this.maxWorkgroupsPerDimension) {
      throw new Error(
        `swapLeading: ${dim0}x${dim1}x${D} needs ${workgroups} workgroups and two dimensions of ` +
          `${this.maxWorkgroupsPerDimension} are not enough — see issue #211`,
      );
    }
    const out = this.take(dim0 * dim1 * D);
    await this.dispatch(ops, this.kernels.permute, [x, out, this.uniform([
      ["u32", dim0], ["u32", dim1], ["u32", D],
    ])], [x0, y0]);
    return out;
  }

  private async rope(ops: ResidentOp[], x: Mat, seq: number, positions: GPUBuffer): Promise<Mat> {
    const c = this.manifest.config;
    const out = this.take(seq * c.num_attention_heads * c.attention_head_dim);
    const width = c.num_attention_heads * c.attention_head_dim;
    // Chunked by **token**, which rope is separable over. `positions` is
    // `[N, 4]`, so its slices are 16 bytes a token and a chunk has to be a
    // multiple of 16 tokens to stay 256-byte aligned — a tighter step than the
    // activations alone would need.
    const perDispatch = this.maxWorkgroupsPerDimension * WG;
    const tokensPerChunk = Math.max(16, Math.floor(perDispatch / (width / 2) / 16) * 16);
    // Binding order and struct read off `ops/rope/wgsl/axes.wgsl`, not
    // remembered: `axis_dims` is binding 1 and `positions` is binding 2, and
    // Params has **five** fields. Written from memory this had them swapped and
    // one field short, which is a silent NaN rather than a validation error —
    // the shapes are all `array<f32>` and a short uniform reads garbage.
    for (let start = 0; start < seq; start += tokensPerChunk) {
      const count = Math.min(tokensPerChunk, seq - start);
      await this.dispatchSliced(ops, this.kernels.ropeAxes, [
        { buffer: x.buffer, offset: start * width * 4, size: count * width * 4 },
        { buffer: this.w("rope.axisDims"), offset: 0, size: this.w("rope.axisDims").size },
        { buffer: positions, offset: start * 4 * 4, size: count * 4 * 4 },
        { buffer: out, offset: start * width * 4, size: count * width * 4 },
        {
          buffer: this.uniform([
            ["u32", count], ["u32", c.num_attention_heads], ["u32", c.attention_head_dim],
            ["u32", 4], ["f32", c.rope_theta],
          ]),
          offset: 0, size: UNIFORM_BYTES,
        },
      ], [Math.ceil((count * width / 2) / WG)]);
    }
    return { buffer: out, rows: seq, cols: width };
  }

  /**
   * `q`, `k` and `v` arrive already `[seq, heads * headDim]` — token-major,
   * the layout every caller here holds them in — so `flashAttentionToken`
   * reads them directly. Issue #223: this used to run three `swapLeading`
   * copies in and one out, four buffers at 549 MB each at 19,027 rows, which
   * is most of what stopped 512x896 x 120 frames from fitting in 32 GB.
   */
  private async attention(
    ops: ResidentOp[], q: Mat, k: Mat, v: Mat, seq: number, keep: GPUBuffer[] = [],
  ): Promise<Mat> {
    const c = this.manifest.config;
    const heads = c.num_attention_heads;
    const headDim = c.attention_head_dim;
    const attended = this.take(seq * heads * headDim);
    await this.dispatch(ops, this.kernels.flashAttentionToken, [
      q.buffer, k.buffer, v.buffer, this.w("attn.noMask"), attended,
      this.uniform([
        ["u32", heads], ["u32", seq], ["u32", seq], ["u32", headDim], ["u32", headDim],
        ["f32", 1 / Math.sqrt(headDim)],
        // One packed sequence, one attention document, no mask and no causality
        // — the processor's own note.
        ["u32", 0], ["i32", 0], ["u32", 1], ["u32", 1], ["u32", 1],
      ]),
    ], flashGrid(seq, heads, 1));
    // q, k and v are read only by the dispatch above. Issue #223.
    this.recycle(q.buffer);
    this.recycle(k.buffer);
    this.recycle(v.buffer);
    // **Split, once the flash dispatch has read q, k and v.** Down from eight
    // buffers in quarantine (three head swaps plus the five that fed them) to
    // three, now that the kernel reads token-major directly rather than
    // needing head-major copies of all three.
    if (seq > this.splitBlockAboveRows && keep.length > 0) {
      await this.flush(ops, [attended, ...keep]);
    }
    return { buffer: attended, rows: seq, cols: heads * headDim };
  }

  /**
   * The SwiGLU both stacks use: `hidden * silu(gate)`, **hidden first**.
   *
   * The two halves are separate matrices in the conversion — `ff.hidden` and
   * `ff.gate` — because slicing them out of one `[28672, hidden]` product needs
   * a dispatch per row, and at 1,024 rows over 50 blocks that is 100,000 of
   * them a step. `examples/h3-video`'s decoder splits `ff.w1` for the same
   * reason.
   */
  private async feedForward(ops: ResidentOp[], x: Mat, prefix: string): Promise<Mat> {
    const c = this.manifest.config;
    const seq = x.rows;
    const value = await this.linear(ops, x, `${prefix}ff.hidden.weight`, null, c.ffn_dim);
    const gate = await this.linear(ops, x, `${prefix}ff.gate.weight`, null, c.ffn_dim);
    const activated = this.take(seq * c.ffn_dim);
    await this.dispatchFlat(ops, this.kernels.activation, seq, c.ffn_dim, [
      { buffer: gate.buffer, sliced: true }, { buffer: activated, sliced: true },
    ], (count) => [["u32", count * c.ffn_dim], ["u32", ACTIVATION.silu], ["f32", 1]]);
    // `gate` is read only by the activation above. Issue #223: at 19,027 rows
    // each of these is 1,095 MB and four of them alive is the wall the
    // attention path was moved off.
    this.recycle(gate.buffer);
    const gated = await this.pointwise(
      ops, value, { buffer: activated, rows: seq, cols: c.ffn_dim }, ELEMENTWISE.multiply,
    );
    this.recycle(value.buffer);
    this.recycle(activated);
    const out = await this.linear(ops, gated, `${prefix}ff.net.2.weight`, null, c.hidden_size);
    this.recycle(gated.buffer);
    return out;
  }

  /**
   * RMS-norm over each head's channels, with the checkpoint's per-channel
   * weights.
   *
   * **One workgroup per head-row**, so the grid is `seq * heads` — at 56 heads
   * that passes 65,535 at **1,171 tokens**, which a 256x256 clip with two
   * references already is. Split on head-row boundaries: each is
   * `attention_head_dim` floats, 512 bytes at 128, so every slice is 256-byte
   * aligned without rounding.
   *
   * The failure it replaces was an invalid *command buffer*, which takes every
   * dispatch recorded beside it with it — fifteen steps of plausible timings
   * and a velocity computed from pool debris. Issue #211.
   */
  private async qkNorm(ops: ResidentOp[], x: Mat, seq: number, weight: GPUBuffer): Promise<Mat> {
    const c = this.manifest.config;
    const dim = c.attention_head_dim;
    const rows = seq * c.num_attention_heads;
    const out = this.take(rows * dim);
    const rowBytes = dim * 4;
    const perChunk = Math.min(rows, this.maxWorkgroupsPerDimension);
    for (let start = 0; start < rows; start += perChunk) {
      const count = Math.min(perChunk, rows - start);
      await this.dispatchSliced(ops, this.kernels.rmsnorm, [
        { buffer: x.buffer, offset: start * rowBytes, size: count * rowBytes },
        { buffer: weight, offset: 0, size: weight.size },
        { buffer: out, offset: start * rowBytes, size: count * rowBytes },
        {
          buffer: this.uniform([["u32", count], ["u32", dim], ["f32", c.qk_norm_eps], ["u32", 1]]),
          offset: 0, size: UNIFORM_BYTES,
        },
      ], [count]);
    }
    return { buffer: out, rows: seq, cols: c.num_attention_heads * dim };
  }

  /**
   * One block, `[seq, hidden]` to `[seq, hidden]`.
   *
   * The same arrangement as `model.ts`'s `h3DitBlock`, with the six modulation
   * vectors read straight out of the table instead of being projected.
   */
  private async block(
    ops: ResidentOp[],
    x: Mat,
    index: number,
    table: GPUBuffer,
    stepOffsetRows: number,
    runs: Run[],
    positions: GPUBuffer,
  ): Promise<Mat> {
    const c = this.manifest.config;
    const seq = x.rows;
    const width = c.num_attention_heads * c.attention_head_dim;
    const p = `transformer_blocks.${index}.`;

    const normed = await this.norm(ops, x, this.w(`${p}norm1.weight`), c.norm_eps);
    // `x * (1 + scale) + shift`, and the table already holds `1 + scale`.
    const scaled = await this.modulateChunk(
      ops, normed, table, stepOffsetRows, runs, 1, 6, ELEMENTWISE.multiply, this.take(seq * c.hidden_size));
    this.recycle(normed.buffer);
    const shifted = await this.modulateChunk(
      ops, scaled, table, stepOffsetRows, runs, 0, 6, ELEMENTWISE.add, this.take(seq * c.hidden_size));
    this.recycle(scaled.buffer);

    let q = await this.linear(ops, shifted, `${p}attn.to_q.weight`, null, width);
    let k = await this.linear(ops, shifted, `${p}attn.to_k.weight`, null, width);
    const v = await this.linear(ops, shifted, `${p}attn.to_v.weight`, null, width);
    // Each of these reads its input once and writes a new buffer, so the input
    // is dead the moment the dispatch is recorded. Issue #223.
    q = this.consume(q, await this.qkNorm(ops, q, seq, this.w(`${p}attn.norm_q.weight`)));
    k = this.consume(k, await this.qkNorm(ops, k, seq, this.w(`${p}attn.norm_k.weight`)));
    q = this.consume(q, await this.rope(ops, q, seq, positions));
    k = this.consume(k, await this.rope(ops, k, seq, positions));

    // `shifted` fed the three projections and nothing else.
    this.recycle(shifted.buffer);
    // **Split, before the attention.** The projections' outputs and the
    // `qkNorm` ones -- four buffers at `heads * head_dim` -- died on the way
    // here (issue #223).
    if (seq > this.splitBlockAboveRows) {
      await this.flush(ops, [q.buffer, k.buffer, v.buffer, x.buffer, positions]);
    }
    const merged = await this.attention(ops, q, k, v, seq, [x.buffer, positions]);
    // **Split one: the attention is finished with.** `attention` already
    // split once it recycled q, k and v; this is a second chance in case
    // nothing forced a submit in between (`flush` is a no-op on an empty
    // `ops`, so this costs nothing when the one above already ran).
    if (seq > this.splitBlockAboveRows) await this.flush(ops, [merged.buffer, x.buffer, positions]);
    const projected = await this.linear(ops, merged, `${p}attn.to_out.0.weight`, null, c.hidden_size);
    this.recycle(merged.buffer);
    const gated = await this.modulateChunk(
      ops, projected, table, stepOffsetRows, runs, 2, 6, ELEMENTWISE.multiply, this.take(seq * c.hidden_size));
    this.recycle(projected.buffer);
    let hidden = await this.pointwise(ops, x, gated, ELEMENTWISE.add);
    this.recycle(gated.buffer);

    // **Split two: the block's own halfway point.** Everything above is dead once the
    // residual has been added, and nothing below has been allocated yet, so
    // this is where the peak can be cut without keeping anything alive across
    // it. `x` is deliberately not kept: the caller replaces it with what this
    // returns.
    if (seq > this.splitBlockAboveRows) await this.flush(ops, [hidden.buffer, positions]);

    const normed2 = await this.norm(ops, hidden, this.w(`${p}norm2.weight`), c.norm_eps);
    const scaled2 = await this.modulateChunk(
      ops, normed2, table, stepOffsetRows, runs, 4, 6, ELEMENTWISE.multiply, this.take(seq * c.hidden_size));
    this.recycle(normed2.buffer);
    const shifted2 = await this.modulateChunk(
      ops, scaled2, table, stepOffsetRows, runs, 3, 6, ELEMENTWISE.add, this.take(seq * c.hidden_size));
    this.recycle(scaled2.buffer);

    const ff = await this.feedForward(ops, shifted2, p);
    this.recycle(shifted2.buffer);
    // **Split three: the feed-forward's four `ffn_dim` buffers are done.** At
    // 19,027 rows each is 1,095 MB, which is the difference between fitting and
    // not.
    if (seq > this.splitBlockAboveRows) await this.flush(ops, [ff.buffer, hidden.buffer, positions]);
    const gated2 = await this.modulateChunk(
      ops, ff, table, stepOffsetRows, runs, 5, 6, ELEMENTWISE.multiply, this.take(seq * c.hidden_size));
    this.recycle(ff.buffer);
    // **`hidden` is not recycled**: the add reads it and writes a new buffer,
    // and the one it writes is what this returns.
    hidden = await this.pointwise(ops, hidden, gated2, ELEMENTWISE.add);
    this.recycle(gated2.buffer);
    return hidden;
  }

  /** One plain refiner block: no AdaLN, no rotary. */
  private async refinerBlock(ops: ResidentOp[], x: Mat, index: number): Promise<Mat> {
    const c = this.manifest.config;
    const seq = x.rows;
    const width = c.num_attention_heads * c.attention_head_dim;
    const p = `token_refiner.refiner_blocks.${index}.`;

    const normed = await this.norm(ops, x, this.w(`${p}norm1.weight`), c.norm_eps);
    let q = await this.linear(ops, normed, `${p}attn.to_q.weight`, null, width);
    let k = await this.linear(ops, normed, `${p}attn.to_k.weight`, null, width);
    const v = await this.linear(ops, normed, `${p}attn.to_v.weight`, null, width);
    q = await this.qkNorm(ops, q, seq, this.w(`${p}attn.norm_q.weight`));
    k = await this.qkNorm(ops, k, seq, this.w(`${p}attn.norm_k.weight`));
    // **No rope.** The text stream carries no position grid, and rotating it
    // anyway is a mistake every shape survives.
    const merged = await this.attention(ops, q, k, v, seq);
    const projected = await this.linear(ops, merged, `${p}attn.to_out.0.weight`, null, c.hidden_size);
    let hidden = await this.pointwise(ops, x, projected, ELEMENTWISE.add);

    const normed2 = await this.norm(ops, hidden, this.w(`${p}norm2.weight`), c.norm_eps);
    const ff = await this.feedForward(ops, normed2, p);
    hidden = await this.pointwise(ops, hidden, ff, ELEMENTWISE.add);
    return hidden;
  }

  /** A device buffer holding `data`, kept until the next `release`. */
  private upload(data: Float32Array): GPUBuffer {
    const buffer = this.take(data.length);
    this.device.upload(buffer, 0, data);
    return buffer;
  }

  /**
   * One velocity, for the video rows and the audio rows.
   *
   * `steps` names the schedule the tables were converted for and `stepIndex`
   * says where in it this call is — the modulation table is read at that row
   * rather than projected, because `adaln_proj` is not resident.
   */
  async forward(inputs: DitForwardInputs): Promise<{ video: Float32Array; audio: Float32Array }> {
    const c = this.manifest.config;
    const { layout } = inputs;
    const seq = layout.seq;
    const hidden = c.hidden_size;
    const patchDim = c.in_channels * c.patch_size[0] * c.patch_size[1] * c.patch_size[2];
    const numVideo = layout.videoIndices.length;
    const numAudio = layout.audioIndices.length;
    const numText = layout.textIndices.length;

    if (!this.manifest.stepCounts.includes(inputs.steps)) {
      throw new Error(
        `this conversion holds tables for ${this.manifest.stepCounts.join(", ")} steps, not ${inputs.steps} — ` +
          `re-run convert_dit.py with --steps ${inputs.steps}`,
      );
    }
    this.ensureMask(seq);
    this.submitMs = 0;
    this.readbackMs = 0;
    this.dispatches = 0;
    const startedAt = performance.now();

    const ops: ResidentOp[] = [];
    const adalnIndices = new Int32Array(seq);
    for (let i = 0; i < seq; i += 1) {
      adalnIndices[i] = layout.timestepIndices[i]! * MODALITY_NUM + layout.tokenTags[i]!;
    }
    const runs = modulationRuns(adalnIndices);
    // `timestepIndices` alone, for `norm_out` — which modulates per *timestep*,
    // not per (timestep, modality).
    const normOutRuns = modulationRuns(layout.timestepIndices);

    const levels = this.manifest.schedules[String(inputs.steps)]!.levelsPerStep.length;
    if (inputs.stepIndex < 0 || inputs.stepIndex >= levels) {
      throw new Error(`step ${inputs.stepIndex} is outside the ${levels} the ${inputs.steps}-step schedule has`);
    }
    // The table is `[steps, maxLevels * 3, 6 * hidden]`, so a step is
    // `maxLevels * 3` rows in. Two for a `t2va` conversion and four for a
    // `ref2va` one; older conversions predate the field and were all `t2va`.
    const maxLevels = this.manifest.schedules[String(inputs.steps)]!.maxLevels ?? 2;
    const stepOffsetRows = inputs.stepIndex * maxLevels * MODALITY_NUM;

    const positions = this.upload(ropeAxesPositions(layout.positionIds, seq));

    // 1. Project each modality; refine the text stream.
    const videoEmbeds = await this.linear(
      ops, { buffer: this.upload(inputs.video), rows: numVideo, cols: patchDim },
      "proj_in.weight", this.w("proj_in.bias"), hidden);
    const audioEmbeds = await this.linear(
      ops, { buffer: this.upload(inputs.audio), rows: numAudio, cols: c.audio_in_channels },
      "audio_proj_in.weight", this.w("audio_proj_in.bias"), hidden);
    let textEmbeds = await this.linear(
      ops, { buffer: this.upload(inputs.text), rows: numText, cols: c.text_dim },
      "context_embedder.weight", this.w("context_embedder.bias"), hidden);
    for (let i = 0; i < c.num_refiner_layers; i += 1) textEmbeds = await this.refinerBlock(ops, textEmbeds, i);
    textEmbeds = await this.norm(ops, textEmbeds, this.w("token_refiner.final_norm.weight"), c.final_norm_eps);

    // 2. Scatter into one packed buffer. Each modality's rows are contiguous in
    // the layouts this builds, so a scatter is a copy per run rather than an
    // indexed write — and `copyRuns` throws rather than guessing if they are not.
    const packed = this.take(seq * hidden);
    copyRuns(ops, videoEmbeds.buffer, packed, layout.videoIndices, hidden);
    copyRuns(ops, audioEmbeds.buffer, packed, layout.audioIndices, hidden);
    copyRuns(ops, textEmbeds.buffer, packed, layout.textIndices, hidden);

    let x: Mat = { buffer: packed, rows: seq, cols: hidden };
    await this.flush(ops, [x.buffer, positions]);

    // 3. The block stack.
    for (let i = 0; i < this.manifest.layers; i += 1) {
      x = await this.block(ops, x, i, this.table(`adaln.${inputs.steps}.${i}`), stepOffsetRows, runs, positions);
      if ((i + 1) % this.blocksPerSubmit === 0) await this.flush(ops, [x.buffer, positions]);
    }
    await this.flush(ops, [x.buffer, positions]);

    // 4. `norm_out`, then both heads over every row.
    const normOut = this.table(`normOut.${inputs.steps}`);
    const normed = await this.norm(ops, x, this.w("norm_out.norm.weight"), c.final_norm_eps);
    const scaled = await this.modulateChunk(
      ops, normed, normOut, inputs.stepIndex * maxLevels, normOutRuns, 1, 2,
      ELEMENTWISE.multiply, this.take(seq * hidden));
    const modulated = await this.modulateChunk(
      ops, scaled, normOut, inputs.stepIndex * maxLevels, normOutRuns, 0, 2,
      ELEMENTWISE.add, this.take(seq * hidden));

    const videoAll = await this.linear(ops, modulated, "proj_out.weight", this.w("proj_out.bias"), patchDim);
    const audioAll = await this.linear(
      ops, modulated, "audio_proj_out.weight", this.w("audio_proj_out.bias"), c.audio_in_channels);

    // 5. Select each modality's rows — the same runs, in the other direction.
    const videoOut = this.take(numVideo * patchDim);
    const audioOut = this.take(numAudio * c.audio_in_channels);
    gatherRuns(ops, videoAll.buffer, videoOut, layout.videoIndices, patchDim);
    gatherRuns(ops, audioAll.buffer, audioOut, layout.audioIndices, c.audio_in_channels);

    const videoStaging = this.device.createStorageBuffer(
      numVideo * patchDim * 4, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);
    const audioStaging = this.device.createStorageBuffer(
      numAudio * c.audio_in_channels * 4, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);
    const readAt = performance.now();
    const [video, audio] = await this.device.batch(ops, [
      { staging: videoStaging, source: videoOut, sourceOffset: 0, length: numVideo * patchDim, type: "f32" },
      { staging: audioStaging, source: audioOut, sourceOffset: 0, length: numAudio * c.audio_in_channels, type: "f32" },
    ]);
    this.readbackMs += performance.now() - readAt;
    this.dispatches += ops.length;
    ops.length = 0;
    this.release();
    videoStaging.destroy();
    audioStaging.destroy();
    // Everything that is not the two submits and the readback: bind groups,
    // pipeline lookups, uniform writes. Named rather than left implicit,
    // because `examples/h3-video` found 136 ms of it in a 264 ms decode.
    this.recordMs = performance.now() - startedAt - this.submitMs - this.readbackMs;
    return { video: video as Float32Array, audio: audio as Float32Array };
  }
}

/**
 * `(t, h, w)` per row widened to the four axes `ropeAxes` takes.
 *
 * The fourth is pinned at zero and covers the channels H3 leaves unrotated.
 */
export function ropeAxesPositions(positionIds: Float32Array, seq: number): Float32Array {
  const out = new Float32Array(seq * 4);
  for (let token = 0; token < seq; token += 1) {
    for (let axis = 0; axis < 3; axis += 1) out[token * 4 + axis] = positionIds[token * 3 + axis]!;
  }
  return out;
}

/** Maximal ascending runs of consecutive indices. */
function runsOf(indices: Int32Array): { from: number; to: number; length: number }[] {
  const runs: { from: number; to: number; length: number }[] = [];
  for (let i = 0; i < indices.length; i += 1) {
    const last = runs[runs.length - 1];
    if (last && indices[i]! === last.to + last.length && i === last.from + last.length) last.length += 1;
    else runs.push({ from: i, to: indices[i]!, length: 1 });
  }
  return runs;
}

/** Copies `[n, width]` rows into the packed buffer at the positions `indices` names. */
function copyRuns(ops: ResidentOp[], src: GPUBuffer, dst: GPUBuffer, indices: Int32Array, width: number): void {
  for (const run of runsOf(indices)) {
    ops.push({
      kind: "copy",
      src, srcOffset: run.from * width * 4,
      dst, dstOffset: run.to * width * 4,
      size: run.length * width * 4,
    });
  }
}

/** The other direction: reads the rows `indices` names out of a packed buffer. */
function gatherRuns(ops: ResidentOp[], src: GPUBuffer, dst: GPUBuffer, indices: Int32Array, width: number): void {
  for (const run of runsOf(indices)) {
    ops.push({
      kind: "copy",
      src, srcOffset: run.to * width * 4,
      dst, dstOffset: run.from * width * 4,
      size: run.length * width * 4,
    });
  }
}
