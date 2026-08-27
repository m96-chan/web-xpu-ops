/**
 * MiniMax-H3's visual VAE encoder, on the GPU: a video in, the latent's moments
 * out.
 *
 * Issue #214. `encoder.ts` is the same arithmetic on the CPU, held to
 * `EncoderFCN3D` and `quant_conv` at **2.432e-5**; this is the resident
 * version, built the way `examples/h3-video/src/decoder-gpu.ts` is so a
 * divergence between the two is readable.
 *
 * **The CPU one is a reference and stays one.** It is a single-threaded loop
 * over `ops/conv`'s scalar `conv3d`, and it takes **120.5 s on an 8x32x32
 * clip** — R2V has to encode a reference per request, so it needed a device.
 * Nothing here replaces it: it is what this file is compared against, and it is
 * held to the model in turn.
 *
 * **No new kernel.** `conv3d`, `pad`, `group_norm`, `activation`,
 * `elementwise` and `permute` cover the whole encoder — which was checked
 * against the four the CPU version calls before any of this was written.
 *
 * ## Three things the assembly has to get right on its own
 *
 * - **`Downsample3D` pads asymmetrically before it strides.** With
 *   `space_stride == 2` it prepends nothing and appends one column and one row,
 *   then runs a `k=3, stride=2` convolution whose spatial padding is zero. A
 *   symmetric pad gives the same output *size* and a different alignment, which
 *   is a video that comes back shifted half a pixel per level.
 * - **The padding for an ordinary `k=3` convolution is reflect in space and
 *   causal in time** — two zero frames in front, none behind. `ops/pad` takes
 *   one axis at a time, and W then H then D is what reproduces a single
 *   multi-axis `F.pad`.
 * - **Group norm runs per frame.** `[C, D, H, W]` is `_merge_time_to_batch`'d
 *   to `[D, C, H*W]` and back. On the CPU that is two index loops; here it is
 *   two `permute` dispatches, because `ops/group_norm` normalises over `C/G`
 *   channels of one batch item and the batch has to be the frame.
 *
 * The 1x1 convolutions — `nin_shortcut` and `quant_conv` — take **no padding at
 * all**: `BaseConv3d.forward` short-circuits to `nn.Conv3d` when every padding
 * is zero, so in particular no causal frames. Padding them would change the
 * length.
 */
import type { ResidentDevice, ResidentOp } from "../../../harness/resident.js";
import { params } from "../../../harness/wgsl.js";
import { encodeConditioning, type Moments } from "./conditioning.js";
import { ACTIVATION } from "../../../ops/activation/index.js";
import { ELEMENTWISE } from "../../../ops/elementwise/index.js";

export const ENCODER_KERNEL_SOURCES: { key: keyof EncoderKernels; op: string; entry: string }[] = [
  { key: "conv3d", op: "conv", entry: "conv3d" },
  { key: "pad", op: "pad", entry: "kernel" },
  { key: "groupNorm", op: "group_norm", entry: "kernel" },
  { key: "activation", op: "activation", entry: "kernel" },
  { key: "elementwise", op: "elementwise", entry: "kernel" },
  { key: "permute", op: "permute", entry: "kernel" },
];

export interface EncoderKernels {
  conv3d: string;
  pad: string;
  groupNorm: string;
  activation: string;
  elementwise: string;
  permute: string;
}

export interface EncoderGpuManifest {
  config: {
    ch: number; ch_mult: number[]; num_res_blocks: number;
    space_down: number[]; time_down: number[]; in_channels: number;
    z_channels: number; groups: number; norm_eps: number;
  };
  tensors: { name: string; offset: number; count: number }[];
  elements: number;
  /**
   * `vae_clip_length` and `vae_token_drop` from the released `video_vae`
   * config, for `encodeConditioning`. Issue #216.
   *
   * Optional because manifests written before it existed do not carry them, and
   * `conditioningChunking` refuses rather than defaulting: 17 and 3 are what
   * *this* checkpoint ships, and a silent default is how a second checkpoint
   * with different chunking would be encoded wrongly and never say so.
   */
  clipLength?: number;
  tokenDrop?: number;
}

/**
 * The chunk geometry a conditioning encode uses, or an error naming the fix.
 *
 * Not defaulted. The released `Ref2VA/video_vae/config.json` says
 * `vae_clip_length 17` and `vae_token_drop 3`, and those numbers belong to that
 * checkpoint -- writing them in here would encode the next checkpoint's
 * references with the wrong geometry and return a well-shaped tensor while
 * doing it, which is the failure #216 is about.
 */
export function conditioningChunking(
  manifest: EncoderGpuManifest,
): { clipLength: number; tokenDrop: number } {
  const { clipLength, tokenDrop } = manifest;
  if (typeof clipLength !== "number" || typeof tokenDrop !== "number") {
    throw new Error(
      "encoder manifest has no `clipLength`/`tokenDrop`: it predates issue #216. " +
        "Re-run `tools/gen_resnet_golden.py --whole`, which writes them.",
    );
  }
  return { clipLength, tokenDrop };
}

const WG = 256;
const UNIFORM_BYTES = 128;
/** 65,535 everywhere measured — see issue #211. */
const MAX_WORKGROUPS = 65535;
/** `ops/pad`'s mode enum. */
const PAD_CONSTANT = 0;
const PAD_REFLECT = 1;

/** A tensor and the shape that is live in it. */
interface Volume { buffer: GPUBuffer; C: number; D: number; H: number; W: number }

/** `[block_in, block_mid]` per level — the channel counts the model derives. */
export function channelPlanOf(config: EncoderGpuManifest["config"]): { blockIn: number[]; blockMid: number[] } {
  const blockMid = config.ch_mult.map((m) => config.ch * m);
  return { blockIn: [blockMid[0]!, ...blockMid.slice(0, -1)], blockMid };
}

export class EncoderGpu {
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
    private readonly kernels: EncoderKernels,
    readonly manifest: EncoderGpuManifest,
  ) {}

  static async create(
    device: ResidentDevice,
    kernels: EncoderKernels,
    manifest: EncoderGpuManifest,
    read: (offsetBytes: number, byteLength: number) => Uint8Array | Promise<Uint8Array>,
    onProgress?: (doneBytes: number, totalBytes: number) => void | Promise<void>,
  ): Promise<EncoderGpu> {
    const self = new EncoderGpu(device, kernels, manifest);
    let done = 0;
    for (const entry of manifest.tensors) {
      const data = await read(entry.offset * 4, entry.count * 4);
      if (data.byteLength !== entry.count * 4) {
        throw new Error(`encoder.bin: ${entry.name} read ${data.byteLength} of ${entry.count * 4}`);
      }
      const buffer = device.createStorageBuffer(data.byteLength);
      device.upload(buffer, 0, data);
      self.weights.set(entry.name, buffer);
      done += entry.count * 4;
      if (onProgress) await onProgress(done, manifest.elements * 4);
    }
    return self;
  }

  destroy(): void {
    for (const buffer of this.weights.values()) buffer.destroy();
    for (const list of this.pool.values()) for (const buffer of list) buffer.destroy();
    for (const buffer of this.lent) buffer.destroy();
    for (const buffer of this.lentUniforms) buffer.destroy();
    for (const buffer of this.freeUniforms) buffer.destroy();
  }

  private w(name: string): GPUBuffer {
    const buffer = this.weights.get(name);
    if (!buffer) throw new Error(`the encoder has no tensor named "${name}"`);
    return buffer;
  }

  private take(elements: number): GPUBuffer {
    const GRAIN = 4 << 20;
    const bytes = Math.max(GRAIN, Math.ceil((Math.max(4, elements) * 4) / GRAIN) * GRAIN);
    const buffer = this.pool.get(bytes)?.pop() ?? this.device.createStorageBuffer(bytes);
    this.lent.push(buffer);
    return buffer;
  }

  private release(keep: GPUBuffer[]): void {
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

  /**
   * `ops/pad` on one axis.
   *
   * The flat range is tiled in two dimensions because a single row of
   * workgroups runs out at 65,535 — `stride_y` is how the kernel folds `(x, y)`
   * back into one index, and the host owns it because only the host knows how
   * wide it dispatched.
   */
  private async padAxis(
    ops: ResidentOp[], input: GPUBuffer,
    outer: number, L: number, inner: number, before: number, after: number, mode: number,
  ): Promise<GPUBuffer> {
    const Lout = L + before + after;
    const total = outer * Lout * inner;
    const out = this.take(total);
    const [x, y] = this.tiles(total);
    await this.dispatch(ops, this.kernels.pad, [
      input, out,
      this.uniform([
        ["u32", outer], ["u32", L], ["u32", inner], ["u32", before], ["u32", Lout],
        ["u32", mode], ["f32", 0], ["u32", x * WG],
      ]),
    ], [x, y]);
    return out;
  }

  /**
   * `BaseConv3d`'s padding for a `k=3, padding=1` convolution: reflect on W,
   * reflect on H, then **two zero frames in front and none behind**.
   */
  private async padForCausalConv(
    ops: ResidentOp[], v: Volume, spatial: number, temporalBefore: number,
  ): Promise<{ buffer: GPUBuffer; D: number; H: number; W: number }> {
    let buffer = v.buffer;
    let { D, H, W } = v;
    if (spatial > 0) {
      buffer = await this.padAxis(ops, buffer, v.C * D * H, W, 1, spatial, spatial, PAD_REFLECT);
      W += 2 * spatial;
      buffer = await this.padAxis(ops, buffer, v.C * D, H, W, spatial, spatial, PAD_REFLECT);
      H += 2 * spatial;
    }
    if (temporalBefore > 0) {
      buffer = await this.padAxis(ops, buffer, v.C, D, H * W, temporalBefore, 0, PAD_CONSTANT);
      D += temporalBefore;
    }
    return { buffer, D, H, W };
  }

  /** One `conv3d` dispatch over an already-padded volume — the kernel pads nothing. */
  private async conv(
    ops: ResidentOp[],
    input: GPUBuffer, Cin: number, Cout: number, D: number, H: number, W: number,
    weight: GPUBuffer, bias: GPUBuffer,
    K: [number, number, number], stride: [number, number, number] = [1, 1, 1],
  ): Promise<Volume> {
    const [KD, KH, KW] = K;
    const Dout = Math.floor((D - KD) / stride[0]) + 1;
    const Hout = Math.floor((H - KH) / stride[1]) + 1;
    const Wout = Math.floor((W - KW) / stride[2]) + 1;
    const out = this.take(Cout * Dout * Hout * Wout);
    const z = Cout * Dout;
    if (Hout > MAX_WORKGROUPS || z > MAX_WORKGROUPS) {
      throw new Error(`conv: ${Cout}x${Dout}x${Hout}x${Wout} exceeds the grid limit — see issue #211`);
    }
    await this.dispatch(ops, this.kernels.conv3d, [
      input, weight, bias, out,
      this.uniform([
        ["u32", Cin], ["u32", Cout], ["u32", D], ["u32", H], ["u32", W],
        ["u32", KD], ["u32", KH], ["u32", KW],
        ["u32", Dout], ["u32", Hout], ["u32", Wout],
        ["u32", stride[0]], ["u32", stride[1]], ["u32", stride[2]],
        // No padding: everything this file convolves has already been padded by
        // `ops/pad`, because reflect is not something a convolution can express.
        ["u32", 0], ["u32", 0], ["u32", 0],
        ["u32", 1], ["u32", 1], ["u32", 1],
        ["u32", Cin], ["u32", Cout],
        ["u32", 0], ["u32", 0],
      ]),
    ], [Math.ceil(Wout / WG), Hout, z]);
    return { buffer: out, C: Cout, D: Dout, H: Hout, W: Wout };
  }

  /**
   * A flat dispatch, tiled into `(x, y)` when one row of workgroups is not
   * enough.
   *
   * The first level of a 256x256 reference is 128 channels of 65,536 pixels a
   * frame, which passes the 65,535-workgroup ceiling (#211) at five frames.
   * `permute`, `activation` and `elementwise` fold `y` back in from
   * `num_workgroups`, so the host only has to choose the tiling.
   */
  private tiles(count: number): [number, number] {
    const wanted = Math.ceil(count / WG);
    const x = Math.min(wanted, MAX_WORKGROUPS);
    const y = Math.ceil(wanted / x);
    if (y > MAX_WORKGROUPS) throw new Error(`${count} elements need ${wanted} workgroups — see issue #211`);
    return [x, y];
  }

  /** `[dim0][dim1][inner]` to `[dim1][dim0][inner]`. */
  private async swapLeading(
    ops: ResidentOp[], x: GPUBuffer, dim0: number, dim1: number, inner: number,
  ): Promise<GPUBuffer> {
    const count = dim0 * dim1 * inner;
    const out = this.take(count);
    await this.dispatch(ops, this.kernels.permute, [
      x, out, this.uniform([["u32", dim0], ["u32", dim1], ["u32", inner]]),
    ], this.tiles(count));
    return out;
  }

  /**
   * Group norm over each frame independently.
   *
   * `_merge_time_to_batch`: the frame has to be the batch, because
   * `ops/group_norm` normalises over `C/G` channels of one batch item and H3
   * normalises each frame on its own. Two permutes rather than a strided read,
   * for the same reason the CPU version has two index loops.
   */
  private async perFrameGroupNorm(
    ops: ResidentOp[], v: Volume, weight: GPUBuffer, bias: GPUBuffer, groups: number, eps: number,
  ): Promise<Volume> {
    const plane = v.H * v.W;
    const merged = await this.swapLeading(ops, v.buffer, v.C, v.D, plane);
    const normed = this.take(v.C * v.D * plane);
    if (v.D * groups > MAX_WORKGROUPS) {
      throw new Error(`perFrameGroupNorm: ${v.D} frames of ${groups} groups exceeds the grid limit`);
    }
    await this.dispatch(ops, this.kernels.groupNorm, [
      merged, weight, bias, normed,
      this.uniform([["u32", v.D], ["u32", v.C], ["u32", plane], ["u32", groups], ["f32", eps]]),
    ], [v.D * groups]);
    return { buffer: await this.swapLeading(ops, normed, v.D, v.C, plane), C: v.C, D: v.D, H: v.H, W: v.W };
  }

  /** A flat elementwise pass, tiled the way every other flat dispatch here is. */
  private async flat(
    ops: ResidentOp[], code: string, buffers: GPUBuffer[], count: number,
    uniform: Parameters<typeof params>[0],
  ): Promise<void> {
    await this.dispatch(ops, code, [...buffers, this.uniform(uniform)], this.tiles(count));
  }

  private async silu(ops: ResidentOp[], v: Volume): Promise<Volume> {
    const count = v.C * v.D * v.H * v.W;
    const out = this.take(count);
    await this.flat(ops, this.kernels.activation, [v.buffer, out], count,
      [["u32", count], ["u32", ACTIVATION.silu], ["f32", 1]]);
    return { buffer: out, C: v.C, D: v.D, H: v.H, W: v.W };
  }

  private async add(ops: ResidentOp[], a: Volume, b: Volume): Promise<Volume> {
    const count = a.C * a.D * a.H * a.W;
    const out = this.take(count);
    await this.flat(ops, this.kernels.elementwise, [a.buffer, b.buffer, out], count,
      [["u32", count], ["u32", ELEMENTWISE.add]]);
    return { buffer: out, C: a.C, D: a.D, H: a.H, W: a.W };
  }

  /** `pad -> conv3d`, the `k=3, padding=1` convolution the whole encoder is made of. */
  private async conv3x3x3(
    ops: ResidentOp[], v: Volume, Cout: number, weight: GPUBuffer, bias: GPUBuffer,
  ): Promise<Volume> {
    const padded = await this.padForCausalConv(ops, v, 1, 2);
    return this.conv(ops, padded.buffer, v.C, Cout, padded.D, padded.H, padded.W, weight, bias, [3, 3, 3]);
  }

  /** One encoder resnet block, `[Cin, D, H, W]` to `[Cout, D, H, W]`. */
  private async resnetBlock(
    ops: ResidentOp[], v: Volume, prefix: string, outChannels: number, groups: number, eps: number,
  ): Promise<Volume> {
    const inChannels = v.C;
    let h = await this.perFrameGroupNorm(
      ops, v, this.w(`${prefix}norm1.weight`), this.w(`${prefix}norm1.bias`), groups, eps);
    h = await this.silu(ops, h);
    h = await this.conv3x3x3(ops, h, outChannels, this.w(`${prefix}conv1.weight`), this.w(`${prefix}conv1.bias`));

    h = await this.perFrameGroupNorm(
      ops, h, this.w(`${prefix}norm2.weight`), this.w(`${prefix}norm2.bias`), groups, eps);
    h = await this.silu(ops, h);
    h = await this.conv3x3x3(ops, h, outChannels, this.w(`${prefix}conv2.weight`), this.w(`${prefix}conv2.bias`));

    // **The 1x1 shortcut takes no padding**, and in particular no causal
    // frames: `BaseConv3d.forward` short-circuits to `nn.Conv3d` when every
    // padding is zero. Padding it would be a length change.
    let residual = v;
    if (inChannels !== outChannels) {
      residual = await this.conv(
        ops, v.buffer, inChannels, outChannels, v.D, v.H, v.W,
        this.w(`${prefix}nin_shortcut.weight`), this.w(`${prefix}nin_shortcut.bias`), [1, 1, 1]);
    }
    return this.add(ops, residual, h);
  }

  /**
   * `Downsample3D`: pad one column and one row **after**, then stride.
   *
   * The asymmetry is the model's. A symmetric pad of one on each side gives the
   * same output size and a different alignment — a picture shifted half a pixel
   * per level, which is exactly the failure that is invisible until the whole
   * chain is compared.
   */
  private async downsample(
    ops: ResidentOp[], v: Volume, Cout: number, timeStride: number, spaceStride: number, prefix: string,
  ): Promise<Volume> {
    let buffer = v.buffer;
    let { H, W } = v;
    if (spaceStride === 2) {
      buffer = await this.padAxis(ops, buffer, v.C * v.D * H, W, 1, 0, 1, PAD_REFLECT);
      W += 1;
      buffer = await this.padAxis(ops, buffer, v.C * v.D, H, W, 0, 1, PAD_REFLECT);
      H += 1;
    }
    // `padding=(1, 0, 0)`: nothing in space, two causal frames in front.
    buffer = await this.padAxis(ops, buffer, v.C, v.D, H * W, 2, 0, PAD_CONSTANT);
    const D = v.D + 2;
    return this.conv(
      ops, buffer, v.C, Cout, D, H, W,
      this.w(`${prefix}conv.weight`), this.w(`${prefix}conv.bias`),
      [3, 3, 3], [timeStride, spaceStride, spaceStride]);
  }

  /**
   * A video `[inChannels, T, H, W]` in, `[2 * zChannels, T/4, H/16, W/16]` out.
   *
   * `quant_conv` is folded in for the same reason `encoder.ts` folds it in:
   * `encode` in the model is `quant_conv(encoder(x))`, and splitting them would
   * invite one to be forgotten.
   */
  async encode(video: Float32Array, T: number, H: number, W: number): Promise<{
    data: Float32Array; C: number; D: number; H: number; W: number;
  }> {
    const c = this.manifest.config;
    const { blockMid } = channelPlanOf(c);
    const startedAt = performance.now();
    this.submitMs = 0;
    this.readbackMs = 0;
    this.dispatches = 0;

    const ops: ResidentOp[] = [];
    const input = this.take(video.length);
    this.device.upload(input, 0, video);
    let v: Volume = { buffer: input, C: c.in_channels, D: T, H, W };
    v = await this.conv3x3x3(ops, v, blockMid[0]!, this.w("conv_in.weight"), this.w("conv_in.bias"));
    await this.flush(ops, [v.buffer]);

    for (let level = 0; level < c.ch_mult.length; level += 1) {
      for (let block = 0; block < c.num_res_blocks; block += 1) {
        v = await this.resnetBlock(
          ops, v, `down.${level}.block.${block}.`, blockMid[level]!, c.groups, c.norm_eps);
        await this.flush(ops, [v.buffer]);
      }
      // The model adds a downsample only when it changes the shape, and a 1x1
      // convolution only when the level changes channel count — which, with
      // `block_out == block_mid`, it never does. One branch, not two.
      if (c.space_down[level]! * c.time_down[level]! > 1) {
        v = await this.downsample(
          ops, v, blockMid[level]!, c.time_down[level]!, c.space_down[level]!, `down.${level}.downsample.`);
        await this.flush(ops, [v.buffer]);
      }
    }

    v = await this.perFrameGroupNorm(
      ops, v, this.w("norm_out.weight"), this.w("norm_out.bias"), c.groups, c.norm_eps);
    v = await this.silu(ops, v);
    v = await this.conv3x3x3(ops, v, 2 * c.z_channels, this.w("conv_out.weight"), this.w("conv_out.bias"));
    // `quant_conv`, a 1x1x1 Conv3d with no padding at all.
    v = await this.conv(
      ops, v.buffer, v.C, 2 * c.z_channels, v.D, v.H, v.W,
      this.w("quant_conv.weight"), this.w("quant_conv.bias"), [1, 1, 1]);

    const count = v.C * v.D * v.H * v.W;
    const staging = this.device.createStorageBuffer(
      count * 4, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);
    const readAt = performance.now();
    const [out] = await this.device.batch(ops, [
      { staging, source: v.buffer, sourceOffset: 0, length: count, type: "f32" },
    ]);
    this.readbackMs += performance.now() - readAt;
    this.dispatches += ops.length;
    ops.length = 0;
    this.release([]);
    staging.destroy();
    this.recordMs = performance.now() - startedAt - this.submitMs - this.readbackMs;
    return { data: out as Float32Array, C: v.C, D: v.D, H: v.H, W: v.W };
  }

  /**
   * The conditioning path: `encode_temporal`, not `encode`.
   *
   * **Issue #216. `encode` above is the wrong entry point for a reference with
   * more than one frame**, and holding it to `EncoderFCN3D` + `quant_conv` --
   * which `verify-encode-gpu.ts` does, at 9.537e-6 -- cannot see that, because
   * `EncoderFCN3D` is what `encode` wraps. The model's own `encode_base` calls
   * `encode` only for a single image; for a frame stack it calls
   * `encode_temporal`, transcribed here from `video_vae/klvae.py` lines
   * 461-493. The released checkpoint ships `vae_clip_length 17` and
   * `vae_token_drop 3`; the `isolated_*` flags that guard the other branches
   * are absent from its config and default to false.
   *
   * The clip is padded up to a multiple of `clip_length` by **repeating its
   * last frame**, each 17-frame chunk is encoded on its own -- the causal state
   * restarts, so chunk two knows nothing of chunk one -- and `token_drop`
   * latent frames come off the end of the concatenation.
   *
   * **Measured** (`tools/measure_temporal_chunking.py`, the released weights,
   * this against `encode` alone): at 48 pixel frames, the shortest video
   * reference the model card allows, both return 12 latent frames and they
   * differ by **17.9% rms**, worst element 5.62. 68 frames: 19.0%. 85 frames:
   * 21.5%. At 17, 22, 51 and 120 frames the *shapes* disagree outright. The
   * shapes coinciding at 48 and 68 is arithmetic, not agreement, and it is why
   * nothing downstream ever complained.
   *
   * A loop over `encode` rather than chunking inside the dispatch graph: the
   * reference is a loop over an encode, the chunks are independent by
   * construction, and a version fused into the kernels would be a second thing
   * to hold to the first.
   */
  async encodeConditioning(video: Float32Array, T: number, H: number, W: number): Promise<Moments> {
    return encodeConditioning(
      (clip, frames, h, w) => this.encode(clip, frames, h, w),
      video, T, H, W, conditioningChunking(this.manifest),
    );
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
}

/**
 * The mean half of the moments — the latent a decoder is given.
 *
 * `DiagonalGaussianDistribution(moments).sample()` draws from
 * `mean + exp(0.5 * logvar) * noise`; the mean is the first `zChannels` of the
 * channel axis, and handing a decoder the second half produces something
 * plausible and wrong.
 */
export function latentMeanOf(
  moments: { data: Float32Array; D: number; H: number; W: number }, zChannels: number,
): Float32Array {
  return moments.data.slice(0, zChannels * moments.D * moments.H * moments.W);
}
