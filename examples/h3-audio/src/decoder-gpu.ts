/**
 * The same decoder, on the GPU, dispatch for dispatch against `decoder.ts`.
 *
 * Issue #200. The reference is 21 seconds for 0.2 s of audio, which is what a
 * reference is for; this is the one a browser runs. It is deliberately the same
 * shape as `decoder.ts` — the same function names, the same order, the same
 * arguments — because the way to know a port is right is to be able to read the
 * two side by side, and `decoder-gpu.test.ts` compares them on the same latent.
 *
 * **No kernel here is new.** `ops/conv`, `ops/conv_transpose`, `ops/pad`,
 * `ops/snake` and `ops/axpy` cover the whole decoder. Two things that would
 * otherwise have needed one are folded away instead:
 *
 * - **The `ratio *` after the anti-aliasing upsample rides on the filter.** A
 *   convolution is linear in its weight, so scaling the twelve taps once at
 *   load scales every output.
 * - **The slice after it is the transposed convolution's `padding`.** The two
 *   trims are equal for this filter (15 and 15), and a symmetric crop of the
 *   output is what `padding` means to a transposed convolution. `decoder.ts`
 *   throws if a future filter makes them differ, rather than quietly dropping a
 *   sample off the front — a phase error no unit test hears.
 *
 * The final clamp to [-1, 1] happens on the CPU after readback. It is one pass
 * over the finished waveform, which is being copied to the host anyway to be
 * played, and `ops/activation` has no clamp — adding one to run it on the
 * device would be a kernel for nothing.
 */
import type { ResidentDevice, ResidentOp } from "../../../harness/resident.js";
import { params } from "../../../harness/wgsl.js";
import { amplePadding, type AudioVaeManifest, type AudioVaeWeights } from "./decoder.js";

/** The WGSL each stage needs, read by `kernels-node.ts` or fetched by the page. */
export const AUDIO_KERNEL_SOURCES: { key: keyof AudioKernels; op: string; entry: string }[] = [
  { key: "conv1d", op: "conv", entry: "kernel" },
  { key: "convTranspose1d", op: "conv_transpose", entry: "kernel" },
  { key: "pad", op: "pad", entry: "kernel" },
  { key: "snakeBeta", op: "snake", entry: "beta" },
  { key: "axpy", op: "axpy", entry: "inplace" },
  { key: "axpyOut", op: "axpy", entry: "kernel" },
];

export interface AudioKernels {
  conv1d: string;
  convTranspose1d: string;
  pad: string;
  snakeBeta: string;
  axpy: string;
  axpyOut: string;
}

const WG = 256;
const PAD_MODE = { constant: 0, reflect: 1, replicate: 2 } as const;
/** The x extent `ops/pad`'s kernel tiles its flat range into. */
const PAD_TILE_X = 256 * 256;

/** A device-side tensor: a buffer and the length that is actually live in it. */
interface Slab {
  buffer: GPUBuffer;
  C: number;
  L: number;
}

/**
 * Weights on the device, and the scratch to run through.
 *
 * Built once; `decode` is then a pure sequence of dispatches. Scratch buffers
 * are pooled by size rather than allocated per layer — a 7-stage decoder with
 * 21 AMP blocks asks for a few thousand intermediates, and creating a
 * `GPUBuffer` for each was measured on the LLM engine (issue #118) to cost more
 * than the dispatches do.
 */
export class AudioVaeGpu {
  private readonly weightBuffers = new Map<string, GPUBuffer>();
  /** `filter.expand(C)`, per channel count, at scale 1 and at scale `ratio`. */
  private readonly filters = new Map<string, GPUBuffer>();
  private readonly pool = new Map<number, GPUBuffer[]>();
  private readonly lent: GPUBuffer[] = [];
  /** Uniform buffers of this decode. Kept apart from `lent`: they are a
   *  different usage, and returning one to the storage pool would hand a later
   *  layer a buffer the device refuses to bind as storage. */
  private readonly lentUniforms: GPUBuffer[] = [];
  private readonly zeroBias = new Map<number, GPUBuffer>();

  constructor(
    private readonly device: ResidentDevice,
    private readonly kernels: AudioKernels,
    private readonly manifest: AudioVaeManifest,
    weights: AudioVaeWeights,
  ) {
    for (const entry of manifest.tensors) {
      const data = weights.get(entry.name);
      const buffer = device.createStorageBuffer(data.byteLength);
      device.upload(buffer, 0, data);
      this.weightBuffers.set(entry.name, buffer);

      // `ops/snake` deliberately does not exponentiate — its doc says why: the
      // DACVAE family stores values and the BigVGAN family stores logarithms,
      // and a kernel that chose one would be wrong for the other. This
      // checkpoint's convention is known (`snake_logscale: true`), so the
      // exponential happens here, once, rather than in 127 dispatches.
      if (manifest.snakeLogscale && (entry.name.endsWith(".alpha") || entry.name.endsWith(".beta"))) {
        const exp = Float32Array.from(data, Math.exp);
        const expBuffer = device.createStorageBuffer(exp.byteLength);
        device.upload(expBuffer, 0, exp);
        this.weightBuffers.set(`${entry.name}.exp`, expBuffer);
      } else if (!manifest.snakeLogscale) {
        this.weightBuffers.set(`${entry.name}.exp`, buffer);
      }
    }

    // The anti-aliasing filter, once per channel count the decoder visits and
    // once per scale. Small — twelve taps times at most 1024 channels — and it
    // saves a multiply pass and a broadcast at every one of the 127 activations.
    const filter = weights.get("antialias.filter");
    const channels = new Set<number>();
    for (let i = 0; i < manifest.upsampleRates.length; i += 1) channels.add(manifest.decoderDim / 2 ** (i + 1));
    for (const C of channels) {
      for (const scale of [1, manifest.antialiasRatio]) {
        const taps = new Float32Array(C * filter.length);
        for (let c = 0; c < C; c += 1) {
          for (let k = 0; k < filter.length; k += 1) taps[c * filter.length + k] = scale * filter[k]!;
        }
        const buffer = device.createStorageBuffer(taps.byteLength);
        device.upload(buffer, 0, taps);
        this.filters.set(`${C}:${scale}`, buffer);
      }
    }
  }

  /** A zero bias of `n` entries — the convolutions the checkpoint gives no bias to. */
  private bias(n: number): GPUBuffer {
    let buffer = this.zeroBias.get(n);
    if (!buffer) {
      // WebGPU zero-initialises a buffer it did not map at creation, so this is
      // zeros without an upload. Kept per size rather than one big one because
      // `bindGroup` takes whole buffers by design (see `resident.ts`).
      buffer = this.device.createStorageBuffer(n * 4);
      this.zeroBias.set(n, buffer);
    }
    return buffer;
  }

  private take(elements: number): GPUBuffer {
    // Rounded to a power of two so the pool has a handful of sizes rather than
    // one per layer; the decoder's lengths are all products of 2 and 5.
    const bytes = 2 ** Math.ceil(Math.log2(Math.max(1, elements) * 4));
    const free = this.pool.get(bytes);
    const buffer = free?.pop() ?? this.device.createStorageBuffer(bytes);
    this.lent.push(buffer);
    return buffer;
  }

  /** Returns everything `take` handed out. Called once per decode, not per layer. */
  private release(): void {
    for (const buffer of this.lent) {
      const bytes = buffer.size;
      const free = this.pool.get(bytes) ?? [];
      free.push(buffer);
      this.pool.set(bytes, free);
    }
    this.lent.length = 0;
    // Uniforms are small and there is one per dispatch, so they are destroyed
    // rather than pooled: a pool keyed by size would collide with the storage
    // pool's keys and hand a later dispatch a buffer of the wrong usage.
    for (const buffer of this.lentUniforms) buffer.destroy();
    this.lentUniforms.length = 0;
  }

  destroy(): void {
    for (const buffer of this.weightBuffers.values()) buffer.destroy();
    for (const buffer of this.filters.values()) buffer.destroy();
    for (const buffer of this.zeroBias.values()) buffer.destroy();
    for (const list of this.pool.values()) for (const buffer of list) buffer.destroy();
    for (const buffer of this.lent) buffer.destroy();
    for (const buffer of this.lentUniforms) buffer.destroy();
  }

  private uniform(values: Parameters<typeof params>[0]): GPUBuffer {
    const data = params(values);
    const buffer = this.device.createUniformBuffer(data.byteLength);
    this.device.upload(buffer, 0, new Uint8Array(data));
    this.lentUniforms.push(buffer);
    return buffer;
  }

  private async conv(
    ops: ResidentOp[],
    code: string,
    x: Slab,
    weight: GPUBuffer,
    bias: GPUBuffer,
    Cout: number,
    K: number,
    { stride = 1, padding = 0, dilation = 1, groups = 1, transposed = false } = {},
  ): Promise<Slab> {
    const Lout = transposed
      ? (x.L - 1) * stride - 2 * padding + dilation * (K - 1) + 1
      : Math.floor((x.L + 2 * padding - dilation * (K - 1) - 1) / stride) + 1;
    const out = this.take(Cout * Lout);
    const pipeline = await this.device.pipelineFor(code);
    ops.push({
      kind: "dispatch",
      pipeline,
      bindGroup: await this.device.bindGroup(pipeline, [x.buffer, weight, bias, out, this.uniform([
        ["u32", x.C], ["u32", Cout], ["u32", x.L], ["u32", K], ["u32", Lout],
        ["u32", stride], ["u32", padding], ["u32", dilation],
        ["u32", x.C / groups], ["u32", Cout / groups], ["u32", 0], ["u32", 0],
      ])]),
      workgroups: [Math.ceil(Lout / WG), Cout, 1],
    });
    return { buffer: out, C: Cout, L: Lout };
  }

  private async padAxis(ops: ResidentOp[], x: Slab, before: number, after: number): Promise<Slab> {
    const Lout = before + x.L + after;
    const total = x.C * Lout;
    const out = this.take(total);
    const groupsX = Math.min(PAD_TILE_X, Math.ceil(total / WG) * WG) / WG;
    const pipeline = await this.device.pipelineFor(this.kernels.pad);
    ops.push({
      kind: "dispatch",
      pipeline,
      bindGroup: await this.device.bindGroup(pipeline, [x.buffer, out, this.uniform([
        ["u32", x.C], ["u32", x.L], ["u32", 1], ["u32", before], ["u32", Lout],
        ["u32", PAD_MODE.replicate], ["f32", 0], ["u32", PAD_TILE_X],
      ])]),
      workgroups: [groupsX, Math.ceil(total / PAD_TILE_X), 1],
    });
    return { buffer: out, C: x.C, L: Lout };
  }

  /** `ops/snake`'s beta entry, with the logarithms already exponentiated on the host. */
  private async snake(ops: ResidentOp[], x: Slab, alpha: GPUBuffer, beta: GPUBuffer): Promise<Slab> {
    const out = this.take(x.C * x.L);
    const pipeline = await this.device.pipelineFor(this.kernels.snakeBeta);
    ops.push({
      kind: "dispatch",
      pipeline,
      bindGroup: await this.device.bindGroup(pipeline, [x.buffer, alpha, beta, out, this.uniform([
        ["u32", 1], ["u32", x.C], ["u32", x.L],
      ])]),
      workgroups: [Math.ceil((x.C * x.L) / WG), 1, 1],
    });
    return { buffer: out, C: x.C, L: x.L };
  }

  /** `y += a * x`, in place — the residual add and the multi-receptive-field mean. */
  private async accumulate(ops: ResidentOp[], y: GPUBuffer, x: GPUBuffer, n: number, a: number): Promise<void> {
    const pipeline = await this.device.pipelineFor(this.kernels.axpy);
    ops.push({
      kind: "dispatch",
      pipeline,
      bindGroup: await this.device.bindGroup(pipeline, [x, y, this.uniform([["u32", n], ["f32", a]])]),
      workgroups: [Math.ceil(n / WG), 1, 1],
    });
  }

  /** `out = a * x`, as `y + a*x` with a zero `y`. One dispatch, one buffer. */
  private async scale(ops: ResidentOp[], x: Slab, a: number): Promise<Slab> {
    const n = x.C * x.L;
    const out = this.take(n);
    const pipeline = await this.device.pipelineFor(this.kernels.axpyOut);
    ops.push({
      kind: "dispatch",
      pipeline,
      bindGroup: await this.device.bindGroup(pipeline, [x.buffer, this.bias(n), out, this.uniform([["u32", n], ["f32", a]])]),
      workgroups: [Math.ceil(n / WG), 1, 1],
    });
    return { buffer: out, C: x.C, L: x.L };
  }

  private async antialiasedSnake(ops: ResidentOp[], x: Slab, prefix: string): Promise<Slab> {
    const { antialiasRatio: ratio, antialiasKernelSize: K } = this.manifest;
    const padWidth = Math.floor(K / ratio) - 1;
    const trim = padWidth * ratio + Math.floor((K - ratio) / 2);

    const padded = await this.padAxis(ops, x, padWidth, padWidth);
    const upper = await this.conv(
      ops,
      this.kernels.convTranspose1d,
      padded,
      this.filters.get(`${x.C}:${ratio}`)!,
      this.bias(x.C),
      x.C,
      K,
      { stride: ratio, padding: trim, groups: x.C, transposed: true },
    );
    const activated = await this.snake(
      ops,
      upper,
      this.weightBuffers.get(`${prefix}.alpha.exp`)!,
      this.weightBuffers.get(`${prefix}.beta.exp`)!,
    );
    // `LowPassFilter1d`'s padding is asymmetric for an even kernel: 5 before and
    // 6 after at K=12. It is kept because it is upstream's, **not** because a
    // test can see it — mutating the 6 to a 5 leaves the golden unchanged, and
    // that is a property of the shape rather than a gap in the test. The
    // upsample makes `upperLength` exactly `2 * L` (the crop is chosen to), and
    // a stride-2 window over an even length never reaches the final padded
    // sample. Recorded rather than hidden: at a different ratio, or an odd
    // kernel, it would matter.
    const downLeft = Math.floor(K / 2) - (K % 2 === 0 ? 1 : 0);
    const downPadded = await this.padAxis(ops, activated, downLeft, Math.floor(K / 2));
    return this.conv(
      ops,
      this.kernels.conv1d,
      downPadded,
      this.filters.get(`${x.C}:1`)!,
      this.bias(x.C),
      x.C,
      K,
      { stride: ratio, groups: x.C },
    );
  }

  private async ampBlock(ops: ResidentOp[], x: Slab, index: number, kernel: number, dilations: number[]): Promise<Slab> {
    let current = x;
    for (let c = 0; c < dilations.length; c += 1) {
      const dilation = dilations[c]!;
      let t = await this.antialiasedSnake(ops, current, `resblocks.${index}.act.${2 * c}`);
      t = await this.conv(
        ops,
        this.kernels.conv1d,
        t,
        this.weightBuffers.get(`resblocks.${index}.convs1.${c}.weight`)!,
        this.weightBuffers.get(`resblocks.${index}.convs1.${c}.bias`)!,
        t.C,
        kernel,
        { dilation, padding: amplePadding(kernel, dilation) },
      );
      t = await this.antialiasedSnake(ops, t, `resblocks.${index}.act.${2 * c + 1}`);
      t = await this.conv(
        ops,
        this.kernels.conv1d,
        t,
        this.weightBuffers.get(`resblocks.${index}.convs2.${c}.weight`)!,
        this.weightBuffers.get(`resblocks.${index}.convs2.${c}.bias`)!,
        t.C,
        kernel,
        { padding: amplePadding(kernel, 1) },
      );
      // `x = xt + x`: the residual accumulates into the block's own output, so
      // the running tensor is the one that was just written and no copy happens.
      await this.accumulate(ops, t.buffer, current.buffer, t.C * t.L, 1);
      current = t;
    }
    return current;
  }

  /** A latent `[1, latentChannels, T]` in, a waveform out. Clamped on the host. */
  async decode(latent: Float32Array, T: number): Promise<Float32Array> {
    const m = this.manifest;
    const ops: ResidentOp[] = [];

    const input = this.take(latent.length);
    this.device.upload(input, 0, latent);

    let x: Slab = await this.conv(
      ops,
      this.kernels.conv1d,
      { buffer: input, C: m.latentChannels, L: T },
      this.weightBuffers.get("dec_in_proj.weight")!,
      this.weightBuffers.get("dec_in_proj.bias")!,
      m.latentDim,
      1,
    );
    x = await this.conv(
      ops,
      this.kernels.conv1d,
      x,
      this.weightBuffers.get("conv_pre.weight")!,
      this.weightBuffers.get("conv_pre.bias")!,
      m.decoderDim,
      7,
      { padding: 3 },
    );

    for (let i = 0; i < m.upsampleRates.length; i += 1) {
      const stride = m.upsampleRates[i]!;
      const K = m.upsampleKernelSizes[i]!;
      x = await this.conv(
        ops,
        this.kernels.convTranspose1d,
        x,
        this.weightBuffers.get(`ups.${i}.weight`)!,
        this.weightBuffers.get(`ups.${i}.bias`)!,
        m.decoderDim / 2 ** (i + 1),
        K,
        { stride, padding: (K - stride) / 2, transposed: true },
      );

      // The three blocks are averaged. The first writes the accumulator and the
      // other two add into it, all at 1/n — one pass fewer than summing and then
      // scaling, and no zeroed buffer to start from.
      const n = m.resblockKernelSizes.length;
      let accumulator: Slab | null = null;
      for (let j = 0; j < n; j += 1) {
        const block = await this.ampBlock(ops, x, i * n + j, m.resblockKernelSizes[j]!, m.resblockDilations[j]!);
        if (accumulator === null) accumulator = block;
        else await this.accumulate(ops, accumulator.buffer, block.buffer, block.C * block.L, 1);
      }
      // One scaled copy rather than scaling the first block in place: `y += a*x`
      // with `y` and `x` the same buffer would bind one buffer as both `read`
      // and `read_write`, which WebGPU rejects — and `ops/axpy`'s own doc
      // records that the failure is a silently invalidated command buffer and a
      // readback of zeros, not an error anybody is told about.
      x = await this.scale(ops, accumulator!, 1 / n);
    }

    x = await this.antialiasedSnake(ops, x, "activation_post");
    const waveform = await this.conv(
      ops,
      this.kernels.conv1d,
      x,
      this.weightBuffers.get("conv_post.weight")!,
      this.bias(1),
      1,
      7,
      { padding: 3 },
    );

    const staging = this.device.createStorageBuffer(waveform.L * 4, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);
    const [raw] = await this.device.batch(ops, [
      { staging, source: waveform.buffer, sourceOffset: 0, length: waveform.L, type: "f32" },
    ]);
    staging.destroy();
    this.release();

    const out = raw as Float32Array;
    if (m.useTanhAtFinal) return Float32Array.from(out, Math.tanh);
    return Float32Array.from(out, (v) => Math.min(1, Math.max(-1, v)));
  }
}
