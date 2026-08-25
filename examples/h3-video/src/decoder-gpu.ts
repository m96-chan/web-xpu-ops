/**
 * MiniMax-H3's visual VAE decoder, on the GPU: a latent in, video frames out.
 *
 * Issue #200. `block.ts` is the same block on the CPU, checked against the
 * model's own output to 1.192e-7; this is 36 of them plus the embedding, the
 * projection and the two reshapes that turn tokens back into pixels. The two
 * files are deliberately the same shape so a divergence is readable.
 *
 * **No new kernel.** `matmul`, `rmsnorm`, `layernorm`, `flash_attention`,
 * `rope`'s axes entry, `activation`, `elementwise` and `permute` cover the whole
 * decoder.
 *
 * ## What the converter did so this file did not have to
 *
 * Three things that would each be a strided copy per token, done once at
 * conversion instead (`tools/convert_decoder.py`):
 *
 * - **`to_qkv` is split into three projections.** The model stores one matrix
 *   and reads it as `view(B, L, -1, 3 * dim_head).chunk(3, dim=-1)`, so head `h`
 *   owns contiguous `[q, k, v]`. Splitting the *weight* gives three ordinary
 *   matmuls whose outputs land contiguous.
 * - **`ff.w1` is split into gate and up**, for the same reason. The gate is the
 *   first half.
 * - **Every weight is stored `[in, out]`**, which is what `ops/matmul` reads.
 *
 * And the RoPE permutation is in the q and k weights, as `permuteForRope` does
 * for Anima.
 *
 * ## The two reshapes
 *
 * `_pack_tensors_3d(x, 1, 1)` at the input is a **channels-last flatten**:
 * `[1, 24, T, H, W]` becomes `[T·H·W, 24]`. At the output,
 * `_unpack_tensors_3d(o, 16, 4, …)` scatters each token's 3072 channels into a
 * `3 × 4 × 16 × 16` block of the video. Both are index arithmetic on the host
 * around a device-side buffer — the input is 24 channels and the output is read
 * back to be displayed anyway.
 */
import type { ResidentDevice, ResidentOp } from "../../../harness/resident.js";
import { params } from "../../../harness/wgsl.js";
import { ACTIVATION } from "../../../ops/activation/index.js";
import { ELEMENTWISE } from "../../../ops/elementwise/index.js";
import { FLASH_GENERATION, flashGrid } from "../../../ops/flash_attention/index.js";
import { matmulQ8Grid } from "../../../ops/matmul/index.js";

export const VIDEO_KERNEL_SOURCES: { key: keyof VideoKernels; op: string; entry: string }[] = [
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

export interface VideoKernels {
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

export interface VideoDecoderManifest {
  config: {
    heads: number;
    dim_head: number;
    num_layers: number;
    patch_size: number;
    patch_size_t: number;
    in_channels: number;
    out_channels: number;
    num_register_tokens: number;
    eps: number;
    rope_theta: number;
    rope_dim_ratio: number;
  };
  /** `"f32"` or `"q8"` — which layout the weight matrices are in. */
  dtype: string;
  dim: number;
  ffnHidden: number;
  ropeAxisDims: number[];
  pixelMean: number[];
  pixelStd: number[];
  tensors: { name: string; shape: number[]; offset: number; count: number }[];
  elements: number;
}

const WG = 256;
/** `ops/matmul`'s tile: `BM = 64`, `BN = 128`, 512 threads. */
const MM_BM = 64;
const MM_BN = 128;

/** A device-side matrix: a buffer and the shape that is live in it. */
interface Mat {
  buffer: GPUBuffer;
  rows: number;
  cols: number;
}

/**
 * The decoder's weights on the device, and the scratch to run through.
 *
 * 9.69 GB of f32 stays uploaded between decodes — the same arrangement
 * `dit-resident.ts` uses for Anima, and for the same measured reason:
 * re-uploading per forward was the difference between 0.2 s and 20 s a step.
 */
export class VideoDecoderGpu {
  private readonly weights = new Map<string, GPUBuffer>();
  private readonly pool = new Map<number, GPUBuffer[]>();
  private readonly lent: GPUBuffer[] = [];
  private readonly lentUniforms: GPUBuffer[] = [];
  /**
   * Blocks recorded into one command buffer.
   *
   * Everything a batch records has to stay live until it is submitted, so this
   * trades scratch memory against submit count directly. Set from the measured
   * default; `--blocks-per-submit` on `verify-decode.ts` sweeps it.
   */
  blocksPerSubmit = 1;

  private constructor(
    private readonly device: ResidentDevice,
    private readonly kernels: VideoKernels,
    private readonly manifest: VideoDecoderManifest,
  ) {}

  /**
   * Uploads the weights **one tensor at a time**, awaiting each.
   *
   * A factory rather than a constructor because the browser reads them from a
   * folder and a constructor cannot await — but the syntax is not the reason it
   * matters. Pre-reading the file into JavaScript would put 9.69 GB in the heap
   * before the first byte reached the device, and a single `Float32Array` over
   * it would be 2.4 billion elements, past what a typed array is guaranteed to
   * hold. Each tensor is uploaded and dropped.
   *
   * `onProgress` is how a page shows the half-minute this takes; without it the
   * tab looks frozen.
   */
  static async create(
    device: ResidentDevice,
    kernels: VideoKernels,
    manifest: VideoDecoderManifest,
    read: (offsetBytes: number, byteLength: number) => Uint8Array | Promise<Uint8Array>,
    onProgress?: (doneBytes: number, totalBytes: number) => void | Promise<void>,
  ): Promise<VideoDecoderGpu> {
    const self = new VideoDecoderGpu(device, kernels, manifest);
    let done = 0;
    for (const entry of manifest.tensors) {
      // **Bytes, not floats.** A `q8` tensor is packed int8 in `u32` words, and
      // reading those through a `Float32Array` would run every word through
      // f32's NaN canonicalisation on the way to the device -- a corruption
      // that only shows on the bit patterns that happen to be signalling NaNs,
      // which is to say on some weights and not others.
      const data = await read(entry.offset * 4, entry.count * 4);
      if (data.byteLength !== entry.count * 4) {
        throw new Error(`${entry.name}: read ${data.byteLength} bytes where the manifest says ${entry.count * 4}`);
      }
      const buffer = device.createStorageBuffer(data.byteLength);
      device.upload(buffer, 0, data);
      self.weights.set(entry.name, buffer);
      done += entry.count * 4;
      if (onProgress) await onProgress(done, manifest.elements * 4);
    }
    self.addConstants();
    return self;
  }

  /** The buffers the checkpoint has no tensor for. */
  private addConstants(): void {
    const { device, manifest } = this;
    // Two constants the checkpoint has no tensor for. `qk_norm_affine: false`
    // means the QK norms have no weights, and a vector of ones multiplies
    // without rounding in f32 -- one code path instead of two. `axis_dims` is
    // the rope kernel's own binding.
    const ones = new Float32Array(manifest.config.dim_head).fill(1);
    const onesBuffer = device.createStorageBuffer(ones.byteLength);
    device.upload(onesBuffer, 0, ones);
    this.weights.set("qk.ones", onesBuffer);

    const axisDims = Uint32Array.from(manifest.ropeAxisDims);
    const axisBuffer = device.createStorageBuffer(axisDims.byteLength);
    device.upload(axisBuffer, 0, axisDims);
    this.weights.set("rope.axisDims", axisBuffer);

    // `ops/flash_attention` takes an additive mask whatever the geometry, and
    // reads `S` entries of it. A row of zeros serves for "no mask", which is
    // this decoder's case -- but it has to be long enough. A read past the end
    // returns zero on this device, which is the right answer by accident and
    // would be the wrong one somewhere else.
    this.ensureMask(4096);
  }

  private maskBytes = 0;

  /** Grows the zero mask to cover a sequence of `seq` keys. */
  private ensureMask(seq: number): void {
    const bytes = seq * 4;
    if (bytes <= this.maskBytes) return;
    this.weights.get("attn.noMask")?.destroy();
    this.weights.set("attn.noMask", this.device.createStorageBuffer(bytes));
    this.maskBytes = bytes;
  }

  private w(name: string): GPUBuffer {
    const buffer = this.weights.get(name);
    // Named rather than undefined: a missing weight otherwise becomes a
    // zero-length read somewhere thirty layers down.
    if (!buffer) throw new Error(`decoder.bin has no tensor named "${name}"`);
    return buffer;
  }

  private take(elements: number): GPUBuffer {
    // Rounded to a multiple of 4 MB rather than to a power of two. The
    // decoder's widths are `dim`, `ffnHidden` and `heads * dim_head` times a
    // token count, so a power of two wastes up to half of every buffer -- at
    // 16,384 tokens that is 537 MB asked for and 1 GB allocated, thirty-one
    // times over, and the run fails where the multiple fits.
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
    for (const buffer of this.lentUniforms) buffer.destroy();
    this.lentUniforms.length = 0;
  }

  /**
   * Submits what has been recorded and returns the scratch, keeping `keep`.
   *
   * **One submit per block, not one per decode.** Every buffer a recorded
   * dispatch reads has to stay untouched until the command buffer runs, so
   * nothing inside a batch can be reused — and a 36-block decode records about
   * a thousand intermediates. At 24 tokens they are small enough not to notice;
   * at 1,029 the run allocated 464 buffers before Dawn started handing back
   * invalid ones, which is how this was found.
   *
   * `dit-resident.ts` splits Anima's forward the same way for the same reason.
   */
  private async flush(ops: ResidentOp[], keep: GPUBuffer[]): Promise<void> {
    if (ops.length === 0) return;
    const at = performance.now();
    await this.device.batch(ops, []);
    this.submitMs += performance.now() - at;
    this.dispatches += ops.length;
    ops.length = 0;
    this.release(keep);
  }

  /** Where a decode's wall clock went. Reset at the top of `decode`. */
  submitMs = 0;
  recordMs = 0;
  dispatches = 0;

  destroy(): void {
    for (const buffer of this.weights.values()) buffer.destroy();
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

  /**
   * No per-call timers here, deliberately.
   *
   * A `performance.now()` pair straddling an `await` charges whatever else the
   * event loop runs to whatever is being awaited. Instrumenting this function
   * that way attributed **502 ms to `pipelineFor`**, which a tight loop then
   * priced at **0.1 µs a call** — the same class of wrong answer
   * `accountForForward` was written for. `decode` times its phases instead, and
   * the split it reports is a whole-phase one.
   */
  private async dispatch(
    ops: ResidentOp[],
    code: string,
    buffers: GPUBuffer[],
    workgroups: [number] | [number, number] | [number, number, number],
  ): Promise<void> {
    const pipeline = await this.device.pipelineFor(code);
    ops.push({ kind: "dispatch", pipeline, bindGroup: await this.device.bindGroup(pipeline, buffers), workgroups });
  }

  /**
   * `a @ W`, then `+ bias` broadcast over rows.
   *
   * The two quantisations want **opposite layouts**, which is why the weight is
   * named rather than passed: `ops/matmul` reads `b` as `[K, N]`, so the f32
   * conversion transposes; `matmulQ8` reads `[M, K/4]` with one scale per `M`,
   * which is `nn.Linear`'s own `[out, in]` untransposed. The converter emits
   * whichever the manifest says, and this picks the kernel to match.
   */
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
    await this.dispatch(ops, this.kernels.rows, [out, bias, biased, this.uniform([
      ["u32", a.rows], ["u32", N], ["u32", ELEMENTWISE.add],
    ])], [Math.ceil((a.rows * N) / WG)]);
    return { buffer: biased, rows: a.rows, cols: N };
  }

  private async norm(ops: ResidentOp[], x: Mat, weight: GPUBuffer, eps: number, groups = 1): Promise<Mat> {
    const out = this.take(x.rows * x.cols);
    await this.dispatch(ops, this.kernels.rmsnorm, [x.buffer, weight, out, this.uniform([
      ["u32", x.rows], ["u32", x.cols], ["f32", eps], ["u32", groups],
    ])], [x.rows]);
    return { buffer: out, rows: x.rows, cols: x.cols };
  }

  private async rows(ops: ResidentOp[], x: Mat, vector: GPUBuffer, kind: number): Promise<Mat> {
    const out = this.take(x.rows * x.cols);
    await this.dispatch(ops, this.kernels.rows, [x.buffer, vector, out, this.uniform([
      ["u32", x.rows], ["u32", x.cols], ["u32", kind],
    ])], [Math.ceil((x.rows * x.cols) / WG)]);
    return { buffer: out, rows: x.rows, cols: x.cols };
  }

  private async pointwise(ops: ResidentOp[], a: Mat, b: Mat, kind: number): Promise<Mat> {
    const out = this.take(a.rows * a.cols);
    await this.dispatch(ops, this.kernels.elementwise, [a.buffer, b.buffer, out, this.uniform([
      ["u32", a.rows * a.cols], ["u32", kind],
    ])], [Math.ceil((a.rows * a.cols) / WG)]);
    return { buffer: out, rows: a.rows, cols: a.cols };
  }

  /** `[rows, heads, headDim]` <-> `[heads, rows, headDim]`. */
  private async swapLeading(ops: ResidentOp[], x: GPUBuffer, dim0: number, dim1: number, D: number): Promise<GPUBuffer> {
    const out = this.take(dim0 * dim1 * D);
    await this.dispatch(ops, this.kernels.permute, [x, out, this.uniform([
      ["u32", dim0], ["u32", dim1], ["u32", D],
    ])], [Math.ceil((dim0 * dim1 * D) / WG)]);
    return out;
  }

  private async block(ops: ResidentOp[], x: Mat, index: number, positions: GPUBuffer, mask: GPUBuffer): Promise<Mat> {
    const c = this.manifest.config;
    const { dim, ffnHidden } = this.manifest;
    const width = c.heads * c.dim_head;
    const seq = x.rows;
    const p = `blocks.${index}.`;

    const normed = await this.norm(ops, x, this.w(`${p}norm1.weight`), c.eps);
    let q = await this.linear(ops, normed, `${p}q.weight`, this.w(`${p}q.bias`), width);
    let k = await this.linear(ops, normed, `${p}k.weight`, this.w(`${p}k.bias`), width);
    const v = await this.linear(ops, normed, `${p}v.weight`, this.w(`${p}v.bias`), width);

    // QK-norm over each head's channels, with no weights (`qk_norm_affine:
    // false`). `groups` is what makes one dispatch normalise `seq * heads` rows
    // of `dim_head` inside a `[seq, width]` buffer.
    q = await this.qkNorm(ops, q, seq, c.heads, c.dim_head, c.eps);
    k = await this.qkNorm(ops, k, seq, c.heads, c.dim_head, c.eps);

    q = await this.rope(ops, q, seq, c.heads, c.dim_head, positions);
    k = await this.rope(ops, k, seq, c.heads, c.dim_head, positions);

    // `[seq, heads, dim_head]` -> `[heads, seq, dim_head]`, which is the layout
    // `ops/flash_attention` reads.
    const qh = await this.swapLeading(ops, q.buffer, seq, c.heads, c.dim_head);
    const kh = await this.swapLeading(ops, k.buffer, seq, c.heads, c.dim_head);
    const vh = await this.swapLeading(ops, v.buffer, seq, c.heads, c.dim_head);

    const attended = this.take(c.heads * seq * c.dim_head);
    await this.dispatch(ops, this.kernels.flashAttention, [qh, kh, vh, mask, attended, this.uniform([
      ["u32", c.heads], ["u32", seq], ["u32", seq], ["u32", c.dim_head], ["u32", c.dim_head],
      ["f32", 1 / Math.sqrt(c.dim_head)],
      // `causal_decoder: false` in the checkpoint's own config -- every token
      // sees every token, suffix tokens included.
      ["u32", 0], ["i32", 0], ["u32", 1], ["u32", 1], ["u32", 1],
    ])], flashGrid(seq, c.heads, 1));

    const merged = await this.swapLeading(ops, attended, c.heads, seq, c.dim_head);
    const projected = await this.linear(
      ops, { buffer: merged, rows: seq, cols: width }, `${p}out.weight`, this.w(`${p}out.bias`), dim,
    );
    // LayerScale: a per-channel parameter that multiplies the branch before it
    // is added back. Initialised to zero, so dropping it is a block that starts
    // from scratch every forward.
    const scaled = await this.rows(ops, projected, this.w(`${p}scale1`), ELEMENTWISE.multiply);
    let hidden = await this.pointwise(ops, x, scaled, ELEMENTWISE.add);

    const normed2 = await this.norm(ops, hidden, this.w(`${p}norm2.weight`), c.eps);
    const gate = await this.linear(ops, normed2, `${p}gate.weight`, this.w(`${p}gate.bias`), ffnHidden);
    const up = await this.linear(ops, normed2, `${p}up.weight`, this.w(`${p}up.bias`), ffnHidden);

    const activated = this.take(seq * ffnHidden);
    await this.dispatch(ops, this.kernels.activation, [gate.buffer, activated, this.uniform([
      ["u32", seq * ffnHidden], ["u32", ACTIVATION.silu], ["f32", 1],
    ])], [Math.ceil((seq * ffnHidden) / WG)]);

    const gated = await this.pointwise(
      ops, { buffer: activated, rows: seq, cols: ffnHidden }, up, ELEMENTWISE.multiply,
    );
    const ff = await this.linear(ops, gated, `${p}w2.weight`, this.w(`${p}w2.bias`), dim);
    const scaled2 = await this.rows(ops, ff, this.w(`${p}scale2`), ELEMENTWISE.multiply);
    hidden = await this.pointwise(ops, hidden, scaled2, ELEMENTWISE.add);
    return hidden;
  }

  /** RMS-norm over each head's `dim_head` channels, no weights. */
  private async qkNorm(ops: ResidentOp[], x: Mat, seq: number, heads: number, headDim: number, eps: number): Promise<Mat> {
    const out = this.take(seq * heads * headDim);
    await this.dispatch(ops, this.kernels.rmsnorm, [x.buffer, this.w("qk.ones"), out, this.uniform([
      ["u32", seq * heads], ["u32", headDim], ["f32", eps], ["u32", 1],
    ])], [seq * heads]);
    return { buffer: out, rows: seq, cols: heads * headDim };
  }

  /**
   * A latent `[in_channels, T, H, W]` in, the decoder's raw pixels out.
   *
   * The output is in the model's **normalised** space: `decode()` returns the
   * ViT's output and `get_denormalize_transform` is applied by the caller that
   * wants an image. `denormalise` below is that transform.
   */
  async decode(latent: Float32Array, dims: [number, number, number]): Promise<Float32Array> {
    const c = this.manifest.config;
    const { dim } = this.manifest;
    const [T, H, W] = dims;
    const patches = T * H * W;
    const suffix = 1 + c.num_register_tokens;
    const seq = patches + suffix;
    if (latent.length !== c.in_channels * patches) {
      throw new Error(`decode: latent holds ${latent.length} values, ${c.in_channels} x ${T} x ${H} x ${W} is ${c.in_channels * patches}`);
    }

    const ops: ResidentOp[] = [];
    this.submitMs = 0;
    this.dispatches = 0;
    const decodeStart = performance.now();
    this.ensureMask(seq);

    // `_pack_tensors_3d(z, 1, 1)` is a channels-last flatten: the latent
    // arrives `[C, T, H, W]` and the decoder wants `[T*H*W, C]`. Done on the
    // host because it is 24 channels and happens once.
    const packed = new Float32Array(patches * c.in_channels);
    for (let channel = 0; channel < c.in_channels; channel += 1) {
      for (let token = 0; token < patches; token += 1) {
        packed[token * c.in_channels + channel] = latent[channel * patches + token]!;
      }
    }
    const input = this.take(packed.length);
    this.device.upload(input, 0, packed);

    // `post_quant_conv` is a 1x1x1 Conv3d, which is a per-token linear.
    const quantised = await this.linear(
      ops, { buffer: input, rows: patches, cols: c.in_channels },
      "post_quant.weight", this.w("post_quant.bias"), c.in_channels,
    );
    const embedded = await this.linear(ops, quantised, "x_embedder.weight", this.w("x_embedder.bias"), dim,
    );

    // The suffix: four learned register tokens and one **zero** cls token,
    // which `ViT3DDecoder.forward` builds as `torch.zeros_like(hidden[:, 0:1])`
    // rather than as a parameter. A port that used a learned one would be
    // reading a tensor the checkpoint does not have.
    const withSuffix = this.take(seq * dim);
    ops.push({ kind: "copy", src: embedded.buffer, srcOffset: 0, dst: withSuffix, dstOffset: 0, size: patches * dim * 4 });
    ops.push({
      kind: "copy", src: this.w("register_tokens"), srcOffset: 0,
      dst: withSuffix, dstOffset: patches * dim * 4, size: c.num_register_tokens * dim * 4,
    });
    // The cls token is left as the zeros a fresh buffer already holds -- but
    // this buffer is pooled, so it is cleared explicitly.
    const zeros = new Float32Array(dim);
    this.device.upload(withSuffix, (patches + c.num_register_tokens) * dim * 4, zeros);

    const positions = this.take(seq * this.manifest.ropeAxisDims.length);
    this.device.upload(positions, 0, tokenAngles(dims, this.manifest.ropeAxisDims.length, suffix));

    let hidden: Mat = { buffer: withSuffix, rows: seq, cols: dim };
    for (let i = 0; i < c.num_layers; i += 1) {
      hidden = await this.block(ops, hidden, i, positions, this.w("attn.noMask"));
      // `positions` is read by every block, so it is kept alongside the
      // residual stream; everything else goes back to the pool.
      //
      // How often this happens is a trade, and `blocksPerSubmit` is where it is
      // made: everything a batch records stays live until it is submitted, so a
      // larger group holds proportionally more scratch and issues
      // proportionally fewer submits. Measured, not assumed -- see the README.
      if ((i + 1) % this.blocksPerSubmit === 0 || i === c.num_layers - 1) {
        await this.flush(ops, [hidden.buffer, positions]);
      }
    }

    // `norm_out` is a LayerNorm -- weight **and** bias -- where every norm
    // inside a block is an RMSNorm with weight only.
    const normed = this.take(seq * dim);
    await this.dispatch(ops, this.kernels.layernorm, [
      hidden.buffer, this.w("norm_out.weight"), this.w("norm_out.bias"), normed,
      this.uniform([["u32", seq], ["u32", dim], ["f32", c.eps]]),
    ], [seq]);

    const patchDim = c.out_channels * c.patch_size_t * c.patch_size * c.patch_size;
    const projected = await this.linear(
      ops, { buffer: normed, rows: seq, cols: dim }, "proj_out.weight", this.w("proj_out.bias"), patchDim,
    );

    // Only the patches are read back: the suffix tokens exist to give the
    // attention somewhere to put global information and are dropped here, as
    // `output[:, :num_patches, :]` does.
    const staging = this.device.createStorageBuffer(
      patches * patchDim * 4, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    );
    const [tokens] = await this.device.batch(ops, [
      { staging, source: projected.buffer, sourceOffset: 0, length: patches * patchDim, type: "f32" },
    ]);
    staging.destroy();
    this.release();
    // Everything that is not the queue: building bind groups, writing uniforms,
    // and the host-side reshapes. Named rather than left in the gap, because a
    // gap reads as GPU time.
    this.recordMs = performance.now() - decodeStart - this.submitMs;

    return unpackPatches(tokens as Float32Array, dims, c.out_channels, c.patch_size_t, c.patch_size);
  }

  private async rope(ops: ResidentOp[], x: Mat, seq: number, heads: number, headDim: number, positions: GPUBuffer): Promise<Mat> {
    const out = this.take(seq * heads * headDim);
    const axisDims = this.manifest.ropeAxisDims;
    await this.dispatch(ops, this.kernels.ropeAxes, [x.buffer, this.w("rope.axisDims"), positions, out, this.uniform([
      ["u32", seq], ["u32", heads], ["u32", headDim], ["u32", axisDims.length],
      ["f32", this.manifest.config.rope_theta],
    ])], [Math.ceil((seq * heads * (headDim / 2)) / WG)]);
    return { buffer: out, rows: seq, cols: heads * headDim };
  }
}

/**
 * `create_token_ids` times `2*pi` — the angle each token sits at, per axis.
 *
 * `2 * (i + 0.5) / n - 1` normalises each axis to `(-1, 1)`, which is why
 * `ops/rope`'s positions are `f32` (issue #200). Suffix tokens sit at zero on
 * every axis, so they are rotated by the identity; so is the fourth axis, which
 * is how `rope_dim_ratio: 0.75` is covered without a second code path.
 */
export function tokenAngles(
  [T, H, W]: [number, number, number],
  axes: number,
  suffix: number,
): Float32Array {
  const coordinate = (i: number, n: number): number => 2 * Math.PI * ((2 * (i + 0.5)) / n - 1);
  const out = new Float32Array((T * H * W + suffix) * axes);
  let at = 0;
  for (let t = 0; t < T; t += 1) {
    for (let h = 0; h < H; h += 1) {
      for (let w = 0; w < W; w += 1) {
        out[at * axes] = coordinate(t, T);
        out[at * axes + 1] = coordinate(h, H);
        out[at * axes + 2] = coordinate(w, W);
        at += 1;
      }
    }
  }
  return out;
}

/**
 * `_unpack_tensors_3d`: each token's `3 * 4 * 16 * 16` channels into its block.
 *
 * The model's `view(...).permute(0, 4, 1, 5, 2, 6, 3, 7).reshape(...)`, written
 * as the index arithmetic it is. On the host because the result is being read
 * back to be displayed regardless.
 */
export function unpackPatches(
  tokens: Float32Array,
  [T, H, W]: [number, number, number],
  channels: number,
  patchT: number,
  patch: number,
): Float32Array {
  const frames = T * patchT;
  const height = H * patch;
  const width = W * patch;
  const out = new Float32Array(channels * frames * height * width);
  const patchDim = channels * patchT * patch * patch;
  let token = 0;
  for (let t = 0; t < T; t += 1) {
    for (let h = 0; h < H; h += 1) {
      for (let w = 0; w < W; w += 1) {
        for (let c = 0; c < channels; c += 1) {
          for (let pt = 0; pt < patchT; pt += 1) {
            for (let ph = 0; ph < patch; ph += 1) {
              for (let pw = 0; pw < patch; pw += 1) {
                const from = token * patchDim + ((c * patchT + pt) * patch + ph) * patch + pw;
                const to = ((c * frames + t * patchT + pt) * height + h * patch + ph) * width + w * patch + pw;
                out[to] = tokens[from]!;
              }
            }
          }
        }
        token += 1;
      }
    }
  }
  return out;
}

/** `x * std + mean`, the ImageNet denormalisation `decode_videos` applies. */
export function denormalise(pixels: Float32Array, channels: number, mean: number[], std: number[]): Float32Array {
  const per = pixels.length / channels;
  const out = new Float32Array(pixels.length);
  for (let c = 0; c < channels; c += 1) {
    for (let i = 0; i < per; i += 1) out[c * per + i] = pixels[c * per + i]! * std[c]! + mean[c]!;
  }
  return out;
}
