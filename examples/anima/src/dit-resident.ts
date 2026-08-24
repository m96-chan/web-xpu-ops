/**
 * Anima's DiT forward with its weights and activations on the device.
 *
 * `dit.ts` is the definition of correct and takes 814 seconds for 64 image
 * tokens on the CPU reference ops — 52 blocks, each with self-attention, cross
 * attention, three adaLN modulations and an MLP. Nothing usable comes out of
 * that.
 *
 * The arrangement is `examples/zimage/src/dit-resident.ts`'s, which measured a
 * forward from 990 s to 0.17 s: ops are *recorded* into one command encoder per
 * block, activations live in `GPUBuffer`s from a pool, weights upload once and
 * persist across steps, and only the final latent is read back. What is not
 * shared is the block itself — cross-attention, three modulations, LayerNorm
 * instead of RMSNorm and a plain GELU MLP are Anima's, and writing them as a
 * parameterisation of Z-Image's block would leave one file where a reader has
 * to hold two models in their head.
 *
 * Correctness is `fixtures/forward.*`, the same golden the CPU port is held to.
 * The two are never compared against each other: two ports that drift the same
 * way agree with each other and with nothing else.
 */
import type { ResidentDevice, ResidentOp } from "../../../harness/resident.js";
import { params } from "../../../harness/wgsl.js";
import { ACTIVATION } from "../../../ops/activation/index.js";
import { ELEMENTWISE } from "../../../ops/elementwise/index.js";
import type { DitKernels } from "../../zimage/src/dit-gpu.js";
import { type AnimaConfig, type AnimaInput, type AnimaTrace, patchify, timestepEmbedding, unpatchify } from "./dit.js";
import { ropeAxisDims, ropeBases } from "./block.js";

const WG = 256;
const TILE = 16;
const MAX_WORKGROUPS = 65535;

interface Slot {
  buffer: GPUBuffer;
  bytes: number;
}

export interface AnimaResidentStats {
  dispatches: number;
  submits: number;
  poolSlots: number;
  poolBytes: number;
  weightBuffers: number;
  uploadedBytes: number;
}

/** A source that can hand back the packed q8 form as well as dense f32. */
export interface AnimaWeightSource {
  get(name: string): Float32Array;
  has(name: string): boolean;
  shapeOf(name: string): number[] | undefined;
  packedQ8(name: string): { codes: Uint32Array; scale: Float32Array; N: number; K: number } | null;
}

/**
 * Activation buffers, reused across batches but **never within one**.
 *
 * Ops here are recorded, not executed: handing a slot back while an unsubmitted
 * dispatch still names it lets a later dispatch overwrite a buffer an earlier
 * one is about to read. `examples/zimage` tried releasing mid-block and got
 * `used in submit while destroyed` — the driver catching what the reasoning
 * missed. Release happens at the boundary, and anything cleverer needs liveness
 * tracked rather than argued.
 */
class BufferPool {
  readonly #device: ResidentDevice;
  #free: Slot[] = [];
  #inFlight: Slot[] = [];
  #created = 0;

  constructor(device: ResidentDevice) {
    this.#device = device;
  }

  take(elements: number): Slot {
    const bytes = Math.max(16, elements * 4);
    let bestAt = -1;
    for (let i = 0; i < this.#free.length; i += 1) {
      if (this.#free[i]!.bytes < bytes) continue;
      if (bestAt < 0 || this.#free[i]!.bytes < this.#free[bestAt]!.bytes) bestAt = i;
    }
    const slot =
      bestAt >= 0
        ? this.#free.splice(bestAt, 1)[0]!
        : ((): Slot => {
            this.#created += 1;
            return { buffer: this.#device.createStorageBuffer(bytes), bytes };
          })();
    this.#inFlight.push(slot);
    return slot;
  }

  release(keep: Slot[]): void {
    const kept = new Set(keep);
    for (const slot of this.#inFlight) {
      if (!kept.has(slot)) this.#free.push(slot);
    }
    this.#inFlight = [...kept];
  }

  get created(): number {
    return this.#created;
  }

  get bytes(): number {
    let total = 0;
    for (const slot of [...this.#free, ...this.#inFlight]) total += slot.bytes;
    return total;
  }

  destroy(): void {
    for (const slot of [...this.#free, ...this.#inFlight]) slot.buffer.destroy();
    this.#free = [];
    this.#inFlight = [];
  }
}

/** Frees weights held across forwards. The caller owns them. */
export function releaseAnimaWeights(held: Map<string, GPUBuffer>): void {
  for (const buffer of held.values()) buffer.destroy();
  held.clear();
}

/** `animaForward`, recorded rather than dispatched. */
export async function animaForwardResident(
  device: ResidentDevice,
  K: DitKernels,
  cfg: AnimaConfig,
  weights: AnimaWeightSource,
  input: AnimaInput,
  stats?: AnimaResidentStats,
  /** Weights that survive between forwards — the point of the word "resident". */
  held?: Map<string, GPUBuffer>,
  trace?: AnimaTrace,
  onProgress?: (label: string, done: number, total: number) => void,
): Promise<Float32Array> {
  const { modelChannels: dim, numHeads, adalnLoraDim, inChannels, patchSpatial, patchTemporal, normEps } = cfg;
  const headDim = dim / numHeads;
  const { T, H, W } = input;
  const seq = (T / patchTemporal) * (H / patchSpatial) * (W / patchSpatial);
  const contextSeq = input.context.length / cfg.crossattnEmbChannels;
  const totalC = inChannels + (cfg.concatPaddingMask ? 1 : 0);
  const patchDim = patchTemporal * patchSpatial * patchSpatial * totalC;
  const axisDims = ropeAxisDims(headDim);
  const thetaPerAxis = ropeBases(axisDims, cfg.ropeExtrapolation);

  const pool = new BufferPool(device);
  const pipelines = new Map<string, GPUComputePipeline>();
  const weightBuffers = held ?? new Map<string, GPUBuffer>();
  const uniforms: GPUBuffer[] = [];
  let uniformAt = 0;
  let ops: ResidentOp[] = [];
  let uploaded = 0;
  let dispatches = 0;

  const pipelineFor = async (code: string): Promise<GPUComputePipeline> => {
    let pipeline = pipelines.get(code);
    if (!pipeline) {
      pipeline = await device.pipelineFor(code);
      pipelines.set(code, pipeline);
    }
    return pipeline;
  };

  const uniform = (data: ArrayBuffer): GPUBuffer => {
    if (uniformAt === uniforms.length) uniforms.push(device.createUniformBuffer(256));
    const buffer = uniforms[uniformAt]!;
    uniformAt += 1;
    device.upload(buffer, 0, new Uint8Array(data));
    return buffer;
  };

  const kernelNames = new Map<string, string>(Object.entries(K).map(([name, code]) => [code, name]));
  const record = async (
    code: string,
    buffers: GPUBuffer[],
    workgroups: [number] | [number, number] | [number, number, number],
  ): Promise<void> => {
    const over = workgroups.findIndex((n) => n > MAX_WORKGROUPS);
    if (over >= 0) {
      // Issue #112: going over is not always an error a caller sees, and Dawn's
      // own message names a dispatch rather than the op that made it.
      throw new Error(
        `animaForwardResident: ${kernelNames.get(code) ?? "?"} wants ${workgroups[over]} workgroups on ` +
          `dimension ${"xyz"[over]}, and the limit is ${MAX_WORKGROUPS} (issue #112).`,
      );
    }
    const pipeline = await pipelineFor(code);
    ops.push({ kind: "dispatch", pipeline, bindGroup: await device.bindGroup(pipeline, buffers), workgroups });
    dispatches += 1;
  };

  const flush = async (keep: Slot[], capture?: { name: keyof AnimaTrace; slot: Slot; length: number }): Promise<void> => {
    if (ops.length === 0 && !capture) return;
    if (trace && capture) {
      const staging = device.createStorageBuffer(capture.length * 4, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);
      const [read] = await device.batch(ops, [
        { staging, source: capture.slot.buffer, sourceOffset: 0, length: capture.length, type: "f32" },
      ]);
      trace[capture.name] = (read as Float32Array).slice();
      staging.destroy();
    } else if (ops.length > 0) {
      await device.batch(ops, []);
    }
    ops = [];
    uniformAt = 0;
    pool.release(keep);
  };

  /** A weight's buffer, uploading only if the device does not have it. */
  const weightBuffer = (name: string, make: () => ArrayBufferView): GPUBuffer => {
    const existing = weightBuffers.get(name);
    if (existing) return existing;
    const data = make();
    const buffer = device.createStorageBuffer(data.byteLength);
    device.upload(buffer, 0, data);
    uploaded += data.byteLength;
    weightBuffers.set(name, buffer);
    return buffer;
  };

  const upload = (data: Float32Array | Int32Array | Uint32Array): Slot => {
    const slot = pool.take(data.length);
    device.upload(slot.buffer, 0, data);
    uploaded += data.byteLength;
    return slot;
  };

  /** A weight as a `Slot`, so a bias can be passed where an activation is expected. */
  const weightSlot = (name: string, make: () => ArrayBufferView = () => weights.get(name)): Slot => ({
    buffer: weightBuffer(name, make),
    bytes: 0,
  });

  // --- one dispatch each ---

  const project = async (name: string, x: Slot, rows: number, inDim: number, outDim: number): Promise<Slot> => {
    const codesName = `${name}#c`;
    if (weightBuffers.has(codesName)) {
      // Already on the device: dispatch without asking the source for anything.
      const shape = weights.shapeOf(name);
      const [N, Kd] = shape && shape.length === 2 ? shape : [outDim, inDim];
      const out = pool.take(rows * N!);
      await record(
        K.matmulQ8,
        [
          x.buffer,
          weightBuffers.get(codesName)!,
          weightBuffers.get(`${name}#s`)!,
          out.buffer,
          uniform(params([["u32", rows], ["u32", N!], ["u32", Kd!]])),
        ],
        [Math.ceil(N! / TILE), Math.ceil(rows / TILE)],
      );
      return out;
    }

    const packed = weights.packedQ8(name);
    if (packed) {
      const out = pool.take(rows * packed.N);
      await record(
        K.matmulQ8,
        [
          x.buffer,
          weightBuffer(codesName, () => packed.codes),
          weightBuffer(`${name}#s`, () => packed.scale),
          out.buffer,
          uniform(params([["u32", rows], ["u32", packed.N], ["u32", packed.K]])),
        ],
        [Math.ceil(packed.N / TILE), Math.ceil(rows / TILE)],
      );
      return out;
    }

    // The f32 tail. `matmul` wants `[K, N]` and the checkpoint stores `[N, K]`,
    // so the transpose happens once per weight and lives on the device.
    const out = pool.take(rows * outDim);
    await record(
      K.matmul,
      [
        x.buffer,
        weightBuffer(`${name}#T`, () => {
          const weight = weights.get(name);
          const wT = new Float32Array(inDim * outDim);
          for (let o = 0; o < outDim; o += 1) {
            for (let i = 0; i < inDim; i += 1) wT[i * outDim + o] = weight[o * inDim + i]!;
          }
          return wT;
        }),
        out.buffer,
        uniform(params([["u32", rows], ["u32", outDim], ["u32", inDim]])),
      ],
      [Math.ceil(outDim / TILE), Math.ceil(rows / TILE)],
    );
    return out;
  };

  const rmsnorm = async (x: Slot, name: string, N: number, D: number): Promise<Slot> => {
    const out = pool.take(N * D);
    await record(
      K.rmsnorm,
      [x.buffer, weightBuffer(name, () => weights.get(name)), out.buffer, uniform(params([["u32", N], ["u32", D], ["f32", normEps]]))],
      [N],
    );
    return out;
  };

  /** LayerNorm with `elementwise_affine=False` — ones and zeros are the identity. */
  const layernormNoAffine = async (x: Slot, N: number, D: number): Promise<Slot> => {
    const out = pool.take(N * D);
    await record(
      K.layernorm,
      [
        x.buffer,
        weightBuffer("#lnWeight", () => new Float32Array(D).fill(1)),
        weightBuffer("#lnBias", () => new Float32Array(D)),
        out.buffer,
        uniform(params([["u32", N], ["u32", D], ["f32", 1e-6]])),
      ],
      [N],
    );
    return out;
  };

  /**
   * Splits a flat, per-element dispatch that would exceed the workgroup limit.
   *
   * Every op below is elementwise: lane `i` reads index `i` and writes index
   * `i`, with no state between lanes, so a split is exact rather than an
   * approximation — only the number of submits changes.
   *
   * Needed at Anima's shipped resolution and not at the fixture's: 832x1216 is
   * 3,952 tokens, and the MLP's 8,192-wide activation is 32.4M elements, or
   * 126,464 workgroups against a limit of 65,535. The `#112` guard named the op
   * and this is the answer to it, carried over from
   * `examples/zimage/src/dit-resident.ts` which met the same wall first.
   */
  const chunkedFlat = async (
    code: string,
    n: number,
    slice: (offset: number, count: number) => { buffer: GPUBuffer; offset: number; size: number }[],
  ): Promise<void> => {
    const perDispatch = MAX_WORKGROUPS * WG;
    const pipeline = await pipelineFor(code);
    // 64 elements is 256 bytes — the binding-offset alignment. A whole number
    // of workgroups is not required, only a whole number of elements.
    const step = n <= perDispatch ? n : Math.floor(perDispatch / 64) * 64;
    for (let at = 0; at < n; at += step) {
      const count = Math.min(step, n - at);
      ops.push({
        kind: "dispatch",
        pipeline,
        bindGroup: await device.bindGroupSliced(pipeline, slice(at, count)),
        workgroups: [Math.ceil(count / WG)],
      });
      dispatches += 1;
    }
  };

  const elementwise = async (a: Slot, b: Slot, n: number, kind: number): Promise<Slot> => {
    const out = pool.take(n);
    await chunkedFlat(K.elementwise, n, (at, count) => [
      { buffer: a.buffer, offset: at * 4, size: count * 4 },
      { buffer: b.buffer, offset: at * 4, size: count * 4 },
      { buffer: out.buffer, offset: at * 4, size: count * 4 },
      { buffer: uniform(params([["u32", count], ["u32", kind]])), offset: 0, size: 256 },
    ]);
    return out;
  };

  /**
   * `a` is `[S, D]`, `b` is `[D]` broadcast down the rows.
   *
   * Split by **rows**, not by elements: `b` is indexed by column, so a chunk
   * has to start at a row boundary or every lane in it reads the wrong entry of
   * `b`. That is why this is not `chunkedFlat`.
   */
  const rowsOp = async (a: Slot, b: Slot, S: number, D: number, kind: number): Promise<Slot> => {
    const out = pool.take(S * D);
    const perDispatch = MAX_WORKGROUPS * WG;
    if (S * D <= perDispatch) {
      await record(
        K.rows,
        [a.buffer, b.buffer, out.buffer, uniform(params([["u32", S], ["u32", D], ["u32", kind]]))],
        [Math.ceil((S * D) / WG)],
      );
      return out;
    }
    const rowBytes = D * 4;
    if (rowBytes % 256 !== 0) {
      throw new Error(
        `animaForwardResident: rows must be split but a row is ${rowBytes} bytes, not a multiple of 256.`,
      );
    }
    const rowsPer = Math.max(1, Math.floor(perDispatch / D));
    const pipeline = await pipelineFor(K.rows);
    for (let row = 0; row < S; row += rowsPer) {
      const count = Math.min(rowsPer, S - row);
      ops.push({
        kind: "dispatch",
        pipeline,
        bindGroup: await device.bindGroupSliced(pipeline, [
          { buffer: a.buffer, offset: row * rowBytes, size: count * rowBytes },
          { buffer: b.buffer, offset: 0, size: rowBytes },
          { buffer: out.buffer, offset: row * rowBytes, size: count * rowBytes },
          { buffer: uniform(params([["u32", count], ["u32", D], ["u32", kind]])), offset: 0, size: 256 },
        ]),
        workgroups: [Math.ceil((count * D) / WG)],
      });
      dispatches += 1;
    }
    return out;
  };

  const activation = async (x: Slot, n: number, kind: number): Promise<Slot> => {
    const out = pool.take(n);
    await chunkedFlat(K.activation, n, (at, count) => [
      { buffer: x.buffer, offset: at * 4, size: count * 4 },
      { buffer: out.buffer, offset: at * 4, size: count * 4 },
      { buffer: uniform(params([["u32", count], ["u32", kind], ["f32", 1]])), offset: 0, size: 256 },
    ]);
    return out;
  };

  const permute = async (x: Slot, dim0: number, dim1: number, D: number): Promise<Slot> => {
    const out = pool.take(dim0 * dim1 * D);
    await record(
      K.permute,
      [x.buffer, out.buffer, uniform(params([["u32", dim0], ["u32", dim1], ["u32", D]]))],
      [Math.ceil((dim0 * dim1 * D) / WG)],
    );
    return out;
  };

  const zeroBias = weightBuffer(`#zeroBias:${Math.max(seq, contextSeq)}`, () =>
    new Float32Array(Math.max(seq, contextSeq, 64)),
  );

  /**
   * Attention over all heads, `L` queries against `S` keys.
   *
   * Batched by a VRAM budget rather than taken whole: the scores are
   * `H * L * S` floats on the device, and at a large image that is gigabytes
   * next to the weights.
   */
  const SCORE_BUDGET = 512 * 1024 * 1024;
  const attention = async (q: Slot, k: Slot, v: Slot, L: number, S: number): Promise<Slot> => {
    const out = pool.take(numHeads * L * headDim);
    const perHead = L * S * 4;
    const perBatch = Math.max(1, Math.min(numHeads, Math.floor(SCORE_BUDGET / Math.max(perHead, 1))));
    const qBytes = L * headDim * 4;
    const kvBytes = S * headDim * 4;
    const scoresPipeline = await pipelineFor(K.scores);
    const contextPipeline = await pipelineFor(K.context);
    const probs = pool.take(perBatch * L * S);

    for (let h0 = 0; h0 < numHeads; h0 += perBatch) {
      const count = Math.min(perBatch, numHeads - h0);
      ops.push({
        kind: "dispatch",
        pipeline: scoresPipeline,
        bindGroup: await device.bindGroupSliced(scoresPipeline, [
          { buffer: q.buffer, offset: h0 * qBytes, size: count * qBytes },
          { buffer: k.buffer, offset: h0 * kvBytes, size: count * kvBytes },
          { buffer: zeroBias, offset: 0, size: Math.max(S * 4, 256) },
          { buffer: probs.buffer, offset: 0, size: count * L * S * 4 },
          {
            buffer: uniform(
              params([
                ["u32", count], ["u32", L], ["u32", S], ["u32", headDim],
                ["f32", 1 / Math.sqrt(headDim)],
                ["u32", 0], ["i32", 0],
                ["u32", 1], ["u32", 1], ["u32", 1],
              ]),
            ),
            offset: 0,
            size: 256,
          },
        ]),
        workgroups: [L, count, 1],
      });
      ops.push({
        kind: "dispatch",
        pipeline: contextPipeline,
        bindGroup: await device.bindGroupSliced(contextPipeline, [
          { buffer: probs.buffer, offset: 0, size: count * L * S * 4 },
          { buffer: v.buffer, offset: h0 * kvBytes, size: count * kvBytes },
          { buffer: out.buffer, offset: h0 * qBytes, size: count * qBytes },
          { buffer: uniform(params([["u32", count], ["u32", L], ["u32", S], ["u32", headDim]])), offset: 0, size: 256 },
        ]),
        workgroups: [L, count, 1],
      });
      dispatches += 2;
    }
    return out;
  };

  /**
   * Three-axis RoPE, one dispatch per axis, in place over the head.
   *
   * Each axis has its own base — this checkpoint's extrapolation ratios are 4.0
   * for h and w against 1.0 for t — and `ops/rope`'s `axes` entry takes one
   * base for the whole head. Splitting is exact: rope never mixes channels
   * across a pair, and no pair straddles an axis boundary.
   *
   * **No gather.** The first version copied each axis's channels out of every
   * head into a contiguous buffer, rotated, and copied back — `seq * heads * 3`
   * copies each way, and a `copy size` validation error the moment the pool
   * handed back a slot bigger than the one the offsets were computed for.
   * Instead the whole head is passed to `ropeAxes` with the **real** axis
   * split, and the axes whose base differs are handled by running it three
   * times with the other axes' positions set so they do not rotate.
   *
   * An angle of zero is the identity, and `ropeAxes` computes `pos * base^...`
   * — so a position of 0 leaves those channels untouched whatever the base is.
   * Three passes, each rotating one axis and leaving the other two alone.
   */
  const applyRope = async (x: Slot, positions: Slot[], rows: number): Promise<Slot> => {
    const axisBuffer = weightBuffer(`#axes:${axisDims.join(",")}`, () => Uint32Array.from(axisDims));
    // `head_dim`, not `width`. The first version passed the whole row and asked
    // the kernel to read `rows * numHeads * width` floats — sixteen times what
    // the buffer holds, which Dawn caught as a copy-size overrun rather than as
    // silently wrong numbers.
    const length = rows * numHeads * headDim;
    const workgroups = Math.ceil(length / 2 / WG);

    let current = x;
    for (const [axis] of axisDims.entries()) {
      // No padding: the kernel returns early past `total_pairs`, so a tail lane
      // reads nothing. `out` is exactly the tensor, same shape as the input.
      const out = pool.take(length);
      await record(
        K.ropeAxes,
        [
          current.buffer,
          axisBuffer,
          positions[axis]!.buffer,
          out.buffer,
          uniform(
            params([
              ["u32", rows], ["u32", numHeads], ["u32", headDim], ["u32", axisDims.length],
              ["f32", thetaPerAxis[axis]!],
            ]),
          ),
        ],
        [workgroups],
      );
      current = out;
    }
    return current;
  };

  // ============================================================
  // The forward. Structure follows `animaForward` exactly.
  // ============================================================

  const totalSteps = cfg.numBlocks + 2;
  let stepsDone = 0;
  const progress = (label: string): void => {
    stepsDone += 1;
    onProgress?.(label, stepsDone, totalSteps);
  };

  progress("timestep");
  const sample = upload(timestepEmbedding(input.t, dim, cfg.maxPeriod));
  const hidden = await activation(
    await project("net.t_embedder.1.linear_1.weight", sample, 1, dim, dim), dim, ACTIVATION.silu,
  );
  const adalnLora = await project("net.t_embedder.1.linear_2.weight", hidden, 1, dim, 3 * dim);
  const emb = await rmsnorm(sample, "net.t_embedding_norm.weight", 1, dim);
  await flush([emb, adalnLora], trace ? { name: "tEmbed", slot: emb, length: dim } : undefined);

  progress("patch embedder");
  let x = await project(
    "net.x_embedder.proj.1.weight",
    upload(patchify(input.latent, inChannels, T, H, W, patchSpatial, patchTemporal, cfg.concatPaddingMask)),
    seq,
    patchDim,
    dim,
  );
  if (weights.has("net.x_embedder.proj.1.bias")) {
    x = await rowsOp(x, weightSlot("net.x_embedder.proj.1.bias"), seq, dim, ELEMENTWISE.add);
  }

  const context = upload(input.context);
  /**
   * One `[seq, 3]` position array per axis, with the **other two axes zeroed**.
   *
   * `ropeAxes` takes one base for the whole head, and Anima's three axes do not
   * share one. Zeroing an axis's positions makes its angle zero, which is the
   * identity whatever base is in force — so pass number `i` rotates axis `i`
   * with axis `i`'s base and leaves the rest exactly as it found them.
   *
   * Four axes of slack past the end, as `ops/rope/testing.ts` does: a lane that
   * ran past `seq` reads there, and zeros would look like the identity rotation
   * rather than like a bug.
   */
  const positionSlots: Slot[] = axisDims.map((_, axis) => {
    const ids = new Int32Array(seq * 3 + axisDims.length * 4).fill(9999);
    let at = 0;
    for (let t = 0; t < T / patchTemporal; t += 1) {
      for (let h = 0; h < H / patchSpatial; h += 1) {
        for (let w = 0; w < W / patchSpatial; w += 1) {
          const coords = [t, h, w];
          for (let a = 0; a < 3; a += 1) ids[at + a] = a === axis ? coords[a]! : 0;
          at += 3;
        }
      }
    }
    return upload(ids);
  });

  const mlpHidden = weights.shapeOf("net.blocks.0.mlp.layer1.weight")?.[0] ?? 4 * dim;
  const width = numHeads * headDim;
  const kept = [emb, adalnLora, context, ...positionSlots];
  await flush([...kept, x]);

  for (let index = 0; index < cfg.numBlocks; index += 1) {
    progress(`block ${index + 1}/${cfg.numBlocks}`);
    const p = `net.blocks.${index}.`;

    /** One adaLN LoRA, chunked `shift, scale, gate`. */
    const modulation = async (which: string): Promise<{ shift: Slot; scale: Slot; gate: Slot }> => {
      const activated = await activation(emb, dim, ACTIVATION.silu);
      const inner = await project(`${p}adaln_modulation_${which}.1.weight`, activated, 1, dim, adalnLoraDim);
      const projected = await project(`${p}adaln_modulation_${which}.2.weight`, inner, 1, adalnLoraDim, 3 * dim);
      const biased = await elementwise(projected, adalnLora, 3 * dim, ELEMENTWISE.add);
      const chunk = (at: number): Slot => {
        const out = pool.take(dim);
        ops.push({ kind: "copy", src: biased.buffer, srcOffset: at * dim * 4, dst: out.buffer, dstOffset: 0, size: dim * 4 });
        return out;
      };
      return { shift: chunk(0), scale: chunk(1), gate: chunk(2) };
    };

    /** `norm(x) * (1 + scale) + shift`. */
    const modulate = async (normed: Slot, scale: Slot, shift: Slot, rows: number): Promise<Slot> => {
      const one = weightSlot("#ones", () => new Float32Array(dim).fill(1));
      const scalePlusOne = await elementwise(scale, one, dim, ELEMENTWISE.add);
      const scaled = await rowsOp(normed, scalePlusOne, rows, dim, ELEMENTWISE.multiply);
      return rowsOp(scaled, shift, rows, dim, ELEMENTWISE.add);
    };

    const attend = async (
      query: Slot,
      keyValue: Slot,
      prefix: string,
      L: number,
      S: number,
      kvDim: number,
      withRope: boolean,
    ): Promise<Slot> => {
      let q = await project(`${p}${prefix}.q_proj.weight`, query, L, dim, width);
      let k = await project(`${p}${prefix}.k_proj.weight`, keyValue, S, kvDim, width);
      const v = await project(`${p}${prefix}.v_proj.weight`, keyValue, S, kvDim, width);

      q = await rmsnorm(q, `${p}${prefix}.q_norm.weight`, L * numHeads, headDim);
      k = await rmsnorm(k, `${p}${prefix}.k_norm.weight`, S * numHeads, headDim);
      if (withRope) {
        q = await applyRope(q, positionSlots, L);
        k = await applyRope(k, positionSlots, S);
      }

      const attended = await attention(
        await permute(q, L, numHeads, headDim),
        await permute(k, S, numHeads, headDim),
        await permute(v, S, numHeads, headDim),
        L,
        S,
      );
      return project(`${p}${prefix}.output_proj.weight`, await permute(attended, numHeads, L, headDim), L, width, dim);
    };

    // --- self-attention ---
    {
      const mod = await modulation("self_attn");
      const normed = await layernormNoAffine(x, seq, dim);
      // The same tensor is both query and key/value — self-attention is the
      // case where `context is None` upstream (`model.py:63`). Computing the
      // modulation twice would be the same numbers at twice the cost.
      const modulated = await modulate(normed, mod.scale, mod.shift, seq);
      const result = await attend(modulated, modulated, "self_attn", seq, seq, dim, true);
      x = await elementwise(x, await rowsOp(result, mod.gate, seq, dim, ELEMENTWISE.multiply), seq * dim, ELEMENTWISE.add);
    }

    // --- cross-attention: no rope, keys and values from the context ---
    {
      const mod = await modulation("cross_attn");
      const normed = await layernormNoAffine(x, seq, dim);
      const result = await attend(await modulate(normed, mod.scale, mod.shift, seq), context, "cross_attn", seq, contextSeq, cfg.crossattnEmbChannels, false);
      x = await elementwise(x, await rowsOp(result, mod.gate, seq, dim, ELEMENTWISE.multiply), seq * dim, ELEMENTWISE.add);
    }

    // --- MLP: Linear -> GELU -> Linear, no gate ---
    {
      const mod = await modulation("mlp");
      const normed = await layernormNoAffine(x, seq, dim);
      const modulated = await modulate(normed, mod.scale, mod.shift, seq);
      // `nn.GELU()` with no argument is the **exact** one, not the tanh
      // approximation — `ops/activation` has both.
      const activated = await activation(
        await project(`${p}mlp.layer1.weight`, modulated, seq, dim, mlpHidden), seq * mlpHidden, ACTIVATION.gelu,
      );
      const result = await project(`${p}mlp.layer2.weight`, activated, seq, mlpHidden, dim);
      x = await elementwise(x, await rowsOp(result, mod.gate, seq, dim, ELEMENTWISE.multiply), seq * dim, ELEMENTWISE.add);
    }

    const capture =
      trace && index === 0
        ? ({ name: "afterBlock0", slot: x, length: seq * dim } as const)
        : trace && index === cfg.numBlocks - 1
          ? ({ name: "afterBlocks", slot: x, length: seq * dim } as const)
          : undefined;
    await flush([...kept, x], capture);
  }

  // --- final layer: two chunks, shift and scale, no gate ---
  const finalActivated = await activation(emb, dim, ACTIVATION.silu);
  const finalInner = await project("net.final_layer.adaln_modulation.1.weight", finalActivated, 1, dim, adalnLoraDim);
  const finalProjected = await project("net.final_layer.adaln_modulation.2.weight", finalInner, 1, adalnLoraDim, 2 * dim);
  // The LoRA's **first two thirds** — `adaln_lora[:, :, :2*D]` (`predict2.py:381`).
  const loraPrefix = pool.take(2 * dim);
  ops.push({ kind: "copy", src: adalnLora.buffer, srcOffset: 0, dst: loraPrefix.buffer, dstOffset: 0, size: 2 * dim * 4 });
  const finalMod = await elementwise(finalProjected, loraPrefix, 2 * dim, ELEMENTWISE.add);
  const finalShift = pool.take(dim);
  const finalScale = pool.take(dim);
  ops.push({ kind: "copy", src: finalMod.buffer, srcOffset: 0, dst: finalShift.buffer, dstOffset: 0, size: dim * 4 });
  ops.push({ kind: "copy", src: finalMod.buffer, srcOffset: dim * 4, dst: finalScale.buffer, dstOffset: 0, size: dim * 4 });

  const normedFinal = await layernormNoAffine(x, seq, dim);
  const one = weightSlot("#ones", () => new Float32Array(dim).fill(1));
  const scaled = await rowsOp(
    normedFinal,
    await elementwise(finalScale, one, dim, ELEMENTWISE.add),
    seq,
    dim,
    ELEMENTWISE.multiply,
  );
  const shifted = await rowsOp(scaled, finalShift, seq, dim, ELEMENTWISE.add);

  const outPatchDim = patchSpatial * patchSpatial * patchTemporal * cfg.outChannels;
  let projected = await project("net.final_layer.linear.weight", shifted, seq, dim, outPatchDim);
  if (weights.has("net.final_layer.linear.bias")) {
    projected = await rowsOp(projected, weightSlot("net.final_layer.linear.bias"), seq, outPatchDim, ELEMENTWISE.add);
  }

  // The only readback in the whole forward.
  const staging = device.createStorageBuffer(seq * outPatchDim * 4, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);
  const [out] = await device.batch(ops, [
    { staging, source: projected.buffer, sourceOffset: 0, length: seq * outPatchDim, type: "f32" },
  ]);
  ops = [];

  if (stats) {
    stats.dispatches = dispatches;
    stats.submits = device.stats.submits;
    stats.poolSlots = pool.created;
    stats.poolBytes = pool.bytes;
    stats.weightBuffers = weightBuffers.size;
    stats.uploadedBytes = uploaded;
  }

  const latent = unpatchify(out as Float32Array, cfg.outChannels, T, H, W, patchSpatial, patchTemporal);
  pool.destroy();
  staging.destroy();
  for (const buffer of uniforms) buffer.destroy();
  if (!held) for (const buffer of weightBuffers.values()) buffer.destroy();
  return latent;
}
