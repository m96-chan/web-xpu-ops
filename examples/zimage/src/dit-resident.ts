/**
 * The DiT forward with its activations kept on the device.
 *
 * `dit-gpu.ts` dispatches through `harness/wgsl.ts`'s `Runner`, which is a test
 * harness: it uploads every binding, submits, waits, and **reads every output
 * back to the CPU**. That is the right shape for checking one kernel against a
 * reference and the wrong shape for a 34-layer forward. Measured at 1,039
 * tokens:
 *
 *     dispatches      1,209
 *     uploaded        28.57 GB
 *     read back       19.85 GB
 *     inside run()    92% of wall time
 *
 * Of the 48 GB moved, the weights are 6.17 GB. **The other ~42 GB is
 * activations going out to the CPU and straight back in**, because each
 * dispatch's output is the next one's input and the harness has no way to say
 * so.
 *
 * This file says so. Activations live in `GPUBuffer`s, a layer's dispatches are
 * recorded into one command encoder, and only the final latent is read back.
 * Weights are uploaded once per forward instead of once per dispatch.
 *
 * Correctness is `fixtures/forward.*` — the same golden the CPU and
 * `dit-gpu.ts` ports are held to. The three are never compared against each
 * other: two ports that drift the same way agree with each other and with
 * nothing else.
 */
import type { ResidentDevice, ResidentOp } from "../../../harness/resident.js";
import { params } from "../../../harness/wgsl.js";
import { ACTIVATION } from "../../../ops/activation/index.js";
import { ELEMENTWISE } from "../../../ops/elementwise/index.js";
import type { DitConfig, DitInput } from "./dit.js";
import type { DitKernels, PackedWeightSource } from "./dit-gpu.js";
import { captionPositionIds, imagePositionIds, patchify, timestepEmbedding, unpatchify } from "./dit.js";

const WG = 256;
const TILE = 16;

/** A device buffer and the bytes it can hold. */
interface Slot {
  buffer: GPUBuffer;
  bytes: number;
}

export interface ResidentDitStats {
  dispatches: number;
  submits: number;
  /** Every buffer the device made — weights, uniforms and activations together. */
  buffersCreated: number;
  /** Activation slots only. This is the one that says whether the pool works. */
  poolSlots: number;
  /** Buffers holding weights: two per q8 tensor, one per f32. */
  weightBuffers: number;
  uploadedBytes: number;
  readBackBytes: number;
}

/**
 * Activation buffers, reused **across batches but never within one**.
 *
 * That distinction is the whole hazard. Ops here are *recorded*, not executed:
 * handing a slot back while it is still named by an unsubmitted dispatch would
 * let a later dispatch in the same batch write to a buffer an earlier one is
 * about to read. So slots are only released at a batch boundary, where the GPU
 * has already consumed them.
 *
 * Within a layer that means every intermediate holds its own buffer — about 20
 * of them, the largest `seq * ffnHidden` — and they all come back at the end of
 * the layer.
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

  /** Releases everything recorded so far except `keep` — call only after a submit. */
  release(keep: Slot[]): void {
    const kept = new Set(keep);
    for (const slot of this.#inFlight) {
      if (kept.has(slot)) continue;
      this.#free.push(slot);
    }
    this.#inFlight = [...kept];
  }

  get created(): number {
    return this.#created;
  }

  destroy(): void {
    for (const slot of [...this.#free, ...this.#inFlight]) slot.buffer.destroy();
    this.#free = [];
    this.#inFlight = [];
  }
}

/**
 * `ditForwardGpu`'s structure, recorded rather than dispatched.
 *
 * Same config, weights and input; held to the same golden. Returns the
 * unpatchified latent, which is the only thing that comes back to the CPU.
 */
export async function ditForwardResident(
  device: ResidentDevice,
  K: DitKernels,
  cfg: DitConfig,
  weights: PackedWeightSource,
  input: DitInput,
  stats?: ResidentDitStats,
  prefetch?: (prefix: string, label: string) => Promise<void>,
  /**
   * Weights that survive between forwards.
   *
   * Uploading them per forward was the first version and is wrong for the job:
   * a generation is eight forwards, so 6.17 GB crossed the bus eight times for
   * bytes that never change. Passing the same map back in makes that
   * once — the actual point of the word "resident".
   *
   * The caller owns it, and `releaseDitWeights` is what frees it, because a
   * function that allocates 6.17 GB of VRAM and also decides when to drop it
   * would be deciding for a caller that knows better.
   */
  held?: Map<string, GPUBuffer>,
  /**
   * Called at every layer boundary, whether or not weights had to be fetched.
   *
   * Progress and weight-loading were the same callback until the weights
   * became resident — at which point the first generation reported progress
   * and every one after it silently showed nothing, because the hook that
   * drove the bar was the hook that had just been made unnecessary. Two
   * concerns, two callbacks.
   */
  onProgress?: (label: string, done: number, total: number) => void,
  /**
   * Reads a named intermediate back, for bisecting a mismatch.
   *
   * Off by default and never on a hot path: it forces a submit and a readback
   * at the checkpoint, which is exactly what the rest of this file avoids. It
   * exists because a 34-layer forward that reports only its last tensor tells
   * you that something is wrong and nothing about where — the same reason
   * `verify-forward.ts` compares in order.
   */
  trace?: Record<string, Float32Array>,
): Promise<Float32Array> {
  const { dim, nHeads, patchSize, inChannels, normEps } = cfg;
  const headDim = dim / nHeads;
  const width = nHeads * headDim;
  const { F, H, W } = input;
  const hTokens = H / patchSize;
  const wTokens = W / patchSize;
  const xSeq = F * hTokens * wTokens;
  const capSeq = input.capMask.length;
  const patchDim = patchSize * patchSize * inChannels;

  const pool = new BufferPool(device);
  const pipelines = new Map<string, GPUComputePipeline>();
  const weightBuffers = held ?? new Map<string, GPUBuffer>();
  const transposed = new Map<string, Float32Array>();
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

  /**
   * A uniform buffer for one dispatch.
   *
   * Allocated once and rewritten, because `queue.writeBuffer` is ordered
   * against the submit that follows it — but only until that submit happens,
   * which is why the counter resets at a batch boundary and not before.
   */
  const uniform = (data: ArrayBuffer): GPUBuffer => {
    if (uniformAt === uniforms.length) uniforms.push(device.createUniformBuffer(256));
    const buffer = uniforms[uniformAt]!;
    uniformAt += 1;
    device.upload(buffer, 0, new Uint8Array(data));
    return buffer;
  };

  /**
   * WebGPU's per-dimension workgroup limit, and the reason this is checked here
   * rather than discovered at submit time.
   *
   * Issue #112: going over is not always an error a caller sees — some devices
   * invalidate the command buffer and hand back a stale output buffer, which
   * reads as plausible numbers. Dawn does report it, but by then the message
   * names a dispatch and not the op that made it. Checked at the point where
   * the op still has a name.
   */
  const MAX_WORKGROUPS = 65535;
  const kernelNames = new Map<string, string>(Object.entries(K).map(([name, code]) => [code, name]));

  const record = async (
    code: string,
    buffers: GPUBuffer[],
    workgroups: [number] | [number, number] | [number, number, number],
  ): Promise<void> => {
    const over = workgroups.findIndex((n) => n > MAX_WORKGROUPS);
    if (over >= 0) {
      throw new Error(
        `ditForwardResident: ${kernelNames.get(code) ?? "?"} wants ${workgroups[over]} workgroups on ` +
          `dimension ${"xyz"[over]}, and the limit is ${MAX_WORKGROUPS} (issue #112). ` +
          `Dispatch was [${workgroups.join(", ")}].`,
      );
    }
    const pipeline = await pipelineFor(code);
    ops.push({ kind: "dispatch", pipeline, bindGroup: await device.bindGroup(pipeline, buffers), workgroups });
    dispatches += 1;
  };

  /** Submits, then lets the pool and the uniforms be reused. */
  const flush = async (keep: Slot[], capture?: { name: string; slot: Slot; length: number }): Promise<void> => {
    if (ops.length === 0 && !capture) return;
    if (trace && capture) {
      const staging = device.createStorageBuffer(
        capture.length * 4,
        GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      );
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

  /**
   * Calls `prefetch` only when this prefix's weights are not already on the GPU.
   *
   * `prefetch` exists so a browser can pull a layer's tensors out of its disk
   * cache before the synchronous accessors read them. Once those tensors are
   * *resident*, that read is pure waste — and it is not small: 6.17 GB per
   * step, at a measured 1,156 MB/s, is five seconds of every step spent
   * fetching bytes the device already holds.
   *
   * The test is whether the map the caller is holding already names a buffer
   * from this prefix. That is the same map `weightBuffer` fills, so there is
   * one source of truth about what is resident rather than a second set to
   * keep in sync with it.
   */
  /** Total layer boundaries, for the progress fraction. */
  const totalSteps = cfg.nRefinerLayers * 2 + cfg.nLayers + 3;
  let stepsDone = 0;

  const needWeights = async (prefix: string, label: string): Promise<void> => {
    stepsDone += 1;
    onProgress?.(label, stepsDone, totalSteps);
    if (!prefetch) return;
    for (const name of weightBuffers.keys()) {
      if (name.startsWith(prefix)) return;
    }
    await prefetch(prefix, label);
  };

  /**
   * A weight's buffer, uploading it only if the device does not have it.
   *
   * `make` is a thunk rather than the data because on a hit **nothing should
   * touch the CPU side at all** — reading it would be the disk fetch this
   * whole arrangement exists to skip, and skipping the prefetch while still
   * calling `weights.get` is how the second generation threw
   * "read before it was preloaded".
   */
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

  /**
   * A weight as a `Slot`, so it can be passed where an activation is expected.
   *
   * Biases and the pad token are weights: constant across every step, and
   * therefore things to upload once rather than per forward. They were going
   * through `upload()` into the activation pool, which re-read and re-sent them
   * every time.
   */
  const weightSlot = (name: string, make: () => ArrayBufferView = () => weights.get(name)): Slot => {
    const buffer = weightBuffer(name, make);
    return { buffer, bytes: 0 };
  };

  const upload = (data: Float32Array | Int32Array | Uint32Array): Slot => {
    const slot = pool.take(data.length);
    device.upload(slot.buffer, 0, data);
    uploaded += data.byteLength;
    return slot;
  };

  // --- one dispatch each ---

  const project = async (name: string, x: Slot, rows: number, inDim: number, outDim: number): Promise<Slot> => {
    // Already on the device: dispatch without asking the weight source for
    // anything. Calling `packedQ8` here would read it back off disk for the
    // shapes alone, which is the cost this path exists to avoid — so the shapes
    // come from the manifest and the buffers from the map.
    const codesName = `${name}#c`;
    if (weightBuffers.has(codesName)) {
      const shape = weights.shapeOf?.(name);
      const [N, Kdim] = shape && shape.length === 2 ? shape : [outDim, inDim];
      const out = pool.take(rows * N!);
      await record(
        K.matmulQ8,
        [
          x.buffer,
          weightBuffers.get(codesName)!,
          weightBuffers.get(`${name}#s`)!,
          out.buffer,
          uniform(params([["u32", rows], ["u32", N!], ["u32", Kdim!]])),
        ],
        [Math.ceil(N! / TILE), Math.ceil(rows / TILE)],
      );
      return out;
    }
    if (weightBuffers.has(`${name}#T`)) {
      const out = pool.take(rows * outDim);
      await record(
        K.matmul,
        [x.buffer, weightBuffers.get(`${name}#T`)!, out.buffer, uniform(params([["u32", rows], ["u32", outDim], ["u32", inDim]]))],
        [Math.ceil(outDim / TILE), Math.ceil(rows / TILE)],
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
          weightBuffer(`${name}#c`, () => packed.codes),
          weightBuffer(`${name}#s`, () => packed.scale),
          out.buffer,
          uniform(params([["u32", rows], ["u32", packed.N], ["u32", packed.K]])),
        ],
        [Math.ceil(packed.N / TILE), Math.ceil(rows / TILE)],
      );
      return out;
    }
    // The f32 tail the converter leaves unquantized. `matmul` wants `[K, N]`
    // and the checkpoint stores `[N, K]`, so the transpose is done once per
    // weight and cached rather than per dispatch.
    let wT = transposed.get(name);
    if (!wT) {
      const weight = weights.get(name);
      wT = new Float32Array(inDim * outDim);
      for (let o = 0; o < outDim; o += 1) {
        for (let i = 0; i < inDim; i += 1) wT[i * outDim + o] = weight[o * inDim + i]!;
      }
      transposed.set(name, wT);
    }
    const out = pool.take(rows * outDim);
    await record(
      K.matmul,
      [x.buffer, weightBuffer(`${name}#T`, () => wT!), out.buffer, uniform(params([["u32", rows], ["u32", outDim], ["u32", inDim]]))],
      [Math.ceil(outDim / TILE), Math.ceil(rows / TILE)],
    );
    return out;
  };

  /**
   * RMSNorm, split by rows when there are more than a dispatch can carry.
   *
   * The kernel is one workgroup per row, so `N` *is* the dispatch — and QK-Norm
   * counts rows per head, not per token: at 1024x1024 that is 4096 tokens x 30
   * heads = **122,880**, against a limit of 65,535 (issue #112). At 512 it is
   * 30,720 and fits, which is exactly the shape of bug that ships.
   *
   * Rows are independent, so a split is exact rather than an approximation.
   * Each chunk binds its own slice of the input and output, which
   * `createBindGroup` allows at a 256-byte offset — `D * 4` is a multiple of
   * 256 for every shape here (`headDim` 128, `dim` 3840), and it is asserted
   * rather than assumed because a shape where it is not would silently fail
   * validation somewhere else.
   */
  const rmsnorm = async (x: Slot, name: string, N: number, D: number): Promise<Slot> => {
    const out = pool.take(N * D);
    const weight = weightBuffer(name, () => weights.get(name));
    if (N <= MAX_WORKGROUPS) {
      await record(
        K.rmsnorm,
        [x.buffer, weight, out.buffer, uniform(params([["u32", N], ["u32", D], ["f32", normEps]]))],
        [N],
      );
      return out;
    }

    const rowBytes = D * 4;
    if (rowBytes % 256 !== 0) {
      throw new Error(
        `ditForwardResident: rmsnorm needs ${N} rows, which must be split, but a row is ${rowBytes} ` +
          `bytes and a binding offset must be a multiple of 256.`,
      );
    }
    const perChunk = Math.min(MAX_WORKGROUPS, Math.floor((256 * 1024 * 1024) / rowBytes));
    const pipeline = await pipelineFor(K.rmsnorm);
    for (let row = 0; row < N; row += perChunk) {
      const rows = Math.min(perChunk, N - row);
      ops.push({
        kind: "dispatch",
        pipeline,
        bindGroup: await device.bindGroupSliced(pipeline, [
          { buffer: x.buffer, offset: row * rowBytes, size: rows * rowBytes },
          { buffer: weight, offset: 0, size: rowBytes },
          { buffer: out.buffer, offset: row * rowBytes, size: rows * rowBytes },
          { buffer: uniform(params([["u32", rows], ["u32", D], ["f32", normEps]])), offset: 0, size: 256 },
        ]),
        workgroups: [rows],
      });
      dispatches += 1;
    }
    return out;
  };

  /**
   * Splits a flat, per-element dispatch that would exceed the workgroup limit.
   *
   * Every op below is elementwise: lane `i` reads index `i` and writes index
   * `i`, with no state between lanes, so a split is exact rather than an
   * approximation — only the number of submits changes. The chunk is a
   * multiple of 64 elements so each slice's byte offset is a multiple of 256,
   * which is what `bindGroupSliced` requires.
   *
   * `dit-gpu.ts` solves the same problem by slicing the `Float32Array` before
   * upload; there is no array here, which is the point, so the slice moves to
   * the binding.
   */
  const chunkedFlat = async (
    code: string,
    n: number,
    slice: (offset: number, count: number) => { buffer: GPUBuffer; offset: number; size: number }[],
  ): Promise<void> => {
    const perDispatch = MAX_WORKGROUPS * WG;
    if (n <= perDispatch) {
      const pipeline = await pipelineFor(code);
      ops.push({
        kind: "dispatch",
        pipeline,
        bindGroup: await device.bindGroupSliced(pipeline, slice(0, n)),
        workgroups: [Math.ceil(n / WG)],
      });
      dispatches += 1;
      return;
    }
    // 64 elements is 256 bytes — the binding-offset alignment — and a whole
    // number of workgroups is not required, only a whole number of elements.
    const step = Math.floor(perDispatch / 64) * 64;
    const pipeline = await pipelineFor(code);
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
   * has to start at a row boundary or every lane in it reads the wrong entry
   * of `b`. That is the difference between this and `chunkedFlat` above, and
   * it is why they are not one function.
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
      throw new Error(`ditForwardResident: rows must be split but a row is ${rowBytes} bytes, not a multiple of 256.`);
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

  /**
   * Multi-axis RoPE.
   *
   * The kernel works two channels per lane and rounds the dispatch up to a
   * whole workgroup, so the buffer carries that slack — `ops/rope/testing.ts`
   * does the same. The positions get four axes of slack for the reason its doc
   * gives: a lane past `N` reads there, and zeros would look like the identity
   * rotation rather than like a bug.
   */
  const ropeAxes = async (x: Slot, seq: number, positions: Slot, axisSlots: GPUBuffer, slots: number): Promise<Slot> => {
    const out = pool.take(slots);
    await record(
      K.ropeAxes,
      [
        x.buffer,
        axisSlots,
        positions.buffer,
        out.buffer,
        uniform(
          params([
            ["u32", seq], ["u32", nHeads], ["u32", headDim], ["u32", cfg.ropeAxesDims.length],
            ["f32", cfg.ropeTheta],
          ]),
        ),
      ],
      [slots / (WG * 2)],
    );
    return out;
  };

  const axisDims = weightBuffer("#axisDims", () => Uint32Array.from(cfg.ropeAxesDims));

  /**
   * Non-causal attention over all heads.
   *
   * Batched, and without the host-memory cap `dit-gpu.ts` needs: nothing is
   * read back here, so the scores never become a JavaScript allocation. The
   * only ceiling left is the device's own storage-binding limit, which at these
   * shapes is not close.
   */
  /**
   * Non-causal attention, in head-sized batches.
   *
   * The scores are `H * L * S` floats and that is **VRAM**, not host memory —
   * `dit-gpu.ts` caps this because every output came back to JavaScript, and
   * here nothing does. At 1024x1024 the whole batch is 2.03 GB of scores next
   * to 6.17 GB of resident weights, on a card with 13.4 GB free, and the
   * allocation that failed was a `vkAllocateMemory`.
   *
   * So the batch is sized by bytes rather than taken whole. Heads are
   * independent, and `[H, L, D]` is head-major, so a chunk is a contiguous
   * range of both input and output — the split is exact.
   */
  const SCORE_BUDGET = 512 * 1024 * 1024;
  const attention = async (q: Slot, k: Slot, v: Slot, seq: number, zeroBias: GPUBuffer): Promise<Slot> => {
    const out = pool.take(nHeads * seq * headDim);
    const perHeadScores = seq * seq * 4;
    const perBatch = Math.max(1, Math.min(nHeads, Math.floor(SCORE_BUDGET / Math.max(perHeadScores, 1))));
    const qkBytes = seq * headDim * 4;

    const scoresPipeline = await pipelineFor(K.scores);
    const contextPipeline = await pipelineFor(K.context);
    const probs = pool.take(perBatch * seq * seq);

    for (let h0 = 0; h0 < nHeads; h0 += perBatch) {
      const count = Math.min(perBatch, nHeads - h0);
      ops.push({
        kind: "dispatch",
        pipeline: scoresPipeline,
        bindGroup: await device.bindGroupSliced(scoresPipeline, [
          { buffer: q.buffer, offset: h0 * qkBytes, size: count * qkBytes },
          { buffer: k.buffer, offset: h0 * qkBytes, size: count * qkBytes },
          { buffer: zeroBias, offset: 0, size: Math.max(seq * 4, 256) },
          { buffer: probs.buffer, offset: 0, size: count * seq * seq * 4 },
          {
            buffer: uniform(
              params([
                ["u32", count], ["u32", seq], ["u32", seq], ["u32", headDim],
                ["f32", 1 / Math.sqrt(headDim)],
                ["u32", 0], ["i32", 0],
                ["u32", 1], ["u32", 1], ["u32", 1],
              ]),
            ),
            offset: 0,
            size: 256,
          },
        ]),
        workgroups: [seq, count, 1],
      });
      ops.push({
        kind: "dispatch",
        pipeline: contextPipeline,
        bindGroup: await device.bindGroupSliced(contextPipeline, [
          { buffer: probs.buffer, offset: 0, size: count * seq * seq * 4 },
          { buffer: v.buffer, offset: h0 * qkBytes, size: count * qkBytes },
          { buffer: out.buffer, offset: h0 * qkBytes, size: count * qkBytes },
          {
            buffer: uniform(params([["u32", count], ["u32", seq], ["u32", seq], ["u32", headDim]])),
            offset: 0,
            size: 256,
          },
        ]),
        workgroups: [seq, count, 1],
      });
      dispatches += 2;
    }
    return out;
  };

  /**
   * `[S, H, D]` to `[H, S, D]`, on the GPU.
   *
   * `ops/permute` is exactly this shape (`[dim0, dim1, D] -> [dim1, dim0, D]`).
   * `dit-gpu.ts` does it in JavaScript, which costs a readback — the one thing
   * this file exists to avoid — so the reshape that was free there is a
   * dispatch here, and the op for it already existed.
   */
  const permute = async (x: Slot, dim0: number, dim1: number, D: number): Promise<Slot> => {
    const out = pool.take(dim0 * dim1 * D);
    await record(
      K.permute,
      [x.buffer, out.buffer, uniform(params([["u32", dim0], ["u32", dim1], ["u32", D]]))],
      [Math.ceil((dim0 * dim1 * D) / WG)],
    );
    return out;
  };

  // **Keyed by size.** These two are shape-dependent and live in the same cache
  // as the weights, which survives across forwards — so a page that generates
  // at 256 and then at 512 would reuse a buffer sized for 271 tokens and bind a
  // range past its end. Caught by validation rather than read as zeros, which
  // is the good case, and fixed by making the name say the size.
  const zeroBias = weightBuffer(`#zeroBias:${xSeq + capSeq}`, () => new Float32Array(Math.max(xSeq + capSeq, 256)));
  /** Zeros, for padding an attention output back to full length. */
  const zeroTail = weightBuffer(`#zeroTail:${xSeq + capSeq}`, () => new Float32Array((xSeq + capSeq) * width));

  /** One block, recorded. Mirrors `dit-gpu.ts`'s `block`, including the trim. */
  const block = async (
    prefix: string,
    modulated: boolean,
    x: Slot,
    adaln: Slot | null,
    positions: Slot,
    seq: number,
    ffnHidden: number,
    /**
     * How many leading tokens are real.
     *
     * **Trims, does not mask** — `modules/attention.py` slices q/k/v to the
     * valid length, discards the mask and pads the output back with zeros, so
     * a padded row's attention output is exactly zero and the residual leaves
     * it untouched. `zimageBlock`'s own doc has the two measurements that
     * separate this from masking.
     *
     * Writing this block without the parameter — which is what the first
     * version of this file did — measures 1.234 at `afterContextRefiner`,
     * against the 1.21 that ignoring the padding measured in `dit.ts`. Same
     * bug, same number, second time.
     */
    validSeq: number,
  ): Promise<Slot> => {
    const w = (name: string): string => `${prefix}${name}`;
    const ropeSlots = Math.ceil((seq * nHeads * headDim) / 2 / WG) * WG * 2;

    let scaleMsa: Slot;
    let gateMsa: Slot;
    let scaleMlp: Slot;
    let gateMlp: Slot;
    if (adaln === null) {
      // `modulation=False` — `context_refiner`'s layers. Ones multiply without
      // rounding in f32, so one path serves both rather than two branches for
      // the residuals to disagree in.
      const ones = weightSlot("#ones", () => new Float32Array(dim).fill(1));
      scaleMsa = ones;
      gateMsa = ones;
      scaleMlp = ones;
      gateMlp = ones;
    } else {
      const mod = await project(w("adaLN_modulation.0.weight"), adaln, 1, cfg.adalnEmbedDim, 4 * dim);
      const biased = await rowsOp(mod, weightSlot(w("adaLN_modulation.0.bias")), 1, 4 * dim, ELEMENTWISE.add);
      // The four chunks are contiguous in one buffer; each dispatch below reads
      // its own by binding a slice-shaped uniform. Splitting on the GPU would
      // need a copy per chunk, so instead the chunks are separated by four
      // `permute`-free `elementwiseRows` reads over `[1, 4*dim]` — see the
      // offsets threaded through `chunkOf`.
      const chunkOf = async (index: number): Promise<Slot> => {
        const out = pool.take(dim);
        rec_copy(biased, index * dim * 4, out, 0, dim * 4);
        return out;
      };
      scaleMsa = await chunkOf(0);
      gateMsa = await activation(await chunkOf(1), dim, ACTIVATION.tanh);
      scaleMlp = await chunkOf(2);
      gateMlp = await activation(await chunkOf(3), dim, ACTIVATION.tanh);
      const one = weightSlot("#ones", () => new Float32Array(dim).fill(1));
      scaleMsa = await elementwise(scaleMsa, one, dim, ELEMENTWISE.add);
      scaleMlp = await elementwise(scaleMlp, one, dim, ELEMENTWISE.add);
    }

    const normed1 = await rmsnorm(x, w("attention_norm1.weight"), seq, dim);
    const scaled1 = await rowsOp(normed1, scaleMsa, seq, dim, ELEMENTWISE.multiply);

    const q0 = await project(w("attention.to_q.weight"), scaled1, seq, dim, width);
    const k0 = await project(w("attention.to_k.weight"), scaled1, seq, dim, width);
    const v0 = await project(w("attention.to_v.weight"), scaled1, seq, dim, width);

    const q1 = await rmsnorm(q0, w("attention.norm_q.weight"), seq * nHeads, headDim);
    const k1 = await rmsnorm(k0, w("attention.norm_k.weight"), seq * nHeads, headDim);
    const q2 = await ropeAxes(q1, seq, positions, axisDims, ropeSlots);
    const k2 = await ropeAxes(k1, seq, positions, axisDims, ropeSlots);

    // The live tokens are a prefix, so `[seq, H, D]`'s first `live` rows are
    // contiguous and `permute` can simply be told there are fewer of them.
    const live = validSeq;
    const attended = await attention(
      await permute(q2, live, nHeads, headDim),
      await permute(k2, live, nHeads, headDim),
      await permute(v0, live, nHeads, headDim),
      live,
      zeroBias,
    );
    const mergedLive = await permute(attended, nHeads, live, headDim);

    let merged = mergedLive;
    if (live < seq) {
      // Padded back with zeros. Copied from a buffer that is zero by
      // construction rather than cleared per call: `createStorageBuffer` gives
      // zeroed memory, and the tail is never written after that.
      merged = pool.take(seq * width);
      ops.push({
        kind: "copy", src: zeroTail, srcOffset: 0,
        dst: merged.buffer, dstOffset: 0, size: seq * width * 4,
      });
      ops.push({
        kind: "copy", src: mergedLive.buffer, srcOffset: 0,
        dst: merged.buffer, dstOffset: 0, size: live * width * 4,
      });
    }

    const projected = await project(w("attention.to_out.0.weight"), merged, seq, width, dim);
    const normed2 = await rmsnorm(projected, w("attention_norm2.weight"), seq, dim);
    const gated1 = await rowsOp(normed2, gateMsa, seq, dim, ELEMENTWISE.multiply);
    const h1 = await elementwise(x, gated1, seq * dim, ELEMENTWISE.add);

    const normed3 = await rmsnorm(h1, w("ffn_norm1.weight"), seq, dim);
    const scaled2 = await rowsOp(normed3, scaleMlp, seq, dim, ELEMENTWISE.multiply);
    const gate = await activation(
      await project(w("feed_forward.w1.weight"), scaled2, seq, dim, ffnHidden), seq * ffnHidden, ACTIVATION.silu,
    );
    const up = await project(w("feed_forward.w3.weight"), scaled2, seq, dim, ffnHidden);
    const ffn = await project(
      w("feed_forward.w2.weight"), await elementwise(gate, up, seq * ffnHidden, ELEMENTWISE.multiply), seq, ffnHidden, dim,
    );
    const normed4 = await rmsnorm(ffn, w("ffn_norm2.weight"), seq, dim);
    const gated2 = await rowsOp(normed4, gateMlp, seq, dim, ELEMENTWISE.multiply);
    return elementwise(h1, gated2, seq * dim, ELEMENTWISE.add);
  };

  /** A GPU-to-GPU copy recorded into the same encoder — `ResidentOp`'s `copy`. */
  function rec_copy(src: Slot, srcOffset: number, dst: Slot, dstOffset: number, size: number): void {
    ops.push({ kind: "copy", src: src.buffer, srcOffset, dst: dst.buffer, dstOffset, size });
  }


  // ============================================================
  // The forward. Structure follows `ditForwardGpu` exactly.
  // ============================================================

  await needWeights("t_embedder.", "timestep embedder");
  await needWeights(`all_x_embedder.${patchSize}-1.`, "patch embedder");
  const key = `${patchSize}-1`;
  const midDim = weights.shapeOf?.("t_embedder.mlp.0.bias")?.[0] ?? weights.get("t_embedder.mlp.0.bias").length;

  const tFreq = upload(timestepEmbedding(input.t * cfg.tScale, cfg.frequencyEmbeddingSize, cfg.maxPeriod));
  const tMid = await rowsOp(
    await project("t_embedder.mlp.0.weight", tFreq, 1, cfg.frequencyEmbeddingSize, midDim),
    weightSlot("t_embedder.mlp.0.bias"), 1, midDim, ELEMENTWISE.add,
  );
  const adalnInput = await rowsOp(
    await project("t_embedder.mlp.2.weight", await activation(tMid, midDim, ACTIVATION.silu), 1, midDim, cfg.adalnEmbedDim),
    weightSlot("t_embedder.mlp.2.bias"), 1, cfg.adalnEmbedDim, ELEMENTWISE.add,
  );

  let x = await rowsOp(
    await project(
      `all_x_embedder.${key}.weight`,
      upload(patchify(input.latent, inChannels, F, H, W, patchSize, 1)), xSeq, patchDim, dim,
    ),
    weightSlot(`all_x_embedder.${key}.bias`), xSeq, dim, ELEMENTWISE.add,
  );
  await flush([adalnInput, x], { name: "adalnInput", slot: adalnInput, length: cfg.adalnEmbedDim });

  const ffnHidden = weights.shapeOf?.("layers.0.feed_forward.w1.weight")?.[0] ??
    weights.get("layers.0.feed_forward.w1.weight").length / dim;

  /** Positions with the slack `ops/rope`'s own helper gives them. */
  const positionSlots = (ids: Int32Array): Slot => {
    const slack = new Int32Array(ids.length + cfg.ropeAxesDims.length * 4).fill(9999);
    slack.set(ids);
    return upload(slack);
  };

  const xPositions = positionSlots(imagePositionIds(F, hTokens, wTokens, capSeq));
  for (let i = 0; i < cfg.nRefinerLayers; i += 1) {
    await needWeights(`noise_refiner.${i}.`, `noise refiner ${i + 1}/${cfg.nRefinerLayers}`);
    x = await block(`noise_refiner.${i}.`, true, x, adalnInput, xPositions, xSeq, ffnHidden, xSeq);
    await flush(
      [adalnInput, x, xPositions],
      i === cfg.nRefinerLayers - 1 ? { name: "afterNoiseRefiner", slot: x, length: xSeq * dim } : undefined,
    );
  }

  await needWeights("cap_embedder.", "caption embedder");
  let capValid = 0;
  while (capValid < capSeq && input.capMask[capValid]) capValid += 1;
  for (let i = capValid; i < capSeq; i += 1) {
    if (input.capMask[i]) throw new Error(`ditForwardResident: capMask must be a prefix of real tokens (token ${i}).`);
  }

  // The caption's padded rows carry `cap_pad_token`. Built on the host and
  // uploaded whole, because it is one small tensor and the alternative is a
  // scatter dispatch for something that is not on any hot path.
  const capNormed = await rmsnorm(upload(input.capFeats), "cap_embedder.0.weight", capSeq, cfg.capFeatDim);
  let cap = await rowsOp(
    await project("cap_embedder.1.weight", capNormed, capSeq, cfg.capFeatDim, dim),
    weightSlot("cap_embedder.1.bias"), capSeq, dim, ELEMENTWISE.add,
  );
  await flush([adalnInput, x, xPositions, cap]);
  if (capValid < capSeq) {
    await needWeights("cap_pad_token", "caption pad token");
    const source = weightSlot(`#capPad:${capSeq - capValid}`, () => {
      const padded = new Float32Array((capSeq - capValid) * dim);
      const padToken = weights.get("cap_pad_token");
      for (let i = 0; i < capSeq - capValid; i += 1) padded.set(padToken, i * dim);
      return padded;
    });
    const padded = { byteLength: (capSeq - capValid) * dim * 4 };
    ops.push({
      kind: "copy", src: source.buffer, srcOffset: 0,
      dst: cap.buffer, dstOffset: capValid * dim * 4, size: padded.byteLength,
    });
  }

  const capPositions = positionSlots(captionPositionIds(capSeq));
  for (let i = 0; i < cfg.nRefinerLayers; i += 1) {
    await needWeights(`context_refiner.${i}.`, `context refiner ${i + 1}/${cfg.nRefinerLayers}`);
    cap = await block(`context_refiner.${i}.`, false, cap, null, capPositions, capSeq, ffnHidden, capValid);
    await flush(
      [adalnInput, x, xPositions, cap, capPositions],
      i === cfg.nRefinerLayers - 1 ? { name: "afterContextRefiner", slot: cap, length: capSeq * dim } : undefined,
    );
  }

  // --- unified stack ---
  const unifiedSeq = xSeq + capSeq;
  let unified = pool.take(unifiedSeq * dim);
  ops.push({ kind: "copy", src: x.buffer, srcOffset: 0, dst: unified.buffer, dstOffset: 0, size: xSeq * dim * 4 });
  ops.push({ kind: "copy", src: cap.buffer, srcOffset: 0, dst: unified.buffer, dstOffset: xSeq * dim * 4, size: capSeq * dim * 4 });

  const allPositions = new Int32Array(unifiedSeq * 3);
  allPositions.set(imagePositionIds(F, hTokens, wTokens, capSeq), 0);
  allPositions.set(captionPositionIds(capSeq), xSeq * 3);
  const positions = positionSlots(allPositions);
  await flush([adalnInput, unified, positions]);

  for (let i = 0; i < cfg.nLayers; i += 1) {
    await needWeights(`layers.${i}.`, `layer ${i + 1}/${cfg.nLayers}`);
    unified = await block(`layers.${i}.`, true, unified, adalnInput, positions, unifiedSeq, ffnHidden, xSeq + capValid);
    await flush(
      [adalnInput, unified, positions],
      i === 0
        ? { name: "afterLayer0", slot: unified, length: unifiedSeq * dim }
        : i === cfg.nLayers - 1
          ? { name: "afterLayers", slot: unified, length: unifiedSeq * dim }
          : undefined,
    );
  }

  // --- final layer ---
  await needWeights(`all_final_layer.${patchSize}-1.`, "final layer");
  const scale = await elementwise(
    await rowsOp(
      await project(
        `all_final_layer.${key}.adaLN_modulation.1.weight`,
        await activation(adalnInput, cfg.adalnEmbedDim, ACTIVATION.silu), 1, cfg.adalnEmbedDim, dim,
      ),
      weightSlot(`all_final_layer.${key}.adaLN_modulation.1.bias`), 1, dim, ELEMENTWISE.add,
    ),
    weightSlot("#ones", () => new Float32Array(dim).fill(1)), dim, ELEMENTWISE.add,
  );

  const normed = pool.take(unifiedSeq * dim);
  await record(
    K.layernorm,
    [
      unified.buffer,
      weightBuffer("#lnWeight", () => new Float32Array(dim).fill(1)),
      weightBuffer("#lnBias", () => new Float32Array(dim)),
      normed.buffer,
      uniform(params([["u32", unifiedSeq], ["u32", dim], ["f32", 1e-6]])),
    ],
    [unifiedSeq],
  );
  const scaled = await rowsOp(normed, scale, unifiedSeq, dim, ELEMENTWISE.multiply);
  const projected = await rowsOp(
    await project(`all_final_layer.${key}.linear.weight`, scaled, unifiedSeq, dim, patchDim),
    weightSlot(`all_final_layer.${key}.linear.bias`), unifiedSeq, patchDim, ELEMENTWISE.add,
  );

  // The only readback in the whole forward.
  const staging = device.createStorageBuffer(
    xSeq * patchDim * 4,
    GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  );
  const [out] = await device.batch(ops, [
    { staging, source: projected.buffer, sourceOffset: 0, length: xSeq * patchDim, type: "f32" },
  ]);
  ops = [];

  if (stats) {
    stats.dispatches = dispatches;
    stats.submits = device.stats.submits;
    stats.buffersCreated = device.stats.buffersCreated;
    stats.poolSlots = pool.created;
    stats.weightBuffers = weightBuffers.size;
    stats.uploadedBytes = uploaded;
    stats.readBackBytes = xSeq * patchDim * 4;
  }

  const latent = unpatchify(out as Float32Array, inChannels, F, H, W, patchSize, 1);
  pool.destroy();
  staging.destroy();
  for (const buffer of uniforms) buffer.destroy();
  // The weights are deliberately **not** destroyed when the caller is holding
  // them: that is what `held` is for. Without a caller they are this forward's
  // own and go now.
  if (!held) for (const buffer of weightBuffers.values()) buffer.destroy();
  return latent;
}

/** Frees weights held across forwards. The caller owns them; this is how they go. */
export function releaseDitWeights(held: Map<string, GPUBuffer>): void {
  for (const buffer of held.values()) buffer.destroy();
  held.clear();
}
