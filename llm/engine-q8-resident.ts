import { kernel, params, runnerFromResident, type ResidentDevice, type ResidentOp, type Runner } from "../harness/index.js";
import type { LlamaConfig } from "./config.js";
import { LlamaEngineQ8 } from "./engine-q8.js";
import { MAX_WORKGROUPS_PER_DISPATCH } from "./kernels.js";
import type { KVCache } from "./kv-cache.js";
import {
  assertWeightShapesQ8,
  cloneQuantizedLinear,
  gatherDequantRows,
  packInt8Rows,
  type LlamaWeightsQ8,
  type QuantizedLinear,
} from "./weights-q8.js";

/**
 * Issue #110: `LlamaEngineQ8`'s decode path, restructured so one generated
 * token costs one `queue.submit` and one readback (the final logits alone),
 * instead of the ~155 GPU↔CPU round trips `LlamaEngineQ8.forward` pays per
 * token today (one `Runner.run()` — its own buffer allocation, submit and
 * readback — per kernel dispatch, per layer).
 *
 * A **separate class**, not a mode flag: `LlamaEngineQ8Resident` holds
 * structurally different state (persistent `GPUBuffer`s, pipelines and bind
 * groups built once at construction) and a structurally different decode
 * loop (one flat list of GPU ops per token, no `await` between kernels)
 * from `LlamaEngineQ8`'s per-op `await run(...)` chain — the same reasoning
 * `LlamaEngineQ8`'s own class doc gives for not being a mode flag on
 * `LlamaEngine`.
 *
 * ## Scope: decode only, per issue #110
 *
 * Prefill is unchanged — "プリフィルは現行方式のまま(別ISSUE)". This class
 * does not reimplement it: the first `forward()` call (however many tokens
 * long) is delegated whole to a private `LlamaEngineQ8` instance, and this
 * class only takes over from the *second* `forward()` call onward, where
 * every call must be exactly one token (`LlamaEngineQ8`'s own decode
 * contract, `tokens.length === 1`). See `LlamaEngineQ8#cacheState` for the
 * handoff this reuses rather than re-deriving prefill's control flow.
 *
 * ## Buffer design: one buffer per tensor, not one fused buffer sliced by offset
 *
 * `LlamaEngineQ8`'s CPU-side engine (and its non-resident dispatch wrappers,
 * `llm/kernels.ts`) fuse Q/K/V and gate/up into one packed weight so one
 * `matvecQ8` dispatch computes all three/two at once (`llm/reshape.ts#concatRows`'s
 * doc: fewer *dispatches*, because each one used to cost a full
 * submit+readback round trip). That reasoning does not carry over here: this
 * class already pays one submit for an entire token's ~20-dispatch chain per
 * layer, so a fused QKV dispatch buys nothing a resident engine cares about,
 * and it costs something real — `harness/resident.ts#bindGroup`'s doc
 * records the measurement: splitting a fused output by a byte offset into
 * one buffer only works when that offset happens to be a multiple of
 * `minStorageBufferOffsetAlignment` (256 bytes here), which `kvDim`-sized
 * splits do **not** generally satisfy (the tiny fixture's own `kvDim = 32`
 * floats lands `v`'s slice at byte 384). So Q, K, V, gate and up projections
 * each get their own weight buffer, own scale buffer and own output buffer
 * — a few more `matvecQ8` dispatches per layer than `LlamaEngineQ8` issues,
 * all bandwidth-bound against the same total weight bytes, and correct
 * regardless of how the model's dimensions happen to divide.
 *
 * ## One device, not two
 *
 * `create()` takes a single `ResidentDevice` — no separate `Runner` for the
 * prefill delegate. `LlamaEngineQ8` is written against `Runner["run"]`, so
 * an earlier version of this class took one of those alongside the
 * `ResidentDevice`, and constructing both (two native `webgpu` `GPUDevice`s
 * in one process) reproducibly crashed this repository's Node/Dawn binding
 * partway through prefill — not on every run, and not always at the same
 * call, which is the signature of binding instability under load (the same
 * failure family issue #38/#49/#107 document) rather than a logic bug.
 * `harness/resident.ts#runnerFromResident` adapts the one `ResidentDevice`
 * into a `Runner["run"]` instead, so prefill and decode share a single
 * device — see that function's own doc for the measurement.
 *
 * ## KV-cache writes: a GPU-to-GPU copy, not `queue.writeBuffer`
 *
 * The new token's K (after RoPE) and V already live in a GPU buffer —
 * written by this same token's `rope`/`matvecQ8` dispatches, a few ops
 * earlier in the same batch. Routing that through `queue.writeBuffer` would
 * mean reading it back to the CPU first (defeating "logits-only readback",
 * issue #110's own phrase) just to hand the same bytes back to the GPU.
 * `encoder.copyBufferToBuffer`, recorded into the same batch as every
 * dispatch around it, moves them without leaving the device — see
 * `harness/resident.ts#ResidentOp`'s `"copy"` variant.
 *
 * The KV cache itself keeps `KVCache`'s own layout (`kv-cache.ts`'s doc):
 * `[kvHeads, maxSeqLen, headDim]` per layer, head-major, so a head's positions
 * are contiguous but heads are not — a new position's write is `kvHeads`
 * small copies (one per head), not one. `maxSeqLen` positions are always the
 * causal-mask's job to hide, never the buffer's: `ops/gqa`'s kernels take
 * `S` as both the loop bound *and* the per-head stride
 * (`ops/gqa/wgsl/scores.wgsl`: `let k_head = kv_head * params.S * params.D`),
 * so a cache addressed by `maxSeqLen` has to be *read* with `S = maxSeqLen`
 * too, not the position actually filled — this class always passes
 * `S = maxSeqLen` to `gqaScores`/`gqaContext` and relies on the causal mask
 * (`j <= i + query_offset`, unconditionally true for every step this class
 * runs) to skip the dot product for every unwritten position, which
 * `ops/gqa/wgsl/scores.wgsl`'s own `if` already does before ever reading `k`
 * there. The cost this accepts: every decode step's attention dispatches
 * scan `maxSeqLen` positions, not `position + 1` — bounded, and (per the
 * PR's roofline table) small next to the weight traffic `matvecQ8` moves,
 * but real, and not something #110's "no new kernels" scope leaves room to
 * avoid without a stride parameter `ops/gqa` does not have.
 */

const CODE = {
  rmsnorm: kernel(new URL("../ops/rmsnorm/index.ts", import.meta.url)),
  matvecQ8: kernel(new URL("../ops/matvec/index.ts", import.meta.url), "q8"),
  rope: kernel(new URL("../ops/rope/index.ts", import.meta.url)),
  gqaScores: kernel(new URL("../ops/gqa/index.ts", import.meta.url), "scores"),
  gqaContext: kernel(new URL("../ops/gqa/index.ts", import.meta.url), "context"),
  activation: kernel(new URL("../ops/activation/index.ts", import.meta.url)),
  elementwise: kernel(new URL("../ops/elementwise/index.ts", import.meta.url)),
};

/** `ops/activation/reference.ts#ACTIVATION.silu`, copied rather than imported to avoid pulling in `ops/activation`'s whole module graph for one constant — same value, checked against `llm/kernels.ts`'s own `ACTIVATION` re-export. */
const SILU = 1;
/** `ops/elementwise/reference.ts#ELEMENTWISE`. */
const ELEMENTWISE_ADD = 0;
const ELEMENTWISE_MULTIPLY = 1;

/** `runRope`'s uniform layout (`llm/kernels.ts`): field index 3 is `pos_offset` (u32). Copied, not re-derived — rule 2. */
const ROPE_POS_OFFSET_BYTE = 3 * 4;
/** `runGqa`'s scores uniform layout: field index 7 is `query_offset` (i32). */
const GQA_QUERY_OFFSET_BYTE = 7 * 4;

function uniformOf(device: ResidentDevice, fields: ["u32" | "i32" | "f32", number][]): GPUBuffer {
  const data = params(fields);
  const buffer = device.createUniformBuffer(data.byteLength);
  device.upload(buffer, 0, new Uint8Array(data));
  return buffer;
}

/** `[rowStart, rowStart + rowCount)` of a `QuantizedLinear`'s rows, as a view — no copy. */
function sliceLinear(linear: QuantizedLinear, rowStart: number, rowCount: number, K: number): QuantizedLinear {
  return {
    codes: linear.codes.subarray(rowStart * K, (rowStart + rowCount) * K),
    scale: linear.scale.subarray(rowStart, rowStart + rowCount),
  };
}

/** Packs, uploads and binds one `matvecQ8` projection: weight, scale, the shared vector/output buffers it reads and writes, and the uniform (`M`, `K`) it shares with every layer's same projection. */
async function buildProjection(
  device: ResidentDevice,
  pipeline: GPUComputePipeline,
  linear: QuantizedLinear,
  N: number,
  K: number,
  uniform: GPUBuffer,
  vectorBuf: GPUBuffer,
  outBuf: GPUBuffer,
): Promise<GPUBindGroup> {
  const packed = packInt8Rows(linear.codes, N, K);
  const weightBuf = device.createStorageBuffer(packed.byteLength);
  device.upload(weightBuf, 0, packed);
  const scaleBuf = device.createStorageBuffer(linear.scale.byteLength);
  device.upload(scaleBuf, 0, linear.scale);
  return device.bindGroup(pipeline, [weightBuf, scaleBuf, vectorBuf, outBuf, uniform]);
}

interface SharedResident {
  rmsnormPipeline: GPUComputePipeline;
  matvecPipeline: GPUComputePipeline;
  ropePipeline: GPUComputePipeline;
  gqaScoresPipeline: GPUComputePipeline;
  gqaContextPipeline: GPUComputePipeline;
  activationPipeline: GPUComputePipeline;
  elementwisePipeline: GPUComputePipeline;

  hiddenA: GPUBuffer;
  hiddenB: GPUBuffer;
  normedBuf: GPUBuffer;
  normed2Buf: GPUBuffer;
  qOutBuf: GPUBuffer;
  kOutBuf: GPUBuffer;
  vOutBuf: GPUBuffer;
  qRopedBuf: GPUBuffer;
  kRopedBuf: GPUBuffer;
  attnOutBuf: GPUBuffer;
  projOutBuf: GPUBuffer;
  gateOutBuf: GPUBuffer;
  upOutBuf: GPUBuffer;
  gateActBuf: GPUBuffer;
  gatedBuf: GPUBuffer;
  downOutBuf: GPUBuffer;
  finalNormedBuf: GPUBuffer;

  ropeQUniform: GPUBuffer;
  ropeKUniform: GPUBuffer;
  gqaScoresUniform: GPUBuffer;

  ropeQGroup: GPUBindGroup;
  ropeKGroup: GPUBindGroup;
  add1Group: GPUBindGroup;
  siluGroup: GPUBindGroup;
  mulGroup: GPUBindGroup;
  add2Group: GPUBindGroup;
  finalNormGroup: GPUBindGroup;
}

interface LayerResident {
  kCacheBuf: GPUBuffer;
  vCacheBuf: GPUBuffer;
  attnNormGroup: GPUBindGroup;
  ffnNormGroup: GPUBindGroup;
  wqGroup: GPUBindGroup;
  wkGroup: GPUBindGroup;
  wvGroup: GPUBindGroup;
  scoresGroup: GPUBindGroup;
  contextGroup: GPUBindGroup;
  woGroup: GPUBindGroup;
  gateGroup: GPUBindGroup;
  upGroup: GPUBindGroup;
  downGroup: GPUBindGroup;
}

interface LmHeadChunk {
  rowCount: number;
  group: GPUBindGroup;
  outBuf: GPUBuffer;
  staging: GPUBuffer;
}

export class LlamaEngineQ8Resident {
  private tokensSoFar = 0;
  private prefillWeights: LlamaWeightsQ8 | null;

  private constructor(
    private readonly config: LlamaConfig,
    private readonly device: ResidentDevice,
    private readonly legacyRun: Runner["run"],
    weights: LlamaWeightsQ8,
    private readonly embedTokens: QuantizedLinear,
    private readonly shared: SharedResident,
    private readonly layers: LayerResident[],
    private readonly lmHeadChunks: LmHeadChunk[],
  ) {
    // Retained only until the first `forward()` call runs prefill (see the
    // class doc): `LlamaEngineQ8`'s own constructor needs the full,
    // unpacked `LlamaWeightsQ8` to build *its* resident packed weights, and
    // this class cannot know the prompt's tokens (so cannot run prefill)
    // before that first call — dropped afterward (`runPrefill` below) so it
    // does not outlive the one delegation that needs it.
    this.prefillWeights = weights;
  }

  /** Positions already resident in the KV cache — 0 before the first `forward`. */
  get position(): number {
    return this.tokensSoFar;
  }

  static async create(config: LlamaConfig, weights: LlamaWeightsQ8, device: ResidentDevice): Promise<LlamaEngineQ8Resident> {
    assertWeightShapesQ8(config, weights);
    const { numLayers, hiddenSize, numHeads, numKvHeads, headDim, ffnHidden, vocabSize, maxSeqLen, ropeTheta, rmsNormEps } = config;
    const qDim = numHeads * headDim;
    const kvDim = numKvHeads * headDim;

    const [rmsnormPipeline, matvecPipeline, ropePipeline, gqaScoresPipeline, gqaContextPipeline, activationPipeline, elementwisePipeline] =
      await Promise.all([
        device.pipelineFor(CODE.rmsnorm),
        device.pipelineFor(CODE.matvecQ8),
        device.pipelineFor(CODE.rope),
        device.pipelineFor(CODE.gqaScores),
        device.pipelineFor(CODE.gqaContext),
        device.pipelineFor(CODE.activation),
        device.pipelineFor(CODE.elementwise),
      ]);

    // f32 activation buffers, sized for the N = 1 decode this class runs —
    // every one reused across all `numLayers` layers and every decode step
    // (see the class doc: no per-layer, no per-token allocation).
    const hiddenA = device.createStorageBuffer(hiddenSize * 4);
    const hiddenB = device.createStorageBuffer(hiddenSize * 4);
    const normedBuf = device.createStorageBuffer(hiddenSize * 4);
    const normed2Buf = device.createStorageBuffer(hiddenSize * 4);
    const qOutBuf = device.createStorageBuffer(qDim * 4);
    const kOutBuf = device.createStorageBuffer(kvDim * 4);
    const vOutBuf = device.createStorageBuffer(kvDim * 4);
    const qRopedBuf = device.createStorageBuffer(qDim * 4);
    const kRopedBuf = device.createStorageBuffer(kvDim * 4);
    const attnOutBuf = device.createStorageBuffer(qDim * 4);
    const projOutBuf = device.createStorageBuffer(hiddenSize * 4);
    const gateOutBuf = device.createStorageBuffer(ffnHidden * 4);
    const upOutBuf = device.createStorageBuffer(ffnHidden * 4);
    const gateActBuf = device.createStorageBuffer(ffnHidden * 4);
    const gatedBuf = device.createStorageBuffer(ffnHidden * 4);
    const downOutBuf = device.createStorageBuffer(hiddenSize * 4);
    const finalNormedBuf = device.createStorageBuffer(hiddenSize * 4);
    // `rope`'s uncached-table binding — always bound, never read
    // (`cache_positions = 0`), see `ops/rope/wgsl/kernel.wgsl`'s doc.
    // Relies on WebGPU's spec guarantee that a fresh `GPUBuffer` reads as
    // zero, so no upload is needed for a buffer nothing ever inspects.
    const dummyCacheBuf = device.createStorageBuffer(8);
    // The additive attention bias `ops/gqa` takes: always the all-zero "no
    // mask" case (this engine has no padding, batch size 1) — zero-init,
    // same reasoning as `dummyCacheBuf`.
    const maskBuf = device.createStorageBuffer(maxSeqLen * 4);
    const probsBuf = device.createStorageBuffer(numHeads * maxSeqLen * 4);

    const rmsUniform = uniformOf(device, [["u32", 1], ["u32", hiddenSize], ["f32", rmsNormEps], ["u32", 1]]);
    const qUniform = uniformOf(device, [["u32", qDim], ["u32", hiddenSize]]);
    const kUniform = uniformOf(device, [["u32", kvDim], ["u32", hiddenSize]]);
    const vUniform = uniformOf(device, [["u32", kvDim], ["u32", hiddenSize]]);
    const oUniform = uniformOf(device, [["u32", hiddenSize], ["u32", qDim]]);
    const gateUniform = uniformOf(device, [["u32", ffnHidden], ["u32", hiddenSize]]);
    const upUniform = uniformOf(device, [["u32", ffnHidden], ["u32", hiddenSize]]);
    const downUniform = uniformOf(device, [["u32", hiddenSize], ["u32", ffnHidden]]);
    const siluUniform = uniformOf(device, [["u32", ffnHidden], ["u32", SILU], ["f32", 1]]);
    const addUniform = uniformOf(device, [["u32", hiddenSize], ["u32", ELEMENTWISE_ADD]]);
    const mulUniform = uniformOf(device, [["u32", ffnHidden], ["u32", ELEMENTWISE_MULTIPLY]]);

    // `pos_offset` (index 3) is rewritten every decode step
    // (`ROPE_POS_OFFSET_BYTE`); every other field is architecture-wide and
    // never changes — see `llm/kernels.ts#runRope` for the field order this
    // copies (rule 2).
    const ropeQUniform = uniformOf(device, [
      ["u32", 1], ["u32", numHeads], ["u32", headDim], ["u32", 0], ["u32", 0],
      ["f32", ropeTheta], ["f32", 1], ["f32", 0], ["f32", 1], ["f32", 1],
      ["u32", 0], ["u32", numHeads],
    ]);
    const ropeKUniform = uniformOf(device, [
      ["u32", 1], ["u32", numKvHeads], ["u32", headDim], ["u32", 0], ["u32", 0],
      ["f32", ropeTheta], ["f32", 1], ["f32", 0], ["f32", 1], ["f32", 1],
      ["u32", 0], ["u32", numKvHeads],
    ]);
    // `S = maxSeqLen` always — see the class doc's KV-cache section for why.
    // `query_offset` (index 7) is rewritten every decode step.
    const gqaScoresUniform = uniformOf(device, [
      ["u32", numHeads], ["u32", numKvHeads], ["u32", 1], ["u32", maxSeqLen], ["u32", headDim],
      ["f32", 1 / Math.sqrt(headDim)], ["u32", 1], ["i32", 0], ["u32", 1], ["u32", 1], ["u32", 1],
    ]);
    const gqaContextUniform = uniformOf(device, [
      ["u32", numHeads], ["u32", numKvHeads], ["u32", 1], ["u32", maxSeqLen], ["u32", headDim],
    ]);

    const ropeQGroup = await device.bindGroup(ropePipeline, [qOutBuf, dummyCacheBuf, qRopedBuf, ropeQUniform]);
    const ropeKGroup = await device.bindGroup(ropePipeline, [kOutBuf, dummyCacheBuf, kRopedBuf, ropeKUniform]);
    const add1Group = await device.bindGroup(elementwisePipeline, [hiddenA, projOutBuf, hiddenB, addUniform]);
    const siluGroup = await device.bindGroup(activationPipeline, [gateOutBuf, gateActBuf, siluUniform]);
    const mulGroup = await device.bindGroup(elementwisePipeline, [gateActBuf, upOutBuf, gatedBuf, mulUniform]);
    const add2Group = await device.bindGroup(elementwisePipeline, [hiddenB, downOutBuf, hiddenA, addUniform]);

    const layers: LayerResident[] = [];
    for (const layer of weights.layers) {
      const attnNormBuf = device.createStorageBuffer(hiddenSize * 4);
      device.upload(attnNormBuf, 0, layer.attnNorm);
      const ffnNormBuf = device.createStorageBuffer(hiddenSize * 4);
      device.upload(ffnNormBuf, 0, layer.ffnNorm);

      const attnNormGroup = await device.bindGroup(rmsnormPipeline, [hiddenA, attnNormBuf, normedBuf, rmsUniform]);
      const ffnNormGroup = await device.bindGroup(rmsnormPipeline, [hiddenB, ffnNormBuf, normed2Buf, rmsUniform]);
      const wqGroup = await buildProjection(device, matvecPipeline, layer.wq, qDim, hiddenSize, qUniform, normedBuf, qOutBuf);
      const wkGroup = await buildProjection(device, matvecPipeline, layer.wk, kvDim, hiddenSize, kUniform, normedBuf, kOutBuf);
      const wvGroup = await buildProjection(device, matvecPipeline, layer.wv, kvDim, hiddenSize, vUniform, normedBuf, vOutBuf);
      const woGroup = await buildProjection(device, matvecPipeline, layer.wo, hiddenSize, qDim, oUniform, attnOutBuf, projOutBuf);
      const gateGroup = await buildProjection(device, matvecPipeline, layer.wGate, ffnHidden, hiddenSize, gateUniform, normed2Buf, gateOutBuf);
      const upGroup = await buildProjection(device, matvecPipeline, layer.wUp, ffnHidden, hiddenSize, upUniform, normed2Buf, upOutBuf);
      const downGroup = await buildProjection(device, matvecPipeline, layer.wDown, hiddenSize, ffnHidden, downUniform, gatedBuf, downOutBuf);

      const kCacheBuf = device.createStorageBuffer(numKvHeads * maxSeqLen * headDim * 4);
      const vCacheBuf = device.createStorageBuffer(numKvHeads * maxSeqLen * headDim * 4);
      const scoresGroup = await device.bindGroup(gqaScoresPipeline, [qRopedBuf, kCacheBuf, maskBuf, probsBuf, gqaScoresUniform]);
      const contextGroup = await device.bindGroup(gqaContextPipeline, [probsBuf, vCacheBuf, attnOutBuf, gqaContextUniform]);

      layers.push({
        kCacheBuf, vCacheBuf, attnNormGroup, ffnNormGroup, wqGroup, wkGroup, wvGroup, scoresGroup, contextGroup, woGroup, gateGroup, upGroup, downGroup,
      });
    }

    const finalNormBuf = device.createStorageBuffer(hiddenSize * 4);
    device.upload(finalNormBuf, 0, weights.finalNorm);
    const finalNormGroup = await device.bindGroup(rmsnormPipeline, [hiddenA, finalNormBuf, finalNormedBuf, rmsUniform]);

    // `lm_head`'s `M = vocabSize` can exceed `maxComputeWorkgroupsPerDimension`
    // (65,535 — issue #106's own bug, `llm/kernels.ts#MAX_WORKGROUPS_PER_DISPATCH`);
    // chunked into separate buffers rather than one buffer sliced by a row
    // offset for the same alignment reason every other split in this file
    // is (`buildProjection`'s callers above) — a chunk boundary at row
    // 65,535 lands at byte 262,140, not a multiple of 256 either.
    const lmHeadChunks: LmHeadChunk[] = [];
    for (let rowStart = 0; rowStart < vocabSize; rowStart += MAX_WORKGROUPS_PER_DISPATCH) {
      const rowCount = Math.min(MAX_WORKGROUPS_PER_DISPATCH, vocabSize - rowStart);
      const chunkUniform = uniformOf(device, [["u32", rowCount], ["u32", hiddenSize]]);
      const outBuf = device.createStorageBuffer(rowCount * 4);
      const staging = device.createStorageBuffer(rowCount * 4, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
      const group = await buildProjection(
        device, matvecPipeline, sliceLinear(weights.lmHead, rowStart, rowCount, hiddenSize), rowCount, hiddenSize, chunkUniform, finalNormedBuf, outBuf,
      );
      lmHeadChunks.push({ rowCount, group, outBuf, staging });
    }

    const shared: SharedResident = {
      rmsnormPipeline, matvecPipeline, ropePipeline, gqaScoresPipeline, gqaContextPipeline, activationPipeline, elementwisePipeline,
      hiddenA, hiddenB, normedBuf, normed2Buf, qOutBuf, kOutBuf, vOutBuf, qRopedBuf, kRopedBuf, attnOutBuf, projOutBuf,
      gateOutBuf, upOutBuf, gateActBuf, gatedBuf, downOutBuf, finalNormedBuf,
      ropeQUniform, ropeKUniform, gqaScoresUniform,
      ropeQGroup, ropeKGroup, add1Group, siluGroup, mulGroup, add2Group, finalNormGroup,
    };

    return new LlamaEngineQ8Resident(
      config, device, runnerFromResident(device), weights, cloneQuantizedLinear(weights.embedTokens), shared, layers, lmHeadChunks,
    );
  }

  async forward(tokens: number[]): Promise<Float32Array[]> {
    if (tokens.length === 0) throw new Error("LlamaEngineQ8Resident.forward: tokens must be non-empty");
    if (this.prefillWeights) return this.runPrefill(tokens);
    if (tokens.length !== 1) {
      throw new Error("LlamaEngineQ8Resident.forward: after prefill, every call must be exactly one token (decode)");
    }
    return this.runDecodeStep(tokens[0]!);
  }

  /** Delegates whole to `LlamaEngineQ8` — see the class doc's "Scope" section. */
  private async runPrefill(tokens: number[]): Promise<Float32Array[]> {
    const weights = this.prefillWeights!;
    const legacy = new LlamaEngineQ8(this.config, weights, this.legacyRun);
    const logits = await legacy.forward(tokens);
    const { cache, position } = legacy.cacheState;
    await this.uploadKvCache(cache, position);
    this.tokensSoFar = position;
    // Dropped now that both this class's own resident buffers (built in
    // `create()`) and `legacy` (built just above) have read what they
    // needed from it — see the constructor's doc.
    this.prefillWeights = null;
    return logits;
  }

  private async uploadKvCache(cache: KVCache, position: number): Promise<void> {
    if (position === 0) return;
    const { numLayers, numKvHeads, headDim, maxSeqLen } = this.config;
    for (let l = 0; l < numLayers; l += 1) {
      const { k, v } = cache.read(l, position);
      const layer = this.layers[l]!;
      for (let h = 0; h < numKvHeads; h += 1) {
        const src = h * position * headDim;
        const dstBytes = h * maxSeqLen * headDim * 4;
        this.device.upload(layer.kCacheBuf, dstBytes, k.subarray(src, src + position * headDim));
        this.device.upload(layer.vCacheBuf, dstBytes, v.subarray(src, src + position * headDim));
      }
    }
  }

  /**
   * One token, one `queue.submit`, logits-only readback — issue #110's core
   * ask. Every buffer, pipeline and bind group referenced here was built
   * once in `create()`; the only per-step allocation is the plain JS `ops`
   * array itself (references and a handful of changing numbers — `at`, the
   * KV-cache write position — not GPU resources), and the small CPU-side
   * embedding gather (`gatherDequantRows`, one row) `LlamaEngineQ8` already
   * does the same way.
   */
  private async runDecodeStep(tokenId: number): Promise<Float32Array[]> {
    const { numLayers, numKvHeads, headDim, hiddenSize, ffnHidden, numHeads, maxSeqLen, vocabSize } = this.config;
    const at = this.tokensSoFar;
    const s = this.shared;

    const embedVec = gatherDequantRows(this.embedTokens, [tokenId], hiddenSize);
    this.device.upload(s.hiddenA, 0, embedVec);
    this.device.upload(s.ropeQUniform, ROPE_POS_OFFSET_BYTE, new Uint32Array([at]));
    this.device.upload(s.ropeKUniform, ROPE_POS_OFFSET_BYTE, new Uint32Array([at]));
    this.device.upload(s.gqaScoresUniform, GQA_QUERY_OFFSET_BYTE, new Int32Array([at]));

    const ops: ResidentOp[] = [];
    const dispatch = (pipeline: GPUComputePipeline, bindGroup: GPUBindGroup, workgroups: [number] | [number, number] | [number, number, number]) =>
      ops.push({ kind: "dispatch", pipeline, bindGroup, workgroups });
    const copy = (src: GPUBuffer, srcOffset: number, dst: GPUBuffer, dstOffset: number, size: number) =>
      ops.push({ kind: "copy", src, srcOffset, dst, dstOffset, size });

    for (let l = 0; l < numLayers; l += 1) {
      const layer = this.layers[l]!;
      dispatch(s.rmsnormPipeline, layer.attnNormGroup, [1]);
      dispatch(s.matvecPipeline, layer.wqGroup, [numHeads * headDim]);
      dispatch(s.matvecPipeline, layer.wkGroup, [numKvHeads * headDim]);
      dispatch(s.matvecPipeline, layer.wvGroup, [numKvHeads * headDim]);
      dispatch(s.ropePipeline, s.ropeQGroup, [Math.ceil((numHeads * headDim) / 2 / 256)]);
      dispatch(s.ropePipeline, s.ropeKGroup, [Math.ceil((numKvHeads * headDim) / 2 / 256)]);
      for (let h = 0; h < numKvHeads; h += 1) {
        copy(s.kRopedBuf, h * headDim * 4, layer.kCacheBuf, (h * maxSeqLen + at) * headDim * 4, headDim * 4);
        copy(s.vOutBuf, h * headDim * 4, layer.vCacheBuf, (h * maxSeqLen + at) * headDim * 4, headDim * 4);
      }
      dispatch(s.gqaScoresPipeline, layer.scoresGroup, [1, numHeads, 1]);
      dispatch(s.gqaContextPipeline, layer.contextGroup, [1, numHeads, 1]);
      dispatch(s.matvecPipeline, layer.woGroup, [hiddenSize]);
      dispatch(s.elementwisePipeline, s.add1Group, [Math.ceil(hiddenSize / 256)]);
      dispatch(s.rmsnormPipeline, layer.ffnNormGroup, [1]);
      dispatch(s.matvecPipeline, layer.gateGroup, [ffnHidden]);
      dispatch(s.matvecPipeline, layer.upGroup, [ffnHidden]);
      dispatch(s.activationPipeline, s.siluGroup, [Math.ceil(ffnHidden / 256)]);
      dispatch(s.elementwisePipeline, s.mulGroup, [Math.ceil(ffnHidden / 256)]);
      dispatch(s.matvecPipeline, layer.downGroup, [hiddenSize]);
      dispatch(s.elementwisePipeline, s.add2Group, [Math.ceil(hiddenSize / 256)]);
    }
    dispatch(s.rmsnormPipeline, s.finalNormGroup, [1]);
    for (const chunk of this.lmHeadChunks) dispatch(s.matvecPipeline, chunk.group, [chunk.rowCount]);

    const readback = this.lmHeadChunks.map((chunk) => ({
      staging: chunk.staging, source: chunk.outBuf, sourceOffset: 0, length: chunk.rowCount, type: "f32" as const,
    }));
    const results = await this.device.batch(ops, readback);

    const logits = new Float32Array(vocabSize);
    let offset = 0;
    for (const r of results) {
      logits.set(r as Float32Array, offset);
      offset += r.length;
    }

    this.tokensSoFar += 1;
    return [logits];
  }
}
