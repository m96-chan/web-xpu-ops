import { kernel, params, type ResidentDevice, type ResidentOp } from "../harness/index.js";
import { ACTIVATION } from "../ops/activation/index.js";
import { ELEMENTWISE } from "../ops/elementwise/index.js";
import type { LlamaConfig } from "./config.js";
import { MAX_WORKGROUPS_PER_DISPATCH } from "./kernels.js";
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
 * ## Prefill is resident too, as of issue #117
 *
 * Issue #110 scoped prefill out — "プリフィルは現行方式のまま(別ISSUE)" — and an
 * earlier version of this class delegated the first `forward()` call whole to
 * a private `LlamaEngineQ8` instance, then uploaded its CPU-side `KVCache`
 * into this class's own GPU buffers. That paid `LlamaEngineQ8`'s per-op
 * `await run(...)` round trips for every dispatch in the prompt's pass
 * through every layer — ~360 of them for a 76-token prompt at
 * Sarashina2.2-1B's 24 layers, measured at 12.6s (issue #117's own numbers) —
 * for exactly the reason issue #110 already fixed for decode.
 *
 * `runPrefillResident` (below) is that fix applied to prefill: every prompt
 * token's pass through every layer is encoded into the *same* flat op list
 * `runDecodeStep` builds, and the whole prompt costs one `queue.submit`, one
 * readback — the **final** token's logits alone (`greedyGenerate`,
 * `examples/llm-demo`, and this class's own decode step only ever read
 * `prefillLogits[prefillLogits.length - 1]`, never an earlier prefill
 * position — see `forward`'s return-shape note). `LlamaEngineQ8` is no
 * longer used by this class at all.
 *
 * Prefill keeps `LlamaEngineQ8`'s **matmul** path rather than switching to
 * `matvecQ8` per token: `matvecQ8` re-reads a projection's whole weight once
 * per query vector, so running it once per prompt token would multiply the
 * weight traffic prefill moves by `N` — turning a 360-token prompt into 360x
 * the decode bandwidth, the opposite of the goal. `matmul` reads each
 * projection's weight once and computes against all `N` rows at once, same
 * as `LlamaEngineQ8`'s own prefill branch; only *how many round trips it
 * costs* changes here, not which kernel. Weight for `matmul`'s `b` operand is
 * dequantized-then-transposed from the packed int8 form once per layer per
 * `forward()` call that runs prefill — never per token, matching
 * `LlamaEngineQ8#project`'s own accepted cost (that class's doc: "この
 * transient dequantが1回のgenerationにつき1回で済む").
 *
 * ## Reshape: a GPU permute kernel, not a CPU round trip
 *
 * `matmul`'s projections produce `[N, heads, dim]` token-major (a token's
 * `heads * dim` output channels are one contiguous output row), and `rope`
 * reads and writes that same layout unchanged — but `ops/gqa` wants
 * `[heads, N, dim]` head-major (`llm/reshape.ts#splitHeadsMajor`'s doc has
 * the full explanation, shared with `LlamaEngineQ8`). That class reshapes on
 * the CPU (`llm/reshape.ts`'s own doc: "None of these are compute... so they
 * run on the CPU"), which is fine when every dispatch already pays a round
 * trip, and wrong here — reading Q/K/V back to reshape and re-uploading
 * would put a CPU round trip exactly where this class removes one.
 * `ops/permute/wgsl/kernel.wgsl` (issue #117, `ops/permute/reference.ts`)
 * is that reshape as one GPU dispatch.
 *
 * ## Prefill's own KV is scanned at S = N, not S = maxSeqLen
 *
 * Prefill's attention only ever needs the tokens in *this* prompt — nothing
 * before them exists yet (this class's own decode contract: `forward()`
 * accepts more than one token exactly once, as the first call). So
 * `runPrefillResident` binds `gqaScores`/`gqaContext` against **transient**
 * `[kvHeads, N, headDim]` buffers (this call's own freshly projected,
 * RoPE'd K and V), not the persistent `maxSeqLen`-strided cache — `S = N`
 * there, already the tight bound, and `sEff` (issue #117, see below) stays
 * at its default (`sEff = S`) for prefill's own attention. The persistent
 * cache is written *separately*, once attention has read from the tight
 * buffers: `kvHeads` `copyBufferToBuffer` calls per layer (one per head, each
 * moving that head's whole `N`-position block in one contiguous copy — the
 * same shape of copy `runDecodeStep` already does for one position, scaled
 * to `N`), so decode's own persistent, `maxSeqLen`-strided `kCacheBuf`/
 * `vCacheBuf` are ready for `runDecodeStep`'s first call the moment prefill's
 * `batch()` resolves.
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
 * ## One device, always
 *
 * `create()` takes a single `ResidentDevice`, still. A pre-#117 version of
 * this class needed a *second* device dressed up as a `Runner["run"]`
 * (`harness/resident.ts#runnerFromResident`) purely so `LlamaEngineQ8` had
 * something to call for its delegated prefill — two native `webgpu`
 * `GPUDevice`s in one process reproducibly crashed this repository's
 * Node/Dawn binding partway through prefill (issue #38/#49/#107's failure
 * family). Prefill going resident removed the delegate, and with it the
 * only reason this class ever touched a second device or `runnerFromResident`
 * — this class no longer imports it, and after #117 nothing in this
 * repository does (`harness/resident.test.ts` is now its only exerciser,
 * PR #119 review item 8). It stays in `harness/resident.ts` as a general
 * `ResidentDevice`-to-`Runner["run"]` adapter for whichever future caller
 * needs that shape next, with its own direct test rather than this class's
 * history as its only coverage.
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
 * small copies (one per head), not one. `ops/gqa`'s kernels take `S` as both
 * the loop bound *and* the per-head stride
 * (`ops/gqa/wgsl/scores.wgsl`: `let k_head = kv_head * params.S * params.D`),
 * so a cache addressed by `maxSeqLen` has to be *read* with `S = maxSeqLen`
 * as the stride — but not scanned that far: `ops/gqa`'s `sEff` parameter
 * (issue #117) separates the two, and this class passes `S = maxSeqLen`,
 * `sEff = position + 1` to `gqaScores`/`gqaContext` every step
 * (`GQA_SCORES_S_EFF_BYTE`/`GQA_CONTEXT_S_EFF_BYTE` below). Before `sEff`
 * existed, every decode step's attention dispatches scanned all `maxSeqLen`
 * positions regardless of how many were real, relying only on the causal
 * mask to skip the dot product past the real ones — issue #116's own
 * roofline table measured that at 672 MiB/token on Sarashina2.2-1B's shape,
 * the single largest unaccounted term in decode's bandwidth budget. `sEff`
 * removes the scan itself rather than skipping work inside it.
 */

const CODE = {
  rmsnorm: kernel(new URL("../ops/rmsnorm/index.ts", import.meta.url)),
  matvecQ8: kernel(new URL("../ops/matvec/index.ts", import.meta.url), "q8"),
  matmul: kernel(new URL("../ops/matmul/index.ts", import.meta.url)),
  rope: kernel(new URL("../ops/rope/index.ts", import.meta.url)),
  gqaScores: kernel(new URL("../ops/gqa/index.ts", import.meta.url), "scores"),
  gqaContext: kernel(new URL("../ops/gqa/index.ts", import.meta.url), "context"),
  activation: kernel(new URL("../ops/activation/index.ts", import.meta.url)),
  elementwise: kernel(new URL("../ops/elementwise/index.ts", import.meta.url)),
  // Issue #117's prefill reshape — see `ops/permute/wgsl/kernel.wgsl`'s own doc.
  permute: kernel(new URL("../ops/permute/index.ts", import.meta.url)),
  // Issue #117's prefill weight prep — see `ops/dequant_transpose/reference.ts`'s own doc.
  dequantTranspose: kernel(new URL("../ops/dequant_transpose/index.ts", import.meta.url)),
};

/** `runRope`'s uniform layout (`llm/kernels.ts`): field index 3 is `pos_offset` (u32). Copied, not re-derived — rule 2. */
const ROPE_POS_OFFSET_BYTE = 3 * 4;
/** `runGqa`'s scores uniform layout: field index 7 is `query_offset` (i32). */
const GQA_QUERY_OFFSET_BYTE = 7 * 4;
/**
 * `runGqa`'s scores uniform layout: field index 11 is `s_eff` (u32) — issue
 * #117, appended after `mask_rows` so `GQA_QUERY_OFFSET_BYTE` above did not
 * have to move.
 */
const GQA_SCORES_S_EFF_BYTE = 11 * 4;
/** `runGqa`'s context uniform layout: field index 5 is `s_eff` (u32), same reasoning. */
const GQA_CONTEXT_S_EFF_BYTE = 5 * 4;

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

/** Must match `TILE` in `ops/matmul/wgsl/kernel.wgsl` — copied per rule 2, same as `llm/kernels.ts#MATMUL_TILE`. */
const MATMUL_TILE = 16;

/**
 * The WebGPU spec's guaranteed-supported minimum for `maxStorageBufferBindingSize`
 * (128 MiB) — PR #119 review, item 2's conservative bound for `probsBuf`
 * (`runPrefillResident`'s own attention-matrix buffer, `numHeads * N * N * 4`
 * bytes: quadratic in `N`, unlike every other prefill buffer). `ResidentDevice`
 * (`harness/resident.ts`) does not expose the adapter's actual *negotiated*
 * limit — it requests up to 2 GiB but is capped by whatever the adapter
 * itself reports, and this class has no way to ask what that turned out to
 * be. Checking against the spec floor rather than the 2 GiB ceiling means
 * this can reject a prompt some GPUs could actually have handled, but never
 * the reverse — the same "minimal, correctness-first, precision left for a
 * follow-up" trade `MAX_WORKGROUPS_PER_DISPATCH` below already makes for a
 * ceiling this class *can* check exactly.
 */
const DEFAULT_MAX_STORAGE_BINDING_BYTES = 128 * 1024 * 1024;

/**
 * One `matmul` projection's *shape* — everything `runPrefillResident` can
 * set up once and reuse for every layer that has a projection this shape,
 * because none of it depends on which layer's weight is about to flow
 * through it: `weightTBuf` is where that layer's dequantized-and-transposed
 * weight lands (`ops/dequant_transpose`'s `[inFeatures, outFeatures]` f32
 * output, `matmul`'s `b` operand), and `matmulGroup` is the `matmul`
 * dispatch's own bind group, which only ever reads `weightTBuf` — never the
 * per-layer packed weight that fills it.
 *
 * PR #119 review, item 7: an earlier version created a fresh `weightTBuf`
 * (and its `matmul` bind group) inside the per-layer loop, once per layer
 * per projection — 24 layers x 7 projections' worth of `[inFeatures,
 * outFeatures]` f32 buffers alive simultaneously (none of them `destroy()`ed
 * until GC got around to it, which a native WebGPU binding is not
 * guaranteed to treat as "freed" promptly), measured at Sarashina2.2-1B's
 * shape to peak VRAM around 5-7 GiB for one prefill call. One `weightTBuf`
 * per *shape* instead — `setupMatmulProjectionShape` below, called once
 * per projection before the layer loop starts — needs only as many
 * buffers as there are distinct `(outFeatures, inFeatures)` pairs among
 * `wq`/`wk`/`wv`/`wo`/`wGate`/`wUp`/`wDown` (five at Sarashina2.2-1B's
 * shape, `wk`/`wv` and `wGate`/`wUp` sharing one each): ~122 MiB total,
 * independent of `numLayers`. Reusing the same buffer across layers is
 * safe because `runPrefillResident`'s own per-layer loop is a plain `for`
 * with an `await` inside it — layer `l + 1`'s dequant dispatch is never
 * even *built*, let alone pushed into `ops`, until layer `l`'s own
 * `Promise.all` (which includes that layer's matmul dispatch push) has
 * resolved, so a shape's `weightTBuf` is always fully read by the layer
 * that just wrote it before the next layer's dequant overwrites it — the
 * GPU executes `ops` in exactly that push order, same guarantee the
 * KV-cache copies already rely on.
 */
interface MatmulProjectionShape {
  weightTBuf: GPUBuffer;
  matmulGroup: GPUBindGroup;
  outFeatures: number;
  inFeatures: number;
}

async function setupMatmulProjectionShape(
  device: ResidentDevice,
  matmulPipeline: GPUComputePipeline,
  outFeatures: number,
  inFeatures: number,
  matmulUniform: GPUBuffer,
  aBuf: GPUBuffer,
  outBuf: GPUBuffer,
): Promise<MatmulProjectionShape> {
  const weightTBuf = device.createStorageBuffer(inFeatures * outFeatures * 4);
  const matmulGroup = await device.bindGroup(matmulPipeline, [aBuf, weightTBuf, outBuf, matmulUniform]);
  return { weightTBuf, matmulGroup, outFeatures, inFeatures };
}

/**
 * One layer's weight, dequantized-and-transposed into `shape.weightTBuf` —
 * `setupMatmulProjectionShape`'s per-layer counterpart. Packs `linear`'s raw
 * int8 codes (a plain per-row byte copy, no arithmetic —
 * `ops/dequant_transpose/reference.ts`'s doc measured it at ~170ms total
 * across Sarashina2.2-1B's 24 layers), uploads the packed weight and scale,
 * records an `ops/dequant_transpose` dispatch — on the GPU, not the three
 * CPU passes (`packInt8Rows` then `dequantizePackedQ8` then
 * `transposeRowMajor`) `LlamaEngineQ8#project`'s own prefill branch takes
 * (that doc's own measurement: ~100ms/layer, ~2.5s for 24 layers, on the
 * critical path of every prefill call).
 *
 * `weightBuf`/`scaleBuf`/`dequantUniform` are this call's own transient
 * buffers — unlike `shape.weightTBuf`, layer-specific and used exactly
 * once, so the caller collects them (this function's return value) to
 * `destroy()` once `batch()` has resolved (PR #119 review, item 7's other
 * half): letting 24 layers' worth of packed weights and scales (small
 * relative to `weightTBuf`, but not free — ~1.2 GiB together at
 * Sarashina2.2-1B's shape) sit until GC is the same unmeasured-lifetime
 * risk `weightTBuf` reuse above exists to remove from the buffer that
 * mattered more.
 */
async function dequantIntoShape(
  device: ResidentDevice,
  dequantPipeline: GPUComputePipeline,
  shape: MatmulProjectionShape,
  linear: QuantizedLinear,
  pushDequant: (bindGroup: GPUBindGroup, workgroups: [number]) => void,
): Promise<GPUBuffer[]> {
  const packed = packInt8Rows(linear.codes, shape.outFeatures, shape.inFeatures);
  const weightBuf = device.createStorageBuffer(packed.byteLength);
  device.upload(weightBuf, 0, packed);
  const scaleBuf = device.createStorageBuffer(linear.scale.byteLength);
  device.upload(scaleBuf, 0, linear.scale);
  const dequantUniform = uniformOf(device, [["u32", shape.outFeatures], ["u32", shape.inFeatures]]);
  const dequantGroup = await device.bindGroup(dequantPipeline, [weightBuf, scaleBuf, shape.weightTBuf, dequantUniform]);
  pushDequant(dequantGroup, [Math.ceil((shape.outFeatures * shape.inFeatures) / 256)]);
  return [weightBuf, scaleBuf, dequantUniform];
}

interface SharedResident {
  rmsnormPipeline: GPUComputePipeline;
  matvecPipeline: GPUComputePipeline;
  ropePipeline: GPUComputePipeline;
  gqaScoresPipeline: GPUComputePipeline;
  gqaContextPipeline: GPUComputePipeline;
  activationPipeline: GPUComputePipeline;
  elementwisePipeline: GPUComputePipeline;
  /** Prefill only — see `runPrefillResident`. */
  matmulPipeline: GPUComputePipeline;
  /** Prefill only — see `runPrefillResident`. */
  permutePipeline: GPUComputePipeline;
  /** Prefill only — see `runPrefillResident`/`buildMatmulProjection`. */
  dequantTransposePipeline: GPUComputePipeline;

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
  gqaContextUniform: GPUBuffer;

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
    weights: LlamaWeightsQ8,
    private readonly embedTokens: QuantizedLinear,
    private readonly shared: SharedResident,
    private readonly layers: LayerResident[],
    private readonly lmHeadChunks: LmHeadChunk[],
  ) {
    // Retained only until the first `forward()` call runs prefill (see the
    // class doc): this class cannot know the prompt's tokens (so cannot
    // dequantize-and-transpose each layer's projection weight for `matmul`,
    // `runPrefillResident`'s own first step) before that first call —
    // dropped afterward so it does not outlive the one call that needs it
    // (the same reasoning `LlamaEngineQ8`'s own class doc gives for not
    // holding onto the unpacked `LlamaWeightsQ8` past construction).
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

    const [
      rmsnormPipeline, matvecPipeline, ropePipeline, gqaScoresPipeline, gqaContextPipeline, activationPipeline, elementwisePipeline,
      matmulPipeline, permutePipeline, dequantTransposePipeline,
    ] = await Promise.all([
        device.pipelineFor(CODE.rmsnorm),
        device.pipelineFor(CODE.matvecQ8),
        device.pipelineFor(CODE.rope),
        device.pipelineFor(CODE.gqaScores),
        device.pipelineFor(CODE.gqaContext),
        device.pipelineFor(CODE.activation),
        device.pipelineFor(CODE.elementwise),
        // Prefill only (issue #117's resident prefill, `runPrefillResident`
        // below) — pipeline creation does not depend on `N`, only the
        // dispatch workgroup counts and buffer sizes do, so these are built
        // once here alongside decode's, not per `forward()` call.
        device.pipelineFor(CODE.matmul),
        device.pipelineFor(CODE.permute),
        device.pipelineFor(CODE.dequantTranspose),
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
    const siluUniform = uniformOf(device, [["u32", ffnHidden], ["u32", ACTIVATION.silu], ["f32", 1]]);
    const addUniform = uniformOf(device, [["u32", hiddenSize], ["u32", ELEMENTWISE.add]]);
    const mulUniform = uniformOf(device, [["u32", ffnHidden], ["u32", ELEMENTWISE.multiply]]);

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
    // `query_offset` (index 7) and `s_eff` (index 11 — issue #117) are both
    // rewritten every decode step, to `at` and `at + 1` respectively: the
    // cache holds exactly `at + 1` valid positions (0..at) once this step's
    // own K/V have been written, so `s_eff = at + 1` is the tight bound —
    // not `maxSeqLen`, which is what made every decode step scan the whole
    // cache regardless of how much of it was real (issue #116's roofline
    // table: 672 MiB/token of that scan on Sarashina2.2-1B's shape).
    const gqaScoresUniform = uniformOf(device, [
      ["u32", numHeads], ["u32", numKvHeads], ["u32", 1], ["u32", maxSeqLen], ["u32", headDim],
      ["f32", 1 / Math.sqrt(headDim)], ["u32", 1], ["i32", 0], ["u32", 1], ["u32", 1], ["u32", 1], ["u32", maxSeqLen],
    ]);
    const gqaContextUniform = uniformOf(device, [
      ["u32", numHeads], ["u32", numKvHeads], ["u32", 1], ["u32", maxSeqLen], ["u32", headDim], ["u32", maxSeqLen],
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
      matmulPipeline, permutePipeline, dequantTransposePipeline,
      hiddenA, hiddenB, normedBuf, normed2Buf, qOutBuf, kOutBuf, vOutBuf, qRopedBuf, kRopedBuf, attnOutBuf, projOutBuf,
      gateOutBuf, upOutBuf, gateActBuf, gatedBuf, downOutBuf, finalNormedBuf,
      ropeQUniform, ropeKUniform, gqaScoresUniform, gqaContextUniform,
      ropeQGroup, ropeKGroup, add1Group, siluGroup, mulGroup, add2Group, finalNormGroup,
    };

    return new LlamaEngineQ8Resident(
      config, device, weights, cloneQuantizedLinear(weights.embedTokens), shared, layers, lmHeadChunks,
    );
  }

  /**
   * Prefill's return shape is `[finalPositionLogits]` — **one** element,
   * regardless of how many tokens the prompt had, unlike `LlamaEngineQ8.forward`'s
   * one-per-token array. Issue #117's own "readbackは最終位置logitsのみ": every
   * real caller (`llm/engine.ts#greedyGenerate`, `examples/llm-demo/src/main.ts`)
   * only ever reads `prefillLogits[prefillLogits.length - 1]`, so this
   * remains a drop-in replacement for both — `[x][0]` and `[x][x.length - 1]`
   * name the same element when `x.length === 1`. A caller that wanted every
   * prefill position's logits (nothing in this repository does) would need a
   * different method; that is a real capability this class does not have
   * after issue #117, not an oversight — see the class doc's "Prefill is
   * resident too" section for why recovering it would cost back the
   * round-trip issue #117 removes.
   */
  async forward(tokens: number[]): Promise<Float32Array[]> {
    if (tokens.length === 0) throw new Error("LlamaEngineQ8Resident.forward: tokens must be non-empty");
    if (this.prefillWeights) return this.runPrefillResident(tokens);
    if (tokens.length !== 1) {
      throw new Error("LlamaEngineQ8Resident.forward: after prefill, every call must be exactly one token (decode)");
    }
    return this.runDecodeStep(tokens[0]!);
  }

  /**
   * All `N` prompt tokens, every layer, one `queue.submit`, one readback —
   * the final position's logits alone. See the class doc's "Prefill is
   * resident too" / "Reshape" / "Prefill's own KV" sections for the design;
   * this method is their assembly.
   *
   * Every buffer here is transient — allocated once at the top of this call
   * (sized by `N`, which `create()` cannot know) and left for GC once this
   * `await` resolves. That is unlike `runDecodeStep`'s buffers (built once in
   * `create()`, reused every step): prefill runs exactly once per generation
   * (this class's own decode contract, enforced in `forward`), so there is
   * no repeated call to amortize allocation across — the "resident" property
   * that matters here is *within* this one call (one submit for the whole
   * prompt), not across calls.
   */
  private async runPrefillResident(tokens: number[], debugAllPositions = false): Promise<Float32Array[]> {
    const weights = this.prefillWeights!;
    const { numLayers, hiddenSize, numHeads, numKvHeads, headDim, ffnHidden, maxSeqLen, vocabSize, ropeTheta, rmsNormEps } = this.config;
    const N = tokens.length;
    if (N > maxSeqLen) throw new Error(`LlamaEngineQ8Resident.forward: prompt length ${N} exceeds maxSeqLen=${maxSeqLen}`);
    const qDim = numHeads * headDim;
    const kvDim = numKvHeads * headDim;
    // PR #119 review, items 1-2: two WebGPU ceilings this method's own
    // `N`-scaled dispatches/buffers can reach *before* `maxSeqLen` does, on
    // a real config — both previously unguarded, both the same failure
    // shape as issue #106's own bug (`llm/kernels.ts#MAX_WORKGROUPS_PER_DISPATCH`'s
    // doc): a validation error outside any error scope invalidates the
    // whole batch silently, so the visible symptom is not a thrown error
    // but `forward()` resolving with zero/stale logits. Checked here,
    // before any `N`-scaled buffer exists, rather than left to surface that
    // way. `llm/engine-q8-resident.limits.wgsl.test.ts` is the regression
    // coverage for both (its own module doc has the exact thresholds).
    const widestNScaledDispatch = N * Math.max(qDim, kvDim, hiddenSize, ffnHidden);
    if (Math.ceil(widestNScaledDispatch / 256) > MAX_WORKGROUPS_PER_DISPATCH) {
      throw new Error(
        `LlamaEngineQ8Resident.forward: prompt length ${N} needs ${Math.ceil(widestNScaledDispatch / 256)} workgroups for one dispatch, past WebGPU's maxComputeWorkgroupsPerDimension=${MAX_WORKGROUPS_PER_DISPATCH} — this prefill path does not chunk that dispatch yet (tracked separately, see the PR)`,
      );
    }
    const probsBufBytes = numHeads * N * N * 4;
    if (probsBufBytes > DEFAULT_MAX_STORAGE_BINDING_BYTES) {
      throw new Error(
        `LlamaEngineQ8Resident.forward: prompt length ${N} needs a ${probsBufBytes}-byte attention (probsBuf) storage buffer, past the WebGPU spec's guaranteed maxStorageBufferBindingSize=${DEFAULT_MAX_STORAGE_BINDING_BYTES} — some adapters negotiate more (ResidentDevice does not expose which), so this is a conservative rejection, not a hard ceiling on capable hardware`,
      );
    }
    const device = this.device;
    const s = this.shared;

    // ---- N-sized scratch, reused across every layer of this call (not across calls — see this method's doc). ----
    const hiddenA = device.createStorageBuffer(N * hiddenSize * 4);
    const hiddenB = device.createStorageBuffer(N * hiddenSize * 4);
    const normedBuf = device.createStorageBuffer(N * hiddenSize * 4);
    const normed2Buf = device.createStorageBuffer(N * hiddenSize * 4);
    const qOutBuf = device.createStorageBuffer(N * qDim * 4);
    const kOutBuf = device.createStorageBuffer(N * kvDim * 4);
    const vOutBuf = device.createStorageBuffer(N * kvDim * 4);
    const qRopedBuf = device.createStorageBuffer(N * qDim * 4);
    const kRopedBuf = device.createStorageBuffer(N * kvDim * 4);
    const qHeadMajorBuf = device.createStorageBuffer(N * qDim * 4);
    const kHeadMajorBuf = device.createStorageBuffer(N * kvDim * 4);
    const vHeadMajorBuf = device.createStorageBuffer(N * kvDim * 4);
    // Attention matrix at prefill's own tight stride S = N (not maxSeqLen —
    // see the class doc's "Prefill's own KV" section).
    const probsBuf = device.createStorageBuffer(numHeads * N * N * 4);
    const attnHeadMajorBuf = device.createStorageBuffer(N * qDim * 4);
    const attnTokenMajorBuf = device.createStorageBuffer(N * qDim * 4);
    const projOutBuf = device.createStorageBuffer(N * hiddenSize * 4);
    const gateOutBuf = device.createStorageBuffer(N * ffnHidden * 4);
    const upOutBuf = device.createStorageBuffer(N * ffnHidden * 4);
    const gateActBuf = device.createStorageBuffer(N * ffnHidden * 4);
    const gatedBuf = device.createStorageBuffer(N * ffnHidden * 4);
    const downOutBuf = device.createStorageBuffer(N * hiddenSize * 4);
    const finalNormedAllBuf = device.createStorageBuffer(N * hiddenSize * 4);
    // Same reasoning as `create()`'s `dummyCacheBuf`/`maskBuf`: unread or
    // all-zero, so a fresh (zero-initialized) buffer needs no upload.
    const dummyCacheBuf = device.createStorageBuffer(8);
    const maskBuf = device.createStorageBuffer(N * 4);

    const embed = gatherDequantRows(this.embedTokens, tokens, hiddenSize);
    device.upload(hiddenA, 0, embed);

    // ---- Uniforms constant across every layer of this call. ----
    const rmsUniform = uniformOf(device, [["u32", N], ["u32", hiddenSize], ["f32", rmsNormEps], ["u32", 1]]);
    const qMmUniform = uniformOf(device, [["u32", N], ["u32", qDim], ["u32", hiddenSize]]);
    const kMmUniform = uniformOf(device, [["u32", N], ["u32", kvDim], ["u32", hiddenSize]]);
    const vMmUniform = uniformOf(device, [["u32", N], ["u32", kvDim], ["u32", hiddenSize]]);
    const oMmUniform = uniformOf(device, [["u32", N], ["u32", hiddenSize], ["u32", qDim]]);
    const gateMmUniform = uniformOf(device, [["u32", N], ["u32", ffnHidden], ["u32", hiddenSize]]);
    const upMmUniform = uniformOf(device, [["u32", N], ["u32", ffnHidden], ["u32", hiddenSize]]);
    const downMmUniform = uniformOf(device, [["u32", N], ["u32", hiddenSize], ["u32", ffnHidden]]);
    const siluUniform = uniformOf(device, [["u32", N * ffnHidden], ["u32", ACTIVATION.silu], ["f32", 1]]);
    const addUniform = uniformOf(device, [["u32", N * hiddenSize], ["u32", ELEMENTWISE.add]]);
    const mulUniform = uniformOf(device, [["u32", N * ffnHidden], ["u32", ELEMENTWISE.multiply]]);
    // `pos_offset = 0` always: prefill is only ever the first `forward()`
    // call (this class's own contract), so every prompt token's absolute
    // position is its index in `tokens`.
    const ropeQUniform = uniformOf(device, [
      ["u32", N], ["u32", numHeads], ["u32", headDim], ["u32", 0], ["u32", 0],
      ["f32", ropeTheta], ["f32", 1], ["f32", 0], ["f32", 1], ["f32", 1],
      ["u32", 0], ["u32", numHeads],
    ]);
    const ropeKUniform = uniformOf(device, [
      ["u32", N], ["u32", numKvHeads], ["u32", headDim], ["u32", 0], ["u32", 0],
      ["f32", ropeTheta], ["f32", 1], ["f32", 0], ["f32", 1], ["f32", 1],
      ["u32", 0], ["u32", numKvHeads],
    ]);
    // S = N here (prefill's own tight buffers, not the maxSeqLen-strided
    // cache), so sEff's default (sEff = S) is already the tight bound — no
    // truncation to ask for. See the class doc's "Prefill's own KV" section.
    const gqaScoresUniform = uniformOf(device, [
      ["u32", numHeads], ["u32", numKvHeads], ["u32", N], ["u32", N], ["u32", headDim],
      ["f32", 1 / Math.sqrt(headDim)], ["u32", 1], ["i32", 0], ["u32", 1], ["u32", 1], ["u32", 1], ["u32", N],
    ]);
    const gqaContextUniform = uniformOf(device, [
      ["u32", numHeads], ["u32", numKvHeads], ["u32", N], ["u32", N], ["u32", headDim], ["u32", N],
    ]);
    // Split (token-major -> head-major): [N, heads, D] -> [heads, N, D].
    const splitQUniform = uniformOf(device, [["u32", N], ["u32", numHeads], ["u32", headDim]]);
    const splitKvUniform = uniformOf(device, [["u32", N], ["u32", numKvHeads], ["u32", headDim]]);
    // Merge (head-major -> token-major): [heads, N, D] -> [N, heads, D].
    const mergeUniform = uniformOf(device, [["u32", numHeads], ["u32", N], ["u32", headDim]]);

    // ---- Bind groups constant across every layer of this call. ----
    const ropeQGroup = await device.bindGroup(s.ropePipeline, [qOutBuf, dummyCacheBuf, qRopedBuf, ropeQUniform]);
    const ropeKGroup = await device.bindGroup(s.ropePipeline, [kOutBuf, dummyCacheBuf, kRopedBuf, ropeKUniform]);
    const splitQGroup = await device.bindGroup(s.permutePipeline, [qRopedBuf, qHeadMajorBuf, splitQUniform]);
    const splitKGroup = await device.bindGroup(s.permutePipeline, [kRopedBuf, kHeadMajorBuf, splitKvUniform]);
    const splitVGroup = await device.bindGroup(s.permutePipeline, [vOutBuf, vHeadMajorBuf, splitKvUniform]);
    const gqaScoresGroup = await device.bindGroup(s.gqaScoresPipeline, [qHeadMajorBuf, kHeadMajorBuf, maskBuf, probsBuf, gqaScoresUniform]);
    const gqaContextGroup = await device.bindGroup(s.gqaContextPipeline, [probsBuf, vHeadMajorBuf, attnHeadMajorBuf, gqaContextUniform]);
    const mergeGroup = await device.bindGroup(s.permutePipeline, [attnHeadMajorBuf, attnTokenMajorBuf, mergeUniform]);
    const add1Group = await device.bindGroup(s.elementwisePipeline, [hiddenA, projOutBuf, hiddenB, addUniform]);
    const siluGroup = await device.bindGroup(s.activationPipeline, [gateOutBuf, gateActBuf, siluUniform]);
    const mulGroup = await device.bindGroup(s.elementwisePipeline, [gateActBuf, upOutBuf, gatedBuf, mulUniform]);
    const add2Group = await device.bindGroup(s.elementwisePipeline, [hiddenB, downOutBuf, hiddenA, addUniform]);

    const ops: ResidentOp[] = [];
    const dispatch = (pipeline: GPUComputePipeline, bindGroup: GPUBindGroup, workgroups: [number] | [number, number] | [number, number, number]) =>
      ops.push({ kind: "dispatch", pipeline, bindGroup, workgroups });
    const copy = (src: GPUBuffer, srcOffset: number, dst: GPUBuffer, dstOffset: number, size: number) =>
      ops.push({ kind: "copy", src, srcOffset, dst, dstOffset, size });
    const wg256 = (elements: number) => Math.ceil(elements / 256);
    const matmulWg = (outFeatures: number): [number, number] => [Math.ceil(outFeatures / MATMUL_TILE), Math.ceil(N / MATMUL_TILE)];

    // One `weightTBuf`/`matmulGroup` per projection *shape*, reused by every
    // layer — see `MatmulProjectionShape`'s own doc (PR #119 review, item 7).
    const [wqShape, wkShape, wvShape, woShape, gateShape, upShape, downShape] = await Promise.all([
      setupMatmulProjectionShape(device, s.matmulPipeline, qDim, hiddenSize, qMmUniform, normedBuf, qOutBuf),
      setupMatmulProjectionShape(device, s.matmulPipeline, kvDim, hiddenSize, kMmUniform, normedBuf, kOutBuf),
      setupMatmulProjectionShape(device, s.matmulPipeline, kvDim, hiddenSize, vMmUniform, normedBuf, vOutBuf),
      setupMatmulProjectionShape(device, s.matmulPipeline, hiddenSize, qDim, oMmUniform, attnTokenMajorBuf, projOutBuf),
      setupMatmulProjectionShape(device, s.matmulPipeline, ffnHidden, hiddenSize, gateMmUniform, normed2Buf, gateOutBuf),
      setupMatmulProjectionShape(device, s.matmulPipeline, ffnHidden, hiddenSize, upMmUniform, normed2Buf, upOutBuf),
      setupMatmulProjectionShape(device, s.matmulPipeline, hiddenSize, ffnHidden, downMmUniform, gatedBuf, downOutBuf),
    ]);
    // Every layer's own transient buffers (packed weight/scale/uniform per
    // projection, plus the small norm-gain buffers) — `destroy()`ed once
    // `batch()` below has resolved, not left for GC (PR #119 review, item 7).
    const transientBuffers: GPUBuffer[] = [];

    for (let l = 0; l < numLayers; l += 1) {
      const lw = weights.layers[l]!;
      const layer = this.layers[l]!;

      // Norm-gain buffers are per-layer and small (hiddenSize floats) —
      // re-created here rather than reusing `layer.attnNormGroup`'s own
      // (decode-only) buffer, which is bound at N = 1 shapes this call
      // cannot reuse.
      const attnNormBuf = device.createStorageBuffer(hiddenSize * 4);
      device.upload(attnNormBuf, 0, lw.attnNorm);
      const ffnNormBuf = device.createStorageBuffer(hiddenSize * 4);
      device.upload(ffnNormBuf, 0, lw.ffnNorm);
      transientBuffers.push(attnNormBuf, ffnNormBuf);
      // All nine bind groups requested together, not one `await` at a time:
      // `ResidentDevice.bindGroup` validates through `pushErrorScope`/
      // `await popErrorScope()` (`harness/resident.ts`'s own doc explains
      // why — catching an invalid binding here rather than at the eventual
      // `batch()`), and that round trip measured real per-call latency at
      // Sarashina2.2-1B's scale (~24 layers x 9 groups = 216 sequential
      // awaits otherwise). `Promise.all` still issues them in order and
      // still balances every push/pop correctly: each of the nine calls
      // below is synchronous (dequant, upload, `pushErrorScope`,
      // `createBindGroup`, the `popErrorScope()` *call*) right up to its own
      // `await`, so by the time control reaches the next array element the
      // previous one has already pushed and popped its own scope — nesting
      // never overlaps, only the *wait* for each validation result does.
      // Each `dequantIntoShape` call below also records its own
      // `ops/dequant_transpose` dispatch via `pushDequant` — synchronously,
      // before that call's own promise resolves, so every dequant dispatch
      // this layer needs is in `ops` before this `await` returns and the
      // `dispatch(matmul, ...)` calls just below run (see that function's
      // own doc for why the ordering is safe without awaiting each call
      // individually).
      const pushDequant = (pipeline: GPUComputePipeline, bindGroup: GPUBindGroup, workgroups: [number]) =>
        dispatch(pipeline, bindGroup, workgroups);
      const [attnNormGroup, ffnNormGroup, wqTransient, wkTransient, wvTransient, woTransient, gateTransient, upTransient, downTransient] = await Promise.all([
        device.bindGroup(s.rmsnormPipeline, [hiddenA, attnNormBuf, normedBuf, rmsUniform]),
        device.bindGroup(s.rmsnormPipeline, [hiddenB, ffnNormBuf, normed2Buf, rmsUniform]),
        dequantIntoShape(device, s.dequantTransposePipeline, wqShape, lw.wq, (bg, wg) => pushDequant(s.dequantTransposePipeline, bg, wg)),
        dequantIntoShape(device, s.dequantTransposePipeline, wkShape, lw.wk, (bg, wg) => pushDequant(s.dequantTransposePipeline, bg, wg)),
        dequantIntoShape(device, s.dequantTransposePipeline, wvShape, lw.wv, (bg, wg) => pushDequant(s.dequantTransposePipeline, bg, wg)),
        dequantIntoShape(device, s.dequantTransposePipeline, woShape, lw.wo, (bg, wg) => pushDequant(s.dequantTransposePipeline, bg, wg)),
        dequantIntoShape(device, s.dequantTransposePipeline, gateShape, lw.wGate, (bg, wg) => pushDequant(s.dequantTransposePipeline, bg, wg)),
        dequantIntoShape(device, s.dequantTransposePipeline, upShape, lw.wUp, (bg, wg) => pushDequant(s.dequantTransposePipeline, bg, wg)),
        dequantIntoShape(device, s.dequantTransposePipeline, downShape, lw.wDown, (bg, wg) => pushDequant(s.dequantTransposePipeline, bg, wg)),
      ]);
      transientBuffers.push(...wqTransient, ...wkTransient, ...wvTransient, ...woTransient, ...gateTransient, ...upTransient, ...downTransient);

      dispatch(s.rmsnormPipeline, attnNormGroup, [N]);
      dispatch(s.matmulPipeline, wqShape.matmulGroup, matmulWg(qDim));
      dispatch(s.matmulPipeline, wkShape.matmulGroup, matmulWg(kvDim));
      dispatch(s.matmulPipeline, wvShape.matmulGroup, matmulWg(kvDim));
      dispatch(s.ropePipeline, ropeQGroup, [wg256((N * qDim) / 2)]);
      dispatch(s.ropePipeline, ropeKGroup, [wg256((N * kvDim) / 2)]);
      dispatch(s.permutePipeline, splitQGroup, [wg256(N * qDim)]);
      dispatch(s.permutePipeline, splitKGroup, [wg256(N * kvDim)]);
      dispatch(s.permutePipeline, splitVGroup, [wg256(N * kvDim)]);
      dispatch(s.gqaScoresPipeline, gqaScoresGroup, [N, numHeads, 1]);
      dispatch(s.gqaContextPipeline, gqaContextGroup, [N, numHeads, 1]);
      dispatch(s.permutePipeline, mergeGroup, [wg256(N * qDim)]);
      dispatch(s.matmulPipeline, woShape.matmulGroup, matmulWg(hiddenSize));
      dispatch(s.elementwisePipeline, add1Group, [wg256(N * hiddenSize)]);
      dispatch(s.rmsnormPipeline, ffnNormGroup, [N]);
      dispatch(s.matmulPipeline, gateShape.matmulGroup, matmulWg(ffnHidden));
      dispatch(s.matmulPipeline, upShape.matmulGroup, matmulWg(ffnHidden));
      dispatch(s.activationPipeline, siluGroup, [wg256(N * ffnHidden)]);
      dispatch(s.elementwisePipeline, mulGroup, [wg256(N * ffnHidden)]);
      dispatch(s.matmulPipeline, downShape.matmulGroup, matmulWg(hiddenSize));
      dispatch(s.elementwisePipeline, add2Group, [wg256(N * hiddenSize)]);

      // Into the persistent, maxSeqLen-strided cache `runDecodeStep` reads —
      // one contiguous copy per head (this layer's whole N-position block),
      // not one per position: `kHeadMajorBuf`/`vHeadMajorBuf` are already
      // head-major, so head h's N positions are one contiguous run on both
      // ends of the copy. See the class doc's "Prefill's own KV" section.
      for (let h = 0; h < numKvHeads; h += 1) {
        copy(kHeadMajorBuf, h * N * headDim * 4, layer.kCacheBuf, h * maxSeqLen * headDim * 4, N * headDim * 4);
        copy(vHeadMajorBuf, h * N * headDim * 4, layer.vCacheBuf, h * maxSeqLen * headDim * 4, N * headDim * 4);
      }
    }

    const finalNormBuf = device.createStorageBuffer(hiddenSize * 4);
    device.upload(finalNormBuf, 0, weights.finalNorm);
    const finalNormGroup = await device.bindGroup(s.rmsnormPipeline, [hiddenA, finalNormBuf, finalNormedAllBuf, rmsUniform]);
    dispatch(s.rmsnormPipeline, finalNormGroup, [N]);

    let logits: Float32Array[];
    if (debugAllPositions) {
      // Test/debug-only path — see `debugAllPositionLogits`'s own doc for
      // why this exists and why it is never what `forward()` itself runs.
      // `lm_head` as one more projection shape, `M = N` instead of the 1
      // row `s.finalNormedBuf`/`this.lmHeadChunks` are sized for, reading
      // `finalNormedAllBuf` directly rather than the single-row copy the
      // production path below makes. Its own shape (never reused — this
      // path runs at most once per call), so no separate `setup*` step.
      const lmHeadMmUniform = uniformOf(device, [["u32", N], ["u32", vocabSize], ["u32", hiddenSize]]);
      const logitsAllBuf = device.createStorageBuffer(N * vocabSize * 4);
      const lmHeadShape = await setupMatmulProjectionShape(device, s.matmulPipeline, vocabSize, hiddenSize, lmHeadMmUniform, finalNormedAllBuf, logitsAllBuf);
      const lmHeadTransient = await dequantIntoShape(device, s.dequantTransposePipeline, lmHeadShape, weights.lmHead, (bg, wg) => dispatch(s.dequantTransposePipeline, bg, wg));
      transientBuffers.push(...lmHeadTransient);
      dispatch(s.matmulPipeline, lmHeadShape.matmulGroup, matmulWg(vocabSize));

      const stagingAll = device.createStorageBuffer(N * vocabSize * 4, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
      const [allLogitsFlat] = await device.batch(ops, [
        { staging: stagingAll, source: logitsAllBuf, sourceOffset: 0, length: N * vocabSize, type: "f32" },
      ]);
      logits = [];
      for (let t = 0; t < N; t += 1) logits.push((allLogitsFlat as Float32Array).slice(t * vocabSize, (t + 1) * vocabSize));
      transientBuffers.push(lmHeadShape.weightTBuf);
    } else {
      // Only the last row goes through lm_head — issue #117's "readbackは
      // 最終位置logitsのみ", and `forward`'s own doc for why every real
      // caller only ever wants that row. `s.finalNormedBuf`/`this.lmHeadChunks`
      // are the exact buffer and bind groups `runDecodeStep` uses, reused
      // here rather than duplicated: lm_head is by far the largest
      // projection (`vocabSize` rows), so computing it for every prompt
      // position instead of one would undo most of what going resident
      // buys prefill.
      copy(finalNormedAllBuf, (N - 1) * hiddenSize * 4, s.finalNormedBuf, 0, hiddenSize * 4);
      for (const chunk of this.lmHeadChunks) dispatch(s.matvecPipeline, chunk.group, [chunk.rowCount]);

      const readback = this.lmHeadChunks.map((chunk) => ({
        staging: chunk.staging, source: chunk.outBuf, sourceOffset: 0, length: chunk.rowCount, type: "f32" as const,
      }));
      const results = await device.batch(ops, readback);

      const finalLogits = new Float32Array(vocabSize);
      let offset = 0;
      for (const r of results) {
        finalLogits.set(r as Float32Array, offset);
        offset += r.length;
      }
      logits = [finalLogits];
    }

    // PR #119 review, item 7: every layer's transient dequant/scale/norm
    // buffer and every projection shape's `weightTBuf` — `batch()` above has
    // resolved, so the GPU has finished reading all of them; nothing here is
    // referenced by anything this call still needs (the persistent KV cache
    // and `s.finalNormedBuf`/`lmHeadChunks` writes are copies, already
    // landed in buffers this class owns for the long term, not these).
    for (const buf of transientBuffers) buf.destroy();
    for (const shape of [wqShape, wkShape, wvShape, woShape, gateShape, upShape, downShape]) shape.weightTBuf.destroy();

    this.tokensSoFar = N;
    // Dropped now that this call has read every layer's weight it needed —
    // see the constructor's doc.
    this.prefillWeights = null;
    return logits;
  }

  /**
   * Test/debug-only: every prefill position's logits, not just the final
   * one `forward()`'s own contract returns (issue #117's "readbackは最終
   * 位置logitsのみ"). Exists for `engine-q8-resident.wgsl.test.ts`'s
   * full-`N`-position correctness gate (PR #119 review, item 4) — a bug
   * confined to one early query position would not reliably surface in a
   * check of the final position and whatever decode derives from it, and
   * restoring that coverage without this method would mean reverting the
   * production optimisation `forward()`'s single-position contract exists
   * for. Never called by `forward()`, `runDecodeStep`, or anything else
   * this class's own production path reaches — recomputes `lm_head` for
   * every position via `matmul` (`runPrefillResident`'s own
   * `debugAllPositions` branch), which is exactly the "computing it for
   * every prompt position... undo[es] most of what going resident buys
   * prefill" cost that branch's sibling comment describes; harmless at the
   * tiny fixture's `vocabSize`, not something real generation should do.
   *
   * Same one-call contract as `forward()`'s own prefill branch otherwise:
   * only valid as this instance's first call, and consumes
   * `this.prefillWeights` the same way.
   */
  async debugAllPositionLogits(tokens: number[]): Promise<Float32Array[]> {
    if (!this.prefillWeights) {
      throw new Error("LlamaEngineQ8Resident.debugAllPositionLogits: only valid before prefill has run");
    }
    return this.runPrefillResident(tokens, true);
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
    // `LlamaEngine`/`LlamaEngineQ8`'s own `forward()` reject a call once
    // `posOffset + N > maxSeqLen` (see those files) before touching the KV
    // cache; this decode path bypasses `KVCache.write` entirely (a GPU-to-GPU
    // copy instead — see the class doc), so nothing else catches this. Past
    // this point, `at` unchecked would land the KV-cache copy below
    // (`(h * maxSeqLen + at) * headDim * 4`) into the next head's region, or
    // past the end of the buffer for the last head — a validation error with
    // no error scope around `batch()`'s encoder, which invalidates the whole
    // command buffer silently and resolves with the *previous* step's stale
    // logits rather than failing (PR #116 review, item 1;
    // `engine-q8-resident.wgsl.test.ts`'s "maxSeqLen boundary" test is the
    // regression coverage).
    if (at + 1 > maxSeqLen) {
      throw new Error(`LlamaEngineQ8Resident.forward: position ${at + 1} exceeds maxSeqLen=${maxSeqLen}`);
    }
    const s = this.shared;

    const embedVec = gatherDequantRows(this.embedTokens, [tokenId], hiddenSize);
    this.device.upload(s.hiddenA, 0, embedVec);
    this.device.upload(s.ropeQUniform, ROPE_POS_OFFSET_BYTE, new Uint32Array([at]));
    this.device.upload(s.ropeKUniform, ROPE_POS_OFFSET_BYTE, new Uint32Array([at]));
    this.device.upload(s.gqaScoresUniform, GQA_QUERY_OFFSET_BYTE, new Int32Array([at]));
    // Issue #117: `s_eff = at + 1` — this step's cache write (below) fills
    // position `at`, so positions `0..at` are the whole valid cache, and
    // `ops/gqa`'s scan need not reach past it to `maxSeqLen`. Both uniforms
    // carry it because `scores.wgsl` and `context.wgsl` are separate
    // dispatches reading separate params buffers (see their own docs — they
    // must agree, or `context.wgsl` reads `probs` columns `scores.wgsl` never
    // wrote this step).
    const sEff = new Uint32Array([at + 1]);
    this.device.upload(s.gqaScoresUniform, GQA_SCORES_S_EFF_BYTE, sEff);
    this.device.upload(s.gqaContextUniform, GQA_CONTEXT_S_EFF_BYTE, sEff);

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
