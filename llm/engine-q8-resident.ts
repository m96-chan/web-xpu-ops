import { kernel, params, type BatchProfile, type ResidentDevice, type ResidentOp } from "../harness/index.js";
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
 * Issue #131: an opt-in, per-call breakdown of one `forward()` call's own
 * wall time — the ~1.2s prefill fixed cost issue #131 exists to attribute,
 * not estimate. Passed by the caller (`undefined` by default, every real
 * caller in this repository today) and filled in as `runPrefillResident`/
 * `runDecodeStep` run; `forward()` never reads it back, so it costs nothing
 * to plumb through a call site that does not want it.
 *
 * Every timed quantity below is `performance.now()`, not inferred from GPU
 * timestamps — issue #131's own background is explicit that the leading
 * candidate (`matmulQ8IntoShape`'s `packInt8Rows` + `queue.writeBuffer`) is
 * CPU/transfer work a GPU timestamp query cannot see at all (it sits
 * *before* the compute pass that timestamp brackets), so a wall-clock
 * breakdown of the whole call is load-bearing here, not a fallback for a
 * device that lacks `timestamp-query` (that fallback is `gpuEntries: []` +
 * `gpuTimestampsSupported: false`, both below, and is a separate concern
 * from why `packMs`/`uploadMs`/`bindGroupMs` are wall-clock unconditionally).
 */
export interface ForwardProfile {
  /**
   * **Input, set by the caller before `forward()`, never touched by this
   * file except to read it** — unlike every field below, which
   * `runPrefillResident`/`runDecodeStep` reset to a fresh value at the start
   * of every call (see those methods' own note on reuse safety).
   *
   * `false` (`createForwardProfile()`'s own default) gets every CPU-side
   * field below (`packEntries`, `uploadMs`, `bindGroupMs`, `layerSetupMs`,
   * `submitToDoneMs`, `readbackMs`, `totalMs`) at effectively no cost over
   * an unprofiled call — `runPrefillResident`/`runDecodeStep` batch every
   * dispatch the same single/few-pass way production does, because no
   * per-dispatch `labels` array is even built. `true` additionally asks for
   * `gpuEntries`, which costs a dedicated compute pass **per labeled
   * dispatch** (`BatchProfile`'s own doc: `GPUComputePassTimestampWrites`
   * only covers a whole pass, not a point inside one) — real GPU-side
   * overhead, ~500 extra pass boundaries at Sarashina2.2-1B's 24-layer
   * scale, that inflates `submitToDoneMs` and this call's own `totalMs`
   * measurably.
   *
   * PR #141 review: an earlier version of this file always built the
   * `labels` array whenever any `profile` was passed at all, so the mere
   * act of asking for `packMs`/`layerSetupMs` silently paid the GPU
   * pass-splitting tax too — the measured effect was severe enough that the
   * CPU-only fields (`packInt8Rows` alone) came out *larger* than an
   * entirely unprofiled prefill's own total from an earlier PR's
   * measurement, which cannot be right for a sub-phase of that same call.
   * Splitting the two costs apart is what this flag is for: a caller who
   * wants the wall-clock CPU breakdown without paying for (or muddying it
   * with) the GPU pass-splitting overhead sets nothing; a caller who
   * explicitly wants the GPU-side kernel-time shares sets this `true` and
   * reads `totalMs` from *that* run only as "GPU-breakdown-mode total", not
   * as the production-shaped number the CPU-only fields are compared
   * against.
   */
  wantGpuBreakdown: boolean;
  /** One entry per `matmulQ8IntoShape` call this `forward()` made (prefill's own per-layer, per-projection weight pack) — `packInt8Rows`'s own CPU cost, issue #131 item 1. Empty for a decode step: `runDecodeStep` never calls `matmulQ8IntoShape` (its weights were packed once, in `create()`, via `buildProjection` — see that function's own doc). */
  packEntries: { layer: number; proj: string; ms: number; bytes: number }[];
  /** Cumulative `performance.now()` time across every `device.upload()` (`queue.writeBuffer`) call this `forward()` made, and the total bytes those calls wrote — issue #131 item 2. This is the *enqueue* cost (the synchronous call into `queue.writeBuffer`), not a measurement of when the device-side copy actually lands; `submitToDoneMs` below is where that device-side cost surfaces, folded in with every dispatch's own GPU time. */
  uploadMs: number;
  uploadBytes: number;
  /**
   * Sum of every individual `device.bindGroup(...)` call's own `performance.now()`
   * duration — issue #131 item 3's "24 rounds of `Promise.all(device.bindGroup())`"
   * for prefill (nine per layer), or decode's much smaller "each dispatch's
   * group already exists" case (decode does not call `bindGroup` in its own
   * steady-state loop — see `LayerResident`'s own doc — so this stays ~0
   * there, a useful contrast).
   *
   * **Not wall-clock, and can legitimately exceed `totalMs`**: prefill issues
   * nine of these per layer through one `Promise.all(...)`, so their
   * individual await windows overlap in real time — summing them adds up
   * each call's own round trip (`pushErrorScope`/`createBindGroup`/`await
   * popErrorScope()`, `harness/resident.ts#bindGroup`'s own doc) as if they
   * ran one after another, which is not what the wall clock saw. Measured
   * directly: this sum reached ~11.6s on one real 76-token prefill whose own
   * `totalMs` was ~2.2s (this repository's own #131 measurement run) —
   * proof the concurrency is real and this field alone cannot be read as "X
   * seconds of the call's own wall time". `layerSetupMs` below is the
   * wall-clock figure for the phase this overlaps with; use that one to
   * answer "how much of `totalMs`".
   */
  bindGroupMs: number;
  bindGroupCalls: number;
  /**
   * Wall-clock `performance.now()` time spent inside `runPrefillResident`'s
   * own per-layer `await Promise.all([...])` (issue #131 item 3), summed
   * across every layer — unlike `bindGroupMs` above, this *is* additive
   * against `totalMs`, because it brackets the whole concurrent block once
   * per layer rather than each of the nine calls inside it individually.
   * Covers that block's own CPU work too (`packInt8Rows` runs synchronously,
   * before its own call's `await`, inside the same `Promise.all` entries —
   * see `matmulQ8IntoShape`'s doc), so this is the number to compare against
   * `totalMs` for "how much of one `forward()` call this phase actually
   * cost", not `packMs`/`bindGroupMs` read in isolation. `0` for a decode
   * step: `runDecodeStep` has no equivalent per-token block (every group it
   * binds was built once, in `create()`).
   */
  layerSetupMs: number;
  /** Set once `device.batch(...)` resolves: GPU submit-to-completion wait and the readback `mapAsync` phase, timed separately (`BatchProfileSink`'s own doc) — issue #131 item 4's CPU-visible half. `null` until `batch()` writes them. */
  submitToDoneMs: number | null;
  readbackMs: number | null;
  /** Per-labeled-dispatch GPU time from timestamp-query (`label` is `"L<layer>:<kernel>"`, or `"final_norm"`/`"lm_head_chunk<i>"` for the ops outside the per-layer loop) — issue #131 item 4's GPU-visible half. Always `[]` when `gpuTimestampsSupported` is `false`; see `BatchProfileSink.gpuEntries`'s own doc for why that is not the same as "measured zero". */
  gpuEntries: { label: string; seconds: number }[];
  /** `device.timestampsSupported` at the time this call ran — carried alongside `gpuEntries` so a report can say "measured, GPU total was very small" instead of misreading `gpuEntries: []` as "GPU took no time" on a device that never supports the query at all. */
  gpuTimestampsSupported: boolean;
  /** Whole-call wall time, `performance.now()`-timed around this entire `forward()` call — the number this issue's own title names. */
  totalMs: number;
}

export function createForwardProfile(): ForwardProfile {
  return {
    wantGpuBreakdown: false,
    packEntries: [],
    uploadMs: 0,
    uploadBytes: 0,
    bindGroupMs: 0,
    bindGroupCalls: 0,
    layerSetupMs: 0,
    submitToDoneMs: null,
    readbackMs: null,
    gpuEntries: [],
    gpuTimestampsSupported: false,
    totalMs: 0,
  };
}

/**
 * Resets every *output* field of `profile` in place — everything except
 * `wantGpuBreakdown`, the one field the caller sets and this file only
 * reads (see that field's own doc) — called once at the very start of
 * `runPrefillResident`/`runDecodeStep` whenever `profile` is given.
 *
 * PR #141 review, item 7: without this, a `ForwardProfile` reused across
 * two `forward()` calls (the same object passed twice, deliberately or by
 * mistake) mixed two different call's data in an inconsistent way —
 * `packEntries`/`bindGroupMs`/`bindGroupCalls`/`layerSetupMs`/`uploadMs`/
 * `uploadBytes` *accumulate* (`+=`/`.push`) across both calls, while
 * `submitToDoneMs`/`readbackMs`/`gpuEntries`/`totalMs` *overwrite* and so
 * only ever reflect the second call — neither behaviour is obviously
 * correct, and a caller could easily read the object without realising
 * which fields meant what. Resetting every output field to
 * `createForwardProfile()`'s own zero state at the top of every profiled
 * call makes "one `forward()` call, one profile snapshot" true regardless
 * of whether the caller passes a fresh object or reuses one.
 */
function resetForwardProfileOutputs(profile: ForwardProfile, timestampsSupported: boolean): void {
  profile.packEntries = [];
  profile.uploadMs = 0;
  profile.uploadBytes = 0;
  profile.bindGroupMs = 0;
  profile.bindGroupCalls = 0;
  profile.layerSetupMs = 0;
  profile.submitToDoneMs = null;
  profile.readbackMs = null;
  profile.gpuEntries = [];
  profile.gpuTimestampsSupported = timestampsSupported;
  profile.totalMs = 0;
}

/**
 * Wraps `device.upload`/`device.bindGroup` with `performance.now()` timing
 * into `sink` (issue #131 items 2-3) — every other member of `device`
 * (`createStorageBuffer`, `pipelineFor`, `batch`, `stats`, `destroy`,
 * `timestampsSupported`) passes straight through unwrapped. `batch` itself
 * is profiled separately, via the `BatchProfile` argument `runPrefillResident`/
 * `runDecodeStep` build alongside their own `ops`/`labels` arrays (item 4) —
 * this wrapper has no visibility into which `ops` entries batch() will
 * dispatch, so it cannot time that half itself.
 *
 * A plain object spread, not a class: `createResidentDevice`/
 * `createBrowserResidentDevice` both return object literals whose methods
 * close over their own `device`/`stats` rather than reading `this`, so
 * spreading and overriding two of them here is safe — nothing here depends
 * on identity beyond "has these methods".
 */
function instrumentDevice(device: ResidentDevice, sink: ForwardProfile): ResidentDevice {
  return {
    ...device,
    upload(buffer, offset, data) {
      const t0 = performance.now();
      device.upload(buffer, offset, data);
      sink.uploadMs += performance.now() - t0;
      sink.uploadBytes += data.byteLength;
    },
    async bindGroup(pipeline, buffers) {
      const t0 = performance.now();
      const group = await device.bindGroup(pipeline, buffers);
      sink.bindGroupMs += performance.now() - t0;
      sink.bindGroupCalls += 1;
      return group;
    },
  };
}

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
 * Prefill keeps a **matmul**-shaped path rather than switching to `matvecQ8`
 * per token: `matvecQ8` re-reads a projection's whole weight once per query
 * vector, so running it once per prompt token would multiply the weight
 * traffic prefill moves by `N` — turning a 360-token prompt into 360x the
 * decode bandwidth, the opposite of the goal. `matmulQ8` (issue #128; before
 * it, plain `matmul`) reads each projection's weight once and computes
 * against all `N` rows at once, same as `LlamaEngineQ8`'s own prefill
 * branch; only *how many round trips it costs* changes here, not the shape
 * of the contraction. Issue #117 read the packed int8 weight by
 * dequantizing-and-transposing it into `matmul`'s plain-f32 `b` operand once
 * per layer per `forward()` call that runs prefill (never per token,
 * matching `LlamaEngineQ8#project`'s own accepted cost — that class's doc:
 * "この transient dequantが1回のgenerationにつき1回で済む") — but that pass
 * itself, a full write of `inFeatures * outFeatures` f32 values immediately
 * followed by a full read of the same, turned out to be the dominant term in
 * prefill's own fixed cost regardless of prompt length (issue #128's own
 * background: 76-token and 365-token prompts measured at effectively the
 * same ~1.2s). `matmulQ8` (`ops/matmul/reference.ts`'s own doc) removes that
 * pass: the kernel dequantizes in-kernel from the same packed int8 layout
 * `matvecQ8` already reads, so there is no intermediate f32 buffer to write
 * or read at all — only the packed weight and its per-row scale, uploaded
 * once per layer per `forward()` call (`matmulQ8IntoShape` below), the same
 * "once per generation, not per token" cost `LlamaEngineQ8#project`'s doc
 * already accepted, just without the transpose in between.
 *
 * ## Reshape: a GPU permute kernel, not a CPU round trip
 *
 * `matmulQ8`'s projections produce `[N, heads, dim]` token-major (a token's
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
 *
 * ## `reset()`: a second (third, ...) independent generation, same instance
 *
 * Issue #120: before this, every new independent prompt needed a whole new
 * `ResidentDevice` and `LlamaEngineQ8Resident.create()` call — `forward()`'s
 * own contract was "prefill exactly once, then decode only", so there was no
 * other way back to "accepts a new prefill". alibi-ai's chat integration
 * measured what that costs in practice: 17-33s per independent turn,
 * dominated by `create()`'s own `device.upload()` calls rebuilding every
 * persistent `matvecQ8` weight buffer from the ~1.4 GiB checkpoint (issue
 * #120's own background) — for a workload (a chat: many short, independent
 * generations) where the model itself never changes between turns.
 *
 * `reset()` sets `tokensSoFar` back to 0 and re-arms `forward()`'s prefill
 * routing, and touches nothing else: every pipeline, bind group and
 * persistent buffer `create()` built (`this.shared`, `this.layers`,
 * `this.lmHeadChunks` — the weight buffers `buildProjection` populated via
 * `device.upload()`, specifically, since those are the expensive ones) is
 * still exactly what the next generation's `runDecodeStep`/`runPrefillResident`
 * calls read and write. Old KV cache contents are left in place, not
 * cleared — see `reset()`'s own doc for why that is still correct rather
 * than an oversight, and `llm/engine-q8-resident.reset.wgsl.test.ts` for the
 * `+Infinity`-poisoned proof.
 *
 * Not free, still: `runPrefillResident`'s own transient, `N`-sized buffer
 * allocation and its per-layer packed-weight pack-and-upload
 * (`matmulQ8IntoShape`, issue #128) both re-run on every generation,
 * `reset()`-triggered or not — that was already true
 * before issue #120 (every generation pays it exactly once, on its own
 * first `forward()` call), and `reset()` does not change it. What `reset()`
 * removes is strictly the `create()` call itself and the weight re-upload
 * inside it, which is the term alibi-ai's own measurement named as
 * dominant.
 */

const CODE = {
  rmsnorm: kernel(new URL("../ops/rmsnorm/index.ts", import.meta.url)),
  matvecQ8: kernel(new URL("../ops/matvec/index.ts", import.meta.url), "q8"),
  // Issue #111: decode-only fused entry points on the same `ops/matvec` op —
  // see `ops/matvec/wgsl/q8_ffn.wgsl`/`q8_residual.wgsl`'s own docs. Prefill
  // (`runPrefillResident`) keeps the unfused `matmulQ8`+`activation`+
  // `elementwise` path deliberately — issue #111's own "スコープ外: プリフィル
  // 専用最適化" — so these two are read only by `runDecodeStep`'s bind groups.
  matvecQ8Ffn: kernel(new URL("../ops/matvec/index.ts", import.meta.url), "q8_ffn"),
  matvecQ8Residual: kernel(new URL("../ops/matvec/index.ts", import.meta.url), "q8_residual"),
  rope: kernel(new URL("../ops/rope/index.ts", import.meta.url)),
  gqaScores: kernel(new URL("../ops/gqa/index.ts", import.meta.url), "scores"),
  gqaContext: kernel(new URL("../ops/gqa/index.ts", import.meta.url), "context"),
  activation: kernel(new URL("../ops/activation/index.ts", import.meta.url)),
  elementwise: kernel(new URL("../ops/elementwise/index.ts", import.meta.url)),
  // Issue #117's prefill reshape — see `ops/permute/wgsl/kernel.wgsl`'s own doc.
  permute: kernel(new URL("../ops/permute/index.ts", import.meta.url)),
  // Issue #128's prefill weight path — reads the packed int8 weight directly,
  // no separate dequant-transpose pass. See `ops/matmul/reference.ts#matmulQ8`'s
  // own doc.
  matmulQ8: kernel(new URL("../ops/matmul/index.ts", import.meta.url), "q8"),
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

/**
 * Packs, uploads and binds one `matvecQ8Ffn` projection (issue #111): two
 * weights (gate, up), sharing one `vector` input and writing one fused
 * `silu(gate) * up` output — `buildProjection`'s own doc explains why gate
 * and up get their own weight/scale buffers rather than one fused buffer
 * (the same `minStorageBufferOffsetAlignment` reasoning applies here
 * unchanged, one layer up: the *pair* is fused into one dispatch, but each
 * half is still its own buffer). Binding order matches
 * `ops/matvec/wgsl/q8_ffn.wgsl`'s own bindings 0-6, and `uniform` is the
 * *existing* `{N: ffnHidden, K: hiddenSize}` uniform `create()` already
 * builds for the old `gateGroup` (`gateUniform`) — this op's `Params` shape
 * is identical to `q8`'s, so no new uniform buffer is needed.
 */
async function buildFfnProjection(
  device: ResidentDevice,
  pipeline: GPUComputePipeline,
  gate: QuantizedLinear,
  up: QuantizedLinear,
  N: number,
  K: number,
  uniform: GPUBuffer,
  vectorBuf: GPUBuffer,
  outBuf: GPUBuffer,
): Promise<GPUBindGroup> {
  const packedGate = packInt8Rows(gate.codes, N, K);
  const weightGateBuf = device.createStorageBuffer(packedGate.byteLength);
  device.upload(weightGateBuf, 0, packedGate);
  const scaleGateBuf = device.createStorageBuffer(gate.scale.byteLength);
  device.upload(scaleGateBuf, 0, gate.scale);
  const packedUp = packInt8Rows(up.codes, N, K);
  const weightUpBuf = device.createStorageBuffer(packedUp.byteLength);
  device.upload(weightUpBuf, 0, packedUp);
  const scaleUpBuf = device.createStorageBuffer(up.scale.byteLength);
  device.upload(scaleUpBuf, 0, up.scale);
  return device.bindGroup(pipeline, [weightGateBuf, scaleGateBuf, weightUpBuf, scaleUpBuf, vectorBuf, outBuf, uniform]);
}

/**
 * Packs, uploads and binds one `matvecQ8Residual` projection (issue #111):
 * same shape as `buildProjection`, plus a `residualBuf` input bound between
 * `vector` and `output` — `ops/matvec/wgsl/q8_residual.wgsl`'s own binding
 * order. `uniform` is, again, an existing `{N, K}` uniform (`oUniform` for
 * `wo`, `downUniform` for `wDown`) — `q8_residual`'s `Params` struct is
 * identical to `q8`'s.
 */
async function buildResidualProjection(
  device: ResidentDevice,
  pipeline: GPUComputePipeline,
  linear: QuantizedLinear,
  N: number,
  K: number,
  uniform: GPUBuffer,
  vectorBuf: GPUBuffer,
  residualBuf: GPUBuffer,
  outBuf: GPUBuffer,
): Promise<GPUBindGroup> {
  const packed = packInt8Rows(linear.codes, N, K);
  const weightBuf = device.createStorageBuffer(packed.byteLength);
  device.upload(weightBuf, 0, packed);
  const scaleBuf = device.createStorageBuffer(linear.scale.byteLength);
  device.upload(scaleBuf, 0, linear.scale);
  return device.bindGroup(pipeline, [weightBuf, scaleBuf, vectorBuf, residualBuf, outBuf, uniform]);
}

/** Must match `TILE` in `ops/matmul/wgsl/q8.wgsl` — copied per rule 2, same as `llm/kernels.ts#MATMUL_TILE`. */
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
 * One `matmulQ8` projection's *shape* — everything `runPrefillResident` can
 * set up once and reuse for every layer that has a projection this shape:
 * `aBuf`/`outBuf`/`uniform` are the activation buffers and the `(N, M, K)`
 * uniform, none of which depend on which layer's weight is about to flow
 * through the dispatch. `outFeatures`/`inFeatures` are kept alongside for
 * `matmulQ8IntoShape` below to size and pack that layer's own weight.
 *
 * Issue #128: this replaces `MatmulProjectionShape`'s pre-#128 `weightTBuf` —
 * a shared `[inFeatures, outFeatures]` f32 buffer that `ops/dequant_transpose`
 * dequantized-and-transposed a layer's packed weight *into*, once per layer,
 * for plain `matmul` to then read back out in full. `matmulQ8` (`ops/matmul/reference.ts`'s
 * own doc) reads the packed int8 weight directly — same wire format
 * `matvecQ8`'s decode-side `buildProjection` already uploads — so there is no
 * shared GPU-side scratch buffer left to reuse across layers at all: a
 * layer's weight now goes straight from `packInt8Rows` (CPU) to its own
 * transient `weightBuf`/`scaleBuf` (`matmulQ8IntoShape` below), the same
 * shape `dequantIntoShape` used for its *own* per-layer transient
 * weight/scale before dequantizing them — except now nothing dequantizes
 * them, `matmulQ8`'s dispatch just reads them, so the GPU write-then-read of
 * `inFeatures * outFeatures` f32 values that pass paid on every `forward()`
 * call is gone.
 */
interface MatmulQ8ProjectionShape {
  aBuf: GPUBuffer;
  outBuf: GPUBuffer;
  uniform: GPUBuffer;
  outFeatures: number;
  inFeatures: number;
}

function matmulQ8ProjectionShape(
  outFeatures: number,
  inFeatures: number,
  matmulUniform: GPUBuffer,
  aBuf: GPUBuffer,
  outBuf: GPUBuffer,
): MatmulQ8ProjectionShape {
  return { aBuf, outBuf, uniform: matmulUniform, outFeatures, inFeatures };
}

/**
 * One layer's `matmulQ8` bind group, built fresh every call (unlike
 * `runDecodeStep`'s `buildProjection`, built once in `create()`): a fresh
 * `weightBuf`/`scaleBuf` per layer per projection, packed the same way
 * `buildProjection` packs its own resident decode-side weight
 * (`packInt8Rows`, `matvecQ8`'s wire format — a plain per-row byte copy, no
 * arithmetic, `ops/dequant_transpose/reference.ts`'s own doc measured this
 * step at ~170ms total across Sarashina2.2-1B's 24 layers), bound straight
 * into `matmulQ8` alongside `shape`'s own constant `aBuf`/`outBuf`/`uniform`.
 *
 * A fresh buffer and a fresh bind group per layer, not a shared per-shape
 * buffer reused via `device.upload` the way a resident weight might suggest
 * — `ResidentDevice.upload`'s own doc says why that would be wrong here:
 * `queue.writeBuffer` runs immediately, not in `ops`' own push order, so
 * uploading layer `l + 1`'s bytes into a buffer layer `l`'s own (not yet
 * submitted) `matmulQ8` dispatch still reads would read the wrong layer's
 * weight the moment the whole batch actually submits. `dequantIntoShape`
 * (pre-#128) avoided this by writing through a GPU *dispatch* recorded into
 * `ops`, which executes in push order; a plain upload has no such op to
 * record. `weightBuf`/`scaleBuf` are collected into the caller's
 * `transientBuffers` and `destroy()`ed once `batch()` resolves, the same
 * lifetime `dequantIntoShape`'s own transient pair had (minus the third,
 * `dequantUniform`, which no longer exists).
 */
async function matmulQ8IntoShape(
  device: ResidentDevice,
  matmulQ8Pipeline: GPUComputePipeline,
  shape: MatmulQ8ProjectionShape,
  linear: QuantizedLinear,
  /** Issue #131 item 1: when given, times this call's own `packInt8Rows` and records it under `layer`/`proj` — `undefined` for every call site that does not care (this function's own callers pass it only from a profiled `forward()`). */
  packProfile?: { sink: ForwardProfile; layer: number; proj: string },
): Promise<{ bindGroup: GPUBindGroup; transient: GPUBuffer[] }> {
  const packStart = packProfile ? performance.now() : 0;
  const packed = packInt8Rows(linear.codes, shape.outFeatures, shape.inFeatures);
  if (packProfile) {
    packProfile.sink.packEntries.push({ layer: packProfile.layer, proj: packProfile.proj, ms: performance.now() - packStart, bytes: packed.byteLength });
  }
  const weightBuf = device.createStorageBuffer(packed.byteLength);
  device.upload(weightBuf, 0, packed);
  const scaleBuf = device.createStorageBuffer(linear.scale.byteLength);
  device.upload(scaleBuf, 0, linear.scale);
  const bindGroup = await device.bindGroup(matmulQ8Pipeline, [shape.aBuf, weightBuf, scaleBuf, shape.outBuf, shape.uniform]);
  return { bindGroup, transient: [weightBuf, scaleBuf] };
}

interface SharedResident {
  rmsnormPipeline: GPUComputePipeline;
  matvecPipeline: GPUComputePipeline;
  /** Issue #111, decode only — see `CODE.matvecQ8Ffn`'s own doc. */
  matvecFfnPipeline: GPUComputePipeline;
  /** Issue #111, decode only — see `CODE.matvecQ8Residual`'s own doc. */
  matvecResidualPipeline: GPUComputePipeline;
  ropePipeline: GPUComputePipeline;
  gqaScoresPipeline: GPUComputePipeline;
  gqaContextPipeline: GPUComputePipeline;
  activationPipeline: GPUComputePipeline;
  elementwisePipeline: GPUComputePipeline;
  /** Prefill only — see `runPrefillResident`. */
  permutePipeline: GPUComputePipeline;
  /** Prefill only — see `runPrefillResident`/`matmulQ8IntoShape`. */
  matmulQ8Pipeline: GPUComputePipeline;

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
  /**
   * Issue #111: `matvecQ8Ffn`'s fused `silu(gate) * up` output, read
   * straight back in as `down_proj`'s `vector` input — the un-fused path's
   * `gateOutBuf`/`upOutBuf`/`gateActBuf` intermediates (matvecQ8(gate),
   * matvecQ8(up), then activation(silu) into a *separate* buffer before the
   * multiply) no longer exist for decode; this is the only buffer between
   * the FFN's two projections now.
   */
  gatedBuf: GPUBuffer;
  finalNormedBuf: GPUBuffer;

  ropeQUniform: GPUBuffer;
  ropeKUniform: GPUBuffer;
  gqaScoresUniform: GPUBuffer;
  gqaContextUniform: GPUBuffer;

  ropeQGroup: GPUBindGroup;
  ropeKGroup: GPUBindGroup;
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
  /** Issue #111: `matvecQ8Residual` — `hiddenB = hiddenA + wo · attnOutBuf`, fusing the old `woGroup` matvec and the old `add1Group` elementwise add into one dispatch. */
  woGroup: GPUBindGroup;
  /** Issue #111: `matvecQ8Ffn` — `gatedBuf = silu(wGate · normed2Buf) * (wUp · normed2Buf)`, fusing the old `gateGroup`/`upGroup` matvecs and the old `siluGroup`/`mulGroup` dispatches into one. */
  ffnGroup: GPUBindGroup;
  /** Issue #111: `matvecQ8Residual` — `hiddenA = hiddenB + wDown · gatedBuf`, fusing the old `downGroup` matvec and the old `add2Group` elementwise add into one dispatch. */
  downGroup: GPUBindGroup;
}

interface LmHeadChunk {
  rowCount: number;
  group: GPUBindGroup;
  outBuf: GPUBuffer;
  staging: GPUBuffer;
}

export class LlamaEngineQ8Resident {
  /**
   * `forward()`'s own routing state, not merely a position counter: `0`
   * means "the next `forward()` call must be prefill", non-zero means
   * "decode". PR #126 review, item 4: an earlier version of this class kept
   * a *separate* `hasPrefilled` boolean for that routing decision, redundant
   * with `tokensSoFar` at every real call site — the one place they could
   * diverge was `debugAllPositionLogits([])` (an empty prompt, which
   * `forward()` itself rejects but that debug method did not), which left
   * `tokensSoFar === 0` but `hasPrefilled === true`, an inconsistent state
   * nothing else in this class expected. Routing on `tokensSoFar === 0`
   * directly makes that state unreachable rather than merely unlikely: `N =
   * 0` prefill leaves `tokensSoFar` at `0`, so the *next* call is still
   * routed as prefill, exactly the self-consistent behaviour a real empty
   * prompt would need anyway.
   */
  private tokensSoFar = 0;
  /**
   * PR #126 review, item 3: `reset()`'s own writes are synchronous, but
   * `runPrefillResident`/`runDecodeStep` only write `this.tokensSoFar` (and,
   * before this field existed, `this.hasPrefilled`) *after* their own
   * `await device.batch(...)` resolves — so a `reset()` call landing while a
   * `forward()` call is still in flight would otherwise be silently undone
   * the moment that in-flight call's own state-write ran, with no error and
   * no trace. Bumped by every `reset()` call; each `forward()`-family method
   * captures the value at its own start and refuses to write state (or
   * return logits as if they were still valid) if it has changed by the
   * time that method's own GPU work resolves — see `assertSameEpoch`.
   */
  private generationEpoch = 0;

  private constructor(
    private readonly config: LlamaConfig,
    private readonly device: ResidentDevice,
    /**
     * Held for this instance's whole lifetime, not dropped after the first
     * `forward()` call the way a pre-#120 version of this field
     * (`prefillWeights: LlamaWeightsQ8 | null`) did. `runPrefillResident`
     * needs these CPU-side packed weights (`codes`/`scale` per layer) every
     * time it runs — to pack and upload each layer's projection for
     * `matmulQ8` (issue #128), since prefill's own transient weight buffers
     * are not the persistent `matvecQ8` ones `create()` already built into
     * `this.layers` — and
     * issue #120 asks for exactly that: `reset()` followed by a *second*
     * prefill, not a one-shot.
     *
     * Costs nothing extra to keep: `weights` is a reference to the exact
     * object the caller passed into `create()`, not a clone, and every real
     * caller in this repository (`examples/llm-demo/src/main.ts`'s `loaded`)
     * already keeps its own reference to it for the whole page session
     * regardless — this field holding a second reference to the same
     * ~1.4 GiB object adds a pointer, not another ~1.4 GiB. The pre-#120
     * reasoning for dropping it ("does not outlive the one call that needs
     * it") only ever saved memory for a hypothetical caller that itself
     * gave up its own reference right after `create()`; no caller here does.
     */
    private readonly weights: LlamaWeightsQ8,
    private readonly embedTokens: QuantizedLinear,
    private readonly shared: SharedResident,
    private readonly layers: LayerResident[],
    private readonly lmHeadChunks: LmHeadChunk[],
  ) {}

  /** Positions already resident in the KV cache — 0 before the first `forward`, and again immediately after `reset()`. */
  get position(): number {
    return this.tokensSoFar;
  }

  /**
   * Issue #120: back to `create()`'s own initial state — position 0, next
   * `forward()` accepted as a new prefill — **without** rebuilding anything
   * `create()` built. Every pipeline, bind group and persistent buffer
   * (`this.shared`, `this.layers`, `this.lmHeadChunks`) stays exactly as it
   * was; `runPrefillResident`'s transient, `N`-sized buffers for the new
   * prompt are still allocated fresh (they always were, once per prefill —
   * see that method's own doc), and its per-layer packed-weight
   * pack-and-upload still re-runs (needs `this.weights`, kept resident for exactly
   * this — see that field's own doc) — neither of those is the ~1.4 GiB
   * weight re-upload issue #120 exists to remove, which was **`create()`
   * itself**: a whole new `ResidentDevice`, a whole new set of persistent
   * `matvecQ8` weight buffers built via `device.upload()`. `reset()` means
   * that call never happens again for a second, independent generation.
   *
   * Old KV cache contents are **not** cleared here, deliberately — nothing
   * clears them. The next generation's own prefill only ever *writes* into
   * `layer.kCacheBuf`/`vCacheBuf` (`runPrefillResident`'s per-head
   * `copyBufferToBuffer`, never a read), and decode's attention is bounded
   * by `sEff = position + 1` (issue #117), which after `reset()` starts
   * from the same 0 `tokensSoFar` a freshly `create()`d engine would — so
   * the old generation's leftover bytes, wherever they still physically
   * sit in the buffer, are simply never in scanning range again. Proved by
   * `llm/engine-q8-resident.reset.wgsl.test.ts`, which poisons those bytes
   * with `+Infinity` before `reset()` rather than trusting this argument on
   * its own (`context.wgsl` reads `v` unconditionally within its bound — a
   * scan that reached the poison would return `NaN`, not a plausible wrong
   * number).
   *
   * **Not safe to call while a `forward()` call on this same instance is
   * still in flight** (PR #126 review, item 3) — `reset()` this instance
   * mid-generation, e.g. a chat UI's "new conversation" button firing while
   * the previous turn's decode step is still awaiting the GPU. Call it only
   * once every `forward()` promise you have started has settled (`await`ed
   * or rejected). This method does not detect that misuse itself (it has no
   * way to know whether a `forward()` call is outstanding), but the
   * in-flight `forward()` call detects being reset out from under it —
   * `assertSameEpoch` throws rather than writing stale position/routing
   * state or returning logits computed against a generation that no longer
   * exists from this instance's point of view.
   */
  reset(): void {
    this.generationEpoch += 1;
    this.tokensSoFar = 0;
  }

  /**
   * PR #126 review, item 3: throws if `reset()` ran on this instance after
   * `epoch` (captured by the caller at the *start* of its own `forward()`
   * work) was captured — meaning this call's result is stale and must not
   * write `this.tokensSoFar`, or be returned as though it reflects the
   * engine's current generation. Called once, immediately after the one
   * `await device.batch(...)`/`await` chain each of `runPrefillResident`'s
   * two branches and `runDecodeStep` ends with — after that point nothing
   * further in this class blocks, so one check there covers the entire
   * window `reset()` could have raced against.
   *
   * This *cannot* undo the GPU-side KV cache writes a stale call's own
   * `device.batch(...)` already made by the time this runs — those
   * `copyBufferToBuffer`s were already recorded and submitted before
   * `reset()` was even called (this method only runs after that submit's
   * own `await` resolves). That is not a gap: those writes become exactly
   * the kind of "old generation's leftover bytes" `reset()`'s own doc
   * already explains are harmless — whatever *real* generation runs next
   * overwrites that same cache region with its own prefill before decode
   * ever reads it, the same guarantee `reset.wgsl.test.ts`'s poison tests
   * exercise deliberately. What this check protects is purely the JS-side
   * bookkeeping (`tokensSoFar`) and this call's own return value, not the
   * GPU buffers.
   */
  private assertSameEpoch(epoch: number, caller: string): void {
    if (this.generationEpoch !== epoch) {
      throw new Error(
        `LlamaEngineQ8Resident.${caller}: reset() was called while this call was still in flight — its result is stale; its own position update has been discarded (its GPU-side KV writes already landed, but the next real generation's own prefill will overwrite that same region before anything reads it — see reset()'s own doc)`,
      );
    }
  }

  /**
   * Test/debug-only: overwrites every layer's entire KV cache
   * (`kCacheBuf`/`vCacheBuf`, every position, not just the ones a given
   * generation has actually written) with `value`. Exists for
   * `reset.wgsl.test.ts`'s own correctness proof — see `reset()`'s own doc
   * — so that a `sEff` bound leaking past where it should stops being "a
   * number that happens to look plausible" and becomes `NaN` on contact
   * (`+Infinity` through `context.wgsl`'s unconditional `v` read). Never
   * called by `forward()`, `reset()`, or anything else this class's own
   * production path reaches — writing directly into the persistent cache
   * outside `runPrefillResident`'s/`runDecodeStep`'s own copy operations is
   * not something real generation should ever do.
   */
  debugPoisonKVCache(value: number): void {
    const { numKvHeads, maxSeqLen, headDim } = this.config;
    const poison = new Float32Array(numKvHeads * maxSeqLen * headDim).fill(value);
    for (const layer of this.layers) {
      this.device.upload(layer.kCacheBuf, 0, poison);
      this.device.upload(layer.vCacheBuf, 0, poison);
    }
  }

  static async create(config: LlamaConfig, weights: LlamaWeightsQ8, device: ResidentDevice): Promise<LlamaEngineQ8Resident> {
    assertWeightShapesQ8(config, weights);
    const { numLayers, hiddenSize, numHeads, numKvHeads, headDim, ffnHidden, vocabSize, maxSeqLen, ropeTheta, rmsNormEps } = config;
    const qDim = numHeads * headDim;
    const kvDim = numKvHeads * headDim;

    const [
      rmsnormPipeline, matvecPipeline, matvecFfnPipeline, matvecResidualPipeline,
      ropePipeline, gqaScoresPipeline, gqaContextPipeline, activationPipeline, elementwisePipeline,
      permutePipeline, matmulQ8Pipeline,
    ] = await Promise.all([
        device.pipelineFor(CODE.rmsnorm),
        device.pipelineFor(CODE.matvecQ8),
        // Issue #111, decode only — see `CODE.matvecQ8Ffn`/`CODE.matvecQ8Residual`'s own doc.
        device.pipelineFor(CODE.matvecQ8Ffn),
        device.pipelineFor(CODE.matvecQ8Residual),
        device.pipelineFor(CODE.rope),
        device.pipelineFor(CODE.gqaScores),
        device.pipelineFor(CODE.gqaContext),
        device.pipelineFor(CODE.activation),
        device.pipelineFor(CODE.elementwise),
        // Prefill only (issue #117's resident prefill, `runPrefillResident`
        // below) — pipeline creation does not depend on `N`, only the
        // dispatch workgroup counts and buffer sizes do, so these are built
        // once here alongside decode's, not per `forward()` call.
        device.pipelineFor(CODE.permute),
        device.pipelineFor(CODE.matmulQ8),
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
    // Issue #111: decode's only FFN intermediate now — `matvecQ8Ffn` writes
    // `silu(gate) * up` straight here, and `matvecQ8Residual` reads it
    // straight back out as `down_proj`'s vector input. The old
    // `gateOutBuf`/`upOutBuf`/`gateActBuf`/`projOutBuf`/`downOutBuf`
    // intermediates (separate matvecQ8 outputs, a separate silu output, a
    // separate pre-residual-add projection output) no longer exist for
    // decode — see `SharedResident.gatedBuf`'s own doc.
    const gatedBuf = device.createStorageBuffer(ffnHidden * 4);
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
    // Issue #111: `gateUniform`'s `{N: ffnHidden, K: hiddenSize}` shape is
    // shared by *both* halves of the fused FFN kernel (`q8_ffn`'s `Params`
    // struct has one `{N, K}`, read by both the gate and up weight — see
    // `buildFfnProjection`'s own doc), so there is no separate `upUniform`
    // to build: gate and up projections have the exact same shape at every
    // real config (`wGate`/`wUp` are always `[ffnHidden, hiddenSize]`), and
    // a second, byte-identical uniform buffer would only be another buffer
    // to keep in sync, never a different value.
    const gateUniform = uniformOf(device, [["u32", ffnHidden], ["u32", hiddenSize]]);
    const downUniform = uniformOf(device, [["u32", hiddenSize], ["u32", ffnHidden]]);
    // Issue #111: `gateUniform`/`oUniform`/`downUniform` above are reused
    // as-is by the fused decode kernels below — `q8_ffn`'s and
    // `q8_residual`'s `Params` structs are both `{N, K}`, identical to
    // plain `q8`'s, so no new uniform buffers are needed for them. The old
    // `siluUniform`/`addUniform`/`mulUniform` triple (decode's un-fused
    // activation/elementwise dispatches) is gone with the dispatches that
    // read it — `runPrefillResident` below declares its own copies of all
    // three locally, unaffected, since prefill's FFN stays un-fused (issue
    // #111's own "スコープ外: プリフィル専用最適化").

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
      // Issue #111: `hiddenB = hiddenA + wo · attnOutBuf` — fused
      // matvecQ8+residual, replacing the old `woGroup` (plain matvecQ8 into
      // `projOutBuf`) plus a separate `add1Group` elementwise dispatch.
      const woGroup = await buildResidualProjection(device, matvecResidualPipeline, layer.wo, hiddenSize, qDim, oUniform, attnOutBuf, hiddenA, hiddenB);
      // Issue #111: `gatedBuf = silu(wGate · normed2Buf) * (wUp · normed2Buf)`
      // — fused gate+up+silu+multiply, replacing the old `gateGroup`/
      // `upGroup` (two plain matvecQ8 dispatches) plus the old
      // `siluGroup`/`mulGroup` (activation, then elementwise multiply).
      const ffnGroup = await buildFfnProjection(device, matvecFfnPipeline, layer.wGate, layer.wUp, ffnHidden, hiddenSize, gateUniform, normed2Buf, gatedBuf);
      // Issue #111: `hiddenA = hiddenB + wDown · gatedBuf` — fused
      // matvecQ8+residual, same shape as `woGroup` above, replacing the old
      // `downGroup` (plain matvecQ8 into `downOutBuf`) plus `add2Group`.
      const downGroup = await buildResidualProjection(device, matvecResidualPipeline, layer.wDown, hiddenSize, ffnHidden, downUniform, gatedBuf, hiddenB, hiddenA);

      const kCacheBuf = device.createStorageBuffer(numKvHeads * maxSeqLen * headDim * 4);
      const vCacheBuf = device.createStorageBuffer(numKvHeads * maxSeqLen * headDim * 4);
      const scoresGroup = await device.bindGroup(gqaScoresPipeline, [qRopedBuf, kCacheBuf, maskBuf, probsBuf, gqaScoresUniform]);
      const contextGroup = await device.bindGroup(gqaContextPipeline, [probsBuf, vCacheBuf, attnOutBuf, gqaContextUniform]);

      layers.push({
        kCacheBuf, vCacheBuf, attnNormGroup, ffnNormGroup, wqGroup, wkGroup, wvGroup, scoresGroup, contextGroup, woGroup, ffnGroup, downGroup,
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
      rmsnormPipeline, matvecPipeline, matvecFfnPipeline, matvecResidualPipeline,
      ropePipeline, gqaScoresPipeline, gqaContextPipeline, activationPipeline, elementwisePipeline,
      permutePipeline, matmulQ8Pipeline,
      hiddenA, hiddenB, normedBuf, normed2Buf, qOutBuf, kOutBuf, vOutBuf, qRopedBuf, kRopedBuf, attnOutBuf,
      gatedBuf, finalNormedBuf,
      ropeQUniform, ropeKUniform, gqaScoresUniform, gqaContextUniform,
      ropeQGroup, ropeKGroup, finalNormGroup,
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
   *
   * Routes on `this.tokensSoFar === 0`, not on whether `this.weights` is
   * present — issue #120's `reset()` sets `tokensSoFar` back to `0` without
   * touching `this.weights`, which stays available for exactly this (a
   * second prefill); see both fields' own doc.
   */
  /**
   * `profile` (issue #131): optional, and never read by this method itself —
   * passed straight to whichever of `runPrefillResident`/`runDecodeStep`
   * this call routes to, so a driving script can ask for a per-call
   * breakdown of where this `forward()` call's own wall time went (`
   * ForwardProfile`'s own doc). `undefined` (every call site in this
   * repository outside issue #131's own measurement script) costs nothing:
   * every profiled branch below is behind a truthy check on this same
   * argument.
   */
  async forward(tokens: number[], profile?: ForwardProfile): Promise<Float32Array[]> {
    if (tokens.length === 0) throw new Error("LlamaEngineQ8Resident.forward: tokens must be non-empty");
    if (this.tokensSoFar === 0) return this.runPrefillResident(tokens, false, profile);
    if (tokens.length !== 1) {
      throw new Error("LlamaEngineQ8Resident.forward: after prefill, every call must be exactly one token (decode)");
    }
    return this.runDecodeStep(tokens[0]!, profile);
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
   * prompt), not across calls. Issue #120's `reset()` means "once per
   * generation" no longer means "once per this class instance's lifetime" —
   * this method's own transient allocation and per-layer packed-weight
   * pack-and-upload both re-run, unchanged, on every generation after a `reset()`; see
   * that method's own doc for which cost `reset()` actually removes (it is
   * not this one).
   */
  private async runPrefillResident(tokens: number[], debugAllPositions = false, profile?: ForwardProfile): Promise<Float32Array[]> {
    // Issue #131: whole-call wall time, stopped at this method's very last
    // line (after `this.tokensSoFar`/`this.weights` bookkeeping, so it
    // covers exactly what a caller's own `performance.now()` bracket around
    // `forward()` would have measured).
    const callStart = profile ? performance.now() : 0;
    // PR #126 review, item 3: captured before this call's first `await`, so
    // a `reset()` racing against this call is unambiguously "before" or
    // "after" this snapshot — see `assertSameEpoch`'s own doc.
    const epoch = this.generationEpoch;
    const weights = this.weights;
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
    // Issue #131 items 2-3: every `device.upload`/`device.bindGroup` call
    // below is timed transparently once `profile` is given — see
    // `instrumentDevice`'s own doc. Every other caller (`profile` undefined)
    // gets `this.device` back unchanged, same object, no wrapper allocated.
    const device = profile ? instrumentDevice(this.device, profile) : this.device;
    if (profile) resetForwardProfileOutputs(profile, this.device.timestampsSupported);
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
    // Issue #131 item 4 / PR #141 review item 3: only built when the caller
    // explicitly opted into the GPU breakdown (`ForwardProfile.wantGpuBreakdown`
    // — see that field's own doc for why this is a *second*, separate
    // opt-in from `profile` itself). `undefined` here means `device.batch(...)`
    // below never sees a `labels` array at all, so it batches every dispatch
    // the same single/few-pass way an unprofiled call does — no per-dispatch
    // pass-splitting tax for a caller who only wants the CPU-side fields.
    // Kept parallel to `ops` (one push per `ops.push`, a `copy` entry always
    // `null`) whenever it does exist, same as before.
    const labels: (string | null)[] | undefined = profile?.wantGpuBreakdown ? [] : undefined;
    const dispatch = (
      pipeline: GPUComputePipeline,
      bindGroup: GPUBindGroup,
      workgroups: [number] | [number, number] | [number, number, number],
      label?: string,
    ) => {
      ops.push({ kind: "dispatch", pipeline, bindGroup, workgroups });
      labels?.push(label ?? null);
    };
    const copy = (src: GPUBuffer, srcOffset: number, dst: GPUBuffer, dstOffset: number, size: number) => {
      ops.push({ kind: "copy", src, srcOffset, dst, dstOffset, size });
      labels?.push(null);
    };
    const wg256 = (elements: number) => Math.ceil(elements / 256);
    const matmulWg = (outFeatures: number): [number, number] => [Math.ceil(outFeatures / MATMUL_TILE), Math.ceil(N / MATMUL_TILE)];

    // One `aBuf`/`outBuf`/`uniform` grouping per projection *shape*, reused
    // by every layer — see `MatmulQ8ProjectionShape`'s own doc. Plain
    // construction, not `Promise.all`: unlike pre-#128's `weightTBuf`, there
    // is no GPU buffer or bind group here that needs a device round trip.
    const wqShape = matmulQ8ProjectionShape(qDim, hiddenSize, qMmUniform, normedBuf, qOutBuf);
    const wkShape = matmulQ8ProjectionShape(kvDim, hiddenSize, kMmUniform, normedBuf, kOutBuf);
    const wvShape = matmulQ8ProjectionShape(kvDim, hiddenSize, vMmUniform, normedBuf, vOutBuf);
    const woShape = matmulQ8ProjectionShape(hiddenSize, qDim, oMmUniform, attnTokenMajorBuf, projOutBuf);
    const gateShape = matmulQ8ProjectionShape(ffnHidden, hiddenSize, gateMmUniform, normed2Buf, gateOutBuf);
    const upShape = matmulQ8ProjectionShape(ffnHidden, hiddenSize, upMmUniform, normed2Buf, upOutBuf);
    const downShape = matmulQ8ProjectionShape(hiddenSize, ffnHidden, downMmUniform, gatedBuf, downOutBuf);
    // Every layer's own transient buffers (packed weight/scale per
    // projection, plus the small norm-gain buffers) — `destroy()`ed once
    // `batch()` below has resolved, not left for GC (PR #119 review, item 7,
    // carried over unchanged by issue #128 — the transient-buffer lifetime
    // discipline did not depend on which kernel reads them).
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
      // below is synchronous (pack, upload, `pushErrorScope`,
      // `createBindGroup`, the `popErrorScope()` *call*) right up to its own
      // `await`, so by the time control reaches the next array element the
      // previous one has already pushed and popped its own scope — nesting
      // never overlaps, only the *wait* for each validation result does.
      // Issue #131: wall-clock around the whole concurrent block, not each
      // call inside it — see `ForwardProfile.layerSetupMs`'s own doc for why
      // this is the additive number and `bindGroupMs` (summed per-call) is
      // not.
      const layerSetupStart = profile ? performance.now() : 0;
      const [attnNormGroup, ffnNormGroup, wq, wk, wv, wo, gate, up, down] = await Promise.all([
        device.bindGroup(s.rmsnormPipeline, [hiddenA, attnNormBuf, normedBuf, rmsUniform]),
        device.bindGroup(s.rmsnormPipeline, [hiddenB, ffnNormBuf, normed2Buf, rmsUniform]),
        matmulQ8IntoShape(device, s.matmulQ8Pipeline, wqShape, lw.wq, profile && { sink: profile, layer: l, proj: "wq" }),
        matmulQ8IntoShape(device, s.matmulQ8Pipeline, wkShape, lw.wk, profile && { sink: profile, layer: l, proj: "wk" }),
        matmulQ8IntoShape(device, s.matmulQ8Pipeline, wvShape, lw.wv, profile && { sink: profile, layer: l, proj: "wv" }),
        matmulQ8IntoShape(device, s.matmulQ8Pipeline, woShape, lw.wo, profile && { sink: profile, layer: l, proj: "wo" }),
        matmulQ8IntoShape(device, s.matmulQ8Pipeline, gateShape, lw.wGate, profile && { sink: profile, layer: l, proj: "wGate" }),
        matmulQ8IntoShape(device, s.matmulQ8Pipeline, upShape, lw.wUp, profile && { sink: profile, layer: l, proj: "wUp" }),
        matmulQ8IntoShape(device, s.matmulQ8Pipeline, downShape, lw.wDown, profile && { sink: profile, layer: l, proj: "wDown" }),
      ]);
      if (profile) profile.layerSetupMs += performance.now() - layerSetupStart;
      transientBuffers.push(...wq.transient, ...wk.transient, ...wv.transient, ...wo.transient, ...gate.transient, ...up.transient, ...down.transient);

      dispatch(s.rmsnormPipeline, attnNormGroup, [N], profile && `L${l}:rmsnorm_attn`);
      dispatch(s.matmulQ8Pipeline, wq.bindGroup, matmulWg(qDim), profile && `L${l}:matmulQ8_wq`);
      dispatch(s.matmulQ8Pipeline, wk.bindGroup, matmulWg(kvDim), profile && `L${l}:matmulQ8_wk`);
      dispatch(s.matmulQ8Pipeline, wv.bindGroup, matmulWg(kvDim), profile && `L${l}:matmulQ8_wv`);
      dispatch(s.ropePipeline, ropeQGroup, [wg256((N * qDim) / 2)], profile && `L${l}:rope_q`);
      dispatch(s.ropePipeline, ropeKGroup, [wg256((N * kvDim) / 2)], profile && `L${l}:rope_k`);
      dispatch(s.permutePipeline, splitQGroup, [wg256(N * qDim)], profile && `L${l}:permute_split_q`);
      dispatch(s.permutePipeline, splitKGroup, [wg256(N * kvDim)], profile && `L${l}:permute_split_k`);
      dispatch(s.permutePipeline, splitVGroup, [wg256(N * kvDim)], profile && `L${l}:permute_split_v`);
      dispatch(s.gqaScoresPipeline, gqaScoresGroup, [N, numHeads, 1], profile && `L${l}:gqa_scores`);
      dispatch(s.gqaContextPipeline, gqaContextGroup, [N, numHeads, 1], profile && `L${l}:gqa_context`);
      dispatch(s.permutePipeline, mergeGroup, [wg256(N * qDim)], profile && `L${l}:permute_merge`);
      dispatch(s.matmulQ8Pipeline, wo.bindGroup, matmulWg(hiddenSize), profile && `L${l}:matmulQ8_wo`);
      dispatch(s.elementwisePipeline, add1Group, [wg256(N * hiddenSize)], profile && `L${l}:elementwise_add1`);
      dispatch(s.rmsnormPipeline, ffnNormGroup, [N], profile && `L${l}:rmsnorm_ffn`);
      dispatch(s.matmulQ8Pipeline, gate.bindGroup, matmulWg(ffnHidden), profile && `L${l}:matmulQ8_gate`);
      dispatch(s.matmulQ8Pipeline, up.bindGroup, matmulWg(ffnHidden), profile && `L${l}:matmulQ8_up`);
      dispatch(s.activationPipeline, siluGroup, [wg256(N * ffnHidden)], profile && `L${l}:activation_silu`);
      dispatch(s.elementwisePipeline, mulGroup, [wg256(N * ffnHidden)], profile && `L${l}:elementwise_mul`);
      dispatch(s.matmulQ8Pipeline, down.bindGroup, matmulWg(hiddenSize), profile && `L${l}:matmulQ8_down`);
      dispatch(s.elementwisePipeline, add2Group, [wg256(N * hiddenSize)], profile && `L${l}:elementwise_add2`);

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
    dispatch(s.rmsnormPipeline, finalNormGroup, [N], profile && "final_norm");

    let logits: Float32Array[];
    if (debugAllPositions) {
      // Test/debug-only path — see `debugAllPositionLogits`'s own doc for
      // why this exists and why it is never what `forward()` itself runs.
      // `lm_head` as one more projection shape, `M = N` instead of the 1
      // row `s.finalNormedBuf`/`this.lmHeadChunks` are sized for, reading
      // `finalNormedAllBuf` directly rather than the single-row copy the
      // production path below makes. Its own shape (never reused — this
      // path runs at most once per call).
      const lmHeadMmUniform = uniformOf(device, [["u32", N], ["u32", vocabSize], ["u32", hiddenSize]]);
      const logitsAllBuf = device.createStorageBuffer(N * vocabSize * 4);
      const lmHeadShape = matmulQ8ProjectionShape(vocabSize, hiddenSize, lmHeadMmUniform, finalNormedAllBuf, logitsAllBuf);
      const lmHead = await matmulQ8IntoShape(device, s.matmulQ8Pipeline, lmHeadShape, weights.lmHead, profile && { sink: profile, layer: numLayers, proj: "lm_head" });
      transientBuffers.push(...lmHead.transient);
      dispatch(s.matmulQ8Pipeline, lmHead.bindGroup, matmulWg(vocabSize), profile && "lm_head");

      const stagingAll = device.createStorageBuffer(N * vocabSize * 4, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
      const [allLogitsFlat] = await device.batch(
        ops,
        [{ staging: stagingAll, source: logitsAllBuf, sourceOffset: 0, length: N * vocabSize, type: "f32" }],
        profile && { labels, sink: profile },
      );
      logits = [];
      for (let t = 0; t < N; t += 1) logits.push((allLogitsFlat as Float32Array).slice(t * vocabSize, (t + 1) * vocabSize));
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
      this.lmHeadChunks.forEach((chunk, i) => dispatch(s.matvecPipeline, chunk.group, [chunk.rowCount], profile && `lm_head_chunk${i}`));

      const readback = this.lmHeadChunks.map((chunk) => ({
        staging: chunk.staging, source: chunk.outBuf, sourceOffset: 0, length: chunk.rowCount, type: "f32" as const,
      }));
      const results = await device.batch(ops, readback, profile && { labels, sink: profile });

      const finalLogits = new Float32Array(vocabSize);
      let offset = 0;
      for (const r of results) {
        finalLogits.set(r as Float32Array, offset);
        offset += r.length;
      }
      logits = [finalLogits];
    }

    // PR #119 review, item 7 (carried over by issue #128): every layer's
    // transient packed-weight/scale/norm buffer — `batch()` above has
    // resolved, so the GPU has finished reading all of them; nothing here is
    // referenced by anything this call still needs (the persistent KV cache
    // and `s.finalNormedBuf`/`lmHeadChunks` writes are copies, already
    // landed in buffers this class owns for the long term, not these). No
    // separate `weightTBuf`-style shared buffer to destroy any more — issue
    // #128 removed it; `MatmulQ8ProjectionShape` holds only buffers this
    // class already owned before this call started.
    for (const buf of transientBuffers) buf.destroy();

    this.assertSameEpoch(epoch, debugAllPositions ? "debugAllPositionLogits" : "forward");
    this.tokensSoFar = N;
    // `this.weights` is *not* dropped here (issue #120) — see that field's
    // own doc for why keeping the reference costs nothing and why
    // `reset()` needs it available for a second prefill. Routing now reads
    // `this.tokensSoFar` directly (PR #126 review, item 4), so there is no
    // separate flag to flip.
    if (profile) profile.totalMs = performance.now() - callStart;
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
   * every position via `matmulQ8` (`runPrefillResident`'s own
   * `debugAllPositions` branch), which is exactly the "computing it for
   * every prompt position... undo[es] most of what going resident buys
   * prefill" cost that branch's sibling comment describes; harmless at the
   * tiny fixture's `vocabSize`, not something real generation should do.
   *
   * Same one-call-per-generation contract as `forward()`'s own prefill
   * branch otherwise: valid as this instance's first call, or again after
   * `reset()` (issue #120) — checked against `this.tokensSoFar === 0`, the
   * same state `forward()` itself routes on (PR #126 review, item 4), not
   * against `this.weights` (always present now — see that field's own doc).
   */
  async debugAllPositionLogits(tokens: number[]): Promise<Float32Array[]> {
    if (this.tokensSoFar !== 0) {
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
  private async runDecodeStep(tokenId: number, profile?: ForwardProfile): Promise<Float32Array[]> {
    const callStart = profile ? performance.now() : 0;
    const { numLayers, numKvHeads, headDim, hiddenSize, ffnHidden, numHeads, maxSeqLen, vocabSize } = this.config;
    const at = this.tokensSoFar;
    // PR #126 review, item 3: same reasoning as `runPrefillResident`'s own
    // `epoch` capture — see `assertSameEpoch`'s doc.
    const epoch = this.generationEpoch;
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
    // Issue #131: same instrumented-wrapper opt-in as `runPrefillResident` —
    // see `instrumentDevice`'s own doc. Decode's steady-state loop calls no
    // `device.bindGroup` at all (every group was built once, in `create()` —
    // `LayerResident`'s own doc) and only a handful of small `upload`s
    // (below), so `packEntries`/`bindGroupMs` staying near-zero here, next
    // to prefill's own, is itself part of this issue's answer.
    const device = profile ? instrumentDevice(this.device, profile) : this.device;
    if (profile) resetForwardProfileOutputs(profile, this.device.timestampsSupported);

    const embedVec = gatherDequantRows(this.embedTokens, [tokenId], hiddenSize);
    device.upload(s.hiddenA, 0, embedVec);
    device.upload(s.ropeQUniform, ROPE_POS_OFFSET_BYTE, new Uint32Array([at]));
    device.upload(s.ropeKUniform, ROPE_POS_OFFSET_BYTE, new Uint32Array([at]));
    device.upload(s.gqaScoresUniform, GQA_QUERY_OFFSET_BYTE, new Int32Array([at]));
    // Issue #117: `s_eff = at + 1` — this step's cache write (below) fills
    // position `at`, so positions `0..at` are the whole valid cache, and
    // `ops/gqa`'s scan need not reach past it to `maxSeqLen`. Both uniforms
    // carry it because `scores.wgsl` and `context.wgsl` are separate
    // dispatches reading separate params buffers (see their own docs — they
    // must agree, or `context.wgsl` reads `probs` columns `scores.wgsl` never
    // wrote this step).
    const sEff = new Uint32Array([at + 1]);
    device.upload(s.gqaScoresUniform, GQA_SCORES_S_EFF_BYTE, sEff);
    device.upload(s.gqaContextUniform, GQA_CONTEXT_S_EFF_BYTE, sEff);

    const ops: ResidentOp[] = [];
    // PR #141 review item 3 / `ForwardProfile.wantGpuBreakdown`'s own doc —
    // same gating as `runPrefillResident`'s own `labels`.
    const labels: (string | null)[] | undefined = profile?.wantGpuBreakdown ? [] : undefined;
    const dispatch = (
      pipeline: GPUComputePipeline,
      bindGroup: GPUBindGroup,
      workgroups: [number] | [number, number] | [number, number, number],
      label?: string,
    ) => {
      ops.push({ kind: "dispatch", pipeline, bindGroup, workgroups });
      labels?.push(label ?? null);
    };
    const copy = (src: GPUBuffer, srcOffset: number, dst: GPUBuffer, dstOffset: number, size: number) => {
      ops.push({ kind: "copy", src, srcOffset, dst, dstOffset, size });
      labels?.push(null);
    };

    for (let l = 0; l < numLayers; l += 1) {
      const layer = this.layers[l]!;
      dispatch(s.rmsnormPipeline, layer.attnNormGroup, [1], profile && `L${l}:rmsnorm_attn`);
      dispatch(s.matvecPipeline, layer.wqGroup, [numHeads * headDim], profile && `L${l}:matvecQ8_wq`);
      dispatch(s.matvecPipeline, layer.wkGroup, [numKvHeads * headDim], profile && `L${l}:matvecQ8_wk`);
      dispatch(s.matvecPipeline, layer.wvGroup, [numKvHeads * headDim], profile && `L${l}:matvecQ8_wv`);
      dispatch(s.ropePipeline, s.ropeQGroup, [Math.ceil((numHeads * headDim) / 2 / 256)], profile && `L${l}:rope_q`);
      dispatch(s.ropePipeline, s.ropeKGroup, [Math.ceil((numKvHeads * headDim) / 2 / 256)], profile && `L${l}:rope_k`);
      for (let h = 0; h < numKvHeads; h += 1) {
        copy(s.kRopedBuf, h * headDim * 4, layer.kCacheBuf, (h * maxSeqLen + at) * headDim * 4, headDim * 4);
        copy(s.vOutBuf, h * headDim * 4, layer.vCacheBuf, (h * maxSeqLen + at) * headDim * 4, headDim * 4);
      }
      dispatch(s.gqaScoresPipeline, layer.scoresGroup, [1, numHeads, 1], profile && `L${l}:gqa_scores`);
      dispatch(s.gqaContextPipeline, layer.contextGroup, [1, numHeads, 1], profile && `L${l}:gqa_context`);
      // Issue #111: `hiddenB = hiddenA + wo · attnOutBuf` — one dispatch
      // (`matvecQ8Residual`) in place of the old `matvecQ8(wo)` +
      // `elementwise(add)` pair. See `LayerResident.woGroup`'s own doc.
      dispatch(s.matvecResidualPipeline, layer.woGroup, [hiddenSize], profile && `L${l}:matvecQ8Residual_wo`);
      dispatch(s.rmsnormPipeline, layer.ffnNormGroup, [1], profile && `L${l}:rmsnorm_ffn`);
      // Issue #111: `gatedBuf = silu(wGate · normed2Buf) * (wUp · normed2Buf)`
      // — one dispatch (`matvecQ8Ffn`) in place of the old two `matvecQ8`
      // dispatches plus `activation(silu)` plus `elementwise(multiply)`. See
      // `LayerResident.ffnGroup`'s own doc.
      dispatch(s.matvecFfnPipeline, layer.ffnGroup, [ffnHidden], profile && `L${l}:matvecQ8Ffn`);
      // Issue #111: `hiddenA = hiddenB + wDown · gatedBuf` — same fusion as
      // `woGroup` above. See `LayerResident.downGroup`'s own doc.
      dispatch(s.matvecResidualPipeline, layer.downGroup, [hiddenSize], profile && `L${l}:matvecQ8Residual_down`);
    }
    dispatch(s.rmsnormPipeline, s.finalNormGroup, [1], profile && "final_norm");
    this.lmHeadChunks.forEach((chunk, i) => dispatch(s.matvecPipeline, chunk.group, [chunk.rowCount], profile && `lm_head_chunk${i}`));

    const readback = this.lmHeadChunks.map((chunk) => ({
      staging: chunk.staging, source: chunk.outBuf, sourceOffset: 0, length: chunk.rowCount, type: "f32" as const,
    }));
    const results = await device.batch(ops, readback, profile && { labels, sink: profile });

    const logits = new Float32Array(vocabSize);
    let offset = 0;
    for (const r of results) {
      logits.set(r as Float32Array, offset);
      offset += r.length;
    }

    // PR #126 review, item 3: `+= 1`, not an absolute set — a stale call
    // that slipped past this guard would not merely revert `tokensSoFar` to
    // an old value, it would push a *new* generation's counter one past
    // where it should be, so this check matters even more here than in
    // `runPrefillResident`'s own absolute `this.tokensSoFar = N`.
    this.assertSameEpoch(epoch, "forward");
    this.tokensSoFar += 1;
    if (profile) profile.totalMs = performance.now() - callStart;
    return [logits];
  }
}
