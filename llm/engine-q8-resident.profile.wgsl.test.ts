import { beforeAll, describe, expect } from "vitest";
import { agree, residentTest, useResidentGpu } from "../harness/index.js";
import { createForwardProfile, LlamaEngineQ8Resident, type ForwardProfile } from "./engine-q8-resident.js";
import { loadTinyFixtureQ8 } from "./fixture-q8.js";

/**
 * Issue #131: `ForwardProfile` is an opt-in breakdown of one `forward()`
 * call's own wall time — CPU pack (`packInt8Rows`), upload (`queue.writeBuffer`),
 * bind-group creation, GPU submit-to-completion wait, readback, and (when
 * `timestamp-query` is negotiated) per-dispatch GPU time. These tests gate
 * two things a real-model measurement cannot check for itself: that asking
 * for a profile does not change what `forward()` computes (issue #131's own
 * scope is measurement-only — a profiler that perturbs the answer would be
 * exactly the kind of "積分よりバグ" this repository's rule 8 exists to catch),
 * and that the profile's own shape matches what `runPrefillResident`/
 * `runDecodeStep` actually do (per layer/projection counts, tied to the tiny
 * fixture's own `numLayers=2`, `vocabSize=256` config — see `config.ts`).
 */
describe("llm/engine-q8-resident / ForwardProfile (issue #131)", () => {
  const getResident = useResidentGpu();
  const fixture = loadTinyFixtureQ8();
  let engine: LlamaEngineQ8Resident | undefined;

  beforeAll(async () => {
    const resident = getResident();
    if (!resident) return;
    engine = await LlamaEngineQ8Resident.create(fixture.config, fixture.weights, resident);
  });

  residentTest("a profiled prefill produces identical logits to an unprofiled one", async () => {
    const unprofiled = await LlamaEngineQ8Resident.create(fixture.config, fixture.weights, getResident()!);
    const unprofiledLogits = await unprofiled.forward(fixture.promptTokens);

    const profile = createForwardProfile();
    const profiledLogits = await engine!.forward(fixture.promptTokens, profile);

    // Bit-for-bit, not merely within tolerance: profiling adds no arithmetic
    // to the dispatch chain (labels only decide pass boundaries in
    // `batch()`), so the computed values must be exactly the same, the
    // stronger check the coordinator's own #130 finding (prefill/decode
    // logits are bit-identical across a structurally-equivalent change)
    // establishes as the right bar here.
    expect(Array.from(profiledLogits[0]!)).toEqual(Array.from(unprofiledLogits[0]!));
  });

  residentTest("prefill's profile reports one packEntries row per layer per matmulQ8 projection", async () => {
    const profile = createForwardProfile();
    await engine!.forward(fixture.promptTokens, profile);

    const { numLayers } = fixture.config;
    // wq, wk, wv, wo, wGate, wUp, wDown — `runPrefillResident`'s own seven
    // `matmulQ8IntoShape` calls per layer (production path; `lm_head` only
    // packs through `matmulQ8IntoShape` in the debug-only `debugAllPositionLogits`
    // path, not here).
    expect(profile.packEntries).toHaveLength(numLayers * 7);
    const projByLayer = new Map<number, Set<string>>();
    for (const entry of profile.packEntries) {
      expect(entry.ms).toBeGreaterThanOrEqual(0);
      expect(entry.bytes).toBeGreaterThan(0);
      if (!projByLayer.has(entry.layer)) projByLayer.set(entry.layer, new Set());
      projByLayer.get(entry.layer)!.add(entry.proj);
    }
    expect([...projByLayer.keys()].sort((a, b) => a - b)).toEqual(Array.from({ length: numLayers }, (_, i) => i));
    for (const projs of projByLayer.values()) {
      expect([...projs].sort()).toEqual(["wDown", "wGate", "wUp", "wk", "wo", "wq", "wv"]);
    }

    expect(profile.uploadBytes).toBeGreaterThan(0);
    expect(profile.uploadMs).toBeGreaterThanOrEqual(0);
    // Nine bind groups per layer (`attnNormGroup`, `ffnNormGroup`, plus the
    // seven `matmulQ8IntoShape` calls' own internal `device.bindGroup`) —
    // issue #131 item 3's own "24 rounds" description at Sarashina2.2-1B's
    // 24-layer scale, `numLayers` here instead of 24 for the tiny fixture —
    // plus a fixed number of once-per-call groups built outside the loop
    // (rope x2, permute split x3, gqa x2, merge, four elementwise/activation
    // groups, final norm): at least `numLayers * 9` of the total.
    expect(profile.bindGroupCalls).toBeGreaterThanOrEqual(numLayers * 9);
    expect(profile.bindGroupMs).toBeGreaterThanOrEqual(0);
    // Wall-clock, unlike `bindGroupMs` (see that field's own doc) — must be
    // strictly less than `totalMs` since it is one phase inside the call,
    // not a sum of overlapping individual awaits.
    expect(profile.layerSetupMs).toBeGreaterThan(0);
    expect(profile.layerSetupMs).toBeLessThan(profile.totalMs);
    expect(profile.submitToDoneMs).not.toBeNull();
    expect(profile.submitToDoneMs).toBeGreaterThanOrEqual(0);
    expect(profile.readbackMs).not.toBeNull();
    expect(profile.readbackMs).toBeGreaterThanOrEqual(0);
    expect(profile.totalMs).toBeGreaterThan(0);
  });

  residentTest("prefill's GPU breakdown, when timestamp-query is supported, has one entry per labeled dispatch", async () => {
    const resident = getResident()!;
    const profile = createForwardProfile();
    await engine!.forward(fixture.promptTokens, profile);

    expect(profile.gpuTimestampsSupported).toBe(resident.timestampsSupported);
    if (!resident.timestampsSupported) {
      expect(profile.gpuEntries).toEqual([]);
      return;
    }
    const { numLayers } = fixture.config;
    // 21 labeled dispatches per layer (see `runPrefillResident`'s own
    // `dispatch(...)` call sites) + 1 final_norm + 1 lm_head chunk (the tiny
    // fixture's vocabSize=256 fits in a single `lmHeadChunks` entry — see
    // `MAX_WORKGROUPS_PER_DISPATCH`).
    expect(profile.gpuEntries.length).toBe(numLayers * 21 + 1 + 1);
    const labels = new Set(profile.gpuEntries.map((e) => e.label));
    expect(labels.has("L0:matmulQ8_wq")).toBe(true);
    expect(labels.has("final_norm")).toBe(true);
    expect(labels.has("lm_head_chunk0")).toBe(true);
    for (const entry of profile.gpuEntries) expect(entry.seconds).toBeGreaterThan(0);
  });

  residentTest("decode's profile shows near-zero CPU pack/bindGroup cost, unlike prefill's own", async () => {
    // Relies on the prefill test above having already run on this shared
    // `engine` (vitest runs `it`s in one `describe` serially) — decode
    // needs `position > 0`.
    const profile = createForwardProfile();
    await engine!.forward([fixture.decodeTokens[0]!], profile);

    // `runDecodeStep` never calls `matmulQ8IntoShape` — every weight buffer
    // it reads was packed once, in `create()`'s own `buildProjection` calls,
    // not per `forward()` call (this file's class doc, "Prefill is resident
    // too" section, and this issue's own review comment: the same bytes
    // `buildProjection` already uploaded for decode are what issue #131
    // itself is asking whether prefill could reuse).
    expect(profile.packEntries).toEqual([]);
    // Steady-state decode binds no new bind groups either (`LayerResident`'s
    // own doc) — every group used below was built once in `create()`.
    expect(profile.bindGroupCalls).toBe(0);
    expect(profile.bindGroupMs).toBe(0);
    expect(profile.layerSetupMs).toBe(0);
    // Decode still uploads a handful of small uniforms/the embedding row
    // every step (`s.hiddenA`, both rope position uniforms, both `sEff`
    // uniforms) — non-zero, but tiny next to prefill's per-layer weight
    // re-upload.
    expect(profile.uploadBytes).toBeGreaterThan(0);
    expect(profile.submitToDoneMs).not.toBeNull();
    expect(profile.readbackMs).not.toBeNull();
    expect(profile.totalMs).toBeGreaterThan(0);

    if (profile.gpuTimestampsSupported) {
      const { numLayers } = fixture.config;
      // 12 labeled dispatches per layer (`runDecodeStep`'s own doc: "12→per
      // layer") + final_norm + 1 lm_head chunk.
      expect(profile.gpuEntries.length).toBe(numLayers * 12 + 1 + 1);
    }
  });
});
