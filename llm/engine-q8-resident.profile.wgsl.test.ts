import { beforeAll, describe, expect } from "vitest";
import { agree, residentTest, useResidentGpu } from "../harness/index.js";
import { createForwardProfile, LlamaEngineQ8Resident, type ForwardProfile } from "./engine-q8-resident.js";
import { loadTinyFixtureQ8 } from "./fixture-q8.js";

/**
 * Issue #131: `ForwardProfile` is an opt-in breakdown of one `forward()`
 * call's own wall time — CPU pack (`packInt8Rows`), upload (`queue.writeBuffer`),
 * bind-group creation, GPU submit-to-completion wait, readback, and (when
 * both `timestamp-query` is negotiated *and* the caller opts into
 * `wantGpuBreakdown`) per-dispatch GPU time. These tests gate two things a
 * real-model measurement cannot check for itself: that asking for a profile
 * does not change what `forward()` computes (issue #131's own scope is
 * measurement-only — a profiler that perturbs the answer would be exactly
 * the kind of "積分よりバグ" this repository's rule 8 exists to catch), and
 * that the profile's own shape matches what `runPrefillResident`/
 * `runDecodeStep` actually do (per layer/projection counts, tied to the tiny
 * fixture's own `numLayers=2`, `vocabSize=256` config — see `config.ts`).
 *
 * PR #141 review: `wantGpuBreakdown` defaults `false` (`createForwardProfile()`'s
 * own default), and only the two tests below that actually read `gpuEntries`
 * set it `true` — every other test leaves it off, so their own `totalMs`/
 * `layerSetupMs` reflect the cheap, production-shaped single/few-pass
 * batching an unprofiled call would also use, not the per-dispatch
 * pass-splitting `wantGpuBreakdown: true` costs (`ForwardProfile`'s own doc
 * has the full reasoning — an earlier version of this file did not have
 * this distinction, and its own `packInt8Rows`-alone numbers came out
 * larger than an unrelated PR's entirely unprofiled prefill total).
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
    // Issue #131 review: every `residentTest` below starts with `reset()`
    // (issue #120), unconditionally — the same "one shared engine, every
    // test resets first" pattern `reset.wgsl.test.ts` establishes, so no
    // test here depends on running immediately after another one in a
    // specific order. A no-op on this first test (the engine is already at
    // `tokensSoFar = 0` straight out of `beforeAll`'s `create()`), but kept
    // for the same reason `reset.wgsl.test.ts`'s own first test keeps it:
    // consistency beats relying on position-in-file.
    engine!.reset();
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

  /**
   * Issue #142: `runPrefillResident`'s seven per-layer projections
   * (`wq`/`wk`/`wv`/`wo`/gate/up/`wDown`) no longer go through
   * `matmulQ8IntoShape` — `bindMatmulQ8` binds `this.layers[l]`'s own
   * resident `ResidentWeight` (built once in `create()`) directly, with no
   * `packInt8Rows` call and no `device.upload` for the weight/scale bytes.
   * `profile.packEntries` (issue #131's own per-`packInt8Rows`-call log)
   * must therefore be **empty** for a plain (non-debug) prefill call —
   * `lm_head`'s debug-only `matmulQ8IntoShape` call
   * (`debugAllPositionLogits`, `runPrefillResident`'s own
   * `debugAllPositions` branch) is the only remaining caller, and this test
   * never reaches it (it calls `forward()`, not `debugAllPositionLogits`).
   *
   * This test used to assert the *opposite* — `numLayers * 7` pack
   * entries, one per projection per layer — because it predates issue #142
   * (PR #141/issue #131 landed first, measuring `matmulQ8IntoShape`'s own
   * per-call repack cost before #142 removed it). Left un-updated, that
   * old assertion is not a false negative here: it started failing exactly
   * because #142 does what it claims (`expected [] to have a length of 14
   * but got +0` — the pre-#142 assertion, reproduced during this PR's own
   * review before this rewrite). Renamed and rewritten to assert the
   * post-#142 invariant directly, so a regression that reintroduces
   * per-call packing (reverting `bindMatmulQ8` back to `matmulQ8IntoShape`
   * for these seven) fails *this* test again — the same "observation point
   * must break when the behavior it names breaks" property
   * `engine-q8-resident.residentweight.wgsl.test.ts`'s own
   * `buffersCreated` assertion already holds for buffer allocation; this is
   * the same property for CPU pack time and transfer bytes. Confirmed by
   * mutation: temporarily reverting the seven `bindMatmulQ8` calls back to
   * `matmulQ8IntoShape` during this rewrite measured `packEntries.length
   * === 14` and `uploadBytes === 81544` (matching the pre-#142 shape
   * exactly — `2 layers × 7 projections`, and the packed weight+scale byte
   * total for this fixture's tiny shapes); reverting the mutation restored
   * `packEntries: []` and `uploadBytes: 3720`.
   */
  residentTest("prefill's profile reports zero packEntries — issue #142's resident weight buffers, not a per-call repack", async () => {
    // Issue #131 review: the previous test already ran this shared `engine`
    // through one prefill (`forward(fixture.promptTokens, ...)`), so without
    // `reset()` (issue #120) this call would route into `runDecodeStep`
    // (`tokensSoFar !== 0`) and be rejected by #126's own "after prefill,
    // every call must be exactly one token" guard — reproduced and confirmed
    // by running this file as a whole (not one `it` at a time), the same
    // fresh-engine-per-generation pattern `reset.wgsl.test.ts` establishes.
    engine!.reset();
    const profile = createForwardProfile();
    await engine!.forward(fixture.promptTokens, profile);

    const { numLayers } = fixture.config;
    // See this test's own doc above — issue #142's own claim, asserted
    // directly rather than merely narrated.
    expect(profile.packEntries).toEqual([]);

    // `uploadBytes` still counts the small per-layer norm-gain buffers
    // (`attnNormBuf`/`ffnNormBuf`, `hiddenSize` floats each) and the
    // embedding-row/final-norm uploads `runPrefillResident` always does —
    // so `> 0` alone would not distinguish "no weight upload" from "weight
    // upload still happening, plus these". `UPLOAD_BYTES_WEIGHT_FREE_CEILING`
    // is the mutation-tested bound instead: measured 3720 bytes with
    // `bindMatmulQ8` (this fixture, `numLayers: 2`), 81544 bytes with the
    // pre-#142 `matmulQ8IntoShape` mutation (see this test's own doc) — the
    // ceiling sits strictly between the two, with headroom on the resident
    // side for unrelated future norm-buffer additions and a wide margin
    // below the weight-upload side (~22x the resident measurement).
    const UPLOAD_BYTES_WEIGHT_FREE_CEILING = 20_000;
    expect(profile.uploadBytes).toBeGreaterThan(0);
    expect(profile.uploadBytes).toBeLessThan(UPLOAD_BYTES_WEIGHT_FREE_CEILING);
    expect(profile.uploadMs).toBeGreaterThanOrEqual(0);
    // Nine bind groups per layer (`attnNormGroup`, `ffnNormGroup`, plus the
    // seven `bindMatmulQ8` calls' own internal `device.bindGroup` — no
    // pack/upload ahead of them any more, issue #142) —
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

  residentTest("prefill's GPU breakdown, when opted into and timestamp-query is supported, has at most one entry per labeled dispatch", async () => {
    // Same reset()-before-reprefill requirement as the previous test — see
    // its own comment.
    engine!.reset();
    const resident = getResident()!;
    const profile = createForwardProfile();
    // Explicit opt-in (PR #141 review) — without this, `labels` is never
    // built at all and `gpuEntries` stays `[]` regardless of hardware
    // support (`ForwardProfile.wantGpuBreakdown`'s own doc).
    profile.wantGpuBreakdown = true;
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
    // `MAX_WORKGROUPS_PER_DISPATCH`). An *upper* bound, not exact equality:
    // `harness/resident.ts#batch` drops any query pair whose measured delta
    // is exactly zero (the driver declining to serve it — that function's
    // own doc), which is a real, hardware-dependent possibility for the
    // smallest dispatches in this tiny fixture, not a bug this test should
    // fail on (PR #141 review, item 4 — this exact assertion was flagged as
    // liable to go red by chance for that reason).
    const expectedMax = numLayers * 21 + 1 + 1;
    expect(profile.gpuEntries.length).toBeGreaterThan(0);
    expect(profile.gpuEntries.length).toBeLessThanOrEqual(expectedMax);
    const labels = new Set(profile.gpuEntries.map((e) => e.label));
    // `matmulQ8` is the one kernel family this issue's own measurement finds
    // GPU-dominant among (`README`'s "Where prefill's ~1.2s..." section) —
    // its dispatches are the least likely of any label here to measure an
    // exact-zero delta, so asserting its presence (rather than every label)
    // keeps this robust against the same zero-filtering noted above.
    expect(labels.has("L0:matmulQ8_wq")).toBe(true);
    for (const entry of profile.gpuEntries) expect(entry.seconds).toBeGreaterThan(0);
  });

  residentTest("decode's profile shows zero bind-group cost, unlike prefill's own — packEntries is empty for both, as of issue #142", async () => {
    // Self-contained, not relying on a preceding test's own side effect
    // (issue #131 review) — `reset()` (issue #120) then an *unprofiled*
    // prefill to reach a real decode-eligible position (`tokensSoFar > 0`);
    // the profile under test below is only the one decode step that
    // follows. A `reset()` immediately followed by a single-token
    // `forward()` would *not* exercise decode at all — `tokensSoFar === 0`
    // still routes a 1-token call into prefill (`forward()`'s own doc), so
    // this needs a real multi-token prefill first, the same reason
    // `reset.wgsl.test.ts`'s own decode-touching tests all re-prefill after
    // `reset()` rather than jumping straight to a single-token call.
    engine!.reset();
    await engine!.forward(fixture.promptTokens);

    const profile = createForwardProfile();
    profile.wantGpuBreakdown = true;
    await engine!.forward([fixture.decodeTokens[0]!], profile);

    // `runDecodeStep` never calls `matmulQ8IntoShape` — every weight buffer
    // it reads was packed once, in `create()`'s own `buildProjection` calls,
    // not per `forward()` call (this file's class doc, "Prefill is resident
    // too" section, and this issue's own review comment: the same bytes
    // `buildProjection` already uploaded for decode are what issue #131
    // itself is asking whether prefill could reuse). As of issue #142,
    // prefill's own production path reached the identical conclusion for
    // its seven per-layer projections (`bindMatmulQ8` reuses those same
    // resident buffers instead of calling `matmulQ8IntoShape`) — so this
    // assertion is no longer decode-specific; the sibling test above
    // ("prefill's profile reports zero packEntries") checks the same thing
    // for prefill. `ForwardProfile.packEntries`'s own doc has the one
    // remaining exception (`debugAllPositionLogits`'s debug-only path).
    expect(profile.packEntries).toEqual([]);
    // Steady-state decode binds no new bind groups either (`LayerResident`'s
    // own doc) — every group used below was built once in `create()`. This
    // *is* still a real contrast with prefill (`bindGroupCalls` stays at
    // `numLayers * 9`+ there, issue #142 does not change that — only
    // whether those bind groups sit behind a fresh pack/upload or a
    // resident buffer).
    expect(profile.bindGroupCalls).toBe(0);
    expect(profile.bindGroupMs).toBe(0);
    expect(profile.layerSetupMs).toBe(0);
    // Decode still uploads a handful of small uniforms/the embedding row
    // every step (`s.hiddenA`, both rope position uniforms, both `sEff`
    // uniforms) — non-zero, but tiny. Pre-#142 this was also tiny next to
    // prefill's own per-layer weight re-upload (~1 GiB at real-model scale);
    // post-#142, prefill's own `uploadBytes` is tiny too (issue #142's own
    // point), so this is no longer a decode-vs-prefill contrast, just a
    // sanity check that decode still uploads *something* every step.
    expect(profile.uploadBytes).toBeGreaterThan(0);
    expect(profile.submitToDoneMs).not.toBeNull();
    expect(profile.readbackMs).not.toBeNull();
    expect(profile.totalMs).toBeGreaterThan(0);

    if (profile.gpuTimestampsSupported) {
      const { numLayers } = fixture.config;
      // 12 labeled dispatches per layer (`runDecodeStep`'s own doc: "12→per
      // layer") + final_norm + 1 lm_head chunk — an upper bound, not exact
      // equality, for the same zero-delta-filtering reason the prefill GPU
      // breakdown test above documents.
      expect(profile.gpuEntries.length).toBeGreaterThan(0);
      expect(profile.gpuEntries.length).toBeLessThanOrEqual(numLayers * 12 + 1 + 1);
    }
  });
});
