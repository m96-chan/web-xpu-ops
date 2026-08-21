import { beforeAll, describe, expect } from "vitest";
import { residentTest, useResidentGpu } from "../harness/index.js";
import { argmax } from "./engine.js";
import { LlamaEngineQ8Resident } from "./engine-q8-resident.js";
import { loadTinyFixtureQ8 } from "./fixture-q8.js";

/**
 * Issue #142's own two correctness/perf gates, kept in one file (a separate
 * file, not more `residentTest`s in `engine-q8-resident.wgsl.test.ts`/
 * `.reset.wgsl.test.ts` — same "total GPU object count" reasoning those two
 * files' own module docs already give, PR #116 review item 9): one shared
 * engine, built once in `beforeAll`, exercised by both tests below in order.
 *
 * ## Gate 1: bit-for-bit logits — an in-session A/B diff, not a golden file
 *
 * `runPrefillResident`'s `matmulQ8` bind group used to be built from a
 * *fresh* `packInt8Rows(linear.codes, ...)` on every `forward()` call
 * (`matmulQ8IntoShape`, pre-#142). Issue #142 replaced that with
 * `bindMatmulQ8`, which binds the exact `GPUBuffer`s `buildProjection`
 * uploaded once in `create()` for decode's own `matvecQ8` bind groups —
 * `engine-q8-resident.ts#bindMatmulQ8`'s own doc has the full argument for
 * why the bytes are identical either way. Not the fixture's own
 * Python-derived `prefillLogits`/`decodeLogits` (`engine-q8-resident.wgsl.test.ts`'s
 * own gate, which allows `TOLERANCE = {rel: 1e-2, abs: 5e-3}` because that
 * comparison crosses a reference-implementation boundary, PyTorch vs this
 * repository's own WGSL) — this file's own comparison does not cross any
 * such boundary, so `toEqual` (no tolerance) is the right bar, per #130's
 * own established criterion: "1ビットでも動いたら丸めではなく設計の違い"
 * (issue #142's own background).
 *
 * An earlier version of this test asserted that against a **golden JSON
 * fixture** — a literal capture of this exact fixture's prefill/decode
 * logits, taken once from this repository's own `matmulQ8IntoShape`-based
 * code on one machine (`git stash` back to the commit immediately before
 * issue #142's own changes). Review caught the problem with that shape:
 * prefill runs through `rmsnorm`'s `rsqrt`, `rope`'s `sin`/`cos`, and
 * `gqa`'s softmax `exp` — all vendor/driver-dependent at the ULP level
 * (rule 2, "GPU の挙動は特に推測しない"), so a literal-float fixture is only
 * guaranteed reproducible **on the machine that captured it**, not across
 * adapters, and not necessarily across a driver update on the *same*
 * machine either. `engine-q8-resident.wgsl.test.ts`'s own sibling gate
 * already accounts for exactly this by comparing against the Python
 * fixture with a tolerance, not exact equality — pinning a *second*,
 * stricter, single-machine-only fixture next to it was inconsistent with
 * that.
 *
 * `debugPrefillWithPackedWeights` (`engine-q8-resident.ts`, issue #142,
 * test/debug-only) is the fix: it runs the exact same prefill through the
 * pre-#142 `matmulQ8IntoShape` path instead of `bindMatmulQ8`, on the
 * *same* engine instance, in the *same* test, on the *same* device. The
 * claim issue #142 actually needs — "these two paths compute the same
 * thing" — only requires the two to agree with **each other**, on whatever
 * this session's own device happens to compute; it never needs a value
 * that survives being written to disk and read back on a different
 * machine. An in-session diff proves exactly that claim and nothing more,
 * which is also exactly what portability requires.
 *
 * ## Gate 2: no new GPU resource per prefill call
 *
 * `matmulQ8IntoShape` used to call `device.createStorageBuffer` twice
 * (weight, scale) per projection per layer, every `forward()` call that ran
 * prefill — `7` projections (`wq`/`wk`/`wv`/`wo`/gate/up/`wDown`) `× 2`
 * buffers `× numLayers`, `28` extra buffers for this fixture's `numLayers:
 * 2`. `bindMatmulQ8` calls neither `createStorageBuffer` nor `upload` at
 * all — it only builds a bind group against buffers `create()` already
 * made. `resident.stats.buffersCreated`'s own doc names exactly this
 * property ("プリフィルが新規GPUリソースを確保しない") as something a test can hold
 * the class honest to, the same way `harness/resident.test.ts`'s own
 * "buffers created before batch() are not recreated by repeated batches"
 * test holds `runDecodeStep`'s steady state honest. This one *is* a plain
 * integer count, not a float — no ULP/vendor portability concern, so a
 * measured numeric threshold is the right shape here, unlike Gate 1 above.
 *
 * `THRESHOLD` below is a **measured** bound, not a guess (rule 2): running
 * this fixture's own prefill through the pre-#142 `matmulQ8IntoShape`
 * path (`git stash` to the commit immediately before this file's own
 * changes) measured `stats.buffersCreated` increasing by **75** per prefill
 * call; the post-#142 `bindMatmulQ8` path measures **47** — the same `28`
 * (`2 × 7 × numLayers`) difference the paragraph above predicts. `THRESHOLD
 * = 60` sits strictly between the two: comfortably above 47 (room for
 * unrelated, legitimate scratch-buffer additions to `runPrefillResident` in
 * a future change) and comfortably below 75 (so a regression that
 * reintroduces per-call weight packing — reverting `bindMatmulQ8` back to
 * `matmulQ8IntoShape` for the seven production projections — fails this
 * assertion, confirmed by temporarily making exactly that revert during
 * this test's own development and watching this test go red).
 */
const THRESHOLD = 60;

describe("llm/engine-q8-resident / issue #142 — resident weight buffers in prefill", () => {
  const getResident = useResidentGpu();
  const fixture = loadTinyFixtureQ8();

  let engine: LlamaEngineQ8Resident | undefined;

  beforeAll(async () => {
    const resident = getResident();
    if (!resident) return;
    engine = await LlamaEngineQ8Resident.create(fixture.config, fixture.weights, resident);
  });

  residentTest("prefill and decode logits are bit-for-bit identical whether matmulQ8's weight comes from bindMatmulQ8 (resident) or matmulQ8IntoShape (packed) — same session, same device", async () => {
    // First generation on the shared engine: the pre-#142 packed-weight
    // path (`debugPrefillWithPackedWeights`'s own doc — test/debug-only,
    // never reachable from `forward()`). `tokensSoFar` starts at 0 straight
    // out of `beforeAll`'s `create()`, so this is a valid first prefill.
    const packedPrefill = await engine!.debugPrefillWithPackedWeights(fixture.promptTokens);
    const packedDecode: Float32Array[] = [];
    let next = argmax(packedPrefill[0]!);
    for (let s = 0; s < fixture.decodeTokens.length; s += 1) {
      // eslint-disable-next-line no-await-in-loop
      const [logits] = await engine!.forward([next]);
      packedDecode.push(logits!);
      next = argmax(logits!);
    }

    // Second, independent generation on the same engine instance (issue
    // #120's `reset()`), same tokens: the production resident-weight path.
    engine!.reset();
    const residentPrefill = await engine!.forward(fixture.promptTokens);
    const residentDecode: Float32Array[] = [];
    next = argmax(residentPrefill[0]!);
    for (let s = 0; s < fixture.decodeTokens.length; s += 1) {
      // eslint-disable-next-line no-await-in-loop
      const [logits] = await engine!.forward([next]);
      residentDecode.push(logits!);
      next = argmax(logits!);
    }

    // Bit-for-bit, not merely within tolerance — see this file's own "Gate
    // 1" doc above for why exact equality is the right bar here.
    expect(Array.from(residentPrefill[0]!)).toEqual(Array.from(packedPrefill[0]!));
    residentDecode.forEach((got, s) => {
      expect(Array.from(got)).toEqual(Array.from(packedDecode[s]!));
    });
  });

  residentTest("a second prefill (after reset()) allocates well under the pre-#142 per-call weight-repack cost", async (resident) => {
    // Relies on the previous test having already left this shared engine
    // mid-generation (its own resident-path decode loop) — `reset()` here
    // starts a third, independent generation on the same instance (issue
    // #120), which is exactly the case issue #142's own background names
    // ("再パック済みバイト列は create()時点で既にGPU上に存在する" — true on
    // *every* prefill call, not just the first).
    engine!.reset();
    const before = resident.stats.buffersCreated;
    await engine!.forward(fixture.promptTokens);
    const after = resident.stats.buffersCreated;
    const delta = after - before;
    expect(delta, `prefill created ${delta} new GPU buffers — expected well under ${THRESHOLD} (pre-#142 measured 75, post-#142 measured 47 for this fixture)`).toBeLessThan(THRESHOLD);
  });
});
