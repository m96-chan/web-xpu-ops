import { readFileSync } from "node:fs";
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
 * ## Gate 1: bit-for-bit logits
 *
 * `runPrefillResident`'s `matmulQ8` bind group used to be built from a
 * *fresh* `packInt8Rows(linear.codes, ...)` on every `forward()` call
 * (`matmulQ8IntoShape`, pre-#142). Issue #142 replaced that with
 * `bindMatmulQ8`, which binds the exact `GPUBuffer`s `buildProjection`
 * uploaded once in `create()` for decode's own `matvecQ8` bind groups —
 * `engine-q8-resident.ts#bindMatmulQ8`'s own doc has the full argument for
 * why the bytes are identical either way. `tiny_q8.resident_prefill_golden.json`
 * is a literal capture of this exact fixture's prefill + 4-step decode
 * logits, taken from this repository's `matmulQ8IntoShape`-based code
 * (`git stash` back to the commit immediately before issue #142's own
 * changes, same tiny fixture, same token sequence) — not the fixture's own
 * Python-derived `prefillLogits`/`decodeLogits` (`engine-q8-resident.wgsl.test.ts`'s
 * own gate, which allows `TOLERANCE = {rel: 1e-2, abs: 5e-3}` because that
 * comparison crosses a reference-implementation boundary, PyTorch vs this
 * repository's own WGSL). This file's own comparison does not cross any such
 * boundary — both sides are this same `LlamaEngineQ8Resident` reading the
 * same packed int8 bytes through the same `matmulQ8` kernel, only *which
 * buffer* holds those bytes differs — so equality is asserted exactly
 * (`toEqual`, no tolerance), per #130's own established criterion: "1ビット
 * でも動いたら丸めではなく設計の違い" (issue #142's own background).
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
 * test holds `runDecodeStep`'s steady state honest.
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
  const golden = JSON.parse(
    readFileSync(new URL("./fixtures/tiny_q8.resident_prefill_golden.json", import.meta.url), "utf8"),
  ) as { prefillLogits: number[]; decodeLogits: number[][] };

  let engine: LlamaEngineQ8Resident | undefined;

  beforeAll(async () => {
    const resident = getResident();
    if (!resident) return;
    engine = await LlamaEngineQ8Resident.create(fixture.config, fixture.weights, resident);
  });

  residentTest("prefill and decode logits are bit-for-bit identical to the pre-#142 (matmulQ8IntoShape) capture", async () => {
    const prefillLogits = await engine!.forward(fixture.promptTokens);
    expect(Array.from(prefillLogits[0]!)).toEqual(golden.prefillLogits);

    let next = argmax(prefillLogits[0]!);
    for (let s = 0; s < fixture.decodeTokens.length; s += 1) {
      // eslint-disable-next-line no-await-in-loop
      const [logits] = await engine!.forward([next]);
      expect(Array.from(logits!)).toEqual(golden.decodeLogits[s]);
      next = argmax(logits!);
    }
  });

  residentTest("a second prefill (after reset()) allocates well under the pre-#142 per-call weight-repack cost", async (resident) => {
    // Relies on the previous test having already run this shared engine's
    // first prefill + decode — `reset()` here starts a second, independent
    // generation on the same instance (issue #120), which is exactly the
    // case issue #142's own background names ("再パック済みバイト列は
    // create()時点で既にGPU上に存在する" — true on *every* prefill call, not
    // just the first).
    engine!.reset();
    const before = resident.stats.buffersCreated;
    await engine!.forward(fixture.promptTokens);
    const after = resident.stats.buffersCreated;
    const delta = after - before;
    expect(delta, `prefill created ${delta} new GPU buffers — expected well under ${THRESHOLD} (pre-#142 measured 75, post-#142 measured 47 for this fixture)`).toBeLessThan(THRESHOLD);
  });
});
