import { describe, expect } from "vitest";
import { agree, residentTest, useResidentGpu } from "../harness/index.js";
import { argmax } from "./engine.js";
import { LlamaEngineQ8Resident } from "./engine-q8-resident.js";
import { loadTinyFixtureQ8 } from "./fixture-q8.js";

/**
 * Issue #120: `reset()` — a resident engine's weights, pipelines, bind
 * groups and KV/activation buffers survive an independent second (third,
 * ...) generation, instead of every new prompt needing a whole new
 * `ResidentDevice` + `LlamaEngineQ8Resident.create()` (alibi-ai's own
 * chat-integration measurement: 17-33s/round trip, dominated by the 1.4 GiB
 * weight re-upload a fresh `create()` pays — issue #120's own background).
 *
 * A **separate file**, not more `residentTest`s in
 * `engine-q8-resident.wgsl.test.ts` or `.limits.wgsl.test.ts`: the same
 * "total GPU object count, not per-file case count" reason those two files'
 * own docs already give (issue #38/#49/#107's failure family, PR #116
 * review item 9, the `ops/gqa` split in PR #119's second review round) —
 * this file's own single `create()` call already shares a `describe` with
 * nothing else that would add more.
 */

const TOLERANCE = { rel: 1e-2, abs: 5e-3 };

/** Runs the fixture's own prompt through prefill, then every one of its
 * decode tokens, asserting greedy-token and logit parity against the
 * fixture at every step — the same shape `engine-q8-resident.wgsl.test.ts`'s
 * "greedy tokens match the pre-optimization engine" test uses, extracted so
 * this file can run it twice against the *same* engine instance (once
 * before any `reset()`, once after) without duplicating the body. */
async function runFixtureGenerationAndAssert(engine: LlamaEngineQ8Resident, fixture: ReturnType<typeof loadTinyFixtureQ8>): Promise<void> {
  const prefillLogits = await engine.forward(fixture.promptTokens);
  expect(prefillLogits).toHaveLength(1);
  const fixtureLastPrefill = fixture.prefillLogits[fixture.prefillLogits.length - 1]!;
  const prefillWorst = agree(prefillLogits[0]!, fixtureLastPrefill, TOLERANCE);
  expect(prefillWorst, prefillWorst ? `prefill (final position): ${JSON.stringify(prefillWorst)}` : undefined).toBeNull();

  const decodeTokens: number[] = [];
  const decodeLogits: Float32Array[] = [];
  let next = argmax(prefillLogits[0]!);
  for (let s = 0; s < fixture.decodeTokens.length; s += 1) {
    // eslint-disable-next-line no-await-in-loop
    const [logits] = await engine.forward([next]);
    decodeLogits.push(logits!);
    next = argmax(logits!);
    decodeTokens.push(next);
  }
  decodeLogits.forEach((got, t) => {
    const worst = agree(got, fixture.decodeLogits[t]!, TOLERANCE);
    expect(worst, worst ? `decode step ${t}: ${JSON.stringify(worst)}` : undefined).toBeNull();
  });
  expect(decodeTokens).toEqual(fixture.decodeTokens);
}

describe("llm/engine-q8-resident / reset() (issue #120)", () => {
  const getResident = useResidentGpu();
  const fixture = loadTinyFixtureQ8();

  /**
   * The correctness proof issue #120 itself asks for: "旧KVに毒値を書いた状態で
   * reset→新プロンプト実行→フレッシュ構築のエンジンと出力完全一致".
   *
   * Rather than building a *second* engine as the "fresh" comparison target
   * (the total-GPU-object-count instability this file's own module doc
   * warns about), the fixture's own `prefillLogits`/`decodeLogits`/
   * `decodeTokens` — already the ground truth every other engine test in
   * this directory compares against — stand in for "what a freshly built
   * engine would produce", since they were generated independently
   * (`llm/tools/gen_fixture_q8.py`) and do not depend on this engine's own
   * history at all. Running the *exact same* fixture prompt through the
   * *same* engine instance twice, with a `reset()` and a full-buffer
   * `+Infinity` poison in between, and getting the fixture's numbers back
   * unchanged the second time, is the same proof: nothing left over from
   * generation 1 (least of all `+Infinity`) reached generation 2's output.
   *
   * `debugPoisonKVCache` overwrites *every* KV cache slot, not just the ones
   * beyond the second generation's own final position — prefill's own copy
   * into the cache (`runPrefillResident`) only ever *writes*, never *reads*,
   * so poisoning the slots it is about to overwrite is harmless, and doing
   * it anyway is simpler than computing which slots the second run will
   * reach in advance. The slots that matter are the ones *beyond* the
   * second generation's final position (`fixture.promptTokens.length +
   * fixture.decodeTokens.length`, same length as the first, so this test
   * alone does not distinguish "same-length reset" from "no reset at all" —
   * see the second test below for a *shorter* second generation, which
   * does): `context.wgsl` reads `v` unconditionally within its `sEff`
   * bound (`ops/gqa/wgsl-seff.test.ts`'s own doc has the full mechanism),
   * so a `reset()` that failed to bound the second generation's `sEff` to
   * *its own* position — reusing the first generation's `tokensSoFar`
   * instead of the fresh `0`, say — reads `+Infinity` and returns `NaN`,
   * which cannot agree with the fixture's finite numbers under any
   * tolerance.
   */
  residentTest("reset() + the same prompt reproduces the fixture exactly, even with the old KV cache poisoned with +Infinity", async (resident) => {
    const engine = await LlamaEngineQ8Resident.create(fixture.config, fixture.weights, resident);

    await runFixtureGenerationAndAssert(engine, fixture);
    const firstGenPosition = engine.position;
    expect(firstGenPosition).toBe(fixture.promptTokens.length + fixture.decodeTokens.length);

    engine.debugPoisonKVCache(Number.POSITIVE_INFINITY);
    engine.reset();
    expect(engine.position).toBe(0);

    await runFixtureGenerationAndAssert(engine, fixture);
    expect(engine.position).toBe(fixture.promptTokens.length + fixture.decodeTokens.length);
  });

  /**
   * The same proof, but with the second generation *shorter* than the
   * first (prefill only, no decode steps) — so real, un-poisoned positions
   * from generation 1 (`fixture.promptTokens.length` .. `firstGenPosition -
   * 1`, genuine fixture-derived numbers, not `+Infinity`) sit *beyond*
   * generation 2's own final position and must equally not be read. The
   * first test alone cannot catch a `reset()` that forgot to shrink `sEff`
   * back down and instead kept scanning out to the *previous* generation's
   * length: same-length-after-reset would still, by coincidence, read only
   * positions this generation itself just wrote. A shorter second
   * generation does not have that coincidence protecting it.
   */
  residentTest("reset() + a shorter second generation does not read the first generation's leftover positions", async (resident) => {
    const engine = await LlamaEngineQ8Resident.create(fixture.config, fixture.weights, resident);

    await runFixtureGenerationAndAssert(engine, fixture);
    expect(engine.position).toBe(fixture.promptTokens.length + fixture.decodeTokens.length);

    engine.reset();
    const prefillLogits = await engine.forward(fixture.promptTokens);
    expect(prefillLogits).toHaveLength(1);
    const fixtureLastPrefill = fixture.prefillLogits[fixture.prefillLogits.length - 1]!;
    const worst = agree(prefillLogits[0]!, fixtureLastPrefill, TOLERANCE);
    expect(worst, worst ? `prefill (final position) after reset: ${JSON.stringify(worst)}` : undefined).toBeNull();
    expect(engine.position).toBe(fixture.promptTokens.length);
  });

  /**
   * `reset()` puts the engine back in exactly `create()`'s own initial
   * state as far as `forward()`'s own routing can tell: the next call is
   * prefill again (accepts more than one token), not decode (which would
   * reject it — `forward()`'s own `tokens.length !== 1` check, the same
   * contract `engine-q8-resident.wgsl.test.ts`'s "forward() rejects more
   * than one token after prefill has run" test pins for a *fresh* engine).
   * A `reset()` that failed to flip that internal flag back would route this
   * multi-token call into `runDecodeStep` instead, which asserts on `tokens
   * === 1` positionally rather than checking `tokens.length` at all (it
   * reads `tokenId` from `tokens[0]` alone) — the wrong branch would not
   * throw *this* message, or might not throw at all, silently decoding only
   * the prompt's first token and discarding the rest.
   */
  residentTest("reset() re-arms prefill routing: a multi-token forward() after reset() is accepted, not rejected as decode", async (resident) => {
    const engine = await LlamaEngineQ8Resident.create(fixture.config, fixture.weights, resident);
    await engine.forward(fixture.promptTokens);
    await expect(engine.forward([1, 2])).rejects.toThrow(/exactly one token/);

    engine.reset();
    // No throw: routed back into prefill, which accepts any non-empty
    // token list, unlike decode's "exactly one token" contract just above.
    await expect(engine.forward(fixture.promptTokens)).resolves.toHaveLength(1);
  });

  /**
   * `reset()` reuses every persistent GPU object `create()` already built —
   * no second `pipelineFor` call anywhere on a `forward()` or `reset()`
   * path (confirmed by reading `runPrefillResident`/`runDecodeStep`: both
   * only ever reference `this.shared`'s pipelines, built once in `create()`).
   * `stats.pipelinesCreated` is the direct, structural proof: unchanged
   * across a full second generation, the same "decode-loop invariant" shape
   * `harness/resident.test.ts`'s own test of that name already established
   * for a single generation's repeated decode steps, extended here across a
   * `reset()` boundary.
   *
   * `stats.buffersCreated` is deliberately *not* asserted flat here —
   * `runPrefillResident`'s own doc is explicit that its buffers are
   * transient by design (freshly sized by `N`, `destroy()`ed at the end of
   * that same call), so a second prefill call creating more of them is
   * expected, correct behaviour, not evidence of anything `reset()` failed
   * to do. What must not grow is the *pipeline* count, and what must not
   * happen at all — not measurable via `stats`, but true by construction,
   * since `reset()`'s own implementation touches only `tokensSoFar` and one
   * boolean — is a second `ResidentDevice` or a second `create()` call.
   */
  residentTest("reset() creates no new pipeline — every dispatch after it still runs through create()'s own pipelines", async (resident) => {
    const engine = await LlamaEngineQ8Resident.create(fixture.config, fixture.weights, resident);
    await runFixtureGenerationAndAssert(engine, fixture);

    const pipelinesBefore = resident.stats.pipelinesCreated;
    engine.reset();
    await runFixtureGenerationAndAssert(engine, fixture);
    expect(resident.stats.pipelinesCreated).toBe(pipelinesBefore);
  });
});
