import { describe, expect } from "vitest";
import { agree, residentTest, useResidentGpu } from "../harness/index.js";
import { argmax } from "./engine.js";
import { LlamaEngineQ8Resident } from "./engine-q8-resident.js";
import { loadTinyFixtureQ8 } from "./fixture-q8.js";

/**
 * Issue #110's correctness gate: "最適化前エンジンと greedy トークン列完全一致".
 *
 * Compared against the tiny fixture's own `prefillLogits`/`decodeLogits`/
 * `decodeTokens` (`llm/tools/gen_fixture_q8.py`'s output) rather than
 * against a second, live `LlamaEngineQ8` instance built inside this same
 * test — not a weaker check, a **transitive** one:
 * `engine-q8.wgsl.test.ts` already proves `LlamaEngineQ8`'s own greedy
 * tokens equal `fixture.decodeTokens`, on this exact device family, so this
 * file proving `LlamaEngineQ8Resident`'s tokens equal `fixture.decodeTokens`
 * establishes the same "matches the pre-optimization engine" fact the issue
 * asks for, without constructing two independent engines' worth of
 * `matvecQ8`-packed weights and dispatches on one device at once. An
 * earlier version of this file did build a second `LlamaEngineQ8` here
 * (via `runnerFromResident`, `harness/resident.ts`), and running both
 * engines' dispatches back to back reliably crashed this repository's
 * Node/Dawn binding (`terminate called after throwing an instance of
 * 'std::system_error'`) partway through the second engine's prefill — the
 * same failure family issue #38/#49/#107 document, this time triggered by
 * total GPU-object count (`LlamaEngineQ8Resident.create()`'s own ~45
 * pipelines/bind groups for the tiny fixture, plus a second engine's worth
 * on top) rather than by a second native device.
 *
 * Issue #117 removed the delegation to `LlamaEngineQ8` entirely — prefill
 * is resident now (`engine-q8-resident.ts#runPrefillResident`) — so the
 * instability this file's history describes no longer applies to prefill
 * specifically, but the fixture-comparison shape (rather than a live
 * second engine) is kept regardless: it is still the more direct
 * "matches the pre-optimization engine" proof, one engine, one device.
 *
 * `forward()`'s prefill return shape changed with #117 too: **one**
 * element (the final prompt position's logits), not one per prompt token —
 * see `engine-q8-resident.ts#forward`'s own doc. This file's prefill check
 * compares that one element against `fixture.prefillLogits`'s own last
 * element, the same position.
 */

const TOLERANCE = { rel: 1e-2, abs: 5e-3 };

function worstDiff(got: Float32Array, want: Float32Array): { abs: number; rel: number } {
  let worstAbs = 0;
  let worstRel = 0;
  for (let i = 0; i < got.length; i += 1) {
    const abs = Math.abs(got[i]! - want[i]!);
    const rel = abs / Math.max(Math.abs(want[i]!), 1e-6);
    if (abs > worstAbs) worstAbs = abs;
    if (rel > worstRel) worstRel = rel;
  }
  return { abs: worstAbs, rel: worstRel };
}

describe("llm/engine-q8-resident / greedy tokens match the pre-optimization engine's own fixture", () => {
  const getResident = useResidentGpu();

  residentTest("prefill logits agree with the int8 fixture, and decode matches greedily token for token", async (resident) => {
    const fixture = loadTinyFixtureQ8();
    const engine = await LlamaEngineQ8Resident.create(fixture.config, fixture.weights, resident);

    // Issue #117: prefill returns exactly one element — the final prompt
    // position's logits (see this file's own doc, and
    // `engine-q8-resident.ts#forward`'s). Compared against the fixture's own
    // last prefill position, the same one.
    const prefillLogits = await engine.forward(fixture.promptTokens);
    expect(prefillLogits).toHaveLength(1);
    const fixtureLastPrefill = fixture.prefillLogits[fixture.prefillLogits.length - 1]!;
    const worst = agree(prefillLogits[0]!, fixtureLastPrefill, TOLERANCE);
    expect(worst, worst ? `prefill (final position): ${JSON.stringify(worst)}` : undefined).toBeNull();
    console.log("resident int8 prefill worst abs/rel diff:", worstDiff(prefillLogits[0]!, fixtureLastPrefill));

    const decodeTokens: number[] = [];
    const decodeLogits: Float32Array[] = [];
    let next = argmax(prefillLogits[prefillLogits.length - 1]!);
    for (let s = 0; s < fixture.decodeTokens.length; s += 1) {
      // eslint-disable-next-line no-await-in-loop
      const [logits] = await engine.forward([next]);
      decodeLogits.push(logits!);
      next = argmax(logits!);
      decodeTokens.push(next);
    }

    const worstDecode = decodeLogits.reduce(
      (worst, got, t) => {
        const d = worstDiff(got, fixture.decodeLogits[t]!);
        return { abs: Math.max(worst.abs, d.abs), rel: Math.max(worst.rel, d.rel) };
      },
      { abs: 0, rel: 0 },
    );
    console.log("resident int8 decode worst abs/rel diff:", worstDecode);
    decodeLogits.forEach((got, t) => {
      const worst = agree(got, fixture.decodeLogits[t]!, TOLERANCE);
      expect(worst, worst ? `decode step ${t}: ${JSON.stringify(worst)}` : undefined).toBeNull();
    });

    expect(decodeTokens).toEqual(fixture.decodeTokens);
    expect(engine.position).toBe(fixture.promptTokens.length + fixture.decodeTokens.length);
  });

  residentTest("forward() rejects more than one token after prefill has run", async (resident) => {
    const fixture = loadTinyFixtureQ8();
    const engine = await LlamaEngineQ8Resident.create(fixture.config, fixture.weights, resident);
    await engine.forward(fixture.promptTokens);

    await expect(engine.forward([1, 2])).rejects.toThrow(/exactly one token/);
  });
});
