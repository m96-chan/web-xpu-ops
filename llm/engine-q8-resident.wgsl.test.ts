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
 * `matvecQ8`-packed weights and dispatches on one device at once.
 *
 * That "at once" mattered in practice: an earlier version of this file did
 * build a second `LlamaEngineQ8` here (via `runnerFromResident`, issue
 * #110's own single-device fix for the *previous* instability this file
 * hit — see that function's doc), and running both engines' dispatches
 * back to back reliably crashed this repository's Node/Dawn binding
 * (`terminate called after throwing an instance of 'std::system_error'`)
 * partway through the second engine's prefill — the same failure family
 * issue #38/#49/#107 document, this time triggered by total GPU-object
 * count (`LlamaEngineQ8Resident.create()`'s own ~45 pipelines/bind groups
 * for the tiny fixture, plus a second engine's worth on top) rather than by
 * a second native device. `LlamaEngineQ8Resident`'s prefill delegation
 * already exercises `runnerFromResident` end to end on every run below (see
 * `engine-q8-resident.ts#runPrefill`), so dropping the second engine here
 * lost no coverage of that function, only the redundant second dispatch
 * load.
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

    const prefillLogits = await engine.forward(fixture.promptTokens);
    prefillLogits.forEach((got, t) => {
      const worst = agree(got, fixture.prefillLogits[t]!, TOLERANCE);
      expect(worst, worst ? `prefill token ${t}: ${JSON.stringify(worst)}` : undefined).toBeNull();
    });
    const worstPrefill = prefillLogits.reduce(
      (worst, got, t) => {
        const d = worstDiff(got, fixture.prefillLogits[t]!);
        return { abs: Math.max(worst.abs, d.abs), rel: Math.max(worst.rel, d.rel) };
      },
      { abs: 0, rel: 0 },
    );
    console.log("resident int8 prefill worst abs/rel diff:", worstPrefill);

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
