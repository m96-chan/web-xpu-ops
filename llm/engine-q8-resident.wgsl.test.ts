import { beforeAll, describe, expect } from "vitest";
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
 *
 * The two `residentTest`s below that check this fixture (the token-parity
 * gate and the "rejects a second-call multi-token forward" contract) share
 * *one* `LlamaEngineQ8Resident.create()` call, built once in `beforeAll`,
 * for the same "total GPU object count" reason this doc already gives for
 * not building a second engine per test: each `create()` here is ~45
 * pipelines/bind groups, so two of them (one per `residentTest`, the shape
 * an earlier version of this file had) is the same class of avoidable churn
 * as the second-engine case above, not a different one (PR #116 review,
 * item 9). Sharing is safe because the second test's own assertion — "a
 * multi-token `forward()` after prefill throws" — only needs prefill to have
 * already run *once*, not to run again inside that test; the first test's
 * own prefill call already leaves the shared engine in exactly that state by
 * the time the second test runs (vitest runs `it`s inside one `describe`
 * serially, not concurrently, so this ordering is not a race).
 *
 * The maxSeqLen-boundary test at the bottom of this file builds its *own*
 * engine instead of reusing this shared one: it needs a deliberately tiny
 * `maxSeqLen` (PR #116 review, item 1) that the shared fixture's own
 * `config.maxSeqLen` is not, so reusing the shared engine is not an option —
 * this is new coverage, not the redundant duplication the sharing above
 * removes.
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
  const fixture = loadTinyFixtureQ8();
  // Built once, shared by both `residentTest`s below — see this file's
  // module doc for why (PR #116 review, item 9).
  let engine: LlamaEngineQ8Resident | undefined;

  beforeAll(async () => {
    const resident = getResident();
    if (!resident) return; // no adapter — both `residentTest`s below report a skip, same as ever
    engine = await LlamaEngineQ8Resident.create(fixture.config, fixture.weights, resident);
  });

  residentTest("prefill logits agree with the int8 fixture, and decode matches greedily token for token", async () => {
    const prefillLogits = await engine!.forward(fixture.promptTokens);
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
      const [logits] = await engine!.forward([next]);
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
    expect(engine!.position).toBe(fixture.promptTokens.length + fixture.decodeTokens.length);
  });

  residentTest("forward() rejects more than one token after prefill has run", async () => {
    // Relies on the previous test having already run prefill on this same
    // shared `engine` — `forward()`'s own contract only requires prefill to
    // have happened *once* before rejecting a multi-token call (see
    // `engine-q8-resident.ts#forward`), and vitest runs `it`s inside one
    // `describe` serially, so that has always already happened by here.
    await expect(engine!.forward([1, 2])).rejects.toThrow(/exactly one token/);
  });

  /**
   * PR #116 review, item 1 (marked "マージ前必須"): `runDecodeStep` had no
   * `maxSeqLen` boundary check. Once `tokensSoFar` reaches `maxSeqLen`, the
   * KV-cache copy's destination offset (`(h * maxSeqLen + at) * headDim * 4`,
   * `engine-q8-resident.ts#runDecodeStep`) walks into the next head's region
   * (or, for the last head, past the end of the buffer entirely) — a WebGPU
   * validation error with no error scope around it, which invalidates the
   * whole command buffer silently: none of that batch's recorded commands
   * run, including the copy that would refresh the logits staging buffer, so
   * the call *resolves* with the **previous** step's stale logits instead of
   * failing — exactly the "壊れたことに気づけない" failure mode rule 8 exists
   * to keep out, not a crash. `LlamaEngine`/`LlamaEngineQ8`'s own `forward()`
   * already guards this (`posOffset + N > maxSeqLen`, see those files); this
   * engine's decode path was the one place nothing did.
   *
   * Its own engine, not the shared one above: it needs a deliberately tiny
   * `maxSeqLen` (== the fixture's own prompt length, so prefill alone already
   * fills the KV cache to capacity) that the shared engine's config — the
   * fixture's own `maxSeqLen: 32`, far larger than `promptTokens.length +
   * decodeTokens.length` combined — is not. Built from the same shared
   * `resident` device the `describe` block's own `useResidentGpu()` already
   * holds, though, rather than registering a second `useResidentGpu()` in
   * this file: that would reintroduce exactly the extra device
   * create/destroy cycle item 9's engine-sharing above exists to remove
   * (`useResidentGpu`'s own doc — `beforeAll`/`afterAll` are per `describe`,
   * not per file).
   */
  residentTest("runDecodeStep throws once every maxSeqLen position is already resident, instead of returning stale logits", async (resident) => {
    const tightConfig = { ...fixture.config, maxSeqLen: fixture.promptTokens.length };
    const tightEngine = await LlamaEngineQ8Resident.create(tightConfig, fixture.weights, resident);

    // Prefill alone already fills every position this engine has room for
    // (`tokensSoFar === maxSeqLen` immediately after).
    await tightEngine.forward(fixture.promptTokens);
    expect(tightEngine.position).toBe(tightConfig.maxSeqLen);

    await expect(tightEngine.forward([fixture.decodeTokens[0]!])).rejects.toThrow(/maxSeqLen/);
  });
});
