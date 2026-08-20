import { describe, expect } from "vitest";
import { agree, gpuTest, useGpu } from "../harness/index.js";
import { argmax, LlamaEngine } from "./engine.js";
import { loadTinyFixture } from "./fixture.js";

/**
 * The whole point of issue #98: a config-driven llama forward pass, run end to
 * end on a real GPU (Dawn, through `webgpu`), agrees with `transformers`'
 * output on the same weights and the same tokens — logits to a measured
 * tolerance, and greedy-decoded tokens exactly.
 *
 * `llm/kernels.wgsl.test.ts` already checked each op wrapper in isolation
 * against its own reference; this file is what those wrappers add up to.
 *
 * Every `forward` call runs back to back, with no assertion or diagnostic
 * work between them — issue #38 / #49's lesson (`harness/suite.ts`) applies
 * here just as it does inside a single op's test: CPU time between dispatches
 * destabilises this binding, and a prefill call followed immediately by
 * `agree()` over 2048 floats, *then* the first decode dispatch, was measured
 * to make the run past prefill unreliable. All five `forward` calls (prefill
 * plus four greedy decode steps) happen first; every comparison against the
 * fixture happens only once every dispatch this test needs is done.
 */

const fixture = loadTinyFixture();

/**
 * Measured, not assumed (rule 9) — printed by this test every run (see
 * `worstPrefill` / `worstDecode` below). Observed on this device:
 *
 *   prefill (8 tokens, 2 layers, GQA + RoPE + fused-QKV/gate-up matmul path):
 *     abs 1.49e-7   rel 2.98e-4
 *   decode (4 greedy steps, same weights, matvec path):
 *     abs 9.69e-8   rel 3.43e-4
 *
 * Smaller than any *individual* op's own tolerance in this repo (`gqa`'s is
 * `abs 2e-6`, `matmul`'s is `abs 2e-5`, `rope`'s uncached path is `abs 1e-3`)
 * even though this engine chains all of them across two layers — plausible
 * rather than suspicious: `agree()` reports the worst single element, and a
 * tiny random model's logits do not accumulate error the way a trained
 * model's sharper distributions might. `rel 1e-2` / `abs 5e-3` below is
 * roughly 30x the worst `rel` and four orders of magnitude past the worst
 * `abs` actually seen — wide enough that ordinary run-to-run GPU transcendental
 * jitter (documented throughout `ops/rope`, `ops/gqa`) cannot cross it, tight
 * enough that a real bug (wrong head, wrong offset, wrong scale) — which reads
 * as `rel` order 1, not order 1e-4 — still fails loudly. Not widened to make
 * this pass; the passing tolerance came out far tighter than the guess this
 * value started as.
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

describe("llm/engine / fixture parity", () => {
  useGpu();

  gpuTest("prefill logits agree with the transformers fixture, and decode matches greedily token for token", async (run) => {
    const engine = new LlamaEngine(fixture.config, fixture.weights, run);

    // --- Every dispatch, back to back, nothing else in between ---
    const prefillLogits = await engine.forward(fixture.promptTokens);
    const decodeTokens: number[] = [];
    const decodeLogits: Float32Array[] = [];
    let next = argmax(prefillLogits[prefillLogits.length - 1]!);
    for (let s = 0; s < fixture.decodeTokens.length; s += 1) {
      const [logits] = await engine.forward([next]);
      decodeLogits.push(logits!);
      next = argmax(logits!);
      decodeTokens.push(next);
    }

    // --- Every comparison, only now that the device has nothing left to do ---
    const worstPrefill = prefillLogits.reduce(
      (worst, got, t) => {
        const d = worstDiff(got, fixture.prefillLogits[t]!);
        return { abs: Math.max(worst.abs, d.abs), rel: Math.max(worst.rel, d.rel) };
      },
      { abs: 0, rel: 0 },
    );
    console.log("prefill worst abs/rel diff:", worstPrefill);
    prefillLogits.forEach((got, t) => {
      const worst = agree(got, fixture.prefillLogits[t]!, TOLERANCE);
      expect(worst, worst ? `prefill token ${t}: ${JSON.stringify(worst)}` : undefined).toBeNull();
    });

    const worstDecode = decodeLogits.reduce(
      (worst, got, t) => {
        const d = worstDiff(got, fixture.decodeLogits[t]!);
        return { abs: Math.max(worst.abs, d.abs), rel: Math.max(worst.rel, d.rel) };
      },
      { abs: 0, rel: 0 },
    );
    console.log("decode worst abs/rel diff:", worstDecode);
    decodeLogits.forEach((got, t) => {
      const worst = agree(got, fixture.decodeLogits[t]!, TOLERANCE);
      expect(worst, worst ? `decode step ${t}: ${JSON.stringify(worst)}` : undefined).toBeNull();
    });

    expect(decodeTokens).toEqual(fixture.decodeTokens);
  });
});
