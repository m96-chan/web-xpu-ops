import { describe, expect } from "vitest";
import { agree, gpuTest, useGpu } from "../harness/index.js";
import { argmax } from "./engine.js";
import { LlamaEngineQ8 } from "./engine-q8.js";
import { loadTinyFixtureQ8 } from "./fixture-q8.js";

/**
 * Issue #105's int8 counterpart to `engine.wgsl.test.ts`: `LlamaEngineQ8`,
 * run end to end on a real GPU, agrees with a `transformers` forward pass run
 * on the **same int8-quantized-then-dequantized weights**
 * (`llm/tools/gen_fixture_q8.py`) — not the f32-exact fixture `engine.wgsl.test.ts`
 * checks against, because that would be comparing a genuinely quantized
 * engine's output to a reference that never quantized anything.
 *
 * Same "every dispatch first, every assertion after" shape as
 * `engine.wgsl.test.ts`, for the same reason (issue #38/#49, `harness/suite.ts`):
 * this binding degrades when CPU work runs between dispatches on this device.
 */

const fixture = loadTinyFixtureQ8();

/**
 * Measured (rule 9), printed every run below. int8 quantization error is
 * already baked into `fixture.prefillLogits`/`decodeLogits` on the Python
 * side — this tolerance covers only the *residual* disagreement between the
 * TS engine's arithmetic and the Python reference's, not the quantization
 * error itself, so it starts from `engine.wgsl.test.ts`'s own f32 tolerance
 * (`rel 1e-2, abs 5e-3`) rather than a wider one chosen to make this pass.
 * Widened only if a measured worst diff needs it, and only by as much as
 * that measurement shows (documented here once observed).
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

describe("llm/engine-q8 / int8 fixture parity", () => {
  useGpu();

  gpuTest("prefill logits agree with the int8-quantized transformers fixture, and decode matches greedily token for token", async (run) => {
    const engine = new LlamaEngineQ8(fixture.config, fixture.weights, run);

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

    const worstPrefill = prefillLogits.reduce(
      (worst, got, t) => {
        const d = worstDiff(got, fixture.prefillLogits[t]!);
        return { abs: Math.max(worst.abs, d.abs), rel: Math.max(worst.rel, d.rel) };
      },
      { abs: 0, rel: 0 },
    );
    console.log("int8 prefill worst abs/rel diff:", worstPrefill);
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
    console.log("int8 decode worst abs/rel diff:", worstDecode);
    decodeLogits.forEach((got, t) => {
      const worst = agree(got, fixture.decodeLogits[t]!, TOLERANCE);
      expect(worst, worst ? `decode step ${t}: ${JSON.stringify(worst)}` : undefined).toBeNull();
    });

    expect(decodeTokens).toEqual(fixture.decodeTokens);
  });
});
