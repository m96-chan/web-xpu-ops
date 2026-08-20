import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { describe, expect } from "vitest";
import { gpuTest, useGpu } from "../harness/index.js";
import { argmax } from "./engine.js";
import { LlamaEngineQ8 } from "./engine-q8.js";
import { loadConvertedWeightsQ8 } from "./real-model-weights.js";

/**
 * Issue #105's real-model validation: loads the checkpoint
 * `llm/tools/convert_weights.py` converted, prefills a prompt encoded by
 * `llm/tools/encode_prompt.py` (tokenizer wiring is a separate, not-yet-merged
 * issue — #101 — so token IDs come in pre-encoded rather than from a
 * tokenizer this engine owns), greedily decodes `DECODE_STEPS` tokens, and
 * writes the result (token IDs, per-step timing) to a JSON file for offline
 * comparison against `llama-cli` and inspection of the generated Japanese —
 * neither of which this file asserts on, since issue #105's own scope
 * explicitly does not require exact token-for-token agreement with a
 * different (block-quantized, Q8_0) implementation.
 *
 * Skips — does not fail — when the converted checkpoint or the encoded
 * prompt is not present, the same posture `harness/suite.ts#useGpu` takes
 * for a missing GPU adapter: both live outside this repository (issue #105's
 * own instructions place the converted weights in a sibling repo's
 * `third_party/`, gitignored there) and are produced by a manual conversion
 * run, not something CI has.
 *
 * ## A known failure mode on some machines: issue #107
 *
 * On the machine issue #105 was implemented on, this test's first real GPU
 * dispatch reliably crashes this repository's Node+Dawn (`webgpu@0.4.x`)
 * binding — root-caused (see #107) to CPU-bound work of real-model scale
 * (loading + packing a ~1.4 GiB checkpoint, on the order of a second) done
 * in-process before a dispatch, **not** GPU contention, buffer count/size, or
 * a bug in `LlamaEngineQ8` itself (which is verified correct on the tiny
 * fixture — `engine-q8.wgsl.test.ts` — to a tight tolerance). A dedicated
 * repro isolated the trigger to elapsed CPU-bound work alone (confirmed with
 * an unrelated large allocation *and*, separately, a pure compute busy-loop
 * with no allocation at all) and found no workaround inside this
 * architecture (warm-up dispatch, chunking the CPU work with event-loop
 * yields, and deferring device creation until after all the CPU work was
 * done were all tried and none avoided it) — so it is filed as its own issue
 * rather than worked around here. If this test fails with a Dawn thread-pool
 * assertion (`pthread_mutex_lock`, `futex`, `std::system_error`) rather than
 * a normal test assertion, that is #107, not a regression in this file.
 *
 * Repeats the decode loop `REPEATS` times with a **fresh** `LlamaEngineQ8`
 * each time (a fresh KV cache; construction re-packs the same resident
 * weights, which does not change the outcome, only the wall time spent
 * getting there) — issue #105's "GPU共有デスクトップなのでブレは複数回計測で
 * 報告": tok/s is measured this many times so the report can show the spread,
 * not a single number nothing can be compared against.
 */

const REAL_MODEL_Q8_DIR = process.env["ALIBI_SARASHINA_Q8_DIR"]
  ?? "/home/m96-chan/project/technologies.moe/alibi-ai/third_party/webgpu-weights/sarashina2.2-1b-alibi-v1-q8";
const PROMPT_TOKENS_PATH = process.env["ALIBI_PROMPT_TOKENS_JSON"] ?? "";
const RESULT_OUT_PATH = process.env["ALIBI_REAL_MODEL_RESULT_JSON"] ?? "";
const DECODE_STEPS = Number(process.env["ALIBI_DECODE_STEPS"] ?? 20);
const REPEATS = Number(process.env["ALIBI_DECODE_REPEATS"] ?? 1);
const MAX_SEQ_LEN = 4096;

const present = existsSync(`${REAL_MODEL_Q8_DIR}/manifest.json`) && PROMPT_TOKENS_PATH !== "" && existsSync(PROMPT_TOKENS_PATH);

interface RunResult {
  decodeTokens: number[];
  prefillMs: number;
  stepMs: number[];
  /** tok/s over every decode step but the first (warm-up: first step pays for whatever the GPU binding was not yet primed for). */
  tokPerSecExcludingFirstStep: number;
}

// skipIf, not an early return inside the body: `return` reports a green PASS
// with zero assertions run, and this repository has already been bitten by
// exactly that (gpuTest's own `if (!current) return;`). A skip is visible in
// the summary; a vacuous pass tells #106's implementer the real-model gate
// ran when it never did.
describe.skipIf(!present)("LlamaEngineQ8 / real Sarashina2.2-1B-alibi-v1 generation", () => {
  useGpu();

  gpuTest("prefill + greedy decode on the real converted checkpoint", async (run) => {
    const { config, weights } = loadConvertedWeightsQ8(REAL_MODEL_Q8_DIR, MAX_SEQ_LEN);
    const promptData = JSON.parse(readFileSync(PROMPT_TOKENS_PATH, "utf8")) as { promptText: string; tokenIds: number[] };
    const promptTokens = promptData.tokenIds;

    const runs: RunResult[] = [];
    for (let r = 0; r < REPEATS; r += 1) {
      const engine = new LlamaEngineQ8(config, weights, run);

      const prefillStart = performance.now();
      const prefillLogits = await engine.forward(promptTokens);
      const prefillMs = performance.now() - prefillStart;

      let next = argmax(prefillLogits[prefillLogits.length - 1]!);
      const decodeTokens: number[] = [];
      const stepMs: number[] = [];
      for (let s = 0; s < DECODE_STEPS; s += 1) {
        const stepStart = performance.now();
        const [logits] = await engine.forward([next]);
        stepMs.push(performance.now() - stepStart);
        decodeTokens.push(next);
        next = argmax(logits!);
      }

      const warm = stepMs.slice(1);
      const avgWarmMs = warm.length > 0 ? warm.reduce((a, b) => a + b, 0) / warm.length : NaN;
      runs.push({ decodeTokens, prefillMs, stepMs, tokPerSecExcludingFirstStep: 1000 / avgWarmMs });
      console.log(`run ${r + 1}/${REPEATS}: prefill ${prefillMs.toFixed(1)}ms, decode tok/s (warm) ${(1000 / avgWarmMs).toFixed(2)}`);
      console.log(`run ${r + 1}/${REPEATS}: decode tokens`, decodeTokens);

      // Baseline sanity, not a correctness check (see module doc for why
      // exact token agreement is out of scope): every generated id is a real
      // vocabulary entry, and greedy decode always produces exactly the
      // requested number of steps.
      expect(decodeTokens).toHaveLength(DECODE_STEPS);
      for (const t of decodeTokens) {
        expect(t).toBeGreaterThanOrEqual(0);
        expect(t).toBeLessThan(config.vocabSize);
      }
    }

    if (RESULT_OUT_PATH) {
      writeFileSync(RESULT_OUT_PATH, JSON.stringify({ promptText: promptData.promptText, promptTokens, runs }, null, 2));
      console.log(`wrote ${RESULT_OUT_PATH}`);
    }
  });
});
