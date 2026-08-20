import { describe, expect } from "vitest";
import { residentTest, useResidentGpu } from "../harness/index.js";
import type { LlamaConfig } from "./config.js";
import { LlamaEngineQ8Resident } from "./engine-q8-resident.js";
import type { LlamaLayerWeightsQ8, LlamaWeightsQ8, QuantizedLinear } from "./weights-q8.js";

/**
 * PR #119 review, items 1 and 2: `runPrefillResident` validated
 * `N <= maxSeqLen` (`config.maxSeqLen`, an architecture choice) but nothing
 * validated `N` against two *WebGPU* ceilings its own `N`-scaled dispatches
 * and buffers can reach first, on a real config, well before `maxSeqLen`
 * does:
 *
 * - `maxComputeWorkgroupsPerDimension` (`llm/kernels.ts#MAX_WORKGROUPS_PER_DISPATCH`,
 *   65535): several of `runPrefillResident`'s one-dimensional dispatches are
 *   sized `ceil(N * featureWidth / 256)` — at Sarashina2.2-1B's own
 *   `ffnHidden = 6272`, that already exceeds 65535 by `N = 2676`, well
 *   under that config's own `maxSeqLen = 4096`.
 * - `maxStorageBufferBindingSize`: `probsBuf` is `numHeads * N * N * 4`
 *   bytes — quadratic in `N`, unlike every other prefill buffer (linear) —
 *   and exceeds the WebGPU spec's guaranteed-supported default (128 MiB) at
 *   `numHeads = 16`, `N = 1449`.
 *
 * Both failure modes are the same shape as issue #106's own bug
 * (`llm/kernels.ts#MAX_WORKGROUPS_PER_DISPATCH`'s doc): a WebGPU validation
 * error outside any error scope invalidates the whole command buffer
 * silently, so the visible symptom is not a thrown error but a `forward()`
 * call that *resolves* with zero/stale logits. `runPrefillResident` now
 * throws before creating any `N`-scaled buffer once `N` would cross either
 * ceiling.
 *
 * A **separate file**, not two more `residentTest`s in
 * `engine-q8-resident.wgsl.test.ts`: `scripts/test.mjs` runs one vitest
 * *process* per file specifically because this repository's Node/Dawn
 * binding does not tolerate accumulating GPU objects across an unbounded
 * number of `LlamaEngineQ8Resident.create()` calls in one process (that
 * file's own module doc, and PR #116 review item 9) — that file already
 * holds two `create()` calls for its own fixture-parity purposes, and these
 * two tests need their *own*, deliberately unusual configs (a huge
 * `ffnHidden`, a huge `numHeads`) that do not fit either existing engine.
 * Keeping them here costs one more process, not two more engines sharing
 * the crash-prone one this repository has already measured.
 */

function tinyLinear(n: number, k: number): QuantizedLinear {
  return { codes: new Int8Array(n * k), scale: new Float32Array(n).fill(0.01) };
}

/** A minimal, all-zero `LlamaWeightsQ8` matching `config`'s shapes exactly — `assertWeightShapesQ8`'s own contract, nothing more. Values are irrelevant: both tests below expect `forward()` to throw before any dispatch this weight feeds ever runs. */
function tinyWeightsFor(config: LlamaConfig): LlamaWeightsQ8 {
  const { hiddenSize, numHeads, numKvHeads, headDim, ffnHidden, vocabSize, numLayers } = config;
  const qDim = numHeads * headDim;
  const kvDim = numKvHeads * headDim;
  const layer = (): LlamaLayerWeightsQ8 => ({
    attnNorm: new Float32Array(hiddenSize),
    wq: tinyLinear(qDim, hiddenSize),
    wk: tinyLinear(kvDim, hiddenSize),
    wv: tinyLinear(kvDim, hiddenSize),
    wo: tinyLinear(hiddenSize, qDim),
    ffnNorm: new Float32Array(hiddenSize),
    wGate: tinyLinear(ffnHidden, hiddenSize),
    wUp: tinyLinear(ffnHidden, hiddenSize),
    wDown: tinyLinear(hiddenSize, ffnHidden),
  });
  return {
    embedTokens: tinyLinear(vocabSize, hiddenSize),
    finalNorm: new Float32Array(hiddenSize),
    lmHead: tinyLinear(vocabSize, hiddenSize),
    layers: Array.from({ length: numLayers }, layer),
  };
}

describe("llm/engine-q8-resident: prefill N-scale guards (PR #119 review, items 1-2)", () => {
  useResidentGpu();

  residentTest("rejects a prompt whose N-scaled dispatch would exceed maxComputeWorkgroupsPerDimension", async (resident) => {
    // ceil(N * ffnHidden / 256) > 65535 at N = 1000, ffnHidden = 20000
    // (65535 * 256 / 20000 ≈ 838.85) — every other N-scaled dimension
    // (qDim = 32, kvDim = 32, hiddenSize = 32) stays far under the same
    // limit at this N, so only this guard can be what rejects the call.
    const config: LlamaConfig = {
      numLayers: 1, hiddenSize: 32, numHeads: 2, numKvHeads: 2, headDim: 16, ffnHidden: 20000,
      vocabSize: 8, ropeTheta: 10000, rmsNormEps: 1e-5, maxSeqLen: 1000, tieEmbeddings: false,
    };
    const weights = tinyWeightsFor(config);
    const engine = await LlamaEngineQ8Resident.create(config, weights, resident);
    const tokens = new Array(1000).fill(0);

    await expect(engine.forward(tokens)).rejects.toThrow(/workgroup/i);
  });

  residentTest("rejects a prompt whose probsBuf would exceed the WebGPU spec's guaranteed maxStorageBufferBindingSize", async (resident) => {
    // numHeads * N * N * 4 > 128 MiB at numHeads = 100, N = 600
    // (100 * 600 * 600 * 4 = 144,000,000 > 134,217,728). Every N-scaled
    // dispatch stays small at this N (qDim = 400, ffnHidden = 64), so the
    // workgroup guard above cannot be what rejects this one.
    const config: LlamaConfig = {
      numLayers: 1, hiddenSize: 32, numHeads: 100, numKvHeads: 100, headDim: 4, ffnHidden: 64,
      vocabSize: 8, ropeTheta: 10000, rmsNormEps: 1e-5, maxSeqLen: 700, tieEmbeddings: false,
    };
    const weights = tinyWeightsFor(config);
    const engine = await LlamaEngineQ8Resident.create(config, weights, resident);
    const tokens = new Array(600).fill(0);

    await expect(engine.forward(tokens)).rejects.toThrow(/storage buffer|maxStorageBufferBindingSize/i);
  });
});
