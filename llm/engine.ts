import type { Runner } from "../harness/index.js";
import { ACTIVATION } from "../ops/activation/index.js";
import { ELEMENTWISE } from "../ops/elementwise/index.js";
import type { LlamaConfig } from "./config.js";
import { runActivation, runElementwise, runGather, runGqa, runMatMul, runMatVec, runRmsNorm, runRope } from "./kernels.js";
import { KVCache } from "./kv-cache.js";
import { concatRows, mergeHeadsMajor, splitConcatRows, splitHeadsMajor, transposeRowMajor } from "./reshape.js";
import { assertWeightShapes, type LlamaLayerWeights, type LlamaWeights } from "./weights.js";

/**
 * A layer's weights, pre-processed once at construction so `forward` does no
 * host-side matrix algebra of its own:
 *
 * - `wQkv` / `wGateUp` are Q+K+V and gate+up **fused** — `[Wq; Wk; Wv]` and
 *   `[Wgate; Wup]` stacked row-wise (`reshape.ts#concatRows`) — so one
 *   `matvec`/`matmul` dispatch computes what would otherwise be three (or
 *   two) separate ones. `splitConcatRows` un-fuses the output afterwards.
 *   This is a dispatch-count reduction, not a numerical approximation — see
 *   `concatRows`'s own doc for why it is exact — and it is *load-bearing*
 *   rather than an optional speedup: `llm/engine.wgsl.test.ts` documents that
 *   this environment's `webgpu` binding cannot sustain the ~195 dispatches an
 *   unfused prefill-then-4-step-decode run needs, so fusing down to ~124 is
 *   part of getting this engine testable at all here, not a rule-9 violation.
 * - `*T` fields are `matmul`'s `[in, out]` operand order (`reshape.ts#transposeRowMajor`),
 *   for the prefill path; the unfused `[out, in]` weight already serves `matvec`
 *   for decode, so `wQkv`/`wGateUp` are transposed too, once, rather than per call.
 */
interface LayerWeights {
  wQkv: Float32Array;
  wQkvT: Float32Array;
  woT: Float32Array;
  wGateUp: Float32Array;
  wGateUpT: Float32Array;
  wDownT: Float32Array;
}

/**
 * A llama-architecture decoder, config-driven per issue #98: every dimension
 * comes from `LlamaConfig`, and the same code runs Sarashina2.2-1B's shape
 * (once #96's weight converter and tokenizer exist) as the tiny fixture this
 * issue tests against.
 *
 * ## The forward pass, per layer
 *
 * `rmsnorm → QKV projection (fused) → rope(Q, K) → gqa(+KV cache) → O
 * projection → residual → rmsnorm → gate/up projection (fused) → silu(gate) *
 * up → down projection → residual`, then one final `rmsnorm → lm_head` after
 * the last layer.
 *
 * Every arrow is a GPU dispatch through `llm/kernels.ts`; the handful of
 * non-arrows (`llm/reshape.ts`'s transposes and fuse/split helpers, `KVCache`'s
 * compaction) are host-side relabelling, not computed — see their own files
 * for why that is a deliberate scope choice (rule 8) and not a shortcut.
 *
 * ## Prefill vs. decode
 *
 * `forward` takes however many new tokens are given and does not care whether
 * that is 8 (prefill) or 1 (decode): `project` — the shared helper behind
 * every fused-QKV/O/fused-gate-up/down/lm_head projection — picks `matmul`
 * when more than one token is being projected and `matvec` when exactly one
 * is, which is issue #98's "プリフィルはmatmul、デコードはmatvec" stated as
 * code rather than as two call sites that have to stay in sync.
 */
export class LlamaEngine {
  private readonly cache: KVCache;
  private readonly layerWeights: LayerWeights[];
  private readonly lmHeadT: Float32Array;
  private tokensSoFar = 0;

  constructor(
    private readonly config: LlamaConfig,
    private readonly weights: LlamaWeights,
    private readonly run: Runner["run"],
  ) {
    assertWeightShapes(config, weights);
    const { numLayers, numKvHeads, headDim, maxSeqLen, hiddenSize, numHeads, ffnHidden, vocabSize } = config;
    this.cache = new KVCache(numLayers, numKvHeads, headDim, maxSeqLen);

    const qDim = numHeads * headDim;
    const kvDim = numKvHeads * headDim;
    const qkvRows = qDim + 2 * kvDim;
    const gateUpRows = 2 * ffnHidden;

    this.layerWeights = weights.layers.map((layer: LlamaLayerWeights) => {
      const wQkv = concatRows([layer.wq, layer.wk, layer.wv], [qDim, kvDim, kvDim], hiddenSize);
      const wGateUp = concatRows([layer.wGate, layer.wUp], [ffnHidden, ffnHidden], hiddenSize);
      return {
        wQkv,
        wQkvT: transposeRowMajor(wQkv, qkvRows, hiddenSize),
        woT: transposeRowMajor(layer.wo, hiddenSize, qDim),
        wGateUp,
        wGateUpT: transposeRowMajor(wGateUp, gateUpRows, hiddenSize),
        wDownT: transposeRowMajor(layer.wDown, hiddenSize, ffnHidden),
      };
    });
    this.lmHeadT = transposeRowMajor(weights.lmHead, vocabSize, hiddenSize);
  }

  /** Positions already resident in the KV cache — 0 before the first `forward`. */
  get position(): number {
    return this.tokensSoFar;
  }

  /**
   * `hidden` is `[tokens, inFeatures]` token-major; `w` is `[outFeatures, inFeatures]`
   * (`matvec`'s own layout — see `weights.ts`) and `wT` is its `matmul`-ready
   * transpose. Returns `[tokens, outFeatures]` token-major.
   */
  private async project(
    hidden: Float32Array,
    tokens: number,
    w: Float32Array,
    wT: Float32Array,
    outFeatures: number,
    inFeatures: number,
  ): Promise<Float32Array> {
    if (tokens === 1) {
      return runMatVec(this.run, { matrix: w, vector: hidden, M: outFeatures, K: inFeatures });
    }
    return runMatMul(this.run, { a: hidden, b: wT, M: tokens, N: outFeatures, K: inFeatures });
  }

  /**
   * Runs `tokens.length` new tokens through every layer, appends their
   * (post-RoPE) K and post-projection V to the KV cache, and returns one
   * logits row per new token, each `[vocabSize]`.
   *
   * `posOffset` — RoPE's positional angle and `gqa`'s causal `queryOffset` —
   * is `this.position`, the count of tokens already cached; it advances by
   * `tokens.length` before this returns, so the next call continues where
   * this one left off. That is the entire KV-cache-continuation contract, and
   * it is a single number precisely so there is only one place to get it
   * wrong (see `llm/engine.wgsl.test.ts`'s posOffset mutation check).
   */
  async forward(tokens: number[]): Promise<Float32Array[]> {
    const {
      numLayers, hiddenSize, numHeads, numKvHeads, headDim, ffnHidden, vocabSize, ropeTheta, rmsNormEps, maxSeqLen,
    } = this.config;
    const N = tokens.length;
    if (N === 0) throw new Error("LlamaEngine.forward: tokens must be non-empty");
    const posOffset = this.tokensSoFar;
    if (posOffset + N > maxSeqLen) {
      throw new Error(`LlamaEngine.forward: position ${posOffset + N} exceeds maxSeqLen=${maxSeqLen}`);
    }

    const qDim = numHeads * headDim;
    const kvDim = numKvHeads * headDim;

    let hidden = await runGather(this.run, {
      table: this.weights.embedTokens,
      indices: Int32Array.from(tokens),
      rows: vocabSize,
      D: hiddenSize,
    });

    for (let l = 0; l < numLayers; l += 1) {
      const layer = this.weights.layers[l]!;
      const lw = this.layerWeights[l]!;

      // --- Attention block ---
      const normed = await runRmsNorm(this.run, {
        input: hidden, weight: layer.attnNorm, N, D: hiddenSize, eps: rmsNormEps,
      });

      const qkvFlat = await this.project(normed, N, lw.wQkv, lw.wQkvT, qDim + 2 * kvDim, hiddenSize);
      const [qFlat, kFlat, vFlat] = splitConcatRows(qkvFlat, N, [qDim, kvDim, kvDim]);

      // V is never rotated — only Q and K carry position information.
      const qRoped = await runRope(this.run, { input: qFlat!, N, numHeads, headDim, posOffset, thetaBase: ropeTheta });
      const kRoped = await runRope(this.run, {
        input: kFlat!, N, numHeads: numKvHeads, headDim, posOffset, thetaBase: ropeTheta,
      });

      const qHeadMajor = splitHeadsMajor(qRoped, N, numHeads, headDim);
      const kHeadMajor = splitHeadsMajor(kRoped, N, numKvHeads, headDim);
      const vHeadMajor = splitHeadsMajor(vFlat!, N, numKvHeads, headDim);

      this.cache.write(l, posOffset, kHeadMajor, vHeadMajor, N);
      const S = posOffset + N;
      const { k: kCache, v: vCache } = this.cache.read(l, S);

      const attnHeadMajor = await runGqa(this.run, {
        q: qHeadMajor,
        k: kCache,
        v: vCache,
        H: numHeads,
        kvHeads: numKvHeads,
        L: N,
        S,
        D: headDim,
        Dv: headDim,
        causal: true,
        queryOffset: posOffset,
      });
      const attnTokenMajor = mergeHeadsMajor(attnHeadMajor, numHeads, N, headDim);

      const attnOut = await this.project(attnTokenMajor, N, layer.wo, lw.woT, hiddenSize, qDim);
      hidden = await runElementwise(this.run, { a: hidden, b: attnOut, kind: ELEMENTWISE.add });

      // --- MLP block: down(silu(gate(x)) * up(x)) ---
      const normed2 = await runRmsNorm(this.run, {
        input: hidden, weight: layer.ffnNorm, N, D: hiddenSize, eps: rmsNormEps,
      });
      const gateUpFlat = await this.project(normed2, N, lw.wGateUp, lw.wGateUpT, 2 * ffnHidden, hiddenSize);
      const [gate, up] = splitConcatRows(gateUpFlat, N, [ffnHidden, ffnHidden]);
      const gateAct = await runActivation(this.run, { input: gate!, kind: ACTIVATION.silu });
      const gated = await runElementwise(this.run, { a: gateAct, b: up!, kind: ELEMENTWISE.multiply });
      const down = await this.project(gated, N, layer.wDown, lw.wDownT, hiddenSize, ffnHidden);
      hidden = await runElementwise(this.run, { a: hidden, b: down, kind: ELEMENTWISE.add });
    }

    const finalNormed = await runRmsNorm(this.run, {
      input: hidden, weight: this.weights.finalNorm, N, D: hiddenSize, eps: rmsNormEps,
    });
    const logitsFlat = await this.project(finalNormed, N, this.weights.lmHead, this.lmHeadT, vocabSize, hiddenSize);

    this.tokensSoFar += N;

    const logits: Float32Array[] = [];
    for (let t = 0; t < N; t += 1) logits.push(logitsFlat.slice(t * vocabSize, (t + 1) * vocabSize));
    return logits;
  }
}

/** argmax over one row of logits — greedy decoding's entire selection rule. */
export function argmax(logits: Float32Array): number {
  let best = 0;
  for (let i = 1; i < logits.length; i += 1) {
    if (logits[i]! > logits[best]!) best = i;
  }
  return best;
}

/**
 * Prefills `promptTokens` in one `forward` call, then greedily decodes `steps`
 * more tokens one at a time, feeding each step's argmax back in as the next
 * input. `llm/tools/gen_fixture.py`'s decode fixture runs the identical loop
 * against `transformers`, so the two can be compared token for token.
 */
export async function greedyGenerate(
  engine: LlamaEngine,
  promptTokens: number[],
  steps: number,
): Promise<{ prefillLogits: Float32Array[]; tokens: number[]; stepLogits: Float32Array[] }> {
  const prefillLogits = await engine.forward(promptTokens);
  const tokens: number[] = [];
  const stepLogits: Float32Array[] = [];
  let next = argmax(prefillLogits[prefillLogits.length - 1]!);
  for (let s = 0; s < steps; s += 1) {
    const [logits] = await engine.forward([next]);
    stepLogits.push(logits!);
    tokens.push(next);
    next = argmax(logits!);
  }
  return { prefillLogits, tokens, stepLogits };
}
