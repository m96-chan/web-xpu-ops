import type { Runner } from "../harness/index.js";
import { ACTIVATION } from "../ops/activation/index.js";
import { ELEMENTWISE } from "../ops/elementwise/index.js";
import type { LlamaConfig } from "./config.js";
import { runActivation, runElementwise, runMatMul, runMatVecQ8, runRmsNorm, runRope, runGqa } from "./kernels.js";
import { KVCache } from "./kv-cache.js";
import { concatRowsInt8, mergeHeadsMajor, splitConcatRows, splitHeadsMajor, transposeRowMajor } from "./reshape.js";
import {
  assertWeightShapesQ8,
  cloneQuantizedLinear,
  concatScales,
  dequantizePackedQ8,
  gatherDequantRows,
  packInt8Rows,
  type LlamaLayerWeightsQ8,
  type LlamaWeightsQ8,
  type QuantizedLinear,
} from "./weights-q8.js";

/**
 * One int8 projection group, resident after construction: `matvecQ8`'s own
 * packed wire format (`weight.ts`'s `packInt8Rows`) plus its per-row scale.
 *
 * Deliberately the *only* representation kept — not also the unpacked codes
 * `LlamaWeightsQ8` was loaded with. Packing does not change a weight's
 * resident size (`packInt8Rows`'s `[N, ceil(K/4)]` `u32` output is the same
 * order of bytes as `[N, K]` int8 codes), so keeping both would roughly
 * double this engine's weight memory for no benefit — issue #105's stated
 * concern for the real model (Sarashina2.2-1B, ~1.2B parameters). The
 * prefill path (`project` below) reconstructs f32 from this packed form on
 * demand instead (`dequantizePackedQ8`), once per `forward` call that needs
 * it, not once per token.
 */
interface PackedGroup {
  packed: Uint32Array;
  scale: Float32Array;
}

interface LayerWeightsQ8 {
  /** f32, unquantized — small (`hiddenSize` floats), so kept as a direct view rather than copied like `embedTokens` below. */
  attnNorm: Float32Array;
  ffnNorm: Float32Array;
  wQkv: PackedGroup;
  wo: PackedGroup;
  wGateUp: PackedGroup;
  wDown: PackedGroup;
}

/**
 * `LlamaEngine` (`engine.ts`), with every Linear projection's weight held as
 * int8 instead of f32 (issue #105 — the engine half of "weight converter +
 * int8 engine path"). A **separate class**, not a mode flag on `LlamaEngine`:
 * the two hold structurally different resident state (`Float32Array` weights
 * vs. packed `Uint32Array` codes + `Float32Array` scales) and project through
 * different kernels (`runMatVec`/`runMatMul` vs. `runMatVecQ8`/dequant-then-`runMatMul`),
 * so sharing one class would mean every method branching on which kind of
 * weight it holds rather than the type system ruling that out. Every other
 * piece of the forward pass — RoPE, GQA, the KV cache, activations, residual
 * adds — is identical to `LlamaEngine`'s and reused unchanged from `kernels.ts`
 * / `reshape.ts` / `kv-cache.ts`, since none of those touch a projection
 * weight's dtype.
 *
 * ## Decode vs. prefill (W8A32: int8 weight, f32 activation throughout)
 *
 * Decode (`tokens === 1`) dispatches `matvecQ8` directly against the packed,
 * resident weight — issue #97's whole motivation: a quarter the bytes moved
 * per token compared to `LlamaEngine`'s f32 `matvec`.
 *
 * Prefill (`tokens > 1`) dequantizes the needed projection's packed weight
 * into a transient f32 matrix (`dequantizePackedQ8`), transposes it
 * (`transposeRowMajor`, matching `LlamaEngine`'s own prefill layout), and
 * runs `matmul` — issue #105's own stated scope: "プリフィルは当面「行スケール
 * dequantしてf32 matmul」でもよい(正しさ優先、判断を記録)". A prefill-int8
 * kernel (`matmul` reading packed codes directly, skipping this transient f32
 * copy) is explicitly out of this issue's scope, left for a follow-up. Because
 * `greedyGenerate` calls `forward` with `tokens.length > 1` exactly once per
 * generation (the initial prompt) and with exactly `1` for every decode step
 * after it, this transient dequant happens once per generation, not once per
 * token — the cost this design accepts is bounded, not per-step.
 *
 * ## The embedding table: CPU-side gather, not a GPU dispatch
 *
 * `embedTokens` is quantized like every other weight but is never packed or
 * fed to `matvecQ8` — `gatherDequantRows` dequantizes only the rows a
 * `forward` call's `tokens` actually name, entirely on the CPU, rather than
 * dispatching `runGather` against a fully-dequantized `[vocabSize, hiddenSize]`
 * table (733 MiB for Sarashina2.2-1B, for a call that reads at most `maxSeqLen`
 * of its rows) — issue #105's own stated design: "gatherはCPU側で行い、デコード
 * 毎に1行dequantしてuploadする設計でよい".
 */
export class LlamaEngineQ8 {
  private readonly cache: KVCache;
  private readonly layerWeights: LayerWeightsQ8[];
  private readonly lmHead: PackedGroup;
  private readonly embedTokens: QuantizedLinear;
  private readonly finalNorm: Float32Array;
  private tokensSoFar = 0;

  /**
   * `weights` is deliberately **not** a parameter property (`private readonly
   * weights: LlamaWeightsQ8`) — that would create a `this.weights` field
   * holding the whole object for this instance's entire lifetime, and every
   * quantized weight in it is a view into one shared buffer covering the
   * *entire* checkpoint's codes (`weights-q8-io.ts#buildLlamaWeightsQ8`).
   * Retaining that reference at all — even just to reach `embedTokens` later —
   * would keep the whole buffer (~1.4 GiB for Sarashina2.2-1B) resident
   * alongside the packed weights this constructor builds from it, which is
   * exactly the double-memory outcome the class doc above says this engine
   * avoids. `weights` is read only inside this constructor; everything
   * `forward` needs afterward is copied out (`cloneQuantizedLinear` for
   * `embedTokens`) or referenced from the small, separate norms buffer
   * (`attnNorm`/`ffnNorm`/`finalNorm` — cheap enough, at `hiddenSize` floats
   * each, not to bother copying).
   */
  constructor(
    private readonly config: LlamaConfig,
    weights: LlamaWeightsQ8,
    private readonly run: Runner["run"],
  ) {
    assertWeightShapesQ8(config, weights);
    const { numLayers, numKvHeads, headDim, maxSeqLen, hiddenSize, numHeads, ffnHidden, vocabSize } = config;
    this.cache = new KVCache(numLayers, numKvHeads, headDim, maxSeqLen);

    const qDim = numHeads * headDim;
    const kvDim = numKvHeads * headDim;
    const qkvRows = qDim + 2 * kvDim;
    const gateUpRows = 2 * ffnHidden;

    this.layerWeights = weights.layers.map((layer: LlamaLayerWeightsQ8) => {
      const qkvCodes = concatRowsInt8([layer.wq.codes, layer.wk.codes, layer.wv.codes], [qDim, kvDim, kvDim], hiddenSize);
      const qkvScale = concatScales([layer.wq.scale, layer.wk.scale, layer.wv.scale]);
      const gateUpCodes = concatRowsInt8([layer.wGate.codes, layer.wUp.codes], [ffnHidden, ffnHidden], hiddenSize);
      const gateUpScale = concatScales([layer.wGate.scale, layer.wUp.scale]);
      return {
        attnNorm: layer.attnNorm,
        ffnNorm: layer.ffnNorm,
        wQkv: { packed: packInt8Rows(qkvCodes, qkvRows, hiddenSize), scale: qkvScale },
        wo: { packed: packInt8Rows(layer.wo.codes, hiddenSize, qDim), scale: layer.wo.scale },
        wGateUp: { packed: packInt8Rows(gateUpCodes, gateUpRows, hiddenSize), scale: gateUpScale },
        wDown: { packed: packInt8Rows(layer.wDown.codes, hiddenSize, ffnHidden), scale: layer.wDown.scale },
      };
    });
    this.lmHead = { packed: packInt8Rows(weights.lmHead.codes, vocabSize, hiddenSize), scale: weights.lmHead.scale };
    this.embedTokens = cloneQuantizedLinear(weights.embedTokens);
    this.finalNorm = weights.finalNorm;
  }

  /** Positions already resident in the KV cache — 0 before the first `forward`. */
  get position(): number {
    return this.tokensSoFar;
  }

  /**
   * `hidden` is `[tokens, inFeatures]` token-major; `group` is `matvecQ8`'s
   * packed `[outFeatures, ceil(inFeatures/4)]` weight plus its `[outFeatures]`
   * scale. Returns `[tokens, outFeatures]` token-major — see the class doc for
   * why the two branches differ in more than which kernel they call.
   */
  private async project(
    hidden: Float32Array,
    tokens: number,
    group: PackedGroup,
    outFeatures: number,
    inFeatures: number,
  ): Promise<Float32Array> {
    if (tokens === 1) {
      return runMatVecQ8(this.run, { weight: group.packed, scale: group.scale, vector: hidden, M: outFeatures, K: inFeatures });
    }
    const dequant = dequantizePackedQ8(group.packed, group.scale, outFeatures, inFeatures);
    const wT = transposeRowMajor(dequant, outFeatures, inFeatures);
    return runMatMul(this.run, { a: hidden, b: wT, M: tokens, N: outFeatures, K: inFeatures });
  }

  /** See `LlamaEngine.forward` (`engine.ts`) — identical control flow, int8 projections. */
  async forward(tokens: number[]): Promise<Float32Array[]> {
    const {
      numLayers, hiddenSize, numHeads, numKvHeads, headDim, ffnHidden, vocabSize, ropeTheta, rmsNormEps, maxSeqLen,
    } = this.config;
    const N = tokens.length;
    if (N === 0) throw new Error("LlamaEngineQ8.forward: tokens must be non-empty");
    const posOffset = this.tokensSoFar;
    if (posOffset + N > maxSeqLen) {
      throw new Error(`LlamaEngineQ8.forward: position ${posOffset + N} exceeds maxSeqLen=${maxSeqLen}`);
    }

    const qDim = numHeads * headDim;
    const kvDim = numKvHeads * headDim;

    let hidden = gatherDequantRows(this.embedTokens, tokens, hiddenSize);

    for (let l = 0; l < numLayers; l += 1) {
      const lw = this.layerWeights[l]!;

      const normed = await runRmsNorm(this.run, {
        input: hidden, weight: lw.attnNorm, N, D: hiddenSize, eps: rmsNormEps,
      });

      const qkvFlat = await this.project(normed, N, lw.wQkv, qDim + 2 * kvDim, hiddenSize);
      const [qFlat, kFlat, vFlat] = splitConcatRows(qkvFlat, N, [qDim, kvDim, kvDim]);

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

      const attnOut = await this.project(attnTokenMajor, N, lw.wo, hiddenSize, qDim);
      hidden = await runElementwise(this.run, { a: hidden, b: attnOut, kind: ELEMENTWISE.add });

      const normed2 = await runRmsNorm(this.run, {
        input: hidden, weight: lw.ffnNorm, N, D: hiddenSize, eps: rmsNormEps,
      });
      const gateUpFlat = await this.project(normed2, N, lw.wGateUp, 2 * ffnHidden, hiddenSize);
      const [gate, up] = splitConcatRows(gateUpFlat, N, [ffnHidden, ffnHidden]);
      const gateAct = await runActivation(this.run, { input: gate!, kind: ACTIVATION.silu });
      const gated = await runElementwise(this.run, { a: gateAct, b: up!, kind: ELEMENTWISE.multiply });
      const down = await this.project(gated, N, lw.wDown, hiddenSize, ffnHidden);
      hidden = await runElementwise(this.run, { a: hidden, b: down, kind: ELEMENTWISE.add });
    }

    const finalNormed = await runRmsNorm(this.run, {
      input: hidden, weight: this.finalNorm, N, D: hiddenSize, eps: rmsNormEps,
    });
    const logitsFlat = await this.project(finalNormed, N, this.lmHead, vocabSize, hiddenSize);

    this.tokensSoFar += N;

    const logits: Float32Array[] = [];
    for (let t = 0; t < N; t += 1) logits.push(logitsFlat.slice(t * vocabSize, (t + 1) * vocabSize));
    return logits;
  }
}
