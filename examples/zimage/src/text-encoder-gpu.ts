/**
 * The same Qwen3 encoder, on the GPU.
 *
 * `text-encoder.ts` is the definition of correct and takes 136 seconds for a
 * 15-token prompt — most of a CPU generation's wall clock, and long enough that
 * a browser tab would look hung. The structure below is that file's, dispatch
 * for dispatch, against the same golden.
 *
 * No new kernel: `ops/gqa` already has the grouped attention, and every other
 * step is an op this repository ships. The one thing worth naming is that
 * **QK-Norm's weight must be permuted with the projection**, exactly as on the
 * CPU side — see `text-encoder.ts` for why, and for the 1.002e-1 it costs to
 * forget.
 */
import type { Runner } from "../../../harness/wgsl.js";
import { params } from "../../../harness/wgsl.js";
import { ACTIVATION } from "../../../ops/activation/index.js";
import { ELEMENTWISE } from "../../../ops/elementwise/index.js";
import type { Qwen3Config, Qwen3LayerWeights } from "./text-encoder.js";
import { matmulGrid } from "../../../ops/matmul/index.js";

type Run = Runner["run"];

/** The kernels this encoder dispatches. Same arrangement as `DitKernels`. */
export interface EncoderKernels {
  rmsnorm: string;
  matmul: string;
  activation: string;
  elementwise: string;
  rope: string;
  gqaScores: string;
  gqaContext: string;
}

export const ENCODER_KERNEL_SOURCES: { key: keyof EncoderKernels; op: string; entry: string }[] = [
  { key: "rmsnorm", op: "rmsnorm", entry: "kernel" },
  { key: "matmul", op: "matmul", entry: "kernel" },
  { key: "activation", op: "activation", entry: "kernel" },
  { key: "elementwise", op: "elementwise", entry: "kernel" },
  { key: "rope", op: "rope", entry: "kernel" },
  { key: "gqaScores", op: "gqa", entry: "scores" },
  { key: "gqaContext", op: "gqa", entry: "context" },
];

const WG = 256;
const TILE = 16;

async function matmulOp(
  run: Run, K: EncoderKernels, x: Float32Array, wT: Float32Array, rows: number, inDim: number, outDim: number,
): Promise<Float32Array> {
  const [y] = await run({
    code: K.matmul,
    bindings: [
      { kind: "storage", data: x },
      { kind: "storage", data: wT },
      { kind: "out", type: "f32", length: rows * outDim },
      { kind: "uniform", data: params([["u32", rows], ["u32", outDim], ["u32", inDim]]) },
    ],
    workgroups: matmulGrid(rows, outDim),
  });
  return y as Float32Array;
}

/** `nn.Linear` without bias; `W` is `[out, in]` and the transpose is on the CPU. */
async function linear(
  run: Run, K: EncoderKernels, x: Float32Array, weight: Float32Array, rows: number, inDim: number, outDim: number,
): Promise<Float32Array> {
  const wT = new Float32Array(inDim * outDim);
  for (let o = 0; o < outDim; o += 1) {
    for (let i = 0; i < inDim; i += 1) wT[i * outDim + o] = weight[o * inDim + i]!;
  }
  return matmulOp(run, K, x, wT, rows, inDim, outDim);
}

async function rmsnormOp(
  run: Run, K: EncoderKernels, input: Float32Array, weight: Float32Array, N: number, D: number, eps: number,
): Promise<Float32Array> {
  const [out] = await run({
    code: K.rmsnorm,
    bindings: [
      { kind: "storage", data: input },
      { kind: "storage", data: weight },
      { kind: "out", type: "f32", length: N * D },
      { kind: "uniform", data: params([["u32", N], ["u32", D], ["f32", eps]]) },
    ],
    workgroups: [N],
  });
  return out as Float32Array;
}

async function elementwiseOp(
  run: Run, K: EncoderKernels, a: Float32Array, b: Float32Array, kind: number,
): Promise<Float32Array> {
  const [out] = await run({
    code: K.elementwise,
    bindings: [
      { kind: "storage", data: a },
      { kind: "storage", data: b },
      { kind: "out", type: "f32", length: a.length },
      { kind: "uniform", data: params([["u32", a.length], ["u32", kind]]) },
    ],
    workgroups: [Math.ceil(a.length / WG)],
  });
  return out as Float32Array;
}

async function activationOp(
  run: Run, K: EncoderKernels, input: Float32Array, kind: number,
): Promise<Float32Array> {
  const [out] = await run({
    code: K.activation,
    bindings: [
      { kind: "storage", data: input },
      { kind: "out", type: "f32", length: input.length },
      { kind: "uniform", data: params([["u32", input.length], ["u32", kind], ["f32", 1]]) },
    ],
    workgroups: [Math.ceil(input.length / WG)],
  });
  return out as Float32Array;
}

/** Single-axis RoPE, `ops/rope`'s own dispatch shape. */
async function ropeOp(
  run: Run, K: EncoderKernels, input: Float32Array, N: number, numHeads: number, headDim: number, thetaBase: number,
): Promise<Float32Array> {
  const length = N * numHeads * headDim;
  const workgroups = Math.ceil(length / 2 / WG);
  const slots = workgroups * WG * 2;
  const padded = new Float32Array(slots);
  padded.set(input);
  const [out] = await run({
    code: K.rope,
    bindings: [
      { kind: "storage", data: padded },
      // The cos/sin cache. `ops/rope` takes one at binding 1 and reads it only
      // when `cachedPositions` is non-zero; an empty table with that count at
      // zero is the "compute the angles in the kernel" path, which is what a
      // 15-token prompt wants. A missing binding here is a validation error
      // rather than a wrong number, which is the good case.
      { kind: "storage", data: new Float32Array(1) },
      { kind: "out", type: "f32", length: slots },
      {
        // `ops/rope/testing.ts`'s own layout, copied rather than re-derived:
        // geometry, then the cached-positions count, then the frequency
        // scaling group (base, interpolation, both ramp ends, attention
        // factor), then the head range appended last — that ordering is
        // load-bearing and the appended pair is why it looks the way it does.
        kind: "uniform",
        data: params([
          ["u32", N], ["u32", numHeads], ["u32", headDim], ["u32", 0],
          ["u32", 0],
          ["f32", thetaBase], ["f32", 1],
          ["f32", 0], ["f32", 0], ["f32", 1],
          ["u32", 0], ["u32", numHeads],
        ]),
      },
    ],
    workgroups: [workgroups],
  });
  return (out as Float32Array).subarray(0, length);
}

function splitHeads(x: Float32Array, seq: number, heads: number, dim: number): Float32Array {
  const out = new Float32Array(seq * heads * dim);
  for (let s = 0; s < seq; s += 1) {
    for (let h = 0; h < heads; h += 1) {
      for (let d = 0; d < dim; d += 1) out[(h * seq + s) * dim + d] = x[(s * heads + h) * dim + d]!;
    }
  }
  return out;
}

function mergeHeads(x: Float32Array, seq: number, heads: number, dim: number): Float32Array {
  const out = new Float32Array(seq * heads * dim);
  for (let h = 0; h < heads; h += 1) {
    for (let s = 0; s < seq; s += 1) {
      for (let d = 0; d < dim; d += 1) out[(s * heads + h) * dim + d] = x[(h * seq + s) * dim + d]!;
    }
  }
  return out;
}

/** Causal grouped attention, per query head — `ops/gqa`'s own dispatch shape. */
async function groupedAttention(
  run: Run,
  K: EncoderKernels,
  q: Float32Array,
  k: Float32Array,
  v: Float32Array,
  heads: number,
  kvHeads: number,
  L: number,
  D: number,
): Promise<Float32Array> {
  const [probs] = await run({
    code: K.gqaScores,
    bindings: [
      { kind: "storage", data: q },
      { kind: "storage", data: k },
      { kind: "storage", data: new Float32Array(L) },
      { kind: "out", type: "f32", length: heads * L * L },
      {
        // `ops/gqa/testing.ts`'s layout. No batch field — the kernel takes it
        // from the dispatch's z dimension.
        kind: "uniform",
        data: params([
          ["u32", heads], ["u32", kvHeads], ["u32", L], ["u32", L], ["u32", D],
          ["f32", 1 / Math.sqrt(D)],
          ["u32", 1], ["i32", 0],
          ["u32", 1], ["u32", 1], ["u32", 1],
          ["u32", L],
        ]),
      },
    ],
    workgroups: [L, heads, 1],
  });
  const [out] = await run({
    code: K.gqaContext,
    bindings: [
      { kind: "storage", data: probs as Float32Array },
      { kind: "storage", data: v },
      { kind: "out", type: "f32", length: heads * L * D },
      {
        kind: "uniform",
        data: params([
          ["u32", heads], ["u32", kvHeads], ["u32", L], ["u32", L], ["u32", D], ["u32", L],
        ]),
      },
    ],
    workgroups: [L, heads, 1],
  });
  return out as Float32Array;
}

/**
 * Weights, possibly fetched.
 *
 * `embed` and `layer` may return a promise: in the browser a layer's tensors
 * arrive over HTTP, and this function is already async, so awaiting them here
 * costs nothing and saves the browser a second copy of the loop. The Node
 * caller returns them directly and `await` passes those through unchanged.
 */
export interface Qwen3GpuWeights {
  embed(ids: Int32Array): Float32Array | Promise<Float32Array>;
  numLayers: number;
  layer(index: number): Qwen3LayerWeights | Promise<Qwen3LayerWeights>;
  /**
   * `model.norm`, applied after the last layer — **optional, and absent for
   * Z-Image**, which reads `hidden_states[-2]` from before it. The mirror of
   * `Qwen3Weights.finalNorm`; see its doc for why the two models differ.
   */
  finalNorm?: Float32Array | Promise<Float32Array>;
}

/** `qwen3Encode`, dispatch for dispatch. */
export async function qwen3EncodeGpu(
  run: Run,
  K: EncoderKernels,
  cfg: Qwen3Config,
  weights: Qwen3GpuWeights,
  tokenIds: Int32Array,
  onLayer?: (index: number) => void | Promise<void>,
): Promise<Float32Array> {
  const { hiddenSize, numHeads, numKvHeads, headDim, ffnHidden, rmsNormEps, ropeTheta } = cfg;
  const seq = tokenIds.length;
  const qDim = numHeads * headDim;
  const kvDim = numKvHeads * headDim;

  let hidden = await weights.embed(tokenIds);
  for (let l = 0; l <= cfg.stopAfterLayer; l += 1) {
    const w = await weights.layer(l);

    const normed = await rmsnormOp(run, K, hidden, w.input_layernorm, seq, hiddenSize, rmsNormEps);
    let q = await linear(run, K, normed, w.q_proj, seq, hiddenSize, qDim);
    let k = await linear(run, K, normed, w.k_proj, seq, hiddenSize, kvDim);
    const v = await linear(run, K, normed, w.v_proj, seq, hiddenSize, kvDim);

    q = await rmsnormOp(run, K, q, w.q_norm, seq * numHeads, headDim, rmsNormEps);
    k = await rmsnormOp(run, K, k, w.k_norm, seq * numKvHeads, headDim, rmsNormEps);
    q = await ropeOp(run, K, q, seq, numHeads, headDim, ropeTheta);
    k = await ropeOp(run, K, k, seq, numKvHeads, headDim, ropeTheta);

    const attn = await groupedAttention(
      run, K,
      splitHeads(q, seq, numHeads, headDim),
      splitHeads(k, seq, numKvHeads, headDim),
      splitHeads(v, seq, numKvHeads, headDim),
      numHeads, numKvHeads, seq, headDim,
    );
    const projected = await linear(
      run, K, mergeHeads(attn, seq, numHeads, headDim), w.o_proj, seq, qDim, hiddenSize,
    );
    hidden = await elementwiseOp(run, K, hidden, projected, ELEMENTWISE.add);

    const normed2 = await rmsnormOp(run, K, hidden, w.post_attention_layernorm, seq, hiddenSize, rmsNormEps);
    const gate = await activationOp(
      run, K, await linear(run, K, normed2, w.gate_proj, seq, hiddenSize, ffnHidden), ACTIVATION.silu,
    );
    const up = await linear(run, K, normed2, w.up_proj, seq, hiddenSize, ffnHidden);
    const down = await linear(
      run, K, await elementwiseOp(run, K, gate, up, ELEMENTWISE.multiply), w.down_proj, seq, ffnHidden, hiddenSize,
    );
    hidden = await elementwiseOp(run, K, hidden, down, ELEMENTWISE.add);
    await onLayer?.(l);
  }
  // `model.norm` when the caller supplies it, never the LM head. Z-Image takes
  // `hidden_states[-2]`, from before both; Anima takes the last layer, from
  // after the norm.
  if (weights.finalNorm) {
    hidden = await rmsnormOp(run, K, hidden, await weights.finalNorm, seq, hiddenSize, rmsNormEps);
  }
  return hidden;
}
