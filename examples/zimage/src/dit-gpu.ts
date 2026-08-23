/**
 * The same DiT forward, on the GPU.
 *
 * `dit.ts` is the definition of correct and is unusably slow by design: 990
 * seconds for 70 tokens on the CPU reference ops. Eight denoising steps of that
 * is two and a half hours for one small image, so nothing comes out of this
 * example without the GPU.
 *
 * The structure is `dit.ts`'s, dispatch for dispatch. No new kernel is written
 * — if this needed one, that would be a missing op, and saying so is more
 * useful than hiding it here. Every binding order and uniform layout is copied
 * from that op's own `wgsl.test.ts` rather than re-derived (rule 2): those
 * files are where the layouts are pinned, and a second reading is a second
 * chance to get one wrong.
 *
 * ## Weights are uploaded per dispatch, and that is a choice with a number
 *
 * `harness/resident.ts` exists for weights that stay on the device, and this
 * does not use it. One forward moves about 13.7 GB across the bus — every
 * layer's projections, 34 times — which at this machine's measured transfer
 * rate is seconds, not minutes, and is small next to what it buys: no bind
 * group bookkeeping for thirty-odd dispatch shapes, and the same `Runner`
 * interface `examples/zimage-vae` already uses.
 *
 * It is the obvious thing to fix first if this needs to be faster, and the
 * README says so with the measurement beside it rather than leaving "GPU" to
 * imply "fast".
 *
 * ## The dispatch cap
 *
 * WebGPU allows 65535 workgroups per dimension and going over is **not an
 * error a caller sees** — the submit does nothing and the output buffer keeps
 * whatever it held (issue #112). The VAE decoder walked into this at 256x256.
 * Here the flat elementwise dispatches are chunked for the same reason; the
 * attention and matmul dispatches are two-dimensional and stay far below it at
 * any resolution this can run.
 */
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import type { Runner } from "../../../harness/wgsl.js";
import { params } from "../../../harness/wgsl.js";
import { ACTIVATION } from "../../../ops/activation/index.js";
import { ELEMENTWISE } from "../../../ops/elementwise/index.js";
import type { DitConfig, DitInput, DitTrace, WeightSource } from "./dit.js";

/** A weight source that can also hand back the packed q8 form — see `linearPacked`. */
export interface PackedWeightSource extends WeightSource {
  packedQ8(name: string): { codes: Uint32Array; scale: Float32Array; N: number; K: number } | null;
}
import { captionPositionIds, imagePositionIds, patchify, timestepEmbedding, unpatchify } from "./dit.js";

type Run = Runner["run"];

const opsRoot = new URL("../../../ops/", import.meta.url);

/** As `decoder-gpu.ts` does it: read the WGSL directly, so this runs under plain node. */
function wgsl(op: string, entry = "kernel"): string {
  return readFileSync(fileURLToPath(new URL(`${op}/wgsl/${entry}.wgsl`, opsRoot)), "utf8");
}

const CODE = {
  rmsnorm: wgsl("rmsnorm"),
  layernorm: wgsl("layernorm"),
  matmul: wgsl("matmul"),
  activation: wgsl("activation"),
  elementwise: wgsl("elementwise"),
  rows: wgsl("elementwise", "rows"),
  ropeAxes: wgsl("rope", "axes"),
  dequantTranspose: wgsl("dequant_transpose"),
  scores: wgsl("attention", "scores"),
  context: wgsl("attention", "context"),
};

const WG = 256;
const TILE = 16;
const MAX_ELEMS_PER_DISPATCH = 65535 * WG;

async function chunked(
  len: number,
  one: (offset: number, n: number) => Promise<Float32Array>,
): Promise<Float32Array> {
  const out = new Float32Array(len);
  for (let offset = 0; offset < len; offset += MAX_ELEMS_PER_DISPATCH) {
    const n = Math.min(MAX_ELEMS_PER_DISPATCH, len - offset);
    out.set(await one(offset, n), offset);
  }
  return out;
}

/**
 * `y = x @ W^T` with `W` still packed, dequantised and transposed on the GPU.
 *
 * `ops/dequant_transpose` turns `[out, in]` q8 codes straight into `matmul`'s
 * `[in, out]` f32 operand in one dispatch. The version below it — dequantise on
 * the CPU, transpose on the CPU, upload f32 — was measured as most of a GPU
 * forward's wall time, which is the shape of answer that makes "we moved it to
 * the GPU" mean very little on its own.
 */
async function linearPacked(
  run: Run,
  x: Float32Array,
  packed: { codes: Uint32Array; scale: Float32Array; N: number; K: number },
  rows: number,
  bias?: Float32Array,
): Promise<Float32Array> {
  const { codes, scale, N: outDim, K: inDim } = packed;
  const out = new Float32Array(rows * outDim);
  const wordsPerRow = Math.ceil(inDim / 4);

  // `dequant_transpose` is a flat dispatch over `outDim * inDim`, which for the
  // FFN's 10240x3840 is 153,600 workgroups against a limit of 65,535 — issue
  // #112's failure, and it is not hypothetical: the first version of this file
  // hit it and read back a stale buffer at 9.9e-2 instead of 2.7e-6.
  //
  // Split along the **output** features. Those are whole rows of the packed
  // weight, so each chunk is a contiguous slice of both `codes` and `scale`,
  // and each chunk's `matmul` produces whole columns of the answer — assembled
  // here with a copy of `rows * outDim` floats, which is the activation and not
  // the weight.
  const maxOut = Math.max(1, Math.floor((65535 * WG) / inDim));

  for (let o0 = 0; o0 < outDim; o0 += maxOut) {
    const chunk = Math.min(maxOut, outDim - o0);
    const [wT] = await run({
      code: CODE.dequantTranspose,
      bindings: [
        { kind: "storage", data: codes.subarray(o0 * wordsPerRow, (o0 + chunk) * wordsPerRow) },
        { kind: "storage", data: scale.subarray(o0, o0 + chunk) },
        { kind: "out", type: "f32", length: chunk * inDim },
        { kind: "uniform", data: params([["u32", chunk], ["u32", inDim]]) },
      ],
      workgroups: [Math.ceil((chunk * inDim) / WG)],
    });
    const [y] = await run({
      code: CODE.matmul,
      bindings: [
        { kind: "storage", data: x },
        { kind: "storage", data: wT as Float32Array },
        { kind: "out", type: "f32", length: rows * chunk },
        { kind: "uniform", data: params([["u32", rows], ["u32", chunk], ["u32", inDim]]) },
      ],
      workgroups: [Math.ceil(chunk / TILE), Math.ceil(rows / TILE)],
    });
    const part = y as Float32Array;
    for (let r = 0; r < rows; r += 1) {
      out.set(part.subarray(r * chunk, (r + 1) * chunk), r * outDim + o0);
    }
  }

  if (bias) {
    for (let r = 0; r < rows; r += 1) {
      for (let o = 0; o < outDim; o += 1) out[r * outDim + o] = out[r * outDim + o]! + bias[o]!;
    }
  }
  return out;
}

/** `y = x @ W^T (+ b)`; `W` is `[out, in]` and the transpose stays on the CPU. */
async function linear(
  run: Run,
  x: Float32Array,
  weight: Float32Array,
  rows: number,
  inDim: number,
  outDim: number,
  bias?: Float32Array,
): Promise<Float32Array> {
  const wT = new Float32Array(inDim * outDim);
  for (let o = 0; o < outDim; o += 1) {
    for (let i = 0; i < inDim; i += 1) wT[i * outDim + o] = weight[o * inDim + i]!;
  }
  const [y] = await run({
    code: CODE.matmul,
    bindings: [
      { kind: "storage", data: x },
      { kind: "storage", data: wT },
      { kind: "out", type: "f32", length: rows * outDim },
      { kind: "uniform", data: params([["u32", rows], ["u32", outDim], ["u32", inDim]]) },
    ],
    workgroups: [Math.ceil(outDim / TILE), Math.ceil(rows / TILE)],
  });
  const out = y as Float32Array;
  if (bias) {
    for (let r = 0; r < rows; r += 1) {
      for (let o = 0; o < outDim; o += 1) out[r * outDim + o] = out[r * outDim + o]! + bias[o]!;
    }
  }
  return out;
}

async function rmsnorm(
  run: Run,
  input: Float32Array,
  weight: Float32Array,
  N: number,
  D: number,
  eps: number,
): Promise<Float32Array> {
  const [out] = await run({
    code: CODE.rmsnorm,
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

async function elementwise(run: Run, a: Float32Array, b: Float32Array, kind: number): Promise<Float32Array> {
  return chunked(a.length, async (offset, n) => {
    const [part] = await run({
      code: CODE.elementwise,
      bindings: [
        { kind: "storage", data: a.subarray(offset, offset + n) },
        { kind: "storage", data: b.subarray(offset, offset + n) },
        { kind: "out", type: "f32", length: n },
        { kind: "uniform", data: params([["u32", n], ["u32", kind]]) },
      ],
      workgroups: [Math.ceil(n / WG)],
    });
    return part as Float32Array;
  });
}

/** `a` is `[S, D]`, `b` is `[D]` and is broadcast down the rows. */
async function elementwiseRows(
  run: Run,
  a: Float32Array,
  b: Float32Array,
  S: number,
  D: number,
  kind: number,
): Promise<Float32Array> {
  const [out] = await run({
    code: CODE.rows,
    bindings: [
      { kind: "storage", data: a },
      { kind: "storage", data: b },
      { kind: "out", type: "f32", length: S * D },
      { kind: "uniform", data: params([["u32", S], ["u32", D], ["u32", kind]]) },
    ],
    workgroups: [Math.ceil((S * D) / WG)],
  });
  return out as Float32Array;
}

async function activation(run: Run, input: Float32Array, kind: number): Promise<Float32Array> {
  return chunked(input.length, async (offset, n) => {
    const [part] = await run({
      code: CODE.activation,
      bindings: [
        { kind: "storage", data: input.subarray(offset, offset + n) },
        { kind: "out", type: "f32", length: n },
        { kind: "uniform", data: params([["u32", n], ["u32", kind], ["f32", 1]]) },
      ],
      workgroups: [Math.ceil(n / WG)],
    });
    return part as Float32Array;
  });
}

/**
 * Multi-axis RoPE.
 *
 * The kernel works two channels per lane and rounds the dispatch up to a whole
 * workgroup, so the input is padded to that slack — `ops/rope/testing.ts`'s own
 * `axesScenario` does the same, and reading back only the real length is what
 * keeps the slack out of the result. The positions carry four axes of slack for
 * the same reason its doc gives: a lane that ran past `N` reads there, and
 * zeros would look like the identity rotation rather than like a bug.
 */
async function ropeAxes(
  run: Run,
  input: Float32Array,
  N: number,
  numHeads: number,
  axisDims: number[],
  positions: Int32Array,
  thetaBase: number,
): Promise<Float32Array> {
  const headDim = axisDims.reduce((sum, dim) => sum + dim, 0);
  const length = N * numHeads * headDim;
  const workgroups = Math.ceil(length / 2 / WG);
  const slots = workgroups * WG * 2;

  const padded = new Float32Array(slots);
  padded.set(input);
  const positionSlack = new Int32Array(positions.length + axisDims.length * 4).fill(9999);
  positionSlack.set(positions);

  const [out] = await run({
    code: CODE.ropeAxes,
    bindings: [
      { kind: "storage", data: padded },
      { kind: "storage", data: Uint32Array.from(axisDims) },
      { kind: "storage", data: positionSlack },
      { kind: "out", type: "f32", length: slots },
      {
        kind: "uniform",
        data: params([
          ["u32", N], ["u32", numHeads], ["u32", headDim], ["u32", axisDims.length],
          ["f32", thetaBase],
        ]),
      },
    ],
    workgroups: [workgroups],
  });
  return (out as Float32Array).subarray(0, length);
}

/** `[S, H*D]` token-major to `[H, S, D]` head-major. */
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

/**
 * Non-causal attention over `H` heads.
 *
 * Dispatched per head rather than as one batch, because the scores buffer is
 * `L * S` per head and materialising all thirty at once is what
 * `maxStorageBufferBindingSize` (512 MiB, measured) stops first: at 4096 tokens
 * one head's scores are already 64 MiB.
 */
async function attention(
  run: Run,
  q: Float32Array,
  k: Float32Array,
  v: Float32Array,
  heads: number,
  L: number,
  S: number,
  D: number,
): Promise<Float32Array> {
  const out = new Float32Array(heads * L * D);
  const zeroBias = new Float32Array(S);
  const scale = 1 / Math.sqrt(D);
  for (let h = 0; h < heads; h += 1) {
    const qh = q.subarray(h * L * D, (h + 1) * L * D);
    const kh = k.subarray(h * S * D, (h + 1) * S * D);
    const vh = v.subarray(h * S * D, (h + 1) * S * D);
    const [probs] = await run({
      code: CODE.scores,
      bindings: [
        { kind: "storage", data: qh },
        { kind: "storage", data: kh },
        { kind: "storage", data: zeroBias },
        { kind: "out", type: "f32", length: L * S },
        {
          kind: "uniform",
          data: params([
            ["u32", 1], ["u32", L], ["u32", S], ["u32", D],
            ["f32", scale],
            ["u32", 0], ["i32", 0],
            ["u32", 1], ["u32", 1], ["u32", 1],
          ]),
        },
      ],
      workgroups: [L, 1, 1],
    });
    const [context] = await run({
      code: CODE.context,
      bindings: [
        { kind: "storage", data: probs as Float32Array },
        { kind: "storage", data: vh },
        { kind: "out", type: "f32", length: L * D },
        { kind: "uniform", data: params([["u32", 1], ["u32", L], ["u32", S], ["u32", D]]) },
      ],
      workgroups: [L, 1, 1],
    });
    out.set(context as Float32Array, h * L * D);
  }
  return out;
}

/**
 * One projection, taken packed when the source has it that way.
 *
 * Falling back to dense f32 rather than requiring q8 keeps this able to run a
 * `--format mixed` conversion, where the same tensor is q4 — the fallback is
 * slower and correct, which is the right way round.
 */
async function project(
  run: Run,
  src: PackedWeightSource,
  name: string,
  x: Float32Array,
  rows: number,
  inDim: number,
  outDim: number,
  bias?: Float32Array,
): Promise<Float32Array> {
  const packed = src.packedQ8(name);
  if (packed) return linearPacked(run, x, packed, rows, bias);
  return linear(run, x, src.get(name), rows, inDim, outDim, bias);
}

/** One block. Mirrors `zimageBlock`, including `validSeq`'s trim-not-mask. */
async function block(
  run: Run,
  cfg: { dim: number; nHeads: number; headDim: number; ffnHidden: number; normEps: number; ropeAxesDims: number[]; ropeTheta: number },
  src: PackedWeightSource,
  prefix: string,
  modulated: boolean,
  x: Float32Array,
  adalnInput: Float32Array | null,
  positions: Int32Array,
  seq: number,
  validSeq: number,
): Promise<Float32Array> {
  const { dim, nHeads, headDim, ffnHidden, normEps, ropeAxesDims, ropeTheta } = cfg;
  const width = nHeads * headDim;
  const w = (name: string): Float32Array => src.get(`${prefix}${name}`);
  const p = (name: string): string => `${prefix}${name}`;

  let scaleMsa: Float32Array;
  let gateMsa: Float32Array;
  let scaleMlp: Float32Array;
  let gateMlp: Float32Array;
  void modulated;
  if (adalnInput === null) {
    const ones = () => new Float32Array(dim).fill(1);
    scaleMsa = ones();
    gateMsa = ones();
    scaleMlp = ones();
    gateMlp = ones();
  } else {
    const mod = await project(
      run, src, p("adaLN_modulation.0.weight"), adalnInput, 1, adalnInput.length, 4 * dim,
      w("adaLN_modulation.0.bias"),
    );
    const chunk = (i: number) => mod.slice(i * dim, (i + 1) * dim);
    scaleMsa = chunk(0);
    gateMsa = await activation(run, chunk(1), ACTIVATION.tanh);
    scaleMlp = chunk(2);
    gateMlp = await activation(run, chunk(3), ACTIVATION.tanh);
    for (let i = 0; i < dim; i += 1) {
      scaleMsa[i] = 1 + scaleMsa[i]!;
      scaleMlp[i] = 1 + scaleMlp[i]!;
    }
  }

  const normed1 = await rmsnorm(run, x, w("attention_norm1.weight"), seq, dim, normEps);
  const scaled1 = await elementwiseRows(run, normed1, scaleMsa, seq, dim, ELEMENTWISE.multiply);

  let q = await project(run, src, p("attention.to_q.weight"), scaled1, seq, dim, width);
  let k = await project(run, src, p("attention.to_k.weight"), scaled1, seq, dim, width);
  const v = await project(run, src, p("attention.to_v.weight"), scaled1, seq, dim, width);

  q = await rmsnorm(run, q, w("attention.norm_q.weight"), seq * nHeads, headDim, normEps);
  k = await rmsnorm(run, k, w("attention.norm_k.weight"), seq * nHeads, headDim, normEps);
  q = await ropeAxes(run, q, seq, nHeads, ropeAxesDims, positions, ropeTheta);
  k = await ropeAxes(run, k, seq, nHeads, ropeAxesDims, positions, ropeTheta);

  const live = validSeq;
  const attended = await attention(
    run,
    splitHeads(q.subarray(0, live * width), live, nHeads, headDim),
    splitHeads(k.subarray(0, live * width), live, nHeads, headDim),
    splitHeads(v.subarray(0, live * width), live, nHeads, headDim),
    nHeads, live, live, headDim,
  );
  const merged = new Float32Array(seq * width);
  merged.set(mergeHeads(attended, live, nHeads, headDim), 0);

  const projected = await project(run, src, p("attention.to_out.0.weight"), merged, seq, width, dim);
  const normed2 = await rmsnorm(run, projected, w("attention_norm2.weight"), seq, dim, normEps);
  const gated1 = await elementwiseRows(run, normed2, gateMsa, seq, dim, ELEMENTWISE.multiply);
  let h = await elementwise(run, x, gated1, ELEMENTWISE.add);

  const normed3 = await rmsnorm(run, h, w("ffn_norm1.weight"), seq, dim, normEps);
  const scaled2 = await elementwiseRows(run, normed3, scaleMlp, seq, dim, ELEMENTWISE.multiply);
  const gate = await activation(
    run, await project(run, src, p("feed_forward.w1.weight"), scaled2, seq, dim, ffnHidden), ACTIVATION.silu,
  );
  const up = await project(run, src, p("feed_forward.w3.weight"), scaled2, seq, dim, ffnHidden);
  const ffn = await project(
    run, src, p("feed_forward.w2.weight"),
    await elementwise(run, gate, up, ELEMENTWISE.multiply), seq, ffnHidden, dim,
  );
  const normed4 = await rmsnorm(run, ffn, w("ffn_norm2.weight"), seq, dim, normEps);
  const gated2 = await elementwiseRows(run, normed4, gateMlp, seq, dim, ELEMENTWISE.multiply);
  h = await elementwise(run, h, gated2, ELEMENTWISE.add);
  return h;
}


/** `ditForward`, dispatch for dispatch. */
export async function ditForwardGpu(
  run: Run,
  cfg: DitConfig,
  weights: PackedWeightSource,
  input: DitInput,
  trace?: DitTrace,
): Promise<Float32Array> {
  const { dim, nHeads, patchSize, inChannels, normEps } = cfg;
  const headDim = dim / nHeads;
  const { F, H, W } = input;
  const hTokens = H / patchSize;
  const wTokens = W / patchSize;
  const xSeq = F * hTokens * wTokens;
  const capSeq = input.capMask.length;
  const patchDim = patchSize * patchSize * inChannels;

  const tFreq = timestepEmbedding(input.t * cfg.tScale, cfg.frequencyEmbeddingSize, cfg.maxPeriod);
  const tMid = await project(
    run, weights, "t_embedder.mlp.0.weight", tFreq, 1, cfg.frequencyEmbeddingSize,
    weights.get("t_embedder.mlp.0.bias").length, weights.get("t_embedder.mlp.0.bias"),
  );
  const adalnInput = await project(
    run, weights, "t_embedder.mlp.2.weight", await activation(run, tMid, ACTIVATION.silu),
    1, tMid.length, cfg.adalnEmbedDim, weights.get("t_embedder.mlp.2.bias"),
  );
  if (trace) trace.adalnInput = adalnInput;

  const key = `${patchSize}-1`;
  let x = await project(
    run, weights, `all_x_embedder.${key}.weight`,
    patchify(input.latent, inChannels, F, H, W, patchSize, 1), xSeq, patchDim, dim,
    weights.get(`all_x_embedder.${key}.bias`),
  );

  const xPositions = imagePositionIds(F, hTokens, wTokens, capSeq);
  const ffnHidden = weights.get("layers.0.feed_forward.w1.weight").length / dim;
  const blockCfg = { dim, nHeads, headDim, ffnHidden, normEps, ropeAxesDims: cfg.ropeAxesDims, ropeTheta: cfg.ropeTheta };

  for (let i = 0; i < cfg.nRefinerLayers; i += 1) {
    x = await block(run, blockCfg, weights, `noise_refiner.${i}.`, true, x, adalnInput, xPositions, xSeq, xSeq);
  }
  if (trace) trace.afterNoiseRefiner = x.slice();

  const capNormed = await rmsnorm(
    run, input.capFeats, weights.get("cap_embedder.0.weight"), capSeq, cfg.capFeatDim, normEps,
  );
  let cap = await project(
    run, weights, "cap_embedder.1.weight", capNormed, capSeq, cfg.capFeatDim, dim,
    weights.get("cap_embedder.1.bias"),
  );

  const capPad = weights.get("cap_pad_token");
  let capValid = 0;
  while (capValid < capSeq && input.capMask[capValid]) capValid += 1;
  for (let i = capValid; i < capSeq; i += 1) {
    if (input.capMask[i]) throw new Error(`ditForwardGpu: capMask must be a prefix of real tokens (token ${i}).`);
  }
  for (let i = capValid; i < capSeq; i += 1) {
    for (let d = 0; d < dim; d += 1) cap[i * dim + d] = capPad[d]!;
  }

  const capPositions = captionPositionIds(capSeq);
  for (let i = 0; i < cfg.nRefinerLayers; i += 1) {
    cap = await block(
      run, blockCfg, weights, `context_refiner.${i}.`, false, cap, null, capPositions, capSeq, capValid,
    );
  }
  if (trace) trace.afterContextRefiner = cap.slice();

  const unifiedSeq = xSeq + capSeq;
  let unified: Float32Array = new Float32Array(unifiedSeq * dim);
  unified.set(x, 0);
  unified.set(cap, xSeq * dim);
  const positions = new Int32Array(unifiedSeq * 3);
  positions.set(xPositions, 0);
  positions.set(capPositions, xSeq * 3);
  const unifiedValid = xSeq + capValid;

  for (let i = 0; i < cfg.nLayers; i += 1) {
    unified = await block(
      run, blockCfg, weights, `layers.${i}.`, true, unified, adalnInput, positions, unifiedSeq, unifiedValid,
    );
    if (trace && i === 0) trace.afterLayer0 = unified.slice();
  }
  if (trace) trace.afterLayers = unified.slice();

  const scale = await project(
    run, weights, `all_final_layer.${key}.adaLN_modulation.1.weight`,
    await activation(run, adalnInput, ACTIVATION.silu), 1, cfg.adalnEmbedDim, dim,
    weights.get(`all_final_layer.${key}.adaLN_modulation.1.bias`),
  );
  for (let d = 0; d < dim; d += 1) scale[d] = 1 + scale[d]!;

  const [normed] = await run({
    code: CODE.layernorm,
    bindings: [
      { kind: "storage", data: unified },
      { kind: "storage", data: new Float32Array(dim).fill(1) },
      { kind: "storage", data: new Float32Array(dim) },
      { kind: "out", type: "f32", length: unifiedSeq * dim },
      { kind: "uniform", data: params([["u32", unifiedSeq], ["u32", dim], ["f32", 1e-6]]) },
    ],
    workgroups: [unifiedSeq],
  });
  const scaled = await elementwiseRows(run, normed as Float32Array, scale, unifiedSeq, dim, ELEMENTWISE.multiply);

  const projected = await project(
    run, weights, `all_final_layer.${key}.linear.weight`, scaled, unifiedSeq, dim, patchDim,
    weights.get(`all_final_layer.${key}.linear.bias`),
  );
  return unpatchify(projected.slice(0, xSeq * patchDim), inChannels, F, H, W, patchSize, 1);
}
