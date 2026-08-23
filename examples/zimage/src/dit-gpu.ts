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
import type { Runner } from "../../../harness/wgsl.js";
import { params } from "../../../harness/wgsl.js";
import { ACTIVATION } from "../../../ops/activation/index.js";
import { ELEMENTWISE } from "../../../ops/elementwise/index.js";
import type { DitConfig, DitInput, DitTrace, WeightSource } from "./dit.js";

/** A weight source that can also hand back the packed q8 form — see `linearPacked`. */
export interface PackedWeightSource extends WeightSource {
  packedQ8(name: string): { codes: Uint32Array; scale: Float32Array; N: number; K: number } | null;
  /**
   * A tensor's shape without reading it.
   *
   * The manifest already carries every shape, and the one thing this forward
   * needs before it has fetched anything is the FFN's width. Deriving that from
   * `get(...).length` costs a 100 MB fetch to learn one integer — and in the
   * browser it happens before that layer's prefetch, which is an error rather
   * than merely wasteful.
   */
  shapeOf?(name: string): number[] | undefined;
}

/**
 * Called before each block, with the prefix of the tensors it is about to read.
 *
 * `get` and `packedQ8` are synchronous because the CPU path reads a file, and
 * making them async would put an `await` on every weight access in `dit.ts` for
 * the benefit of one caller. The browser's weights arrive over `fetch`, so it
 * needs somewhere to do that: this hook is awaited, fills the source's cache,
 * and the synchronous accessors then hit it.
 *
 * It is also where a progress bar goes, which is the other thing a browser
 * needs and a file read does not.
 */
export type LayerPrefetch = (prefix: string, label: string) => Promise<void>;
import { captionPositionIds, imagePositionIds, patchify, timestepEmbedding, unpatchify } from "./dit.js";

type Run = Runner["run"];

/**
 * The kernels this forward dispatches, by name.
 *
 * Passed in rather than read from disk, because the browser has no `fs` and
 * this file is the same code in both places. `kernels-node.ts` reads them off
 * the filesystem for Node; the browser build fetches them. Making it a
 * parameter is what stops the browser version from being a second, drifting
 * copy of this file.
 */
export interface DitKernels {
  rmsnorm: string;
  layernorm: string;
  matmul: string;
  activation: string;
  elementwise: string;
  rows: string;
  ropeAxes: string;
  matmulQ8: string;
  scores: string;
  context: string;
}

/** The kernel names, so both loaders fetch the same set. */
export const DIT_KERNEL_SOURCES: { key: keyof DitKernels; op: string; entry: string }[] = [
  { key: "rmsnorm", op: "rmsnorm", entry: "kernel" },
  { key: "layernorm", op: "layernorm", entry: "kernel" },
  { key: "matmul", op: "matmul", entry: "kernel" },
  { key: "activation", op: "activation", entry: "kernel" },
  { key: "elementwise", op: "elementwise", entry: "kernel" },
  { key: "rows", op: "elementwise", entry: "rows" },
  { key: "ropeAxes", op: "rope", entry: "axes" },
  { key: "matmulQ8", op: "matmul", entry: "q8" },
  { key: "scores", op: "attention", entry: "scores" },
  { key: "context", op: "attention", entry: "context" },
];

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
 * `y = x @ W^T` straight from the packed weight — `ops/matmul`'s `q8` entry.
 *
 * The version this replaced dequantised and transposed on the GPU
 * (`ops/dequant_transpose`) and then ran `matmul`. Correct, and it moved nine
 * times the bytes it needed to: the dequantised `[K, N]` operand is **four
 * times** the packed weight, and in this harness it came back to the CPU after
 * the first dispatch and went up again for the second.
 *
 * That was measured rather than reasoned about. Batching attention across heads
 * cut the dispatch count from 3,610 to 1,638 and changed the wall clock by
 * nothing at all — 7.1 ms per dispatch became 14.1 ms — which is what a
 * bandwidth-bound loop looks like when you take away half its calls. The
 * dispatch count was never the problem.
 *
 * `matmulQ8` reads the packed codes and the per-row scales directly, so the
 * weight crosses the bus once, at its stored size, and nothing is read back
 * between two halves of one projection.
 */
async function linearPacked(
  run: Run,
  K: DitKernels,
  x: Float32Array,
  packed: { codes: Uint32Array; scale: Float32Array; N: number; K: number },
  rows: number,
  bias?: Float32Array,
): Promise<Float32Array> {
  const { codes, scale, N: outDim, K: inDim } = packed;
  const [y] = await run({
    code: K.matmulQ8,
    bindings: [
      { kind: "storage", data: x },
      { kind: "storage", data: codes },
      { kind: "storage", data: scale },
      { kind: "out", type: "f32", length: rows * outDim },
      // `[N, M, K]` — rows of the activation, output features, contracted size.
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

/** `y = x @ W^T (+ b)`; `W` is `[out, in]` and the transpose stays on the CPU. */
async function linear(
  run: Run,
  K: DitKernels,
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
    code: K.matmul,
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
  K: DitKernels,
  input: Float32Array,
  weight: Float32Array,
  N: number,
  D: number,
  eps: number,
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

async function elementwise(run: Run, K: DitKernels, a: Float32Array, b: Float32Array, kind: number): Promise<Float32Array> {
  return chunked(a.length, async (offset, n) => {
    const [part] = await run({
      code: K.elementwise,
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
  K: DitKernels,
  a: Float32Array,
  b: Float32Array,
  S: number,
  D: number,
  kind: number,
): Promise<Float32Array> {
  const [out] = await run({
    code: K.rows,
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

async function activation(run: Run, K: DitKernels, input: Float32Array, kind: number): Promise<Float32Array> {
  return chunked(input.length, async (offset, n) => {
    const [part] = await run({
      code: K.activation,
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
  K: DitKernels,
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
    code: K.ropeAxes,
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
 * Non-causal attention over `H` heads, in **one** dispatch pair.
 *
 * The first version of this looped over heads, passing `H = 1` each time. That
 * was 30 dispatch pairs per block and 2,040 per forward — 57% of the 3,610
 * dispatches a forward made, in a harness where each one costs a submit, a
 * fence and a readback (measured: 7.1 ms per dispatch, 95% of wall time). The
 * GPU was not busy; it was being asked one question at a time.
 *
 * The reason for the loop was `maxStorageBufferBindingSize`, and it does not
 * apply at these shapes: all thirty heads' scores at 271 tokens are 8.8 MB
 * against a 512 MiB ceiling. So the batch is taken whole, and the loop is kept
 * for the case where it would not fit — a fallback with a threshold rather than
 * a rule applied everywhere in case it might matter.
 */
async function attention(
  run: Run,
  K: DitKernels,
  q: Float32Array,
  k: Float32Array,
  v: Float32Array,
  heads: number,
  L: number,
  S: number,
  D: number,
): Promise<Float32Array> {
  const scale = 1 / Math.sqrt(D);
  const scoresBytes = heads * L * S * 4;
  // The device's own ceiling is not visible from here, so this is the spec's
  // guaranteed minimum rather than a guess at the adapter's — under it, every
  // device works; over it, the per-head path is taken and is merely slower.
  const CAP = 128 * 1024 * 1024;

  const dispatch = async (h0: number, count: number): Promise<Float32Array> => {
    const qh = q.subarray(h0 * L * D, (h0 + count) * L * D);
    const kh = k.subarray(h0 * S * D, (h0 + count) * S * D);
    const vh = v.subarray(h0 * S * D, (h0 + count) * S * D);
    const [probs] = await run({
      code: K.scores,
      bindings: [
        { kind: "storage", data: qh },
        { kind: "storage", data: kh },
        { kind: "storage", data: new Float32Array(S) },
        { kind: "out", type: "f32", length: count * L * S },
        {
          kind: "uniform",
          data: params([
            ["u32", count], ["u32", L], ["u32", S], ["u32", D],
            ["f32", scale],
            ["u32", 0], ["i32", 0],
            ["u32", 1], ["u32", 1], ["u32", 1],
          ]),
        },
      ],
      workgroups: [L, count, 1],
    });
    const [context] = await run({
      code: K.context,
      bindings: [
        { kind: "storage", data: probs as Float32Array },
        { kind: "storage", data: vh },
        { kind: "out", type: "f32", length: count * L * D },
        { kind: "uniform", data: params([["u32", count], ["u32", L], ["u32", S], ["u32", D]]) },
      ],
      workgroups: [L, count, 1],
    });
    return context as Float32Array;
  };

  if (scoresBytes <= CAP) return dispatch(0, heads);

  const perBatch = Math.max(1, Math.floor(CAP / (L * S * 4)));
  const out = new Float32Array(heads * L * D);
  for (let h0 = 0; h0 < heads; h0 += perBatch) {
    const count = Math.min(perBatch, heads - h0);
    out.set(await dispatch(h0, count), h0 * L * D);
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
  K: DitKernels,
  src: PackedWeightSource,
  name: string,
  x: Float32Array,
  rows: number,
  inDim: number,
  outDim: number,
  bias?: Float32Array,
): Promise<Float32Array> {
  const packed = src.packedQ8(name);
  if (packed) return linearPacked(run, K, x, packed, rows, bias);
  return linear(run, K, x, src.get(name), rows, inDim, outDim, bias);
}

/** One block. Mirrors `zimageBlock`, including `validSeq`'s trim-not-mask. */
async function block(
  run: Run,
  K: DitKernels,
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
      run, K, src, p("adaLN_modulation.0.weight"), adalnInput, 1, adalnInput.length, 4 * dim,
      w("adaLN_modulation.0.bias"),
    );
    const chunk = (i: number) => mod.slice(i * dim, (i + 1) * dim);
    scaleMsa = chunk(0);
    gateMsa = await activation(run, K, chunk(1), ACTIVATION.tanh);
    scaleMlp = chunk(2);
    gateMlp = await activation(run, K, chunk(3), ACTIVATION.tanh);
    for (let i = 0; i < dim; i += 1) {
      scaleMsa[i] = 1 + scaleMsa[i]!;
      scaleMlp[i] = 1 + scaleMlp[i]!;
    }
  }

  const normed1 = await rmsnorm(run, K, x, w("attention_norm1.weight"), seq, dim, normEps);
  const scaled1 = await elementwiseRows(run, K, normed1, scaleMsa, seq, dim, ELEMENTWISE.multiply);

  let q = await project(run, K, src, p("attention.to_q.weight"), scaled1, seq, dim, width);
  let k = await project(run, K, src, p("attention.to_k.weight"), scaled1, seq, dim, width);
  const v = await project(run, K, src, p("attention.to_v.weight"), scaled1, seq, dim, width);

  q = await rmsnorm(run, K, q, w("attention.norm_q.weight"), seq * nHeads, headDim, normEps);
  k = await rmsnorm(run, K, k, w("attention.norm_k.weight"), seq * nHeads, headDim, normEps);
  q = await ropeAxes(run, K, q, seq, nHeads, ropeAxesDims, positions, ropeTheta);
  k = await ropeAxes(run, K, k, seq, nHeads, ropeAxesDims, positions, ropeTheta);

  const live = validSeq;
  const attended = await attention(
    run, K,
    splitHeads(q.subarray(0, live * width), live, nHeads, headDim),
    splitHeads(k.subarray(0, live * width), live, nHeads, headDim),
    splitHeads(v.subarray(0, live * width), live, nHeads, headDim),
    nHeads, live, live, headDim,
  );
  const merged = new Float32Array(seq * width);
  merged.set(mergeHeads(attended, live, nHeads, headDim), 0);

  const projected = await project(run, K, src, p("attention.to_out.0.weight"), merged, seq, width, dim);
  const normed2 = await rmsnorm(run, K, projected, w("attention_norm2.weight"), seq, dim, normEps);
  const gated1 = await elementwiseRows(run, K, normed2, gateMsa, seq, dim, ELEMENTWISE.multiply);
  let h = await elementwise(run, K, x, gated1, ELEMENTWISE.add);

  const normed3 = await rmsnorm(run, K, h, w("ffn_norm1.weight"), seq, dim, normEps);
  const scaled2 = await elementwiseRows(run, K, normed3, scaleMlp, seq, dim, ELEMENTWISE.multiply);
  const gate = await activation(
    run, K, await project(run, K, src, p("feed_forward.w1.weight"), scaled2, seq, dim, ffnHidden), ACTIVATION.silu,
  );
  const up = await project(run, K, src, p("feed_forward.w3.weight"), scaled2, seq, dim, ffnHidden);
  const ffn = await project(
    run, K, src, p("feed_forward.w2.weight"),
    await elementwise(run, K, gate, up, ELEMENTWISE.multiply), seq, ffnHidden, dim,
  );
  const normed4 = await rmsnorm(run, K, ffn, w("ffn_norm2.weight"), seq, dim, normEps);
  const gated2 = await elementwiseRows(run, K, normed4, gateMlp, seq, dim, ELEMENTWISE.multiply);
  h = await elementwise(run, K, h, gated2, ELEMENTWISE.add);
  return h;
}


/** `ditForward`, dispatch for dispatch. */
export async function ditForwardGpu(
  run: Run,
  K: DitKernels,
  cfg: DitConfig,
  weights: PackedWeightSource,
  input: DitInput,
  trace?: DitTrace,
  prefetch?: LayerPrefetch,
): Promise<Float32Array> {
  const { dim, nHeads, patchSize, inChannels, normEps } = cfg;
  const headDim = dim / nHeads;
  const { F, H, W } = input;
  const hTokens = H / patchSize;
  const wTokens = W / patchSize;
  const xSeq = F * hTokens * wTokens;
  const capSeq = input.capMask.length;
  const patchDim = patchSize * patchSize * inChannels;

  // Before the first read, not after it: `project` and `get` are synchronous
  // and the browser's cache is filled by this hook, so a prefetch placed below
  // its own tensor's use throws rather than fetching late. It did.
  await prefetch?.("t_embedder.", "timestep embedder");
  await prefetch?.(`all_x_embedder.${patchSize}-1.`, "patch embedder");

  const tFreq = timestepEmbedding(input.t * cfg.tScale, cfg.frequencyEmbeddingSize, cfg.maxPeriod);
  const tMid = await project(
    run, K, weights, "t_embedder.mlp.0.weight", tFreq, 1, cfg.frequencyEmbeddingSize,
    weights.get("t_embedder.mlp.0.bias").length, weights.get("t_embedder.mlp.0.bias"),
  );
  const adalnInput = await project(
    run, K, weights, "t_embedder.mlp.2.weight", await activation(run, K, tMid, ACTIVATION.silu),
    1, tMid.length, cfg.adalnEmbedDim, weights.get("t_embedder.mlp.2.bias"),
  );
  if (trace) trace.adalnInput = adalnInput;

  const key = `${patchSize}-1`;
  let x = await project(
    run, K, weights, `all_x_embedder.${key}.weight`,
    patchify(input.latent, inChannels, F, H, W, patchSize, 1), xSeq, patchDim, dim,
    weights.get(`all_x_embedder.${key}.bias`),
  );

  const xPositions = imagePositionIds(F, hTokens, wTokens, capSeq);
  // `[ffnHidden, dim]`. From the manifest when the source has it — see
  // `shapeOf` — and otherwise from the tensor, which is what the Node loader
  // does anyway.
  const ffnHidden =
    weights.shapeOf?.("layers.0.feed_forward.w1.weight")?.[0] ??
    weights.get("layers.0.feed_forward.w1.weight").length / dim;
  const blockCfg = { dim, nHeads, headDim, ffnHidden, normEps, ropeAxesDims: cfg.ropeAxesDims, ropeTheta: cfg.ropeTheta };

  for (let i = 0; i < cfg.nRefinerLayers; i += 1) {
    await prefetch?.(`noise_refiner.${i}.`, `noise refiner ${i + 1}/${cfg.nRefinerLayers}`);
    x = await block(run, K, blockCfg, weights, `noise_refiner.${i}.`, true, x, adalnInput, xPositions, xSeq, xSeq);
  }
  if (trace) trace.afterNoiseRefiner = x.slice();

  await prefetch?.("cap_embedder.", "caption embedder");
  await prefetch?.("cap_pad_token", "caption pad token");
  const capNormed = await rmsnorm(
    run, K, input.capFeats, weights.get("cap_embedder.0.weight"), capSeq, cfg.capFeatDim, normEps,
  );
  let cap = await project(
    run, K, weights, "cap_embedder.1.weight", capNormed, capSeq, cfg.capFeatDim, dim,
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
    await prefetch?.(`context_refiner.${i}.`, `context refiner ${i + 1}/${cfg.nRefinerLayers}`);
    cap = await block(
      run, K, blockCfg, weights, `context_refiner.${i}.`, false, cap, null, capPositions, capSeq, capValid,
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
    await prefetch?.(`layers.${i}.`, `layer ${i + 1}/${cfg.nLayers}`);
    unified = await block(
      run, K, blockCfg, weights, `layers.${i}.`, true, unified, adalnInput, positions, unifiedSeq, unifiedValid,
    );
    if (trace && i === 0) trace.afterLayer0 = unified.slice();
  }
  if (trace) trace.afterLayers = unified.slice();

  await prefetch?.(`all_final_layer.${patchSize}-1.`, "final layer");
  const scale = await project(
    run, K, weights, `all_final_layer.${key}.adaLN_modulation.1.weight`,
    await activation(run, K, adalnInput, ACTIVATION.silu), 1, cfg.adalnEmbedDim, dim,
    weights.get(`all_final_layer.${key}.adaLN_modulation.1.bias`),
  );
  for (let d = 0; d < dim; d += 1) scale[d] = 1 + scale[d]!;

  const [normed] = await run({
    code: K.layernorm,
    bindings: [
      { kind: "storage", data: unified },
      { kind: "storage", data: new Float32Array(dim).fill(1) },
      { kind: "storage", data: new Float32Array(dim) },
      { kind: "out", type: "f32", length: unifiedSeq * dim },
      { kind: "uniform", data: params([["u32", unifiedSeq], ["u32", dim], ["f32", 1e-6]]) },
    ],
    workgroups: [unifiedSeq],
  });
  const scaled = await elementwiseRows(run, K, normed as Float32Array, scale, unifiedSeq, dim, ELEMENTWISE.multiply);

  const projected = await project(
    run, K, weights, `all_final_layer.${key}.linear.weight`, scaled, unifiedSeq, dim, patchDim,
    weights.get(`all_final_layer.${key}.linear.bias`),
  );
  return unpatchify(projected.slice(0, xSeq * patchDim), inChannels, F, H, W, patchSize, 1);
}
