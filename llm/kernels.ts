import { kernel, params, type Runner } from "../harness/index.js";
import type { ActivationKind } from "../ops/activation/index.js";
import type { ElementwiseKind } from "../ops/elementwise/index.js";

/**
 * GPU dispatch wrappers around the ops this engine composes, one function per
 * kernel entry point.
 *
 * Every binding order, uniform field order and workgroup count below is
 * copied from that op's own `wgsl.test.ts` (or, for `gqa`, its two-file
 * `scores`/`context` split) — not re-derived, per rule 2. Where a comment
 * cites a fact about buffer sizing ("exact-length buffers are safe here"), it
 * is because the kernel's own `.wgsl` was read to confirm the guard exists,
 * not assumed from the op's test file alone; see the op's `wgsl/kernel.wgsl`
 * for the guard itself.
 *
 * Every dispatch here uses an output buffer sized to the *real* element count,
 * not to the dispatch width rounded up to a workgroup. Each of these seven
 * kernels was read and guards both its reads and its writes against the
 * uniform-supplied size (not the buffer's own length) before touching memory,
 * so a thread past the real data returns/discards rather than reading or
 * writing garbage — the padding some of their own `wgsl.test.ts` files add
 * exists to *exercise* those guards, not because a caller needs to supply it.
 */

function opKernel(op: string, entry = "kernel"): string {
  return kernel(new URL(`../ops/${op}/index.ts`, import.meta.url), entry);
}

const CODE = {
  gather: opKernel("gather"),
  rmsnorm: opKernel("rmsnorm"),
  matvec: opKernel("matvec"),
  matvecQ8: opKernel("matvec", "q8"),
  matmul: opKernel("matmul"),
  rope: opKernel("rope"),
  gqaScores: opKernel("gqa", "scores"),
  gqaContext: opKernel("gqa", "context"),
  activation: opKernel("activation"),
  elementwise: opKernel("elementwise"),
};

const asF32 = (x: Float32Array | Int32Array | Uint32Array): Float32Array => x as Float32Array;

export interface GatherArgs {
  /** `[rows, D]` row-major. */
  table: Float32Array;
  indices: Int32Array;
  rows: number;
  D: number;
}

export async function runGather(run: Runner["run"], { table, indices, rows, D }: GatherArgs): Promise<Float32Array> {
  const N = indices.length;
  const [out] = await run({
    code: CODE.gather,
    bindings: [
      { kind: "storage", data: table },
      { kind: "storage", data: indices },
      { kind: "out", type: "f32", length: N * D },
      { kind: "uniform", data: params([["u32", N], ["u32", D], ["u32", rows]]) },
    ],
    workgroups: [Math.ceil((N * D) / 256)],
  });
  return asF32(out!);
}

export interface RmsNormArgs {
  input: Float32Array;
  /** `[groups, D]`; `groups` defaults to 1, one shared gamma. */
  weight: Float32Array;
  N: number;
  D: number;
  eps: number;
  groups?: number;
}

export async function runRmsNorm(
  run: Runner["run"],
  { input, weight, N, D, eps, groups = 1 }: RmsNormArgs,
): Promise<Float32Array> {
  const [out] = await run({
    code: CODE.rmsnorm,
    bindings: [
      { kind: "storage", data: input },
      { kind: "storage", data: weight },
      { kind: "out", type: "f32", length: N * D },
      { kind: "uniform", data: params([["u32", N], ["u32", D], ["f32", eps], ["u32", groups]]) },
    ],
    workgroups: [N],
  });
  return asF32(out!);
}

/**
 * `runMatVec`/`runMatVecQ8` dispatch one workgroup per output row
 * (`workgroups: [M]`, no tiling — see `ops/matvec/wgsl/kernel.wgsl` and
 * `q8.wgsl`'s own docs), which is exact and cheap until `M` exceeds
 * `maxComputeWorkgroupsPerDimension`. Every WebGPU implementation measured
 * against this repo — Dawn on Linux/NVIDIA (Node, and Chrome) — reports that
 * limit as `65535`, the spec's own required default; hard-coded here rather
 * than threaded through from an adapter because `Runner["run"]` (this
 * module's only handle on the device) carries no way to ask it, by design —
 * `llm/kernels.ts` and its two `Runner` implementations (Node's
 * `harness/wgsl.ts`, the browser demo's `browser-runtime.ts`) do not
 * otherwise know anything device-specific.
 *
 * A dispatch past the limit is not rejected loudly: WebGPU reports the
 * validation error asynchronously through the device's own error callback,
 * not by throwing where `dispatchWorkgroups` is called, so the visible
 * *symptom* is a compute pass that silently does nothing — every row of
 * `output` stays at its buffer's zero-initialized default. First hit running
 * this engine's real target model (Sarashina2.2-1B, `vocabSize=102400`) in a
 * browser (issue #106): `lm_head`'s decode-time projection
 * (`LlamaEngineQ8`'s `project`, `tokens === 1`) dispatches `matvecQ8` with
 * `M=102400`, and every decode step after the first (which the prefill's
 * `matmul` path — tiled, unaffected — still gets right) produced the same
 * wrong token, because argmax over an all-zero logits vector always picks
 * index 0. Below the limit this is a single dispatch, identical to before.
 */
const MAX_WORKGROUPS_PER_DISPATCH = 65535;

/**
 * Runs a one-workgroup-per-row dispatch over `M` rows, splitting into
 * multiple dispatches of at most `MAX_WORKGROUPS_PER_DISPATCH` rows each
 * when `M` exceeds it, and concatenating the results back into one `[M]`
 * output — see `MAX_WORKGROUPS_PER_DISPATCH`'s own doc for why this exists.
 * No WGSL changes: the shader has no idea how many dispatches contributed to
 * `output`, and does not need to — each chunk's dispatch writes rows
 * `[0, rowCount)` of its own `"out"` binding, and `dispatchChunk` is given
 * the row range to source its *inputs* from.
 */
async function dispatchRowsChunked(
  M: number,
  dispatchChunk: (rowStart: number, rowCount: number) => Promise<Float32Array>,
): Promise<Float32Array> {
  if (M <= MAX_WORKGROUPS_PER_DISPATCH) return dispatchChunk(0, M);
  const result = new Float32Array(M);
  for (let rowStart = 0; rowStart < M; rowStart += MAX_WORKGROUPS_PER_DISPATCH) {
    const rowCount = Math.min(MAX_WORKGROUPS_PER_DISPATCH, M - rowStart);
    result.set(await dispatchChunk(rowStart, rowCount), rowStart);
  }
  return result;
}

export interface MatVecArgs {
  /** `[M, K]` row-major. */
  matrix: Float32Array;
  vector: Float32Array;
  M: number;
  K: number;
}

export async function runMatVec(run: Runner["run"], { matrix, vector, M, K }: MatVecArgs): Promise<Float32Array> {
  return dispatchRowsChunked(M, async (rowStart, rowCount) => {
    const [out] = await run({
      code: CODE.matvec,
      bindings: [
        { kind: "storage", data: matrix.subarray(rowStart * K, (rowStart + rowCount) * K) },
        { kind: "storage", data: vector },
        { kind: "out", type: "f32", length: rowCount },
        { kind: "uniform", data: params([["u32", rowCount], ["u32", K]]) },
      ],
      workgroups: [rowCount],
    });
    return asF32(out!);
  });
}

export interface MatVecQ8Args {
  /** `[M, ceil(K/4)]` u32, row-major, `matvecQ8`'s own packed wire format (`ops/matvec/reference.ts#packQ8`). */
  weight: Uint32Array;
  /** `[M]`, one absmax-derived scale per row. */
  scale: Float32Array;
  vector: Float32Array;
  M: number;
  K: number;
}

/**
 * `matvecQ8` (issue #97), used by `LlamaEngineQ8`'s decode path (issue #105):
 * the same GEMV as `runMatVec`, with the weight held as int8 codes packed
 * four to a `u32` word instead of f32 — a quarter the bytes moved per decode
 * step, which is what makes decode bandwidth-bound rather than
 * compute-bound at this weight size. Binding order and uniform layout are
 * copied from `ops/matvec/q8.wgsl.test.ts`, not re-derived (rule 2).
 */
export async function runMatVecQ8(run: Runner["run"], { weight, scale, vector, M, K }: MatVecQ8Args): Promise<Float32Array> {
  const wordsPerRow = Math.ceil(K / 4);
  return dispatchRowsChunked(M, async (rowStart, rowCount) => {
    const [out] = await run({
      code: CODE.matvecQ8,
      bindings: [
        { kind: "storage", data: weight.subarray(rowStart * wordsPerRow, (rowStart + rowCount) * wordsPerRow) },
        { kind: "storage", data: scale.subarray(rowStart, rowStart + rowCount) },
        { kind: "storage", data: vector },
        { kind: "out", type: "f32", length: rowCount },
        { kind: "uniform", data: params([["u32", rowCount], ["u32", K]]) },
      ],
      workgroups: [rowCount],
    });
    return asF32(out!);
  });
}

export interface MatMulArgs {
  /** `[M, K]` row-major. */
  a: Float32Array;
  /** `[K, N]` row-major. */
  b: Float32Array;
  M: number;
  N: number;
  K: number;
}

/** Must match `TILE` in `ops/matmul/wgsl/kernel.wgsl`. */
const MATMUL_TILE = 16;

export async function runMatMul(run: Runner["run"], { a, b, M, N, K }: MatMulArgs): Promise<Float32Array> {
  const [out] = await run({
    code: CODE.matmul,
    bindings: [
      { kind: "storage", data: a },
      { kind: "storage", data: b },
      { kind: "out", type: "f32", length: M * N },
      { kind: "uniform", data: params([["u32", M], ["u32", N], ["u32", K]]) },
    ],
    workgroups: [Math.ceil(N / MATMUL_TILE), Math.ceil(M / MATMUL_TILE)],
  });
  return asF32(out!);
}

export interface RopeArgs {
  /** `[N, numHeads, headDim]`, token-major. */
  input: Float32Array;
  N: number;
  numHeads: number;
  headDim: number;
  /** Position of the first token — the KV-cache continuation offset. */
  posOffset: number;
  thetaBase: number;
}

/**
 * Plain RoPE only — no cache table, no NTK/YaRN, no head range. This engine's
 * scope (issue #98) is the llama architecture's unscaled RoPE; `ops/rope`'s
 * other parameters exist for models this engine does not target yet.
 *
 * The cache binding is a two-float dummy with `cache_positions = 0`, which
 * `ops/rope/wgsl/kernel.wgsl` documents as "the uncached kernel, unchanged" —
 * read to confirm the fallback path recomputes rather than reading a table of
 * the wrong size, and that `layout: "auto"` still needs *something* bound at
 * that slot because the shader references it unconditionally.
 */
export async function runRope(run: Runner["run"], { input, N, numHeads, headDim, posOffset, thetaBase }: RopeArgs): Promise<Float32Array> {
  const totalPairs = (N * numHeads * headDim) / 2;
  const [out] = await run({
    code: CODE.rope,
    bindings: [
      { kind: "storage", data: input },
      { kind: "storage", data: new Float32Array(2) },
      { kind: "out", type: "f32", length: input.length },
      {
        kind: "uniform",
        data: params([
          ["u32", N],
          ["u32", numHeads],
          ["u32", headDim],
          ["u32", posOffset],
          ["u32", 0], // cache_positions: 0 disables the cache
          ["f32", thetaBase], // effective_base: unscaled RoPE uses thetaBase itself
          ["f32", 1], // interpolation_factor: 1, no YaRN
          ["f32", 0], // ramp_low: arbitrary when interpolation_factor is 1
          ["f32", 1], // ramp_high: likewise
          ["f32", 1], // attention_factor: 1, no YaRN gain
          ["u32", 0], // head_offset: rotate from head 0
          ["u32", numHeads], // head_count: rotate every head
        ]),
      },
    ],
    workgroups: [Math.ceil(totalPairs / 256)],
  });
  return asF32(out!);
}

export interface GqaArgs {
  /** `[H, L, D]` head-major (`B = 1` throughout this engine). */
  q: Float32Array;
  /** `[kvHeads, S, D]` head-major. */
  k: Float32Array;
  /** `[kvHeads, S, Dv]` head-major. */
  v: Float32Array;
  H: number;
  kvHeads: number;
  L: number;
  S: number;
  D: number;
  Dv: number;
  causal: boolean;
  /** Absolute position of query row 0 within the key sequence. */
  queryOffset: number;
  /** Defaults to `1/sqrt(D)`, `ops/gqa`'s own default. */
  scale?: number;
}

/**
 * Two dispatches — `scores` (QK^T, scale, causal mask, softmax) then `context`
 * (probs @ V) — because `ops/gqa` is two `.wgsl` files for the reason #46
 * records: `layout: "auto"` drops a binding an entry point never references,
 * so one shader with two entry points would silently lose one of them.
 *
 * `B = 1` throughout: this engine processes one sequence at a time (issue #98's
 * scope), so every `[B, ...]` shape in `ops/gqa` degenerates to its `B = 1`
 * case and the mask is always the `[B, 1, 1, S]` all-zero "no mask" bias —
 * causal is expressed by the `causal` flag, and this engine has no padding
 * mask to add (no batching yet).
 */
export async function runGqa(run: Runner["run"], args: GqaArgs): Promise<Float32Array> {
  const { q, k, v, H, kvHeads, L, S, D, Dv, causal, queryOffset } = args;
  const B = 1;
  const scale = args.scale ?? 1 / Math.sqrt(D);
  const mask = new Float32Array(B * S);

  const [probs] = await run({
    code: CODE.gqaScores,
    bindings: [
      { kind: "storage", data: q },
      { kind: "storage", data: k },
      { kind: "storage", data: mask },
      { kind: "out", type: "f32", length: B * H * L * S },
      {
        kind: "uniform",
        data: params([
          ["u32", H],
          ["u32", kvHeads],
          ["u32", L],
          ["u32", S],
          ["u32", D],
          ["f32", scale],
          ["u32", causal ? 1 : 0],
          ["i32", queryOffset],
          ["u32", B],
          ["u32", 1],
          ["u32", 1],
        ]),
      },
    ],
    workgroups: [L, H, B],
  });

  const [output] = await run({
    code: CODE.gqaContext,
    bindings: [
      { kind: "storage", data: asF32(probs!) },
      { kind: "storage", data: v },
      { kind: "out", type: "f32", length: B * H * L * Dv },
      { kind: "uniform", data: params([["u32", H], ["u32", kvHeads], ["u32", L], ["u32", S], ["u32", Dv]]) },
    ],
    workgroups: [L, H, B],
  });
  return asF32(output!);
}

export interface ActivationArgs {
  input: Float32Array;
  kind: ActivationKind;
  alpha?: number;
}

export async function runActivation(run: Runner["run"], { input, kind, alpha = 1 }: ActivationArgs): Promise<Float32Array> {
  const N = input.length;
  const [out] = await run({
    code: CODE.activation,
    bindings: [
      { kind: "storage", data: input },
      { kind: "out", type: "f32", length: N },
      { kind: "uniform", data: params([["u32", N], ["u32", kind], ["f32", alpha]]) },
    ],
    workgroups: [Math.ceil(N / 256)],
  });
  return asF32(out!);
}

export interface ElementwiseArgs {
  a: Float32Array;
  b: Float32Array;
  kind: ElementwiseKind;
}

export async function runElementwise(run: Runner["run"], { a, b, kind }: ElementwiseArgs): Promise<Float32Array> {
  const N = a.length;
  const [out] = await run({
    code: CODE.elementwise,
    bindings: [
      { kind: "storage", data: a },
      { kind: "storage", data: b },
      { kind: "out", type: "f32", length: N },
      { kind: "uniform", data: params([["u32", N], ["u32", kind]]) },
    ],
    workgroups: [Math.ceil(N / 256)],
  });
  return asF32(out!);
}
