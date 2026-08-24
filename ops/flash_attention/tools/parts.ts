/**
 * The pieces every flash-attention generation here is built from.
 *
 * Issue #177. FA2 and FA3 differ in **when** the tile loop does things, not in
 * what it does: the same q staging, the same scores, the same online softmax
 * recurrence, the same accumulate, the same epilogue. Writing those out twice
 * would let the two drift, and the one that must not drift is the recurrence —
 * it is `reference.ts`'s, which the tests hold both generations to, and it is
 * not the sort of thing to improvise per file.
 *
 * So the fragments live here and the schedule lives in `fa2.wgsl.ts` and
 * `fa3.wgsl.ts`. Each fragment takes the expressions it needs to address a
 * staging buffer, because FA3 has two of them and FA2 has one.
 */
import { sweepsFor, type FlashShape, type Generation } from "./shape.js";

export const MASKED = "-3.402823e+38";

/**
 * Bindings, constants and workgroup arrays.
 *
 * `maxD` sizes the workgroup arrays and nothing else: WGSL fixes those at
 * compile time, but every *loop* below runs to `params.D`, so one program
 * serves every head dimension up to `maxD`. The first version baked `D` in and
 * turned a general op into a specialised one — `ops/flash_attention`'s own
 * tests cover D of 8, 16 and 64, and all of them broke.
 */
export function preamble(shape: FlashShape, maxD: number, maxDv: number, generation: Generation): string {
  const { bq, tileS, threads } = shape;
  const buffers = generation === "fa3" ? 2 : 1;
  return `
struct Params {
  H: u32, L: u32, S: u32, D: u32, Dv: u32,
  scale: f32, causal: u32, query_offset: i32,
  mask_batch: u32, mask_heads: u32, mask_rows: u32,
}

@group(0) @binding(0) var<storage, read> q: array<f32>;
@group(0) @binding(1) var<storage, read> k: array<f32>;
@group(0) @binding(2) var<storage, read> v: array<f32>;
@group(0) @binding(3) var<storage, read> mask: array<f32>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;
@group(0) @binding(5) var<uniform> params: Params;

const BQ: u32 = ${bq}u;
const TILE_S: u32 = ${tileS}u;
const THREADS: u32 = ${threads}u;
const MAX_D: u32 = ${maxD}u;
const MAX_DV: u32 = ${maxDv}u;
const K_STRIDE: u32 = ${tileS * maxD}u;
const V_STRIDE: u32 = ${tileS * maxDv}u;
const MASKED: f32 = ${MASKED};

var<workgroup> sq: array<f32, ${bq * maxD}>;
var<workgroup> sk: array<f32, ${buffers * tileS * maxD}>;
var<workgroup> sv: array<f32, ${buffers * tileS * maxDv}>;
var<workgroup> ss: array<f32, ${bq * tileS}>;
// The softmax state, per row, **shared**.
//
// It used to be m[BQ] and l[BQ] in every thread's registers, which is where BQ
// stopped: they are per-row values, so every thread owning a channel of the
// same row held its own identical copy. At BQ=16 that is 32 f32 of duplicate
// state per thread on top of the accumulators, and BQ=32 measured *slower*
// than BQ=16 for that reason while workgroup storage sat two-thirds empty.
var<workgroup> smax: array<f32, ${bq}>;
var<workgroup> ssum: array<f32, ${bq}>;
var<workgroup> scorr: array<f32, ${bq}>;
`;
}

/** Head/batch addressing and the q tile, which is staged once and read every tile. */
export function prologue(shape: FlashShape, threads: number, maxDv: number): string {
  return `  let q0 = wg.x * BQ;
  let batch = wg.z;
  let h = wg.y;
  let head = batch * params.H + h;
  let k_head = head * params.S * params.D;
  let v_head = head * params.S * params.Dv;

  let mb = select(0u, batch, params.mask_batch > 1u);
  let mh = select(0u, h, params.mask_heads > 1u);

  for (var t = tid; t < BQ * params.D; t = t + THREADS) {
    let r = t / params.D;
    let d = t % params.D;
    let row = q0 + r;
    sq[t] = select(0.0, q[(head * params.L + row) * params.D + d], row < params.L);
  }

  // One running maximum, sum and accumulator per (row this thread owns,
  // channel this thread owns). A thread owns one channel per sweep, and sweeps
  // until Dv is covered — Dv can exceed the workgroup (the tests go to 300),
  // which the first version silently truncated to the first THREADS channels.
  const SWEEPS: u32 = ${sweepsFor(threads, maxDv)}u;
  var acc: array<array<f32, SWEEPS>, BQ>;
  for (var r = 0u; r < BQ; r = r + 1u) {
    for (var p = 0u; p < SWEEPS; p = p + 1u) { acc[r][p] = 0.0; }
  }
  if (tid < BQ) { smax[tid] = MASKED; ssum[tid] = 0.0; }

  let tiles = (params.S + TILE_S - 1u) / TILE_S;`;
}

/**
 * Stages keys and values for the tile beginning at `base` into buffer `buf`.
 *
 * `indent` only so the generated WGSL reads like something a person wrote; a
 * kernel nobody can read is a kernel nobody can check against the reference.
 */
export function stageTile(baseExpr: string, bufExpr: string, indent: string): string {
  return `${indent}// D and Dv can differ; k and v are staged with their own strides rather
${indent}// than one shared constant.
${indent}for (var e = tid; e < TILE_S * params.D; e = e + THREADS) {
${indent}  let j = (${baseExpr}) + e / params.D;
${indent}  let d = e % params.D;
${indent}  sk[(${bufExpr}) * K_STRIDE + e] = select(0.0, k[k_head + j * params.D + d], j < params.S);
${indent}}
${indent}for (var e = tid; e < TILE_S * params.Dv; e = e + THREADS) {
${indent}  let j = (${baseExpr}) + e / params.Dv;
${indent}  let d = e % params.Dv;
${indent}  sv[(${bufExpr}) * V_STRIDE + e] = select(0.0, v[v_head + j * params.Dv + d], j < params.S);
${indent}}`;
}

/** `BQ * TILE_S` scores, spread over every thread, both operands from workgroup memory. */
export function scores(baseExpr: string, bufExpr: string, indent: string): string {
  return `${indent}for (var e = tid; e < BQ * TILE_S; e = e + THREADS) {
${indent}  let r = e / TILE_S;
${indent}  let slot = e % TILE_S;
${indent}  let j = (${baseExpr}) + slot;
${indent}  let row = q0 + r;
${indent}  var value = MASKED;
${indent}  if (j < params.S && row < params.L
${indent}      && (params.causal == 0u || i32(j) <= i32(row) + params.query_offset)) {
${indent}    var dot = 0.0;
${indent}    for (var d = 0u; d < params.D; d = d + 1u) {
${indent}      dot = fma(sq[r * params.D + d], sk[(${bufExpr}) * K_STRIDE + slot * params.D + d], dot);
${indent}    }
${indent}    let mr = select(0u, row, params.mask_rows > 1u);
${indent}    value = dot * params.scale + mask[((mb * params.mask_heads + mh) * params.mask_rows + mr) * params.S + j];
${indent}  }
${indent}  ss[e] = value;
${indent}}`;
}

/**
 * The online softmax recurrence — `reference.ts`'s, and the one thing here that
 * is not a scheduling choice.
 *
 * One thread per row updates that row's state and turns the scores into weights
 * in place, so the accumulate loop reads weights rather than recomputing `exp`
 * once per channel. The `1 / l` division is deferred to the epilogue, which is
 * FlashAttention-2's change to FlashAttention-1 and is why the FA2 kernel is
 * called that.
 */
export function softmax(indent: string): string {
  return `${indent}if (tid < BQ) {
${indent}  var best = MASKED;
${indent}  for (var slot = 0u; slot < TILE_S; slot = slot + 1u) { best = max(best, ss[tid * TILE_S + slot]); }
${indent}  let m_new = max(smax[tid], best);
${indent}  let corr = select(exp(smax[tid] - m_new), 0.0, smax[tid] == MASKED);
${indent}  var sum = 0.0;
${indent}  for (var slot = 0u; slot < TILE_S; slot = slot + 1u) {
${indent}    let s = ss[tid * TILE_S + slot];
${indent}    let w = select(exp(s - m_new), 0.0, s == MASKED);
${indent}    ss[tid * TILE_S + slot] = w;
${indent}    sum = sum + w;
${indent}  }
${indent}  ssum[tid] = ssum[tid] * corr + sum;
${indent}  scorr[tid] = corr;
${indent}  smax[tid] = m_new;
${indent}}`;
}

/** Folds this tile's weighted values into the accumulators. See `FlashShape.accumulate`. */
export function accumulate(mode: "row" | "key", bufExpr: string, indent: string): string {
  if (mode === "row") {
    // Rows outermost: each staged value of v is read once per row.
    return `${indent}for (var r = 0u; r < BQ; r = r + 1u) {
${indent}  let corr = scorr[r];
${indent}  var weighted: array<f32, SWEEPS>;
${indent}  for (var p = 0u; p < SWEEPS; p = p + 1u) { weighted[p] = 0.0; }
${indent}  for (var slot = 0u; slot < TILE_S; slot = slot + 1u) {
${indent}    let w = ss[r * TILE_S + slot];
${indent}    var p = 0u;
${indent}    for (var c = tid; c < params.Dv; c = c + THREADS) {
${indent}      weighted[p] = fma(w, sv[(${bufExpr}) * V_STRIDE + slot * params.Dv + c], weighted[p]);
${indent}      p = p + 1u;
${indent}    }
${indent}  }
${indent}  for (var p = 0u; p < SWEEPS; p = p + 1u) { acc[r][p] = acc[r][p] * corr + weighted[p]; }
${indent}}`;
  }
  // Keys outermost: v is read once and every row uses it from a register. The
  // rescale by corr has to happen before the tile rather than folded into it,
  // since acc is touched by every key now.
  return `${indent}for (var r = 0u; r < BQ; r = r + 1u) {
${indent}  let corr = scorr[r];
${indent}  for (var p = 0u; p < SWEEPS; p = p + 1u) { acc[r][p] = acc[r][p] * corr; }
${indent}}
${indent}for (var slot = 0u; slot < TILE_S; slot = slot + 1u) {
${indent}  var p = 0u;
${indent}  for (var c = tid; c < params.Dv; c = c + THREADS) {
${indent}    let value = sv[(${bufExpr}) * V_STRIDE + slot * params.Dv + c];
${indent}    for (var r = 0u; r < BQ; r = r + 1u) {
${indent}      acc[r][p] = fma(ss[r * TILE_S + slot], value, acc[r][p]);
${indent}    }
${indent}    p = p + 1u;
${indent}  }
${indent}}`;
}

/** The deferred `1 / l`, and the write out. */
export function epilogue(): string {
  return `  workgroupBarrier();
  for (var r = 0u; r < BQ; r = r + 1u) {
    let row = q0 + r;
    if (row >= params.L) { continue; }
    let denom = ssum[r];
    var p = 0u;
    for (var c = tid; c < params.Dv; c = c + THREADS) {
      output[(head * params.L + row) * params.Dv + c] = select(acc[r][p] / denom, 0.0, denom == 0.0);
      p = p + 1u;
    }
  }`;
}
