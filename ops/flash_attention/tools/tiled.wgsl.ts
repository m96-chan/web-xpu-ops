/**
 * A flash-attention kernel with its shape as a parameter, for sweeping.
 *
 * Issue #177. `wgsl/kernel.wgsl` reaches 3.3% of an RTX 5090's measured
 * roofline and is 80.7% of an Anima forward once the two matmuls are tiled.
 *
 * **Reading it suggested three fixes, all of which were worth nothing.** The
 * score phase uses 64 of 256 threads, the accumulate phase 128 of 256, and `q`
 * is re-read from global `S` times. Fixing all three measured **1.0x**. What
 * the arithmetic says instead:
 *
 *     one workgroup = one query row
 *     FLOPs   2 * S * D  (scores) + 2 * S * Dv (accumulate) = 2.02 M
 *     bytes   S * D * 4  (all of k) + S * Dv * 4 (all of v) = 4.05 MB
 *     -> 0.50 FLOP/byte, against a roofline crossover of 30
 *
 * It is memory-bound by a factor of sixty, so no amount of arranging the
 * arithmetic moves it. `k` and `v` are read once per *query row* — 3,952 times
 * per head — and the only thing that changes that is carrying more than one
 * query in a workgroup, which divides the traffic by exactly that number.
 *
 *     BQ = 8  ->  4 FLOP/byte
 *     BQ = 32 -> 16 FLOP/byte
 *
 * That is FlashAttention's prefill tile, and what llama.cpp's paper calls its
 * "tile path" as against its decode path. This template is that: `BQ` query
 * rows per workgroup, with the key and value tiles staged in workgroup memory
 * so every one of the `BQ` rows reads them from there rather than from global.
 *
 * The online recurrence is unchanged and is not the sort of thing to improvise:
 * it is `reference.ts`'s, which the tests hold both to. Each of the `BQ` rows
 * carries its own running maximum and sum.
 */

export interface FlashShape {
  /** Query rows per workgroup — the whole point. */
  bq: number;
  /** Keys staged per pass. */
  tileS: number;
  /** Threads per workgroup. */
  threads: number;
}

export const WORKGROUP_STORAGE_LIMIT = 49152;
export const INVOCATION_LIMIT = 1024;

/** Workgroup storage in bytes for a given head dimension. */
export function flashStorageBytes({ bq, tileS }: FlashShape, D: number): number {
  // q rows, the staged k and v tiles, one score per (row, key), and the
  // per-row softmax state — a running maximum, a running sum, and the
  // correction the tile's new maximum implies.
  return (bq * D + tileS * D + tileS * D + bq * tileS + bq * 3) * 4;
}

/** Why a shape cannot be dispatched, or null. */
export function rejectReason(shape: FlashShape, D: number): string | null {
  if (shape.threads > INVOCATION_LIMIT) return `${shape.threads} invocations exceeds ${INVOCATION_LIMIT}`;
  const bytes = flashStorageBytes(shape, D);
  if (bytes > WORKGROUP_STORAGE_LIMIT) return `${bytes} B exceeds ${WORKGROUP_STORAGE_LIMIT}`;
  if ((shape.tileS * D) % shape.threads !== 0) return "k/v tile is not a multiple of the thread count";
  if ((shape.bq * shape.tileS) % shape.threads !== 0) return "score tile is not a multiple of the thread count";
  return null;
}

/**
 * `q: [B, H, L, D]`, `k: [B, H, S, D]`, `v: [B, H, S, Dv]` to
 * `output: [B, H, L, Dv]`, dispatched `[ceil(L / BQ), H, B]`.
 *
 * **The grid is not one workgroup per query** — it is per `BQ` queries.
 *
 * `maxD` sizes the workgroup arrays and nothing else: WGSL fixes those at
 * compile time, but every *loop* here runs to `params.D`, so one program serves
 * every head dimension up to `maxD`. The first version baked `D` in and turned
 * a general op into a specialised one — `ops/flash_attention`'s own tests cover
 * D of 8, 16 and 64, and all of them broke.
 */
export function tiledFlash(shape: FlashShape, maxD: number, maxDv = maxD): string {
  const { bq, tileS, threads } = shape;
  const D = maxD;
  // How many channels one thread carries. Dv can exceed the workgroup.
  const sweeps = Math.max(1, Math.ceil(maxDv / threads));
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
const MAX_D: u32 = ${D}u;
const MASKED: f32 = -3.402823e+38;

var<workgroup> sq: array<f32, ${bq * D}>;
var<workgroup> sk: array<f32, ${tileS * D}>;
var<workgroup> sv: array<f32, ${tileS * maxDv}>;
var<workgroup> ss: array<f32, ${bq * tileS}>;
// The softmax state, per row, **shared**.
//
// It used to be m[BQ] and l[BQ] in every thread's registers, which is
// where BQ stopped: they are per-row values, so every thread owning a channel
// of the same row held its own identical copy. At BQ=16 that is 32 f32 of
// duplicate state per thread on top of the accumulators, and BQ=32 measured
// *slower* than BQ=16 for that reason while workgroup storage sat two-thirds
// empty.
var<workgroup> smax: array<f32, ${bq}>;
var<workgroup> ssum: array<f32, ${bq}>;
var<workgroup> scorr: array<f32, ${bq}>;

@compute @workgroup_size(${threads})
fn main(
  @builtin(workgroup_id) wg: vec3<u32>,
  @builtin(local_invocation_index) tid: u32,
) {
  let q0 = wg.x * BQ;
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
  // channel this thread owns). Each thread owns one channel of every row.
  // A thread owns one channel per sweep, and sweeps until Dv is covered — Dv
  // can exceed the workgroup (the tests go to 300), which the first version
  // silently truncated to the first THREADS channels.
  const SWEEPS: u32 = ${sweeps}u;
  var acc: array<array<f32, SWEEPS>, BQ>;
  for (var r = 0u; r < BQ; r = r + 1u) {
    for (var p = 0u; p < SWEEPS; p = p + 1u) { acc[r][p] = 0.0; }
  }
  if (tid < BQ) { smax[tid] = MASKED; ssum[tid] = 0.0; }

  let tiles = (params.S + TILE_S - 1u) / TILE_S;
  for (var t = 0u; t < tiles; t = t + 1u) {
    let base = t * TILE_S;
    workgroupBarrier();
    // D and Dv can differ; k and v are staged with their own strides rather
    // than one shared constant.
    for (var e = tid; e < TILE_S * params.D; e = e + THREADS) {
      let j = base + e / params.D;
      let d = e % params.D;
      sk[e] = select(0.0, k[k_head + j * params.D + d], j < params.S);
    }
    for (var e = tid; e < TILE_S * params.Dv; e = e + THREADS) {
      let j = base + e / params.Dv;
      let d = e % params.Dv;
      sv[e] = select(0.0, v[v_head + j * params.Dv + d], j < params.S);
    }
    workgroupBarrier();

    // Scores: BQ * TILE_S of them, spread over every thread. Both operands
    // come out of workgroup memory, which is the entire point.
    for (var e = tid; e < BQ * TILE_S; e = e + THREADS) {
      let r = e / TILE_S;
      let slot = e % TILE_S;
      let j = base + slot;
      let row = q0 + r;
      var value = MASKED;
      if (j < params.S && row < params.L
          && (params.causal == 0u || i32(j) <= i32(row) + params.query_offset)) {
        var dot = 0.0;
        for (var d = 0u; d < params.D; d = d + 1u) {
          dot = fma(sq[r * params.D + d], sk[slot * params.D + d], dot);
        }
        let mr = select(0u, row, params.mask_rows > 1u);
        value = dot * params.scale + mask[((mb * params.mask_heads + mh) * params.mask_rows + mr) * params.S + j];
      }
      ss[e] = value;
    }
    workgroupBarrier();

    // One thread per row updates that row's state and turns the scores into
    // weights in place, so the accumulate loop below reads weights rather than
    // recomputing exp once per channel.
    if (tid < BQ) {
      var best = MASKED;
      for (var slot = 0u; slot < TILE_S; slot = slot + 1u) { best = max(best, ss[tid * TILE_S + slot]); }
      let m_new = max(smax[tid], best);
      let corr = select(exp(smax[tid] - m_new), 0.0, smax[tid] == MASKED);
      var sum = 0.0;
      for (var slot = 0u; slot < TILE_S; slot = slot + 1u) {
        let s = ss[tid * TILE_S + slot];
        let w = select(exp(s - m_new), 0.0, s == MASKED);
        ss[tid * TILE_S + slot] = w;
        sum = sum + w;
      }
      ssum[tid] = ssum[tid] * corr + sum;
      scorr[tid] = corr;
      smax[tid] = m_new;
    }
    workgroupBarrier();

    for (var r = 0u; r < BQ; r = r + 1u) {
      let corr = scorr[r];
      var weighted: array<f32, SWEEPS>;
      for (var p = 0u; p < SWEEPS; p = p + 1u) { weighted[p] = 0.0; }
      for (var slot = 0u; slot < TILE_S; slot = slot + 1u) {
        let w = ss[r * TILE_S + slot];
        var p = 0u;
        for (var c = tid; c < params.Dv; c = c + THREADS) {
          weighted[p] = fma(w, sv[slot * params.Dv + c], weighted[p]);
          p = p + 1u;
        }
      }
      for (var p = 0u; p < SWEEPS; p = p + 1u) { acc[r][p] = acc[r][p] * corr + weighted[p]; }
    }
  }

  workgroupBarrier();
  for (var r = 0u; r < BQ; r = r + 1u) {
    let row = q0 + r;
    if (row >= params.L) { continue; }
    let denom = ssum[r];
    var p = 0u;
    for (var c = tid; c < params.Dv; c = c + THREADS) {
      output[(head * params.L + row) * params.Dv + c] = select(acc[r][p] / denom, 0.0, denom == 0.0);
      p = p + 1u;
    }
  }
}
`;
}
