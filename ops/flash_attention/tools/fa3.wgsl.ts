/**
 * FlashAttention-3's portable half: software pipelining.
 *
 * Issue #177. FA3 rests on three things, and **two of them are CUDA hardware
 * features that WebGPU does not expose.** Saying which is which up front is the
 * point of this file having its own header:
 *
 *   - **Warp specialisation with TMA.** A producer warp group issues
 *     asynchronous bulk copies while consumer warp groups compute. WGSL has no
 *     asynchronous copy and no way to address a subgroup as a scheduling unit —
 *     `workgroupBarrier` is the only synchronisation there is. **Not possible.**
 *   - **FP8 with incoherent processing.** This device does not report
 *     `shader-f16`, let alone f8. **Not possible.**
 *   - **Ping-pong scheduling**: overlap one tile's softmax and score arithmetic
 *     with the *next* tile's loads, so the memory latency is spent doing
 *     something. This needs no special instruction — only double-buffered
 *     staging and a loop that issues the next load before consuming the
 *     current one. **This is what this kernel is.**
 *
 * On an op that is memory-bound by a factor of sixty (see `fa2.wgsl.ts` for the
 * arithmetic), hiding the load latency is the one of the three that would have
 * mattered here anyway.
 *
 * The saving is also in barriers: FA2 stages, waits, scores, waits, softmaxes,
 * waits, accumulates, waits — four per tile. Here the staging of tile `t+1`
 * shares a barrier with the scoring of tile `t`, so there are three.
 *
 * **Whether it is faster is a measurement, not an argument.** `bench.ts` sweeps
 * both generations against the same reference in the same rounds.
 */
import { accumulate, epilogue, preamble, prologue, scores, softmax, stageTile } from "./parts.js";
import type { FlashShape } from "./shape.js";

/**
 * `q: [B, H, L, D]`, `k: [B, H, S, D]`, `v: [B, H, S, Dv]` to
 * `output: [B, H, L, Dv]`, dispatched `[ceil(L / BQ), H, B]` — the same
 * interface as FA2, so a caller swaps one for the other and nothing else.
 */
export function fa3Flash(shape: FlashShape, maxD: number, maxDv = maxD): string {
  // Registers per thread for the prefetched tile, when it is prefetched into
  // registers. The staging loops stride by THREADS, so a thread touches at
  // most this many elements — `params.D` is bounded by `maxD` at dispatch.
  const vec = shape.scoreReads === "vec4";
  // In vec4 mode a "unit" of k is four channels, so a thread touches a quarter
  // as many of them.
  const kUnits = vec ? Math.ceil(maxD / 4) : maxD;
  const pk = Math.max(1, Math.ceil((shape.tileS * kUnits) / shape.threads));
  const pv = Math.max(1, Math.ceil((shape.tileS * maxDv) / shape.threads));

  // Straight into the idle half of the buffer, before the scores. The store
  // waits on the load, so this only overlaps if the compiler hoists the score
  // arithmetic above it — which it is free to do (sk, sv, sq and ss are
  // distinct workgroup variables and WGSL has no aliasing between them) but is
  // not obliged to.
  const DIRECT_ISSUE = `      if (t + 1u < tiles) {
${stageTile(shape, "(t + 1u) * TILE_S", "1u - cur", "        ")}
      }`;
  const DIRECT_LAND = "";

  // Into registers first. The global loads are issued here, the scores and the
  // softmax happen, and only then does the thread wait on them to store — so
  // the arithmetic covers the latency whatever the compiler decides.
  const KEY_INTO_REGISTERS = vec
    ? `        for (var e = tid; e < TILE_S * d4n; e = e + THREADS) {
          let j = next + e / d4n;
          let c0 = (e % d4n) * 4u;
          var val = vec4<f32>(0.0, 0.0, 0.0, 0.0);
          if (j < params.S) {
            for (var c = 0u; c < 4u; c = c + 1u) {
              let d = c0 + c;
              if (d < params.D) { val[c] = k[k_head + j * params.D + d]; }
            }
          }
          pk[p] = val;
          p = p + 1u;
        }`
    : `        for (var e = tid; e < TILE_S * params.D; e = e + THREADS) {
          let j = next + e / params.D;
          let d = e % params.D;
          pk[p] = select(0.0, k[k_head + j * params.D + d], j < params.S);
          p = p + 1u;
        }`;
  // The same padded row stride the staging path uses; `e` walks the tile
  // densely, the destination does not.
  const unitsPerRow = vec ? "d4n" : "params.D";
  const KEY_OUT_OF_REGISTERS = `        for (var e = tid; e < TILE_S * ${unitsPerRow}; e = e + THREADS) {
          sk[(1u - cur) * K_STRIDE + (e / ${unitsPerRow}) * ROW + (e % ${unitsPerRow})] = pk[p];
          p = p + 1u;
        }`;

  const REGISTER_ISSUE = `      var pk: array<${vec ? "vec4<f32>" : "f32"}, ${pk}>;
      var pv: array<f32, ${pv}>;
      if (t + 1u < tiles) {
        let next = (t + 1u) * TILE_S;
        var p = 0u;
${KEY_INTO_REGISTERS}
        p = 0u;
        for (var e = tid; e < TILE_S * params.Dv; e = e + THREADS) {
          let j = next + e / params.Dv;
          let d = e % params.Dv;
          pv[p] = select(0.0, v[v_head + j * params.Dv + d], j < params.S);
          p = p + 1u;
        }
      }`;
  const REGISTER_LAND = `      if (t + 1u < tiles) {
        var p = 0u;
${KEY_OUT_OF_REGISTERS}
        p = 0u;
        for (var e = tid; e < TILE_S * params.Dv; e = e + THREADS) {
          sv[(1u - cur) * V_STRIDE + e] = pv[p];
          p = p + 1u;
        }
      }`;

  const issue = shape.prefetch === "registers" ? REGISTER_ISSUE : DIRECT_ISSUE;
  const land = shape.prefetch === "registers" ? REGISTER_LAND : DIRECT_LAND;

  return `// FlashAttention-3's pipelining, minus the parts that need CUDA. Generated by
// ops/flash_attention/tools/generate.ts — edit tools/fa3.wgsl.ts and
// regenerate rather than editing this file.
//
// Layout:
//   q:      [B, H, L, D]  f32, row-major
//   k:      [B, H, S, D]  f32
//   v:      [B, H, S, Dv] f32
//   mask:   broadcast over [mask_batch, mask_heads, mask_rows, S]
//   output: [B, H, L, Dv] f32
${preamble(shape, maxD, maxDv, "fa3")}
@compute @workgroup_size(${shape.threads})
fn main(
  @builtin(workgroup_id) wg: vec3<u32>,
  @builtin(local_invocation_index) tid: u32,
) {
${prologue(shape, shape.threads, maxDv)}

  // Prime the pipeline: tile 0 into half 0, before the loop.
${stageTile(shape, "0u", "0u", "  ")}

  var cur = 0u;
  for (var t = 0u; t < tiles; t = t + 1u) {
    let base = t * TILE_S;
    // Half \`cur\` now holds tile t, and half \`1 - cur\` was last read during
    // the previous iteration's accumulate — so both "tile t is visible" and
    // "the other half is free to overwrite" hold past this one barrier.
    workgroupBarrier();

${issue}

${scores(shape, "base", "cur", "    ")}
    workgroupBarrier();

${softmax("    ")}
    workgroupBarrier();

${accumulate(shape.accumulate, "cur", "    ")}
${land}
    cur = 1u - cur;
  }

${epilogue()}
}
`;
}
