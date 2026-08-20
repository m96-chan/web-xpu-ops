import { describe, expect, it } from "vitest";
import { gpuTest, useGpu } from "../../harness/index.js";
import { groupedAttention, type GroupedAttentionArgs, type GroupedAttentionResult } from "./reference.js";
import { check, prepare, wave } from "./testing.js";

/**
 * `sEff` (issue #117's whole point): the GPU dispatch cases and the CPU-only
 * validation/equivalence checks, split out of `wgsl.test.ts` into their own
 * file for the dispatch budget (#68) — see `testing.ts`'s own doc for the
 * full history, including the earlier, incorrect diagnosis (module-collect-time
 * CPU cost) this split supersedes.
 */

/**
 * `S` is allocated large (a stand-in for `maxSeqLen`), and every case below
 * sets `sEff` to exactly what the query rows need — the tightest legal value,
 * `min(S, L + queryOffset)` (`sEff`'s own doc in reference.ts). K and V past
 * `sEff` are poisoned, but the two kernels need *different* poison to be
 * caught, because they read past-`sEff` memory under genuinely different
 * conditions:
 *
 * - `scores.wgsl` only reads `k` (and `mask`) inside
 *   `if (causal == 0u || j <= i + query_offset)` — and `sEff`'s own contract
 *   requires `causal == true` with `sEff >= that same bound` whenever
 *   `sEff < S`, which means **every** position `[sEff, S)` is already
 *   causally excluded for every row, for any legal `sEff`. So a `k` value at
 *   `k`'s address is never read there regardless of whether the loop that
 *   would reach it stops at `s_eff` or continues to `S` — the poison in `k`
 *   below is a decoy, kept only so a reader checking this file does not
 *   wonder why `k` looks unpoisoned. Reverting `scores.wgsl`'s loop bound
 *   from `s_eff` back to `S` is real (it does more iterations and more
 *   `probs` writes — the bandwidth issue #116 measured), but it is *not*
 *   something any legal input can catch by comparing `probs` values: the
 *   extra iterations all take the `else` branch and write the same `MASKED`
 *   sentinel a softmax turns into the same `0` a fresh buffer already held.
 *   `seffEquivalence()` below is the correctness proof for `scores.wgsl`
 *   instead of a poison check, for exactly this reason.
 * - `context.wgsl` reads `v` **unconditionally** —
 *   `acc += probs[p_row + j] * v[v_head + j * params.Dv + c]` has no `if` —
 *   because by the time `context.wgsl` runs, `scores.wgsl` has already
 *   turned every masked column into a plain `0.0` in `probs`, and
 *   `context.wgsl` never re-derives which columns those were. `0 * poison`
 *   is `0` for any finite poison, so a *finite* decoy would be silently
 *   swallowed here too — this is why `v`'s poison below is `+Infinity`:
 *   `0 * Infinity` is `NaN` in IEEE 754, and `NaN` cannot round-trip through
 *   `agree()`'s tolerance check as a pass (`harness/agree.ts`: a `NaN`
 *   difference fails both the `abs` and `rel` bounds, since every comparison
 *   against `NaN` is `false`). A kernel that scanned `S` instead of `s_eff`
 *   in `context.wgsl` reads that `Infinity`, multiplies it by the `0` at that
 *   column, and produces `NaN` in `output` — caught here, at every case,
 *   while every other test in `wgsl.test.ts` (none of which sets `sEff`, or
 *   pass finite `v`) stays green.
 */
const SEFF_K_DECOY = 1e4;

const SEFF_CASES: { name: string; args: GroupedAttentionArgs }[] = (() => {
  const decode = (() => {
    // Decode-shaped: L = 1, causal, queryOffset = the position being
    // generated. sEff = queryOffset + 1 is exactly what LlamaEngineQ8Resident's
    // decode step passes (llm/engine-q8-resident.ts) — everything at or past
    // it in the cache has not been written yet.
    const [B, H, kvHeads, L, S, D, Dv] = [1, 4, 2, 1, 16, 8, 8];
    const queryOffset = 5;
    const sEff = queryOffset + 1;
    return {
      name: `decode-shaped: L=1 causal@${queryOffset}, sEff=${sEff} of S=${S}`,
      args: {
        q: wave(B * H * L * D, 0.37),
        // Row-major per KV head: live rows [0, sEff), decoy rows [sEff, S) —
        // never read under any legal sEff (see this file's own doc).
        k: Float32Array.from({ length: B * kvHeads * S * D }, (_, idx) => {
          const j = Math.floor(idx / D) % S;
          return j < sEff ? wave(D, 0.11, 0.5 + idx * 1e-6)[idx % D]! : SEFF_K_DECOY;
        }),
        // Row-major per KV head: live rows [0, sEff), +Infinity rows
        // [sEff, S) — the mutation-sensitive poison (see this file's own doc).
        v: Float32Array.from({ length: B * kvHeads * S * Dv }, (_, idx) => {
          const j = Math.floor(idx / Dv) % S;
          return j < sEff ? wave(Dv, 0.23, 1.25 + idx * 1e-6)[idx % Dv]! : Number.POSITIVE_INFINITY;
        }),
        B, H, kvHeads, L, S, D, Dv, causal: true, queryOffset, sEff,
      } satisfies GroupedAttentionArgs,
    };
  })();

  const prefill = (() => {
    // Prefill-shaped: L > 1 in one batched dispatch, sEff set to exactly the
    // last row's reach (L + queryOffset) — the tight bound every row in the
    // batch is allowed to use (earlier rows need less, and the per-element
    // causal check already stops them there).
    const [B, H, kvHeads, L, S, D, Dv] = [1, 4, 2, 5, 20, 8, 8];
    const queryOffset = 3;
    const sEff = L + queryOffset;
    return {
      name: `prefill-shaped: L=${L} causal@${queryOffset}, sEff=${sEff} of S=${S}`,
      args: {
        q: wave(B * H * L * D, 0.41),
        k: Float32Array.from({ length: B * kvHeads * S * D }, (_, idx) => {
          const j = Math.floor(idx / D) % S;
          return j < sEff ? wave(D, 0.13, 0.5 + idx * 1e-6)[idx % D]! : SEFF_K_DECOY;
        }),
        v: Float32Array.from({ length: B * kvHeads * S * Dv }, (_, idx) => {
          const j = Math.floor(idx / Dv) % S;
          return j < sEff ? wave(Dv, 0.29, 1.25 + idx * 1e-6)[idx % Dv]! : Number.POSITIVE_INFINITY;
        }),
        B, H, kvHeads, L, S, D, Dv, causal: true, queryOffset, sEff,
      } satisfies GroupedAttentionArgs,
    };
  })();

  return [decode, prefill];
})();

const SEFF_TESTS = SEFF_CASES.map(({ name, args }) => prepare(name, args));

/**
 * `sEff = n` must equal `S = n` outright (K/V sliced to `n`, no poison): the
 * scanned positions are summed identically, not merely "the poisoned tail was
 * avoided" (see `SEFF_CASES`'s doc for why that is a materially different
 * claim from the poison check above).
 *
 * A plain function, not a module-scope `const ... = SEFF_CASES.map(...)`:
 * an earlier version computed every case's `groupedAttention()` pair (this
 * function's own body, twice per case) at module *collect* time, alongside
 * `SEFF_CASES`'/`SEFF_TESTS`'s own module-scope work — suspected, at the time,
 * to reproduce this file's Node/Dawn worker instability. That suspicion did
 * not hold up (PR #119's second review round: the actual cause is #68 — too
 * many GPU dispatches in one file, unrelated to CPU cost at collect time —
 * and this whole file's *existence*, not this function's laziness, is what
 * fixes it). The lazy shape stays regardless, since it is still the right one
 * — the same one every other per-case computation in this file already uses
 * (`check()`, `prepare()`'s own callers) — just not, in hindsight, load-bearing
 * for the instability it was first written to fix.
 */
function seffEquivalence(args: GroupedAttentionArgs): { truncated: GroupedAttentionResult; shrunk: GroupedAttentionResult } {
  const { B, kvHeads, D, Dv, sEff } = args;
  const n = sEff!;
  const kFull = args.k;
  const vFull = args.v;
  const kShrunk = new Float32Array(B * kvHeads * n * D);
  const vShrunk = new Float32Array(B * kvHeads * n * Dv);
  for (let bh = 0; bh < B * kvHeads; bh += 1) {
    kShrunk.set(kFull.subarray(bh * args.S * D, bh * args.S * D + n * D), bh * n * D);
    vShrunk.set(vFull.subarray(bh * args.S * Dv, bh * args.S * Dv + n * Dv), bh * n * Dv);
  }
  return {
    truncated: groupedAttention(args),
    shrunk: groupedAttention({ ...args, S: n, k: kShrunk, v: vShrunk, sEff: undefined }),
  };
}

describe("gqa / wgsl / sEff", () => {
  useGpu();

  for (const p of SEFF_TESTS) {
    gpuTest(p.name, async (run) => {
      await check(run, p);
    });
  }

  for (const { name, args } of SEFF_CASES) {
    it(`sEff-truncated output equals S shrunk to sEff at ${name}`, () => {
      const { truncated, shrunk } = seffEquivalence(args);
      const { B, H, L, S } = args;
      const n = args.sEff!;
      // shrunk's probs buffer is `[.., n]` wide (S was shrunk to n outright);
      // truncated's is `[.., S]` wide with the tail at 0 (never scanned — see
      // `sEff`'s own doc). Pad shrunk out to the same shape to compare.
      const padded = new Float32Array(B * H * L * S);
      for (let row = 0; row < B * H * L; row += 1) padded.set(shrunk.probs.subarray(row * n, (row + 1) * n), row * S);
      expect(Array.from(truncated.probs)).toEqual(Array.from(padded));
      expect(Array.from(truncated.output)).toEqual(Array.from(shrunk.output));
    });
  }

  it("rejects an sEff outside [1, S]", () => {
    const [B, H, kvHeads, L, S, D, Dv] = [1, 4, 2, 1, 5, 4, 4];
    const args = {
      q: new Float32Array(B * H * L * D), k: new Float32Array(B * kvHeads * S * D), v: new Float32Array(B * kvHeads * S * Dv),
      B, H, kvHeads, L, S, D, Dv, causal: true,
    };
    expect(() => groupedAttention({ ...args, sEff: 0 })).toThrow(/sEff must be an integer in \[1, S=5\]/);
    expect(() => groupedAttention({ ...args, sEff: 6 })).toThrow(/sEff must be an integer in \[1, S=5\]/);
    expect(() => groupedAttention({ ...args, sEff: 2.5 })).toThrow(/sEff must be an integer in \[1, S=5\]/);
    expect(() => groupedAttention({ ...args, sEff: 5 })).not.toThrow();
  });

  it("rejects a truncated sEff without causal=true", () => {
    const [B, H, kvHeads, L, S, D, Dv] = [1, 4, 2, 1, 5, 4, 4];
    const args = {
      q: new Float32Array(B * H * L * D), k: new Float32Array(B * kvHeads * S * D), v: new Float32Array(B * kvHeads * S * Dv),
      B, H, kvHeads, L, S, D, Dv,
    };
    expect(() => groupedAttention({ ...args, sEff: 3 })).toThrow(/requires causal=true/);
    expect(() => groupedAttention({ ...args, sEff: 5 })).not.toThrow();
  });

  it("rejects a causal sEff too small for the last query row's reach", () => {
    // L=3, queryOffset=4: the last row (i=2) may attend up to key 2+4=6, so
    // sEff must be at least 7.
    const [B, H, kvHeads, L, S, D, Dv] = [1, 4, 2, 3, 10, 4, 4];
    const args = {
      q: new Float32Array(B * H * L * D), k: new Float32Array(B * kvHeads * S * D), v: new Float32Array(B * kvHeads * S * Dv),
      B, H, kvHeads, L, S, D, Dv, causal: true, queryOffset: 4,
    };
    expect(() => groupedAttention({ ...args, sEff: 6 })).toThrow(/must be at least 7/);
    expect(() => groupedAttention({ ...args, sEff: 7 })).not.toThrow();
    // S itself can be the tighter bound when S - 1 < L - 1 + queryOffset.
    expect(() => groupedAttention({ ...args, S: 6, k: new Float32Array(B * kvHeads * 6 * D), v: new Float32Array(B * kvHeads * 6 * Dv), sEff: 6 })).not.toThrow();
  });
});
