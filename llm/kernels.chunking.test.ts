import { describe, expect, it } from "vitest";
import type { Dispatch } from "../harness/wgsl.js";
import { runMatVec, runMatVecQ8 } from "./kernels.js";

/**
 * `runMatVec`/`runMatVecQ8`'s row-chunking (see `kernels.ts`'s own doc on
 * `MAX_WORKGROUPS_PER_DISPATCH` / `dispatchRowsChunked`) against a **fake**
 * `Runner["run"]` — no GPU, no `harness`, no `useGpu()`.
 *
 * Deliberately not a `gpuTest`. A real GPU test at this scale was written
 * first (`gpuTest`, real device, `M = MAX_WORKGROUPS_PER_DISPATCH + 2`,
 * checked against `matvec()`/`matvecQ8()`) and it is *not* what proved this
 * fix correct: a real dispatch at (or requesting) 65,535 workgroups
 * reproducibly crashed this repository's Node/Dawn binding — three runs in a
 * row, "The futex facility returned an unexpected error code." before any
 * assertion could run — the same Node/Dawn fragility family as issue
 * #38/#49/#107, and the exact reason real-model-scale verification for issue
 * #106 moved to a browser instead of staying in this Node suite. That test
 * was removed rather than left in the suite to flake red on every run here;
 * `examples/llm-demo`'s real generation against the real checkpoint
 * (`vocabSize=102400`, the actual `M` that triggered this bug) is this fix's
 * real-hardware proof, and it runs in a browser precisely because Node/Dawn
 * cannot be trusted at this scale — see the PR for that run's output.
 *
 * What stayed in Node, here: this test's `run` never touches a GPU, it
 * inspects and echoes back exactly what `runMatVec`/`runMatVecQ8` handed it,
 * so what it proves is the **slicing and reassembly** — that every dispatch
 * covers the right row range, no row is sent twice, no row is skipped, and
 * the concatenated result lands back in the right order. That bookkeeping is
 * where the actual defect was (issue #106: `lm_head`'s `M=102400` decode
 * dispatch silently produced all zeros because the whole dispatch exceeded
 * `maxComputeWorkgroupsPerDimension` and nothing chunked it) — not in the
 * WGSL math, which `q8.wgsl.test.ts` and `kernels.wgsl.test.ts` already check
 * on real hardware at small, Dawn-safe scale.
 */

const MAX_WORKGROUPS_PER_DISPATCH = 65535;

describe("runMatVec / runMatVecQ8 row-chunking (mocked Runner, no GPU)", () => {
  it("runMatVec: a single dispatch when M is within the limit", async () => {
    const calls: Dispatch[] = [];
    const run = async (dispatch: Dispatch) => {
      calls.push(dispatch);
      return [new Float32Array(dispatch.workgroups[0])];
    };
    const M = 5;
    const K = 3;
    await runMatVec(run, { matrix: new Float32Array(M * K), vector: new Float32Array(K), M, K });
    expect(calls).toHaveLength(1);
    expect(calls[0]!.workgroups).toEqual([M]);
  });

  it("runMatVec: splits M > 65535 into chunks that exactly tile [0, M), in order, with no gap or overlap", async () => {
    const K = 1; // one column, so matrix[row] is that row's own tag value.
    const M = MAX_WORKGROUPS_PER_DISPATCH + 2;
    const matrix = Float32Array.from({ length: M }, (_, row) => row);

    const calls: Dispatch[] = [];
    const run = async (dispatch: Dispatch) => {
      calls.push(dispatch);
      // Echo the sliced matrix straight back as "output" — a stand-in for
      // the identity function, so the concatenated result reveals exactly
      // which rows each dispatch believed it was covering.
      const sliced = dispatch.bindings[0] as { kind: "storage"; data: Float32Array };
      return [Float32Array.from(sliced.data)];
    };

    const got = await runMatVec(run, { matrix, vector: new Float32Array(K), M, K });

    expect(calls).toHaveLength(2);
    expect(calls[0]!.workgroups).toEqual([MAX_WORKGROUPS_PER_DISPATCH]);
    expect(calls[1]!.workgroups).toEqual([2]);
    // Every dispatch's own uniform ("N", "K") must describe *its* row count,
    // not the original M — a shader reading params.N to bound its own
    // writes would corrupt memory past the chunk's real output length
    // otherwise. `params()` packs u32 little-endian at offset 0.
    const uniform0 = calls[0]!.bindings[3] as { kind: "uniform"; data: ArrayBuffer };
    const uniform1 = calls[1]!.bindings[3] as { kind: "uniform"; data: ArrayBuffer };
    expect(new DataView(uniform0.data).getUint32(0, true)).toBe(MAX_WORKGROUPS_PER_DISPATCH);
    expect(new DataView(uniform1.data).getUint32(0, true)).toBe(2);

    expect(got).toHaveLength(M);
    expect(Array.from(got)).toEqual(Array.from({ length: M }, (_, row) => row));
  });

  it("runMatVecQ8: a single dispatch when M is within the limit", async () => {
    const calls: Dispatch[] = [];
    const run = async (dispatch: Dispatch) => {
      calls.push(dispatch);
      return [new Float32Array(dispatch.workgroups[0])];
    };
    const M = 5;
    const K = 8;
    const wordsPerRow = Math.ceil(K / 4);
    await runMatVecQ8(run, {
      weight: new Uint32Array(M * wordsPerRow),
      scale: new Float32Array(M),
      vector: new Float32Array(K),
      M,
      K,
    });
    expect(calls).toHaveLength(1);
    expect(calls[0]!.workgroups).toEqual([M]);
  });

  it("runMatVecQ8: splits M > 65535 into chunks that exactly tile [0, M), slicing `weight` by words-per-row and `scale` by row", async () => {
    const K = 8;
    const wordsPerRow = Math.ceil(K / 4); // 2
    const M = MAX_WORKGROUPS_PER_DISPATCH + 2;
    // scale[row] tags the row (unpacked f32 — easy to assert on directly,
    // unlike `weight`'s packed int8 codes).
    const scale = Float32Array.from({ length: M }, (_, row) => row);
    const weight = new Uint32Array(M * wordsPerRow); // contents unused by this mock

    const calls: Dispatch[] = [];
    const run = async (dispatch: Dispatch) => {
      calls.push(dispatch);
      const weightSlice = dispatch.bindings[0] as { kind: "storage"; data: Uint32Array };
      const scaleSlice = dispatch.bindings[1] as { kind: "storage"; data: Float32Array };
      // The two slices must agree on how many rows they each cover.
      expect(weightSlice.data.length).toBe(scaleSlice.data.length * wordsPerRow);
      return [Float32Array.from(scaleSlice.data)];
    };

    const got = await runMatVecQ8(run, { weight, scale, vector: new Float32Array(K), M, K });

    expect(calls).toHaveLength(2);
    expect(calls[0]!.workgroups).toEqual([MAX_WORKGROUPS_PER_DISPATCH]);
    expect(calls[1]!.workgroups).toEqual([2]);
    const weight0 = calls[0]!.bindings[0] as { kind: "storage"; data: Uint32Array };
    const weight1 = calls[1]!.bindings[0] as { kind: "storage"; data: Uint32Array };
    expect(weight0.data.length).toBe(MAX_WORKGROUPS_PER_DISPATCH * wordsPerRow);
    expect(weight1.data.length).toBe(2 * wordsPerRow);

    expect(got).toHaveLength(M);
    expect(Array.from(got)).toEqual(Array.from({ length: M }, (_, row) => row));
  });
});
