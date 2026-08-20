import { describe, expect } from "vitest";
import { agree, gpuTest, useGpu } from "../harness/index.js";
import { ACTIVATION } from "../ops/activation/index.js";
import { ELEMENTWISE } from "../ops/elementwise/index.js";
import { gather } from "../ops/gather/index.js";
import { groupedAttention } from "../ops/gqa/index.js";
import { matmul } from "../ops/matmul/index.js";
import { matvec, matvecQ8, packQ8 } from "../ops/matvec/index.js";
import { rmsnorm } from "../ops/rmsnorm/index.js";
import { rope } from "../ops/rope/index.js";
import {
  runActivation,
  runElementwise,
  runGather,
  runGqa,
  runMatMul,
  runMatVec,
  runMatVecQ8,
  runRmsNorm,
  runRope,
} from "./kernels.js";

/**
 * Each `llm/kernels.ts` wrapper against the same op's reference — not the
 * op's own `wgsl.test.ts` again, but that the *wrapper* (binding order,
 * uniform packing, workgroup count) round-trips through the shared `Runner`
 * correctly. `llm/engine.wgsl.test.ts` then checks the whole forward pass
 * against a transformers-generated fixture; this file isolates a wiring
 * mistake in one wrapper from a modelling mistake in the engine.
 */

const wave = (n: number, k: number, phase = 0) => Float32Array.from({ length: n }, (_, i) => Math.sin(i * k + phase) * 1.5);

describe("llm/kernels wrappers", () => {
  useGpu();

  gpuTest("runGather agrees with gather()", async (run) => {
    const [rows, D, N] = [7, 5, 11];
    const table = wave(rows * D, 0.37);
    const indices = Int32Array.from({ length: N }, (_, i) => (i * 3 + 1) % rows);
    const got = await runGather(run, { table, indices, rows, D });
    const want = gather({ table, indices, rows, D });
    expect(agree(got, want)).toBeNull();
  });

  gpuTest("runRmsNorm agrees with rmsnorm()", async (run) => {
    const [N, D] = [4, 32];
    const input = wave(N * D, 0.29);
    const weight = Float32Array.from({ length: D }, (_, i) => 0.5 + Math.cos(i * 0.1) * 0.3);
    const eps = 1e-5;
    const got = await runRmsNorm(run, { input, weight, N, D, eps });
    const want = rmsnorm({ input, weight, N, D, eps });
    expect(agree(got, want)).toBeNull();
  });

  gpuTest("runMatVec agrees with matvec()", async (run) => {
    const [M, K] = [6, 40];
    const matrix = wave(M * K, 0.37);
    const vector = wave(K, 0.11, 0.8);
    const got = await runMatVec(run, { matrix, vector, M, K });
    const want = matvec({ matrix, vector, M, K });
    expect(agree(got, want)).toBeNull();
  });

  gpuTest("runMatVecQ8 agrees with matvecQ8()", async (run) => {
    const [M, K] = [6, 40];
    const codes = Int32Array.from({ length: M * K }, (_, i) => Math.round(Math.sin(i * 0.37) * 100));
    const weight = packQ8({ codes, N: M, K });
    const scale = Float32Array.from({ length: M }, (_, i) => 0.02 * (i + 1) + 0.007);
    const vector = wave(K, 0.11, 0.8);
    const got = await runMatVecQ8(run, { weight, scale, vector, M, K });
    const want = matvecQ8({ weight, scale, vector, N: M, K });
    expect(agree(got, want, { rel: 1e-4, abs: 5e-5 })).toBeNull();
  });

  gpuTest("runMatMul agrees with matmul()", async (run) => {
    const [M, N, K] = [5, 7, 33];
    const a = wave(M * K, 0.37);
    const b = wave(K * N, 0.11, 0.5);
    const got = await runMatMul(run, { a, b, M, N, K });
    const want = matmul({ a, b, M, N, K });
    expect(agree(got, want, { rel: 1e-5, abs: 2e-5 })).toBeNull();
  });

  gpuTest("runRope agrees with rope() at posOffset=0", async (run) => {
    const [N, numHeads, headDim] = [3, 4, 16];
    const input = wave(N * numHeads * headDim, 0.29);
    const thetaBase = 10000;
    const got = await runRope(run, { input, N, numHeads, headDim, posOffset: 0, thetaBase });
    const want = rope({ input, N, numHeads, headDim, posOffset: 0, thetaBase });
    expect(agree(got, want, { abs: 1e-3 })).toBeNull();
  });

  gpuTest("runRope honours posOffset (KV-cache continuation)", async (run) => {
    const [N, numHeads, headDim] = [2, 2, 8];
    const input = wave(N * numHeads * headDim, 0.31);
    const thetaBase = 10000;
    const posOffset = 5;
    const got = await runRope(run, { input, N, numHeads, headDim, posOffset, thetaBase });
    const want = rope({ input, N, numHeads, headDim, posOffset, thetaBase });
    expect(agree(got, want, { abs: 1e-3 })).toBeNull();
  });

  gpuTest("runGqa agrees with groupedAttention() for a GQA grouping", async (run) => {
    const [H, kvHeads, L, S, D, Dv] = [4, 2, 3, 3, 8, 8];
    const q = wave(H * L * D, 0.37);
    const k = wave(kvHeads * S * D, 0.11, 0.5);
    const v = wave(kvHeads * S * Dv, 0.23, 1.25);
    const got = await runGqa(run, { q, k, v, H, kvHeads, L, S, D, Dv, causal: true, queryOffset: 0 });
    const want = groupedAttention({ q, k, v, B: 1, H, kvHeads, L, S, D, Dv, causal: true });
    expect(agree(got, want.output, { rel: 1e-5, abs: 2e-6 })).toBeNull();
  });

  gpuTest("runActivation agrees with activation() for silu", async (run) => {
    const input = wave(37, 0.41);
    const got = await runActivation(run, { input, kind: ACTIVATION.silu });
    const want = new Float32Array(input.map((x) => x / (1 + Math.exp(-x))));
    expect(agree(got, want)).toBeNull();
  });

  gpuTest("runElementwise agrees with elementwise() for add and multiply", async (run) => {
    const a = wave(41, 0.31);
    const b = wave(41, 0.19, 0.4);
    const gotAdd = await runElementwise(run, { a, b, kind: ELEMENTWISE.add });
    const wantAdd = new Float32Array(a.map((x, i) => x + b[i]!));
    expect(agree(gotAdd, wantAdd)).toBeNull();

    const gotMul = await runElementwise(run, { a, b, kind: ELEMENTWISE.multiply });
    const wantMul = new Float32Array(a.map((x, i) => x * b[i]!));
    expect(agree(gotMul, wantMul)).toBeNull();
  });
});
