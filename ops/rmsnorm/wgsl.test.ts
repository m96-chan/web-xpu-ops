import { describe, expect, it } from "vitest";
import { expectAgrees, gpuTest, kernel, params, useGpu } from "../../harness/index.js";
import { rmsnorm } from "./reference.js";

const code = kernel(import.meta.url);
const wave = (n: number, k = 0.37) => Float32Array.from({ length: n }, (_, i) => Math.sin(i * k) * 2);

/**
 * `[G, D]` gamma. Every group is offset by `g`, so no two groups share a value,
 * and it still varies along `D` inside a group — a weight that is uniform in
 * either direction passes with the indexing wrong.
 *
 * The tail past `G * D` is a sentinel rather than absent: this GPU reads zero
 * out of range, and a zero gamma produces a zero output, which is hard to tell
 * from a kernel that never ran. A read one group too far now returns something
 * loud instead.
 */
const SENTINEL = -1000;
const groupWeight = (G: number, D: number) =>
  Float32Array.from({ length: (G + 1) * D }, (_, i) =>
    i >= G * D ? SENTINEL : 0.5 + Math.cos((i % D) * 0.11) * 0.4 + Math.floor(i / D));

describe("rmsnorm / wgsl", () => {
  useGpu();

  // D spans the 256-wide workgroup deliberately: below it, exactly on it, and
  // not a multiple of it. The strided loop and the tree reduction each behave
  // differently depending on which, and only one of those is the common case.
  //
  // These pack the pre-#78 three-word params, with no group count at all. That
  // is the compatibility contract, not an oversight: the uniform's fourth word
  // is zero, and the kernel reads zero as one group. A caller written before
  // the parameter existed keeps its behaviour without being recompiled.
  for (const [N, D] of [[2, 8], [1, 256], [3, 300], [2, 2560]] as const) {
    gpuTest(`agrees with the reference at N=${N} D=${D}`, async (run) => {
      const input = wave(N * D);
      const weight = Float32Array.from({ length: D }, (_, i) => 0.5 + Math.cos(i * 0.11) * 0.4);
      const eps = 1e-5;
      await expectAgrees(
        run,
        {
          code,
          bindings: [
            { kind: "storage", data: input },
            { kind: "storage", data: weight },
            { kind: "out", type: "f32", length: N * D },
            { kind: "uniform", data: params([["u32", N], ["u32", D], ["f32", eps]]) },
          ],
          workgroups: [N],
        },
        [rmsnorm({ input, weight, N, D, eps })],
      );
    });
  }

  gpuTest("keeps eps meaningful on an all-but-zero row", async (run) => {
    // Without this case, removing eps from the shader entirely still passes:
    // on ordinary input sumSquares/D dwarfs it. Here it is the only thing
    // between the reciprocal square root and a division by zero.
    const D = 8;
    const input = new Float32Array(D);
    const weight = new Float32Array(D).fill(1);
    const eps = 1e-5;
    await expectAgrees(
      run,
      {
        code,
        bindings: [
          { kind: "storage", data: input },
          { kind: "storage", data: weight },
          { kind: "out", type: "f32", length: D },
          { kind: "uniform", data: params([["u32", 1], ["u32", D], ["f32", eps]]) },
        ],
        workgroups: [1],
      },
      [rmsnorm({ input, weight, N: 1, D, eps })],
    );
  });

  // #78: QK-norm gives every head its own gamma. Flattened to [N, D] that is a
  // weight repeating every G rows — G = H for input laid out [B, S, H, Dh],
  // which is the only layout this op accepts (see reference.ts).
  //
  // N is B*S*G in the first case and deliberately *not* a multiple of G in the
  // second: the reference says the weight keeps repeating, and a rule stated
  // only in prose is a rule nothing checks.
  for (const [N, D, G] of [[12, 8, 4], [6, 300, 3], [6, 8, 4], [5, 256, 1]] as const) {
    gpuTest(`applies a per-group weight at N=${N} D=${D} G=${G}`, async (run) => {
      const input = wave(N * D);
      const weight = groupWeight(G, D);
      const eps = 1e-5;
      await expectAgrees(
        run,
        {
          code,
          bindings: [
            { kind: "storage", data: input },
            { kind: "storage", data: weight },
            { kind: "out", type: "f32", length: N * D },
            { kind: "uniform", data: params([["u32", N], ["u32", D], ["f32", eps], ["u32", G]]) },
          ],
          workgroups: [N],
        },
        [rmsnorm({ input, weight, N, D, eps, groups: G })],
      );
    });
  }

  gpuTest("reads an explicit G=1 the same as no group count at all", async (run) => {
    // The two ways of saying "today's behaviour" have to agree, or the
    // compatibility path above is a second implementation nobody compares.
    const [N, D] = [3, 8];
    const input = wave(N * D);
    const weight = Float32Array.from({ length: D }, (_, i) => 0.5 + Math.cos(i * 0.11) * 0.4);
    const eps = 1e-5;
    await expectAgrees(
      run,
      {
        code,
        bindings: [
          { kind: "storage", data: input },
          { kind: "storage", data: weight },
          { kind: "out", type: "f32", length: N * D },
          { kind: "uniform", data: params([["u32", N], ["u32", D], ["f32", eps], ["u32", 1]]) },
        ],
        workgroups: [N],
      },
      [rmsnorm({ input, weight, N, D, eps })],
    );
  });

  it("reads groups=0 as one group, as the kernel does", () => {
    // The kernel's `max(params.G, 1u)` cannot be killed by a mutation on this
    // device — Tint lowers `x % 0` to `0`, which is the same answer — so the
    // contract is pinned here, where zero and one are genuinely different
    // arithmetic. If the reference ever stops agreeing, the two have drifted.
    const [N, D] = [4, 8];
    const input = wave(N * D);
    const weight = Float32Array.from({ length: D }, (_, i) => 0.5 + Math.cos(i * 0.11) * 0.4);
    const eps = 1e-5;
    expect(rmsnorm({ input, weight, N, D, eps, groups: 0 })).toEqual(
      rmsnorm({ input, weight, N, D, eps, groups: 1 }),
    );
  });

  it("the per-group cases would fail against a single shared gamma", () => {
    // Guards the test data, not the kernel. If the weight above ever became
    // uniform across groups, every per-group case would still pass with the
    // indexing wrong, which is exactly rule 1's "passes for the wrong reason".
    const [N, D, G] = [12, 8, 4];
    const input = wave(N * D);
    const weight = groupWeight(G, D);
    const eps = 1e-5;
    const grouped = rmsnorm({ input, weight, N, D, eps, groups: G });
    const shared = rmsnorm({ input, weight, N, D, eps });
    // Row 0 is group 0 either way; row 1 is where they must diverge.
    expect(grouped.slice(0, D)).toEqual(shared.slice(0, D));
    expect(grouped.slice(D, 2 * D)).not.toEqual(shared.slice(D, 2 * D));
  });
});
