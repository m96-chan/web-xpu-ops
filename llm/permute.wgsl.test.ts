import { describe, expect } from "vitest";
import { agree, gpuTest, useGpu } from "../harness/index.js";
import { runPermute } from "./kernels.js";
import { mergeHeadsMajor, splitHeadsMajor } from "./reshape.js";

/**
 * `llm/wgsl/permute.wgsl` against the CPU functions it exists to replace —
 * `splitHeadsMajor`/`mergeHeadsMajor` (`llm/reshape.ts`) *are* the reference
 * here (both already proven correct by the engines that use them; see that
 * file's doc), not a re-derivation, per rule 2.
 */

const wave = (n: number, k: number, phase = 0) => Float32Array.from({ length: n }, (_, i) => Math.sin(i * k + phase) * 1.5);

describe("llm/permute (wgsl)", () => {
  useGpu();

  const SHAPES = [
    { tokens: 1, heads: 1, dim: 1, why: "one of everything" },
    { tokens: 5, heads: 3, dim: 4, why: "a small GQA-unrelated shape" },
    { tokens: 3, heads: 16, dim: 112, why: "Sarashina2.2-1B's Q shape (numHeads, headDim)" },
    { tokens: 3, heads: 8, dim: 112, why: "Sarashina2.2-1B's K/V shape (numKvHeads, headDim)" },
    { tokens: 7, heads: 2, dim: 3, why: "total not a multiple of the 256-wide workgroup" },
    { tokens: 1, heads: 1, dim: 300, why: "total past one workgroup, D alone wider than it" },
  ];

  for (const { tokens, heads, dim, why } of SHAPES) {
    gpuTest(`runPermute(dim0=tokens, dim1=heads) agrees with splitHeadsMajor at tokens=${tokens} heads=${heads} dim=${dim} — ${why}`, async (run) => {
      const input = wave(tokens * heads * dim, 0.37);
      const want = splitHeadsMajor(input, tokens, heads, dim);
      const got = await runPermute(run, { input, dim0: tokens, dim1: heads, D: dim });
      const worst = agree(got, want, { rel: 0, abs: 0 });
      expect(worst, worst ? JSON.stringify(worst) : undefined).toBeNull();
    });

    gpuTest(`runPermute(dim0=heads, dim1=tokens) agrees with mergeHeadsMajor at tokens=${tokens} heads=${heads} dim=${dim} — ${why}`, async (run) => {
      // Head-major input, the shape `mergeHeadsMajor`'s own doc gives it:
      // `[heads, tokens, dim]`.
      const input = wave(heads * tokens * dim, 0.53, 0.2);
      const want = mergeHeadsMajor(input, heads, tokens, dim);
      const got = await runPermute(run, { input, dim0: heads, dim1: tokens, D: dim });
      const worst = agree(got, want, { rel: 0, abs: 0 });
      expect(worst, worst ? JSON.stringify(worst) : undefined).toBeNull();
    });
  }

  /**
   * The one thing a block transpose can get wrong that a same-shape square
   * case cannot reveal: swapping `dim0`/`dim1` themselves rather than the
   * elements between them. At `dim0 == dim1` a transposed-shape bug is
   * invisible (the output buffer is the same size and shape either way);
   * this case uses different sizes for the two axes, so a kernel that wrote
   * `output[(i0 * dim1 + i1) * D + d]` (identity — the "did not transpose at
   * all" bug) would overrun `output`'s real length rather than merely
   * misplace elements, and one that swapped which uniform field feeds which
   * loop variable produces a visibly different permutation, not a subtly
   * wrong one.
   */
  gpuTest("dim0 != dim1 really transposes, not just relabels", async (run) => {
    const [tokens, heads, dim] = [4, 9, 2];
    const input = wave(tokens * heads * dim, 0.61, 1.1);
    const want = splitHeadsMajor(input, tokens, heads, dim);
    const got = await runPermute(run, { input, dim0: tokens, dim1: heads, D: dim });
    expect(Array.from(got)).toEqual(Array.from(want));
    // And is not simply the identity — the whole point of the op is that
    // this differs from the input's own layout.
    expect(Array.from(got)).not.toEqual(Array.from(input));
  });
});
