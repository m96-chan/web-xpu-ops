import { describe, expect, it } from "vitest";
import { agree, expectAgrees, gpuTest, kernel, params, useGpu } from "../../harness/index.js";
import { mergeHeadsMajor, splitHeadsMajor } from "../../llm/reshape.js";
import { permute } from "./reference.js";

const code = kernel(import.meta.url);

/**
 * Every element distinct, so a misplaced one is a failure rather than a
 * coincidence — this op does no arithmetic, only decides where a value
 * lands, and a smooth input like a sine wave hides a small displacement
 * behind a small difference (`ops/transpose/wgsl.test.ts` measured the same
 * thing first; copied here rather than re-derived, rule 2).
 */
const distinct = (n: number) => Float32Array.from({ length: n }, (_, i) => (i + 1) * 0.25);

describe("permute / wgsl", () => {
  useGpu();

  const exact = { rel: 0, abs: 0 };

  const SHAPES = [
    { dim0: 1, dim1: 1, D: 1, why: "one of everything" },
    { dim0: 5, dim1: 3, D: 4, why: "a small shape" },
    { dim0: 3, dim1: 16, D: 112, why: "Sarashina2.2-1B's Q shape (tokens, numHeads, headDim)" },
    { dim0: 3, dim1: 8, D: 112, why: "Sarashina2.2-1B's K/V shape (tokens, numKvHeads, headDim)" },
    { dim0: 7, dim1: 2, D: 3, why: "total not a multiple of the 256-wide workgroup" },
    { dim0: 1, dim1: 1, D: 300, why: "total past one workgroup, D alone wider than it" },
    { dim0: 4, dim1: 9, D: 2, why: "dim0 != dim1, so a transposed-shape bug cannot hide behind a square case" },
  ];

  for (const { dim0, dim1, D, why } of SHAPES) {
    gpuTest(`agrees with the reference at dim0=${dim0} dim1=${dim1} D=${D} — ${why}`, async (run) => {
      const input = distinct(dim0 * dim1 * D);
      const want = permute({ input, dim0, dim1, D });
      await expectAgrees(
        run,
        {
          code,
          bindings: [
            { kind: "storage", data: input },
            { kind: "out", type: "f32", length: dim0 * dim1 * D },
            { kind: "uniform", data: params([["u32", dim0], ["u32", dim1], ["u32", D]]) },
          ],
          workgroups: [Math.ceil((dim0 * dim1 * D) / 256)],
        },
        [want],
        exact,
      );
    });
  }

  gpuTest("really transposes dim0/dim1, not just relabels them", async (run) => {
    const [dim0, dim1, D] = [4, 9, 2];
    const input = distinct(dim0 * dim1 * D);
    const want = permute({ input, dim0, dim1, D });
    const [got] = await run({
      code,
      bindings: [
        { kind: "storage", data: input },
        { kind: "out", type: "f32", length: dim0 * dim1 * D },
        { kind: "uniform", data: params([["u32", dim0], ["u32", dim1], ["u32", D]]) },
      ],
      workgroups: [Math.ceil((dim0 * dim1 * D) / 256)],
    });
    expect(Array.from(got as Float32Array)).toEqual(Array.from(want));
    // And is not simply the identity — the whole point of the op is that
    // this differs from the input's own layout.
    expect(Array.from(got as Float32Array)).not.toEqual(Array.from(input));
  });

  /**
   * `llm/reshape.ts#splitHeadsMajor`/`mergeHeadsMajor` under their own
   * names — the functions this op exists to replace with one GPU dispatch
   * for issue #117's resident prefill (see `ops/permute/reference.ts`'s
   * doc). Checked directly against this kernel, not just against
   * `permute()` above, so a drift between this op's own reference and the
   * llm-specific functions it is meant to stand in for cannot go unnoticed.
   */
  const RESHAPE_SHAPES = [
    { tokens: 1, heads: 1, dim: 1 },
    { tokens: 5, heads: 3, dim: 4 },
    { tokens: 3, heads: 16, dim: 112 },
    { tokens: 3, heads: 8, dim: 112 },
  ];

  for (const { tokens, heads, dim } of RESHAPE_SHAPES) {
    gpuTest(`matches splitHeadsMajor at tokens=${tokens} heads=${heads} dim=${dim}`, async (run) => {
      const input = distinct(tokens * heads * dim);
      const want = splitHeadsMajor(input, tokens, heads, dim);
      await expectAgrees(
        run,
        {
          code,
          bindings: [
            { kind: "storage", data: input },
            { kind: "out", type: "f32", length: tokens * heads * dim },
            { kind: "uniform", data: params([["u32", tokens], ["u32", heads], ["u32", dim]]) },
          ],
          workgroups: [Math.ceil((tokens * heads * dim) / 256)],
        },
        [want],
        exact,
      );
    });

    gpuTest(`matches mergeHeadsMajor at tokens=${tokens} heads=${heads} dim=${dim}`, async (run) => {
      // Head-major input, the shape `mergeHeadsMajor`'s own doc gives it:
      // `[heads, tokens, dim]`.
      const input = distinct(heads * tokens * dim);
      const want = mergeHeadsMajor(input, heads, tokens, dim);
      await expectAgrees(
        run,
        {
          code,
          bindings: [
            { kind: "storage", data: input },
            { kind: "out", type: "f32", length: heads * tokens * dim },
            { kind: "uniform", data: params([["u32", heads], ["u32", tokens], ["u32", dim]]) },
          ],
          workgroups: [Math.ceil((heads * tokens * dim) / 256)],
        },
        [want],
        exact,
      );
    });
  }

  /**
   * A dispatch too wide for one row of workgroups.
   *
   * 65,535 is the ceiling on every backend measured (#211), which at 256
   * threads a workgroup is 16.7 M elements — a video VAE's first level reaches
   * that on a 256x256 reference with five frames. Tiling it into `(x, y)` only
   * works if the kernel folds `y` back in, and the fold uses `num_workgroups`
   * rather than a uniform so that **every existing one-dimensional caller keeps
   * working unchanged**: at `[n]` the y extent is 1 and `gid.y` is 0.
   */
  gpuTest("folds a two-dimensional dispatch back into one index", async (run) => {
    const dim0 = 300;
    const dim1 = 7;
    const D = 129;
    const total = dim0 * dim1 * D;
    const tiles = Math.ceil(total / 256);
    // Deliberately narrow, so most of the work is on rows the old kernel never
    // reached: with `gid.y` ignored only the first 8 tiles are ever written.
    const x = 8;
    const y = Math.ceil(tiles / x);
    expect(y).toBeGreaterThan(1);
    const input = distinct(total);
    await expectAgrees(
      run,
      {
        code,
        bindings: [
          { kind: "storage", data: input },
          { kind: "out", type: "f32", length: total },
          { kind: "uniform", data: params([["u32", dim0], ["u32", dim1], ["u32", D]]) },
        ],
        workgroups: [x, y],
      },
      [permute({ input, dim0, dim1, D })],
      exact,
    );
  });

  it("rejects an input sized for the wrong dim0/dim1/D", () => {
    expect(() => permute({ input: new Float32Array(5), dim0: 2, dim1: 3, D: 1 })).toThrow(/expected 6 input elements, got 5/);
  });
});
