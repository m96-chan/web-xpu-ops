import { describe, expect } from "vitest";
import { expectAgrees, gpuTest, kernel, useGpu } from "../../harness/index.js";
import { bindings, CASE_C, CASES, EXACT, expected, scoresFor } from "./cases.js";
import { ctcDecode } from "./reference.js";

const code = kernel(import.meta.url);

/** What the collapse means. The shapes it has to survive are in `wgsl-shapes.test.ts`. */
describe("ctc_decode / wgsl", () => {
  useGpu();

  for (const [name, path, labels] of CASES) {
    gpuTest(name, async (run) => {
      const [B, T, C] = [1, path.length, CASE_C];
      const scores = scoresFor([path], C);
      const want = ctcDecode({ scores, B, T, C });
      // Against the written-down answer as well as the reference, so a kernel
      // and a reference cannot agree on the wrong collapse.
      expect(Array.from(want.tokens.slice(0, labels.length))).toEqual(labels);
      await expectAgrees(
        run,
        { code, bindings: bindings(scores, { B, T, C }), workgroups: [B] },
        expected(want.tokens, want.lengths),
        EXACT,
      );
    });
  }

  gpuTest("honours a blank that is not class 0", async (run) => {
    // The default follows torch.nn.CTCLoss, but TensorFlow's ctc_greedy_decoder
    // puts the blank last and models trained that way exist. With 0 hard-coded
    // as the blank this decodes to something else entirely, so the parameter has
    // to be read rather than assumed.
    const [B, C, blank] = [1, 6, 5];
    const path = [5, 5, 0, 0, 5, 0, 3, 5, 3];
    const T = path.length;
    const scores = scoresFor([path], C);
    const want = ctcDecode({ scores, B, T, C, blank });
    expect(Array.from(want.tokens.slice(0, want.lengths[0]!))).toEqual([0, 0, 3, 3]);
    await expectAgrees(
      run,
      { code, bindings: bindings(scores, { B, T, C, blank }), workgroups: [B] },
      expected(want.tokens, want.lengths),
      EXACT,
    );
  });

  gpuTest("emits the first frame when it is not the blank", async (run) => {
    // Nothing precedes frame 0, and whatever stands in for "the previous label"
    // there decides whether its label survives. Class 0 opens the path while the
    // blank sits elsewhere, so a kernel seeding the previous label with a plain
    // zero swallows it — and every case above still passes, because there the
    // blank is 0 and the two seeds cannot be told apart.
    const [B, C, blank] = [1, 6, 5];
    const path = [0, 5, 0, 2];
    const T = path.length;
    const scores = scoresFor([path], C);
    const want = ctcDecode({ scores, B, T, C, blank });
    expect(Array.from(want.tokens.slice(0, want.lengths[0]!))).toEqual([0, 0, 2]);
    await expectAgrees(
      run,
      { code, bindings: bindings(scores, { B, T, C, blank }), workgroups: [B] },
      expected(want.tokens, want.lengths),
      EXACT,
    );
  });

  gpuTest("keeps the rows of a batch apart", async (run) => {
    // Rows of very different lengths, so a kernel sharing a counter or a
    // previous label across workgroups produces a visibly wrong split rather
    // than an off-by-one buried in the middle of one row.
    const [C, T] = [9, 12];
    const paths = [
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      [1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4],
      [0, 4, 4, 0, 4, 0, 0, 4, 4, 4, 0, 4],
      [7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7],
    ];
    const B = paths.length;
    const scores = scoresFor(paths, C);
    const want = ctcDecode({ scores, B, T, C });
    expect(Array.from(want.lengths)).toEqual([0, 12, 4, 1]);
    await expectAgrees(
      run,
      { code, bindings: bindings(scores, { B, T, C }), workgroups: [B] },
      expected(want.tokens, want.lengths),
      EXACT,
    );
  });
});
