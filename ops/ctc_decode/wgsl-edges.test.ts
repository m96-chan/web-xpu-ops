import { describe, expect } from "vitest";
import { expectAgrees, gpuTest, kernel, useGpu } from "../../harness/index.js";
import { bindings, EXACT, expected, padScores, scoresFor } from "./cases.js";
import { ctcDecode } from "./reference.js";

const code = kernel(import.meta.url);

/**
 * The edges: which class wins, what the argmax starts from, and where the scan
 * stops.
 *
 * One of three GPU files for this op rather than one, because a vitest process
 * on this machine aborts after enough dispatches — see the note in `cases.ts`.
 */
describe("ctc_decode / wgsl / edges", () => {
  useGpu();

  gpuTest("finds a winner past the first lane stride", async (run) => {
    // With one lane per class up to 256, a kernel that never strides still gets
    // every earlier case right. Most of these labels only exist beyond that, and
    // 255 / 256 / 257 sit either side of the boundary itself — 256 and 257 are
    // the second class a lane sees, 255 the last one it sees first.
    const [B, C] = [1, 300];
    const path = [0, 299, 299, 0, 299, 257, 0, 256, 255, 0, 0, 111];
    const T = path.length;
    const scores = scoresFor([path], C);
    const want = ctcDecode({ scores, B, T, C });
    expect(Array.from(want.tokens.slice(0, want.lengths[0]!))).toEqual([
      299, 299, 257, 256, 255, 111,
    ]);
    await expectAgrees(
      run,
      { code, bindings: bindings(scores, { B, T, C }), workgroups: [B] },
      expected(want.tokens, want.lengths),
      EXACT,
    );
  });

  for (const [C, low, high] of [
    // Both classes land on the same lane (20 and 20 + 256), so the tie is
    // settled inside one lane's scan, by it not replacing an equal maximum.
    [300, 20, 276],
    // Different lanes, and — this is the point — the lower class is on the
    // *higher* lane: 260 goes to lane 4, 5 to lane 5. A reduction that keeps
    // whichever slot it already held on a tie answers 260 here and 20 above,
    // so this is the case that says the comparison is on the class index.
    [300, 5, 260],
  ] as const) {
    gpuTest(`takes the lowest index when the top score is tied at C=${C}`, async (run) => {
      const [B, T] = [1, 4];
      const head = Float32Array.from({ length: T * C }, (_, i) => Math.sin(i * 0.37) * 2);
      for (let t = 0; t < T; t += 1) {
        head[t * C + low] = 4;
        head[t * C + high] = 4;
      }
      const scores = padScores(head);
      const want = ctcDecode({ scores, B, T, C });
      // The repeat collapses either way, so the whole answer is which of the two
      // tied classes came out.
      expect(Array.from(want.tokens.slice(0, want.lengths[0]!))).toEqual([low]);
      await expectAgrees(
        run,
        { code, bindings: bindings(scores, { B, T, C }), workgroups: [B] },
        expected(want.tokens, want.lengths),
        EXACT,
      );
    });
  }

  gpuTest("decodes frames whose scores are all negative", async (run) => {
    // The identity for the argmax has to be below every score, and real acoustic
    // models emit log-probabilities — negative everywhere. A kernel seeding its
    // running maximum with zero finds no class at all here, and passes every
    // other test in this op, because their winner sits above zero.
    const [B, C] = [1, 300];
    const path = [7, 7, 0, 7, 299, 0, 0, 12];
    const T = path.length;
    const head = Float32Array.from({ length: T * C }, (_, i) => -20 - Math.abs(Math.sin(i * 0.31)));
    path.forEach((label, t) => {
      head[t * C + label] = -5;
    });
    const scores = padScores(head);
    const want = ctcDecode({ scores, B, T, C });
    expect(Array.from(want.tokens.slice(0, want.lengths[0]!))).toEqual([7, 7, 299, 12]);
    await expectAgrees(
      run,
      { code, bindings: bindings(scores, { B, T, C }), workgroups: [B] },
      expected(want.tokens, want.lengths),
      EXACT,
    );
  });

  gpuTest("reads no frame past T", async (run) => {
    // Every scores buffer in this op runs past `B*T*C`, but only this one says
    // what is out there: four more frames won by a class the real path never
    // uses. A scan running one frame long emits an 11 that nothing else catches.
    const [B, C, T] = [1, 12, 10];
    const path = [0, 2, 2, 0, 2, 5, 0, 0, 6, 6];
    const beyond = [11, 11, 11, 11];
    const scores = scoresFor([[...path, ...beyond]], C);
    const want = ctcDecode({ scores, B, T, C });
    expect(Array.from(want.tokens.slice(0, want.lengths[0]!))).toEqual([2, 2, 5, 6]);
    await expectAgrees(
      run,
      { code, bindings: bindings(scores, { B, T, C }), workgroups: [B] },
      expected(want.tokens, want.lengths),
      EXACT,
    );
  });
});
