import { describe, expect, it } from "vitest";
import { alibi, alibiSlopes } from "./reference.js";

/**
 * The slopes, pinned to published numbers rather than to the kernel.
 *
 * The WGSL test can only say that the shader and the reference compute the same
 * thing — if the reference has the wrong convention, both are wrong together
 * and the suite is green. So the convention itself is asserted here, against
 * the values the paper prints, and against the derivations of the two branches
 * of its algorithm. Touches no GPU.
 *
 * Nothing in PyTorch to defer to (rule 7): ALiBi is not a torch op. The
 * authority is the reference implementation released with the paper, which
 * HuggingFace `transformers` reproduces exactly in BLOOM's
 * `build_alibi_tensor` — the two were compared term by term before this was
 * written, and they agree including on the non-power-of-two branch.
 */
describe("alibi / reference", () => {
  const close = (got: ArrayLike<number>, want: number[]) => {
    expect(got.length).toBe(want.length);
    want.forEach((value, index) => expect(got[index]!).toBeCloseTo(value, 12));
  };

  it("gives the paper's geometric run for 8 heads", () => {
    // Section 3: "for 8 heads the slopes are 1/2, 1/4, 1/8, ..., 1/256" — the
    // one case the paper states outright, and the reason 8 is the anchor.
    close(alibiSlopes(8), [1 / 2, 1 / 4, 1 / 8, 1 / 16, 1 / 32, 1 / 64, 1 / 128, 1 / 256]);
  });

  it("halves the exponent step when the head count doubles", () => {
    // The run always ends at 2^-8 whatever n is; only the step changes. 16
    // heads therefore share the 8-head slopes and interleave the half-steps.
    close(alibiSlopes(16), Array.from({ length: 16 }, (_, h) => 2 ** (-(h + 1) / 2)));
    close(alibiSlopes(4), [2 ** -2, 2 ** -4, 2 ** -6, 2 ** -8]);
    close(alibiSlopes(2), [2 ** -4, 2 ** -8]);
    close(alibiSlopes(1), [2 ** -8]);
  });

  it("appends, rather than interpolates, for a head count that is not a power of two", () => {
    // The branch that implementations quietly get wrong. 12 heads is not
    // "the 16-head run, shortened": it is the whole 8-head run followed by
    // every other slope of the 16-head run.
    const twelve = alibiSlopes(12);
    close(twelve, [
      2 ** -1, 2 ** -2, 2 ** -3, 2 ** -4, 2 ** -5, 2 ** -6, 2 ** -7, 2 ** -8,
      2 ** -0.5, 2 ** -1.5, 2 ** -2.5, 2 ** -3.5,
    ]);

    // Stated as its own expectation because it is the observable consequence:
    // the tail is *larger* than the head, so the sequence is not monotonic and
    // any implementation that sorts it is wrong.
    expect(twelve[8]!).toBeGreaterThan(twelve[0]!);

    // And it is not the longer run truncated, which is the other common shortcut.
    const sixteen = alibiSlopes(16);
    expect(Array.from(twelve)).not.toEqual(Array.from(sixteen).slice(0, 12));
  });

  it("takes the first n-closest of the doubled run, not all of it", () => {
    // 9 heads takes exactly one extra slope; 15 takes seven. Off-by-one in the
    // slice shows up here and nowhere else.
    close(alibiSlopes(9), [
      2 ** -1, 2 ** -2, 2 ** -3, 2 ** -4, 2 ** -5, 2 ** -6, 2 ** -7, 2 ** -8,
      2 ** -0.5,
    ]);
    expect(alibiSlopes(15)).toHaveLength(15);
    close(
      Array.from(alibiSlopes(15)).slice(8),
      [2 ** -0.5, 2 ** -1.5, 2 ** -2.5, 2 ** -3.5, 2 ** -4.5, 2 ** -5.5, 2 ** -6.5],
    );
  });

  it("refuses a head count that would not terminate", () => {
    // log2(0) is -Infinity, which is not an integer, so the recursion would
    // descend forever. Fail loudly instead of hanging the caller.
    expect(() => alibiSlopes(0)).toThrow(/positive integer/);
    expect(() => alibiSlopes(-4)).toThrow(/positive integer/);
    expect(() => alibiSlopes(2.5)).toThrow(/positive integer/);
  });
});

describe("alibi / reference application", () => {
  it("is zero on the diagonal and linear going back", () => {
    // The paper's bias for query i is m * [-(i-1), ..., -2, -1, 0] over the
    // keys: nothing subtracted from the token attending to itself, one slope
    // per step of distance before it. Scores of zero make the bias readable.
    const numHeads = 1;
    const [M, N] = [3, 3];
    const out = alibi({
      scores: new Float32Array(M * N),
      numHeads,
      M,
      N,
      posOffset: 0,
      slopes: [1],
    });
    expect(Array.from(out)).toEqual([
      0, 1, 2,
      -1, 0, 1,
      -2, -1, 0,
    ]);
  });

  it("shifts the query positions by posOffset", () => {
    // Decoding: one query row whose true position is three tokens in, so the
    // diagonal sits at key 3 and every earlier key is penalised by its distance.
    const out = alibi({
      scores: new Float32Array(4),
      numHeads: 1,
      M: 1,
      N: 4,
      posOffset: 3,
      slopes: [1],
    });
    expect(Array.from(out)).toEqual([-3, -2, -1, 0]);
  });

  it("scales each head by its own slope", () => {
    const out = alibi({
      scores: new Float32Array(2 * 1 * 2),
      numHeads: 2,
      M: 1,
      N: 2,
      posOffset: 1,
      slopes: [1, 0.25],
    });
    expect(Array.from(out)).toEqual([-1, 0, -0.25, 0]);
  });

  it("refuses shapes that do not match the buffers", () => {
    expect(() => alibi({ scores: new Float32Array(5), numHeads: 1, M: 2, N: 3, posOffset: 0 }))
      .toThrow(/expected 6 elements, got 5/);
    expect(() =>
      alibi({ scores: new Float32Array(6), numHeads: 2, M: 1, N: 3, posOffset: 0, slopes: [1] }),
    ).toThrow(/expected 2 elements, got 1/);
  });
});
