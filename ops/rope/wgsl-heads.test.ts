import { describe, expect, it } from "vitest";
import { agree, gpuTest, kernel, useGpu } from "../../harness/index.js";
import { TRANSCENDENTAL, YARN_64, scenario } from "./testing.js";

/**
 * The head range: `headOffset` / `headCount`, and the heads outside it.
 *
 * The axis is over **heads**, not over the channels of a head — see
 * `reference.ts`, which says why that distinction is worth writing down. So
 * every assertion here is on whole head rows.
 *
 * Split into its own file for the dispatch budget, like `wgsl-cache.test.ts`;
 * see `testing.ts`.
 */
const code = kernel(import.meta.url);

// 576 pairs, so the 256-wide workgroup is crossed twice and the tail is ragged,
// and eight heads leave room for a range with both ends inside the tensor.
const GEOMETRY = { N: 9, numHeads: 8, headDim: 16, posOffset: 200, thetaBase: 10000 } as const;
const { N, numHeads, headDim } = GEOMETRY;

/**
 * Nowhere zero — `+ 3` puts every element in [1, 5].
 *
 * This is the whole observability argument of this file. `rope` and the kernel
 * both write into a buffer that starts at zero, so a passthrough head nobody
 * copied reads back as zero. Against the wave the other files use, which is
 * exactly 0 at index 0 and small near it, "copied" and "never written" are the
 * same number. Against this one they never are.
 */
const input = Float32Array.from({ length: N * numHeads * headDim }, (_, i) => Math.sin(i * 0.29) * 2 + 3);

/** Every slot of every head outside the range. */
const passthrough = (headOffset: number, headCount: number): number[] => {
  const at: number[] = [];
  for (let token = 0; token < N; token += 1) {
    for (let head = 0; head < numHeads; head += 1) {
      if (head >= headOffset && head < headOffset + headCount) continue;
      const from = (token * numHeads + head) * headDim;
      for (let i = 0; i < headDim; i += 1) at.push(from + i);
    }
  }
  return at;
};

/**
 * The first few passthrough slots that are not bit-identical to the input.
 *
 * Reported as a short list rather than by comparing the two arrays, because the
 * failure has to fit in a test report: a `toEqual` over 720 elements builds a
 * diff big enough to take the GPU worker down with it — measured, six runs out
 * of six hung against a kernel whose only fault was multiplying the copy by
 * 1.0001. Same family as #48; an assertion nobody can read is not an assertion.
 */
const inexact = (actual: ArrayLike<number>, at: number[], limit = 3): string[] => {
  const bad: string[] = [];
  for (const i of at) {
    if (actual[i] !== input[i]) bad.push(`${i}: got ${actual[i]}, want ${input[i]}`);
    if (bad.length === limit) break;
  }
  return bad;
};

// Heads 2, 3, 4 rotate. Both ends of the range are load-bearing: a kernel that
// ignored `head_offset` would rotate 0..2, one that ignored `head_count` would
// rotate 2..7, and both fail on rows the other would get right.
const [MID_OFFSET, MID_COUNT] = [2, 3];
const middle = scenario(code, { ...GEOMETRY, headOffset: MID_OFFSET, headCount: MID_COUNT }, { positions: 0 }, input);
const middleCopied = passthrough(MID_OFFSET, MID_COUNT);

// Irodori-TTS's `_apply_rotary_half`: the first half of the heads rotate and the
// second half stays position-free. With YaRN on top, because the two compose and
// because YaRN's gain of 1.208 must reach the rotated heads and only those — a
// kernel that applied it before the range check fails here and nowhere else.
const half = scenario(
  code,
  { ...GEOMETRY, scaling: YARN_64, headOffset: 0, headCount: numHeads / 2 },
  { positions: 0 },
  input,
);

// The degenerate end: nothing rotates, so the whole tensor is a copy.
const empty = scenario(code, { ...GEOMETRY, headOffset: 3, headCount: 0 }, { positions: 0 }, input);

/**
 * What makes the dispatches below observable, asserted rather than assumed.
 *
 * Runs before the device exists, so it costs the GPU suite nothing and it holds
 * on a machine with no adapter — which matters, since this is the property the
 * issue names and the GPU tests skip themselves when there is no GPU.
 */
describe("rope / head range / what the fixtures pin", () => {
  it("has passthrough data that is nowhere zero and rotated data that moved", () => {
    // Zeros would satisfy "the passthrough heads came back unchanged" even if
    // nothing was ever copied into them.
    expect(middleCopied.every((at) => input[at] !== 0)).toBe(true);
    expect(middleCopied.every((at) => middle.expected[at] === input[at])).toBe(true);

    // And the rotated heads have to be somewhere else, or copying everything
    // would agree with the expectation too. Measured on this fixture: the
    // rotated rows move by 3.2 at position 200 against the 1e-3 allowed.
    const moved = Math.max(
      ...Array.from({ length: N * numHeads * headDim }, (_, at) =>
        Math.floor((at / headDim) % numHeads) >= MID_OFFSET &&
        Math.floor((at / headDim) % numHeads) < MID_OFFSET + MID_COUNT
          ? Math.abs(middle.expected[at]! - input[at]!)
          : 0,
      ),
    );
    expect(moved).toBeGreaterThan(1);

    // The empty range is the input, all of it.
    expect(Array.from(empty.expected.slice(0, input.length))).toEqual(Array.from(input));
  });
});

describe("rope / head range / wgsl", () => {
  useGpu();

  gpuTest("rotates the heads inside the range and copies the ones outside", async (run) => {
    const [actual] = await run(middle.dispatch);
    expect(agree(actual!, middle.expected, TRANSCENDENTAL)).toBeNull();

    // Exactly, not within `TRANSCENDENTAL`: a copy that rounds is not a copy,
    // and the tolerance the rotated heads need is three orders of magnitude
    // wider than anything a passthrough head is entitled to.
    //
    // Counted rather than compared element by element, and that is not a style
    // choice: a failing `toEqual` over 720 elements builds a diff big enough to
    // take the GPU worker down with it — measured, six runs out of six hung
    // with a kernel whose only fault was multiplying the copy by 1.0001. The
    // assertion has to fail small enough to be reported.
    expect(inexact(actual!, middleCopied)).toEqual([]);
    // And nowhere zero, which is the condition the issue names: a passthrough
    // head nobody wrote reads back as zero, and against an input that is itself
    // nowhere zero that is the one thing a copy cannot look like.
    expect(middleCopied.filter((at) => actual![at] === 0).length).toBe(0);
  });

  gpuTest("rotates the first half of the heads under YaRN and leaves the rest alone", async (run) => {
    const [actual] = await run(half.dispatch);
    expect(agree(actual!, half.expected, TRANSCENDENTAL)).toBeNull();
    // YaRN's gain reaches the rotated heads through `agree` above; here it must
    // not have reached these — an exact copy is a copy that was not scaled.
    expect(inexact(actual!, passthrough(0, numHeads / 2))).toEqual([]);
  });

  gpuTest("copies the whole tensor when the range is empty", async (run) => {
    const [actual] = await run(empty.dispatch);
    // Bit-exact, and the padding past the data stays zero — the guard is still
    // the only thing keeping the stray threads of the last workgroup out.
    expect(agree(actual!, empty.expected, { abs: 0, rel: 0 })).toBeNull();
  });
});
