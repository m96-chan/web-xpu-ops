import { describe, expect, it } from "vitest";
import { moeDispatch, moeGather, router } from "./reference.js";

/**
 * The routing policy, written as expectations rather than as prose.
 *
 * Each of these is a decision with more than one defensible answer — tie
 * breaking, which token loses its seat when an expert fills up, whether the
 * gates sum to one. `wgsl.test.ts` checks that the kernel agrees with the
 * reference; this file checks that the reference is the thing that was
 * actually chosen, so that a later "simplification" of the reference has to
 * argue with a failing test rather than with a comment.
 *
 * Touches no GPU, so it costs the suite nothing.
 */
describe("moe / reference", () => {
  it("breaks top-k ties towards the lower expert index", () => {
    // torch.topk documents no order for equal elements. A kernel cannot leave
    // it open — the same token would reach different experts on different runs
    // — so the rule is stated here. An all-equal row is not a curiosity: a
    // freshly initialised router produces one, and so does any mask that
    // writes a constant.
    const { expert, gate } = router({
      logits: Float32Array.from([2, 2, 2, 2]),
      T: 1,
      E: 4,
      k: 3,
      normalize: false,
    });
    expect(Array.from(expert)).toEqual([0, 1, 2]);
    // Equal logits mean equal gates: a quarter each, un-normalised.
    expect(Array.from(gate)).toEqual([0.25, 0.25, 0.25]);
  });

  it("ranks by logit first and index only within a tie", () => {
    // The tie is at the top and again lower down, with a clear winner between
    // them, so an implementation that sorted by index would be caught.
    const { expert } = router({
      logits: Float32Array.from([1, 5, 3, 5, 1]),
      T: 1,
      E: 5,
      k: 5,
      normalize: false,
    });
    expect(Array.from(expert)).toEqual([1, 3, 2, 0, 4]);
  });

  it("normalises the k gates to one, or leaves them as full softmax", () => {
    const logits = Float32Array.from([0, 1, 2, 3]);
    const shape = { logits, T: 1, E: 4, k: 2 } as const;

    const plain = router({ ...shape, normalize: false });
    // exp(3) + exp(2) over exp(0)+exp(1)+exp(2)+exp(3): well under 1, which is
    // the whole point of the Switch gate.
    const total = Array.from(plain.gate).reduce((a, b) => a + b, 0);
    expect(total).toBeLessThan(0.9);
    expect(total).toBeGreaterThan(0.8);

    const normalised = router({ ...shape, normalize: true });
    expect(Array.from(normalised.gate).reduce((a, b) => a + b, 0)).toBeCloseTo(1, 6);
    // Renormalising is a rescale, not a reshuffle: the ratio between the two
    // gates has to survive it.
    expect(normalised.gate[0]! / normalised.gate[1]!).toBeCloseTo(plain.gate[0]! / plain.gate[1]!, 5);
  });

  it("gives a token's higher-ranked choice the seat when an expert fills up", () => {
    // Three tokens, two experts, room for one token each. In rank-major order
    // (GShard) the seats go to the rank-0 choices of tokens 0 and 1. Flattening
    // (token, rank) row-major instead — a stable sort by expert id, which is
    // what Megablocks does — would hand expert 1's seat to token 0's *second*
    // choice and drop token 1 entirely. Both are implementable; they are not
    // the same model.
    const expert = Int32Array.from([0, 1, 1, 0, 0, 1]);
    const x = Float32Array.from([10, 20, 30]);
    const { pos, buffer } = moeDispatch({ x, expert, T: 3, k: 2, E: 2, C: 1, D: 1 });

    expect(Array.from(pos)).toEqual([0, -1, 0, -1, -1, -1]);
    // Row-major order would have given [10, 10] here.
    expect(Array.from(buffer)).toEqual([10, 20]);
  });

  it("keeps counting past capacity, so a drop does not free the seat", () => {
    // Four tokens into one seat. The rule is "the first in this order", not
    // "the first that happened not to be dropped" — if the count stopped
    // advancing at capacity, token 1 would take the seat back from token 0.
    const expert = Int32Array.from([0, 0, 0, 0]);
    const x = Float32Array.from([10, 20, 30, 40]);
    const { pos, buffer } = moeDispatch({ x, expert, T: 4, k: 1, E: 1, C: 1, D: 1 });
    expect(Array.from(pos)).toEqual([0, -1, -1, -1]);
    expect(Array.from(buffer)).toEqual([10]);
  });

  it("drops an expert id outside [0, E) rather than clamping it", () => {
    const expert = Int32Array.from([-1, 5, 1]);
    const x = Float32Array.from([10, 20, 30]);
    const { pos, buffer } = moeDispatch({ x, expert, T: 3, k: 1, E: 2, C: 2, D: 1 });
    expect(Array.from(pos)).toEqual([-1, -1, 0]);
    // Clamping would have put tokens 0 and 2 in expert 0 and 1 respectively.
    expect(Array.from(buffer)).toEqual([0, 0, 30, 0]);
  });

  it("leaves an expert that nobody chose at zero, and produces no NaN", () => {
    // The failure this guards is a kernel that averages over an expert's token
    // count: 0/0 for every expert a short sequence never reaches, and with 64
    // experts that is most of them.
    const logits = Float32Array.from([5, 1, 0, 5, 1, 0]);
    const { expert, gate } = router({ logits, T: 2, E: 3, k: 1, normalize: true });
    expect(Array.from(expert)).toEqual([0, 0]);

    const x = Float32Array.from([1, 2, 3, 4]);
    const shape = { T: 2, k: 1, E: 3, C: 2, D: 2 } as const;
    const { buffer, pos } = moeDispatch({ x, expert, ...shape });
    // Experts 1 and 2 got nothing; their rows stay zero rather than NaN.
    expect(Array.from(buffer.subarray(4))).toEqual([0, 0, 0, 0, 0, 0, 0, 0]);

    const out = moeGather({ y: buffer, expert, pos, gate, ...shape });
    expect(Array.from(out).every(Number.isFinite)).toBe(true);
    // k = 1 normalised means the gate is exactly 1, so the round trip is the
    // identity.
    expect(Array.from(out)).toEqual([1, 2, 3, 4]);
  });

  it("applies the gate once, in gather", () => {
    // Dispatch copies verbatim, so the gate has to be visible exactly once in
    // the round trip. With normalised gates and an identity expert the answer
    // is the input; squaring the gate would give 0.61 of it here, which is the
    // kind of wrong that survives a plausibility check.
    const logits = Float32Array.from([1, 2]);
    const shape = { T: 1, k: 2, E: 2, C: 1, D: 1 } as const;
    const { expert, gate } = router({ logits, T: 1, E: 2, k: 2, normalize: true });
    const x = Float32Array.from([7]);
    const { buffer, pos } = moeDispatch({ x, expert, ...shape });
    // The gate is not in the dispatched copy.
    expect(Array.from(buffer)).toEqual([7, 7]);
    const out = moeGather({ y: buffer, expert, pos, gate, ...shape });
    expect(out[0]).toBeCloseTo(7, 5);
    const squared = moeGather({
      y: buffer,
      expert,
      pos,
      gate: Float32Array.from(gate, (g) => g * g),
      ...shape,
    });
    expect(squared[0]).toBeLessThan(6.9);
  });

  it("fills surplus slots with -1 when k exceeds the expert count", () => {
    const { expert, gate } = router({
      logits: Float32Array.from([1, 2]),
      T: 1,
      E: 2,
      k: 4,
      normalize: true,
    });
    expect(Array.from(expert)).toEqual([1, 0, -1, -1]);
    expect(Array.from(gate).reduce((a, b) => a + b, 0)).toBeCloseTo(1, 6);
    expect(gate[2]).toBe(0);
    expect(gate[3]).toBe(0);
  });

  it("returns a zero row for a token whose every choice was dropped", () => {
    const expert = Int32Array.from([0, 0]);
    // Halves, so that the products below are exact in f32 and the assertion can
    // be equality rather than a tolerance.
    const gate = Float32Array.from([0.5, 0.25]);
    const x = Float32Array.from([1, 2, 3, 4]);
    const shape = { T: 2, k: 1, E: 1, C: 1, D: 2 } as const;
    const { buffer, pos } = moeDispatch({ x, expert, ...shape });
    expect(Array.from(pos)).toEqual([0, -1]);
    const out = moeGather({ y: buffer, expert, pos, gate, ...shape });
    // Token 1 lost its only seat: zeros, not somebody else's row scaled.
    expect(Array.from(out)).toEqual([0.5, 1, 0, 0]);
  });
});
