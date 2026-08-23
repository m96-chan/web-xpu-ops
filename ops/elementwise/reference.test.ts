import { describe, expect, it } from "vitest";
import { ELEMENTWISE, elementwise, elementwiseRows } from "./reference.js";

/**
 * CPU-only ground truth for `ops/elementwise`.
 *
 * The numbers below are not this file's own opinion: they were taken from
 * PyTorch 2.10 in float64 (`torch.arange(6).reshape(2,3) + torch.tensor([10,
 * 100, 1000])`) rather than worked out by hand here, so that rule 7's "the
 * definition already exists, do not invent a second one" is checked against the
 * thing that defines it. They are small enough to read, and every element is
 * distinct, which is what makes a wrong index show as a wrong value rather than
 * as a coincidence.
 */

describe("elementwise (same shape)", () => {
  // Issue #150 added a row-broadcast path beside this one. These two cases
  // exist to pin what the existing callers (`llm/engine.ts`'s residual add,
  // its SwiGLU multiply) already get, so that "the same-shape path did not
  // change" is a green assertion rather than an argument.
  it("adds elementwise", () => {
    const a = new Float32Array([1, 2, 3, 4]);
    const b = new Float32Array([10, 20, 30, 40]);
    expect([...elementwise({ a, b, kind: ELEMENTWISE.add })]).toEqual([11, 22, 33, 44]);
  });

  it("multiplies elementwise", () => {
    const a = new Float32Array([1, 2, 3, 4]);
    const b = new Float32Array([10, 20, 30, 40]);
    expect([...elementwise({ a, b, kind: ELEMENTWISE.multiply })]).toEqual([10, 40, 90, 160]);
  });

  it("throws when the two arrays are different lengths", () => {
    const a = new Float32Array(4);
    const b = new Float32Array(3);
    expect(() => elementwise({ a, b, kind: ELEMENTWISE.add })).toThrow(/length mismatch: 4 vs 3/);
  });

  // The whole reason issue #150 did not make broadcast implicit. A `[2, 3]`
  // activation and a `[3]` bias have lengths 6 and 3; nothing about them says
  // "broadcast" except the caller's intent, and `elementwise` is not allowed to
  // guess it.
  it("still throws for a row-shaped b rather than broadcasting it", () => {
    const a = new Float32Array(6);
    const b = new Float32Array(3);
    expect(() => elementwise({ a, b, kind: ELEMENTWISE.add })).toThrow(/length mismatch/);
  });
});

describe("elementwiseRows (row broadcast)", () => {
  /** `torch.arange(6, dtype=torch.float64).reshape(2, 3)`. */
  const x = () => new Float32Array([0, 1, 2, 3, 4, 5]);
  /** Decades, so a term taken from the wrong column lands nowhere near the right answer. */
  const decades = () => new Float32Array([10, 100, 1000]);

  it("adds b to every row — biasAdd", () => {
    // torch 2.10: (x + b).flatten() -> [10, 101, 1002, 13, 104, 1005]
    const got = elementwiseRows({ a: x(), b: decades(), S: 2, D: 3, kind: ELEMENTWISE.add });
    expect([...got]).toEqual([10, 101, 1002, 13, 104, 1005]);
  });

  it("multiplies every row by b — rowwiseAffine", () => {
    // torch 2.10: (x * b).flatten() -> [0, 100, 2000, 30, 400, 5000]
    const got = elementwiseRows({ a: x(), b: decades(), S: 2, D: 3, kind: ELEMENTWISE.multiply });
    expect([...got]).toEqual([0, 100, 2000, 30, 400, 5000]);
  });

  /**
   * The case that makes the mutation in issue #150's acceptance criteria
   * observable, and the one the other cases cannot cover.
   *
   * `S == D == 3`, so a kernel that indexes `b` by the *row* instead of the
   * column is still in bounds and still returns three finite numbers per row —
   * it just returns the wrong ones. Both answers are written out below, from
   * torch, so the assertion states which convention this op follows instead of
   * merely observing that something happened:
   *
   *   torch  y + c              -> [[1, 3, 5], [4, 6, 8], [7, 9, 11]]   (last dim, this op)
   *   torch  y + c.unsqueeze(1) -> [[1, 2, 3], [5, 6, 7], [9, 10, 11]]  (first dim, not this op)
   *
   * `b[0]`-only ("broadcast the first element") gives a third, different answer
   * again — [[1, 2, 3], [4, 5, 6], [7, 8, 9]] — so all three of the plausible
   * ways to get the broadcast wrong separate here.
   */
  it("applies a different element of b per column, not per row, on a square input", () => {
    const y = new Float32Array([0, 1, 2, 3, 4, 5, 6, 7, 8]);
    const c = new Float32Array([1, 2, 3]);
    const got = elementwiseRows({ a: y, b: c, S: 3, D: 3, kind: ELEMENTWISE.add });
    expect([...got]).toEqual([1, 3, 5, 4, 6, 8, 7, 9, 11]);
  });

  it("uses the same b for every row, however many rows there are", () => {
    // A tall input with rows that differ only by a constant: if any row got a
    // rotated or repeated slice of `b`, the difference between consecutive rows
    // would stop being that constant.
    const S = 64;
    const D = 5;
    const b = new Float32Array([1, 2, 4, 8, 16]);
    const a = new Float32Array(S * D);
    for (let row = 0; row < S; row += 1) for (let col = 0; col < D; col += 1) a[row * D + col] = row;
    const got = elementwiseRows({ a, b, S, D, kind: ELEMENTWISE.add });
    for (let row = 0; row < S; row += 1) {
      for (let col = 0; col < D; col += 1) {
        expect(got[row * D + col]).toBe(row + b[col]!);
      }
    }
  });

  it("reduces to plain elementwise when S is 1", () => {
    const a = new Float32Array([1, 2, 3]);
    const b = new Float32Array([10, 20, 30]);
    const rows = elementwiseRows({ a, b, S: 1, D: 3, kind: ELEMENTWISE.multiply });
    expect([...rows]).toEqual([...elementwise({ a, b, kind: ELEMENTWISE.multiply })]);
  });

  it("applies b as a single scalar to the whole column when D is 1", () => {
    // torch 2.10: torch.arange(4).reshape(4,1) + torch.tensor([5.]) -> [5,6,7,8]
    const a = new Float32Array([0, 1, 2, 3]);
    const b = new Float32Array([5]);
    const got = elementwiseRows({ a, b, S: 4, D: 1, kind: ELEMENTWISE.add });
    expect([...got]).toEqual([5, 6, 7, 8]);
  });

  it("leaves the input alone for an all-zero add and an all-one multiply", () => {
    const a = new Float32Array([1, -2, 3, -4, 5, -6]);
    const zeros = new Float32Array(3);
    const ones = new Float32Array([1, 1, 1]);
    expect([...elementwiseRows({ a, b: zeros, S: 2, D: 3, kind: ELEMENTWISE.add })]).toEqual([...a]);
    expect([...elementwiseRows({ a, b: ones, S: 2, D: 3, kind: ELEMENTWISE.multiply })]).toEqual([...a]);
  });

  it("zeroes everything for an all-zero multiply", () => {
    const a = new Float32Array([1, -2, 3, -4, 5, -6]);
    const zeros = new Float32Array(3);
    const got = elementwiseRows({ a, b: zeros, S: 2, D: 3, kind: ELEMENTWISE.multiply });
    expect([...got]).toEqual([0, -0, 0, -0, 0, -0]);
  });

  it("does not write into its inputs", () => {
    const a = x();
    const b = decades();
    elementwiseRows({ a, b, S: 2, D: 3, kind: ELEMENTWISE.add });
    expect([...a]).toEqual([0, 1, 2, 3, 4, 5]);
    expect([...b]).toEqual([10, 100, 1000]);
  });

  /**
   * Issue #143's class: a wrong argument must throw, not return an empty array
   * or a well-formed wrong answer. Message shape follows `ops/gqa/reference.ts`
   * (rule 7 — the house already picked a format).
   */
  describe("rejects arguments that would otherwise break quietly", () => {
    const b = () => new Float32Array([1, 2, 3]);

    it("throws when b is not one row long", () => {
      expect(() =>
        elementwiseRows({ a: new Float32Array(6), b: new Float32Array(2), S: 2, D: 3, kind: ELEMENTWISE.add }),
      ).toThrow(/elementwiseRows\(\): b must have D=3 elements, got 2/);
    });

    // The specific mistake `elementwise`'s own length check cannot make: `b`
    // the size of the whole activation rather than of one row. PyTorch 2.10
    // refuses this too — a 1-D tensor of 6 against a [2, 3] is
    // "The size of tensor a (3) must match the size of tensor b (6)" — because
    // broadcasting aligns from the right, so `[6]` is not a `[2, 3]`.
    it("throws when b is the size of the whole input rather than of one row", () => {
      expect(() =>
        elementwiseRows({ a: new Float32Array(6), b: new Float32Array(6), S: 2, D: 3, kind: ELEMENTWISE.add }),
      ).toThrow(/elementwiseRows\(\): b must have D=3 elements, got 6/);
    });

    it("throws when a is not S*D long", () => {
      expect(() =>
        elementwiseRows({ a: new Float32Array(7), b: b(), S: 2, D: 3, kind: ELEMENTWISE.add }),
      ).toThrow(/elementwiseRows\(\): a must have S\*D=6 elements for S=2, D=3, got 7/);
    });

    for (const bad of [0, -1, 2.5, Number.NaN] as const) {
      it(`throws when S is ${bad}`, () => {
        expect(() =>
          elementwiseRows({ a: new Float32Array(6), b: b(), S: bad, D: 3, kind: ELEMENTWISE.add }),
        ).toThrow(/elementwiseRows\(\): S must be a positive integer/);
      });

      it(`throws when D is ${bad}`, () => {
        expect(() =>
          elementwiseRows({ a: new Float32Array(6), b: b(), S: 2, D: bad, kind: ELEMENTWISE.add }),
        ).toThrow(/elementwiseRows\(\): D must be a positive integer/);
      });
    }

    // Issue #143's actual reported mechanism: a misspelled argument name makes
    // the dimension `undefined`, `new Float32Array(undefined)` is length 0, and
    // the loop runs zero times. The result is an empty array and no error.
    it("throws when a dimension is missing entirely", () => {
      const args = { a: new Float32Array(6), b: b(), D: 3, kind: ELEMENTWISE.add } as unknown as Parameters<
        typeof elementwiseRows
      >[0];
      expect(() => elementwiseRows(args)).toThrow(/elementwiseRows\(\): S must be a positive integer, got undefined/);
    });
  });
});
