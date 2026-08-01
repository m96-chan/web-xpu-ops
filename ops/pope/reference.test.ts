import { describe, expect, it } from "vitest";
import { legendre, pope } from "./reference.js";

/**
 * The polynomial and the sampling, pinned to closed forms.
 *
 * The WGSL test only says that the shader and the reference agree; if the
 * recurrence or the domain mapping is wrong, they agree on the wrong answer.
 * The first four Legendre polynomials have textbook closed forms, so the
 * recurrence can be checked against something that is not itself a recurrence.
 * Touches no GPU.
 */
describe("pope / reference", () => {
  it("matches the closed forms of the first four Legendre polynomials", () => {
    const closed = [
      (_x: number) => 1,
      (x: number) => x,
      (x: number) => (3 * x ** 2 - 1) / 2,
      (x: number) => (5 * x ** 3 - 3 * x) / 2,
      (x: number) => (35 * x ** 4 - 30 * x ** 2 + 3) / 8,
    ];
    for (const [n, expected] of closed.entries()) {
      for (const x of [-1, -0.75, -0.5, -0.125, 0, 0.125, 0.5, 0.75, 1]) {
        expect(legendre(n, x)).toBeCloseTo(expected(x), 12);
      }
    }
  });

  it("keeps the endpoint identities every Legendre polynomial has", () => {
    // P_n(1) = 1 and P_n(-1) = (-1)^n for all n. A recurrence that drifts shows
    // it here first, and these hold at orders far beyond the closed forms above.
    for (const n of [0, 1, 2, 7, 20, 63]) {
      expect(legendre(n, 1)).toBeCloseTo(1, 10);
      expect(legendre(n, -1)).toBeCloseTo((-1) ** n, 10);
    }
  });

  it("samples the domain from -1 in equal steps of 2/dModel", () => {
    // Equation (14) with the reading recorded in reference.ts: the samples
    // start at -1 and stop one step short of +1. Order 1 makes the sampling
    // directly visible, since P_1(x) = x.
    const dModel = 4;
    const row = pope({ N: 1, dModel, posOffset: 1 });
    expect(Array.from(row)).toEqual([-1, -0.5, 0, 0.5]);
  });

  it("uses the token position as the polynomial order", () => {
    // The assignment that distinguishes PoPE from the sinusoidal scheme: the
    // order advances down the rows, the argument across the columns.
    const table = pope({ N: 4, dModel: 4, posOffset: 0 });
    const row = (t: number) => Array.from(table.slice(t * 4, t * 4 + 4));
    expect(row(0)).toEqual([1, 1, 1, 1]); // P_0 is the constant 1
    expect(row(1)).toEqual([-1, -0.5, 0, 0.5]); // P_1(x) = x
    expect(row(2)).toEqual([1, -0.125, -0.5, -0.125]); // (3x^2 - 1)/2
    // (5x^3 - 3x)/2. The middle entry is negative zero: the recurrence reaches
    // it by subtraction rather than by evaluating the closed form, and IEEE
    // keeps the sign. It compares equal to 0 everywhere that matters, including
    // the harness comparator, so it is recorded rather than rounded away.
    expect(row(3)).toEqual([-1, 0.4375, -0, -0.4375]);
  });

  it("shifts the order by posOffset", () => {
    // The ambiguity the paper leaves open, made a caller's decision. With
    // posOffset = 1 the first token is P_1 rather than the constant P_0, which
    // is the difference between the first token carrying position or not.
    const zeroBased = pope({ N: 3, dModel: 4, posOffset: 0 });
    const oneBased = pope({ N: 3, dModel: 4, posOffset: 1 });
    expect(Array.from(oneBased.slice(0, 8))).toEqual(Array.from(zeroBased.slice(4, 12)));
    expect(Array.from(zeroBased.slice(0, 4))).toEqual([1, 1, 1, 1]);
    expect(Array.from(oneBased.slice(0, 4))).toEqual([-1, -0.5, 0, 0.5]);
  });

  it("stays bounded by 1, which is what makes an f32 kernel possible", () => {
    // |P_n(x)| <= 1 on [-1, 1]. If the recurrence were unstable this is the
    // first thing that would stop being true, and the tolerance in the WGSL
    // test would be hiding it.
    const table = pope({ N: 64, dModel: 48, posOffset: 1 });
    expect(table.every((value) => Math.abs(value) <= 1 + 1e-12)).toBe(true);
  });
});
