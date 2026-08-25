/**
 * `pad`, against torch 2.10.0+cu130.
 *
 * Issue #200. Every expectation below was measured, not read off the docs, and
 * the inputs are small integers so the goldens are exact in f32.
 *
 * The case this file exists for is the last one: **padding two axes in
 * sequence gives what padding both at once gives**. The interface takes one
 * axis, so if that were false the interface would be wrong, and it is not
 * obvious that it is true — a reflection reads its neighbours, so padding H
 * after W means the corners are reflections of already-reflected values.
 */
import { describe, expect, it } from "vitest";
import { pad, padOutputLength, padSourceIndex } from "./reference.js";

/** `[1, 1, 6]` holding 1..6 — torch's `x` in every 1D case below. */
const X = Float32Array.from([1, 2, 3, 4, 5, 6]);
const AXIS = { outer: 1, L: 6, inner: 1 } as const;

describe("pad / reference", () => {
  it("fills with zero outside the data, as torch's default", () => {
    // torch: F.pad(x, (2,3)) -> [0,0,1,2,3,4,5,6,0,0,0]
    expect([...pad({ ...AXIS, input: X, before: 2, after: 3 })]).toEqual([0, 0, 1, 2, 3, 4, 5, 6, 0, 0, 0]);
  });

  it("fills with the value it was given", () => {
    // torch: F.pad(x, (2,3), value=9) -> [9,9,1,2,3,4,5,6,9,9,9]
    expect([...pad({ ...AXIS, input: X, before: 2, after: 3, value: 9 })]).toEqual([9, 9, 1, 2, 3, 4, 5, 6, 9, 9, 9]);
  });

  it("replicate repeats the edge element", () => {
    // torch: F.pad(x, (2,3), mode="replicate") -> [1,1,1,2,3,4,5,6,6,6,6]
    expect([...pad({ ...AXIS, input: X, before: 2, after: 3, mode: "replicate" })]).toEqual([
      1, 1, 1, 2, 3, 4, 5, 6, 6, 6, 6,
    ]);
  });

  it("reflect does NOT repeat the edge element", () => {
    // torch: F.pad(x, (2,3), mode="reflect") -> [3,2,1,2,3,4,5,6,5,4,3]
    // The element before `1` is `2`. Swapping this with `replicate` gives a
    // tensor of the right shape whose interior is entirely correct.
    expect([...pad({ ...AXIS, input: X, before: 2, after: 3, mode: "reflect" })]).toEqual([
      3, 2, 1, 2, 3, 4, 5, 6, 5, 4, 3,
    ]);
  });

  it("reflects to the far edge at the largest padding torch allows", () => {
    // torch: F.pad(x, (5,5), mode="reflect") on L=6 -> the whole axis mirrored.
    expect([...pad({ ...AXIS, input: X, before: 5, after: 5, mode: "reflect" })]).toEqual([
      6, 5, 4, 3, 2, 1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1,
    ]);
  });

  it("pads only one end — the shape a causal convolution needs", () => {
    // torch: F.pad(x, (4,0), mode="replicate") -> [1,1,1,1,1,2,3,4,5,6]
    // H3's temporal padding is this asymmetry: frames before the data, none
    // after, so the output at t cannot see t+1.
    expect([...pad({ ...AXIS, input: X, before: 4, after: 0, mode: "replicate" })]).toEqual([
      1, 1, 1, 1, 1, 2, 3, 4, 5, 6,
    ]);
  });

  it("pads nothing when asked for nothing, in every mode", () => {
    for (const mode of ["constant", "reflect", "replicate"] as const) {
      expect([...pad({ ...AXIS, input: X, before: 0, after: 0, mode })]).toEqual([1, 2, 3, 4, 5, 6]);
    }
  });

  it("pads each slice of `outer` from its own data", () => {
    // torch: F.pad([[1,2,3,4],[10,20,30,40]], (2,2), mode="reflect") on [1,2,4].
    // A kernel that reflected across the whole buffer rather than within each
    // row would pull 10 into the first row's tail.
    const x2 = Float32Array.from([1, 2, 3, 4, 10, 20, 30, 40]);
    expect([...pad({ input: x2, outer: 2, L: 4, inner: 1, before: 2, after: 2, mode: "reflect" })]).toEqual([
      3, 2, 1, 2, 3, 4, 3, 2, 30, 20, 10, 20, 30, 40, 30, 20,
    ]);
  });

  it("carries whole rows when `inner` is more than one", () => {
    // torch: F.pad(x3, (0,0,1,1), mode="reflect") on [1,1,3,4] — the H axis,
    // so each padded position is a whole 4-element row.
    const x3 = Float32Array.from({ length: 12 }, (_, i) => i + 1);
    expect([...pad({ input: x3, outer: 1, L: 3, inner: 4, before: 1, after: 1, mode: "reflect" })]).toEqual([
      5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 5, 6, 7, 8,
    ]);
  });

  it("replicates whole rows too, on one end only", () => {
    // torch: F.pad(x3, (0,0,2,0), mode="replicate") -> the first row twice more.
    const x3 = Float32Array.from({ length: 12 }, (_, i) => i + 1);
    expect([...pad({ input: x3, outer: 1, L: 3, inner: 4, before: 2, after: 0, mode: "replicate" })]).toEqual([
      1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
    ]);
  });

  /**
   * The property the one-axis interface rests on.
   *
   * torch pads every axis in a single call; this pads one. If the two disagreed,
   * a caller composing three calls for a 3D convolution would get a different
   * tensor from the model it is copying — silently, since the shape is the same.
   */
  it("padding W and then H gives what padding both at once gives", () => {
    // torch: F.pad(x3, (1,1,1,1), mode="reflect") on [1,1,3,4] -> [1,1,5,6].
    const x3 = Float32Array.from({ length: 12 }, (_, i) => i + 1);
    const w = pad({ input: x3, outer: 3, L: 4, inner: 1, before: 1, after: 1, mode: "reflect" });
    const both = pad({ input: w, outer: 1, L: 3, inner: 6, before: 1, after: 1, mode: "reflect" });
    expect([...both]).toEqual([
      6, 5, 6, 7, 8, 7,
      2, 1, 2, 3, 4, 3,
      6, 5, 6, 7, 8, 7,
      10, 9, 10, 11, 12, 11,
      6, 5, 6, 7, 8, 7,
    ]);
  });

  describe("what it refuses", () => {
    it("refuses a reflect wider than the axis, where torch raises", () => {
      // torch on L=6: (5,5) is accepted, (6,0) raises "Padding size should be
      // less than the corresponding input dimension".
      expect(() => pad({ ...AXIS, input: X, before: 5, after: 5, mode: "reflect" })).not.toThrow();
      expect(() => pad({ ...AXIS, input: X, before: 6, after: 0, mode: "reflect" })).toThrow(/reflect/);
      expect(() => pad({ ...AXIS, input: X, before: 0, after: 6, mode: "reflect" })).toThrow(/reflect/);
    });

    it("allows the same widths for replicate and constant, which torch does too", () => {
      expect(() => pad({ ...AXIS, input: X, before: 20, after: 0, mode: "replicate" })).not.toThrow();
      expect(() => pad({ ...AXIS, input: X, before: 20, after: 0 })).not.toThrow();
    });

    it("refuses negative padding rather than cropping", () => {
      // torch's constant mode crops on a negative pad — a different operation
      // with a different output length. One meaning per argument.
      expect(() => pad({ ...AXIS, input: X, before: -1, after: 0 })).toThrow(/negative/);
    });

    it("refuses an input whose length does not match its declared shape", () => {
      expect(() => pad({ ...AXIS, input: Float32Array.from([1, 2]), before: 1, after: 1 })).toThrow(/input/);
    });
  });

  describe("padSourceIndex", () => {
    it("is the table the modes are defined by", () => {
      // L=6, before=2: position 0 is two before the data.
      expect(padSourceIndex(0, 6, 2, "constant")).toBe(-1);
      expect(padSourceIndex(0, 6, 2, "replicate")).toBe(0);
      expect(padSourceIndex(0, 6, 2, "reflect")).toBe(2);
      expect(padSourceIndex(1, 6, 2, "reflect")).toBe(1);
      expect(padSourceIndex(2, 6, 2, "reflect")).toBe(0);
      // Inside the data every mode agrees.
      for (const mode of ["constant", "reflect", "replicate"] as const) {
        expect(padSourceIndex(4, 6, 2, mode)).toBe(2);
      }
      // Past the far edge.
      expect(padSourceIndex(8, 6, 2, "constant")).toBe(-1);
      expect(padSourceIndex(8, 6, 2, "replicate")).toBe(5);
      expect(padSourceIndex(8, 6, 2, "reflect")).toBe(4);
    });
  });

  describe("padOutputLength", () => {
    it("is the axis plus both pads", () => {
      expect(padOutputLength({ L: 6, before: 2, after: 3 })).toBe(11);
      expect(padOutputLength({ L: 6, before: 0, after: 0 })).toBe(6);
    });
  });
});
