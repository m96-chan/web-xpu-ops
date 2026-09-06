/**
 * `evictionPlan` in isolation, no GPU.
 *
 * Issue #223. `DitGpu.release()` and `VideoDecoderGpu.release()` only ever
 * grow their free pool, so resident memory is the SUM of every size class's
 * own peak where what a step actually needs at once is the MAX — the
 * attention stretch and the feed-forward stretch of a block never run
 * together, but both stayed resident. `evictionPlan` decides which free
 * buffers to destroy, largest-first, so the pool's free bytes fit a budget;
 * `pool.ts` wires it into both `release()`s.
 */
import { describe, expect, it } from "vitest";
import { evictionPlan } from "./pool.js";

describe("evictionPlan", () => {
  it("returns an exactly empty plan when already under budget", () => {
    const plan = evictionPlan([{ size: 100, count: 3 }], 1000);
    expect(plan.size).toBe(0);
  });

  it("evicts just enough of one over-budget class, not all of it", () => {
    // total = 500; budget 200 is reached exactly after 3 of the 5 are gone
    // (500 -> 400 -> 300 -> 200), so "just enough" is unambiguous here — an
    // off-by-one either stops one short (total 300, still over budget) or
    // goes one past (evicts a 4th, total 100, more than needed).
    const plan = evictionPlan([{ size: 100, count: 5 }], 200);
    expect(plan.get(100)).toBe(3);
    expect(plan.size).toBe(1);
  });

  it("evicts from the largest class first, leaving the smaller untouched if that suffices", () => {
    // large: 2 x 200 = 400; small: 4 x 50 = 200; sum = 600 > budget 500, but
    // each class alone (400, 200) is under 500. Evicting one large buffer
    // (400 -> 200 total, wait: 600 - 200 = 400 <= 500) already fits.
    const plan = evictionPlan(
      [
        { size: 200, count: 2 },
        { size: 50, count: 4 },
      ],
      500,
    );
    expect(plan.get(200)).toBe(1);
    expect(plan.has(50)).toBe(false);
  });

  it("evicts everything at budget 0", () => {
    const free = [
      { size: 200, count: 2 },
      { size: 50, count: 4 },
    ];
    const plan = evictionPlan(free, 0);
    expect(plan.get(200)).toBe(2);
    expect(plan.get(50)).toBe(4);
  });

  it("evicts nothing at budget Infinity", () => {
    const plan = evictionPlan(
      [
        { size: 200, count: 2 },
        { size: 50, count: 4 },
      ],
      Number.POSITIVE_INFINITY,
    );
    expect(plan.size).toBe(0);
  });

  it("never plans to destroy more than exist in a class", () => {
    const plan = evictionPlan([{ size: 100, count: 2 }], 0);
    expect(plan.get(100)).toBeLessThanOrEqual(2);
  });
});
