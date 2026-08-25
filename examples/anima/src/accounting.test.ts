/**
 * The forward breakdown adds up, and cannot add up to more than the wall.
 *
 * Issue #182. A browser forward summed 2014 ms of pass timestamps against
 * 3841 ms of wall and had nothing to say about the other 1827 ms. The fix
 * carries three CPU-side numbers up beside the GPU ones — but two of them
 * **overlap**: `submitToDoneMs` is the wall spent waiting for the queue to
 * drain, so it contains the very GPU execution `byKernel` breaks down. Adding
 * them produces a breakdown reaching past 100%, which reads as authoritative
 * and is not.
 *
 * So the arithmetic is the thing under test, not the plumbing: these run in
 * Node with no GPU, because `accountForForward` is a pure function and the way
 * to know it is right is to hand it numbers whose answer is known.
 */
import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { accountForForward, type AnimaProfile } from "./dit-resident.js";

/** `byKernel` totalling `gpuMs`, split over two kernels so the sum is not the identity. */
function profile(gpuMs: number, cpu: { encodeMs: number; submitToDoneMs: number; readbackMs: number }): AnimaProfile {
  return {
    byKernel: new Map([
      ["a", { seconds: (gpuMs * 0.75) / 1000, dispatches: 3 }],
      ["b", { seconds: (gpuMs * 0.25) / 1000, dispatches: 1 }],
    ]),
    supported: true,
    ...cpu,
  };
}

describe("anima / forward accounting", () => {
  it("splits the wall into parts that sum back to it", () => {
    // 1000 ms wall: 600 inside passes, a 700 ms queue wait containing them,
    // 200 recording, 50 reading back. Named: 600 + 100 + 200 + 50 = 950.
    const acct = accountForForward(profile(600, { encodeMs: 200, submitToDoneMs: 700, readbackMs: 50 }), 1.0);
    expect(acct.inPassesMs).toBeCloseTo(600, 6);
    expect(acct.aroundPassesMs).toBeCloseTo(100, 6);
    expect(acct.unattributedMs).toBeCloseTo(50, 6);
    const sum = acct.inPassesMs + acct.aroundPassesMs + acct.encodeMs + acct.readbackMs + acct.unattributedMs;
    expect(sum).toBeCloseTo(acct.wallMs, 6);
  });

  it("does not double-count the GPU time inside the queue wait", () => {
    // The trap: `submitToDone` (700) already contains the passes (600). A
    // breakdown that added both would name 1500 ms of a 1000 ms forward.
    const p = profile(600, { encodeMs: 200, submitToDoneMs: 700, readbackMs: 50 });
    const acct = accountForForward(p, 1.0);
    expect(acct.inPassesMs + acct.aroundPassesMs).toBeCloseTo(p.submitToDoneMs, 6);
    expect(acct.coverage).toBeLessThanOrEqual(100);
  });

  it("reports the browser's own numbers as a gap rather than as coverage", () => {
    // The measurement that opened #182: 2014 ms of passes in a 3841 ms forward,
    // with the CPU-side fields still zero because nothing recorded them.
    const acct = accountForForward(profile(2014, { encodeMs: 0, submitToDoneMs: 2014, readbackMs: 0 }), 3.841);
    expect(acct.unattributedMs).toBeCloseTo(1827, 0);
    expect(Math.round(acct.coverage)).toBe(52);
  });

  it("never reports a negative row when the timestamps outrun the wait", () => {
    // Timestamps and `performance.now()` are different clocks; the sum of
    // passes can exceed the wait they happened inside. That is an artefact, and
    // a negative row would read as a measurement.
    const acct = accountForForward(profile(900, { encodeMs: 10, submitToDoneMs: 800, readbackMs: 5 }), 1.0);
    expect(acct.aroundPassesMs).toBe(0);
    expect(acct.unattributedMs).toBeGreaterThanOrEqual(0);
  });

  it("never claims more than the wall clock", () => {
    // Every phase overrunning at once still cannot name more than 100%.
    const acct = accountForForward(profile(900, { encodeMs: 900, submitToDoneMs: 900, readbackMs: 900 }), 1.0);
    expect(acct.coverage).toBeLessThanOrEqual(100);
    expect(acct.unattributedMs).toBe(0);
  });

  it("a forward with no wall clock reports no coverage, not a division by zero", () => {
    const acct = accountForForward(profile(0, { encodeMs: 0, submitToDoneMs: 0, readbackMs: 0 }), 0);
    expect(acct.coverage).toBe(0);
    expect(Number.isFinite(acct.coverage)).toBe(true);
  });

  /**
   * The arithmetic above is only worth anything if the numbers reach it.
   *
   * `readbackMs` was written by `batch()` and never carried into `AnimaProfile`,
   * so it was discarded once per batch and the breakdown was missing a phase
   * entirely — the pure-function tests above all pass with it stuck at zero.
   * This reads `collect()` and asserts each CPU field is accumulated from the
   * sink field of the same name, which is the specific way that was missed:
   * a field added to the interface and forgotten one level up.
   */
  it("collect() accumulates every CPU field, not just the ones it started with", () => {
    const source = readFileSync(fileURLToPath(new URL("./dit-resident.ts", import.meta.url)), "utf8");
    for (const field of ["encodeMs", "submitToDoneMs", "readbackMs"] as const) {
      expect(
        source.includes(`profile.${field} += sink.${field}`),
        `dit-resident.ts: AnimaProfile.${field} is never accumulated from sink.${field}, so it stays 0 ` +
          "and the breakdown silently loses that phase",
      ).toBe(true);
    }
  });
});
