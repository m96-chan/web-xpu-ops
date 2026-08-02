import { beforeAll, describe, expect, it } from "vitest";
import { measureRoofline, type Roofline } from "./roofline.js";
import { createRunner, type Runner } from "./wgsl.js";

/**
 * The denominator every performance claim will be divided by.
 *
 * Which makes a wrong one worse than none: an op reported at "40% of roofline"
 * against an inflated ceiling looks like there is headroom to chase that is not
 * there, and against a deflated one looks finished when it is not.
 *
 * The tests below are shaped by what actually went wrong while building it. An
 * earlier version looped over the buffer to make timing comfortable and reported
 * **9 TB/s on a 1.79 TB/s card**, passing every check that only asked for a
 * positive number. So these assert the relationships that a fake ceiling breaks,
 * not merely that a figure came back.
 */

let runner: Runner | null = null;
let measured: Roofline | null = null;

describe("roofline", () => {
  beforeAll(async () => {
    runner = await createRunner();
    if (runner) measured = await measureRoofline(runner);
  }, 120_000);

  it("states a ceiling, or null where the device cannot be timed", async () => {
    if (!runner) return;
    if (measured === null) {
      // Null is a legitimate answer, not a failure: without `timestamp-query`
      // there is nothing honest to report, and rule 9 prefers "unmeasured" to a
      // guess. Assert it is genuinely null rather than a broken object.
      expect(measured).toBeNull();
      return;
    }
    expect(measured.bandwidth).toBeGreaterThan(0);
    expect(measured.compute).toBeGreaterThan(0);
  });

  it("does not report a bandwidth no memory system can deliver", async () => {
    if (!runner || !measured) return;
    // 20 TB/s exceeds any single device shipping today by a wide margin, so this
    // cannot fail on legitimately fast hardware — but it does catch the failure
    // that actually happened, where reads were served without reaching memory.
    // A ceiling that only has to be positive would have passed at 9 TB/s.
    expect(measured.bandwidth).toBeLessThan(20e12);
  });

  it("does not report compute beyond what silicon does", async () => {
    if (!runner || !measured) return;
    // Same shape of guard, on the other axis. 10 PFLOP/s of f32 is far past any
    // single device, so this fails only if the kernel is being elided rather
    // than executed.
    expect(measured.compute).toBeLessThan(10e15);
  });

  it("does not let the bandwidth figure be a compute figure in disguise", async () => {
    if (!runner || !measured) return;
    // The streaming kernel does one 32-bit add per 4 bytes read: 0.25 FLOP/byte.
    // If the reported bandwidth were so large that this implied FLOP rate
    // approached the compute ceiling, "bandwidth" would be describing the same
    // limit "compute" does and the two numbers would be one number.
    //
    // What this does NOT catch, established by mutation rather than assumed: a
    // streaming kernel made deliberately arithmetic-bound survives it, because
    // that makes the measured bandwidth *smaller*, which only satisfies the
    // inequality harder. Catching that direction needs an independent bandwidth
    // reference, and there is none — having one is the problem this file exists
    // to solve. The guard is therefore one-sided, and says so.
    const streamingFlops = measured.bandwidth * 0.25;
    expect(streamingFlops).toBeLessThan(measured.compute / 4);
  });

  it("agrees with itself closely enough to divide by", async () => {
    if (!runner || !measured) return;
    // A ceiling that moves run to run cannot support "this op reached 62%". A
    // second independent calibration must land near the first; 25% is loose
    // enough for a shared GPU and tight enough that a number drawn from noise
    // rather than from the device would fail it.
    const again = await measureRoofline(runner);
    expect(again).not.toBeNull();
    const drift = (a: number, b: number) => Math.abs(a - b) / Math.max(a, b);
    expect(drift(measured.bandwidth, again!.bandwidth)).toBeLessThan(0.25);
    expect(drift(measured.compute, again!.compute)).toBeLessThan(0.25);
  }, 120_000);
});
