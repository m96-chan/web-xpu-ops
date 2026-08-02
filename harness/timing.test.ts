import { beforeAll, describe, expect, it } from "vitest";
import { createRunner, params, type Dispatch, type Runner } from "./wgsl.js";

/**
 * `time()` has to report GPU time, not the host's view of it.
 *
 * Measured on this machine: wrapping `run()` in `performance.now()` charges the
 * dispatch for buffer creation, submission and the mapped-readback round trip —
 * about 1.2 ms, several times a real dispatch. Taking a roofline that way
 * reported 5.3 TB/s on a card whose ceiling is 1.79, which is precisely the
 * authoritative-looking-but-unrelated figure issue #3 exists to prevent.
 */

let runner: Runner | null = null;
let supported = false;

/** Busy arithmetic in registers, with `iterations` the only thing that scales. */
const BUSY = `
struct Params { iterations: u32, seed: f32 }
@group(0) @binding(0) var<storage, read_write> sink: array<f32>;
@group(0) @binding(1) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  var a = params.seed;
  var b = a + 1.0;
  let k = params.seed * 0.5 + 1.0;
  for (var i = 0u; i < params.iterations; i += 1u) {
    a = fma(a, k, 1.0);
    b = fma(b, k, 1.0);
  }
  // Never true, so nothing is written — but the compiler cannot fold the loop
  // away, because it cannot know that without running it.
  if (a + b == params.seed - 999.0) { sink[gid.x % 256u] = a; }
}
`;

const busy = (iterations: number): Dispatch => ({
  code: BUSY,
  bindings: [
    { kind: "out", type: "f32", length: 256 },
    { kind: "uniform", data: params([["u32", iterations], ["f32", 1.0001]]) },
  ],
  workgroups: [128],
});

describe("harness / dispatch timing", () => {
  beforeAll(async () => {
    runner = await createRunner();
    if (!runner) return;
    // One dispatch to get past pipeline creation and any first-use cost, so the
    // measurements below are of steady-state execution.
    supported = (await runner.time(busy(1000))) !== null;
  }, 60_000);

  it("reports a positive duration, or null where the device cannot say", async () => {
    if (!runner) return;
    const seconds = await runner.time(busy(8_000));
    if (seconds === null) {
      // A device without `timestamp-query` must say so rather than guess. Rule 9:
      // an unmeasured figure says it is unmeasured, and a fabricated one is worse
      // than none because it looks authoritative.
      expect(supported).toBe(false);
      return;
    }
    expect(seconds).toBeGreaterThan(0);
    // A dispatch this size is milliseconds. A whole second would mean the units
    // are wrong, which is the failure that would silently corrupt every ratio
    // built on top of this.
    expect(seconds).toBeLessThan(1);
  });

  it("charges more for more work", async () => {
    if (!runner || !supported) return;
    // The claim that makes the number usable at all: it tracks the work, rather
    // than reporting some fixed per-dispatch cost. Sixteen times the arithmetic
    // must be visibly longer — asserted loosely (2x) because the point is that
    // it scales, not what the slope is.
    //
    // Sized small on purpose. Dispatches long enough to take milliseconds, run
    // repeatedly, kill this binding (#68), and this file shares a run with the
    // roofline calibration, which is itself heavy.
    const little = await runner.time(busy(10_000));
    const lots = await runner.time(busy(64_000));
    expect(little).not.toBeNull();
    expect(lots).not.toBeNull();
    expect(lots!).toBeGreaterThan(little! * 2);
  });

  it("excludes the host-side cost that makes wall-clock useless here", async () => {
    if (!runner || !supported) return;
    // The whole reason this exists. The same dispatch, timed both ways: the GPU
    // figure must be meaningfully smaller than the wall-clock one, because the
    // difference is buffer creation, submission and the readback round trip.
    const dispatch = busy(8_000);
    await runner.time(dispatch);
    const started = performance.now();
    const gpu = await runner.time(dispatch);
    const wall = (performance.now() - started) / 1000;
    expect(gpu).not.toBeNull();
    expect(gpu!).toBeLessThan(wall);
  });
});
