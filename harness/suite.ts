import { readFileSync } from "node:fs";
import { afterAll, beforeAll, expect, it } from "vitest";
import { agree, type Tolerance } from "./agree.js";
import { createResidentDevice, type ResidentDevice } from "./resident.js";
import { createRunner, type Dispatch, type Runner } from "./wgsl.js";

/**
 * Cached on globalThis, not in a module variable: vitest gives each test file a
 * fresh module registry, so a module-level cache means one device per file. Dawn
 * does not enjoy that.
 */
const SHARED = Symbol.for("web-xpu-ops.runner");
type Holder = { [SHARED]?: Promise<Runner | null> };

/**
 * One device for the whole run, torn down at the end.
 *
 * Call this once per test file. Adapters are expensive, and a device per test
 * exhausts the driver long before the suite finishes.
 */
export function useGpu(): () => Runner | null {
  beforeAll(async () => {
    const holder = globalThis as Holder;
    holder[SHARED] ??= createRunner();
    current = await holder[SHARED];
  }, 60_000);
  // Released once, at the end of the run. Dawn aborts the process if a device
  // outlives it, and vitest keeps the registry alive between files.
  afterAll(async () => {
    const holder = globalThis as Holder;
    if (!holder[SHARED]) return;
    const runner = await holder[SHARED];
    delete holder[SHARED];
    current = null;
    runner?.destroy();
  });
  return () => current;
}

let current: Runner | null = null;

export function gpuTest(name: string, body: (run: Runner["run"]) => Promise<void>): void {
  it(name, async () => {
    if (!current) return;
    await body(current.run);
  });
}

const RESIDENT_SHARED = Symbol.for("web-xpu-ops.resident-device");
type ResidentHolder = { [RESIDENT_SHARED]?: Promise<ResidentDevice | null> };

/**
 * `useGpu`'s counterpart for `harness/resident.ts` — same one-device-for-the-run,
 * cached-on-globalThis, torn-down-at-the-end shape, for the same reason (Dawn
 * does not tolerate a device per test, or a device outliving the process).
 * A separate cache key and a separate device instance from `useGpu`'s,
 * deliberately: `ResidentDevice` is requested with different buffer-size
 * limits (`resident.ts`'s own doc — a resident engine's weight buffers need
 * headroom `createRunner`'s calibration-sized request does not), and sharing
 * one device between a `Runner`-style caller and a `ResidentDevice`-style
 * caller would mean either one's buffer lifetime assumptions could leak into
 * the other's.
 */
export function useResidentGpu(): () => ResidentDevice | null {
  beforeAll(async () => {
    const holder = globalThis as ResidentHolder;
    holder[RESIDENT_SHARED] ??= createResidentDevice();
    currentResident = await holder[RESIDENT_SHARED];
  }, 60_000);
  afterAll(async () => {
    const holder = globalThis as ResidentHolder;
    if (!holder[RESIDENT_SHARED]) return;
    const device = await holder[RESIDENT_SHARED];
    delete holder[RESIDENT_SHARED];
    currentResident = null;
    device?.destroy();
  });
  return () => currentResident;
}

let currentResident: ResidentDevice | null = null;

export function residentTest(name: string, body: (device: ResidentDevice) => Promise<void>): void {
  it(name, async () => {
    if (!currentResident) return;
    await body(currentResident);
  });
}

/** Reads a kernel next to the calling test. */
export function kernel(url: string | URL, name = "kernel"): string {
  return readFileSync(new URL(`./wgsl/${name}.wgsl`, url), "utf8");
}

/** Runs a dispatch and asserts every output agrees with its reference. */
export async function expectAgrees(
  run: Runner["run"],
  dispatch: Dispatch,
  expected: ArrayLike<number>[],
  /**
   * One tolerance, or one per output. Kernels that emit both integers and
   * floats need the distinction: quantized codes have to match exactly, while
   * the scales beside them are f32 and differ by an ulp.
   */
  tolerance?: Tolerance | (Tolerance | undefined)[],
): Promise<void> {
  const actual = await run(dispatch);
  expect(actual).toHaveLength(expected.length);
  actual.forEach((got, index) => {
    const limit = Array.isArray(tolerance) ? tolerance[index] : tolerance;
    const worst = agree(got, expected[index]!, limit);
    // Reported rather than asserted away: a failure should say which element
    // and by how much, not just that something differed.
    expect(worst, worst ? `output ${index}: ${JSON.stringify(worst)}` : undefined).toBeNull();
  });
}
