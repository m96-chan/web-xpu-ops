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
const RESIDENT_REFCOUNT = Symbol.for("web-xpu-ops.resident-device-refcount");
type ResidentHolder = { [RESIDENT_SHARED]?: Promise<ResidentDevice | null>; [RESIDENT_REFCOUNT]?: number };

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
 *
 * Refcounted, not "destroy in the first file's own `afterAll`": every file
 * that calls this registers its own `beforeAll`/`afterAll` pair (that part is
 * unavoidable, `vitest` hooks are always per-file), and with more than one
 * such file in the run, an unconditional `destroy()` in the first file's
 * `afterAll` tore the shared device down the moment that file's tests
 * finished — the *next* file's `beforeAll` then paid a fresh
 * `createResidentDevice()` (adapter request, device request) to replace it,
 * one extra native device create/destroy cycle per additional file, despite
 * this function's own doc already promising "one device for the run" (PR
 * #116 review, item 8 — flagged as a plausible contributor to this run's
 * `std::system_error` flakes in `llm/engine-q8.wgsl.test.ts`, an unrelated
 * file whose only shared state with this one is the same forked process).
 * Counting `beforeAll`/`afterAll` calls against each other means the device
 * outlives every *individual* file's hooks and is torn down only once, by
 * whichever of this run's resident-using files happens to finish last.
 */
export function useResidentGpu(): () => ResidentDevice | null {
  beforeAll(async () => {
    const holder = globalThis as ResidentHolder;
    holder[RESIDENT_REFCOUNT] = (holder[RESIDENT_REFCOUNT] ?? 0) + 1;
    holder[RESIDENT_SHARED] ??= createResidentDevice();
    currentResident = await holder[RESIDENT_SHARED];
  }, 60_000);
  afterAll(async () => {
    const holder = globalThis as ResidentHolder;
    holder[RESIDENT_REFCOUNT] = (holder[RESIDENT_REFCOUNT] ?? 1) - 1;
    currentResident = null;
    if (!holder[RESIDENT_SHARED]) return;
    if (holder[RESIDENT_REFCOUNT]! > 0) return; // another file's hooks still need this device
    const device = await holder[RESIDENT_SHARED];
    delete holder[RESIDENT_SHARED];
    device?.destroy();
  });
  return () => currentResident;
}

let currentResident: ResidentDevice | null = null;

/**
 * Shared by `residentTest` (and, in spirit, `gpuTest`): reports a visible
 * **skip** rather than a silent pass when the resource a test needs is
 * unavailable. `it(name, async () => { if (!resource) return; ... })` — the
 * shape this replaces — makes that PASS with zero assertions, which is
 * exactly the failure this repository has already hit twice (`gpuTest`'s own
 * identical early-return, and #108's real-model test before it moved to
 * `describe.skipIf`; see `llm/engine-q8.real-model.test.ts`'s comment).
 * `describe.skipIf` is not available here: it decides at collection time,
 * before `useResidentGpu`'s `beforeAll` has resolved whether an adapter
 * exists, so the decision has to be made inside the test body instead —
 * vitest 2.1's per-test `ctx.skip()` (it throws, unwinding the test as
 * `"skip"` rather than `"pass"` or `"fail"`).
 *
 * A plain function, not inlined into `residentTest`, so `harness/suite.test.ts`
 * can check "did skip() get called" without needing a real vitest run missing
 * an adapter — the one thing this repository's own dev machine cannot
 * exercise (it always has one).
 *
 * `ctx` is typed structurally (`{ skip }`) rather than as vitest's own
 * `TestContext`/`TaskContext`: the callback `it()` actually passes is
 * `ExtendedContext<Test>` (`@vitest/runner`'s internal type, not exported
 * as such from the `vitest` package's own surface in a directly reusable
 * shape here), and the one property this function needs from it is `skip`.
 */
export function skipUnlessPresent<T>(ctx: { skip: () => void }, resource: T | null): resource is T {
  if (resource === null) {
    ctx.skip();
    return false;
  }
  return true;
}

export function residentTest(name: string, body: (device: ResidentDevice) => Promise<void>): void {
  it(name, async (ctx) => {
    if (!skipUnlessPresent(ctx, currentResident)) return;
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
