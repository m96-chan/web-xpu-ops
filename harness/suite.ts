import { readFileSync } from "node:fs";
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { agree, type Tolerance } from "./agree.js";
import { variantSuites, type Variant } from "./resolve.js";
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

/** Reads a kernel next to the calling test. */
export function kernel(url: string | URL, name = "kernel"): string {
  return readFileSync(new URL(`./wgsl/${name}.wgsl`, url), "utf8");
}

/**
 * Runs a block of tests once per variant of one entry point beside the caller.
 *
 * The loop comes from the directory, not from a list in the test, so a variant
 * added to `wgsl/` gets the op's reference comparison whether or not anyone
 * remembered to mention it. A target-specific kernel that is fast and wrong is
 * the failure this exists to make impossible, and it only stays impossible while
 * every variant goes through the same reference.
 *
 * ```ts
 * eachVariant(import.meta.url, "kernel", ({ code }) => {
 *   gpuTest("agrees with the reference", async (run) => { ... });
 * });
 * ```
 *
 * The entry point is named rather than defaulted, even for ops that only have
 * `kernel`. An op with two of them — `stft` has `kernel` and `inverse`,
 * `attention` has `scores` and `context` — needs two calls with different
 * dispatch shapes, and a default would let the second one be forgotten silently.
 * Naming it also gives `harness/coverage.ts` something to check against: it
 * catches the other half, an op that grows a variant on an entry point no test
 * loops.
 */
export function eachVariant(
  url: string | URL,
  entry: string,
  define: (found: { variant: Variant; code: string }) => void,
): void {
  for (const found of variantSuites(new URL("./wgsl/", url), entry)) {
    describe(found.variant.name, () => define(found));
  }
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
