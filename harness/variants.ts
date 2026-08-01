import { describe } from "vitest";
import { variantSuites, type Variant } from "./resolve.js";

/**
 * Test-side helpers for ops that ship more than one spelling of a kernel.
 *
 * This lives apart from `harness/suite.ts`, and nothing here is re-exported from
 * `harness/index.ts`, on purpose. `index.ts` is what every op's test imports, and
 * an op that only ever runs `kernel.wgsl` has no reason to pull kernel
 * resolution onto its import graph. Import this directly from the ops that use
 * it:
 *
 * ```ts
 * import { eachVariant } from "../../harness/variants.js";
 * ```
 *
 * The separation is also what keeps `harness/suite.ts` and `harness/index.ts`
 * byte-identical to what they were before resolution existed, so adding this
 * feature changed nothing about how an existing op's test is loaded.
 */

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
