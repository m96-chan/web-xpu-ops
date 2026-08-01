import { readFileSync, readdirSync, statSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { variantsIn } from "./resolve.js";

/**
 * Stops a target-specific kernel from reaching users untested.
 *
 * The failure mode this repository has to design against is a variant that is
 * fast and wrong: someone drops `nvidia.wgsl` into an op, the suite keeps
 * compiling `kernel.wgsl` only, everything stays green, and the first person to
 * find out is whoever owns an NVIDIA card.
 *
 * Two things close it, and only together:
 *
 *  1. `eachVariant` builds the test loop from the directory rather than from a
 *     hand-written list, so a new file is a new reference test by construction.
 *  2. This check, which fails when an op grows a variant and no test in that op
 *     iterates them — the case rule 1 cannot cover, because a test that never
 *     calls `eachVariant` has nothing to build a loop from.
 *
 * It reads the test sources rather than observing a run. That is deliberate:
 * `npm test` gives each file its own process (see `scripts/test.mjs`), so there
 * is no point at which one process could see which variants the others touched.
 *
 * Import it directly, from the one test that runs it. It is deliberately absent
 * from `harness/index.ts`, which is the package's public entry and what every
 * op's test imports: this is a check on this repository's own tree, not
 * something a consumer of the package calls, and there is no reason for it to be
 * on the import graph of every GPU test.
 */
const LOOP = "eachVariant";

function toPath(dir: string | URL): string {
  return typeof dir === "string" ? dir : fileURLToPath(dir);
}

/** Ops carrying a target- or dtype-specific variant that no test iterates. Sorted. */
export function unguardedOps(opsRoot: string | URL): string[] {
  const root = toPath(opsRoot);
  const offenders: string[] = [];
  for (const op of readdirSync(root)) {
    const dir = join(root, op);
    if (!statSync(dir).isDirectory()) continue;
    const variants = variantsIn(join(dir, "wgsl"));
    if (!variants.some((v) => v.target !== null || v.dtype !== null)) continue;
    const tests = readdirSync(dir).filter((file) => file.endsWith(".test.ts"));
    const looped = tests.some((file) => readFileSync(join(dir, file), "utf8").includes(LOOP));
    if (!looped) offenders.push(op);
  }
  return offenders.sort();
}
