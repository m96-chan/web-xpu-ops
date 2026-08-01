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
 * Import it directly, from the one test that runs it. Nothing in this file, in
 * `resolve.ts` or in `variants.ts` is re-exported from `harness/index.ts`, and
 * that is deliberate: `index.ts` is what every op's test imports, so anything
 * re-exported there lands on the import graph of every GPU test. Keeping
 * resolution off it leaves `index.ts` and `suite.ts` byte-identical to what they
 * were before this feature existed — adding it changed nothing about how an
 * existing op's test is loaded, which is the cheapest way to be sure it changed
 * nothing about how one runs.
 */
const LOOP = "eachVariant";

function toPath(dir: string | URL): string {
  return typeof dir === "string" ? dir : fileURLToPath(dir);
}

/**
 * Ops with a target- or dtype-specific variant on an entry point no test loops.
 * Sorted.
 *
 * The check is per entry point, not per op. An op with two of them — `stft`,
 * `attention` — can loop one and leave the other's variants uncompiled, and an
 * op-level check would call that covered. So each entry point that has variants
 * has to be named, which is why `eachVariant` takes the name as a string rather
 * than defaulting to `kernel`.
 */
export function unguardedOps(opsRoot: string | URL): string[] {
  const root = toPath(opsRoot);
  const offenders: string[] = [];
  for (const op of readdirSync(root)) {
    const dir = join(root, op);
    if (!statSync(dir).isDirectory()) continue;

    // Only entry points that actually have a variant can hide one.
    const needing = new Set(
      variantsIn(join(dir, "wgsl"))
        .filter((v) => v.target !== null || v.dtype !== null)
        .map((v) => v.entry),
    );
    if (needing.size === 0) continue;

    const sources = readdirSync(dir)
      .filter((file) => file.endsWith(".test.ts"))
      .map((file) => readFileSync(join(dir, file), "utf8"));
    const covered = (entry: string) =>
      sources.some((src) => src.includes(LOOP) && src.includes(JSON.stringify(entry)));
    if (![...needing].every(covered)) offenders.push(op);
  }
  return offenders.sort();
}
