import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";

/**
 * `examples/llm-demo/src/browser-runtime.ts#WGSL_TABLE` is a hand-written
 * mirror of the **union** of two `CODE` objects — `kernels.ts`'s own (issue
 * #106) and `engine-q8-resident.ts`'s (issue #111 added two decode-only
 * entries there that `kernels.ts` never references) — because both files
 * import `kernel`/`params` from `harness/index.js`, and `build.mjs`'s
 * `harnessBrowserShim` redirects *that specifier* to `browser-runtime.ts`
 * regardless of which file is asking (see `browser-runtime.ts`'s own module
 * doc). WGSL_TABLE must name, per op, the exact union of entry points both
 * files ask for — short by one and the browser demo throws "no bundled WGSL
 * entry ..." partway through a forward pass instead of failing at build
 * time; long by one and nothing catches it locally (an unused import is not
 * a `tsc` error), so this test's own "no more than the union" check below is
 * the only thing that would.
 *
 * This reads all three files as **text** rather than importing them:
 * `kernels.ts`/`engine-q8-resident.ts` pull in `harness/index.ts` (Node/Dawn-only,
 * fine here since this *is* a Node test) and `browser-runtime.ts` statically
 * imports a dozen-plus `.wgsl` files that only esbuild's `text` loader
 * (`examples/llm-demo/build.mjs`) knows how to resolve — vitest has no such
 * loader configured (nor should it: this repo's own suite never needs one),
 * so importing that module here would fail for a reason that has nothing to
 * do with what this test checks. A light regex parse of each file's own
 * object-literal keys is what stays decoupled from both those concerns while
 * still catching real drift.
 */

interface OpTable {
  [op: string]: Set<string>;
}

/** `opKernel("gather")` / `opKernel("matvec", "q8")` calls in `kernels.ts`'s `CODE` object. */
function opsFromKernelsTs(source: string): OpTable {
  const table: OpTable = {};
  const callPattern = /opKernel\(\s*"([^"]+)"\s*(?:,\s*"([^"]+)")?\s*\)/g;
  for (const match of source.matchAll(callPattern)) {
    const op = match[1]!;
    const entry = match[2] ?? "kernel";
    table[op] ??= new Set();
    table[op]!.add(entry);
  }
  return table;
}

/**
 * `xyz: kernel(new URL("../ops/<op>/index.ts", import.meta.url)[, "entry"])`
 * calls in `engine-q8-resident.ts`'s `CODE` object — a different call shape
 * from `kernels.ts`'s `opKernel(op, entry)` helper (that file has no such
 * helper of its own, `CODE`'s values call `harness/suite.ts#kernel` directly
 * with a `new URL(...)`, see that file's own `CODE` object), so it needs its
 * own regex rather than reusing `opsFromKernelsTs`'s.
 */
function opsFromEngineResidentTs(source: string): OpTable {
  const table: OpTable = {};
  const callPattern = /kernel\(\s*new URL\(\s*"\.\.\/ops\/([^/]+)\/index\.ts"[^)]*\)\s*(?:,\s*"([^"]+)")?\s*\)/g;
  for (const match of source.matchAll(callPattern)) {
    const op = match[1]!;
    const entry = match[2] ?? "kernel";
    table[op] ??= new Set();
    table[op]!.add(entry);
  }
  return table;
}

/** `a`'s entries per op, unioned with `b`'s — every op/entry pair either side asks `harness/index.js` (browser: `browser-runtime.ts`) for. */
function unionTables(a: OpTable, b: OpTable): OpTable {
  const table: OpTable = {};
  for (const op of new Set([...Object.keys(a), ...Object.keys(b)])) {
    table[op] = new Set([...(a[op] ?? []), ...(b[op] ?? [])]);
  }
  return table;
}

/** `WGSL_TABLE`'s `{ op: { entry: someImportedIdentifier, ... }, ... }` shape in `browser-runtime.ts`. */
function opsFromBrowserRuntime(source: string): OpTable {
  const start = source.indexOf("export const WGSL_TABLE");
  if (start < 0) throw new Error("kernels.browser-parity: WGSL_TABLE not found in browser-runtime.ts");
  const braceStart = source.indexOf("{", start);
  const braceEnd = source.indexOf("\n};", braceStart);
  const body = source.slice(braceStart, braceEnd);

  const table: OpTable = {};
  const opPattern = /(\w+):\s*\{([^}]*)\}/g;
  for (const opMatch of body.matchAll(opPattern)) {
    const op = opMatch[1]!;
    const entries = new Set<string>();
    const entryPattern = /(\w+):\s*\w+/g;
    for (const entryMatch of opMatch[2]!.matchAll(entryPattern)) {
      entries.add(entryMatch[1]!);
    }
    table[op] = entries;
  }
  return table;
}

describe("llm/kernels.ts + llm/engine-q8-resident.ts CODE <-> examples/llm-demo browser-runtime WGSL_TABLE", () => {
  const kernelsSource = readFileSync(new URL("./kernels.ts", import.meta.url), "utf8");
  const engineResidentSource = readFileSync(new URL("./engine-q8-resident.ts", import.meta.url), "utf8");
  const browserSource = readFileSync(
    new URL("../examples/llm-demo/src/browser-runtime.ts", import.meta.url),
    "utf8",
  );

  const kernelsSide = opsFromKernelsTs(kernelsSource);
  const engineResidentSide = opsFromEngineResidentTs(engineResidentSource);
  // What `browser-runtime.ts`'s WGSL_TABLE has to cover — every op/entry
  // either `CODE` object asks `harness/index.js` for (this file's own doc
  // explains why it is a union, not either side alone).
  const nodeSide = unionTables(kernelsSide, engineResidentSide);
  const browserSide = opsFromBrowserRuntime(browserSource);

  it("parsed at least the twelve known kernels.ts entry points, and engine-q8-resident.ts's own ten plus two fused ones (sanity check on the regexes themselves)", () => {
    // kernels.ts: 10 at issue #106; 11 then 12 since issue #117 added
    // `permute` (`ops/permute`) and `dequantTranspose` (`ops/dequant_transpose`).
    const kernelsTotal = Object.values(kernelsSide).reduce((sum, set) => sum + set.size, 0);
    expect(kernelsTotal).toBe(12);
    // engine-q8-resident.ts: the same ten kernels.ts already covers
    // (rmsnorm, matvecQ8, matmul, rope, gqaScores, gqaContext, activation,
    // elementwise, permute, dequantTranspose) plus issue #111's
    // `matvecQ8Ffn`/`matvecQ8Residual`.
    const engineResidentTotal = Object.values(engineResidentSide).reduce((sum, set) => sum + set.size, 0);
    expect(engineResidentTotal).toBe(12);
  });

  it("names the same ops on both sides", () => {
    expect(Object.keys(browserSide).sort()).toEqual(Object.keys(nodeSide).sort());
  });

  it("names the same entry points per op on both sides", () => {
    // PR #127 review, item 9: this loop's own body is the only place that
    // actually checks per-op entry agreement — `for (const op of
    // Object.keys(nodeSide))` over an *empty* `nodeSide` (both source
    // regexes silently matching nothing) would run zero iterations and this
    // test would report PASS having asserted nothing, the exact
    // zero-assertion shape `harness/suite.ts#skipUnlessPresent`'s own doc
    // already treats as a bug elsewhere in this repo. The "parsed at least
    // twelve" test above would likely also fail in that scenario, but this
    // test should not depend on a sibling test to have already caught it —
    // asserted here, directly, so this test cannot itself pass fail-open.
    expect(Object.keys(nodeSide).length).toBeGreaterThanOrEqual(10);
    for (const op of Object.keys(nodeSide)) {
      expect([...browserSide[op]!].sort()).toEqual([...nodeSide[op]!].sort());
    }
  });
});
