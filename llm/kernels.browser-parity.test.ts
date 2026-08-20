import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";

/**
 * `examples/llm-demo/src/browser-runtime.ts#WGSL_TABLE` is a hand-written
 * mirror of this file's own `CODE` object (see `kernels.ts`'s module doc and
 * `browser-runtime.ts`'s): both must name the exact same ops and, per op,
 * the exact same entry points, or the browser demo silently falls back to
 * throwing "no bundled WGSL for op ..." partway through a forward pass
 * instead of failing at build time.
 *
 * This reads both files as **text** rather than importing either: `kernels.ts`
 * pulls in `harness/index.ts` (Node/Dawn-only, fine here since this *is* a
 * Node test) and `browser-runtime.ts` statically imports ten `.wgsl` files
 * that only esbuild's `text` loader (`examples/llm-demo/build.mjs`) knows
 * how to resolve — vitest has no such loader configured (nor should it: this
 * repo's own suite never needs one), so importing that module here would
 * fail for a reason that has nothing to do with what this test checks. A
 * light regex parse of each file's own object-literal keys is what stays
 * decoupled from both those concerns while still catching real drift.
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

describe("llm/kernels.ts CODE <-> examples/llm-demo browser-runtime WGSL_TABLE", () => {
  const kernelsSource = readFileSync(new URL("./kernels.ts", import.meta.url), "utf8");
  const browserSource = readFileSync(
    new URL("../examples/llm-demo/src/browser-runtime.ts", import.meta.url),
    "utf8",
  );

  const nodeSide = opsFromKernelsTs(kernelsSource);
  const browserSide = opsFromBrowserRuntime(browserSource);

  it("parsed at least the ten known kernel entry points from kernels.ts (sanity check on the regex itself)", () => {
    const totalEntries = Object.values(nodeSide).reduce((sum, set) => sum + set.size, 0);
    expect(totalEntries).toBe(10);
  });

  it("names the same ops on both sides", () => {
    expect(Object.keys(browserSide).sort()).toEqual(Object.keys(nodeSide).sort());
  });

  it("names the same entry points per op on both sides", () => {
    for (const op of Object.keys(nodeSide)) {
      expect([...browserSide[op]!].sort()).toEqual([...nodeSide[op]!].sort());
    }
  });
});
