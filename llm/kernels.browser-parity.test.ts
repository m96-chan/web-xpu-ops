/**
 * `LLM_KERNEL_SOURCES` is what the engines will ask `registerKernelSources`
 * for; `examples/llm-demo/src/browser-runtime.ts#WGSL_TABLE` is what that demo
 * bundles and registers. The two are one set or the page fails at its first
 * dispatch with a message about a table entry — after a 1.4 GiB download.
 *
 * Before issue #224 there were three copies (two `CODE` objects and the
 * table) and this test parsed all three as text. Now the list is a value the
 * engines build their table from, so the Node side is imported, not parsed;
 * the browser side is still read as text because its `.wgsl` imports need a
 * bundler to resolve.
 */
import { readFileSync } from "node:fs";
import { afterEach, describe, expect, it } from "vitest";
import { registerKernelSources, resetKernelSources } from "../harness/api.js";
import { LLM_KERNEL_SOURCES, llmKernels, resetLlmKernels } from "./kernels.js";

interface OpTable {
  [op: string]: Set<string>;
}

function opsFromList(): OpTable {
  const table: OpTable = {};
  for (const { op, entry } of LLM_KERNEL_SOURCES) (table[op] ??= new Set()).add(entry);
  return table;
}

function opsFromBrowserRuntime(source: string): OpTable {
  const start = source.indexOf("export const WGSL_TABLE");
  if (start < 0) throw new Error("kernels.browser-parity: WGSL_TABLE not found in browser-runtime.ts");
  const braceStart = source.indexOf("{", start);
  const braceEnd = source.indexOf("\n};", braceStart);
  const body = source.slice(braceStart, braceEnd);

  const table: OpTable = {};
  const opPattern = /(\w+):\s*\{([^}]*)\}/g;
  for (const opMatch of body.matchAll(opPattern)) {
    const entries = new Set<string>();
    for (const entryMatch of opMatch[2]!.matchAll(/(\w+):\s*\w+/g)) entries.add(entryMatch[1]!);
    table[opMatch[1]!] = entries;
  }
  return table;
}

const sorted = (table: OpTable): Record<string, string[]> =>
  Object.fromEntries(Object.keys(table).sort().map((op) => [op, [...table[op]!].sort()]));

describe("LLM_KERNEL_SOURCES <-> examples/llm-demo browser-runtime WGSL_TABLE", () => {
  afterEach(() => {
    resetKernelSources();
    resetLlmKernels();
  });

  it("names a file that exists for every entry", () => {
    for (const { op, entry } of LLM_KERNEL_SOURCES) {
      const path = new URL(`../ops/${op}/wgsl/${entry}.wgsl`, import.meta.url);
      expect(() => readFileSync(path, "utf8"), `${op}/${entry}`).not.toThrow();
    }
  });

  it("has distinct keys", () => {
    const keys = LLM_KERNEL_SOURCES.map((e) => e.key);
    expect(new Set(keys).size).toBe(keys.length);
  });

  it("asks the registry for exactly the list, and nothing before it is asked", () => {
    const asked: string[] = [];
    registerKernelSources((op, entry) => { asked.push(`${op}/${entry}`); return `// ${op}/${entry}`; });
    expect(asked).toEqual([]);
    const table = llmKernels();
    expect(asked.sort()).toEqual(LLM_KERNEL_SOURCES.map((e) => `${e.op}/${e.entry}`).sort());
    expect(table.matvecQ8Ffn).toBe("// matvec/q8_ffn");
  });

  it("is the demo's WGSL_TABLE, op for op and entry for entry", () => {
    const browserSource = readFileSync(new URL("../examples/llm-demo/src/browser-runtime.ts", import.meta.url), "utf8");
    expect(sorted(opsFromBrowserRuntime(browserSource))).toEqual(sorted(opsFromList()));
  });
});
