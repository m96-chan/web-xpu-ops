/**
 * Node's answer to "where does WGSL come from": the file beside the op.
 *
 * Importing this module registers a reader with `harness/api.ts`'s registry,
 * and that side effect is the whole point of the file. It is a separate
 * module — not inlined in `harness/index.ts` — because the barrel is not the
 * only door into Node code that dispatches: a test with a mocked `Runner`
 * imports `llm/kernels.ts` and nothing from the harness at all, and CI runs
 * every test file in its own process, so "something else in this process
 * registered it" is not a state such a test can rely on. `vitest.config.ts`
 * loads this file before every test; `wgsl.ts` and `resident.ts` import it
 * so a script that creates a device has it too.
 *
 * A browser bundle must never reach this file: it imports `node:fs`.
 * Issue #224.
 */
import { readFileSync } from "node:fs";
import { registerKernelSources, type KernelResolver } from "./api.js";

/** `ops/<op>/wgsl/<entry>.wgsl`, read on first use; `undefined` when there is no such file, so `opKernel` names what is missing. */
export const nodeKernelSources: KernelResolver = (op, entry) => {
  try {
    return readFileSync(new URL(`../ops/${op}/wgsl/${entry}.wgsl`, import.meta.url), "utf8");
  } catch {
    return undefined;
  }
};

registerKernelSources(nodeKernelSources);
