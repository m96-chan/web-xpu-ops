import { readFileSync } from "node:fs";
import { registerKernelSources } from "./api.js";

export { createRunner, params, type Binding, type Dispatch, type Runner } from "./wgsl.js";
export { opKernel, kernelFromUrl, registerKernelSources, resetKernelSources, type KernelResolver, type KernelSources } from "./api.js";
export { agree, type Tolerance } from "./agree.js";
export { useGpu, gpuTest, useResidentGpu, residentTest, kernel, expectAgrees } from "./suite.js";
export {
  createResidentDevice,
  runnerFromResident,
  type BatchProfile,
  type BatchProfileSink,
  type ResidentDevice,
  type ResidentOp,
  type ResidentReadback,
} from "./resident.js";

// Node's answer to "where does WGSL come from": the file beside the op, read
// on first use. Registered here — on import of the harness barrel every test
// and verify script already reaches for — so nothing in `llm/` has to know
// whether it is running under vitest or in a page. A browser bundle never
// imports this file (it would drag in `webgpu`); it registers its own table.
// Issue #224.
registerKernelSources((op, entry) => {
  try {
    return readFileSync(new URL(`../ops/${op}/wgsl/${entry}.wgsl`, import.meta.url), "utf8");
  } catch {
    return undefined;
  }
});
