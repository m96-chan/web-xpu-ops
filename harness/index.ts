// Registers Node's WGSL file reader on import — see that file for why it is a
// module of its own and not a line here.
import "./kernel-sources-node.js";

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
