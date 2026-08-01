export { createRunner, params, type Binding, type Dispatch, type Runner } from "./wgsl.js";
export { agree, type Tolerance } from "./agree.js";
export { useGpu, gpuTest, kernel, eachVariant, expectAgrees } from "./suite.js";
export {
  DTYPES,
  TARGETS,
  describeAdapter,
  detectTarget,
  type AdapterInfoLike,
  type AdapterLike,
  type AdapterReport,
  type Detected,
  type Dtype,
  type Target,
} from "./target.js";
export {
  DEFAULT_ENTRY,
  parseVariant,
  resolve,
  variantSuites,
  variantsIn,
  type Choice,
  type Resolution,
  type Rung,
  type Variant,
} from "./resolve.js";
// `unguardedOps` is deliberately not re-exported here. It is a check on this
// repository's own tree rather than something a consumer of the package calls,
// and `harness/index.ts` is what every op's test imports — see the note in
// harness/coverage.ts about keeping that module off their import graph.
