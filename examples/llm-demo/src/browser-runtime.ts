/**
 * The WGSL this demo bundles, as the `{ [op]: { [entry]: source } }` table
 * `harness/api.ts#registerKernelSources` takes — one static import per
 * `.wgsl` file so esbuild's `text` loader (see `build.mjs`) can inline each
 * as a string. `main.ts` registers it before the first dispatch.
 *
 * `llm/kernels.browser-parity.test.ts` holds this table to
 * `llm/kernels.ts#LLM_KERNEL_SOURCES`, op for op and entry for entry — the
 * list the engines will actually ask for. An entry missing here fails at the
 * page's first dispatch, after the weights downloaded, which is why the test
 * exists.
 *
 * `createBrowserRunner` — the per-dispatch `Runner` over `navigator.gpu` —
 * lives in `examples/web-common/src/browser-runner.ts` since issue #224 and is
 * re-exported below for this demo's `main.ts`.
 */
import gatherKernel from "../../../ops/gather/wgsl/kernel.wgsl";
import rmsnormKernel from "../../../ops/rmsnorm/wgsl/kernel.wgsl";
import matvecKernel from "../../../ops/matvec/wgsl/kernel.wgsl";
import matvecQ8Kernel from "../../../ops/matvec/wgsl/q8.wgsl";
// Issue #111: `llm/engine-q8-resident.ts`'s two decode-only fused entry
// points — not referenced by `llm/kernels.ts#CODE` (that file's `opKernel`
// calls are what `kernels.browser-parity.test.ts` checks against this
// table), but still resolved through this same `harness/index.js` ->
// `browser-runtime.ts` redirect (`build.mjs`'s `harnessBrowserShim` has no
// way to tell which importer is asking), so they need an entry here too —
// see that test file's own doc for how it now covers both `kernels.ts` and
// `engine-q8-resident.ts`.
import matvecQ8FfnKernel from "../../../ops/matvec/wgsl/q8_ffn.wgsl";
import matvecQ8ResidualKernel from "../../../ops/matvec/wgsl/q8_residual.wgsl";
import matmulKernel from "../../../ops/matmul/wgsl/kernel.wgsl";
// Issue #128: `LlamaEngineQ8Resident.runPrefillResident`'s prefill weight
// path — reads the packed int8 weight directly, in-kernel, replacing the
// dequant_transpose+matmul pair below for that one call site (`kernels.ts`
// still asks for both, for parity/integration-test purposes — see those
// imports' own doc a few lines down).
import matmulQ8Kernel from "../../../ops/matmul/wgsl/q8.wgsl";
import ropeKernel from "../../../ops/rope/wgsl/kernel.wgsl";
import gqaScoresKernel from "../../../ops/gqa/wgsl/scores.wgsl";
import gqaContextKernel from "../../../ops/gqa/wgsl/context.wgsl";
import activationKernel from "../../../ops/activation/wgsl/kernel.wgsl";
import elementwiseKernel from "../../../ops/elementwise/wgsl/kernel.wgsl";
// `ops/permute` (issue #117's resident prefill reshape, `ops/permute/reference.ts`'s
// doc has the full story) — `llm/engine-q8-resident.ts` is its only caller
// today, but it is a `kernels.ts#CODE` entry too (kept, per that file's own
// "one function per kernel entry point" scope, even though nothing in
// `LlamaEngineQ8`'s own forward pass calls `runPermute` yet), so it stays
// covered by this table's usual parity check.
import permuteKernel from "../../../ops/permute/wgsl/kernel.wgsl";
// `ops/dequant_transpose` (`ops/dequant_transpose/reference.ts`'s doc has
// the full story) — issue #128 removed its one production caller
// (`LlamaEngineQ8Resident.runPrefillResident`, replaced by `matmulQ8` above),
// but `kernels.ts#CODE` still asks for it (its own `runDequantTranspose`
// wrapper is this op's Node-side integration test path — that file's own
// doc), so it stays in this table too.
import dequantTransposeKernel from "../../../ops/dequant_transpose/wgsl/kernel.wgsl";

export const WGSL_TABLE: Readonly<Record<string, Readonly<Record<string, string>>>> = {
  gather: { kernel: gatherKernel },
  rmsnorm: { kernel: rmsnormKernel },
  matvec: { kernel: matvecKernel, q8: matvecQ8Kernel, q8_ffn: matvecQ8FfnKernel, q8_residual: matvecQ8ResidualKernel },
  matmul: { kernel: matmulKernel, q8: matmulQ8Kernel },
  rope: { kernel: ropeKernel },
  permute: { kernel: permuteKernel },
  dequant_transpose: { kernel: dequantTransposeKernel },
  gqa: { scores: gqaScoresKernel, context: gqaContextKernel },
  activation: { kernel: activationKernel },
  elementwise: { kernel: elementwiseKernel },
};

export { createBrowserRunner, type BrowserRunner, type RunnerStats } from "../../web-common/src/browser-runner.js";
