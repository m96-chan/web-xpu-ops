/**
 * The decoder's kernels, inlined at bundle time.
 *
 * esbuild's `text` loader turns each `import … from "*.wgsl"` into a string
 * literal, so this module needs no filesystem and no fetch — the arrangement
 * `examples/llm-demo` established and every demo since follows.
 *
 * The Node half is `examples/h3-audio/src/kernels-node.ts`, which reads the same
 * five files off disk. Both satisfy `AudioKernels`, so a kernel added on one
 * side and forgotten on the other is a type error rather than a missing
 * dispatch at run time.
 */
import type { AudioKernels } from "../../h3-audio/src/decoder-gpu.js";
import conv1dKernel from "../../../ops/conv/wgsl/kernel.wgsl";
import convTranspose1dKernel from "../../../ops/conv_transpose/wgsl/kernel.wgsl";
import padKernel from "../../../ops/pad/wgsl/kernel.wgsl";
import snakeBetaKernel from "../../../ops/snake/wgsl/beta.wgsl";
import axpyInplaceKernel from "../../../ops/axpy/wgsl/inplace.wgsl";
import axpyKernel from "../../../ops/axpy/wgsl/kernel.wgsl";

export const audioKernels: AudioKernels = {
  conv1d: conv1dKernel,
  convTranspose1d: convTranspose1dKernel,
  pad: padKernel,
  snakeBeta: snakeBetaKernel,
  axpy: axpyInplaceKernel,
  axpyOut: axpyKernel,
};
