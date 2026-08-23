/**
 * The DiT's kernels, inlined at bundle time.
 *
 * The browser half of `DitKernels`. esbuild's `text` loader turns each
 * `import … from "*.wgsl"` into a string literal, so this module needs no
 * filesystem and no fetch — the same arrangement `examples/llm-demo` uses.
 *
 * `kernels-node.ts` is the other half. Both build the same object from
 * `DIT_KERNEL_SOURCES`' names, so a kernel added on one side and forgotten on
 * the other is a type error rather than a missing dispatch at run time.
 */
import type { DitKernels } from "../../zimage/src/dit-gpu.js";
import rmsnormKernel from "../../../ops/rmsnorm/wgsl/kernel.wgsl";
import layernormKernel from "../../../ops/layernorm/wgsl/kernel.wgsl";
import matmulKernel from "../../../ops/matmul/wgsl/kernel.wgsl";
import activationKernel from "../../../ops/activation/wgsl/kernel.wgsl";
import elementwiseKernel from "../../../ops/elementwise/wgsl/kernel.wgsl";
import rowsKernel from "../../../ops/elementwise/wgsl/rows.wgsl";
import ropeAxesKernel from "../../../ops/rope/wgsl/axes.wgsl";
import dequantTransposeKernel from "../../../ops/dequant_transpose/wgsl/kernel.wgsl";
import scoresKernel from "../../../ops/attention/wgsl/scores.wgsl";
import contextKernel from "../../../ops/attention/wgsl/context.wgsl";

export const ditKernels: DitKernels = {
  rmsnorm: rmsnormKernel,
  layernorm: layernormKernel,
  matmul: matmulKernel,
  activation: activationKernel,
  elementwise: elementwiseKernel,
  rows: rowsKernel,
  ropeAxes: ropeAxesKernel,
  dequantTranspose: dequantTransposeKernel,
  scores: scoresKernel,
  context: contextKernel,
};
