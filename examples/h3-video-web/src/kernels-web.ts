/**
 * The decoder's kernels, inlined at bundle time.
 *
 * The Node half is `examples/h3-video/src/kernels-node.ts`, which reads the same
 * files off disk. Both satisfy `VideoKernels`, so a kernel added on one side and
 * forgotten on the other is a type error rather than a missing dispatch.
 */
import type { VideoKernels } from "../../h3-video/src/decoder-gpu.js";
import matmulKernel from "../../../ops/matmul/wgsl/kernel.wgsl";
import rmsnormKernel from "../../../ops/rmsnorm/wgsl/kernel.wgsl";
import layernormKernel from "../../../ops/layernorm/wgsl/kernel.wgsl";
import activationKernel from "../../../ops/activation/wgsl/kernel.wgsl";
import elementwiseKernel from "../../../ops/elementwise/wgsl/kernel.wgsl";
import rowsKernel from "../../../ops/elementwise/wgsl/rows.wgsl";
import ropeAxesKernel from "../../../ops/rope/wgsl/axes.wgsl";
import permuteKernel from "../../../ops/permute/wgsl/kernel.wgsl";
// Both generations ship; `FLASH_GENERATION` says which is dispatched, and the
// bundler needs each import to be static.
import { FLASH_GENERATION } from "../../../ops/flash_attention/index.js";
import fa2Kernel from "../../../ops/flash_attention/wgsl/fa2.wgsl";
import fa3Kernel from "../../../ops/flash_attention/wgsl/fa3.wgsl";

export const videoKernels: VideoKernels = {
  matmul: matmulKernel,
  rmsnorm: rmsnormKernel,
  layernorm: layernormKernel,
  activation: activationKernel,
  elementwise: elementwiseKernel,
  rows: rowsKernel,
  ropeAxes: ropeAxesKernel,
  permute: permuteKernel,
  flashAttention: FLASH_GENERATION === "fa3" ? fa3Kernel : fa2Kernel,
};
