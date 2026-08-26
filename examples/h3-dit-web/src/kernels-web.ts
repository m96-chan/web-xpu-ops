/**
 * The DiT's kernels, inlined at bundle time.
 *
 * The Node half is `examples/h3-dit/src/kernels-node.ts`, which reads the same
 * files off disk. Both satisfy `DitKernels`, so a kernel added on one side and
 * forgotten on the other is a type error rather than a missing dispatch — which
 * is exactly how `examples/h3-video-web` shipped without `matmulQ8` and would
 * have dispatched `undefined` at its first matmul. That page was not on
 * `npm run lint`'s list; every example project's tsconfig is now.
 */
import type { DitKernels } from "../../h3-dit/src/model-gpu.js";
import type { VideoKernels } from "../../h3-video/src/decoder-gpu.js";
import matmulKernel from "../../../ops/matmul/wgsl/kernel.wgsl";
import matmulQ8Kernel from "../../../ops/matmul/wgsl/q8.wgsl";
import rmsnormKernel from "../../../ops/rmsnorm/wgsl/kernel.wgsl";
import layernormKernel from "../../../ops/layernorm/wgsl/kernel.wgsl";
import activationKernel from "../../../ops/activation/wgsl/kernel.wgsl";
import elementwiseKernel from "../../../ops/elementwise/wgsl/kernel.wgsl";
import rowsKernel from "../../../ops/elementwise/wgsl/rows.wgsl";
import ropeAxesKernel from "../../../ops/rope/wgsl/axes.wgsl";
import permuteKernel from "../../../ops/permute/wgsl/kernel.wgsl";
import { FLASH_GENERATION } from "../../../ops/flash_attention/index.js";
import fa2Kernel from "../../../ops/flash_attention/wgsl/fa2.wgsl";
import fa3Kernel from "../../../ops/flash_attention/wgsl/fa3.wgsl";

const flashAttention = FLASH_GENERATION === "fa3" ? fa3Kernel : fa2Kernel;

export const ditKernels: DitKernels = {
  matmul: matmulKernel,
  matmulQ8: matmulQ8Kernel,
  rmsnorm: rmsnormKernel,
  activation: activationKernel,
  elementwise: elementwiseKernel,
  rows: rowsKernel,
  ropeAxes: ropeAxesKernel,
  permute: permuteKernel,
  flashAttention,
};

/** The VAE decoder's set — one more entry (`layernorm`) than the DiT needs. */
export const videoKernels: VideoKernels = {
  ...ditKernels,
  layernorm: layernormKernel,
};
