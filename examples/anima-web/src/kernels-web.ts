/**
 * Every kernel table this demo needs, inlined at bundle time.
 *
 * esbuild's `text` loader turns each `import … from "*.wgsl"` into a string
 * literal, so this module needs no filesystem and no fetch — the arrangement
 * `examples/llm-demo` established and `examples/zimage-web` follows.
 *
 * The Node halves are `examples/zimage/src/kernels-node.ts` and
 * `examples/anima/src/vae-kernels-node.ts`. Both sides satisfy the same
 * interfaces, so a kernel added on one and forgotten on the other is a type
 * error rather than a missing dispatch at run time.
 */
import type { DitKernels } from "../../zimage/src/dit-gpu.js";
import type { EncoderKernels } from "../../zimage/src/text-encoder-gpu.js";
import type { VaeKernels } from "../../anima/src/vae-gpu.js";
import conv2dKernel from "../../../ops/conv/wgsl/conv2d.wgsl";
import upsampleKernel from "../../../ops/upsample/wgsl/kernel.wgsl";
import rmsnormKernel from "../../../ops/rmsnorm/wgsl/kernel.wgsl";
import layernormKernel from "../../../ops/layernorm/wgsl/kernel.wgsl";
import transposeKernel from "../../../ops/transpose/wgsl/kernel.wgsl";
import matmulKernel from "../../../ops/matmul/wgsl/kernel.wgsl";
import matmulQ8Kernel from "../../../ops/matmul/wgsl/q8.wgsl";
import activationKernel from "../../../ops/activation/wgsl/kernel.wgsl";
import elementwiseKernel from "../../../ops/elementwise/wgsl/kernel.wgsl";
import rowsKernel from "../../../ops/elementwise/wgsl/rows.wgsl";
import ropeKernel from "../../../ops/rope/wgsl/kernel.wgsl";
import ropeAxesKernel from "../../../ops/rope/wgsl/axes.wgsl";
import permuteKernel from "../../../ops/permute/wgsl/kernel.wgsl";
import scoresKernel from "../../../ops/attention/wgsl/scores.wgsl";
import contextKernel from "../../../ops/attention/wgsl/context.wgsl";
import gqaScoresKernel from "../../../ops/gqa/wgsl/scores.wgsl";
import gqaContextKernel from "../../../ops/gqa/wgsl/context.wgsl";

export const ditKernels: DitKernels = {
  rmsnorm: rmsnormKernel,
  layernorm: layernormKernel,
  matmul: matmulKernel,
  activation: activationKernel,
  elementwise: elementwiseKernel,
  rows: rowsKernel,
  ropeAxes: ropeAxesKernel,
  matmulQ8: matmulQ8Kernel,
  permute: permuteKernel,
  scores: scoresKernel,
  context: contextKernel,
};

export const encoderKernels: EncoderKernels = {
  rmsnorm: rmsnormKernel,
  matmul: matmulKernel,
  activation: activationKernel,
  elementwise: elementwiseKernel,
  rope: ropeKernel,
  gqaScores: gqaScoresKernel,
  gqaContext: gqaContextKernel,
};

export const vaeKernels: VaeKernels = {
  conv2d: conv2dKernel,
  upsample: upsampleKernel,
  rmsnorm: rmsnormKernel,
  transpose: transposeKernel,
  activation: activationKernel,
  elementwise: elementwiseKernel,
  scores: scoresKernel,
  context: contextKernel,
};
