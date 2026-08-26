/**
 * The encoder's kernels, read off the filesystem.
 *
 * The Node half, the same arrangement `examples/h3-ref2v/src/kernels-node.ts`
 * and `examples/h3-video/src/kernels-node.ts` use: a browser inlines the same
 * files at bundle time, and both sides go through `ENCODER_KERNEL_SOURCES`, so
 * a kernel added on one side and forgotten on the other is a type error rather
 * than a missing dispatch.
 */
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { ENCODER_KERNEL_SOURCES, type EncoderKernels } from "./encoder-gpu.js";

const opsRoot = new URL("../../../ops/", import.meta.url);

export function encoderKernels(): EncoderKernels {
  const out = {} as EncoderKernels;
  for (const { key, op, entry } of ENCODER_KERNEL_SOURCES) {
    out[key] = readFileSync(fileURLToPath(new URL(`${op}/wgsl/${entry}.wgsl`, opsRoot)), "utf8");
  }
  return out;
}
