/** Reads the VAE decoder's kernels off disk, for Node. */
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { type VaeKernels, VAE_KERNEL_SOURCES } from "./vae-gpu.js";

const opsRoot = new URL("../../../ops/", import.meta.url);

export function vaeKernels(): VaeKernels {
  const out = {} as VaeKernels;
  for (const { key, op, entry } of VAE_KERNEL_SOURCES) {
    out[key] = readFileSync(fileURLToPath(new URL(`${op}/wgsl/${entry}.wgsl`, opsRoot)), "utf8");
  }
  return out;
}
