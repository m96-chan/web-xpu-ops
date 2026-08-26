/**
 * The DiT's kernels, read off the filesystem.
 *
 * The Node half. A browser inlines the same files at bundle time; both go
 * through `DIT_KERNEL_SOURCES` so the two cannot load different sets.
 */
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { DIT_KERNEL_SOURCES, type DitKernels } from "./model-gpu.js";

const opsRoot = new URL("../../../ops/", import.meta.url);

export function ditKernels(): DitKernels {
  const out = {} as DitKernels;
  for (const { key, op, entry } of DIT_KERNEL_SOURCES) {
    out[key] = readFileSync(fileURLToPath(new URL(`${op}/wgsl/${entry}.wgsl`, opsRoot)), "utf8");
  }
  return out;
}
