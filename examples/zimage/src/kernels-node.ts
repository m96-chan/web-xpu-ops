/**
 * The DiT's kernels, read off the filesystem.
 *
 * The Node half of `DitKernels`. The browser half fetches the same files over
 * HTTP — see `web/kernels-web.ts`. Both go through `DIT_KERNEL_SOURCES` so the
 * two cannot end up loading different sets, which is the failure a second copy
 * of the list would eventually produce.
 */
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { DIT_KERNEL_SOURCES, type DitKernels } from "./dit-gpu.js";

const opsRoot = new URL("../../../ops/", import.meta.url);

export function ditKernels(): DitKernels {
  const out = {} as DitKernels;
  for (const { key, op, entry } of DIT_KERNEL_SOURCES) {
    out[key] = readFileSync(fileURLToPath(new URL(`${op}/wgsl/${entry}.wgsl`, opsRoot)), "utf8");
  }
  return out;
}
