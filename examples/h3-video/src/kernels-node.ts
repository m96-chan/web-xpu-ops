/**
 * The decoder's kernels, read off the filesystem.
 *
 * The Node half. A browser inlines the same files at bundle time; both go
 * through `VIDEO_KERNEL_SOURCES` so the two cannot load different sets.
 */
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { VIDEO_KERNEL_SOURCES, type VideoKernels } from "./decoder-gpu.js";

const opsRoot = new URL("../../../ops/", import.meta.url);

export function videoKernels(): VideoKernels {
  const out = {} as VideoKernels;
  for (const { key, op, entry } of VIDEO_KERNEL_SOURCES) {
    out[key] = readFileSync(fileURLToPath(new URL(`${op}/wgsl/${entry}.wgsl`, opsRoot)), "utf8");
  }
  return out;
}
