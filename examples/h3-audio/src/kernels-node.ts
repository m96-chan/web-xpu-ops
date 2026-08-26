/**
 * The decoder's kernels, read off the filesystem.
 *
 * The Node half. A browser fetches the same files over HTTP; both go through
 * `AUDIO_KERNEL_SOURCES` so the two cannot end up loading different sets,
 * which is the failure a second copy of the list would eventually produce.
 */
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { AUDIO_KERNEL_SOURCES, type AudioKernels } from "./decoder-gpu.js";

const opsRoot = new URL("../../../ops/", import.meta.url);

export function audioKernels(): AudioKernels {
  const out = {} as AudioKernels;
  for (const { key, op, entry } of AUDIO_KERNEL_SOURCES) {
    out[key] = readFileSync(fileURLToPath(new URL(`${op}/wgsl/${entry}.wgsl`, opsRoot)), "utf8");
  }
  return out;
}
