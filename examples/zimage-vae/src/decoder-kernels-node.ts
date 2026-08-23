/**
 * The decoder's kernels, read off the filesystem.
 *
 * The Node half of `DecoderKernels`; the browser build inlines the same set at
 * bundle time. Both go through `DECODER_KERNEL_SOURCES`, so a kernel added on
 * one side and forgotten on the other is a type error rather than a missing
 * dispatch at run time.
 */
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { DECODER_KERNEL_SOURCES, type DecoderKernels } from "./decoder-gpu.js";

const opsRoot = new URL("../../../ops/", import.meta.url);

export function decoderKernels(): DecoderKernels {
  const out = {} as DecoderKernels;
  for (const { key, op, entry } of DECODER_KERNEL_SOURCES) {
    out[key] = readFileSync(fileURLToPath(new URL(`${op}/wgsl/${entry}.wgsl`, opsRoot)), "utf8");
  }
  return out;
}
