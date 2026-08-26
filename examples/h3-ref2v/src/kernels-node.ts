/**
 * The conditioner's kernels, read off the filesystem.
 *
 * The Node half. A browser inlines the same files at bundle time; both go
 * through `CONDITIONER_KERNEL_SOURCES`, so a kernel added on one side and
 * forgotten on the other is a type error rather than a missing dispatch.
 */
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { CONDITIONER_KERNEL_SOURCES, type ConditionerKernels } from "./conditioner-gpu.js";

const opsRoot = new URL("../../../ops/", import.meta.url);

export function conditionerKernels(): ConditionerKernels {
  const out = {} as ConditionerKernels;
  for (const { key, op, entry } of CONDITIONER_KERNEL_SOURCES) {
    out[key] = readFileSync(fileURLToPath(new URL(`${op}/wgsl/${entry}.wgsl`, opsRoot)), "utf8");
  }
  return out;
}
