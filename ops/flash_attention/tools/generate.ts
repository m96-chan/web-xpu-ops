/**
 * Writes the shipped flash-attention kernels from their generators.
 *
 * Issue #177. Before this existed, `wgsl/kernel.wgsl` was a hand-copied
 * snapshot of a template under `tools/`, and the two were free to disagree —
 * which is the same failure mode as a benchmark that times a kernel other than
 * the one that ships. `wgsl.test.ts` asserts the files on disk are exactly what
 * this writes, so a generator edited without regenerating fails the tests
 * rather than shipping quietly.
 *
 *     npx tsx ops/flash_attention/tools/generate.ts
 */
import { writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { fa2Flash } from "./fa2.wgsl.js";
import { fa3Flash } from "./fa3.wgsl.js";
import { FLASH_TILE } from "../index.js";

/**
 * The shipped kernels, keyed by file name. All three are built at
 * `FLASH_TILE`'s shape.
 *
 * `fa2_token` (#223) is the same `FLASH_TILE` shape with `layout:
 * "tokenMajor"` — see `FLASH_TOKEN_ENTRY` in `../index.ts` for what reads it
 * and why. fa3 gets no token variant; `fa3.wgsl.ts`'s header says why not.
 */
export function shippedKernels(): Record<"fa2" | "fa3" | "fa2_token", string> {
  // `maxHeadDim`/`maxValueDim` are not part of `FlashShape` — they are
  // `fa2Flash`/`fa3Flash`'s separate `maxD`/`maxDv` arguments — so they are
  // pulled out here rather than spread into the token-major shape below,
  // which otherwise carries excess properties `fa2Flash` never asked for.
  const { maxHeadDim, maxValueDim, ...tile } = FLASH_TILE;
  return {
    fa2: fa2Flash(FLASH_TILE, maxHeadDim, maxValueDim),
    fa3: fa3Flash(FLASH_TILE, maxHeadDim, maxValueDim),
    fa2_token: fa2Flash({ ...tile, layout: "tokenMajor" }, maxHeadDim, maxValueDim),
  };
}

if (process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1]) {
  for (const [name, code] of Object.entries(shippedKernels())) {
    const path = fileURLToPath(new URL(`../wgsl/${name}.wgsl`, import.meta.url));
    writeFileSync(path, code);
    console.log(`wrote ${path} (${code.length} B)`);
  }
}
