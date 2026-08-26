/**
 * Every `requestDevice` in this repository asks for the limits the kernels need.
 *
 * Issue #177. The tiled matmul is 512 invocations wide and stages 12 KB, and
 * WebGPU's defaults are 256 and 16384 — so a runtime that does not ask is
 * refused the pipeline outright, at run time, on the caller's machine.
 *
 * This test exists because that is exactly what happened, twice, and the second
 * time the report came from a browser console rather than from anything here.
 * There were **six** call sites and the claim made after fixing three was "all
 * three runtimes now ask". Dawn had already said the same words about
 * `maxBufferSize` in #166.
 *
 * So the count is not asserted from memory. Every file that calls
 * `requestDevice` is found by reading the tree, and each has to name the limits
 * — which means a seventh runtime added later fails here rather than in
 * someone's tab.
 */
import { describe, expect, it } from "vitest";
import { readdirSync, readFileSync, statSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = fileURLToPath(new URL("..", import.meta.url));

/** Directories that hold source rather than build output or dependencies. */
const SKIP = new Set(["node_modules", "dist", ".git", ".claude", "coverage"]);

function sourceFiles(dir: string, found: string[] = []): string[] {
  for (const entry of readdirSync(dir)) {
    if (SKIP.has(entry)) continue;
    const full = path.join(dir, entry);
    if (statSync(full).isDirectory()) sourceFiles(full, found);
    else if (entry.endsWith(".ts") && !entry.endsWith(".test.ts")) found.push(full);
  }
  return found;
}

/**
 * What a compute runtime has to ask for.
 *
 * Not every limit WebGPU has — only the ones a kernel in `ops/` already exceeds
 * at its shipped shape. Adding one here is a deliberate act, and so is the
 * failure it causes in the runtimes that have not caught up.
 */
const REQUIRED = [
  "maxStorageBufferBindingSize",
  "maxBufferSize",
  "maxComputeWorkgroupStorageSize",
  "maxComputeInvocationsPerWorkgroup",
  "maxComputeWorkgroupSizeX",
];

describe("device limits", () => {
  const callers = sourceFiles(root).filter((file) => /adapter\.requestDevice\(/.test(readFileSync(file, "utf8")));

  it("finds every runtime rather than trusting a remembered count", () => {
    // **Five**, down from six, and the drop is the point of writing it here.
    // `anima-web` and `zimage-web` each carried a copy of the browser
    // `ResidentDevice`; the two had already drifted — one had the profiling
    // half and the other had a comment explaining why it did not — so they were
    // merged into `examples/web-common/src/browser-resident.ts` (issue #200).
    // One fewer `requestDevice` is one fewer place to forget a limit.
    //
    // The number is asserted so that adding a sixth is a decision someone makes
    // here, with the list in front of them, rather than a file that quietly
    // never asks for anything.
    expect(callers.length).toBeGreaterThanOrEqual(5);
  });

  for (const file of callers) {
    const relative = path.relative(root, file);
    it(`${relative} asks for the limits the kernels need`, () => {
      const source = readFileSync(file, "utf8");
      for (const limit of REQUIRED) {
        expect(source, `${relative} never names ${limit}`).toContain(limit);
      }
    });
  }
});
