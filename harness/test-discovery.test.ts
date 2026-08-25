/**
 * `npm test` runs everything `vitest.config.ts` says is a test.
 *
 * There are two lists. `vitest.config.ts` has `include` globs, used by
 * `npm run test:file` and by editors. `scripts/test.mjs` has its own walk,
 * because it gives each file its own process (issue #38: Dawn can take a whole
 * worker down, and one file per process is what makes a crash attributable).
 *
 * They drifted. `examples/anima/src` was in the config's globs and not in the
 * runner's roots, so 86 tests — the sampler's schedule, the Qwen BPE and the T5
 * unigram — passed in an editor and **never ran in CI**. Nothing said so: a
 * suite that does not collect a file reports the files it did collect, all
 * green. `scripts/test.mjs` even carries a comment warning about exactly this
 * for `examples/zimage`, written while `examples/anima` was already missing.
 *
 * So the lists are compared rather than trusted.
 */
import { describe, expect, it } from "vitest";
import { readdirSync, readFileSync, statSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";

const root = fileURLToPath(new URL("..", import.meta.url));
const config = readFileSync(path.join(root, "vitest.config.ts"), "utf8");
const runner = readFileSync(path.join(root, "scripts/test.mjs"), "utf8");

/** The `include` array from `vitest.config.ts`, as written. */
function includeGlobs(): string[] {
  const array = config.match(/include:\s*\[([^\]]*)\]/);
  if (!array) return [];
  return [...array[1]!.matchAll(/"([^"]+)"/g)].map((m) => m[1]!);
}

/**
 * The directories `scripts/test.mjs` walks.
 *
 * Read out of the source: the runner cannot be imported, because importing it
 * runs the suite.
 */
function runnerRoots(): string[] {
  const roots: string[] = [];
  for (const name of ["OPS", "HARNESS", "LLM"]) {
    const m = runner.match(new RegExp(`const ${name} = "([^"]+)"`));
    if (m) roots.push(m[1]!);
  }
  const examples = runner.match(/const EXAMPLES_TESTED = \[([^\]]*)\]/);
  if (examples) roots.push(...[...examples[1]!.matchAll(/"([^"]+)"/g)].map((m) => m[1]!));
  return roots;
}

/** Every `*.test.ts` in the tree, ignoring `node_modules` and build output. */
function allTestFiles(dir: string, prefix = ""): string[] {
  const found: string[] = [];
  for (const entry of readdirSync(dir)) {
    if (entry === "node_modules" || entry === "dist" || entry.startsWith(".")) continue;
    const full = path.join(dir, entry);
    const rel = prefix ? `${prefix}/${entry}` : entry;
    if (statSync(full).isDirectory()) found.push(...allTestFiles(full, rel));
    else if (entry.endsWith(".test.ts")) found.push(rel);
  }
  return found;
}

/** A vitest glob as a regex. Only `**` and `*` appear in this config. */
function globToRegExp(glob: string): RegExp {
  const escaped = glob.replace(/[.+^${}()|[\]\\]/g, "\\$&");
  const body = escaped.replace(/\*\*\//g, "(?:.*/)?").replace(/\*/g, "[^/]*");
  return new RegExp(`^${body}$`);
}

const GLOBS = includeGlobs();
const ROOTS = runnerRoots();
const FILES = allTestFiles(root);

describe("test discovery / the two lists agree", () => {
  it("reads both lists", () => {
    expect(GLOBS.length).toBeGreaterThan(0);
    expect(ROOTS.length).toBeGreaterThan(0);
    expect(FILES.length).toBeGreaterThan(50);
  });

  for (const file of FILES) {
    const globbed = GLOBS.some((g) => globToRegExp(g).test(file));
    if (!globbed) continue;
    it(`${file} is collected by scripts/test.mjs`, () => {
      expect(
        ROOTS.some((r) => file === r || file.startsWith(`${r}/`)),
        `${file} matches a vitest include glob but sits under none of scripts/test.mjs's roots ` +
          `(${ROOTS.join(", ")}), so it passes in an editor and never runs in CI`,
      ).toBe(true);
    });
  }
});
