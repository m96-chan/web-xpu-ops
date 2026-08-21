import { readdirSync, readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

/**
 * Stops the published package from promising a module it never builds.
 *
 * Two files decide what a consumer can import, and nothing connected them:
 *
 *  1. `tsconfig.build.json`'s `include` — what gets compiled into `dist/`
 *  2. `package.json`'s `exports` — what a consumer is allowed to reach
 *
 * They are edited at different times for different reasons, and a mismatch is
 * silent in both directions. `npm run build` succeeds either way, the suite is
 * green either way, and `npm pack` produces a tarball either way: an `exports`
 * entry pointing at a file no `include` builds resolves to nothing on the
 * consumer's disk, and the error surfaces as `ERR_MODULE_NOT_FOUND` in someone
 * else's project.
 *
 * That is not hypothetical — it is issue #138. `llm/sampler.ts`,
 * `llm/kv-cache.ts` and `llm/reshape.ts` were reachable from this repository's
 * own tests and from a `file:`-linked consumer (which resolves deep relative
 * paths and so bypasses `exports` entirely), while being absent from the
 * published tarball. The gap was found by a downstream project, not here.
 *
 * Reads both files as data rather than observing a build, for the same reason
 * `coverage.ts` reads test sources: the check has to hold before anything is
 * built, and it has to name which entry is wrong rather than report that some
 * import failed.
 *
 * `.wgsl` entries are exempt — `scripts/assets.mjs` copies those, not `tsc`,
 * so `include` says nothing about them.
 */

const root = new URL("../", import.meta.url);

function readJsonWithComments(name: string): unknown {
  // `tsconfig.build.json` carries `//` comments (the repo keeps its reasoning
  // next to the setting). Strip them rather than adding a JSON5 dependency for
  // one file; string literals in these two files contain no `//`.
  const text = readFileSync(fileURLToPath(new URL(name, root)), "utf8");
  return JSON.parse(text.replace(/^\s*\/\/.*$/gm, ""));
}

/** Every `dist/...` path an `exports` entry can resolve to. */
function exportedDistPaths(exportsMap: Record<string, unknown>): string[] {
  const found: string[] = [];
  const walk = (value: unknown): void => {
    if (typeof value === "string") {
      if (value.startsWith("./dist/")) found.push(value.slice("./".length));
      return;
    }
    if (value && typeof value === "object") Object.values(value).forEach(walk);
  };
  walk(exportsMap);
  return found;
}

/**
 * `dist/llm/sampler.js` becomes `llm/sampler.ts`, and the wildcard form
 * `dist/ops/<star>/index.d.ts` becomes `ops/<star>/index.ts`. The `*` survives
 * the rewrite, so a wildcard export is compared against the wildcard `include`
 * pattern that would build it. (Written as `<star>` here because the literal
 * sequence would close this comment.)
 */
function sourceOf(distPath: string): string {
  return distPath.replace(/^dist\//, "").replace(/\.d\.ts$|\.js$/, ".ts");
}

describe("published surface", () => {
  const pkg = readJsonWithComments("package.json") as {
    exports: Record<string, unknown>;
    files: string[];
  };
  const build = readJsonWithComments("tsconfig.build.json") as { include: string[] };
  const include = new Set(build.include);

  it("builds every module that exports promises", () => {
    const missing = exportedDistPaths(pkg.exports)
      .filter((path) => !path.endsWith(".wgsl"))
      .map(sourceOf)
      .filter((source) => !include.has(source));

    // Named rather than counted: the failure has to say which entry point was
    // promised and never built.
    expect([...new Set(missing)]).toEqual([]);
  });

  it("ships dist", () => {
    // `exports` pointing into `dist/` is only meaningful if `files` carries it
    // — npm would otherwise publish a tarball with none of it.
    expect(pkg.files).toContain("dist");
  });

  it("states the op count the tree actually has", () => {
    // Third time this line has gone stale: 0.1.0 shipped it saying
    // twenty-four when the tree held twenty-seven, that release's own
    // CHANGELOG entry called the drift structural rather than careless —
    // op PRs land in parallel and each is told to leave the shared count
    // alone, so it is correct when written and wrong once the next batch
    // merges — and proposed exactly this assertion. It said twenty-seven
    // again at 0.2.0, against twenty-nine on disk (#119 added `permute`
    // and `dequant_transpose`). The README is what npm renders, so the
    // number is read by people who cannot check it.
    const ops = readdirSync(fileURLToPath(new URL("ops", root)), { withFileTypes: true })
      .filter((entry) => entry.isDirectory()).length;
    const readme = readFileSync(fileURLToPath(new URL("README.md", root)), "utf8");
    const claimed = /^\| WGSL \| (\d+) ops \|/m.exec(readme);

    expect(claimed, "the backends table lost its `| WGSL | N ops |` row").not.toBeNull();
    expect(Number(claimed![1])).toBe(ops);
  });

  it("keeps a subpath for the kernels", () => {
    // `scripts/assets.mjs` guards the copy — it walks `ops/` rather than a
    // hand-written list, and fails the build when an op has no `.wgsl` at all.
    // What it does not check is whether a consumer can reach what it copied:
    // that depends on this `exports` entry, which nothing else observes.
    // Losing it would leave every kernel in the tarball and unreachable,
    // silently, since the backend a WGSL consumer needs is invisible to the
    // type system (issue #138).
    const kernelSubpaths = Object.entries(pkg.exports)
      .filter(([subpath]) => subpath.endsWith(".wgsl"))
      .map(([subpath]) => subpath);

    expect(kernelSubpaths).not.toEqual([]);
  });
});
