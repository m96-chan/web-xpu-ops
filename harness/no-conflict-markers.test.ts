import { describe, expect, it } from "vitest";
import { execFileSync } from "node:child_process";
import { readFileSync, statSync } from "node:fs";

/**
 * No tracked file carries an unresolved merge conflict.
 *
 * This exists because one reached a commit. A resolver script matched two of a
 * file's three conflict hunks -- the third had an *empty* side, which its
 * pattern could not express -- and reported failure, but the shell ran
 * `git commit` regardless, and `npm test` stayed green because **nothing in the
 * suite reads `CHANGELOG.md`**. Lint does not read it either. A file with no
 * test and no type-check has no way to be wrong until a person looks at it.
 *
 * Cheap enough to run every time: it reads the files git already tracks.
 */
describe("the working tree", () => {
  // Built at runtime rather than written out, so this file does not match
  // itself -- the test would otherwise be permanently red on its own source.
  const markers = ["<".repeat(7), "=".repeat(7), ">".repeat(7)];

  const tracked = execFileSync("git", ["ls-files", "-z"], { cwd: `${import.meta.dirname}/..`, encoding: "utf8" })
    .split("\0")
    .filter(Boolean);

  it("tracks files at all", () => {
    // If `git ls-files` returned nothing -- run from the wrong directory, or
    // outside a checkout -- every assertion below would pass over an empty
    // list and this test would be a decoration.
    expect(tracked.length).toBeGreaterThan(100);
  });

  it("has no unresolved conflict markers", () => {
    const root = `${import.meta.dirname}/..`;
    const found: string[] = [];
    for (const path of tracked) {
      const full = `${root}/${path}`;
      // Weights, fixtures and images are tracked in some checkouts and are not
      // text; reading a large binary to look for "<<<<<<<" is waste, and a
      // false positive in one would be unfixable.
      if (!/\.(ts|tsx|js|mjs|cjs|wgsl|json|md|py|yml|yaml|html|css|toml)$/.test(path)) continue;
      if (statSync(full).size > 8_000_000) continue;
      const text = readFileSync(full, "utf8");
      for (const marker of markers) {
        // A marker only counts at the start of a line, which is where git
        // writes it. `=======` under a Markdown heading is a setext underline
        // and legal, so the line must be *exactly* the marker or the marker
        // followed by a space and a label.
        const at = new RegExp(`^${marker}( .*)?$`, "m").exec(text);
        if (at) found.push(`${path}: ${at[0].slice(0, 40)}`);
      }
    }
    expect(found).toEqual([]);
  });
});
