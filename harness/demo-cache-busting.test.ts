/**
 * Every demo server's cache buster actually matches its own page.
 *
 * The browser demos stamp the bundle's mtime onto the `<script src>` because
 * `Cache-Control: no-store` was measured not to be enough — a tab that had
 * loaded the page before the header existed kept executing the bundle it
 * already had. `anima-web` then shipped that mechanism **inert for months**:
 * the server replaced the literal `"./dist/bundle.js"` while its own page said
 * `"/dist/bundle.js"`, so the rewrite matched nothing and returned the input.
 * Nothing failed. The server logged a request, returned 200, and the tab went
 * on running an old bundle.
 *
 * A no-op string replacement is invisible, so the test cannot look for one. It
 * takes the **pattern the server actually uses**, read out of the server's own
 * source, and applies it to the page the server actually serves. Rewriting the
 * pattern to something that no longer matches fails here; so does changing the
 * page's tag.
 */
import { describe, expect, it } from "vitest";
import { readdirSync, readFileSync, existsSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";

const root = fileURLToPath(new URL("..", import.meta.url));
const examples = path.join(root, "examples");

/** Every `examples/*-web` that serves a page and a bundle. */
const demos = readdirSync(examples)
  .filter((name) => name.endsWith("-web"))
  .map((name) => ({
    name,
    server: path.join(examples, name, "server.mjs"),
    page: path.join(examples, name, "index.html"),
  }))
  .filter((d) => existsSync(d.server) && existsSync(d.page));

/**
 * The regex literal the server passes to `.replace`, as the server wrote it.
 *
 * Read rather than restated: a copy of the pattern in this file would pass
 * while the server's own copy matched nothing, which is the bug.
 */
function stampPattern(source: string): RegExp | null {
  const at = source.match(/\.replace\((\/(?:\\.|\[(?:\\.|[^\]])*\]|[^/\\])+\/[gimsuy]*)/);
  if (!at) return null;
  const literal = at[1]!;
  const end = literal.lastIndexOf("/");
  return new RegExp(literal.slice(1, end), literal.slice(end + 1));
}

describe("demo servers / cache busting", () => {
  it("finds the demos", () => {
    expect(demos.map((d) => d.name).sort()).toEqual(["anima-web", "h3-video-web", "zimage-web"]);
  });

  for (const demo of demos) {
    const source = readFileSync(demo.server, "utf8");
    const page = readFileSync(demo.page, "utf8");

    it(`${demo.name}: the page has exactly one bundle tag`, () => {
      const tags = page.match(/<script[^>]*src="[^"]*bundle\.js[^"]*"/g) ?? [];
      expect(tags).toHaveLength(1);
    });

    it(`${demo.name}: the server's stamp pattern matches that tag`, () => {
      const pattern = stampPattern(source);
      expect(pattern, `${demo.name}: no .replace(/.../) to read the pattern from`).not.toBeNull();
      expect(
        pattern!.test(page),
        `${demo.name}: ${pattern} matches nothing in index.html — the cache buster is a no-op, ` +
          "and a stale tab will keep running an old bundle while the server reports serving a new one",
      ).toBe(true);
    });

    it(`${demo.name}: a rewrite that matches nothing is an error, not a pass`, () => {
      // The mechanism failed silently once. Whatever the pattern is, the server
      // has to notice when it does not fire.
      expect(source).toMatch(/if \(stamped === html\) throw/);
    });
  }
});
